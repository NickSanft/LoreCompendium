import asyncio
import html as html_mod
import logging
import os as _os
import queue
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from datetime import datetime

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from urllib.parse import quote as _url_quote

from document_engine import (
    get_chunk_counts,
    get_chunks_for_file,
    get_duplicate_source,
    get_indexed_files,
    initialize_vectorstore,
    query_documents,
    query_documents_scoped,
    similarity_search,
    trigger_reindex,
    INGESTION_QUEUE,
)
from lore_utils import check_ollama_health, DOC_FOLDER, setup_logging, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="templates")
templates.env.filters["basename"] = _os.path.basename
templates.env.filters["urlencode"] = _url_quote


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt_ts(mtime) -> str:
    if not mtime:
        return "unknown"
    return datetime.fromtimestamp(float(mtime)).strftime("%Y-%m-%d %H:%M")


templates.env.filters["fmt_size"] = _fmt_size
templates.env.filters["fmt_ts"] = _fmt_ts

_MAX_QUERY_LENGTH = 500
_STREAM_INTERVAL = 0.5   # seconds between intermediate SSE edits
_jobs: dict[str, dict] = {}   # job_id -> {"sq": Queue, "future": Future}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_query(query: str) -> Optional[str]:
    """Return an error string if the query is invalid, else None."""
    if not query.strip():
        return "Query cannot be empty."
    if len(query) > _MAX_QUERY_LENGTH:
        return f"Query is too long ({len(query)} chars). Max is {_MAX_QUERY_LENGTH}."
    return None


def _resolve_scope(scope: str) -> Optional[str]:
    """Resolve a filename basename to its full manifest path, or None."""
    if not scope:
        return None
    manifest = get_indexed_files()
    import os
    matches = [p for p in manifest if os.path.basename(p) == scope]
    return matches[0] if matches else None


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising vectorstore…")
    asyncio.create_task(asyncio.to_thread(initialize_vectorstore))
    yield
    logger.info("Web app shutting down.")


app = FastAPI(title="Lore Compendium", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Page routes ───────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/search")


@app.get("/search")
async def search_page(request: Request):
    return templates.TemplateResponse(
        request, "search.html",
        {"active_page": "search"},
    )


@app.get("/library")
async def library_page(request: Request):
    manifest = await asyncio.to_thread(get_indexed_files)
    chunk_counts = await asyncio.to_thread(get_chunk_counts)
    files = []
    for path, sig in sorted(manifest.items(), key=lambda x: _os.path.basename(x[0]).lower()):
        files.append({
            "path": path,
            "name": _os.path.basename(path),
            "size": sig.get("size", 0),
            "mtime": sig.get("mtime"),
            "chunks": chunk_counts.get(path, 0),
        })
    return templates.TemplateResponse(
        request, "library.html",
        {"active_page": "library", "files": files,
         "supported": ", ".join(SUPPORTED_EXTENSIONS)},
    )


@app.post("/library/upload", response_class=HTMLResponse)
async def library_upload(request: Request, file: UploadFile = File(...)):
    ext = _os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return HTMLResponse(
            f'<div class="alert alert-error">Unsupported file type: {html_mod.escape(ext)}. '
            f'Allowed: {", ".join(SUPPORTED_EXTENSIONS)}</div>',
            status_code=422,
        )
    save_path = _os.path.join(DOC_FOLDER, file.filename)
    is_update = _os.path.exists(save_path)
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    duplicate = await asyncio.to_thread(get_duplicate_source, save_path)
    notice = ""
    if duplicate and duplicate != file.filename:
        notice = f'<div class="alert alert-error mt-1">⚠ Duplicate of already-indexed <strong>{html_mod.escape(duplicate)}</strong>.</div>'
    elif is_update:
        notice = '<div class="alert alert-info mt-1">🔄 Updated existing file.</div>'

    # Queue for ingestion
    INGESTION_QUEUE.put(("update", save_path))

    size = len(content)
    row_html = templates.env.get_template("partials/file_row.html").render(
        file={
            "path": save_path,
            "name": file.filename,
            "size": size,
            "mtime": None,
            "chunks": 0,
        },
        indexing=True,
    )
    return HTMLResponse(f'{notice}{row_html}')


@app.post("/library/reindex", response_class=HTMLResponse)
async def library_reindex():
    count = await asyncio.to_thread(trigger_reindex)
    if count == 0:
        return HTMLResponse('<span class="alert alert-info">No files found in input folder.</span>')
    return HTMLResponse(f'<span class="alert alert-success">Queued {count} file(s) for re-indexing.</span>')


@app.delete("/library/file/{name}", response_class=HTMLResponse)
async def library_delete(name: str):
    safe_name = _os.path.basename(name)  # prevent path traversal
    file_path = _os.path.join(DOC_FOLDER, safe_name)
    if not _os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    _os.remove(file_path)
    # Watchdog will detect deletion; also queue explicitly for safety
    INGESTION_QUEUE.put(("delete", file_path))
    return HTMLResponse("")  # HTMX hx-swap="delete" removes the row


# ── Chunk explorer ────────────────────────────────────────────────────────────

@app.get("/chunks/{filename}")
async def chunks_page(request: Request, filename: str):
    full_path = await asyncio.to_thread(_resolve_scope, filename)
    if full_path is None:
        raise HTTPException(status_code=404, detail=f"'{filename}' is not indexed.")
    manifest = await asyncio.to_thread(get_indexed_files)
    sig = manifest.get(full_path, {})
    chunk_counts = await asyncio.to_thread(get_chunk_counts)
    return templates.TemplateResponse(
        request, "chunks.html",
        {
            "active_page": "library",
            "filename": filename,
            "full_path": full_path,
            "size": sig.get("size", 0),
            "chunk_count": chunk_counts.get(full_path, 0),
        },
    )


@app.get("/chunks/{filename}/data", response_class=HTMLResponse)
async def chunks_data(filename: str, q: str = ""):
    full_path = await asyncio.to_thread(_resolve_scope, filename)
    if full_path is None:
        return HTMLResponse('<div class="empty-state">File not found in index.</div>')

    if q.strip():
        # Scored by similarity — reuse similarity_search with large k
        raw = await asyncio.to_thread(similarity_search, q.strip(), 100, full_path)
        chunks = [
            {"content": doc.page_content, "metadata": doc.metadata, "score": round(score, 3)}
            for doc, score in raw
        ]
    else:
        # All chunks in document order, no score
        raw = await asyncio.to_thread(get_chunks_for_file, full_path)
        chunks = [{"content": c["content"], "metadata": c["metadata"], "score": None} for c in raw]

    env = templates.env
    t = env.get_template("partials/chunk_list.html")
    return HTMLResponse(t.render(chunks=chunks, query=q))


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Search API ────────────────────────────────────────────────────────────────

@app.get("/search/files", response_class=HTMLResponse)
async def search_files():
    """Return <option> elements for the scope dropdown."""
    import os
    manifest = await asyncio.to_thread(get_indexed_files)
    names = sorted({os.path.basename(p) for p in manifest})
    opts = ['<option value="">All documents</option>']
    for name in names:
        opts.append(f'<option value="{html_mod.escape(name)}">{html_mod.escape(name)}</option>')
    return "\n".join(opts)


@app.post("/search/start", response_class=HTMLResponse)
async def search_start(
    query: str = Form(...),
    scope: str = Form(""),
):
    err = _validate_query(query)
    if err:
        return HTMLResponse(f'<div class="alert alert-error">{html_mod.escape(err)}</div>', status_code=422)

    job_id = str(uuid.uuid4())
    sq: queue.Queue = queue.Queue()
    loop = asyncio.get_running_loop()

    if scope:
        future = loop.run_in_executor(
            None, query_documents_scoped, query, scope, False, sq
        )
    else:
        future = loop.run_in_executor(
            None, query_documents, query, False, sq
        )

    _jobs[job_id] = {"sq": sq, "future": future}

    sse_url = f"/search/stream/{job_id}"
    return HTMLResponse(f'''
<div id="result-stream" hx-ext="sse" sse-connect="{sse_url}">
  <div sse-swap="message" hx-swap="innerHTML">
    <p class="text-muted">⏳ Generating response…</p>
  </div>
</div>
''')


@app.get("/search/stream/{job_id}")
async def search_stream(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _jobs[job_id]
    sq: queue.Queue = job["sq"]
    future = job["future"]

    async def event_gen():
        accumulated = ""
        last_sent = 0.0

        while True:
            got_chunk = False
            while True:
                try:
                    chunk = sq.get_nowait()
                except queue.Empty:
                    break
                if chunk is None:  # sentinel — generation complete
                    escaped = html_mod.escape(accumulated)
                    copy_js = "navigator.clipboard.writeText(this.dataset.text)"
                    final_html = (
                        f'<div class="answer-block">'
                        f'<pre class="answer-text">{escaped}</pre>'
                        f'<div class="result-footer" style="margin-top:.75rem;">'
                        f'<button class="btn btn-secondary btn-sm" '
                        f'onclick="{copy_js}" data-text="{escaped}">Copy</button>'
                        f'</div>'
                        f'</div>'
                    )
                    yield f"data: {final_html}\n\n"
                    _jobs.pop(job_id, None)
                    return
                accumulated += chunk
                got_chunk = True

            # Exit if the background task died without sending the sentinel
            if future.done() and sq.empty():
                if accumulated:
                    escaped = html_mod.escape(accumulated)
                    yield f"data: <pre class='answer-text'>{escaped}</pre>\n\n"
                _jobs.pop(job_id, None)
                return

            # Throttled intermediate update
            if got_chunk and accumulated:
                now = time.monotonic()
                if now - last_sent >= _STREAM_INTERVAL:
                    escaped = html_mod.escape(accumulated)
                    yield f"data: <pre class='answer-text'>{escaped}▌</pre>\n\n"
                    last_sent = now

            await asyncio.sleep(0.05)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ── Vector search ─────────────────────────────────────────────────────────────

@app.post("/vsearch")
async def vsearch(
    request: Request,
    query: str = Form(...),
    scope: str = Form(""),
    k: int = Form(10),
    min_score: float = Form(0.3),
):
    err = _validate_query(query)
    if err:
        return HTMLResponse(f'<div class="alert alert-error">{html_mod.escape(err)}</div>', status_code=422)

    full_path = await asyncio.to_thread(_resolve_scope, scope)
    hits = await asyncio.to_thread(similarity_search, query, k, full_path)

    # Filter by minimum relevance score
    hits = [(doc, score) for doc, score in hits if score >= min_score]

    return templates.TemplateResponse(
        request, "partials/vsearch_results.html",
        {"hits": hits, "query": query, "total": len(hits)},
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    setup_logging(level=logging.INFO)

    errors = check_ollama_health()
    if errors:
        logger.error("Pre-flight checks failed:")
        for err in errors:
            logger.error(err)
        sys.exit(1)

    logger.info("Starting web server at http://127.0.0.1:8080")
    uvicorn.run("web_app:app", host="127.0.0.1", port=8080, reload=False)
