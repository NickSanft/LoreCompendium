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

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from document_engine import (
    get_indexed_files,
    initialize_vectorstore,
    query_documents,
    query_documents_scoped,
    similarity_search,
)
from lore_utils import check_ollama_health, setup_logging

logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="templates")
templates.env.filters["basename"] = _os.path.basename

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
    return templates.TemplateResponse(
        request, "library.html",
        {"active_page": "library"},
    )


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
                    final_html = (
                        f'<div class="answer-block">'
                        f'<pre class="answer-text">{escaped}</pre>'
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
