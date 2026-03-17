import logging
import sys
from contextlib import asynccontextmanager

import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from document_engine import initialize_vectorstore
from lore_utils import check_ollama_health, setup_logging

logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising vectorstore…")
    asyncio.create_task(asyncio.to_thread(initialize_vectorstore))
    yield
    logger.info("Web app shutting down.")


app = FastAPI(title="Lore Compendium", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/search")


@app.get("/search")
async def search_page(request: Request):
    return templates.TemplateResponse(
        "search.html",
        {"request": request, "active_page": "search"},
    )


@app.get("/library")
async def library_page(request: Request):
    return templates.TemplateResponse(
        "library.html",
        {"request": request, "active_page": "library"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


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

    logger.info("Ollama is running. Starting web server at http://127.0.0.1:8080")
    uvicorn.run("web_app:app", host="127.0.0.1", port=8080, reload=False)
