# Lore Compendium - Discord AI Chatbot

Lore Compendium is an AI-powered Discord bot that can answer questions about your documents. Whether you have lore books, PDFs, Word documents, or spreadsheets, the bot searches through them and provides intelligent, cited answers — entirely locally, with no cloud services.

**Supported file formats:** `.docx`, `.pdf`, `.xlsx`, `.csv`, `.txt`, `.md`

---

## Quick Start (For Beginners)

### Step 1: Install Requirements

- **Python 3.13** — [python.org](https://www.python.org/downloads/)
  - On Windows: check "Add Python to PATH" during installation
- **Ollama** — [ollama.com](https://ollama.com/download)
  - Windows: run the installer
  - Mac: `brew install ollama` or download from the website
  - Linux: `curl -fsSL https://ollama.com/install.sh | sh`

### Step 2: Get a Discord Bot Token

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Click **New Application** and give it a name
3. Go to the **Bot** section → click **Reset Token** and copy it
4. Enable **Message Content Intent** under Privileged Gateway Intents
5. Go to **OAuth2 → URL Generator**, select `bot`, choose permissions, and invite it to your server

### Step 3: Run Setup

**Windows:** double-click `setup.bat`
**Mac/Linux:** `./setup.sh`

This installs all libraries, downloads AI models (10–20 min), and walks you through configuration.

### Step 4: Add Documents

Put your files in the `input/` folder.

### Step 5: Start the Bot

**Windows:** double-click `start.bat`
**Mac/Linux:** `./start.sh`

Your bot is now running. Go to Discord and start asking questions!

---

## Discord Commands

| Command | Description |
|---|---|
| `/lore <query>` | Search all indexed documents and generate an answer |
| `/ask <filename> <query>` | Search within a specific document (filename autocompletes) |
| `/search <query> [filename]` | Preview raw matching chunks without LLM generation |
| `/status` | Show which documents are currently indexed |
| `/reindex` | Force a re-scan of the `input/` folder |
| `/help` | Show all commands |

**Conversational mode:** DM the bot or @mention it in a channel for a free-form conversation. The bot automatically searches your documents when relevant.

**Drag-and-drop ingestion:** Drop a supported file into any channel and the bot saves and indexes it automatically, sending a follow-up message when indexing is complete.

---

## Features

### Search & Retrieval
- **Hybrid search** — combines vector (semantic) search with BM25 keyword search, merged via Reciprocal Rank Fusion, so both meaning and exact terms are matched
- **Multi-query retrieval** — generates 2 alternative phrasings of every query to improve recall when document wording differs from the question
- **Context expansion** — retrieved chunks are expanded with their neighbours before being sent to the LLM, providing richer surrounding context
- **Cross-encoder reranking** *(optional)* — if a rerank model is configured, retrieved chunks are reordered by a cross-encoder before generation for higher precision
- **Semantic chunking** — documents are split at topic boundaries rather than fixed character counts, keeping related content together; falls back to character-based splitting for very long sentences

### Answer Quality
- **Source citations** — every answer includes a numbered Sources footer with filename and page/line location
- **Answer faithfulness check** — after generation, a fast model verifies all claims are grounded in the retrieved sources; a warning is shown if not
- **Conversation memory** — `/lore` retains the last 3 Q&A turns per user so follow-up questions are understood in context
- **Query result caching** — identical queries return instantly from an in-memory cache (1-hour TTL, auto-invalidated when documents change)

### Document Management
- **Live sync** — a Watchdog observer detects new, changed, or deleted files in `input/` and updates the index automatically
- **Parallel ingestion** — multiple documents are loaded concurrently via `ProcessPoolExecutor`
- **Duplicate detection** — content-identical files uploaded via Discord are flagged before indexing
- **Tag filtering** — documents can be tagged and queries can be scoped to a specific tag

### Web UI
A local web interface (FastAPI + HTMX) runs alongside the bot:
- **Chunk Explorer** — browse indexed documents and inspect individual chunks
- **Similarity Search** — test retrieval queries with scored results and snippets
- **Settings** — edit model names and bot personality without touching `config.json` directly

---

## Architecture

Two separate LangGraph workflows:

**RAG Pipeline** (`document_engine.py`)
`retrieve → grade_documents → [generate_rag | transform_query]`

- **retrieve**: runs up to 3 query variants through MMR vector search + BM25, merges with RRF, optionally reranks
- **grade_documents**: filters irrelevant chunks in parallel using `fast_llm`
- **transform_query**: rewrites the query with `fast_llm` if no relevant chunks are found (up to 3 retries)
- **generate_rag**: expands chunk context, prepends conversation history, streams the answer, checks faithfulness, appends a Sources footer

**Conversational Agent** (`conversation.py`)
A LangGraph ReAct agent with per-user `MemorySaver` history. The `search_documents` tool delegates to the RAG pipeline when the LLM decides retrieval is needed.

---

## Advanced Setup (For Developers)

### Prerequisites

- Python 3.13
- Ollama running (`ollama serve`)

### Installation

```bash
git clone <repository-url>
cd LoreCompendium

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### Download Models

```bash
cd modelfiles
ollama create -f gpt-oss-20b-modelfile.txt gpt-oss
ollama create -f llama3.2-modelfile.txt llama3.2
ollama pull mxbai-embed-large
cd ..
```

### Configuration

Create `config.json` in the project root:

```json
{
  "discord_bot_token": "your_token_here",
  "role_description": "Bot personality description",
  "thinking_ollama_model": "gpt-oss",
  "fast_ollama_model": "llama3.2",
  "embedding_model": "mxbai-embed-large",
  "rerank_model": ""
}
```

`rerank_model` is optional. Set it to a model like `mxbai-rerank-large` (requires `ollama pull mxbai-rerank-large`) to enable cross-encoder reranking. Leave blank to skip.

All settings can also be edited at runtime via the web UI Settings page.

### Run the Bot

```bash
python discord_main.py
```

On startup the bot runs an Ollama health check, initialises the vector store, and begins indexing any new or changed documents.

### Run the Web UI

```bash
python web_app.py
```

Accessible at `http://localhost:8000` by default.

### Running Tests

```bash
.venv/Scripts/python -m pytest tests/     # Windows
.venv/bin/python -m pytest tests/         # Mac/Linux
```

---

## Data Storage

| Path | Contents |
|---|---|
| `input/` | Drop documents here for indexing |
| `chroma_store/` | Persisted ChromaDB vector store |
| `chroma_store/indexed_files.txt` | JSON manifest tracking indexed files with mtime/size/hash |
| `config.json` | Runtime configuration (not committed) |

---

## License

See repository for license information.

## Contributing

Contributions are welcome. Please open an issue or pull request.
