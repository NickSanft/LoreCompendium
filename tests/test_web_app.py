"""
Tests for web_app.py

Covers route responses, input validation, the /search/files dropdown endpoint,
vector search, and the SSE streaming endpoint. All document_engine calls are
mocked so tests run without Ollama or ChromaDB.
"""
import concurrent.futures
import os
import queue as q_module
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_client():
    """Create a TestClient with the lifespan mocked out."""
    from starlette.testclient import TestClient
    with patch("document_engine.initialize_vectorstore"):
        import web_app
        return TestClient(web_app.app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Basic page routes
# ---------------------------------------------------------------------------

class TestPageRoutes(unittest.TestCase):
    def setUp(self):
        self.client = _make_client()

    def test_root_redirects_to_search(self):
        r = self.client.get("/", follow_redirects=False)
        self.assertEqual(r.status_code, 307)
        self.assertIn("/search", r.headers["location"])

    def test_search_page_returns_200(self):
        r = self.client.get("/search")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Search", r.text)

    def test_library_page_returns_200(self):
        with patch("web_app.get_indexed_files", return_value={}), \
             patch("web_app.get_chunk_counts", return_value={}):
            r = self.client.get("/library")
        self.assertEqual(r.status_code, 200)

    def test_health_returns_ok(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_search_page_contains_form(self):
        r = self.client.get("/search")
        self.assertIn("<form", r.text)
        self.assertIn('name="query"', r.text)

    def test_search_page_contains_mode_toggle(self):
        r = self.client.get("/search")
        self.assertIn('name="mode"', r.text)
        self.assertIn('value="ai"', r.text)
        self.assertIn('value="vector"', r.text)


# ---------------------------------------------------------------------------
# /search/files
# ---------------------------------------------------------------------------

class TestSearchFiles(unittest.TestCase):
    def setUp(self):
        self.client = _make_client()

    def test_returns_all_documents_option(self):
        with patch("web_app.get_indexed_files", return_value={}):
            r = self.client.get("/search/files")
        self.assertEqual(r.status_code, 200)
        self.assertIn("All documents", r.text)

    def test_returns_option_for_each_file(self):
        manifest = {
            "/input/lore.txt": {"mtime": 1.0, "size": 100},
            "/input/campaign.pdf": {"mtime": 2.0, "size": 200},
        }
        with patch("web_app.get_indexed_files", return_value=manifest):
            r = self.client.get("/search/files")
        self.assertIn("lore.txt", r.text)
        self.assertIn("campaign.pdf", r.text)

    def test_options_are_html_elements(self):
        with patch("web_app.get_indexed_files", return_value={"/input/a.txt": {}}):
            r = self.client.get("/search/files")
        self.assertIn("<option", r.text)

    def test_empty_manifest_returns_only_all_option(self):
        with patch("web_app.get_indexed_files", return_value={}):
            r = self.client.get("/search/files")
        self.assertEqual(r.text.count("<option"), 1)


# ---------------------------------------------------------------------------
# POST /search/start
# ---------------------------------------------------------------------------

class TestSearchStart(unittest.TestCase):
    def setUp(self):
        self.client = _make_client()

    def test_empty_query_returns_error(self):
        r = self.client.post("/search/start", data={"query": "", "scope": "", "mode": "ai"})
        self.assertEqual(r.status_code, 422)
        self.assertIn("alert-error", r.text)

    def test_whitespace_query_returns_error(self):
        r = self.client.post("/search/start", data={"query": "   ", "scope": "", "mode": "ai"})
        self.assertEqual(r.status_code, 422)

    def test_overlong_query_returns_error(self):
        long_q = "a" * 501
        r = self.client.post("/search/start", data={"query": long_q, "scope": "", "mode": "ai"})
        self.assertEqual(r.status_code, 422)

    def test_valid_query_returns_sse_div(self):
        with patch("web_app.query_documents") as mock_qd:
            mock_qd.return_value = "some answer"
            r = self.client.post("/search/start", data={"query": "test query", "scope": "", "mode": "ai"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("sse-connect", r.text)
        self.assertIn("/search/stream/", r.text)

    def test_valid_query_creates_job(self):
        import web_app
        with patch("web_app.query_documents"):
            r = self.client.post("/search/start", data={"query": "hello", "scope": "", "mode": "ai"})
        self.assertEqual(r.status_code, 200)
        # Job ID extracted from the sse-connect URL
        import re
        match = re.search(r'/search/stream/([a-f0-9-]+)', r.text)
        self.assertIsNotNone(match)
        job_id = match.group(1)
        self.assertIn(job_id, web_app._jobs)
        # Clean up
        web_app._jobs.pop(job_id, None)

    def test_scoped_query_calls_query_documents_scoped(self):
        import web_app
        manifest = {"/input/notes.txt": {"mtime": 1.0, "size": 50}}
        with patch("web_app.get_indexed_files", return_value=manifest):
            with patch("web_app.query_documents_scoped") as mock_scoped:
                mock_scoped.return_value = "answer"
                r = self.client.post("/search/start", data={
                    "query": "what is this about?",
                    "scope": "notes.txt",
                    "mode": "ai",
                })
        self.assertEqual(r.status_code, 200)
        self.assertIn("sse-connect", r.text)


# ---------------------------------------------------------------------------
# GET /search/stream/{job_id}
# ---------------------------------------------------------------------------

class TestSearchStream(unittest.TestCase):
    def setUp(self):
        self.client = _make_client()

    def _make_done_future(self, result="answer"):
        f = concurrent.futures.Future()
        f.set_result(result)
        return f

    def test_unknown_job_id_returns_404(self):
        r = self.client.get("/search/stream/does-not-exist")
        self.assertEqual(r.status_code, 404)

    def test_streams_sentinel_and_closes(self):
        import web_app
        job_id = "test-stream-1"
        sq = q_module.Queue()
        sq.put("Hello ")
        sq.put("world")
        sq.put(None)  # sentinel
        web_app._jobs[job_id] = {"sq": sq, "future": self._make_done_future()}
        try:
            r = self.client.get(f"/search/stream/{job_id}")
            self.assertIn("data:", r.text)
            self.assertIn("Hello", r.text)
            self.assertIn("world", r.text)
        finally:
            web_app._jobs.pop(job_id, None)

    def test_final_event_contains_answer_block(self):
        import web_app
        job_id = "test-stream-2"
        sq = q_module.Queue()
        sq.put("The answer is 42")
        sq.put(None)
        web_app._jobs[job_id] = {"sq": sq, "future": self._make_done_future()}
        try:
            r = self.client.get(f"/search/stream/{job_id}")
            self.assertIn("answer-block", r.text)
        finally:
            web_app._jobs.pop(job_id, None)

    def test_job_cleaned_up_after_stream(self):
        import web_app
        job_id = "test-stream-cleanup"
        sq = q_module.Queue()
        sq.put(None)  # immediate sentinel
        web_app._jobs[job_id] = {"sq": sq, "future": self._make_done_future()}
        self.client.get(f"/search/stream/{job_id}")
        self.assertNotIn(job_id, web_app._jobs)

    def test_empty_queue_with_done_future_closes_cleanly(self):
        import web_app
        job_id = "test-stream-empty"
        sq = q_module.Queue()  # empty, no sentinel
        web_app._jobs[job_id] = {"sq": sq, "future": self._make_done_future()}
        try:
            r = self.client.get(f"/search/stream/{job_id}")
            # Should respond (possibly empty SSE body) without hanging
            self.assertIsNotNone(r)
        finally:
            web_app._jobs.pop(job_id, None)


# ---------------------------------------------------------------------------
# POST /vsearch
# ---------------------------------------------------------------------------

class TestVsearch(unittest.TestCase):
    def setUp(self):
        self.client = _make_client()

    def _make_doc(self, text="sample text", source="/input/lore.txt", page=None, start_index=None):
        from langchain_core.documents import Document
        meta = {"source": source}
        if page is not None:
            meta["page"] = page
        if start_index is not None:
            meta["start_index"] = start_index
        return Document(page_content=text, metadata=meta)

    def test_empty_query_returns_error(self):
        r = self.client.post("/vsearch", data={"query": "", "scope": "", "k": 10, "min_score": 0.3})
        self.assertEqual(r.status_code, 422)
        self.assertIn("alert-error", r.text)

    def test_overlong_query_returns_error(self):
        r = self.client.post("/vsearch", data={"query": "x" * 501, "scope": "", "k": 10, "min_score": 0.3})
        self.assertEqual(r.status_code, 422)

    def test_returns_results_html(self):
        doc = self._make_doc("A battle occurred near the mountain pass.", page=2)
        with patch("web_app.similarity_search", return_value=[(doc, 0.85)]):
            r = self.client.post("/vsearch", data={"query": "battle", "scope": "", "k": 10, "min_score": 0.0})
        self.assertEqual(r.status_code, 200)
        self.assertIn("lore.txt", r.text)
        self.assertIn("battle occurred", r.text)

    def test_shows_score(self):
        doc = self._make_doc("Sample passage.")
        with patch("web_app.similarity_search", return_value=[(doc, 0.75)]):
            r = self.client.post("/vsearch", data={"query": "sample", "scope": "", "k": 10, "min_score": 0.0})
        self.assertIn("75", r.text)  # score shown as percentage

    def test_empty_results_shows_empty_state(self):
        with patch("web_app.similarity_search", return_value=[]):
            r = self.client.post("/vsearch", data={"query": "nothing", "scope": "", "k": 10, "min_score": 0.0})
        self.assertEqual(r.status_code, 200)
        self.assertIn("No results", r.text)

    def test_min_score_filters_low_results(self):
        doc = self._make_doc("Low relevance text.")
        with patch("web_app.similarity_search", return_value=[(doc, 0.1)]):
            r = self.client.post("/vsearch", data={"query": "anything", "scope": "", "k": 10, "min_score": 0.5})
        self.assertIn("No results", r.text)

    def test_scoped_resolves_full_path(self):
        manifest = {"/input/campaign.pdf": {"mtime": 1.0, "size": 999}}
        doc = self._make_doc(source="/input/campaign.pdf")
        with patch("web_app.get_indexed_files", return_value=manifest):
            with patch("web_app.similarity_search", return_value=[(doc, 0.9)]) as mock_ss:
                self.client.post("/vsearch", data={"query": "q", "scope": "campaign.pdf", "k": 10, "min_score": 0.0})
        mock_ss.assert_called_once()
        _, _, called_filter, *_ = mock_ss.call_args[0]
        self.assertEqual(called_filter, "/input/campaign.pdf")

    def test_result_count_shown(self):
        docs = [(self._make_doc(f"text {i}"), 0.8) for i in range(3)]
        with patch("web_app.similarity_search", return_value=docs):
            r = self.client.post("/vsearch", data={"query": "test", "scope": "", "k": 10, "min_score": 0.0})
        self.assertIn("3", r.text)


# ---------------------------------------------------------------------------
# _validate_query helper
# ---------------------------------------------------------------------------

class TestValidateQuery(unittest.TestCase):
    def test_valid_query_returns_none(self):
        import web_app
        self.assertIsNone(web_app._validate_query("What is the lore?"))

    def test_empty_string_is_invalid(self):
        import web_app
        self.assertIsNotNone(web_app._validate_query(""))

    def test_whitespace_only_is_invalid(self):
        import web_app
        self.assertIsNotNone(web_app._validate_query("   "))

    def test_at_max_length_is_valid(self):
        import web_app
        self.assertIsNone(web_app._validate_query("a" * web_app._MAX_QUERY_LENGTH))

    def test_over_max_length_is_invalid(self):
        import web_app
        self.assertIsNotNone(web_app._validate_query("a" * (web_app._MAX_QUERY_LENGTH + 1)))


# ---------------------------------------------------------------------------
# _resolve_scope helper
# ---------------------------------------------------------------------------

class TestResolveScope(unittest.TestCase):
    def test_empty_scope_returns_none(self):
        import web_app
        self.assertIsNone(web_app._resolve_scope(""))

    def test_matching_basename_returns_full_path(self):
        import web_app
        manifest = {"/input/notes.txt": {}}
        with patch("web_app.get_indexed_files", return_value=manifest):
            result = web_app._resolve_scope("notes.txt")
        self.assertEqual(result, "/input/notes.txt")

    def test_no_match_returns_none(self):
        import web_app
        with patch("web_app.get_indexed_files", return_value={}):
            result = web_app._resolve_scope("missing.txt")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# GET /library
# ---------------------------------------------------------------------------

class TestLibraryPage(unittest.TestCase):
    def setUp(self):
        self.client = _make_client()

    def test_library_page_returns_200(self):
        with patch("web_app.get_indexed_files", return_value={}), \
             patch("web_app.get_chunk_counts", return_value={}):
            r = self.client.get("/library")
        self.assertEqual(r.status_code, 200)

    def test_library_shows_files(self):
        manifest = {"/input/lore.pdf": {"size": 1024, "mtime": 1742000000.0}}
        with patch("web_app.get_indexed_files", return_value=manifest), \
             patch("web_app.get_chunk_counts", return_value={"/input/lore.pdf": 5}):
            r = self.client.get("/library")
        self.assertIn("lore.pdf", r.text)

    def test_library_empty_state_when_no_files(self):
        with patch("web_app.get_indexed_files", return_value={}), \
             patch("web_app.get_chunk_counts", return_value={}):
            r = self.client.get("/library")
        self.assertIn("No documents", r.text)

    def test_library_shows_upload_area(self):
        with patch("web_app.get_indexed_files", return_value={}), \
             patch("web_app.get_chunk_counts", return_value={}):
            r = self.client.get("/library")
        self.assertIn("upload", r.text.lower())


# ---------------------------------------------------------------------------
# POST /library/upload
# ---------------------------------------------------------------------------

class TestLibraryUpload(unittest.TestCase):
    def setUp(self):
        self.client = _make_client()

    def test_unsupported_extension_returns_error(self):
        r = self.client.post(
            "/library/upload",
            files={"file": ("malware.exe", b"data", "application/octet-stream")},
        )
        self.assertEqual(r.status_code, 422)
        self.assertIn("Unsupported", r.text)

    def test_valid_txt_file_saves_and_returns_row(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            with patch("web_app.DOC_FOLDER", tmp), \
                 patch("web_app.get_duplicate_source", return_value=None), \
                 patch("web_app.INGESTION_QUEUE") as mock_q:
                r = self.client.post(
                    "/library/upload",
                    files={"file": ("notes.txt", b"hello world", "text/plain")},
                )
        self.assertEqual(r.status_code, 200)
        self.assertIn("notes.txt", r.text)

    def test_duplicate_detection_shows_warning(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            with patch("web_app.DOC_FOLDER", tmp), \
                 patch("web_app.get_duplicate_source", return_value="original.txt"), \
                 patch("web_app.INGESTION_QUEUE"):
                r = self.client.post(
                    "/library/upload",
                    files={"file": ("copy.txt", b"hello world", "text/plain")},
                )
        self.assertIn("Duplicate", r.text)


# ---------------------------------------------------------------------------
# POST /library/reindex
# ---------------------------------------------------------------------------

class TestLibraryReindex(unittest.TestCase):
    def setUp(self):
        self.client = _make_client()

    def test_reindex_returns_count(self):
        with patch("web_app.trigger_reindex", return_value=3):
            r = self.client.post("/library/reindex")
        self.assertIn("3", r.text)

    def test_reindex_no_files_shows_message(self):
        with patch("web_app.trigger_reindex", return_value=0):
            r = self.client.post("/library/reindex")
        self.assertIn("No files", r.text)


# ---------------------------------------------------------------------------
# DELETE /library/file/{name}
# ---------------------------------------------------------------------------

class TestLibraryDelete(unittest.TestCase):
    def setUp(self):
        self.client = _make_client()

    def test_delete_nonexistent_returns_404(self):
        r = self.client.delete("/library/file/does_not_exist.txt")
        self.assertEqual(r.status_code, 404)

    def test_delete_existing_file_returns_200(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "to_delete.txt")
            with open(target, "w") as f:
                f.write("content")
            with patch("web_app.DOC_FOLDER", tmp), \
                 patch("web_app.INGESTION_QUEUE"):
                r = self.client.delete("/library/file/to_delete.txt")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(os.path.exists(target))

    def test_delete_prevents_path_traversal(self):
        # ../etc/passwd should be sanitised to passwd, which won't exist
        r = self.client.delete("/library/file/..%2Fetc%2Fpasswd")
        self.assertEqual(r.status_code, 404)


# ---------------------------------------------------------------------------
# GET /chunks/{filename}
# ---------------------------------------------------------------------------

class TestChunksPage(unittest.TestCase):
    def setUp(self):
        self.client = _make_client()

    def test_unknown_file_returns_404(self):
        with patch("web_app.get_indexed_files", return_value={}):
            r = self.client.get("/chunks/missing.txt")
        self.assertEqual(r.status_code, 404)

    def test_known_file_returns_200(self):
        manifest = {"/input/lore.txt": {"size": 500, "mtime": 1742000000.0}}
        with patch("web_app.get_indexed_files", return_value=manifest), \
             patch("web_app.get_chunk_counts", return_value={"/input/lore.txt": 3}):
            r = self.client.get("/chunks/lore.txt")
        self.assertEqual(r.status_code, 200)
        self.assertIn("lore.txt", r.text)

    def test_chunk_page_has_back_link(self):
        manifest = {"/input/notes.txt": {"size": 100, "mtime": 1742000000.0}}
        with patch("web_app.get_indexed_files", return_value=manifest), \
             patch("web_app.get_chunk_counts", return_value={}):
            r = self.client.get("/chunks/notes.txt")
        self.assertIn("/library", r.text)


# ---------------------------------------------------------------------------
# GET /chunks/{filename}/data
# ---------------------------------------------------------------------------

class TestChunksData(unittest.TestCase):
    def setUp(self):
        self.client = _make_client()

    def _make_doc(self, text, source="/input/lore.txt", start_index=0):
        from langchain_core.documents import Document
        return Document(page_content=text, metadata={"source": source, "start_index": start_index})

    def test_unknown_file_returns_empty_state(self):
        with patch("web_app.get_indexed_files", return_value={}):
            r = self.client.get("/chunks/missing.txt/data")
        self.assertIn("not found", r.text.lower())

    def test_returns_chunks_html(self):
        manifest = {"/input/lore.txt": {}}
        chunks = [{"content": "First passage.", "metadata": {"source": "/input/lore.txt", "start_index": 0}}]
        with patch("web_app.get_indexed_files", return_value=manifest), \
             patch("web_app.get_chunks_for_file", return_value=chunks):
            r = self.client.get("/chunks/lore.txt/data")
        self.assertIn("First passage", r.text)

    def test_scored_query_uses_similarity_search(self):
        manifest = {"/input/lore.txt": {}}
        doc = self._make_doc("Relevant text.")
        with patch("web_app.get_indexed_files", return_value=manifest), \
             patch("web_app.similarity_search", return_value=[(doc, 0.88)]) as mock_ss:
            r = self.client.get("/chunks/lore.txt/data?q=relevant")
        mock_ss.assert_called_once()
        self.assertIn("Relevant text", r.text)

    def test_scored_chunks_show_percentage(self):
        manifest = {"/input/lore.txt": {}}
        doc = self._make_doc("Some text.")
        with patch("web_app.get_indexed_files", return_value=manifest), \
             patch("web_app.similarity_search", return_value=[(doc, 0.72)]):
            r = self.client.get("/chunks/lore.txt/data?q=some")
        self.assertIn("72", r.text)


# ---------------------------------------------------------------------------
# Tag API
# ---------------------------------------------------------------------------

class TestTagAPI(unittest.TestCase):
    def setUp(self):
        self.client = _make_client()

    def test_get_all_tags_returns_list(self):
        manifest = {"/input/a.txt": {"tags": ["canon", "arc-1"]}, "/input/b.txt": {"tags": ["arc-1", "draft"]}}
        with patch("web_app.get_indexed_files", return_value=manifest):
            r = self.client.get("/library/tags")
        self.assertEqual(r.status_code, 200)
        tags = r.json()
        self.assertIn("canon", tags)
        self.assertIn("draft", tags)
        self.assertEqual(tags, sorted(tags))

    def test_get_all_tags_empty_manifest(self):
        with patch("web_app.get_indexed_files", return_value={}):
            r = self.client.get("/library/tags")
        self.assertEqual(r.json(), [])

    def test_set_tags_unknown_file_returns_404(self):
        with patch("web_app.get_indexed_files", return_value={}):
            r = self.client.put("/library/file/ghost.txt/tags", data={"tags": "canon"})
        self.assertEqual(r.status_code, 404)

    def test_set_tags_returns_pill_html(self):
        manifest = {"/input/notes.txt": {}}
        with patch("web_app.get_indexed_files", return_value=manifest), \
             patch("web_app.set_tags") as mock_set:
            r = self.client.put("/library/file/notes.txt/tags", data={"tags": "canon,arc-1"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("canon", r.text)
        self.assertIn("arc-1", r.text)
        mock_set.assert_called_once()

    def test_set_tags_with_new_tag_appends(self):
        manifest = {"/input/notes.txt": {}}
        with patch("web_app.get_indexed_files", return_value=manifest), \
             patch("web_app.set_tags") as mock_set:
            r = self.client.put("/library/file/notes.txt/tags", data={"base": "canon", "new_tag": "arc-2"})
        self.assertEqual(r.status_code, 200)
        called_tags = mock_set.call_args[0][1]
        self.assertIn("arc-2", called_tags)
        self.assertIn("canon", called_tags)

    def test_set_tags_deduplicates(self):
        manifest = {"/input/a.txt": {}}
        captured = {}
        def fake_set(path, tags):
            captured["tags"] = tags
        with patch("web_app.get_indexed_files", return_value=manifest), \
             patch("web_app.set_tags", side_effect=fake_set):
            self.client.put("/library/file/a.txt/tags", data={"tags": "canon,canon,canon"})
        self.assertEqual(captured["tags"].count("canon"), 1)


# ---------------------------------------------------------------------------
# Settings page
# ---------------------------------------------------------------------------

class TestSettingsPage(unittest.TestCase):
    def setUp(self):
        self.client = _make_client()
        self._cfg = {
            "role_description": "A helpful assistant",
            "thinking_ollama_model": "gpt-oss",
            "fast_ollama_model": "llama3.2",
            "embedding_model": "mxbai-embed-large",
        }

    def test_settings_page_returns_200(self):
        with patch("web_app.get_config", return_value=self._cfg):
            r = self.client.get("/settings")
        self.assertEqual(r.status_code, 200)

    def test_settings_page_shows_all_fields(self):
        with patch("web_app.get_config", return_value=self._cfg):
            r = self.client.get("/settings")
        self.assertIn('name="role_description"', r.text)
        self.assertIn('name="thinking_ollama_model"', r.text)
        self.assertIn('name="fast_ollama_model"', r.text)
        self.assertIn('name="embedding_model"', r.text)

    def test_settings_page_populates_values(self):
        with patch("web_app.get_config", return_value=self._cfg):
            r = self.client.get("/settings")
        self.assertIn("gpt-oss", r.text)
        self.assertIn("llama3.2", r.text)
        self.assertIn("mxbai-embed-large", r.text)

    def test_settings_in_nav(self):
        with patch("web_app.get_config", return_value=self._cfg):
            r = self.client.get("/settings")
        self.assertIn('href="/settings"', r.text)

    def test_save_calls_save_config_with_updates(self):
        captured = {}
        def fake_save(updates):
            captured.update(updates)
        with patch("web_app.save_config", side_effect=fake_save):
            r = self.client.post("/settings", data={
                "role_description": "New personality",
                "thinking_ollama_model": "new-model",
                "fast_ollama_model": "fast-model",
                "embedding_model": "embed-model",
            })
        self.assertEqual(r.status_code, 200)
        self.assertIn("alert-success", r.text)
        self.assertEqual(captured["thinking_ollama_model"], "new-model")
        self.assertEqual(captured["role_description"], "New personality")

    def test_save_empty_field_returns_422(self):
        with patch("web_app.save_config"):
            r = self.client.post("/settings", data={
                "role_description": "ok",
                "thinking_ollama_model": "",   # empty — should fail
                "fast_ollama_model": "fast",
                "embedding_model": "embed",
            })
        self.assertEqual(r.status_code, 422)
        self.assertIn("alert-error", r.text)

    def test_save_does_not_expose_discord_token(self):
        """The settings endpoint must never write discord_bot_token."""
        captured = {}
        def fake_save(updates):
            captured.update(updates)
        with patch("web_app.save_config", side_effect=fake_save):
            self.client.post("/settings", data={
                "role_description": "ok",
                "thinking_ollama_model": "m",
                "fast_ollama_model": "m",
                "embedding_model": "m",
                "discord_bot_token": "leaked-token",  # injected field
            })
        self.assertNotIn("discord_bot_token", captured)


if __name__ == "__main__":
    unittest.main()
