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
        _, _, called_filter = mock_ss.call_args[0]
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


if __name__ == "__main__":
    unittest.main()
