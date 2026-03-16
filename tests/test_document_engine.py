"""
Tests for document_engine.py

Covers manifest I/O, graph routing logic, file loading dispatch, and the
query_documents entry point. Ollama and ChromaDB calls are mocked throughout
so these tests run without any external services.
"""
import json
import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

class TestLoadIndexManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        import document_engine
        self._orig = document_engine.INDEXED_FILES_PATH
        document_engine.INDEXED_FILES_PATH = os.path.join(self.tmp, "indexed_files.txt")

    def tearDown(self):
        import document_engine
        document_engine.INDEXED_FILES_PATH = self._orig

    def _path(self):
        import document_engine
        return document_engine.INDEXED_FILES_PATH

    def test_returns_empty_dict_when_file_missing(self):
        import document_engine
        result = document_engine._load_index_manifest()
        self.assertEqual(result, {})

    def test_returns_empty_dict_for_empty_file(self):
        open(self._path(), "w").close()
        import document_engine
        result = document_engine._load_index_manifest()
        self.assertEqual(result, {})

    def test_parses_valid_json_manifest(self):
        data = {"file_a.txt": {"mtime": 1.0, "size": 100}}
        with open(self._path(), "w", encoding="utf-8") as f:
            json.dump(data, f)
        import document_engine
        result = document_engine._load_index_manifest()
        self.assertEqual(result, data)

    def test_parses_legacy_newline_format(self):
        with open(self._path(), "w", encoding="utf-8") as f:
            f.write("file_a.txt\nfile_b.txt\n")
        import document_engine
        result = document_engine._load_index_manifest()
        self.assertIn("file_a.txt", result)
        self.assertIn("file_b.txt", result)

    def test_returns_empty_dict_for_corrupt_json(self):
        with open(self._path(), "w", encoding="utf-8") as f:
            f.write("{bad json")
        import document_engine
        result = document_engine._load_index_manifest()
        self.assertEqual(result, {})


class TestWriteIndexManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        import document_engine
        self._orig = document_engine.INDEXED_FILES_PATH
        document_engine.INDEXED_FILES_PATH = os.path.join(self.tmp, "indexed_files.txt")

    def tearDown(self):
        import document_engine
        document_engine.INDEXED_FILES_PATH = self._orig

    def test_writes_valid_json(self):
        data = {"some/file.txt": {"mtime": 123.4, "size": 500}}
        import document_engine
        document_engine._write_index_manifest(data)
        with open(document_engine.INDEXED_FILES_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded, data)

    def test_roundtrip(self):
        data = {"a.docx": {"mtime": 1.0, "size": 1}, "b.pdf": {"mtime": 2.0, "size": 2}}
        import document_engine
        document_engine._write_index_manifest(data)
        result = document_engine._load_index_manifest()
        self.assertEqual(result, data)


class TestGetFileSignature(unittest.TestCase):
    def test_returns_mtime_and_size(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello world")
            path = f.name
        try:
            import document_engine
            sig = document_engine._get_file_signature(path)
            self.assertIn("mtime", sig)
            self.assertIn("size", sig)
            self.assertEqual(sig["size"], 11)
        finally:
            os.unlink(path)


class TestUpdateManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        import document_engine
        self._orig = document_engine.INDEXED_FILES_PATH
        document_engine.INDEXED_FILES_PATH = os.path.join(self.tmp, "indexed_files.txt")

    def tearDown(self):
        import document_engine
        document_engine.INDEXED_FILES_PATH = self._orig

    def test_add_action_writes_signature(self):
        with tempfile.NamedTemporaryFile(dir=self.tmp, delete=False, suffix=".txt") as f:
            f.write(b"data")
            path = f.name
        import document_engine
        document_engine._update_manifest("add", path)
        manifest = document_engine._load_index_manifest()
        self.assertIn(path, manifest)
        self.assertIn("mtime", manifest[path])

    def test_delete_action_removes_entry(self):
        with tempfile.NamedTemporaryFile(dir=self.tmp, delete=False, suffix=".txt") as f:
            f.write(b"data")
            path = f.name
        import document_engine
        document_engine._update_manifest("add", path)
        document_engine._update_manifest("delete", path)
        manifest = document_engine._load_index_manifest()
        self.assertNotIn(path, manifest)

    def test_add_nonexistent_file_does_not_crash(self):
        import document_engine
        document_engine._update_manifest("add", "/nonexistent/path/file.txt")
        manifest = document_engine._load_index_manifest()
        self.assertNotIn("/nonexistent/path/file.txt", manifest)


# ---------------------------------------------------------------------------
# Graph routing
# ---------------------------------------------------------------------------

class TestDecideToGenerate(unittest.TestCase):
    def test_returns_generate_when_documents_present(self):
        import document_engine
        state = {"documents": [MagicMock()], "loop_step": 0}
        self.assertEqual(document_engine.decide_to_generate(state), "generate")

    def test_returns_transform_query_when_no_docs_and_under_limit(self):
        import document_engine
        state = {"documents": [], "loop_step": 0}
        self.assertEqual(document_engine.decide_to_generate(state), "transform_query")

    def test_returns_generate_when_no_docs_but_at_limit(self):
        import document_engine
        state = {"documents": [], "loop_step": 3}
        self.assertEqual(document_engine.decide_to_generate(state), "generate")

    def test_returns_generate_when_loop_step_exceeds_limit(self):
        import document_engine
        state = {"documents": [], "loop_step": 5}
        self.assertEqual(document_engine.decide_to_generate(state), "generate")

    def test_missing_loop_step_defaults_to_zero(self):
        import document_engine
        state = {"documents": []}
        self.assertEqual(document_engine.decide_to_generate(state), "transform_query")


# ---------------------------------------------------------------------------
# File loading dispatch
# ---------------------------------------------------------------------------

class TestLoadDocumentByExtension(unittest.TestCase):
    def test_returns_empty_list_for_nonexistent_file(self):
        import document_engine
        result = document_engine.load_document_by_extension("/nonexistent/path/file.pdf")
        self.assertEqual(result, [])

    def test_returns_empty_list_for_unsupported_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            path = f.name
        try:
            import document_engine
            result = document_engine.load_document_by_extension(path)
            self.assertEqual(result, [])
        finally:
            os.unlink(path)

    def _make_temp(self, suffix):
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        f.write(b"dummy content")
        f.close()
        return f.name

    def test_pdf_dispatches_to_pypdf_loader(self):
        path = self._make_temp(".pdf")
        try:
            mock_doc = MagicMock()
            with patch("document_engine.load_pdf", return_value=[mock_doc]) as mock_load:
                import document_engine
                result = document_engine.load_document_by_extension(path)
            mock_load.assert_called_once_with(path)
            self.assertEqual(result, [mock_doc])
        finally:
            os.unlink(path)

    def test_docx_dispatches_to_word_loader(self):
        path = self._make_temp(".docx")
        try:
            mock_doc = MagicMock()
            with patch("document_engine.UnstructuredWordDocumentLoader") as MockLoader:
                MockLoader.return_value.load.return_value = [mock_doc]
                import document_engine
                result = document_engine.load_document_by_extension(path)
            self.assertEqual(result, [mock_doc])
        finally:
            os.unlink(path)

    def test_txt_dispatches_to_text_loader(self):
        path = self._make_temp(".txt")
        try:
            mock_doc = MagicMock()
            with patch("document_engine.TextLoader") as MockLoader:
                MockLoader.return_value.load.return_value = [mock_doc]
                import document_engine
                result = document_engine.load_document_by_extension(path)
            self.assertEqual(result, [mock_doc])
        finally:
            os.unlink(path)

    def test_md_dispatches_to_text_loader(self):
        path = self._make_temp(".md")
        try:
            mock_doc = MagicMock()
            with patch("document_engine.TextLoader") as MockLoader:
                MockLoader.return_value.load.return_value = [mock_doc]
                import document_engine
                result = document_engine.load_document_by_extension(path)
            self.assertEqual(result, [mock_doc])
        finally:
            os.unlink(path)

    def test_loader_exception_returns_empty_list(self):
        path = self._make_temp(".pdf")
        try:
            with patch("document_engine.load_pdf", side_effect=RuntimeError("corrupted")):
                import document_engine
                result = document_engine.load_document_by_extension(path)
            self.assertEqual(result, [])
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# query_documents entry point
# ---------------------------------------------------------------------------

class TestQueryDocuments(unittest.TestCase):
    def _make_stream_output(self, generation="Test answer", docs=None):
        """Build the sequence of dicts that app.stream would yield."""
        if docs is None:
            docs = []
        return [
            {"retrieve": {"documents": docs, "question": "q"}},
            {"grade_documents": {"documents": docs, "question": "q"}},
            {"generate_rag": {"generation": generation, "documents": docs}},
        ]

    def test_returns_answer_string_by_default(self):
        import document_engine
        with patch.object(document_engine, "GLOBAL_VECTORSTORE", MagicMock()):
            with patch.object(document_engine.app, "stream", return_value=self._make_stream_output("Hello!")):
                result = document_engine.query_documents("some question")
        self.assertEqual(result, "Hello!")

    def test_include_sources_returns_dict_with_citations(self):
        from langchain_core.documents import Document
        mock_doc = Document(page_content="chunk text", metadata={"source": "/input/lore.pdf", "page": 0})
        import document_engine
        with patch.object(document_engine, "GLOBAL_VECTORSTORE", MagicMock()):
            with patch.object(document_engine.app, "stream",
                              return_value=self._make_stream_output("Answer", [mock_doc])):
                result = document_engine.query_documents("some question", include_sources=True)
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIn("citations", result)
        self.assertEqual(result["answer"], "Answer")
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0]["file"], "lore.pdf")
        self.assertEqual(result["citations"][0]["location"], "Page 1")

    def test_returns_error_string_on_exception(self):
        import document_engine
        with patch.object(document_engine, "GLOBAL_VECTORSTORE", MagicMock()):
            with patch.object(document_engine.app, "stream", side_effect=RuntimeError("boom")):
                result = document_engine.query_documents("some question")
        self.assertIn("Error", result)

    def test_returns_error_dict_on_exception_with_include_sources(self):
        import document_engine
        with patch.object(document_engine, "GLOBAL_VECTORSTORE", MagicMock()):
            with patch.object(document_engine.app, "stream", side_effect=RuntimeError("boom")):
                result = document_engine.query_documents("q", include_sources=True)
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)

    def test_handles_none_final_state(self):
        """If the graph stream yields nothing, query_documents should not crash."""
        import document_engine
        with patch.object(document_engine, "GLOBAL_VECTORSTORE", MagicMock()):
            with patch.object(document_engine.app, "stream", return_value=iter([])):
                result = document_engine.query_documents("q")
        self.assertEqual(result, "No answer generated.")

    def test_handles_none_final_state_with_include_sources(self):
        import document_engine
        with patch.object(document_engine, "GLOBAL_VECTORSTORE", MagicMock()):
            with patch.object(document_engine.app, "stream", return_value=iter([])):
                result = document_engine.query_documents("q", include_sources=True)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["answer"], "No answer generated.")
        self.assertEqual(result["citations"], [])

    def test_uses_generate_rag_state_over_last_node(self):
        """generate_rag_state should win even if another node outputs last (edge case)."""
        import document_engine
        # Simulate: generate_rag fires, then some extra bookkeeping node fires after
        stream_output = [
            {"retrieve": {"documents": [], "question": "q"}},
            {"generate_rag": {"generation": "Correct answer", "documents": []}},
            {"some_other_node": {"documents": [], "question": "q"}},  # last node, wrong state
        ]
        with patch.object(document_engine, "GLOBAL_VECTORSTORE", MagicMock()):
            with patch.object(document_engine.app, "stream", return_value=iter(stream_output)):
                result = document_engine.query_documents("q")
        self.assertEqual(result, "Correct answer")

    def test_initializes_vectorstore_if_none(self):
        import document_engine
        with patch.object(document_engine, "GLOBAL_VECTORSTORE", None):
            with patch.object(document_engine, "initialize_vectorstore") as mock_init:
                with patch.object(document_engine.app, "stream",
                                  return_value=self._make_stream_output("ok")):
                    document_engine.query_documents("q")
        mock_init.assert_called_once()


# ---------------------------------------------------------------------------
# trigger_reindex
# ---------------------------------------------------------------------------

class TestTriggerReindex(unittest.TestCase):
    def test_queues_supported_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Create a mix of supported and unsupported files
            for name in ["doc.pdf", "notes.txt", "data.xlsx", "ignore.exe", "~$tmp.docx"]:
                open(os.path.join(tmp, name), "w").close()

            import document_engine
            orig_folder = document_engine.DOC_FOLDER
            document_engine.DOC_FOLDER = tmp
            try:
                count = document_engine.trigger_reindex()
            finally:
                document_engine.DOC_FOLDER = orig_folder

        # Only doc.pdf, notes.txt, data.xlsx should be queued (not .exe or ~$ temp files)
        self.assertEqual(count, 3)

    def test_returns_zero_for_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            import document_engine
            orig_folder = document_engine.DOC_FOLDER
            document_engine.DOC_FOLDER = tmp
            try:
                count = document_engine.trigger_reindex()
            finally:
                document_engine.DOC_FOLDER = orig_folder
        self.assertEqual(count, 0)


# ---------------------------------------------------------------------------
# Content-hash duplicate detection
# ---------------------------------------------------------------------------

class TestGetDuplicateSource(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        import document_engine
        self._orig_path = document_engine.INDEXED_FILES_PATH
        document_engine.INDEXED_FILES_PATH = os.path.join(self.tmp, "indexed_files.txt")

    def tearDown(self):
        import document_engine
        document_engine.INDEXED_FILES_PATH = self._orig_path

    def _write_file(self, name: str, content: bytes) -> str:
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as f:
            f.write(content)
        return path

    def test_returns_none_when_no_manifest(self):
        import document_engine
        path = self._write_file("new.txt", b"hello")
        self.assertIsNone(document_engine.get_duplicate_source(path))

    def test_returns_none_when_no_match(self):
        import document_engine
        existing = self._write_file("existing.txt", b"different content")
        document_engine._update_manifest("add", existing)
        new = self._write_file("new.txt", b"totally different")
        self.assertIsNone(document_engine.get_duplicate_source(new))

    def test_detects_identical_content_under_different_name(self):
        import document_engine
        content = b"identical content"
        existing = self._write_file("existing.txt", content)
        document_engine._update_manifest("add", existing)
        new = self._write_file("copy.txt", content)
        result = document_engine.get_duplicate_source(new)
        self.assertEqual(result, "existing.txt")

    def test_does_not_flag_self_as_duplicate(self):
        import document_engine
        content = b"some content"
        path = self._write_file("file.txt", content)
        document_engine._update_manifest("add", path)
        # Same path re-checked should not flag itself
        result = document_engine.get_duplicate_source(path)
        self.assertIsNone(result)

    def test_returns_none_for_nonexistent_file(self):
        import document_engine
        self.assertIsNone(document_engine.get_duplicate_source("/nonexistent/file.txt"))


# ---------------------------------------------------------------------------
# Scoped document query
# ---------------------------------------------------------------------------

class TestQueryDocumentsScoped(unittest.TestCase):
    def _make_stream_output(self, generation="Scoped answer"):
        from langchain_core.documents import Document
        doc = Document(page_content="chunk", metadata={"source": "/input/lore.txt", "start_index": 0})
        return [
            {"retrieve": {"documents": [doc], "question": "q", "scoped_source": "/input/lore.txt"}},
            {"grade_documents": {"documents": [doc], "question": "q"}},
            {"generate_rag": {"generation": generation, "documents": [doc]}},
        ]

    def test_returns_not_found_message_for_unknown_filename(self):
        import document_engine
        with patch.object(document_engine, "GLOBAL_VECTORSTORE", MagicMock()):
            with patch.object(document_engine, "_load_index_manifest", return_value={}):
                result = document_engine.query_documents_scoped("q", "ghost.pdf")
        self.assertIn("ghost.pdf", result)
        self.assertIn("/status", result)

    def test_returns_answer_for_known_filename(self):
        import document_engine
        manifest = {"/input/lore.txt": {"mtime": 1.0, "size": 10}}
        with patch.object(document_engine, "GLOBAL_VECTORSTORE", MagicMock()):
            with patch.object(document_engine, "_load_index_manifest", return_value=manifest):
                with patch.object(document_engine.app, "stream",
                                  return_value=self._make_stream_output("Scoped answer")):
                    result = document_engine.query_documents_scoped("q", "lore.txt")
        self.assertEqual(result, "Scoped answer")

    def test_sets_scoped_source_in_graph_inputs(self):
        import document_engine
        manifest = {"/input/lore.txt": {"mtime": 1.0, "size": 10}}
        captured_inputs = {}

        def capturing_stream(inputs, config=None):
            captured_inputs.update(inputs)
            return iter(self._make_stream_output())

        with patch.object(document_engine, "GLOBAL_VECTORSTORE", MagicMock()):
            with patch.object(document_engine, "_load_index_manifest", return_value=manifest):
                with patch.object(document_engine.app, "stream", side_effect=capturing_stream):
                    document_engine.query_documents_scoped("q", "lore.txt")

        self.assertEqual(captured_inputs.get("scoped_source"), "/input/lore.txt")

    def test_not_found_returns_dict_with_include_sources(self):
        import document_engine
        with patch.object(document_engine, "GLOBAL_VECTORSTORE", MagicMock()):
            with patch.object(document_engine, "_load_index_manifest", return_value={}):
                result = document_engine.query_documents_scoped("q", "ghost.pdf", include_sources=True)
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertEqual(result["citations"], [])


# ---------------------------------------------------------------------------
# Embeddings cache
# ---------------------------------------------------------------------------

class TestGetEmbeddings(unittest.TestCase):
    def setUp(self):
        import document_engine
        self._orig = document_engine._EMBEDDINGS
        document_engine._EMBEDDINGS = None

    def tearDown(self):
        import document_engine
        document_engine._EMBEDDINGS = self._orig

    def test_returns_ollama_embeddings_instance(self):
        from langchain_ollama import OllamaEmbeddings
        import document_engine
        result = document_engine._get_embeddings()
        self.assertIsInstance(result, OllamaEmbeddings)

    def test_returns_same_instance_on_second_call(self):
        import document_engine
        first = document_engine._get_embeddings()
        second = document_engine._get_embeddings()
        self.assertIs(first, second)

    def test_cached_instance_not_recreated(self):
        import document_engine
        mock_embed = MagicMock()
        document_engine._EMBEDDINGS = mock_embed
        result = document_engine._get_embeddings()
        self.assertIs(result, mock_embed)


# ---------------------------------------------------------------------------
# Retrieval k value
# ---------------------------------------------------------------------------

class TestRetrieveUsesK4(unittest.TestCase):
    def test_retrieve_calls_get_retriever_with_k4(self):
        import document_engine
        captured = {}

        def fake_get_retriever(k=4, source_filter=None):
            captured["k"] = k
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = []
            return mock_retriever

        with patch.object(document_engine, "get_retriever", side_effect=fake_get_retriever):
            document_engine.retrieve({"question": "test", "documents": [], "loop_step": 0})

        self.assertEqual(captured.get("k"), 4)


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

class TestShutdown(unittest.TestCase):
    def test_shutdown_puts_sentinel_in_queue(self):
        import document_engine
        # Drain any leftover items queued by earlier tests
        while not document_engine.INGESTION_QUEUE.empty():
            try:
                document_engine.INGESTION_QUEUE.get_nowait()
            except Exception:
                break
        orig_observer = document_engine._OBSERVER
        document_engine._OBSERVER = None
        try:
            document_engine.shutdown()
            sentinel = document_engine.INGESTION_QUEUE.get_nowait()
            self.assertIsNone(sentinel)
        finally:
            document_engine._OBSERVER = orig_observer

    def test_shutdown_stops_observer_if_running(self):
        import document_engine
        mock_observer = MagicMock()
        orig = document_engine._OBSERVER
        document_engine._OBSERVER = mock_observer
        try:
            document_engine.shutdown()
        finally:
            document_engine._OBSERVER = orig
            # Drain sentinel so other tests are not affected
            try:
                document_engine.INGESTION_QUEUE.get_nowait()
            except Exception:
                pass
        mock_observer.stop.assert_called_once()
        mock_observer.join.assert_called_once()

    def test_shutdown_safe_when_observer_is_none(self):
        import document_engine
        orig = document_engine._OBSERVER
        document_engine._OBSERVER = None
        try:
            document_engine.shutdown()  # Should not raise
        finally:
            document_engine._OBSERVER = orig
            try:
                document_engine.INGESTION_QUEUE.get_nowait()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
