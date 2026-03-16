"""
Tests for lore_utils.py

Covers config loading and Ollama health checks.
All external calls (filesystem, network) are isolated via temp files and mocks.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGetKeyFromJsonConfigFile(unittest.TestCase):
    def _call(self, key, default, path):
        """Call the function with a custom config path (bypasses module-level constant)."""
        import lore_utils
        original = lore_utils._CONFIG_PATH
        lore_utils._CONFIG_PATH = path
        try:
            return lore_utils.get_key_from_json_config_file(key, default)
        finally:
            lore_utils._CONFIG_PATH = original

    def test_returns_value_for_existing_key(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"my_key": "my_value"}, f)
            path = f.name
        try:
            result = self._call("my_key", "default", path)
            self.assertEqual(result, "my_value")
        finally:
            os.unlink(path)

    def test_returns_default_for_missing_key(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"other_key": "value"}, f)
            path = f.name
        try:
            result = self._call("my_key", "fallback", path)
            self.assertEqual(result, "fallback")
        finally:
            os.unlink(path)

    def test_returns_default_when_file_missing(self):
        result = self._call("any_key", "fallback", "/nonexistent/path/config.json")
        self.assertEqual(result, "fallback")

    def test_returns_default_for_invalid_json(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("{this is not valid json")
            path = f.name
        try:
            result = self._call("any_key", "fallback", path)
            self.assertEqual(result, "fallback")
        finally:
            os.unlink(path)

    def test_returns_default_for_null_value(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"my_key": None}, f)
            path = f.name
        try:
            result = self._call("my_key", "fallback", path)
            self.assertEqual(result, "fallback")
        finally:
            os.unlink(path)

    def test_returns_default_for_empty_string_value(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"my_key": ""}, f)
            path = f.name
        try:
            result = self._call("my_key", "fallback", path)
            self.assertEqual(result, "fallback")
        finally:
            os.unlink(path)


class TestCheckOllamaHealth(unittest.TestCase):
    def _make_response(self, models: list[str]):
        """Build a fake urllib response for the given model names."""
        payload = json.dumps({"models": [{"name": m} for m in models]}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_no_errors_when_all_models_present(self):
        import lore_utils
        models = [lore_utils.THINKING_OLLAMA_MODEL, lore_utils.FAST_OLLAMA_MODEL, lore_utils.EMBEDDING_MODEL]
        with patch("urllib.request.urlopen", return_value=self._make_response(models)):
            errors = lore_utils.check_ollama_health()
        self.assertEqual(errors, [])

    def test_error_when_model_missing(self):
        import lore_utils
        # Only include the fast and embedding model; thinking model is absent
        models = [lore_utils.FAST_OLLAMA_MODEL, lore_utils.EMBEDDING_MODEL]
        with patch("urllib.request.urlopen", return_value=self._make_response(models)):
            errors = lore_utils.check_ollama_health()
        self.assertTrue(any(lore_utils.THINKING_OLLAMA_MODEL in e for e in errors))

    def test_error_when_ollama_unreachable(self):
        import lore_utils
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            errors = lore_utils.check_ollama_health()
        self.assertTrue(len(errors) >= 1)
        self.assertTrue(any("localhost:11434" in e for e in errors))

    def test_model_matches_with_tag_suffix(self):
        """A model like 'llama3.2:latest' should satisfy a requirement for 'llama3.2'."""
        import lore_utils
        versioned = [
            lore_utils.THINKING_OLLAMA_MODEL + ":latest",
            lore_utils.FAST_OLLAMA_MODEL + ":latest",
            lore_utils.EMBEDDING_MODEL + ":latest",
        ]
        with patch("urllib.request.urlopen", return_value=self._make_response(versioned)):
            errors = lore_utils.check_ollama_health()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
