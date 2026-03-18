"""
Tests for discord_main.py

Covers pure utility functions (split_into_chunks, _validate_query,
_check_rate_limit) since the Discord event handlers require a live bot
connection.
"""
import asyncio
import os
import sys
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSplitIntoChunks(unittest.TestCase):
    def _split(self, text, size=2000):
        from discord_main import split_into_chunks
        return split_into_chunks(text, size)

    def test_short_string_returns_single_chunk(self):
        result = self._split("hello world")
        self.assertEqual(result, ["hello world"])

    def test_exactly_chunk_size_returns_single_chunk(self):
        text = "x" * 2000
        result = self._split(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], text)

    def test_one_over_chunk_size_returns_two_chunks(self):
        text = "x" * 2001
        result = self._split(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "x" * 2000)
        self.assertEqual(result[1], "x")

    def test_chunks_cover_full_content(self):
        text = "ab" * 3000
        result = self._split(text)
        self.assertEqual("".join(result), text)

    def test_all_chunks_within_size_limit(self):
        text = "z" * 9999
        result = self._split(text)
        for chunk in result:
            self.assertLessEqual(len(chunk), 2000)

    def test_empty_string_returns_empty_list(self):
        result = self._split("")
        self.assertEqual(result, [])

    def test_custom_chunk_size(self):
        text = "a" * 10
        result = self._split(text, size=3)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], "aaa")
        self.assertEqual(result[-1], "a")

    # --- Boundary-aware behaviour ---

    def test_prefers_newline_boundary_over_hard_cut(self):
        # Newline is at or past the halfway mark — should split there
        text = "hello\nworld and more text here"
        result = self._split(text, size=10)
        # No chunk may exceed the size limit
        self.assertFalse(any(len(c) > 10 for c in result))
        # The word "hello" must not be split across chunks
        joined = " ".join(result)
        self.assertIn("hello", joined)

    def test_prefers_double_newline_over_single(self):
        # Double newline past the halfway mark should be chosen over single newline
        text = "paragraph one\n\nparagraph two extra words here"
        result = self._split(text, size=20)
        self.assertFalse(any(len(c) > 20 for c in result))
        # "paragraph one" should be intact in some chunk
        self.assertTrue(any("paragraph one" in c for c in result))

    def test_falls_back_to_sentence_boundary(self):
        # No newlines, but a sentence end past the halfway mark of the window
        text = "First sentence. " + "x" * 20
        result = self._split(text, size=25)
        self.assertFalse(any(len(c) > 25 for c in result))
        # "First sentence" should be intact in the first chunk
        self.assertTrue(any("First sentence" in c for c in result))

    def test_content_preserved_with_boundary_splitting(self):
        text = "Line one.\nLine two.\nLine three.\nLine four.\nLine five."
        result = self._split(text, size=30)
        # All chunks within limit
        self.assertFalse(any(len(c) > 30 for c in result))
        # Every line's content should appear in some chunk
        for line in ["Line one", "Line two", "Line three", "Line four", "Line five"]:
            self.assertTrue(any(line in c for c in result), f"'{line}' missing from chunks")


class TestValidateQuery(unittest.TestCase):
    def test_valid_query_returns_none(self):
        from discord_main import _validate_query
        self.assertIsNone(_validate_query("What happened at the Battle of Helm's Deep?"))

    def test_empty_string_is_invalid(self):
        from discord_main import _validate_query
        self.assertIsNotNone(_validate_query(""))

    def test_whitespace_only_is_invalid(self):
        from discord_main import _validate_query
        self.assertIsNotNone(_validate_query("   "))

    def test_query_at_max_length_is_valid(self):
        from discord_main import _validate_query, _MAX_QUERY_LENGTH
        self.assertIsNone(_validate_query("a" * _MAX_QUERY_LENGTH))

    def test_query_over_max_length_is_invalid(self):
        from discord_main import _validate_query, _MAX_QUERY_LENGTH
        result = _validate_query("a" * (_MAX_QUERY_LENGTH + 1))
        self.assertIsNotNone(result)
        self.assertIn(str(_MAX_QUERY_LENGTH), result)


class TestFmtSize(unittest.TestCase):
    def test_bytes(self):
        from discord_main import _fmt_size
        self.assertEqual(_fmt_size(0), "0 B")
        self.assertEqual(_fmt_size(512), "512 B")

    def test_kilobytes(self):
        from discord_main import _fmt_size
        self.assertIn("KB", _fmt_size(2048))

    def test_megabytes(self):
        from discord_main import _fmt_size
        self.assertIn("MB", _fmt_size(2 * 1024 * 1024))

    def test_gigabytes(self):
        from discord_main import _fmt_size
        self.assertIn("GB", _fmt_size(2 * 1024 ** 3))


class TestStreamToInteraction(unittest.IsolatedAsyncioTestCase):
    async def test_edits_with_cursor_then_final_edit(self):
        import queue as q_module
        from discord_main import _stream_to_interaction

        interaction = MagicMock()
        interaction.edit_original_response = AsyncMock()

        sq = q_module.Queue()
        sq.put("Hello ")
        sq.put("world")
        sq.put(None)  # sentinel

        task = asyncio.ensure_future(asyncio.sleep(0))
        await task

        await _stream_to_interaction(interaction, sq, task)
        interaction.edit_original_response.assert_called()

    async def test_exits_cleanly_when_task_done_and_queue_empty(self):
        import queue as q_module
        from discord_main import _stream_to_interaction

        interaction = MagicMock()
        interaction.edit_original_response = AsyncMock()

        sq = q_module.Queue()  # empty queue, no sentinel
        task = asyncio.ensure_future(asyncio.sleep(0))
        await task  # mark done

        # Should return without hanging
        await asyncio.wait_for(_stream_to_interaction(interaction, sq, task), timeout=2.0)


class TestClassifyError(unittest.TestCase):
    def test_connection_error_mentions_ollama(self):
        from discord_main import _classify_error
        result = _classify_error(ConnectionRefusedError("connection refused"))
        self.assertIn("Ollama", result)

    def test_timeout_error_mentions_ollama(self):
        from discord_main import _classify_error
        result = _classify_error(TimeoutError("timeout"))
        self.assertIn("Ollama", result)

    def test_generic_error_gives_fallback_message(self):
        from discord_main import _classify_error
        result = _classify_error(RuntimeError("something else entirely"))
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_connection_string_in_message_triggers_ollama_hint(self):
        from discord_main import _classify_error
        result = _classify_error(RuntimeError("cannot reach connection endpoint"))
        self.assertIn("Ollama", result)


class TestCheckRateLimit(unittest.TestCase):
    def setUp(self):
        import discord_main
        # Reset the dict before each test to avoid cross-test pollution
        discord_main._user_last_query.clear()

    def test_first_request_is_allowed(self):
        from discord_main import _check_rate_limit
        self.assertEqual(_check_rate_limit("user1"), 0.0)

    def test_immediate_second_request_is_blocked(self):
        from discord_main import _check_rate_limit
        _check_rate_limit("user1")
        remaining = _check_rate_limit("user1")
        self.assertGreater(remaining, 0)

    def test_different_users_are_independent(self):
        from discord_main import _check_rate_limit
        _check_rate_limit("user1")
        # user2 has never queried — should be allowed
        self.assertEqual(_check_rate_limit("user2"), 0.0)

    def test_request_allowed_after_cooldown(self):
        import discord_main
        from discord_main import _check_rate_limit, _RATE_LIMIT_SECONDS
        # Manually set the last query time to well in the past
        past = time.time() - (_RATE_LIMIT_SECONDS + 1)
        discord_main._user_last_query["user1"] = past
        self.assertEqual(_check_rate_limit("user1"), 0.0)

    def test_blocked_returns_positive_remaining_time(self):
        from discord_main import _check_rate_limit
        _check_rate_limit("user1")
        remaining = _check_rate_limit("user1")
        self.assertGreater(remaining, 0)
        self.assertLessEqual(remaining, 10)


if __name__ == "__main__":
    unittest.main()
