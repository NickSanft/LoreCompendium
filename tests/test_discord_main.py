"""
Tests for discord_main.py

Covers pure utility functions (split_into_chunks, _validate_query,
_check_rate_limit) since the Discord event handlers require a live bot
connection.
"""
import os
import sys
import time
import unittest
from unittest.mock import patch

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
