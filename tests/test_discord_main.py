"""
Tests for discord_main.py

Only covers pure utility functions (split_into_chunks) since the Discord
event handlers require a live bot connection.
"""
import os
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
