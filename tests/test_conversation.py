"""
Tests for conversation.py

Covers pure helper functions (format_prompt, get_source_info,
get_system_description) and the ask_stuff entry point with a mocked LangGraph
app so no Ollama connection is needed.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGetSourceInfo(unittest.TestCase):
    def test_discord_text_mentions_discord_and_user(self):
        from conversation import get_source_info
        from lore_utils import MessageSource
        result = get_source_info(MessageSource.DISCORD_TEXT, "user123")
        self.assertIn("Discord", result)
        self.assertIn("user123", result)

    def test_discord_voice_includes_word_limit(self):
        from conversation import get_source_info
        from lore_utils import MessageSource
        result = get_source_info(MessageSource.DISCORD_VOICE, "user123")
        self.assertIn("30 words", result)

    def test_local_source_mentions_cli(self):
        from conversation import get_source_info
        from lore_utils import MessageSource
        result = get_source_info(MessageSource.LOCAL, "user123")
        self.assertIn("CLI", result)


class TestFormatPrompt(unittest.TestCase):
    def test_includes_prompt_and_user_id(self):
        from conversation import format_prompt
        from lore_utils import MessageSource
        result = format_prompt("What is lore?", MessageSource.DISCORD_TEXT, "alice")
        self.assertIn("What is lore?", result)
        self.assertIn("alice", result)

    def test_prompt_is_a_string(self):
        from conversation import format_prompt
        from lore_utils import MessageSource
        result = format_prompt("hello", MessageSource.LOCAL, "bob")
        self.assertIsInstance(result, str)


class TestGetSystemDescription(unittest.TestCase):
    def test_contains_system_description_constant(self):
        from conversation import get_system_description
        from lore_utils import SYSTEM_DESCRIPTION
        result = get_system_description()
        self.assertIn(SYSTEM_DESCRIPTION, result)

    def test_mentions_search_documents_tool(self):
        from conversation import get_system_description
        result = get_system_description()
        self.assertIn("search_documents", result)


class TestAskStuff(unittest.TestCase):
    def _make_stream_output(self, content="Test response"):
        mock_msg = MagicMock()
        mock_msg.content = content
        return [{"messages": [mock_msg]}]

    def test_returns_string_response(self):
        import conversation
        from lore_utils import MessageSource
        stream_output = self._make_stream_output("Hello there!")
        with patch.object(conversation.app, "stream", return_value=iter(stream_output)):
            result = conversation.ask_stuff("What is lore?", "user1", MessageSource.DISCORD_TEXT)
        self.assertEqual(result, "Hello there!")

    def test_strips_special_chars_from_user_id(self):
        """User IDs with special characters should not crash ask_stuff."""
        import conversation
        from lore_utils import MessageSource
        stream_output = self._make_stream_output("Response")
        with patch.object(conversation.app, "stream", return_value=iter(stream_output)):
            result = conversation.ask_stuff("q", "user@#$%123", MessageSource.DISCORD_TEXT)
        self.assertIsInstance(result, str)

    def test_returns_empty_string_when_stream_yields_nothing(self):
        import conversation
        from lore_utils import MessageSource
        with patch.object(conversation.app, "stream", return_value=iter([])):
            result = conversation.ask_stuff("q", "user1", MessageSource.DISCORD_TEXT)
        self.assertEqual(result, "")

    def test_passes_cleaned_user_id_as_thread_id(self):
        """ask_stuff should use the cleaned user ID in the LangGraph config."""
        import conversation
        from lore_utils import MessageSource
        stream_output = self._make_stream_output("Response")
        captured_config = {}
        original_stream = conversation.app.stream

        def capturing_stream(inputs, config, **kwargs):
            captured_config.update(config)
            return iter(stream_output)

        with patch.object(conversation.app, "stream", side_effect=capturing_stream):
            conversation.ask_stuff("q", "user!@#123", MessageSource.DISCORD_TEXT)

        thread_id = captured_config.get("configurable", {}).get("thread_id")
        self.assertEqual(thread_id, "user123")


class TestHistoryTrimming(unittest.TestCase):
    def test_history_trimmed_to_max(self):
        """conversation() should trim history to MAX_HISTORY_MESSAGES before passing to the agent."""
        import conversation
        from unittest.mock import MagicMock, patch

        captured_inputs = {}

        def fake_stream(inputs, config=None, stream_mode=None):
            captured_inputs.update(inputs)
            mock_msg = MagicMock()
            mock_msg.content = "reply"
            mock_msg.tool_calls = []
            return iter([{"messages": [mock_msg]}])

        # Build a state with more messages than the limit
        over_limit = conversation.MAX_HISTORY_MESSAGES + 10
        fake_messages = [MagicMock() for _ in range(over_limit)]
        for m in fake_messages:
            m.content = "msg"
            m.tool_calls = []

        state = {"messages": fake_messages}
        config = {"metadata": {"user_id": "u1", "thread_id": "u1"}}

        with patch.object(conversation.conversation_react_agent, "stream", side_effect=fake_stream):
            conversation.conversation(state, config)

        passed_messages = captured_inputs.get("messages", [])
        # Subtract 1 for the system message prepended inside conversation()
        history_passed = [m for m in passed_messages if not (isinstance(m, tuple) and m[0] == "system")]
        self.assertLessEqual(len(history_passed), conversation.MAX_HISTORY_MESSAGES)

    def test_short_history_not_trimmed(self):
        """If history is under the limit, all messages should be passed through."""
        import conversation
        from unittest.mock import MagicMock, patch

        captured_inputs = {}

        def fake_stream(inputs, config=None, stream_mode=None):
            captured_inputs.update(inputs)
            mock_msg = MagicMock()
            mock_msg.content = "reply"
            mock_msg.tool_calls = []
            return iter([{"messages": [mock_msg]}])

        short_history = [MagicMock() for _ in range(5)]
        for m in short_history:
            m.content = "msg"
            m.tool_calls = []

        state = {"messages": short_history}
        config = {"metadata": {"user_id": "u1", "thread_id": "u1"}}

        with patch.object(conversation.conversation_react_agent, "stream", side_effect=fake_stream):
            conversation.conversation(state, config)

        passed_messages = captured_inputs.get("messages", [])
        history_passed = [m for m in passed_messages if not (isinstance(m, tuple) and m[0] == "system")]
        self.assertEqual(len(history_passed), 5)


if __name__ == "__main__":
    unittest.main()
