"""Tests for core/inference.py — ThinkFilter, strip_think_blocks, compression."""

import pytest
from core.inference import ThinkFilter, strip_think_blocks, compress_history


class TestThinkFilter:
    """Test the ThinkFilter streaming processor."""

    def test_basic_thinking_then_response(self):
        """Thinking content is separated from response content."""
        thinking = []
        response = []
        tf = ThinkFilter(
            on_response=lambda t: response.append(t),
            on_thinking=lambda t: thinking.append(t),
        )
        # Simulate streaming: thinking, then </think>, then response
        tf.feed("I need to consider")
        tf.feed(" this carefully</think>")
        tf.feed("Here is my answer.")
        tf.flush()

        assert "".join(response) == "Here is my answer."
        full_think = "".join(thinking)
        assert "consider" in full_think

    def test_no_thinking(self):
        """If model starts with </think> immediately, response is captured."""
        response = []
        tf = ThinkFilter(on_response=lambda t: response.append(t))
        tf.feed("</think>Direct response here.")
        tf.flush()
        assert "Direct response" in "".join(response)

    def test_flush_remaining(self):
        """Flush emits remaining buffer."""
        response = []
        tf = ThinkFilter(on_response=lambda t: response.append(t))
        tf.feed("</think>Hello")
        tf.flush()
        assert "Hello" in "".join(response)

    def test_nested_think_blocks(self):
        """Handle re-entry into thinking mid-response."""
        response = []
        thinking = []
        tf = ThinkFilter(
            on_response=lambda t: response.append(t),
            on_thinking=lambda t: thinking.append(t),
        )
        tf.feed("thinking stuff</think>response part<think>more thinking</think>final part")
        tf.flush()
        assert "response part" in "".join(response)
        assert "final part" in "".join(response)

    def test_only_thinking(self):
        """If stream is entirely thinking, nothing goes to response."""
        response = []
        thinking = []
        tf = ThinkFilter(
            on_response=lambda t: response.append(t),
            on_thinking=lambda t: thinking.append(t),
        )
        tf.feed("All thinking content here")
        tf.flush()
        assert len(response) == 0
        assert len(thinking) > 0


class TestStripThinkBlocks:
    """Test strip_think_blocks utility."""

    def test_removes_complete_think_block(self):
        text = "Hello <think>internal reasoning</think> world"
        assert strip_think_blocks(text) == "Hello  world"

    def test_removes_leading_think_block(self):
        text = "some thinking</think>The actual response"
        assert strip_think_blocks(text) == "The actual response"

    def test_no_think_blocks(self):
        text = "Normal text without thinking"
        assert strip_think_blocks(text) == "Normal text without thinking"

    def test_empty_string(self):
        assert strip_think_blocks("") == ""

    def test_multiple_think_blocks(self):
        text = "<think>first</think>A<think>second</think>B"
        result = strip_think_blocks(text)
        assert "A" in result
        assert "B" in result
        assert "first" not in result
        assert "second" not in result


class TestCompressHistory:
    """Test conversation history compression."""

    def test_short_history_unchanged(self):
        """History shorter than context_window is not compressed."""
        history = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = compress_history(history, context_window=6)
        assert len(result) == len(history)

    def test_long_history_compressed(self):
        """History longer than context_window gets compressed."""
        history = [
            {"role": "system", "content": "System prompt"},
        ]
        for i in range(20):
            history.append({"role": "user", "content": f"Question {i}"})
            history.append({"role": "assistant", "content": f"Answer {i}"})

        result = compress_history(history, context_window=6)
        assert len(result) < len(history)
        # System prompt preserved
        assert result[0]["role"] == "system"

    def test_pins_first_user_message(self):
        """First user message (the task) is pinned."""
        history = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "My original task"},
        ]
        for i in range(20):
            history.append({"role": "assistant", "content": f"Response {i}"})
            history.append({"role": "user", "content": f"Follow-up {i}"})

        result = compress_history(history, context_window=4)
        # First user message should be preserved
        user_msgs = [m for m in result if m["role"] == "user"]
        assert any("My original task" in m["content"] for m in user_msgs)

    def test_compressed_has_summary(self):
        """Compressed history contains a summary of old messages."""
        history = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Task"},
        ]
        for i in range(20):
            history.append({"role": "assistant", "content": f"Answer {i} explanation"})
            history.append({"role": "user", "content": f"Question {i}"})

        result = compress_history(history, context_window=4)
        contents = " ".join(m["content"] for m in result)
        assert "EARLIER CONVERSATION" in contents
