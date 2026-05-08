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


class TestTruncateRepetition:
    """_truncate_repetition must catch real verbatim loops without
    amputating structured output (JSON arrays, equipment schedules, etc.)
    where the same boilerplate repeats with item-specific data between."""

    def test_genuine_loop_truncated(self):
        """Same 80+ char phrase back-to-back 3 times = real loop."""
        from core.inference import _truncate_repetition
        # Exactly 80 chars repeated back-to-back so phrase_len=80 finds
        # consecutive copies with gap=0.
        phrase = "ABCDE" * 16  # 80 chars
        text = phrase * 4
        result = _truncate_repetition(text)
        # Should cut at start of second occurrence — keeps just the first.
        assert result == phrase.rstrip()

    def test_structured_json_with_repeating_boilerplate_kept(self):
        """JSON array of items sharing a long category string must NOT
        be truncated — the data between boilerplate occurrences is
        large enough that the gap-check rejects it as a false loop."""
        from core.inference import _truncate_repetition
        # Three items sharing a long category. Boilerplate is >80 chars.
        # Item-specific data ("01", "04", "11" + qty + model) is the
        # interpolated content between repetitions.
        item_template = (
            '{{"item_number": "{n}", "qty": {q}, "model": "{m}", '
            '"equipment_category": "Underbar Fillers & Drainboards", '
            '"remarks": ""}}'
        )
        items = [
            item_template.format(n="01", q=8, m="KR24-R30"),
            item_template.format(n="04", q=3, m="KR24-R15"),
            item_template.format(n="11", q=1, m="KR24-GS12-PE"),
            item_template.format(n="15", q=1, m="KR24-DR45"),
        ]
        text = "[" + ",\n".join(items) + "]"
        result = _truncate_repetition(text)
        # All four items must survive — the gap between the boilerplate
        # is too large for a real loop.
        assert "KR24-R30" in result
        assert "KR24-R15" in result
        assert "KR24-GS12-PE" in result
        assert "KR24-DR45" in result

    def test_short_text_unchanged(self):
        """Below the 4×min_phrase_len floor, no analysis runs."""
        from core.inference import _truncate_repetition
        text = "short text"
        assert _truncate_repetition(text) == text

    def test_two_occurrences_unchanged(self):
        """Two copies isn't enough — could be a legitimate header/footer
        echo or a structural duplicate."""
        from core.inference import _truncate_repetition
        phrase = "The same eighty character phrase right here exactly to test the threshold."
        # Pad to ensure len >= 4*80
        text = phrase + phrase + (" " * 200)
        # Two occurrences only.  Should not truncate.
        result = _truncate_repetition(text)
        assert result.count(phrase) == 2
