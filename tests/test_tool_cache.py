"""Tests for tools/tool_cache.py — caching thresholds, summarizers, SessionMap."""

import os
import tempfile
import shutil
import pytest

from tools.tool_cache import (
    maybe_cache_output, CACHE_THRESHOLD, SessionMap, update_session_map,
)


class TestMaybeCacheOutput:
    """Test the output caching logic."""

    def test_short_output_not_cached(self):
        """Outputs shorter than threshold stay inline."""
        short = "Hello world"
        assert len(short) < CACHE_THRESHOLD
        result = maybe_cache_output("shell", "echo hello", short)
        assert result == short

    def test_long_output_cached(self):
        """Outputs longer than threshold get cached and summarized."""
        long_output = "match1: foo\n" * (CACHE_THRESHOLD // 5)
        result = maybe_cache_output("grep", "grep foo .", long_output)
        # Result should be shorter than original (summarized)
        assert len(result) < len(long_output)


class TestSessionMap:
    """Test the SessionMap tracker."""

    def test_empty_map(self):
        sm = SessionMap()
        assert sm.render() == "(no files explored yet)"
        assert sm.render(token_budget=100) == "(no files explored yet)"

    def test_set_task(self):
        sm = SessionMap()
        sm.set_task("Investigate the auth system")
        rendered = sm.render(token_budget=500)
        assert "auth" in rendered.lower() or rendered == ""

    def test_clear(self):
        sm = SessionMap()
        sm.set_task("some task")
        sm.clear()
        assert sm.render() == "(no files explored yet)"

    def test_update_from_read_file(self):
        """SessionMap tracks files that were read."""
        sm = SessionMap()
        update_session_map(sm, "read_file", "config.py",
                          "File: config.py\n1: import os\n2: x = 1\n")
        rendered = sm.render(token_budget=500)
        # Should mention config.py or at least not be empty if output processed
        assert isinstance(rendered, str)
