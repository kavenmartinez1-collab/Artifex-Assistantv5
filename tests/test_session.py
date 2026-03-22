"""Tests for core/session.py — save, load, list, auto-save."""

import os
import json
import tempfile
import shutil
import pytest

from core.session import (
    save_session, load_session, list_sessions, find_session, auto_save,
)


@pytest.fixture
def session_dir():
    d = tempfile.mkdtemp(prefix="session_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestSaveAndLoad:
    """Test session save and load round-trip."""

    def test_save_creates_file(self, session_dir):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        path = save_session("test", messages, session_dir=session_dir)
        assert os.path.isfile(path)
        assert path.endswith(".json")

    def test_load_restores_messages(self, session_dir):
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        path = save_session("test", messages, session_dir=session_dir)
        state = load_session(path)

        assert state is not None
        # System messages are stripped on save
        assert len(state.messages) == 2
        assert state.messages[0]["content"] == "Hello"

    def test_save_with_metadata(self, session_dir):
        messages = [{"role": "user", "content": "Test"}]
        metadata = {"model": "qwen3.5-9b", "backend": "transformers"}
        path = save_session("meta", messages, metadata=metadata, session_dir=session_dir)
        state = load_session(path)
        assert state.metadata["model"] == "qwen3.5-9b"

    def test_load_nonexistent(self, session_dir):
        state = load_session(os.path.join(session_dir, "nonexistent.json"))
        assert state is None

    def test_load_corrupt_file(self, session_dir):
        path = os.path.join(session_dir, "corrupt.json")
        with open(path, "w") as f:
            f.write("not valid json{{{")
        state = load_session(path)
        assert state is None


class TestListSessions:
    """Test session listing."""

    def test_empty_dir(self, session_dir):
        sessions = list_sessions(session_dir)
        assert sessions == []

    def test_lists_saved_sessions(self, session_dir):
        for name in ("alpha", "beta", "gamma"):
            save_session(name, [{"role": "user", "content": f"Msg from {name}"}],
                        session_dir=session_dir)

        sessions = list_sessions(session_dir)
        assert len(sessions) == 3
        # Should have name and path
        assert all("name" in s for s in sessions)
        assert all("path" in s for s in sessions)

    def test_sorted_by_recency(self, session_dir):
        import time
        save_session("old", [{"role": "user", "content": "Old"}],
                    session_dir=session_dir)
        time.sleep(0.1)
        save_session("new", [{"role": "user", "content": "New"}],
                    session_dir=session_dir)

        sessions = list_sessions(session_dir)
        assert sessions[0]["name"] == "new"


class TestFindSession:
    """Test session lookup by name or index."""

    def test_find_by_name(self, session_dir):
        save_session("myproject", [{"role": "user", "content": "Test"}],
                    session_dir=session_dir)
        path = find_session("myproject", session_dir)
        assert path is not None

    def test_find_by_index(self, session_dir):
        save_session("first", [{"role": "user", "content": "1"}],
                    session_dir=session_dir)
        path = find_session("1", session_dir)
        assert path is not None

    def test_find_nonexistent(self, session_dir):
        path = find_session("nothing", session_dir)
        assert path is None


class TestAutoSave:
    """Test auto-save rotation."""

    def test_auto_save_creates_file(self, session_dir):
        messages = [{"role": "user", "content": "Auto"}]
        path = auto_save(messages, session_dir=session_dir)
        assert os.path.isfile(path)

    def test_auto_save_rotates(self, session_dir):
        """Multiple auto-saves create rotating files."""
        for i in range(4):
            auto_save([{"role": "user", "content": f"Msg {i}"}],
                     session_dir=session_dir)
        # Should have _autosave_1, _autosave_2, _autosave_3
        files = os.listdir(session_dir)
        autosaves = [f for f in files if "autosave" in f]
        assert len(autosaves) >= 1
