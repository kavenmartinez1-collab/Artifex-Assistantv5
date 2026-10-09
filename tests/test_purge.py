"""core.purge: what it deletes, what it keeps, and which processes it stops."""
import os

import pytest

from core import purge


@pytest.fixture
def tree(tmp_path, monkeypatch):
    out = tmp_path / "output"
    (out / "agent_runs" / "run1").mkdir(parents=True)
    (out / "agent_runs" / "run1" / "notes.txt").write_text("x")
    (out / "uploads").mkdir()
    (out / "uploads" / "a.jpg").write_bytes(b"12345")
    (out / ".gitkeep").write_text("")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "artifex.log").write_text("goal='secret'")
    (logs / ".gitkeep").write_text("")
    extra = tmp_path / "ext"
    extra.mkdir()
    (extra / "artifex-api.log").write_text("chat")
    (extra / "artifex-api.log.1").write_text("chat")
    (extra / "llama-server.log").write_text("keep")
    monkeypatch.setattr(purge, "targets", lambda: [
        ("Output", str(out)), ("Logs", str(logs)),
        ("Extra", str(extra / "artifex-api.log*")),
    ])
    return tmp_path


def test_preview_counts_without_deleting(tree):
    rows = {r["label"]: r for r in purge.preview()}
    assert rows["Output"]["files"] == 2          # .gitkeep not counted
    assert rows["Output"]["bytes"] == 6
    assert rows["Extra"]["files"] == 2
    assert (tree / "output" / "uploads" / "a.jpg").exists()


def test_delete_all_keeps_placeholders_and_unlisted_files(tree):
    count, size, locked = purge.delete_all(retries=1)
    assert (count, locked) == (5, [])
    assert sorted(os.listdir(tree / "output")) == [".gitkeep"]   # empty subdirs gone
    assert os.listdir(tree / "logs") == [".gitkeep"]
    assert os.listdir(tree / "ext") == ["llama-server.log"]      # glob only


class FakeProc:
    def __init__(self, name, cmdline, cwd="C:\\elsewhere"):
        self.info = {"name": name, "cmdline": cmdline}
        self._cwd = cwd

    def cwd(self):
        return self._cwd


@pytest.mark.parametrize("proc,expected", [
    (FakeProc("llama-server.exe", ["llama-server.exe", "-m", "x.gguf"]), True),
    (FakeProc("python.exe", ["python.exe", r"C:\App\main_api.py", "--port", "8000"]), True),
    (FakeProc("python.exe", ["python.exe", "main_gui_qt.py"], cwd=r"C:\App"), True),
    (FakeProc("python.exe", ["python.exe", r"C:\Other\main_api.py"]), False),   # other repo
    (FakeProc("python.exe", ["python.exe", "-m", "core.purge", "--run"]), False),
    (FakeProc("python.exe", ["python.exe", r"C:\App\some_script.py"]), False),
    (FakeProc("chrome.exe", ["chrome.exe"]), False),
])
def test_is_artifex(proc, expected):
    assert purge._is_artifex(proc, os.path.normcase(r"C:\App")) is expected
