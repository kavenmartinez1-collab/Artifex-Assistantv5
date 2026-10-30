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


def test_delete_all_removes_a_workspace_git_repo(tree):
    """Each agent-run workspace is its own git repo (ensure_workspace_git);
    git's object files are read-only, which os.remove refuses on Windows."""
    import stat
    import subprocess
    ws = tree / "output" / "agent_runs" / "run2"
    ws.mkdir()
    subprocess.run(["git", "init", "-q", str(ws)], check=True)
    subprocess.run(["git", "-C", str(ws), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(ws), "config", "user.name", "T"], check=True)
    (ws / "edit.py").write_text("x = 1")
    subprocess.run(["git", "-C", str(ws), "add", "."], check=True)
    subprocess.run(["git", "-C", str(ws), "commit", "-q", "-m", "[agent] edit"], check=True)
    objects = [os.path.join(r, n) for r, _, ns in os.walk(ws / ".git" / "objects") for n in ns]
    assert objects and all(not os.stat(p).st_mode & stat.S_IWRITE for p in objects)

    count, size, locked = purge.delete_all(retries=2)
    assert locked == []
    assert not ws.exists()
    assert sorted(os.listdir(tree / "output")) == [".gitkeep"]


# Built with the OS's own separators: CI runs on Linux, where "C:\App\x.py"
# is neither absolute nor split by basename
APP = os.path.abspath(os.path.join(os.sep, "App"))
OTHER = os.path.abspath(os.path.join(os.sep, "Other"))


class FakeProc:
    def __init__(self, name, cmdline, cwd=os.path.abspath(os.path.join(os.sep, "elsewhere"))):
        self.info = {"name": name, "cmdline": cmdline}
        self._cwd = cwd

    def cwd(self):
        return self._cwd


@pytest.mark.parametrize("proc,expected", [
    (FakeProc("llama-server.exe", ["llama-server.exe", "-m", "x.gguf"]), True),
    (FakeProc("python.exe", ["python.exe", os.path.join(APP, "main_api.py"), "--port", "8000"]), True),
    (FakeProc("python.exe", ["python.exe", "main_gui_qt.py"], cwd=APP), True),
    (FakeProc("python3", ["python3", os.path.join(APP, "main.py")]), True),
    (FakeProc("python.exe", ["python.exe", os.path.join(OTHER, "main_api.py")]), False),  # other repo
    (FakeProc("python.exe", ["python.exe", "-m", "core.purge", "--run"]), False),
    (FakeProc("python.exe", ["python.exe", os.path.join(APP, "some_script.py")]), False),
    (FakeProc("chrome.exe", ["chrome.exe"]), False),
])
def test_is_artifex(proc, expected):
    assert purge._is_artifex(proc, os.path.normcase(APP)) is expected


def test_external_ports_are_the_external_launcher_entries(monkeypatch):
    import core.config
    monkeypatch.setattr(core.config, "get_llama_cpp_models", lambda: {
        "qwen-27b": {"path": "x.gguf", "port": 8081},                                # llama-server: by name
        "flash-next": {"external_launcher": ["powershell", "-File", "s.ps1"], "port": 8085},
        "no-port": {"external_launcher": ["x"]},
        "bad-port": {"external_launcher": ["x"], "port": "8086"},
    })
    assert purge._external_ports() == [8085]


def test_stop_artifex_stops_external_servers_after_artifex(monkeypatch):
    """2026-10-30: a purge left the Strata server (python serve/server.py +
    strata.exe) running - neither llama-server nor an Artifex script."""
    import psutil
    import core.proc_tree
    order = []
    monkeypatch.setattr(psutil, "process_iter", lambda attrs: order.append("artifex") or [])
    monkeypatch.setattr(psutil, "wait_procs", lambda procs, timeout: ([], []))
    monkeypatch.setattr(purge, "_external_ports", lambda: [8085])
    monkeypatch.setattr(core.proc_tree, "stop_port_tree", lambda port, pids=(): order.append(port) or 2)
    assert purge.stop_artifex() == 2
    assert order == ["artifex", 8085]
