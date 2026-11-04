"""core.proc_tree.stop_port_tree: the listener on a port and every process under it."""
import psutil

from core import proc_tree


class FakeProc:
    def __init__(self, pid, children=()):
        self.pid = pid
        self._children = list(children)
        self.terminated = self.killed = False

    def children(self, recursive=False):
        return self._children

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class Conn:
    def __init__(self, port, pid, status=psutil.CONN_LISTEN):
        self.laddr = type("Addr", (), {"port": port})()
        self.pid = pid
        self.status = status


def _fake_psutil(monkeypatch, conns, procs, still_alive=()):
    monkeypatch.setattr(psutil, "net_connections", lambda kind: conns)

    def process(pid):
        if pid not in procs:
            raise psutil.NoSuchProcess(pid)
        return procs[pid]
    monkeypatch.setattr(psutil, "Process", process)
    monkeypatch.setattr(psutil, "wait_procs",
                        lambda ps, timeout: ([p for p in ps if p.pid not in still_alive],
                                             [p for p in ps if p.pid in still_alive]))


def test_stops_the_listener_and_its_children(monkeypatch):
    engine = FakeProc(30)
    server = FakeProc(20, children=[engine])          # python serve/server.py -> strata.exe
    other = FakeProc(40)
    _fake_psutil(monkeypatch,
                 conns=[Conn(8085, 20), Conn(8000, 40), Conn(8085, 50, status="ESTABLISHED")],
                 procs={20: server, 30: engine, 40: other},
                 still_alive={30})
    assert proc_tree.stop_port_tree(8085, timeout=0) == 2
    assert server.terminated and engine.terminated
    assert engine.killed and not server.killed        # what outlived terminate() is killed
    assert not other.terminated                       # another port's listener is left alone


def test_own_pid_and_listener_counted_once(monkeypatch):
    server = FakeProc(20)
    _fake_psutil(monkeypatch, conns=[Conn(8085, 20)], procs={20: server})
    assert proc_tree.stop_port_tree(8085, pids=[20, None], timeout=0) == 1


def test_nothing_listening_and_a_vanished_pid(monkeypatch):
    _fake_psutil(monkeypatch, conns=[], procs={})
    assert proc_tree.stop_port_tree(8085, pids=[99], timeout=0) == 0


# ── orphan_listeners: servers nobody owns any more ─────────────────────────

import os
import subprocess
import sys
import textwrap
import time

import pytest

windows_only = pytest.mark.skipif(os.name != "nt", reason="job objects / orphan scan are Windows-only")

ROOT = os.path.join(os.path.abspath(os.sep), "ArtifexAi")


class ChainProc:
    def __init__(self, pid, name, cmdline, parents=(), cwd=ROOT):
        self.pid, self._name, self._cmdline, self._cwd = pid, name, cmdline, cwd
        self._parents = list(parents)

    def parents(self):
        return self._parents

    def name(self):
        return self._name

    def cmdline(self):
        return self._cmdline

    def cwd(self):
        return self._cwd


def _chain(*specs):
    """specs from the listener up: (pid, name, cmdline)."""
    procs = []
    for pid, name, cmd in reversed(specs):
        procs.insert(0, ChainProc(pid, name, cmd, parents=list(procs)))
    return procs[0]


@windows_only
def test_orphan_is_a_server_whose_launcher_lost_its_parent(monkeypatch):
    # The 2026-11-03 leftover: CLI gone, launcher chain tops out at powershell.
    strata = _chain((20, "python.exe", ["python", r"D:\models\Strata\serve\server.py"]),
                    (19, "python.exe", [r"venv\python", r"D:\models\Strata\serve\server.py"]),
                    (18, "powershell.exe", ["powershell", "-File", r"D:\models\start-strata.ps1"]))
    monkeypatch.setattr(psutil, "net_connections", lambda kind: [Conn(8086, 20)])
    monkeypatch.setattr(psutil, "Process", lambda pid: {20: strata}[pid])
    assert proc_tree.orphan_listeners({8086}, ROOT) == [(8086, 20)]


@windows_only
@pytest.mark.parametrize("chain", [
    # started by a live Artifex front-end (absolute and relative script paths)
    [(20, "llama-server.exe", ["llama-server"]),
     (10, "python.exe", ["python", os.path.join(ROOT, "main_api.py")])],
    [(20, "llama-server.exe", ["llama-server"]),
     (10, "python.exe", ["python", "main.py"])],
    # run by hand from a terminal: reaches explorer
    [(20, "python.exe", ["python", "server.py"]),
     (18, "powershell.exe", ["powershell"]),
     (17, "WindowsTerminal.exe", ["wt"]),
     (2, "explorer.exe", ["explorer"])],
])
def test_owned_or_hand_run_servers_are_left_alone(monkeypatch, chain):
    top = _chain(*chain)
    monkeypatch.setattr(psutil, "net_connections", lambda kind: [Conn(8086, 20)])
    monkeypatch.setattr(psutil, "Process", lambda pid: top)
    assert proc_tree.orphan_listeners({8086}, ROOT) == []


@windows_only
def test_orphan_scan_ignores_other_ports(monkeypatch):
    lone = _chain((20, "llama-server.exe", ["llama-server"]))
    monkeypatch.setattr(psutil, "net_connections", lambda kind: [Conn(9999, 20)])
    monkeypatch.setattr(psutil, "Process", lambda pid: lone)
    assert proc_tree.orphan_listeners({8086}, ROOT) == []


# ── tie_children_to_this_process: a hard-killed front-end takes its servers ─

_FRONTEND = textwrap.dedent("""
    import subprocess, sys, time
    sys.path.insert(0, sys.argv[1])
    from core.proc_tree import tie_children_to_this_process, spawn
    tie_children_to_this_process()
    # launcher -> grandchild, like powershell -> python -> strata.exe
    launcher = spawn([sys.executable, "-c",
        "import subprocess,sys,time;"
        "g=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        "print(g.pid,flush=True);time.sleep(60)"], stdout=subprocess.PIPE, text=True)
    import os
    print(os.getpid(), launcher.pid, launcher.stdout.readline().strip(), flush=True)
    time.sleep(60)
""")


@windows_only
def test_servers_die_when_the_front_end_is_killed():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    front = subprocess.Popen([sys.executable, "-c", _FRONTEND, repo],
                             stdout=subprocess.PIPE, text=True)
    try:
        me, *pids = [int(p) for p in front.stdout.readline().split()]
        assert len(pids) == 2 and all(psutil.pid_exists(p) for p in pids)
        # TerminateProcess on the interpreter itself (a venv python.exe is a
        # redirector in front of it): no atexit, no cleanup code runs.
        psutil.Process(me).kill()
        deadline = time.monotonic() + 10
        while any(psutil.pid_exists(p) for p in pids) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not any(psutil.pid_exists(p) for p in pids)
    finally:
        front.kill()
