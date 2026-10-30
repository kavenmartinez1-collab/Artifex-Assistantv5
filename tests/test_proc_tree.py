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
