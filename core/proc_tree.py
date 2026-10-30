"""
Artifex Assistant V5 — stopping a model server and everything it started.

An external_launcher entry (e.g. Strata) runs its own server: a launcher
script, a python server and the engine exe under it. The engine's unload and
the purge both stop it the same way: by whatever LISTENs on the entry's port
(so a server Artifex adopted after a restart, not its own child, is found
too), plus any pid the caller still holds, each with all of its children.
"""

from __future__ import annotations


def stop_port_tree(port: int, pids=(), timeout: float = 15.0) -> int:
    """Terminate the process listening on TCP `port` and the processes in
    `pids`, with all of their children; kill what is still alive after
    `timeout` seconds. Returns how many processes were stopped."""
    import psutil
    roots = {int(p) for p in pids if p}
    try:
        conns = psutil.net_connections(kind="tcp")
    except (psutil.Error, OSError):
        conns = []
    for conn in conns:
        if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN and conn.pid:
            roots.add(conn.pid)
    procs = {}
    for pid in roots:
        try:
            p = psutil.Process(pid)
            for c in p.children(recursive=True):
                procs[c.pid] = c
            procs[p.pid] = p
        except psutil.NoSuchProcess:
            continue
    for p in procs.values():
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(list(procs.values()), timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    return len(procs)
