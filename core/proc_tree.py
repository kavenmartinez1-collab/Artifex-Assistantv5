"""
Artifex Assistant V5 — stopping a model server and everything it started.

An external_launcher entry (e.g. Strata) runs its own server: a launcher
script, a python server and the engine exe under it. The engine's unload and
the purge both stop it the same way: by whatever LISTENs on the entry's port
(so a server Artifex adopted after a restart, not its own child, is found
too), plus any pid the caller still holds, each with all of its children.
"""

from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)

# Model servers a front-end started that must die with it (see
# tie_children_to_this_process).
_tie_children = False
_job = None
_owned: list = []


def stop_port_tree(port: int | None, pids=(), timeout: float = 15.0) -> int:
    """Terminate the process listening on TCP `port` and the processes in
    `pids`, with all of their children; kill what is still alive after
    `timeout` seconds. Returns how many processes were stopped."""
    import psutil
    roots = {int(p) for p in pids if p}
    try:
        conns = psutil.net_connections(kind="tcp") if port else []
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


def tie_children_to_this_process() -> None:
    """Make model servers this process starts from now on die with it.

    The API keeps its servers across a restart on purpose (the next run
    adopts them), so only the interactive front-ends call this. Without it
    a CLI or GUI that is closed leaves its server behind: the external
    launcher runs with no console, so closing the window does not reach it,
    and the API's queue, which never loaded it, never unloads it either. A
    leftover Strata held 6.7 GB of the 5060 Ti that way until the next
    27B load was refused.

    On Windows the servers go into a kill-on-close job object, which the OS
    tears down however this process ends (exit, Ctrl+C, the window's X, a
    crash). Elsewhere they are stopped at interpreter exit.
    """
    global _tie_children
    if _tie_children:
        return
    _tie_children = True
    if os.name != "nt":
        import atexit
        atexit.register(_stop_owned)


def spawn(cmd, **kwargs):
    """subprocess.Popen for a model server, tied to this process if
    tie_children_to_this_process() was called.

    On Windows the server starts suspended and is resumed only once it is in
    the job: everything it starts (launcher -> python -> engine exe) then
    inherits the job. Assigning it after it runs races its first child —
    measured: a venv python.exe redirector spawned the real interpreter
    before the assignment landed, and that branch outlived the front-end.
    """
    import subprocess
    tie = _tie_children and os.name == "nt"
    if tie:
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | _CREATE_SUSPENDED
    proc = subprocess.Popen(cmd, **kwargs)
    if not _tie_children:
        return proc
    if not tie:
        _owned.append(proc)
        return proc
    try:
        _assign_to_job(proc)
    except OSError as e:
        _log.warning("Could not tie pid=%d to this process (%s); it may "
                     "outlive it", proc.pid, e)
    try:
        import psutil
        psutil.Process(proc.pid).resume()
    except Exception:
        proc.kill()
        raise
    return proc


_CREATE_SUSPENDED = 0x00000004


def _stop_owned() -> None:
    live = [p.pid for p in _owned if p.poll() is None]
    if live:
        stop_port_tree(None, live, timeout=5.0)


def _assign_to_job(proc) -> None:
    import ctypes
    from ctypes import wintypes

    global _job
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if _job is None:
        class _Basic(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _Extended(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _Basic),
                ("IoInfo", ctypes.c_uint64 * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        job = k32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _Extended()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                           ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        # Never closed: the OS closes it when this process ends, and that
        # is what kills the servers.
        _job = job
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    if not k32.AssignProcessToJobObject(_job, int(proc._handle)):
        raise ctypes.WinError(ctypes.get_last_error())


def orphan_listeners(ports, artifex_root: str) -> list[tuple[int, int]]:
    """(port, pid) of each process listening on one of `ports` that no live
    Artifex process owns.

    Owned means some ancestor is this process or a python running a script
    from the Artifex install (main_api.py, main.py, the Qt launcher, ...);
    a server a running front-end started, or a helper the API itself runs,
    is left alone.  Orphaned additionally means the chain was cut: its
    topmost live ancestor is not a session process (explorer, svchost, ...),
    i.e. whatever started it is gone. A server run by hand from a terminal
    still reaches explorer through that terminal and is left alone too.
    Windows only: elsewhere an orphan is re-parented to init and this
    cannot tell it from a service, so nothing is reported.
    """
    import psutil
    if os.name != "nt":
        return []
    root = os.path.normcase(os.path.abspath(artifex_root)) + os.sep
    me = os.getpid()
    ports = set(ports)
    found = []
    try:
        conns = psutil.net_connections(kind="tcp")
    except (psutil.Error, OSError):
        return found
    for conn in conns:
        if not (conn.laddr and conn.laddr.port in ports
                and conn.status == psutil.CONN_LISTEN and conn.pid):
            continue
        try:
            proc = psutil.Process(conn.pid)
            chain = [proc] + proc.parents()
            if any(p.pid == me or _is_artifex_script(p, root) for p in chain):
                continue
            if chain[-1].name().lower() in _SESSION_ROOTS:
                continue
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        found.append((conn.laddr.port, conn.pid))
    return found


# What the topmost live ancestor of anything started from the desktop, a
# terminal or a service is. A server whose chain tops out elsewhere lost
# the process that launched it.
_SESSION_ROOTS = frozenset({
    "system", "smss.exe", "csrss.exe", "wininit.exe", "winlogon.exe",
    "services.exe", "svchost.exe", "explorer.exe", "sihost.exe",
    "userinit.exe", "taskhostw.exe", "runtimebroker.exe",
    "windowsterminal.exe", "openconsole.exe", "conhost.exe",
})


def _is_artifex_script(proc, root: str) -> bool:
    import psutil
    try:
        scripts = [a for a in proc.cmdline()[1:] if a.lower().endswith(".py")]
        cwd = proc.cwd() if any(not os.path.isabs(a) for a in scripts) else ""
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    return any(
        os.path.normcase(os.path.abspath(os.path.join(cwd, a))).startswith(root)
        for a in scripts)
