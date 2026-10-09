"""
Artifex Assistant V5 — Purge: shut Artifex down and delete what it kept.

One implementation behind the phone app, the Qt GUI and the CLI. It runs
as its own detached process (python -m core.purge --run) because it has
to stop the app that asked for it: Windows won't delete a log file the
API or GUI still holds open, and the API can't delete itself mid-request.

Deleted:
  - chats (sessions/), agent runs, uploads and generated content (output/)
  - logs/ (Artifex's own log folder)
  - the web tool cache and the knowledge base (both built from chats)
  - extra files listed in purge_extra.txt (gitignored, one path or glob
    per line), for logs kept outside the repo that carry chat text, e.g.
    the API's redirected console log

Never touched: models, configs, the venv, and tracked placeholder files
(.gitkeep), so the folders keep working after a purge.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

from core.config import BASE_DIR, KNOWLEDGE_DIR, SESSION_DIR

EXTRA_FILE = os.path.join(BASE_DIR, "purge_extra.txt")

# Scripts whose processes count as "Artifex running"
_ENTRY_SCRIPTS = ("main_api.py", "main_gui_qt.py", "main_gui.py", "main.py")


def targets() -> list[tuple[str, str]]:
    """(label, path-or-glob) for everything a purge deletes."""
    from tools.tool_cache import CACHE_DIR
    t = [
        ("Chats", SESSION_DIR),
        ("Agent runs, uploads and generated content", os.path.join(BASE_DIR, "output")),
        ("Artifex logs", os.path.join(BASE_DIR, "logs")),
        ("Web tool cache", CACHE_DIR),
        ("Knowledge base", KNOWLEDGE_DIR),
    ]
    if os.path.isfile(EXTRA_FILE):
        with open(EXTRA_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    t.append(("Extra: " + line, line))
    return t


def _files(path: str) -> list[str]:
    """Every deletable file under a target (keeps .gitkeep & co.)."""
    if os.path.isdir(path):
        out = []
        for root, _, names in os.walk(path):
            out += [os.path.join(root, n) for n in names if not n.startswith(".git")]
        return out
    return [p for p in glob.glob(path) if os.path.isfile(p)]


def preview() -> list[dict]:
    """What a purge would delete right now: label, path, file count, bytes."""
    rows = []
    for label, path in targets():
        files = _files(path)
        size = 0
        for p in files:
            try:
                size += os.path.getsize(p)
            except OSError:
                pass
        rows.append({"label": label, "path": path, "files": len(files), "bytes": size})
    return rows


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def spawn(delay: float = 2.0) -> int:
    """Start the purge as a detached process; returns its pid.

    `delay` lets the caller finish (answer the HTTP request, close the
    window) before its process is stopped.
    """
    flags = 0
    if sys.platform == "win32":
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                 | subprocess.CREATE_NO_WINDOW)
    proc = subprocess.Popen(
        [sys.executable, "-m", "core.purge", "--run", "--delay", str(delay)],
        cwd=BASE_DIR, creationflags=flags, close_fds=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return proc.pid


def _is_artifex(proc, base: str) -> bool:
    try:
        name = (proc.info["name"] or "").lower()
        if name.startswith("llama-server"):
            return True
        if not name.startswith("python"):
            return False
        cmd = proc.info["cmdline"] or []
        script = next((c for c in cmd[1:] if c.lower().endswith(".py")), "")
        if os.path.basename(script).lower() not in _ENTRY_SCRIPTS:
            return False
        path = script if os.path.isabs(script) else os.path.join(proc.cwd(), script)
        return os.path.normcase(os.path.dirname(os.path.abspath(path))) == base
    except Exception:
        return False


def stop_artifex() -> int:
    """Kill the API, GUIs, CLI and llama-server; returns how many."""
    import psutil
    base = os.path.normcase(os.path.abspath(BASE_DIR))
    me = os.getpid()
    victims = [p for p in psutil.process_iter(["pid", "name", "cmdline"])
               if p.pid != me and _is_artifex(p, base)]
    for p in victims:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(victims, timeout=10)
    return len(victims)


def delete_all(retries: int = 5) -> tuple[int, int, list[str]]:
    """Delete every target's files, then its empty subfolders.

    Returns (files deleted, bytes reclaimed, files that stayed locked).
    """
    count = size = 0
    locked: list[str] = []
    for _, path in targets():
        for p in _files(path):
            for attempt in range(retries):
                try:
                    n = os.path.getsize(p)
                    os.remove(p)
                    count += 1
                    size += n
                    break
                except FileNotFoundError:
                    break
                except OSError:
                    if attempt == retries - 1:
                        locked.append(p)
                    else:
                        time.sleep(1)   # a just-killed process may still hold it
        if os.path.isdir(path):
            for root, dirs, _ in os.walk(path, topdown=False):
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))   # only succeeds when empty
                    except OSError:
                        pass
    return count, size, locked


def run(delay: float = 0.0) -> None:
    time.sleep(delay)
    stopped = stop_artifex()
    time.sleep(1)   # let Windows release the killed processes' file handles
    count, size, locked = delete_all()
    print(f"Stopped {stopped} Artifex process(es). Deleted {count} files "
          f"({fmt_bytes(size)}).")
    for p in locked:
        print("  still locked, not deleted:", p)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Shut Artifex down and purge its data.")
    ap.add_argument("--run", action="store_true", help="actually purge (otherwise preview)")
    ap.add_argument("--delay", type=float, default=0.0)
    a = ap.parse_args()
    if a.run:
        run(a.delay)
    else:
        for r in preview():
            print(f"{r['label']}: {r['files']} files, {fmt_bytes(r['bytes'])}  ({r['path']})")
