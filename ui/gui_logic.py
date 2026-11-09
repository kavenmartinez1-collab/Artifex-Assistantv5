"""
Artifex Assistant V5 — GUI decisions that need no Qt.

Small pure functions the PyQt6 window (ui/qt_gui.py) and its workers call,
kept free of Qt imports so they can be tested on a machine with no display.
"""

import sys
import time


def parse_max_tokens(text):
    """The Tokens box as a number: -1 for "max"/blank, a positive int, or
    None when it is neither (the caller tells the user instead of raising
    inside a Qt slot, where the click then silently did nothing)."""
    t = (text or "").strip().lower()
    if t in ("", "max"):
        return -1
    try:
        n = int(t)
    except ValueError:
        return None
    return n if n > 0 else None


def restored_messages(system_message, saved_messages):
    """History for a loaded session: the live system prompt followed by the
    saved turns. Saved sessions carry no system message (core.session strips
    it), so nothing is dropped from the front."""
    return [system_message] + [
        m for m in (saved_messages or [])
        if isinstance(m, dict) and m.get("role") != "system"]


def history_after_cancel(messages, partial):
    """Update `messages` in place after a cancelled generation; returns it.

    The partial answer the user saw is kept as the assistant turn. With no
    visible text at all, the unanswered user turn is dropped instead: the
    model never replied, and two user turns in a row break chat templates
    that require alternating roles.
    """
    if (partial or "").strip():
        messages.append({"role": "assistant", "content": partial})
    elif len(messages) > 1 and messages[-1].get("role") == "user":
        messages.pop()
    return messages


def wait_for_workers(workers, timeout_s=5.0, clock=time.monotonic):
    """Wait for running QThread-like workers, sharing one deadline.

    Each worker needs isRunning() and wait(msecs). Returns the workers still
    running when the time is up (empty list: all stopped).
    """
    deadline = clock() + max(0.0, timeout_s)
    for w in workers:
        if w is None or not w.isRunning():
            continue
        remaining_ms = int(max(0.0, deadline - clock()) * 1000)
        w.wait(max(1, remaining_ms))
    return [w for w in workers if w is not None and w.isRunning()]


def vram_summary(devices):
    """Device-wide VRAM across every adapter: (used_gb, total_gb, fraction, detail).

    `devices` is a list of {"name", "used_mb", "total_mb"} dicts. Returns
    None when there is nothing to show. This is the whole card, all
    processes: a llama-server or Ollama model lives in another process and
    never showed up in this process's torch allocator.
    """
    rows = [d for d in (devices or []) if (d.get("total_mb") or 0) > 0]
    if not rows:
        return None
    used = sum(max(0.0, float(d.get("used_mb") or 0)) for d in rows)
    total = sum(float(d["total_mb"]) for d in rows)
    detail = " + ".join(
        f"{d.get('name') or 'GPU'} {max(0.0, float(d.get('used_mb') or 0)) / 1024:.1f}/"
        f"{float(d['total_mb']) / 1024:.0f}G" for d in rows)
    return used / 1024, total / 1024, max(0.0, min(1.0, used / total)), detail


def read_vram_devices():
    """[{name, used_mb, total_mb}] for every GPU, system-wide.

    Never imports torch (chat-only installs have none; importing it just for
    a meter costs over a second) and never raises.
    """
    try:
        if sys.platform == "win32":
            # DXGI + PDH: all vendors, pure ctypes, no subprocess.
            from core.dxgi_vram import probe_adapters
            return [{"name": a.get("description", "GPU"),
                     "used_mb": a.get("usage_mb", 0.0),
                     "total_mb": a.get("dedicated_mb", 0.0)}
                    for a in probe_adapters()]
        torch = sys.modules.get("torch")   # only if something already loaded it
        if torch is not None and torch.cuda.is_available():
            out = []
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                out.append({"name": torch.cuda.get_device_name(i),
                            "used_mb": (total - free) / 1048576,
                            "total_mb": total / 1048576})
            return out
        from core.gpu_pool import get_pool
        return [{"name": d.get("name", "GPU"),
                 "used_mb": d.get("memory_used_mb", 0.0),
                 "total_mb": d.get("memory_total_mb", 0.0)}
                for d in get_pool().get_device_status()]
    except Exception:
        return []
