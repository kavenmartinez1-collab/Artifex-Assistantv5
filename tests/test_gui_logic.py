"""GUI decisions that need no Qt (ui/gui_logic.py), plus the session
round-trip the GUI's Save / Load buttons rely on."""
import inspect
import os

from core.session import load_session, save_session
from ui import gui_logic
from ui.gui_logic import (
    history_after_cancel, parse_max_tokens, read_vram_devices,
    restored_messages, vram_summary, wait_for_workers,
)


# ── Tokens box ───────────────────────────────────────────────────────────

def test_tokens_box_max_and_blank_mean_unlimited():
    assert parse_max_tokens("max") == -1
    assert parse_max_tokens("  MAX ") == -1
    assert parse_max_tokens("") == -1
    assert parse_max_tokens(None) == -1


def test_tokens_box_number():
    assert parse_max_tokens(" 2048 ") == 2048


def test_tokens_box_garbage_is_reported_not_raised():
    # int("4k") used to raise inside the EXECUTE slot: the click did nothing.
    for bad in ("4k", "abc", "1.5", "0", "-5"):
        assert parse_max_tokens(bad) is None


# ── Save / Load session ──────────────────────────────────────────────────

def test_gui_save_then_load_round_trip(tmp_path):
    """Save writes the chosen folder + name; Load reads that path back and
    keeps every saved turn behind the live system prompt."""
    chosen = tmp_path / "elsewhere" / "my chat.json"
    live = [{"role": "system", "content": "live system prompt"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"}]
    saved = save_session("my chat", live, {}, {"backend": "x"},
                         session_dir=str(chosen.parent), filename=chosen.name)
    assert saved == str(chosen)
    assert os.listdir(chosen.parent) == ["my chat.json"]

    state = load_session(saved)
    assert state is not None
    system = {"role": "system", "content": "a newer system prompt"}
    messages = restored_messages(system, state.messages)
    assert messages[0] is system
    # the first user turn survives (the old code sliced it off as "system")
    assert [m["content"] for m in messages[1:]] == ["first question", "first answer"]


def test_load_by_bare_name_was_never_a_path(tmp_path):
    saved = save_session("s", [{"role": "user", "content": "q"}],
                         session_dir=str(tmp_path), filename="s.json")
    assert load_session("s") is None            # what the GUI used to pass
    assert load_session(saved) is not None      # what it passes now


def test_restored_messages_drops_stray_system_turns():
    out = restored_messages({"role": "system", "content": "S"},
                            [{"role": "system", "content": "old"},
                             {"role": "user", "content": "q"}, "junk"])
    assert out == [{"role": "system", "content": "S"},
                   {"role": "user", "content": "q"}]


# ── Cancel ───────────────────────────────────────────────────────────────

def test_cancel_keeps_the_partial_answer():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]
    history_after_cancel(msgs, "Half an ans")
    assert msgs[-1] == {"role": "assistant", "content": "Half an ans"}
    assert "[Generation cancelled]" not in str(msgs)


def test_cancel_with_nothing_shown_drops_the_unanswered_turn():
    msgs = [{"role": "system", "content": "S"},
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}]
    history_after_cancel(msgs, "   ")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]


def test_cancel_never_removes_the_system_prompt():
    msgs = [{"role": "system", "content": "S"}]
    assert history_after_cancel(msgs, "") == [{"role": "system", "content": "S"}]


# ── Shutdown ─────────────────────────────────────────────────────────────

class _Worker:
    def __init__(self, stops_after_ms=None):
        self.stops_after_ms = stops_after_ms    # None: never stops
        self.running = True
        self.waits = []

    def isRunning(self):
        return self.running

    def wait(self, ms):
        self.waits.append(ms)
        if self.stops_after_ms is not None and ms >= self.stops_after_ms:
            self.running = False
        return not self.running


def test_wait_for_workers_returns_only_the_stuck_ones():
    done = _Worker()
    done.running = False
    quick, stuck = _Worker(stops_after_ms=10), _Worker()
    left = wait_for_workers([None, done, quick, stuck], timeout_s=1.0)
    assert left == [stuck]
    assert done.waits == []            # finished workers are not waited on
    assert quick.waits and stuck.waits


def test_wait_for_workers_shares_one_deadline():
    now = [0.0]

    class _Slow(_Worker):
        def wait(self, ms):
            self.waits.append(ms)
            now[0] += ms / 1000        # uses up everything it was given
            return False

    a, b = _Slow(), _Slow()
    left = wait_for_workers([a, b], timeout_s=2.0, clock=lambda: now[0])
    assert left == [a, b]
    assert a.waits == [2000]
    assert b.waits == [1]              # nothing left: the minimum, not 2 s more


# ── VRAM meter ───────────────────────────────────────────────────────────

def test_vram_summary_adds_up_every_adapter():
    used, total, frac, detail = vram_summary([
        {"name": "Card A", "used_mb": 6144, "total_mb": 8192},
        {"name": "Card B", "used_mb": 2048, "total_mb": 12288},
        {"name": "stub", "used_mb": 0, "total_mb": 0},
    ])
    assert (used, total) == (8.0, 20.0)
    assert frac == 0.4
    assert "Card A 6.0/8G" in detail and "Card B 2.0/12G" in detail
    assert "stub" not in detail


def test_vram_summary_nothing_to_show():
    assert vram_summary([]) is None
    assert vram_summary(None) is None
    assert vram_summary([{"name": "x", "used_mb": 1, "total_mb": 0}]) is None


def test_vram_summary_fraction_is_clamped():
    assert vram_summary([{"name": "x", "used_mb": 9000, "total_mb": 8192}])[2] == 1.0


def test_vram_devices_come_from_the_system_wide_probe(monkeypatch):
    import core.dxgi_vram as dxgi
    monkeypatch.setattr(gui_logic.sys, "platform", "win32")
    monkeypatch.setattr(dxgi, "probe_adapters", lambda: [
        {"description": "Card A", "usage_mb": 5000.0, "dedicated_mb": 8192.0}])
    assert read_vram_devices() == [
        {"name": "Card A", "used_mb": 5000.0, "total_mb": 8192.0}]


def test_vram_devices_never_raise(monkeypatch):
    import core.dxgi_vram as dxgi

    def boom():
        raise OSError("no DXGI here")

    monkeypatch.setattr(gui_logic.sys, "platform", "win32")
    monkeypatch.setattr(dxgi, "probe_adapters", boom)
    assert read_vram_devices() == []


def test_vram_meter_never_imports_torch():
    # chat-only installs have no torch; the old meter did `import torch`.
    src = inspect.getsource(gui_logic)
    assert "import torch" not in src
