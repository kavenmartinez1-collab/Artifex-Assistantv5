"""Qt worker and chat-widget behaviour, run headless (offscreen platform, no
window is ever shown). Skipped where PyQt6 or a Qt platform is unavailable."""
import os
import threading

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtCore", exc_type=ImportError)

try:
    from ui.qt_workers import BackendSwitchWorker, GenerationWorker  # noqa: E402
except ImportError as e:   # a Qt library this machine cannot load
    pytest.skip(f"ui.qt_workers not importable here: {e}", allow_module_level=True)


class _Engine:
    """Streams `chunks` through on_token; records the kwargs it was given."""

    def __init__(self, chunks, starts_in_think=None, cancel_after=None, cancel_event=None):
        self.chunks = chunks
        self.cancel_after = cancel_after
        self.cancel_event = cancel_event
        self.kwargs = None
        if starts_in_think is not None:
            self.stream_starts_in_think = starts_in_think

    def is_loaded(self):
        return True

    def generate_streaming(self, messages, max_tokens, temperature, on_token=None, **kw):
        self.kwargs = kw
        for i, chunk in enumerate(self.chunks):
            if self.cancel_after is not None and i == self.cancel_after:
                self.cancel_event.set()
            on_token(chunk)
        return "".join(self.chunks)


def _run(engine, enable_thinking=True, cancel_event=None, want_telemetry=None):
    """Run the worker's body on this thread and collect what it emitted."""
    worker = GenerationWorker(engine, [{"role": "user", "content": "hi"}], 64, 0.7,
                              enable_thinking=enable_thinking, cancel_event=cancel_event)
    if want_telemetry is not None:
        worker.want_telemetry = want_telemetry
    got = {"response": [], "thinking": [], "finished": [], "error": []}
    worker.response_received.connect(got["response"].append)
    worker.thinking_received.connect(got["thinking"].append)
    worker.finished.connect(got["finished"].append)
    worker.error.connect(got["error"].append)
    worker.run()
    assert got["error"] == []
    return worker, got


def test_answer_of_an_engine_that_starts_outside_think_goes_to_the_chat():
    # llama.cpp / claude CLI emit explicit <think> tags; the worker used to
    # treat every non-Ollama engine as starting INSIDE a think block, so a
    # model with reasoning off streamed its whole answer into the thinking pane.
    _, got = _run(_Engine(["Hello ", "there."], starts_in_think=False))
    assert "".join(got["response"]) == "Hello there."
    assert got["thinking"] == []


def test_explicit_think_block_is_still_split_out():
    _, got = _run(_Engine(["<think>plan</think>", "Answer."], starts_in_think=False))
    assert "".join(got["thinking"]) == "plan"
    assert "".join(got["response"]).strip() == "Answer."


def test_engine_that_prefills_think_still_starts_in_think():
    _, got = _run(_Engine(["plan</think>", "Answer."], starts_in_think=True))
    assert "".join(got["thinking"]) == "plan"
    assert "".join(got["response"]).strip() == "Answer."


def test_thinking_disabled_never_starts_in_think():
    _, got = _run(_Engine(["Just the answer."], starts_in_think=True),
                  enable_thinking=False)
    assert "".join(got["response"]) == "Just the answer."
    assert got["thinking"] == []


def test_cancel_reports_the_partial_answer_not_a_placeholder():
    cancel = threading.Event()
    engine = _Engine(["Half ", "an ", "answer"], starts_in_think=False,
                     cancel_after=2, cancel_event=cancel)
    worker, got = _run(engine, cancel_event=cancel)
    assert worker.cancelled is True
    assert got["finished"] == ["Half an "]
    assert "[Generation cancelled]" not in got["finished"][0]


def test_finished_run_is_not_marked_cancelled():
    worker, got = _run(_Engine(["Done."], starts_in_think=False))
    assert worker.cancelled is False
    assert got["finished"] == ["Done."]


def test_telemetry_is_only_requested_when_wanted():
    off = _Engine(["x"], starts_in_think=False)
    _run(off)                                     # default: tab not showing
    assert off.kwargs["on_telemetry"] is None
    on = _Engine(["x"], starts_in_think=False)
    _run(on, want_telemetry=True)
    assert callable(on.kwargs["on_telemetry"])


def test_backend_switch_worker_reports_models(monkeypatch):
    import core.config as config
    calls = []
    monkeypatch.setattr(config, "set_active_backend", lambda name: calls.append(name))
    monkeypatch.setattr(config, "get_model_names", lambda: ["m1", "m2"])
    worker = BackendSwitchWorker("ollama")
    got = []
    worker.switched.connect(lambda backend, names: got.append((backend, names)))
    worker.run()
    assert calls == ["ollama"]
    assert got == [("ollama", ["m1", "m2"])]


def test_backend_switch_worker_survives_a_failing_probe(monkeypatch):
    import core.config as config

    def boom(name):
        raise OSError("unreachable")

    monkeypatch.setattr(config, "set_active_backend", boom)
    worker = BackendSwitchWorker("ollama")
    got = []
    worker.switched.connect(lambda backend, names: got.append((backend, names)))
    worker.run()
    assert got == [("ollama", [])]


# ── chat widgets (need a QApplication; offscreen, never shown) ────────────

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
    pytest.importorskip("PyQt6.QtMultimedia", exc_type=ImportError)
    from PyQt6.QtWidgets import QApplication
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as e:      # no usable Qt platform on this machine
        pytest.skip(f"no Qt platform: {e}")
    return app


def test_short_reply_is_sized_when_the_stream_ends(qapp):
    from ui.qt_widgets import ChatBubble
    bubble = ChatBubble("assistant")
    bubble.add_text("")
    bubble.append_text("A one-line answer.")
    edit = bubble._stream_widget
    # under 500 chars the throttle never fired: no fixed height at all, so
    # the widget sat at its default 192 px size hint
    assert edit.maximumHeight() > 10000
    bubble.finalize_stream()
    assert edit.minimumHeight() == edit.maximumHeight()
    assert 30 <= edit.maximumHeight() < 120


def test_tail_of_a_long_reply_is_sized_when_the_stream_ends(qapp):
    from ui.qt_widgets import ChatBubble
    bubble = ChatBubble("assistant")
    bubble.add_text("")
    for _ in range(5):
        bubble.append_text("word " * 20 + "\n")           # 505 chars: sized once
    before = bubble._stream_widget.maximumHeight()
    for _ in range(4):
        bubble.append_text("word " * 20 + "\n")           # +404: below the throttle
    assert bubble._stream_widget.maximumHeight() == before   # tail not sized yet
    bubble.finalize_stream()
    assert bubble._stream_widget.maximumHeight() > before
    assert bubble._last_height_update_at_chars == bubble._stream_text_len


def test_finalize_without_a_stream_is_a_no_op(qapp):
    from ui.qt_widgets import ChatBubble
    bubble = ChatBubble("assistant")
    bubble.add_text("plain label")
    bubble.finalize_stream()
    assert bubble._stream_widget is None


def test_at_bottom():
    pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
    pytest.importorskip("PyQt6.QtMultimedia", exc_type=ImportError)
    from ui.qt_widgets import ChatView
    assert ChatView.at_bottom(1000, 1000)
    assert ChatView.at_bottom(980, 1000)
    assert not ChatView.at_bottom(500, 1000)
    assert ChatView.at_bottom(0, 0)


def test_streaming_does_not_pull_a_scrolled_up_view_back_down(qapp):
    from ui.qt_widgets import ChatView
    view = ChatView()
    scheduled = []
    view._scroll_to_bottom = lambda: scheduled.append(1)

    view.scroll_to_bottom()                 # at the bottom: follows
    assert len(scheduled) == 1

    bar = view.verticalScrollBar()
    bar.setRange(0, 1000)
    bar.setValue(1000)
    bar.setValue(200)                       # the user scrolls up to read
    assert view._auto_scroll is False
    view.scroll_to_bottom()                 # a streamed batch arrives
    assert len(scheduled) == 1              # view left where the user put it

    bar.setValue(1000)                      # back at the bottom: follow again
    assert view._auto_scroll is True
    view.scroll_to_bottom()
    assert len(scheduled) == 2

    bar.setValue(200)
    view.add_bubble("assistant")            # output while scrolled up: stay
    assert len(scheduled) == 2
    view.add_bubble("user")                 # the user's own message: jump down
    assert len(scheduled) == 3
    view.scroll_to_bottom(force=True)
    assert len(scheduled) == 4
