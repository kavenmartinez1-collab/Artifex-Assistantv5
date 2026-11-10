"""Main-window rules (ui/qt_gui.ArtifexMainWindow) checked without building
a window: the real methods run against a stand-in `self`. No QApplication,
no display. Skipped where PyQt6 is unavailable."""
import os
import threading
import types
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
pytest.importorskip("PyQt6.QtMultimedia", exc_type=ImportError)
try:
    import ui.qt_gui as gui
except Exception as e:      # a Qt module this machine cannot load
    pytest.skip(f"ui.qt_gui not importable here: {e}", allow_module_level=True)

W = gui.ArtifexMainWindow

_REAL = ("_engine_in_use", "_drop_engine", "_set_current_worker", "_finish_busy",
         "_on_model_changed", "_on_backend_changed", "_on_backend_switched",
         "_on_execute", "_on_action_worker_done", "_on_generation_finished",
         "_on_clear", "_on_vram_relief", "_on_auto_run", "_on_auto_event",
         "_auto_steer", "_auto_confirm_pending", "_on_auto_stop", "_auto_resolve",
         "_on_auto_finished", "_on_auto_new",
         "_shutdown", "_on_load_session", "_on_save_session", "_run_voice_from_audio",
         "_on_stt_result")


def make_win(**state):
    """A stand-in window: widgets are mocks, the listed methods are real."""
    win = MagicMock()
    win._busy = False
    win._auto_running = False
    win._current_action_worker = None
    win._current_worker = None
    win._backend_switching = False
    win._backend_worker = None
    win._retired_workers = []
    win._shutting_down = False
    win._auto_worker = None
    win._auto_control = None
    win._auto_armed_full = False
    win._auto_runner = None
    win._auto_history = None
    win._auto_live = win._auto_row = win._auto_card = win._auto_bubble = None
    win._auto_pending = []
    win._auto_last_text = ""
    win._auto_phase_text = ""
    win._auto_tokens = 0
    win._auto_planned = False
    win._auto_plan_noted = False
    win._ptt_stream = None
    win._resource_monitor = None
    win._mic_recorder._stream = None
    win._pending_tool_outputs = []
    win._cancel_event = threading.Event()
    win.engine = None
    win.messages = [{"role": "system", "content": "S"}]
    win._set_combo_quietly = W._set_combo_quietly
    win._model_combo.findText.return_value = 0
    win._backend_combo.findText.return_value = 0
    for name in _REAL:
        setattr(win, name, types.MethodType(getattr(W, name), win))
    for k, v in state.items():
        setattr(win, k, v)
    return win


def status_of(win):
    return " | ".join(str(c.args[0]) for c in win._set_status.call_args_list)


def engine_mock(loaded=True):
    e = MagicMock()
    e.is_loaded.return_value = loaded
    return e


# ── model / backend switching ────────────────────────────────────────────

def test_model_change_drops_an_engine_that_never_loaded(monkeypatch):
    # e.g. the VRAM gate refused the first model: the engine object stayed,
    # so EXECUTE after picking a smaller model retried the OLD one.
    chosen = []
    monkeypatch.setattr(gui, "set_active_model", chosen.append)
    stale = engine_mock(loaded=False)
    win = make_win(engine=stale)
    win._on_model_changed("smaller-model")
    assert chosen == ["smaller-model"]
    assert win.engine is None


def test_model_change_is_refused_while_generating(monkeypatch):
    chosen = []
    monkeypatch.setattr(gui, "set_active_model", chosen.append)
    monkeypatch.setattr(gui, "get_active_model_name", lambda: "current")
    busy_engine = engine_mock()
    win = make_win(engine=busy_engine, _busy=True)
    win._on_model_changed("other")
    assert chosen == []
    assert win.engine is busy_engine
    busy_engine.unload.assert_not_called()
    assert "Busy" in status_of(win)


def test_model_change_is_refused_during_an_autonomous_run(monkeypatch):
    monkeypatch.setattr(gui, "set_active_model", lambda n: pytest.fail("switched"))
    monkeypatch.setattr(gui, "get_active_model_name", lambda: "current")
    e = engine_mock()
    win = make_win(engine=e, _auto_running=True)
    win._on_model_changed("other")
    e.unload.assert_not_called()


def test_backend_change_runs_off_the_gui_thread(monkeypatch):
    # set_active_backend("ollama") retries with sleeps when Ollama is down;
    # on the GUI thread that froze the window.
    monkeypatch.setattr(gui, "set_active_backend",
                        lambda n: pytest.fail("probed on the GUI thread"))
    monkeypatch.setattr(gui, "get_model_names",
                        lambda: pytest.fail("probed on the GUI thread"))
    made = []

    class _Worker:
        def __init__(self, backend):
            self.backend = backend
            self.started = False
            self.switched = MagicMock()
            self.finished = MagicMock()
            made.append(self)

        def start(self):
            self.started = True

    monkeypatch.setattr(gui, "BackendSwitchWorker", _Worker)
    old = engine_mock()
    win = make_win(engine=old)
    win._on_backend_changed("ollama")
    assert [w.backend for w in made] == ["ollama"] and made[0].started
    assert win._backend_switching is True
    assert win.engine is None
    old.unload.assert_called_once()

    # nothing may start generating until the switch lands
    win._on_execute()
    win._run_text_generation.assert_not_called()

    win._on_backend_switched("ollama", [])
    assert win._backend_switching is False
    assert "no models found" in status_of(win)


def test_backend_change_is_refused_while_busy(monkeypatch):
    monkeypatch.setattr(gui, "BackendSwitchWorker",
                        lambda b: pytest.fail("switch started while busy"))
    monkeypatch.setattr(gui, "get_active_backend", lambda: "llama_cpp")
    e = engine_mock()
    win = make_win(engine=e, _busy=True)
    win._on_backend_changed("ollama")
    e.unload.assert_not_called()
    assert win.engine is e


def test_clear_and_vram_do_not_unload_under_a_running_generation(monkeypatch):
    monkeypatch.setattr(gui, "get_service", lambda: pytest.fail("unloaded pipelines"))
    e = engine_mock()
    win = make_win(engine=e, _busy=True)
    win._on_clear()
    win._on_vram_relief()
    e.unload.assert_not_called()
    assert win.engine is e
    assert win._cancel_event.is_set()          # Clear asks the run to stop
    assert len(win.messages) == 1


# ── one user of the engine at a time ─────────────────────────────────────

def _ready_to_execute(win, tokens="max"):
    win._mode_combo.currentText.return_value = "Chat"
    win._prompt_input.toPlainText.return_value = "hello"
    win._tokens_input.text.return_value = tokens
    win._drop_zone._attached_files = []
    win._temp_slider.value.return_value = 7


def test_execute_starts_a_generation_when_idle():
    win = make_win()
    _ready_to_execute(win)
    win._on_execute()
    win._run_text_generation.assert_called_once()
    assert win._busy is True


def test_execute_is_refused_during_an_autonomous_run():
    win = make_win(_auto_running=True)
    _ready_to_execute(win)
    win._on_execute()
    win._run_text_generation.assert_not_called()
    assert win._busy is False
    assert "autonomous run" in status_of(win)


def test_execute_is_refused_while_actions_run():
    win = make_win(_current_action_worker=MagicMock())
    _ready_to_execute(win)
    win._on_execute()
    win._run_text_generation.assert_not_called()


def test_bad_token_count_is_reported_instead_of_raising():
    win = make_win()
    _ready_to_execute(win, tokens="4k")
    win._on_execute()                           # used to raise ValueError
    win._run_text_generation.assert_not_called()
    assert win._busy is False
    assert "Tokens" in status_of(win)


def test_action_follow_up_never_starts_a_second_generation(monkeypatch):
    monkeypatch.setattr(gui, "GenerationWorker",
                        lambda *a, **k: pytest.fail("second generation started"))
    win = make_win(engine=engine_mock(), _busy=True,
                   _current_action_worker=MagicMock(),
                   _pending_tool_outputs=[("ls", "a.txt")])
    win._on_action_worker_done()
    assert win._current_action_worker is None
    assert win._pending_tool_outputs == [("ls", "a.txt")]    # kept for later
    assert len(win.messages) == 1


def test_transcription_counts_as_busy(monkeypatch):
    started = []

    class _Pipe:
        def __init__(self, *a, **k):
            self.result_ready = MagicMock()
            self.error = MagicMock()
            self.status_changed = MagicMock()

        def start(self):
            started.append(self)

        def isRunning(self):
            return True

    monkeypatch.setattr(gui, "PipelineWorker", _Pipe)
    monkeypatch.setattr(gui, "get_service", lambda: MagicMock())
    win = make_win()
    win._run_voice_from_audio(object())
    assert win._busy is True and len(started) == 1
    win._run_voice_from_audio(object())         # a second push-to-talk
    assert len(started) == 1                    # did not replace the first

    # nothing heard: the busy state is released again
    win._on_stt_result(types.SimpleNamespace(success=True, content="  "))
    assert win._busy is False
    # the worker whose thread is still winding down is parked, not dropped
    assert win._retired_workers == started


# ── finishing ────────────────────────────────────────────────────────────

def test_every_finish_hides_the_progress_bar_and_sizes_the_bubble():
    bubble = MagicMock()
    win = make_win(_busy=True, _current_bubble=bubble)
    win._finish_busy("ERROR")                   # the path a pipeline error takes
    win._progress_bar.setVisible.assert_called_with(False)
    bubble.finalize_stream.assert_called_once()
    assert win._busy is False and win._current_bubble is None


def test_cancel_keeps_the_partial_answer_in_history():
    worker = MagicMock()
    worker.cancelled = True
    worker._thinking_text = ""
    worker.isRunning.return_value = False
    win = make_win(_busy=True, _current_worker=worker, _current_bubble=MagicMock())
    win.messages.append({"role": "user", "content": "q"})
    win._on_generation_finished("Half an answer")
    assert win.messages[-1] == {"role": "assistant", "content": "Half an answer"}
    assert "[Generation cancelled]" not in str(win.messages)
    win.km.add_from_ai_response.assert_not_called()
    assert win._busy is False


def test_finished_generation_is_stored_and_mined_as_before():
    worker = MagicMock()
    worker.cancelled = False
    worker._thinking_text = ""
    worker.isRunning.return_value = False
    win = make_win(_busy=True, _current_worker=worker, _current_bubble=MagicMock())
    win._on_generation_finished("Full answer.")
    assert win.messages[-1] == {"role": "assistant", "content": "Full answer."}
    win.km.add_from_ai_response.assert_called_once_with("Full answer.")


# ── autonomous run ───────────────────────────────────────────────────────

def test_gui_agent_run_uses_medium_reasoning_effort(monkeypatch):
    made = {}

    class _Auto:
        def __init__(self, engine, goal, history, **kw):
            made.update(kw, goal=goal)
            for sig in ("agent_event", "approval_required", "run_finished", "finished"):
                setattr(self, sig, MagicMock())

        def start(self):
            made["started"] = True

    monkeypatch.setattr(gui, "AutonomousWorker", _Auto)
    win = make_win(engine=engine_mock())
    win._auto_input.toPlainText.return_value = "tidy the folder"
    win._auto_level.currentText.return_value = "Guided"
    win._auto_preset.currentText.return_value = "Agent"
    win._auto_thinking.isChecked.return_value = True
    win._auto_plan.isChecked.return_value = False
    win._on_auto_run()
    assert made.get("started") and made["goal"] == "tidy the folder"
    assert made["config"].reasoning_effort == "medium"
    assert made["config"].interactive is True      # a conversation, not a one-shot run
    assert made["runner"] is None                  # first turn makes the runner
    assert win._auto_running is True


def test_agent_run_is_refused_while_chat_is_generating(monkeypatch):
    monkeypatch.setattr(gui, "AutonomousWorker",
                        lambda *a, **k: pytest.fail("run started while busy"))
    win = make_win(_busy=True)
    win._auto_input.toPlainText.return_value = "x"
    win._on_auto_run()
    assert win._auto_running is False


def test_agent_stream_chunks_are_batched_and_keep_their_order():
    from core.agent_loop import AgentEvent
    order = []
    win = make_win()
    win._auto_log_batcher.add.side_effect = lambda t: order.append(("add", t))
    win._auto_log_batcher.flush_now.side_effect = lambda: order.append("flush")
    win._on_auto_event(AgentEvent("assistant_chunk", text="Hel", round=1))
    win._on_auto_event(AgentEvent("assistant_chunk", text="lo", round=1))
    win._auto_stream_text.assert_not_called()   # no per-token insert any more
    win._on_auto_event(AgentEvent("round_start", round=2))
    assert order == [("add", "Hel"), ("add", "lo"), "flush"]
    win._on_auto_event(AgentEvent("thinking_chunk", text="hm", round=2))
    win._auto_think_batcher.add.assert_called_with("hm")


def test_second_turn_reuses_the_conversations_runner(monkeypatch):
    made = {}

    class _Auto:
        def __init__(self, engine, goal, history, **kw):
            made.update(kw, goal=goal, history=history)
            for sig in ("agent_event", "approval_required", "run_finished", "finished"):
                setattr(self, sig, MagicMock())
            self.runner = kw["runner"]

        def start(self):
            pass

    monkeypatch.setattr(gui, "AutonomousWorker", _Auto)
    runner, control, history = MagicMock(), MagicMock(), [{"role": "system", "content": "S"}]
    win = make_win(engine=engine_mock(), _auto_runner=runner, _auto_control=control,
                   _auto_history=history)
    win._auto_input.toPlainText.return_value = "and the tests too"
    win._auto_level.currentText.return_value = "Guided"
    win._auto_preset.currentText.return_value = "Agent"
    win._on_auto_run()
    assert made["runner"] is runner and made["history"] is history
    assert made["config"] is runner.config
    control.reset.assert_called_once()             # clears the last turn's Stop


def test_message_while_working_is_queued_for_the_next_step():
    control = MagicMock()
    win = make_win(_auto_running=True, _auto_control=control)
    win._auto_input.toPlainText.return_value = "use tabs"
    win._on_auto_run()
    control.inject_message.assert_called_once_with("use tabs")
    assert [t for t, _ in win._auto_pending] == ["use tabs"]
    win._auto_user_bubble.assert_called_once_with("use tabs", pending=True)


def test_queued_message_is_confirmed_when_the_loop_reads_it():
    from core.agent_loop import AgentEvent
    bubble = MagicMock()
    win = make_win(_auto_running=True, _auto_pending=[("use tabs", bubble)])
    win._on_auto_event(AgentEvent("user_message", text="use tabs", round=3))
    bubble.set_pending.assert_called_once_with(False)
    assert win._auto_pending == []
    win._auto_user_bubble.assert_not_called()       # not drawn twice


def test_typing_at_an_approval_denies_with_the_reason():
    from core.agent_loop import Decision
    card, worker, control = MagicMock(), MagicMock(), MagicMock()
    card.answered, card.is_gate = False, False
    win = make_win(_auto_running=True, _auto_card=card, _auto_worker=worker,
                   _auto_control=control)
    win._auto_input.toPlainText.return_value = "no, archive it instead"
    win._on_auto_run()
    worker.resolve.assert_called_once_with(Decision.DENY, "no, archive it instead")
    card.mark.assert_called_once_with("denied")
    control.inject_message.assert_not_called()      # it went with the answer
    assert win._auto_card is None


def test_typing_at_a_pause_continues_with_the_message():
    from core.agent_loop import Decision
    card, worker = MagicMock(), MagicMock()
    card.answered, card.is_gate = False, True
    win = make_win(_auto_running=True, _auto_card=card, _auto_worker=worker,
                   _auto_control=MagicMock())
    win._auto_input.toPlainText.return_value = "go on"
    win._on_auto_run()
    worker.resolve.assert_called_once_with(Decision.APPROVE, "go on")


def test_stop_interrupts_and_answers_an_open_approval():
    from core.agent_loop import Decision
    card, worker, control = MagicMock(), MagicMock(), MagicMock()
    card.answered = False
    win = make_win(_auto_running=True, _auto_card=card, _auto_worker=worker,
                   _auto_control=control)
    win._on_auto_stop()
    control.request_stop.assert_called_once()
    worker.resolve.assert_called_once_with(Decision.STOP)


def test_stop_does_nothing_when_idle():
    control = MagicMock()
    win = make_win(_auto_running=False, _auto_control=control)
    win._on_auto_stop()                              # Esc with nothing running
    control.request_stop.assert_not_called()


def test_interrupted_turn_hands_unread_messages_back_and_keeps_the_conversation():
    from core.agent_loop import RunResult
    runner, bubble = MagicMock(), MagicMock()
    win = make_win(_auto_running=True, _auto_runner=runner,
                   _auto_pending=[("later, do X", bubble)])
    win._auto_input.toPlainText.return_value = ""
    win._on_auto_finished(RunResult(status="stopped:user", rounds=1))
    win._auto_input.setPlainText.assert_called_once_with("later, do X")
    win._auto_view.remove_item.assert_called_once_with(bubble)
    assert win._auto_running is False
    assert win._auto_runner is runner                # next message continues it
    assert win.messages == [{"role": "system", "content": "S"}]   # chat tab untouched


def test_new_conversation_is_refused_while_working():
    runner = MagicMock()
    win = make_win(_auto_running=True, _auto_runner=runner)
    win._on_auto_new()
    assert win._auto_runner is runner
    win._auto_view.clear_chat.assert_not_called()


# ── agent conversation widgets ───────────────────────────────────────────

@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_strip_tool_syntax_keeps_the_prose():
    from ui.qt_widgets import strip_tool_syntax
    text = 'I will look.\n@read_file("a.py")\n\nThen run it:\n```bash\npytest -q\n```\nDone.'
    assert strip_tool_syntax(text) == "I will look.\n\nThen run it:\n\nDone."
    # An edit block can hold fences and markers of its own: left alone.
    edit = 'Writing it.\n```edit\nREADME.md\n```bash\nx\n```\n@note("y")\n```'
    assert strip_tool_syntax(edit) == edit
    assert strip_tool_syntax("Just prose with @mention and `code`.") == \
        "Just prose with @mention and `code`."


def test_tool_row_collapsed_line_and_lazy_body(qapp):
    from ui.qt_widgets import ToolRow
    row = ToolRow("shell", "shell: git status", "git status", "LOW")
    assert row._body is None                         # nothing built until opened
    assert row._head.text().startswith("○  shell  git status")
    row.set_state("✓", "ok", "12 lines")
    row.set_output("line\n" * 12)
    assert "✓" in row._head.text() and "12 lines" in row._head.text()
    assert row._body is None and row._preview is None
    row.toggle()
    assert "git status" in row._body.toPlainText() and "── output ──" in row._body.toPlainText()


def test_failed_tool_row_shows_its_last_lines_without_opening(qapp):
    from ui.qt_widgets import ToolRow
    row = ToolRow("python", "python: x", "x", "MEDIUM")
    row.set_state("✗", "bad", "3 lines")
    row.set_output("a\nb\nTraceback: boom", failed=True)
    assert row._preview is not None and "boom" in row._preview.text()


def test_approval_card_emits_the_choice_and_shrinks_once_answered(qapp):
    from ui.qt_widgets import ApprovalCard
    card = ApprovalCard("shell  rm x", "risk HIGH", gate=False)
    got = []
    card.decided.connect(got.append)
    labels = {}
    from PyQt6.QtWidgets import QPushButton
    for b in card.findChildren(QPushButton):
        labels[b.text()] = b
    assert set(labels) == {"Approve", "Always", "Deny"}
    labels["Always"].click()
    assert got == ["always"]
    card.mark("approved")
    assert card.answered and card._title.text() == "APPROVAL · APPROVED"
    assert all(not b.isVisibleTo(card) for b in labels.values())


def test_gate_card_has_only_continue(qapp):
    from ui.qt_widgets import ApprovalCard
    from PyQt6.QtWidgets import QPushButton
    card = ApprovalCard("human gate: 10 actions", gate=True)
    assert [b.text() for b in card.findChildren(QPushButton)] == ["Continue"]
    assert card.is_gate


def test_bubble_markdown_replaces_the_stream_and_keeps_the_fold(qapp):
    from PyQt6.QtCore import Qt
    from ui.qt_widgets import ChatView, ThinkFold
    view = ChatView()
    bubble = view.add_bubble("assistant")
    bubble.add_text("")
    bubble.append_text("**bold** and `code`")
    fold = bubble.add_widget(ThinkFold("thinking text", 12.4), top=True)
    label = bubble.set_markdown("**bold** and `code`")
    assert bubble._stream_widget is None
    assert label.textFormat() == Qt.TextFormat.MarkdownText
    assert fold._head.text() == "▸ Thought for 12s"
    assert bubble._layout.indexOf(fold) == 1         # still right under the header
    # A very long reply stays plain (rich word-wrapped labels that size stall).
    long_label = view.add_bubble("assistant").set_markdown("x" * 9000)
    assert long_label.textFormat() != Qt.TextFormat.MarkdownText


def test_queued_bubble_is_marked_and_cleared(qapp):
    from ui.qt_widgets import ChatView
    view = ChatView()
    first = view.add_bubble("user")
    first.add_text("steer")
    first.set_pending(True)
    assert first.property("pending") == "true" and first._queued_label is not None
    view.add_bubble("assistant")
    first.set_pending(False)
    view.move_to_end(first)
    assert first._queued_label is None and view._bubbles[-1] is first


def test_composer_enter_sends_shift_enter_does_not_esc_interrupts(qapp):
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent
    from ui.qt_widgets import ComposerEdit
    box = ComposerEdit()
    got = []
    box.submitted.connect(lambda: got.append("send"))
    box.escaped.connect(lambda: got.append("esc"))

    def press(key, mods=Qt.KeyboardModifier.NoModifier, text=""):
        box.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, mods, text))

    press(Qt.Key.Key_Return)
    press(Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier, "\n")
    press(Qt.Key.Key_Escape)
    assert got == ["send", "esc"]
    assert box.toPlainText() == "\n"


# ── shutdown ─────────────────────────────────────────────────────────────

def _worker(running):
    w = MagicMock()
    w.isRunning.return_value = running
    return w


def test_shutdown_stops_everything_then_unloads(monkeypatch):
    from core.agent_loop import Decision
    monkeypatch.setattr(gui.os, "_exit", lambda code: pytest.fail("hard exit"))
    e = engine_mock()
    auto, control, monitor = _worker(False), MagicMock(), _worker(False)
    win = make_win(engine=e, _auto_worker=auto, _auto_control=control,
                   _resource_monitor=monitor, _current_worker=_worker(False))
    win._shutdown()
    assert win._cancel_event.is_set()
    control.request_stop.assert_called_once()
    auto.resolve.assert_called_once_with(Decision.STOP)   # a pending approval
    monitor.stop.assert_called_once()
    e.unload.assert_called_once()
    assert win.engine is None

    win._shutdown()                                       # aboutToQuit after close
    control.request_stop.assert_called_once()


def test_shutdown_waits_for_a_running_worker(monkeypatch):
    monkeypatch.setattr(gui.os, "_exit", lambda code: pytest.fail("hard exit"))
    running = MagicMock()
    state = {"running": True}
    running.isRunning.side_effect = lambda: state["running"]
    running.wait.side_effect = lambda ms: state.update(running=False)
    win = make_win(_current_worker=running, engine=engine_mock())
    win._shutdown()
    running.wait.assert_called_once()
    assert win.engine is None


def test_shutdown_never_tears_down_a_thread_that_will_not_stop(monkeypatch):
    exits = []
    monkeypatch.setattr(gui.os, "_exit", exits.append)
    monkeypatch.setattr(gui, "wait_for_workers", lambda workers, timeout_s=5.0:
                        [w for w in workers if w is not None and w.isRunning()])
    e = engine_mock()
    win = make_win(_current_worker=_worker(True), engine=e)
    win._shutdown()
    assert exits == [0]                # a deliberate exit, not a Qt abort
    e.unload.assert_not_called()       # not pulled out from under the thread


# ── sessions ─────────────────────────────────────────────────────────────

def test_save_then_load_through_the_dialogs(monkeypatch, tmp_path):
    target = tmp_path / "picked folder" / "talk.json"
    target.parent.mkdir()
    monkeypatch.setattr(gui.QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(target), "JSON Files (*.json)"))
    monkeypatch.setattr(gui.QFileDialog, "getOpenFileName",
                        lambda *a, **k: (str(target), "JSON Files (*.json)"))
    monkeypatch.setattr(gui, "get_active_model_name", lambda: "m")
    monkeypatch.setattr(gui, "get_active_backend", lambda: "b")

    saver = make_win()
    saver.session_map.to_dict.return_value = {}
    saver.messages += [{"role": "user", "content": "first question"},
                       {"role": "assistant", "content": "first answer"}]
    saver._on_save_session()
    assert target.is_file()                    # the folder and name that were picked
    assert os.listdir(target.parent) == ["talk.json"]

    loader = make_win()
    loader._on_load_session()
    assert "Session loaded" in status_of(loader)
    assert [m["content"] for m in loader.messages] == [
        "S", "first question", "first answer"]
    assert loader._chat_view.add_bubble.call_count == 2
