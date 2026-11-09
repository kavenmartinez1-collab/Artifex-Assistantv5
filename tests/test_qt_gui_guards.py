"""Main-window rules (ui/qt_gui.ArtifexMainWindow) checked without building
a window: the real methods run against a stand-in `self`. No QApplication,
no display. Skipped where PyQt6 is unavailable."""
import os
import threading
import types
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("PyQt6.QtMultimedia")
try:
    import ui.qt_gui as gui
except Exception as e:      # a Qt module this machine cannot load
    pytest.skip(f"ui.qt_gui not importable here: {e}", allow_module_level=True)

W = gui.ArtifexMainWindow

_REAL = ("_engine_in_use", "_drop_engine", "_set_current_worker", "_finish_busy",
         "_on_model_changed", "_on_backend_changed", "_on_backend_switched",
         "_on_execute", "_on_action_worker_done", "_on_generation_finished",
         "_on_clear", "_on_vram_relief", "_on_auto_run", "_on_auto_event",
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
            for sig in ("event", "approval_required", "run_finished", "finished"):
                setattr(self, sig, MagicMock())

        def start(self):
            made["started"] = True

    monkeypatch.setattr(gui, "AutonomousWorker", _Auto)
    win = make_win(engine=engine_mock())
    win._auto_goal.text.return_value = "tidy the folder"
    win._auto_level.currentText.return_value = "Guided"
    win._auto_preset.currentText.return_value = "Agent"
    win._auto_thinking.isChecked.return_value = True
    win._auto_plan.isChecked.return_value = False
    win._on_auto_run()
    assert made.get("started") and made["goal"] == "tidy the folder"
    assert made["config"].reasoning_effort == "medium"
    assert win._auto_running is True


def test_agent_run_is_refused_while_chat_is_generating(monkeypatch):
    monkeypatch.setattr(gui, "AutonomousWorker",
                        lambda *a, **k: pytest.fail("run started while busy"))
    win = make_win(_busy=True)
    win._auto_goal.text.return_value = "x"
    win._on_auto_run()
    assert win._auto_running is False


def test_agent_stream_chunks_are_batched_and_keep_their_order():
    from core.agent_loop import AgentEvent
    order = []
    win = make_win()
    win._auto_log_batcher.add.side_effect = lambda t: order.append(("add", t))
    win._auto_log_batcher.flush_now.side_effect = lambda: order.append("flush")
    win._auto_append.side_effect = lambda t: order.append("append")
    win._on_auto_event(AgentEvent("assistant_chunk", text="Hel", round=1))
    win._on_auto_event(AgentEvent("assistant_chunk", text="lo", round=1))
    win._auto_insert.assert_not_called()        # no per-token insert any more
    win._on_auto_event(AgentEvent("round_start", round=2))
    assert order == [("add", "Hel"), ("add", "lo"), "flush", "append"]
    win._on_auto_event(AgentEvent("thinking_chunk", text="hm", round=2))
    win._auto_think_batcher.add.assert_called_once_with("hm")


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
