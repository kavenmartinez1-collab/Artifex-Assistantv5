"""The CLI's agent conversation: the transcript it prints, what a typed answer
to an approval means, and a turn that runs on a worker thread while the main
thread keeps the keyboard (approvals, Ctrl+C = interrupt, not exit)."""
import builtins
import threading
import types

import pytest

import ui.cli_assistant as cli
from core.agent_loop import AgentEvent, Decision, RunControl
from tools.agent_tools import AgentAction


def _plain(text):
    import re
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _host_out(events, capsys):
    host = cli._ConsoleHost()
    for ev in events:
        host.emit(ev)
    return host, _plain(capsys.readouterr().out)


# ── transcript ──────────────────────────────────────────────────────────────

def test_tool_call_is_one_line_with_a_short_result(capsys):
    act = AgentAction("shell", "git status", "git status")
    out_text = "\n".join(f"line {i}" for i in range(12))
    host, out = _host_out([
        AgentEvent("round_start", round=1),
        AgentEvent("action_proposed", action=act, round=1),
        AgentEvent("action_started", action=act, round=1),
        AgentEvent("action_result", action=act, success=True, output=out_text, round=1),
    ], capsys)
    assert "shell(git status)" in out
    assert "line 0" in out and "line 2" in out and "line 3" not in out
    assert "+9 lines (/expand)" in out
    assert host.last_output == ("shell: git status", out_text)     # /expand has it all


def test_failed_tool_shows_the_end_of_its_output(capsys):
    act = AgentAction("python", "x", "x")
    _, out = _host_out([
        AgentEvent("action_result", action=act, success=False,
                   output="a\nb\nc\nd\nTraceback: boom", round=1)], capsys)
    assert "Traceback: boom" in out and "+2 lines" in out
    assert "\n    a" not in out


def test_long_request_is_printed_in_full(capsys):
    script = "\n".join(f"echo step {i}" for i in range(30))
    act = AgentAction("shell", script, "echo step 0  (+29 more lines)")
    _, out = _host_out([AgentEvent("action_proposed", action=act, round=1)], capsys)
    assert out.count("echo step") == 30                # never cut


def test_thinking_is_kept_for_think_and_not_printed(capsys):
    host, out = _host_out([
        AgentEvent("round_start", round=1),
        AgentEvent("thinking_chunk", text="secret reasoning", round=1),
        AgentEvent("assistant_message", text="Answer.", thinking="secret reasoning",
                   think_s=12.0, round=1)], capsys)
    assert "secret reasoning" not in out
    assert "Thought for 12s" in out and "/think" in out
    assert host.thoughts == [(12.0, "secret reasoning")]
    host.begin_turn()
    assert host.thoughts == []                          # one turn's worth


def test_output_waits_while_the_user_types_a_line(capsys):
    host = cli._ConsoleHost()
    host.hold(True)
    host.emit(AgentEvent("assistant_message", text="Held back.", round=1))
    assert "Held back." not in capsys.readouterr().out
    host.hold(False)
    assert "Held back." in capsys.readouterr().out


def test_a_typed_approval_answer_is_not_echoed_again(capsys):
    host = cli._ConsoleHost()
    act = AgentAction("shell", "rm x", "rm x")
    assert host.answered(act, "no, keep it") == (Decision.DENY, "no, keep it")
    host.emit(AgentEvent("user_message", text="no, keep it", round=1))
    assert "no, keep it" not in capsys.readouterr().out
    host.emit(AgentEvent("user_message", text="another one", round=2))
    assert "> another one" in _plain(capsys.readouterr().out)


# ── what a typed answer means ───────────────────────────────────────────────

@pytest.mark.parametrize("typed, expected", [
    ("y", Decision.APPROVE), ("yes", Decision.APPROVE),
    ("a", Decision.APPROVE_ALWAYS), ("always", Decision.APPROVE_ALWAYS),
    ("", Decision.DENY), ("n", Decision.DENY), ("N", Decision.DENY),
    ("stop", Decision.STOP), ("s", Decision.STOP),
    ("use the other file", (Decision.DENY, "use the other file")),
])
def test_action_prompt_answers(typed, expected):
    act = AgentAction("shell", "rm x", "rm x")
    assert cli._ConsoleHost.read_answer(act, typed) == expected


@pytest.mark.parametrize("typed, expected", [
    ("", Decision.APPROVE), ("y", Decision.APPROVE),
    ("n", Decision.STOP), ("stop", Decision.STOP),
    ("go on but slower", (Decision.APPROVE, "go on but slower")),
])
def test_pause_prompt_answers(typed, expected):
    assert cli._ConsoleHost.read_answer(None, typed) == expected


# ── a turn on a worker thread ───────────────────────────────────────────────

class _Runner:
    """Stands in for AgentRunner: run_turn asks for approval, then waits to
    be released or stopped."""

    def __init__(self, control, script):
        self.control = control
        self.script = script
        self.request_approval = None
        self.thread = None
        self.answers = []

    def run_turn(self, text, history):
        self.thread = threading.current_thread()
        return self.script(self, text, history)


def _result(status="done"):
    return types.SimpleNamespace(status=status, rounds=1, actions_run=0, summary="")


def test_turn_runs_off_the_main_thread_and_approvals_are_asked_on_it(monkeypatch, capsys):
    asked_on = []

    def fake_input(prompt=""):
        asked_on.append(threading.current_thread())
        return "no, not that one"

    monkeypatch.setattr(builtins, "input", fake_input)

    def script(r, text, history):
        r.answers.append(r.request_approval(AgentAction("shell", "rm x", "rm x"), None, ""))
        return _result()

    control = RunControl()
    runner = _Runner(control, script)
    res = cli._drive_turn(runner, "go", [], cli._ConsoleHost(), control)
    assert res.status == "done"
    assert runner.thread is not threading.main_thread()
    assert asked_on == [threading.main_thread()]
    assert runner.answers == [(Decision.DENY, "no, not that one")]


def test_ctrl_c_interrupts_the_turn_instead_of_ending_the_program(monkeypatch, capsys):
    def fake_input(prompt=""):
        raise KeyboardInterrupt              # Ctrl+C at the approval prompt

    monkeypatch.setattr(builtins, "input", fake_input)

    def script(r, text, history):
        r.answers.append(r.request_approval(AgentAction("shell", "rm x", "rm x"), None, ""))
        return _result("stopped:user")

    control = RunControl()
    runner = _Runner(control, script)
    res = cli._drive_turn(runner, "go", [], cli._ConsoleHost(), control)   # no raise
    assert res.status == "stopped:user"
    assert control.stop_requested
    assert runner.answers == [Decision.STOP]
    assert "interrupting" in capsys.readouterr().out


def test_an_error_in_the_turn_is_raised_to_the_caller(capsys):
    def script(r, text, history):
        raise ConnectionError("server gone")

    control = RunControl()
    with pytest.raises(ConnectionError):
        cli._drive_turn(_Runner(control, script), "go", [], cli._ConsoleHost(), control)


def test_a_runner_without_run_turn_still_works():
    class _Old:
        request_approval = None

        def run(self, goal, history):
            return _result()

    control = RunControl()
    assert cli._drive_turn(_Old(), "go", [], cli._ConsoleHost(), control).status == "done"


def test_plain_loop_reads_typed_messages_and_esc(monkeypatch):
    control = RunControl()
    host = cli._ConsoleHost()
    release = threading.Event()
    feed = iter(["use tabs", None, cli._ESC])

    class _Poller:
        composing = False

        def poll(self):
            got = next(feed, None)
            if got is cli._ESC:
                release.set()
            return got

    t = threading.Thread(target=release.wait, daemon=True)
    t.start()
    import queue
    cli._plain_turn_loop(t, queue.Queue(), host, control, _Poller())
    assert control.drain_messages() == ["use tabs"]
    assert control.stop_requested


def test_prompt_toolkit_is_optional():
    # Nothing in requirements installs it; the CLI must work without it.
    assert cli._prompt_toolkit() is None or hasattr(cli._prompt_toolkit(), "PromptSession")
