"""
Tests for the interactive (conversation) mode of core.agent_loop and for the
bounded in-memory run state in api.agent_api.

Scripted engine, faked executor: no model, no server, no real commands.
"""

import re

import pytest

from core.agent_loop import (
    AgentEvent, AgentRunner, AutonomyLevel, Decision, RunConfig, RunControl,
)
from core.sandbox import clear_policy_hooks


@pytest.fixture(autouse=True)
def _clean_hooks():
    clear_policy_hooks()
    yield
    clear_policy_hooks()


class ScriptEngine:
    """Scripted replies. `before[i]` runs just before reply i is produced
    (to queue a user message 'while the model is generating')."""
    stream_starts_in_think = False

    def __init__(self, replies, before=None):
        self.replies = list(replies)
        self.before = before or {}
        self.i = 0
        self.prompts = []          # the messages each generation was given

    def get_context_size(self):
        return 32768

    def generate_streaming(self, messages, max_tokens=0, temperature=0.0, on_token=None):
        self.prompts.append([dict(m) for m in messages])
        hook = self.before.get(self.i)
        if hook:
            hook()
        r = self.replies[self.i] if self.i < len(self.replies) else "Nothing more to do."
        self.i += 1
        if on_token:
            on_token(r)
        # Like the real engines: the stream carries the think block, the
        # return value is the clean reply.
        return re.sub(r"<think>.*?</think>", "", r, flags=re.S)


def make(replies, monkeypatch, *, before=None, interactive=True, max_rounds=8,
         autonomy=AutonomyLevel.FULL_AUTO, approval=None, control=None,
         run_action=None):
    ran = []

    def fake_action(a, **kwargs):
        ran.append(a)
        return run_action(a) if run_action else (True, f"out:{a.display}")

    monkeypatch.setattr("core.agent_loop.run_agent_action", fake_action)
    cfg = RunConfig.default(autonomy)
    cfg.max_rounds = max_rounds
    cfg.interactive = interactive
    cfg.repo_map = False
    cfg.auto_verify = False
    events = []
    engine = ScriptEngine(replies, before)
    runner = AgentRunner(
        engine, build_system_prompt=lambda: "BASE",
        emit=events.append,
        request_approval=approval or (lambda a, d, r: Decision.APPROVE),
        config=cfg, control=control or RunControl(),
    )
    return runner, engine, events, ran


def kinds(events):
    return [e.kind for e in events]


def users(history):
    return [m["content"] for m in history if m["role"] == "user"]


# ── the prompt stays put, the goal is a turn ────────────────────────────────

def test_follow_up_keeps_the_system_prompt_and_the_first_goal(monkeypatch):
    runner, engine, events, _ = make(["First answer.", "Second answer."], monkeypatch)
    history = []
    runner.run_turn("write the report", history)
    runner.run_turn("also add tests", history)

    first_sys = engine.prompts[0][0]["content"]
    second_sys = engine.prompts[1][0]["content"]
    assert first_sys == second_sys                      # prompt cache survives
    assert "GOAL:" not in first_sys
    assert "also add tests" not in second_sys
    assert "AGENT MODE" in first_sys
    # Both requests are ordinary user turns of one history, in order.
    assert users(history)[0] == "write the report"
    assert "also add tests" in users(history)


def test_strict_mode_still_puts_the_goal_in_the_prompt(monkeypatch):
    runner, engine, _, _ = make(['@done("ok")'], monkeypatch, interactive=False)
    runner.run("write the report", [])
    sys_prompt = engine.prompts[0][0]["content"]
    assert "AUTONOMOUS MODE" in sys_prompt
    assert "GOAL:\nwrite the report" in sys_prompt


# ── a prose reply ends the turn ─────────────────────────────────────────────

def test_prose_reply_ends_the_turn_without_a_stall_nudge(monkeypatch):
    # "Let me ..." trips the stall detector in the unattended loop.
    runner, engine, events, _ = make(
        ["Let me explain: the test fails because of a typo."], monkeypatch)
    result = runner.run_turn("why does the test fail?", [])
    assert result.status == "done"
    assert engine.i == 1                                # one round, no nudge
    assert "format_retry" not in kinds(events)


def test_strict_mode_still_nudges_an_announcement(monkeypatch):
    runner, engine, events, _ = make(
        ["Let me start by reading the file.", '@done("ok")'], monkeypatch,
        interactive=False)
    runner.run("do it", [])
    assert "format_retry" in kinds(events)


# ── queued messages are never dropped ───────────────────────────────────────

def test_message_sent_during_the_final_round_is_answered(monkeypatch):
    control = RunControl()
    runner, engine, events, _ = make(
        ["All done.", "Yes, the README too."], monkeypatch, control=control,
        before={0: lambda: control.inject_message("did you update the README?")})
    history = []
    result = runner.run_turn("finish up", history)

    assert result.status == "done"
    assert result.summary == "Yes, the README too."
    assert "did you update the README?" in users(history)
    assert engine.i == 2
    assert control.drain_messages() == []


def test_message_left_after_the_round_cap_starts_the_next_turn(monkeypatch):
    control = RunControl()
    replies = ['@read_file("a.py")', "Stopping there as asked."]
    runner, engine, events, _ = make(
        replies, monkeypatch, control=control, max_rounds=1,
        # Queued after round 1's drain, and round 1 is the last one allowed.
        before={0: lambda: control.inject_message("stop after this file")})
    history = []
    result = runner.run_turn("read everything", history)

    assert "stop after this file" in users(history)
    assert result.status == "done"
    assert [e.text for e in events if e.kind == "user_message"] == ["stop after this file"]


def test_a_stopped_turn_leaves_the_queue_alone(monkeypatch):
    control = RunControl()

    def stop_and_queue():
        control.inject_message("later")
        control.request_stop()

    runner, engine, _, _ = make(["partial"], monkeypatch, control=control,
                                before={0: stop_and_queue})
    result = runner.run_turn("go", [])
    assert result.status == "stopped:user"
    assert engine.i == 1
    assert control.drain_messages() == ["later"]


# ── a mid-task message: answer, then carry on ───────────────────────────────

def test_prose_answer_to_a_mid_task_message_gets_one_carry_on_nudge(monkeypatch):
    control = RunControl()
    replies = [
        '@read_file("a.py")',
        "I'm reading the sources first.",        # answers the question, no action
        '@read_file("b.py")',                    # carries on after the nudge
        "Both files read.",
    ]
    runner, engine, events, ran = make(
        replies, monkeypatch, control=control,
        before={0: lambda: control.inject_message("what are you doing?")})
    history = []
    result = runner.run_turn("read a.py and b.py", history)

    assert result.status == "done"
    assert len(ran) == 2 and "a.py" in ran[0].display and "b.py" in ran[1].display
    nudges = [u for u in users(history) if u.startswith("[automated] If the task")]
    assert len(nudges) == 1


def test_no_carry_on_nudge_when_nothing_was_in_progress(monkeypatch):
    runner, engine, _, _ = make(["Hello."], monkeypatch)
    history = []
    runner.run_turn("hi", history)
    assert engine.i == 1
    assert not any(u.startswith("[automated]") for u in users(history))


# ── per-turn and per-session state ──────────────────────────────────────────

def test_notes_last_for_the_session(monkeypatch):
    runner, _, _, _ = make(['@note("port is 8100")\n@read_file("a.py")', "ok",
                            "still ok"], monkeypatch)
    history = []
    runner.run_turn("look", history)
    assert runner._notes == ["port is 8100"]
    runner.run_turn("again", history)
    assert runner._notes == ["port is 8100"]


def test_gate_and_breaker_are_fresh_each_turn(monkeypatch):
    runner, _, _, _ = make(["one", "two"], monkeypatch)
    history = []
    runner.run_turn("a", history)
    gate, breaker = runner.gate, runner.breaker
    runner.run_turn("b", history)
    assert runner.gate is not gate and runner.breaker is not breaker


# ── thinking is kept with the round ─────────────────────────────────────────

def test_thinking_rides_on_the_assistant_message(monkeypatch):
    runner, _, events, _ = make(["<think>check the port first</think>It is 8100."],
                                monkeypatch)
    runner.run_turn("which port?", [])
    msg = next(e for e in events if e.kind == "assistant_message")
    assert msg.thinking == "check the port first"
    assert msg.text.strip() == "It is 8100."
    assert msg.think_s >= 0.0


def test_long_thinking_is_shortened(monkeypatch):
    from core.agent_loop import THINK_KEEP_CHARS
    big = "x" * (THINK_KEEP_CHARS * 3)
    runner, _, events, _ = make([f"<think>{big}</think>done."], monkeypatch)
    runner.run_turn("go", [])
    msg = next(e for e in events if e.kind == "assistant_message")
    assert len(msg.thinking) < THINK_KEEP_CHARS + 100
    assert "thinking shortened" in msg.thinking


# ── api.agent_api: bounded in-memory state ──────────────────────────────────

def _api_run():
    from api.agent_api import AgentRun
    return AgentRun("goal", "", RunConfig.default(AutonomyLevel.GUIDED))


def test_event_buffer_is_capped_and_indices_keep_counting(monkeypatch):
    import api.agent_api as api
    monkeypatch.setattr(api, "_MAX_EVENTS", 50)
    monkeypatch.setattr(api, "_EVENT_TRIM", 20)
    run = _api_run()
    for n in range(120):
        run.emit(AgentEvent("git", text=f"e{n}"))
    assert len(run.events) <= 50
    assert run.event_end == 120
    assert run.events[-1]["i"] == 119
    assert run.events[0]["i"] == run._ev_base
    assert [e["i"] for e in run.events_from(115)] == [115, 116, 117, 118, 119]
    assert run.snapshot()["event_count"] == 120


def test_thinking_text_is_capped_per_run(monkeypatch):
    import api.agent_api as api
    monkeypatch.setattr(api, "_THINK_RUN_CHARS", 1000)
    run = _api_run()
    for n in range(10):
        run.emit(AgentEvent("assistant_message", text=f"m{n}", thinking="t" * 400))
    kept = [e for e in run.events if e.get("thinking")]
    assert sum(len(e["thinking"]) for e in kept) <= 1000
    assert run.events[0].get("thinking_dropped") is True
    assert run.events[-1]["thinking"] == "t" * 400       # newest is kept


def test_thinking_and_stats_are_serialized():
    run = _api_run()
    run.emit(AgentEvent("assistant_message", text="hi", thinking="hmm", think_s=1.5,
                        stats={"completion_tokens": 12}))
    d = run.events[0]
    assert d["thinking"] == "hmm" and d["think_s"] == 1.5
    assert d["stats"] == {"completion_tokens": 12}


# ── interrupt keeps the conversation ────────────────────────────────────────

def test_interrupt_mid_reply_keeps_the_partial_text_and_the_conversation(monkeypatch):
    control = RunControl()

    class Interrupted(ScriptEngine):
        def generate_streaming(self, messages, max_tokens=0, temperature=0.0, on_token=None):
            if self.i == 0:
                self.prompts.append([dict(m) for m in messages])
                self.i += 1
                on_token("I was half way through ")
                control.request_stop()
                on_token("this sentence")          # raises GenerationAborted
                return "never returned"
            return super().generate_streaming(messages, max_tokens, temperature, on_token)

    runner, engine, events, _ = make([], monkeypatch, control=control)
    runner.engine = engine = Interrupted([None, "Picking up where I left off."])
    history = []
    first = runner.run_turn("write the report", history)
    assert first.status == "stopped:user"
    partial = history[-1]
    assert partial["role"] == "assistant"
    assert partial["content"].startswith("I was half way through")

    control.reset()                                  # what a host does before the next turn
    second = runner.run_turn("carry on", history)
    assert second.status == "done"
    assert users(history)[:2] == ["write the report", "carry on"]
    # The interrupted turn is still in what the model sees.
    assert any(m["content"] == partial["content"] for m in engine.prompts[-1])


def test_interrupt_kills_a_running_command():
    import sys
    import threading
    import time as _time
    from tools import agent_tools

    stop = threading.Event()
    threading.Timer(0.6, stop.set).start()
    agent_tools.set_cancel_check(stop.is_set)
    t0 = _time.monotonic()
    try:
        with pytest.raises(agent_tools.CommandInterrupted):
            agent_tools._run_capture(
                [sys.executable, "-c", "import time; time.sleep(30)"], timeout=60)
    finally:
        agent_tools.set_cancel_check(None)
    assert _time.monotonic() - t0 < 10


def test_command_runs_normally_with_no_interrupt():
    import sys
    from tools import agent_tools
    res = agent_tools._run_capture([sys.executable, "-c", "print('hi')"], timeout=30,
                                   text=True)
    assert res.returncode == 0 and res.stdout.strip() == "hi"


# ── deny with a reason, and typing at a pause ───────────────────────────────

def test_deny_with_a_reason_reaches_the_model(monkeypatch):
    replies = ["```bash\nrm old.txt\n```", "OK, I left old.txt alone."]
    runner, engine, events, ran = make(
        replies, monkeypatch, autonomy=AutonomyLevel.MANUAL,
        approval=lambda a, d, r: (Decision.DENY, "keep that file, archive it instead"))
    history = []
    result = runner.run_turn("clean up", history)

    assert ran == []                                            # nothing executed
    assert result.status == "done"
    feedback = next(u for u in users(history) if "[skipped by user]" in u)
    assert "keep that file, archive it instead" in feedback
    assert "denied" in kinds(events)
    assert [e.text for e in events if e.kind == "user_message"] == [
        "keep that file, archive it instead"]


def test_plain_deny_still_works(monkeypatch):
    runner, _, events, ran = make(
        ["```bash\nrm old.txt\n```", "Skipped."], monkeypatch,
        autonomy=AutonomyLevel.MANUAL, approval=lambda a, d, r: Decision.DENY)
    history = []
    runner.run_turn("clean up", history)
    assert ran == []
    assert any(u.strip().endswith("rm old.txt") or "[skipped by user]" in u
               for u in users(history))
    assert "user_message" not in kinds(events)


def test_text_typed_at_a_gate_pause_becomes_the_next_message(monkeypatch):
    from core.sandbox.human_gate import GateState

    asked = []

    def approval(action, decision, reason):
        asked.append(reason)
        return (Decision.APPROVE, "fine, but skip the tests")

    runner, engine, events, _ = make(
        ['@read_file("a.py")', "Skipping the tests then."], monkeypatch,
        approval=approval)
    # run() (not run_turn) so the tight gate below is the one in use.
    runner.gate = GateState(interval=0, max_actions=1, risk_budget=0)
    history = []
    runner.run("look at a.py", history)
    assert any("human gate" in r for r in asked)
    assert "fine, but skip the tests" in users(history)


# ── api.agent_api: approvals answered by typing ─────────────────────────────

def _answer_later(run, fn):
    import threading
    import time as _time

    def go():
        for _ in range(200):
            if run.pending_approval is not None:
                fn()
                return
            _time.sleep(0.01)
    threading.Thread(target=go, daemon=True).start()


def test_api_message_during_an_action_prompt_is_a_deny_with_reason():
    from tools.agent_tools import AgentAction
    run = _api_run()
    taken = []
    _answer_later(run, lambda: taken.append(run.answer_with_message("use the other file")))
    dec = run.request_approval(AgentAction("shell", "rm x", "rm x"), None, "")
    assert dec == (Decision.DENY, "use the other file")
    assert taken == ["deny"]
    assert run.pending_approval is None and run.status != "awaiting_approval"


def test_api_message_during_a_pause_continues_with_the_text():
    run = _api_run()
    _answer_later(run, lambda: run.answer_with_message("go on, but slower"))
    assert run.request_approval(None, None, "human gate: 10 actions") == (
        Decision.APPROVE, "go on, but slower")


def test_api_message_with_nothing_pending_is_not_an_answer():
    assert _api_run().answer_with_message("hello") is None


def test_api_approve_button_still_returns_a_decision():
    run = _api_run()
    _answer_later(run, lambda: run.answer_approval("always"))
    assert run.request_approval(None, None, "x")[0] == Decision.APPROVE_ALWAYS


# ── chat and agent never share the engine ───────────────────────────────────

def test_agent_start_is_refused_while_a_chat_reply_is_generating(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import api.agent_api as api
    import api.chat_jobs as chat_jobs

    monkeypatch.setattr(api, "_runs", {})
    monkeypatch.setattr(chat_jobs, "live_chat_busy_reason",
                        lambda: "A chat reply is being generated")
    app = FastAPI()
    api.register_agent_routes(app, lambda r: True, lambda: None, "unused")
    r = TestClient(app).post("/v1/agent/runs", json={"goal": "x"})
    assert r.status_code == 409
    assert r.json()["detail"]["chat"] is True
    assert api._runs == {}


def test_chat_job_is_refused_while_the_agent_is_working(monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import api.agent_api as api
    import api.chat_jobs as chat_jobs

    live = _api_run()
    live.status = "running"
    monkeypatch.setattr(api, "_runs", {live.id: live})
    monkeypatch.setattr(chat_jobs, "_jobs", {})

    async def prepare(req):
        return "m"

    app = FastAPI()
    chat_jobs.register_chat_job_routes(app, check_auth=lambda r: True, prepare=prepare,
                                       stream_factory=lambda req, model: None,
                                       session_dir=str(tmp_path))
    client = TestClient(app)
    r = client.post("/v1/chat/jobs", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 409
    assert r.json()["detail"]["agent_run_id"] == live.id
    assert chat_jobs._jobs == {}

    # A finished turn frees the engine for chat again.
    live.status = "done"
    assert api.live_run_busy_reason() is None
