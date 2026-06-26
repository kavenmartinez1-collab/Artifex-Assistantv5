"""
Tests for core.agent_loop — the autonomous agent controller.

Drives the runner with a scripted FakeEngine and a faked executor (no real
shell/edits), exercising completion detection, stop conditions, the autonomy
levels' approval logic, policy blocking via the real ratchet, the circuit
breaker, the human gate, and auto-revert.
"""

import pytest

from core.agent_loop import (
    AgentRunner, RunConfig, RunControl, AutonomyLevel, Decision,
)
from core.sandbox import clear_policy_hooks, RiskLevel
from core.sandbox.circuit_breaker import CircuitBreaker
from core.sandbox.human_gate import GateState
from tools.agent_tools import AgentAction, detect_done


@pytest.fixture(autouse=True)
def _clean_hooks():
    """Isolate each test from globally-registered policy hooks."""
    clear_policy_hooks()
    yield
    clear_policy_hooks()


class FakeEngine:
    """Returns scripted responses; streams each via on_token."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.i = 0

    def get_context_size(self):
        return 8192

    def generate_streaming(self, messages, max_tokens=0, temperature=0.0, on_token=None):
        r = self.responses[self.i] if self.i < len(self.responses) else '@done("out of script")'
        self.i += 1
        if on_token:
            on_token(r)
        return r


def build_runner(responses, monkeypatch, *, autonomy=AutonomyLevel.FULL_AUTO,
                 approval=None, gate=None, breaker=None, max_rounds=8,
                 run_action=None, max_consecutive_failures=3, control=None):
    calls = {"actions": []}

    def default_action(a):
        calls["actions"].append(a)
        return True, f"out:{a.display}"

    monkeypatch.setattr("core.agent_loop.run_agent_action", run_action or default_action)
    cfg = RunConfig.default(autonomy)
    cfg.max_rounds = max_rounds
    cfg.max_consecutive_failures = max_consecutive_failures
    events = []
    runner = AgentRunner(
        FakeEngine(responses), build_system_prompt=lambda: "SYS",
        emit=events.append,
        request_approval=approval or (lambda a, d, r: Decision.APPROVE),
        config=cfg, gate=gate, breaker=breaker, control=control,
    )
    return runner, events, calls


def kinds(events):
    return [e.kind for e in events]


# ── completion detection ────────────────────────────────────────────────────

def test_detect_done_variants():
    assert detect_done('@done("all set")') == "all set"
    assert detect_done("@finish('y')") == "y"
    assert detect_done("@done") == ""
    assert detect_done("no marker here") is None
    assert detect_done("please @done(x) inline") is None   # must be its own line


def test_completes_on_done_marker(monkeypatch):
    runner, events, calls = build_runner(
        ['```bash\nls\n```', '@done("all set")'], monkeypatch)
    res = runner.run("do it", [{"role": "system", "content": "x"}])
    assert res.status == "done"
    assert res.summary == "all set"
    assert "done" in kinds(events)
    assert len(calls["actions"]) == 1          # only the `ls`
    assert res.rounds == 2


def test_no_actions_means_done(monkeypatch):
    runner, events, calls = build_runner(["Here is the final answer."], monkeypatch)
    res = runner.run("q", [{"role": "system", "content": "x"}])
    assert res.status == "done"
    assert res.summary == "Here is the final answer."
    assert calls["actions"] == []


def test_goal_appended_to_history(monkeypatch):
    runner, _, _ = build_runner(['@done("ok")'], monkeypatch)
    hist = [{"role": "system", "content": "x"}]
    runner.run("my goal", hist)
    assert any(m["role"] == "user" and m["content"] == "my goal" for m in hist)


# ── stop conditions ─────────────────────────────────────────────────────────

def test_stops_at_max_rounds(monkeypatch):
    responses = [f'```bash\necho {i}\n```' for i in range(10)]
    runner, events, calls = build_runner(responses, monkeypatch, max_rounds=3)
    res = runner.run("loop", [{"role": "system", "content": "x"}])
    assert res.status == "stopped:max_rounds"
    assert res.rounds == 3
    assert len(calls["actions"]) == 3


def test_stop_control_aborts_before_acting(monkeypatch):
    control = RunControl()
    control.request_stop()
    runner, events, calls = build_runner(
        ['```bash\nls\n```'], monkeypatch, control=control)
    res = runner.run("x", [{"role": "system", "content": "x"}])
    assert res.status == "stopped:user"
    assert calls["actions"] == []


def test_consecutive_failures_stop(monkeypatch):
    def always_fail(a):
        return False, "boom"
    responses = [f'```bash\necho {i}\n```' for i in range(10)]
    runner, events, calls = build_runner(
        responses, monkeypatch, run_action=always_fail, max_consecutive_failures=3)
    res = runner.run("x", [{"role": "system", "content": "x"}])
    assert res.status == "stopped:failures"
    assert res.actions_run == 3                   # stopped after 3rd consecutive failure


# ── autonomy levels ─────────────────────────────────────────────────────────

def test_guided_auto_runs_safe_reads(monkeypatch):
    approvals = []
    def approval(a, d, r):
        approvals.append(r or (a.type if a else "pause"))
        return Decision.APPROVE
    runner, events, calls = build_runner(
        ['@glob("**/*.py")', '@done("d")'], monkeypatch,
        autonomy=AutonomyLevel.GUIDED, approval=approval)
    res = runner.run("x", [{"role": "system", "content": "x"}])
    assert res.status == "done"
    assert approvals == []                       # SAFE read never prompted
    assert len(calls["actions"]) == 1


def test_guided_confirms_medium_risk(monkeypatch):
    seen = []
    def approval(a, d, r):
        seen.append(a.type if a else r)
        return Decision.DENY                      # user rejects the python run
    runner, events, calls = build_runner(
        ['```python\nprint(1)\n```', '@done("d")'], monkeypatch,
        autonomy=AutonomyLevel.GUIDED, approval=approval)
    res = runner.run("x", [{"role": "system", "content": "x"}])
    assert "python" in seen                       # MEDIUM prompted
    assert calls["actions"] == []                 # denied → not executed
    assert "approval_required" in kinds(events)


def test_full_auto_skips_prompts(monkeypatch):
    approvals = []
    runner, events, calls = build_runner(
        ['```python\nprint(1)\n```', '@done("d")'], monkeypatch,
        autonomy=AutonomyLevel.FULL_AUTO,
        approval=lambda a, d, r: approvals.append(r) or Decision.APPROVE)
    runner.run("x", [{"role": "system", "content": "x"}])
    assert approvals == []                        # MEDIUM auto-ran, no prompt
    assert len(calls["actions"]) == 1


def test_full_auto_critical_floor(monkeypatch):
    seen = []
    def approval(a, d, r):
        seen.append((a.type if a else None, d.risk_level if d else None))
        return Decision.DENY
    runner, events, calls = build_runner(
        ['```bash\nrm -rf /tmp/zzz\n```', '@done("d")'], monkeypatch,
        autonomy=AutonomyLevel.FULL_AUTO, approval=approval)
    runner.run("x", [{"role": "system", "content": "x"}])
    assert seen and seen[0][1] == RiskLevel.CRITICAL   # CRITICAL prompted even in full-auto
    assert calls["actions"] == []                       # denied → not run


# ── policy block (real ratchet) ─────────────────────────────────────────────

def test_ratchet_blocks_protected_write(monkeypatch):
    from core.sandbox.ratchet import install as install_ratchet
    install_ratchet()
    runner, events, calls = build_runner(
        ['```bash\necho pwned > core/config.py\n```', '@done("d")'], monkeypatch,
        autonomy=AutonomyLevel.FULL_AUTO)
    res = runner.run("x", [{"role": "system", "content": "x"}])
    assert "blocked" in kinds(events)
    assert calls["actions"] == []                 # blocked → never executed
    assert res.status == "done"


# ── circuit breaker & human gate ────────────────────────────────────────────

def test_circuit_breaker_repetition_stops(monkeypatch):
    def approval(a, d, r):
        return Decision.STOP if r.startswith("circuit breaker") else Decision.APPROVE
    responses = ['```bash\necho loop\n```'] * 6
    runner, events, calls = build_runner(
        responses, monkeypatch, autonomy=AutonomyLevel.FULL_AUTO,
        approval=approval, breaker=CircuitBreaker())
    res = runner.run("x", [{"role": "system", "content": "x"}])
    assert "breaker_tripped" in kinds(events)
    assert res.status == "stopped:breaker"


def test_human_gate_action_cap_stops(monkeypatch):
    def approval(a, d, r):
        return Decision.STOP if r.startswith("human gate") else Decision.APPROVE
    gate = GateState(interval=0, max_actions=2, risk_budget=0)
    responses = [f'```bash\necho {i}\n```' for i in range(6)]
    runner, events, calls = build_runner(
        responses, monkeypatch, autonomy=AutonomyLevel.FULL_AUTO,
        approval=approval, gate=gate)
    res = runner.run("x", [{"role": "system", "content": "x"}])
    assert "gate_pause" in kinds(events)
    assert res.status == "stopped:gate"


# ── auto-commit / auto-revert ───────────────────────────────────────────────

def test_framing_on_adds_preamble_and_goal(monkeypatch):
    runner, _, _ = build_runner(['@done("ok")'], monkeypatch)   # framing default True
    hist = [{"role": "system", "content": "SYS"}]
    runner.run("achieve X", hist)
    assert "AUTONOMOUS MODE" in hist[0]["content"]
    assert "achieve X" in hist[0]["content"]


def test_framing_off_uses_base_prompt(monkeypatch):
    runner, _, _ = build_runner(['answer'], monkeypatch)
    runner.config.framing = False
    hist = [{"role": "system", "content": "SYS"}]
    runner.run("hello", hist)
    assert hist[0]["content"] == "SYS"           # exactly the base prompt, no preamble
    assert "AUTONOMOUS MODE" not in hist[0]["content"]


def test_auto_revert_on_failed_run(monkeypatch):
    reverts = []
    monkeypatch.setattr("core.agent_loop.git_commit_edit",
                        lambda path, summary: (True, f"committed {path}"))
    monkeypatch.setattr("core.agent_loop.git_revert_last",
                        lambda path: (reverts.append(path), (True, f"reverted {path}"))[1])
    runner, _, _ = build_runner(['@done("x")'], monkeypatch)

    pending = []
    edit = AgentAction("edit_file", "foo.py\x00old\x00new", "edit foo.py")
    runner._on_action_complete(edit, True, pending)
    assert pending == ["foo.py"]                  # successful edit was committed

    failed = AgentAction("shell", "pytest", "pytest")
    runner._on_action_complete(failed, False, pending)
    assert reverts == ["foo.py"]                  # failed run rolled the edit back
    assert pending == []
