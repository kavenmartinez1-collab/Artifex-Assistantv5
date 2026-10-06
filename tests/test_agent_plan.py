"""core.agent_plan — plan first, then one fresh context per step."""

from core.agent_loop import AgentRunner, AutonomyLevel, RunConfig
from core.agent_plan import parse_plan


class _Engine:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate_streaming(self, messages, max_tokens, temperature, on_token=None, **kw):
        self.calls.append(messages)
        return self.replies.pop(0)


def _runner(replies, events, **cfg):
    eng = _Engine(replies)
    r = AgentRunner(eng, build_system_prompt=lambda: "sys", emit=events.append,
                    config=RunConfig(autonomy=AutonomyLevel.FULL_AUTO, max_rounds=4,
                                     repo_map=False, plan="auto", **cfg))
    return r, eng


def test_parse_plan():
    assert parse_plan("thinking...\nPLAN:\n1. add X\n2) fix Y\n\nnote") == ["add X", "fix Y"]
    assert parse_plan("no plan here") == []
    assert len(parse_plan("PLAN:\n" + "".join(f"{i}. s{i}\n" for i in range(1, 10)))) == 6


def test_two_steps_each_in_a_fresh_context():
    events = []
    r, eng = _runner(["PLAN:\n1. do A\n2. do B", '@done("did A")', '@done("did B")'], events)
    history = []
    res = r.run("goal text", history)
    assert res.status == "done"
    assert "did A" in res.summary and "did B" in res.summary
    step2 = eng.calls[2]
    joined = "\n".join(m["content"] for m in step2)
    assert "step 2 of 2: do B" in joined and "did A" in joined
    assert len([m for m in step2 if m["role"] == "assistant"]) == 0   # fresh context
    kinds = [e.kind for e in events]
    assert kinds.count("step_done") == 2 and kinds.count("done") == 1 and kinds[-1] == "done"
    assert kinds.count("plan") == 1 and kinds.count("step") == 2


def test_single_step_runs_normally():
    events = []
    r, eng = _runner(["PLAN:\n1. just do it", '@done("ok")'], events)
    res = r.run("goal", [])
    assert res.status == "done" and res.summary == "ok"
    assert "step_done" not in [e.kind for e in events]


def test_unusable_plan_runs_normally():
    events = []
    r, eng = _runner(["I would rather not plan.", '@done("ok")'], events)
    assert r.run("goal", []).status == "done"


def test_follow_up_does_not_replan():
    events = []
    r, eng = _runner(['@done("again")'], events)
    history = [{"role": "user", "content": "g"}, {"role": "assistant", "content": "x"}]
    res = r.run("follow up", history)
    assert res.summary == "again" and len(eng.calls) == 1


def test_failed_step_marks_run_incomplete():
    events = []
    r, eng = _runner(["PLAN:\n1. A\n2. B", "", "", "", '@done("did B")'], events,
                     empty_round_no_think=False)
    res = r.run("goal", [])
    assert res.status != "done"
