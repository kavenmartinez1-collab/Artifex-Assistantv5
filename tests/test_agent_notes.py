"""@note(): facts the agent pins for itself, re-inserted after compaction."""

from core.agent_loop import AgentRunner, AutonomyLevel, RunConfig
from tools.agent_tools import extract_agent_actions


class _Engine:
    def __init__(self, replies):
        self.replies = list(replies)

    def generate_streaming(self, messages, max_tokens, temperature, on_token=None, **kw):
        return self.replies.pop(0)


def test_note_is_parsed_as_its_own_action():
    acts = extract_agent_actions('Found it.\n@note("gpu_pool reads the GGUF keys")\n@read_file("a.py")')
    assert [a.type for a in acts] == ["note", "read_file"]
    assert acts[0].content == "gpu_pool reads the GGUF keys"
    assert extract_agent_actions('inline @note("x") in prose is not a note') == []


def test_runner_keeps_notes_and_continues():
    runner = AgentRunner(_Engine(['@note("plan: fix B")', '@done("ok")']),
                         build_system_prompt=lambda: "sys",
                         config=RunConfig(autonomy=AutonomyLevel.FULL_AUTO, max_rounds=4))
    history = []
    res = runner.run("goal", history)
    assert res.status == "done" and res.rounds == 2
    assert runner._notes == ["plan: fix B"]
    assert any("[note] pinned" in m["content"] for m in history if m["role"] == "user")


def test_pin_notes_goes_after_goal_once():
    runner = AgentRunner(_Engine([]), build_system_prompt=lambda: "sys")
    runner._notes = ["a", "b"]
    hist = [{"role": "system", "content": "s"}, {"role": "user", "content": "goal"},
            {"role": "user", "content": "[EARLIER SESSION — COMPACTED SUMMARY]\n..."}]
    once = runner._pin_notes(hist)
    twice = runner._pin_notes(once)
    assert twice[2]["content"].startswith(runner._NOTES_HEADER)
    assert "- a\n- b" in twice[2]["content"]
    assert sum(m["content"].startswith(runner._NOTES_HEADER) for m in twice) == 1
    assert len(twice) == 4


def test_no_notes_leaves_history_alone():
    runner = AgentRunner(_Engine([]), build_system_prompt=lambda: "sys")
    hist = [{"role": "system", "content": "s"}, {"role": "user", "content": "goal"}]
    assert runner._pin_notes(hist) is hist
