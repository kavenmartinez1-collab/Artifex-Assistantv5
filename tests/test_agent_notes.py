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


def test_unparsed_tool_marker_is_not_a_final_answer():
    # A marker the parser cannot read must get a format retry, not end the run.
    runner = AgentRunner(_Engine(['Let me look.\n@read_file("a.py", lines=3)', '@done("ok")']),
                         build_system_prompt=lambda: "sys",
                         config=RunConfig(autonomy=AutonomyLevel.FULL_AUTO, max_rounds=4,
                                          repo_map=False))
    history = []
    res = runner.run("goal", history)
    assert res.summary == "ok" and res.rounds == 2
    assert any("FORMAT ERROR" in m["content"] for m in history if m["role"] == "user")


def test_collapsed_wrapper_tool_call_is_parsed():
    s = ('Start.\n<tool_call>\nfunction=tool_use\n<tool_call>\n<parameter=tool_name>\n'
         'architecture\n</parameter>\n</function>\n</tool_call>')
    assert [a.type for a in extract_agent_actions(s)] == ["architecture"]
    s2 = ('<tool_call>\n<function=tool_use>\n<parameter=tool_name>read_file</parameter>\n'
          '<parameter=path>core/a.py</parameter>\n</function>\n</tool_call>')
    acts = extract_agent_actions(s2)
    assert acts[0].type == "read_file" and acts[0].content.startswith("core/a.py")


def test_malformed_call_is_removed_before_the_retry():
    broken = '<tool_call>\nfunction=tool_use\n</function>\n</tool_call>'
    eng = _Engine([broken, '@done("ok")'])
    seen = []
    orig = eng.generate_streaming

    def spy(messages, *a, **k):
        seen.append([m["content"] for m in messages])
        return orig(messages, *a, **k)
    eng.generate_streaming = spy
    runner = AgentRunner(eng, build_system_prompt=lambda: "sys",
                         config=RunConfig(autonomy=AutonomyLevel.FULL_AUTO, max_rounds=4,
                                          repo_map=False))
    assert runner.run("goal", []).summary == "ok"
    assert not any("function=tool_use" in c for c in seen[1])
