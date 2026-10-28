"""Agent runs with structured tool calls (core.native_agent_tools).

LFM3-1.7B could not follow the @marker / ```edit``` text protocol: it needed
five rounds and a shell echo to write one file (2026-10-27). An engine entry
marked native_tools now gets real tool definitions; each call becomes the
same AgentAction the text parser produces, so policy and approvals still
see every action.
"""

import json

import pytest

from core.agent_loop import AgentRunner, RunConfig, AutonomyLevel, Decision
from core.native_agent_tools import (
    AGENT_NATIVE_TOOLS, call_to_action, sanitize_tool_pairs,
)
from core.sandbox import clear_policy_hooks


@pytest.fixture(autouse=True)
def _clean_hooks():
    clear_policy_hooks()
    yield
    clear_policy_hooks()


def _call(name, **args):
    return {"id": f"id_{name}", "name": name, "arguments": json.dumps(args)}


# ── call -> AgentAction ─────────────────────────────────────────────────────

@pytest.mark.parametrize("call, kind, content", [
    (_call("run_shell", command="Get-ChildItem"), "shell", "Get-ChildItem"),
    (_call("run_python", code="print(1)"), "python", "print(1)"),
    (_call("read_file", path="a.txt"), "read_file", "a.txt|1"),
    (_call("read_file", path="a.txt", start_line=10, end_line=20), "read_file", "a.txt|L10-20"),
    (_call("write_file", path="new.txt", content="hi"), "edit_file", "new.txt\x00\x00hi"),
    (_call("edit_file", path="a.py", old_text="x = 1", new_text="x = 2"),
     "edit_file", "a.py\x00x = 1\x00x = 2"),
    (_call("list_files", pattern="*.py"), "glob", "*.py"),
    (_call("list_files", pattern="*.py", directory="src"), "glob", "*.py|src"),
    (_call("search_files", pattern="TODO", path="."), "grep", "TODO|.|"),
    (_call("web_search", query="lfm3"), "search", "lfm3"),
    (_call("web_read", ref="2"), "web_read", "2"),
])
def test_calls_map_to_the_text_protocols_actions(call, kind, content):
    action, err, fin = call_to_action(call)
    assert err is None and fin is None
    assert (action.type, action.content) == (kind, content)
    assert action.display


def test_finish_bad_args_and_unknown_tools():
    assert call_to_action(_call("finish", summary="wrote it")) == (None, None, "wrote it")
    _, err, _ = call_to_action(_call("run_shell"))
    assert "missing command" in err
    _, err, _ = call_to_action({"name": "format_disk", "arguments": "{}"})
    assert "unknown tool" in err and "run_shell" in err
    _, err, _ = call_to_action({"name": "read_file", "arguments": "not json"})
    assert "missing path" in err


def test_every_definition_is_handled():
    for tool in AGENT_NATIVE_TOOLS:
        fn = tool["function"]
        args = {k: "x" for k in fn["parameters"]["required"]}
        action, err, fin = call_to_action({"name": fn["name"], "arguments": json.dumps(args)})
        assert err is None, (fn["name"], err)


def test_sanitize_tool_pairs():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "tool", "tool_call_id": "gone", "content": "orphan"},        # call trimmed away
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
        {"role": "tool", "tool_call_id": "a", "content": "ok"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "b"}]},    # result trimmed away
    ]
    out = sanitize_tool_pairs(msgs)
    assert out[1] == {"role": "user", "content": "[TOOL OUTPUT]\norphan"}
    assert out[2]["tool_calls"] and out[3]["role"] == "tool"
    assert "tool_calls" not in out[4]


# ── through the real loop ───────────────────────────────────────────────────

class NativeEngine:
    """Scripted engine that answers with tool calls, like llama-server --jinja."""
    native_tools = True

    def __init__(self, script):
        self.script = list(script)
        self.seen_tools = []
        self.seen_messages = []
        self._last_gen_stats = {}

    def get_context_size(self):
        return 8192

    def generate_streaming(self, messages, max_tokens=0, temperature=0.0,
                           on_token=None, tools=None, **kw):
        self.seen_tools.append(tools)
        self.seen_messages.append([dict(m) for m in messages])
        text, calls = self.script.pop(0) if self.script else ("", [_call("finish", summary="end")])
        self._last_gen_stats = {"tool_calls": calls}
        if on_token and text:
            on_token(text)
        return text


def _runner(engine, monkeypatch, ran):
    def fake_run(a, **kw):
        ran.append(a)
        return True, f"out:{a.display}"
    monkeypatch.setattr("core.agent_loop.run_agent_action", fake_run)
    cfg = RunConfig.default(AutonomyLevel.FULL_AUTO)
    cfg.max_rounds = 6
    events = []
    r = AgentRunner(engine, build_system_prompt=lambda: "TEXT PROTOCOL PROMPT",
                    emit=events.append,
                    request_approval=lambda a, d, rr: Decision.APPROVE, config=cfg)
    return r, events


def test_native_run_writes_reads_and_finishes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    engine = NativeEngine([
        ("", [_call("write_file", path="hello.txt", content="hi from LFM3")]),
        ("", [_call("read_file", path="hello.txt"), _call("list_files", pattern="*")]),
        ("Done.", [_call("finish", summary="wrote and read hello.txt")]),
    ])
    ran = []
    runner, events = _runner(engine, monkeypatch, ran)
    history = []
    result = runner.run("write hello.txt", history)

    assert result.status == "done"
    assert result.summary == "wrote and read hello.txt"
    assert [a.type for a in ran] == ["edit_file", "read_file", "glob"]
    # Real tool definitions went out, and not the text-protocol prompt.
    assert engine.seen_tools[0] and engine.seen_tools[0][0]["function"]["name"] == "run_shell"
    system = engine.seen_messages[0][0]["content"]
    assert "TEXT PROTOCOL PROMPT" not in system and "GOAL:\nwrite hello.txt" in system
    # Results came back as one tool message per call, in order.
    tools_round2 = [m for m in engine.seen_messages[2] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools_round2] == [
        "id_write_file", "id_read_file", "id_list_files"]
    assert "out:read_file" in tools_round2[1]["content"]


def test_bad_call_gets_an_error_result_and_a_retry(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    engine = NativeEngine([
        ("", [_call("run_shell")]),                       # missing command
        ("", [_call("run_shell", command="echo ok")]),
        ("", [_call("finish", summary="ok")]),
    ])
    ran = []
    runner, events = _runner(engine, monkeypatch, ran)
    result = runner.run("say ok", [])
    assert result.status == "done"
    assert [a.content for a in ran] == ["echo ok"]
    retry_msgs = [m for m in engine.seen_messages[1] if m["role"] == "tool"]
    assert "missing command" in retry_msgs[-1]["content"]


def test_plain_engines_keep_the_text_protocol(monkeypatch):
    class Plain:
        def get_context_size(self):
            return 8192

        def generate_streaming(self, messages, max_tokens=0, temperature=0.0, on_token=None):
            return '@done("fine")'
    runner, _ = _runner(Plain(), monkeypatch, [])
    assert runner._native is False
    assert runner.run("x", []).status == "done"
