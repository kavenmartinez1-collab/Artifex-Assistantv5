"""
Chat's prompt-injection floor (api.chat_tools): once a web tool has run in a
reply, code that needs a person is refused (chat cannot ask), and reads keep
working. The executor is faked: nothing really runs.
"""

import pytest

from api import chat_tools
from core.sandbox import clear_policy_hooks
from tools.agent_tools import AgentAction


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    clear_policy_hooks()
    ran = []

    def fake_run(action, **kwargs):
        ran.append(action.type)
        return True, f"out:{action.type}"

    monkeypatch.setattr("tools.agent_tools.run_agent_action", fake_run)
    monkeypatch.delenv("ARTIFEX_UNTRUSTED_GATE", raising=False)
    yield ran
    clear_policy_hooks()


def act(kind, content):
    return AgentAction(type=kind, content=content, display=content)


def test_code_after_web_tool_is_refused_next_round(_isolated):
    state = {}
    chat_tools.execute_chat_tools([act("search", "q")], ".", state)
    assert state["untrusted"] is True
    out = chat_tools.execute_chat_tools(
        [act("shell", "pip install leftpad"), act("read_file", "README.md")],
        ".", state)
    assert "after web content" in out
    assert _isolated == ["search", "read_file"]     # the read still ran


def test_same_round_actions_written_before_the_page_still_run(_isolated):
    state = {}
    chat_tools.execute_chat_tools(
        [act("search", "q"), act("python", "print(1)")], ".", state)
    assert _isolated == ["search", "python"]


def test_no_state_means_no_memory(_isolated):
    chat_tools.execute_chat_tools([act("search", "q")], ".")
    chat_tools.execute_chat_tools([act("python", "print(1)")], ".")
    assert _isolated == ["search", "python"]


def test_gate_off(_isolated, monkeypatch):
    monkeypatch.setenv("ARTIFEX_UNTRUSTED_GATE", "0")
    state = {"untrusted": True}
    chat_tools.execute_chat_tools([act("python", "print(1)")], ".", state)
    assert _isolated == ["python"]


def test_history_with_tool_results_counts():
    assert chat_tools.history_has_tool_results(
        [{"role": "user", "content": "hi"},
         {"role": "tool", "content": "page text"}])
    assert not chat_tools.history_has_tool_results(
        [{"role": "user", "content": "hi"}])
