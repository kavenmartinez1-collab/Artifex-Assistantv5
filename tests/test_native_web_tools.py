"""Native (structured) tool calls for models that never learn @markers.

LFM3-1.7B, told to write @search("query"), echoed the placeholders and then
fell back to its trained XML call format, so no search ever ran
(2026-10-27). Entries marked native_tools get real tool definitions and
their tool_calls are mapped onto the same guarded web tools.
"""

from api.web_tools import NATIVE_WEB_TOOLS, native_calls_to_web_tools


def test_definitions_are_openai_shaped():
    names = [t["function"]["name"] for t in NATIVE_WEB_TOOLS]
    assert names == ["web_search", "web_read"]
    for t in NATIVE_WEB_TOOLS:
        assert t["type"] == "function"
        assert t["function"]["parameters"]["required"]


def test_search_and_read_map_to_web_tools():
    calls = [
        {"id": "a", "name": "web_search", "arguments": '{"query": "tailscale 1.104"}'},
        {"id": "b", "name": "web_read", "arguments": '{"ref": "2"}'},
        {"id": "c", "name": "web_read", "arguments": '{"url": "https://example.com"}'},
    ]
    tools = native_calls_to_web_tools(calls)
    assert [(t["type"], t.get("query") or t.get("ref")) for t in tools] == [
        ("search", "tailscale 1.104"),
        ("web_read", "2"),
        ("web_read", "https://example.com"),
    ]
    assert all(t["raw"] for t in tools)    # dedupe in execute_web_tools keys on raw


def test_tolerates_names_args_and_junk():
    tools = native_calls_to_web_tools([
        {"name": "@search", "arguments": {"q": "already a dict"}},   # dict args, @ prefix
        {"name": "web_search", "arguments": "not json"},               # unparseable
        {"name": "web_search", "arguments": ""},                       # empty
        {"name": "delete_everything", "arguments": '{"path": "/"}'},   # unknown tool
        {"name": "web_read", "arguments": '{"n": 3}'},                  # numeric ref
    ])
    assert [(t["type"], t.get("query") or t.get("ref")) for t in tools] == [
        ("search", "already a dict"),
        ("web_read", "3"),
    ]
    assert native_calls_to_web_tools(None) == []
