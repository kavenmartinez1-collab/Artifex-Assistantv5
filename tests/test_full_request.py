"""full_request: the agent UIs always get the whole tool request, uncut."""
from tools.agent_tools import AgentAction, full_request


def test_long_shell_command_is_not_cut():
    cmd = "echo " + "x" * 2000 + "\nls -la\nrm -f out.txt"
    a = AgentAction("shell", cmd, cmd[:80])
    assert full_request(a) == cmd


def test_python_returns_whole_body():
    code = "\n".join(f"print({i})" for i in range(300))
    assert full_request(AgentAction("python", code, "print(0)  (300 lines)")) == code


def test_edit_shows_both_sides_and_write_shows_content():
    old, new = "a" * 500 + "\nline2", "b" * 500
    full = full_request(AgentAction("edit_file", f"f.py\x00{old}\x00{new}", 'edit: f.py ("aaa...")'))
    assert old in full and new in full and "f.py" in full
    body = "y" * 5000
    full = full_request(AgentAction("edit_file", f"g.txt\x00\x00{body}", "write: g.txt"))
    assert full == f"write_file: g.txt\n{body}"


def test_other_tools_keep_args_not_in_label():
    a = AgentAction("view_image", "pic.png|what colour is the car?", 'view_image: "pic.png"')
    assert "what colour is the car?" in full_request(a)
    a = AgentAction("search", "q", 'search: "q"')
    assert full_request(a) == 'search: "q"'


def test_api_event_carries_uncapped_full_request():
    from api.agent_api import _serialize_event
    from core.agent_loop import AgentEvent
    cmd = "echo " + "z" * 3000
    d = _serialize_event(AgentEvent("action_proposed", action=AgentAction("shell", cmd, cmd[:80])))
    assert d["action"]["full"] == cmd
