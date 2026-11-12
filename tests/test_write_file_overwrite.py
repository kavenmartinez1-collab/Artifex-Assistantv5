"""
The native write_file tool on a file that already exists (agent bench
2026-11-11). It answered "OLD is empty ... use a python code block instead",
words from the text protocol; Qwen3.8-Flash-Next then deleted the file so
that write_file would take it. Now the answer names the way out
(overwrite=true), and that way replaces the file in one step.
"""
from core.native_agent_tools import AGENT_NATIVE_TOOLS, call_to_action
from tools.agent_tools import OVERWRITE_MARK, full_request, run_edit_file


def _call(**args):
    return {"name": "write_file", "arguments": args}


def test_schema_offers_overwrite():
    fn = next(t["function"] for t in AGENT_NATIVE_TOOLS if t["function"]["name"] == "write_file")
    assert fn["parameters"]["properties"]["overwrite"]["type"] == "boolean"
    assert fn["parameters"]["required"] == ["path", "content"]


def test_plain_write_is_unchanged():
    act, err, fin = call_to_action(_call(path="a.txt", content="hi"))
    assert err is None and act.content == "a.txt\x00\x00hi"


def test_overwrite_flag_marks_the_action():
    for flag in (True, "true"):
        act, err, fin = call_to_action(_call(path="a.txt", content="hi", overwrite=flag))
        assert act.type == "edit_file" and act.content == f"a.txt\x00{OVERWRITE_MARK}\x00hi"
        assert "overwrite" in act.display
        assert full_request(act) == "write_file (overwrite): a.txt\nhi"
    act, _, _ = call_to_action(_call(path="a.txt", content="hi", overwrite=False))
    assert act.content == "a.txt\x00\x00hi"


def test_existing_file_refusal_says_how(tmp_path):
    f = tmp_path / "calc.py"
    f.write_text("x = 1\n", encoding="utf-8")
    ok, out = run_edit_file(f"{f}\x00\x00x = 2\n")
    assert not ok
    assert "already exists" in out and "overwrite" in out and "Do not delete" in out
    assert f.read_text(encoding="utf-8") == "x = 1\n"


def test_overwrite_replaces_and_keeps_line_endings(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_bytes(b"\xef\xbb\xbfold\r\nlines\r\n")
    ok, out = run_edit_file(f"{f}\x00{OVERWRITE_MARK}\x00new\ntext")
    assert ok and "Replaced" in out
    assert f.read_bytes() == b"\xef\xbb\xbfnew\r\ntext\r\n"


def test_overwrite_of_python_checks_syntax(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("x = 1\n", encoding="utf-8")
    ok, out = run_edit_file(f"{f}\x00{OVERWRITE_MARK}\x00def (:\n")
    assert not ok and "syntax error" in out
    assert f.read_text(encoding="utf-8") == "x = 1\n"


def test_overwrite_on_a_missing_file_creates_it(tmp_path):
    f = tmp_path / "sub" / "new.txt"
    ok, out = run_edit_file(f"{f}\x00{OVERWRITE_MARK}\x00hello")
    assert ok and "Created" in out
    assert f.read_text(encoding="utf-8") == "hello\n"
