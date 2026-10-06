"""@read_file line ranges, and the line-in-the-path forms models write."""

from tools.agent_tools import extract_agent_actions, run_read_file


def _read(marker, monkeypatch, tmp_path):
    (tmp_path / "m.py").write_text("\n".join(f"line{i}" for i in range(1, 501)))
    monkeypatch.chdir(tmp_path)
    acts = extract_agent_actions(marker)
    assert len(acts) == 1 and acts[0].type == "read_file"
    return run_read_file(acts[0].content)


def test_explicit_range(monkeypatch, tmp_path):
    ok, out = _read('@read_file("m.py", 10, 12)', monkeypatch, tmp_path)
    assert ok and "lines 10-12 of 500" in out
    assert "10| line10" in out and "12| line12" in out and "line13" not in out


def test_start_only_reads_a_window(monkeypatch, tmp_path):
    ok, out = _read('@read_file("m.py", 100)', monkeypatch, tmp_path)
    assert ok and "lines 100-179" in out


def test_range_is_capped(monkeypatch, tmp_path):
    ok, out = _read('@read_file("m.py", 1, 5000)', monkeypatch, tmp_path)
    assert ok and "lines 1-400 of 500" in out


def test_chunk_form_still_works(monkeypatch, tmp_path):
    (tmp_path / "notes.txt").write_text("hello\n")
    monkeypatch.chdir(tmp_path)
    acts = extract_agent_actions('@read_file("notes.txt", chunk=1)')
    ok, out = run_read_file(acts[0].content)
    assert ok and "chunk 1/1" in out and "hello" in out


def test_line_written_into_the_path(monkeypatch, tmp_path):
    ok, out = _read('@read_file("m.py|348")', monkeypatch, tmp_path)
    assert ok and "lines 328-407" in out
    ok, out = _read('@read_file("m.py:20-22")', monkeypatch, tmp_path)
    assert ok and "lines 20-22" in out


def test_past_the_end_and_missing(monkeypatch, tmp_path):
    ok, out = _read('@read_file("m.py", 900, 910)', monkeypatch, tmp_path)
    assert not ok and "past the end" in out
    ok, out = _read('@read_file("nope.py|3")', monkeypatch, tmp_path)
    assert not ok and "File not found" in out
