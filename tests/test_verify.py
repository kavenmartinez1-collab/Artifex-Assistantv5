"""core.verify — automatic checks after agent edits, and their wiring into AgentRunner."""

import os
import subprocess
import sys

import pytest

from core.agent_loop import AgentRunner, AutonomyLevel, RunConfig
from core.verify import EditVerifier


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)


@pytest.fixture
def proj(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calc.py").write_text(
        "from pkg.calc import add\n\n"
        "def test_add():\n    assert add(1, 2) == 3\n\n"
        "def test_already_broken():\n    assert add(1, 1) == 3\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_related_tests_by_name_and_import(proj):
    (proj / "tests" / "test_other.py").write_text("import pkg.calc\n\ndef test_x():\n    pass\n")
    v = EditVerifier(str(proj), sys.executable)
    assert v.related_tests("pkg/calc.py") == ["tests/test_calc.py", "tests/test_other.py"]
    assert v.related_tests("tests/test_calc.py") == ["tests/test_calc.py"]
    assert v.related_tests("README.md") == []


def test_regression_vs_already_failing(proj):
    v = EditVerifier(str(proj), sys.executable)
    v.before_edit(str(proj / "pkg" / "calc.py"))
    (proj / "pkg" / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    rep = v.after_edits([str(proj / "pkg" / "calc.py")])
    assert rep.ran and not rep.clean
    assert rep.regressions == ["test_add"]
    assert rep.still_failing == ["test_already_broken"]
    assert "REGRESSIONS" in rep.render()
    assert "already failing" in rep.render()


def test_clean_edit(proj):
    v = EditVerifier(str(proj), sys.executable)
    v.before_edit("pkg/calc.py")
    (proj / "pkg" / "calc.py").write_text("def add(a, b):\n    return b + a\n")
    rep = v.after_edits(["pkg/calc.py"])
    assert rep.clean and rep.passed == 1 and rep.still_failing == ["test_already_broken"]


def test_broken_import_detected(proj):
    v = EditVerifier(str(proj), sys.executable)
    v.before_edit("pkg/calc.py")
    (proj / "pkg" / "calc.py").write_text("import not_a_real_module_xyz\n")
    rep = v.after_edits(["pkg/calc.py"])
    assert "pkg/calc.py" in rep.broken_imports
    assert "IMPORT BROKEN" in rep.render()


def test_module_that_never_imported_is_not_blamed(proj):
    (proj / "pkg" / "opt.py").write_text("import not_a_real_module_xyz\nX = 1\n")
    v = EditVerifier(str(proj), sys.executable)
    v.before_edit("pkg/opt.py")
    (proj / "pkg" / "opt.py").write_text("import not_a_real_module_xyz\nX = 2\n")
    rep = v.after_edits(["pkg/opt.py"])
    assert rep.broken_imports == {}


def test_new_test_failing_is_reported(proj):
    v = EditVerifier(str(proj), sys.executable)
    v.before_edit("pkg/calc.py")
    (proj / "tests" / "test_calc.py").write_text(
        (proj / "tests" / "test_calc.py").read_text() + "\ndef test_new():\n    assert False\n")
    rep = v.after_edits(["pkg/calc.py"])
    assert rep.new_failing == ["test_new"] and not rep.clean


# ── AgentRunner wiring ───────────────────────────────────────────────────

class _ScriptedEngine:
    def __init__(self, replies):
        self.replies = list(replies)

    def generate_streaming(self, messages, max_tokens, temperature, on_token=None, **kw):
        r = self.replies.pop(0)
        if on_token:
            on_token(r)
        return r


def _edit(path, old, new):
    return f"```edit\nFILE: {path}\nOLD:\n{old}\nNEW:\n{new}\n```\n"


def _runner(replies, **cfg):
    config = RunConfig(autonomy=AutonomyLevel.FULL_AUTO, max_rounds=6, **cfg)
    return AgentRunner(_ScriptedEngine(replies), build_system_prompt=lambda: "sys",
                       config=config)


def test_runner_holds_done_on_regression(proj, monkeypatch):
    monkeypatch.chdir(proj)
    bad = _edit("pkg/calc.py", "    return a + b", "    return a - b")
    good = _edit("pkg/calc.py", "    return a - b", "    return a + b")
    runner = _runner([bad, '@done("changed")', good + '@done("fixed")'])
    history = []
    res = runner.run("make add subtract", history)
    assert res.status == "done" and res.summary == "fixed"
    feedback = [m["content"] for m in history if m["role"] == "user"]
    assert any("REGRESSIONS" in f and "test_add" in f for f in feedback)
    assert any("not done yet" in f for f in feedback)


def test_runner_reverts_broken_import(proj, monkeypatch):
    monkeypatch.chdir(proj)
    bad = _edit("pkg/calc.py", "def add(a, b):", "import not_a_real_module_xyz\ndef add(a, b):")
    runner = _runner([bad, '@done("gave up")'])
    history = []
    runner.run("break it", history)
    assert "not_a_real_module_xyz" not in (proj / "pkg" / "calc.py").read_text()
    assert any("REVERTED" in m["content"] for m in history if m["role"] == "user")


def test_runner_done_hold_is_capped(proj, monkeypatch):
    monkeypatch.chdir(proj)
    bad = _edit("pkg/calc.py", "    return a + b", "    return a - b")
    runner = _runner([bad, '@done("a")', '@done("b")', '@done("c")'], verify_done_retries=2)
    res = runner.run("x", [])
    assert res.status == "done" and res.summary == "c"


def test_runner_auto_verify_off(proj, monkeypatch):
    monkeypatch.chdir(proj)
    bad = _edit("pkg/calc.py", "    return a + b", "    return a - b")
    runner = _runner([bad, '@done("a")'], auto_verify=False)
    res = runner.run("x", [])
    assert res.status == "done" and res.summary == "a"
