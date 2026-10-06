"""core.agent_attempts — best-of-N attempts from one commit, keep one."""

import subprocess

import pytest

from core.agent_loop import AgentRunner, AutonomyLevel, RunConfig


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


@pytest.fixture
def proj(tmp_path, monkeypatch):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calc.py").write_text(
        "from pkg.calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    for args in (("init", "-q"), ("config", "user.email", "t@t"), ("config", "user.name", "t"),
                 ("add", "-A"), ("commit", "-q", "-m", "init")):
        _git(tmp_path, *args)
    monkeypatch.chdir(tmp_path)
    return tmp_path


class _Engine:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate_streaming(self, messages, max_tokens, temperature, on_token=None, **kw):
        self.calls.append(messages)
        return self.replies.pop(0)


def _edit(old, new):
    return f"```edit\nFILE: pkg/calc.py\nOLD:\n{old}\nNEW:\n{new}\n```\n"


def _runner(replies, events=None, **cfg):
    eng = _Engine(replies)
    config = RunConfig(autonomy=AutonomyLevel.FULL_AUTO, max_rounds=4, repo_map=False,
                       attempts=2, **cfg)
    return AgentRunner(eng, build_system_prompt=lambda: "sys",
                       emit=(events.append if events is not None else None), config=config), eng


def test_judge_picks_the_kept_attempt(proj):
    events = []
    r, eng = _runner([
        _edit("    return a + b", "    return a + b  # one") + '@done("one")',
        _edit("    return a + b", "    return b + a  # two") + '@done("two")',
        "Attempt 2 is cleaner.\nBEST: 2",
    ], events, auto_verify=False)
    res = r.run("goal", [])
    assert res.status == "done" and "attempt 2 of 2 kept" in res.summary
    assert "# two" in (proj / "pkg" / "calc.py").read_text()
    assert _git(proj, "rev-parse", "refs/artifex/attempts/1").returncode == 0
    kinds = [e.kind for e in events]
    assert kinds.count("attempt_done") == 2 and kinds[-1] == "done"
    assert "ATTEMPT 1" in eng.calls[-1][1]["content"] and "ATTEMPT 2" in eng.calls[-1][1]["content"]


def test_failing_checks_lose_without_a_judge(proj):
    r, eng = _runner([
        _edit("    return a + b", "    return a - b") + '@done("broke it")',
        '@done("broke it")',   # the same-round @done is refused on regressions
        _edit("    return a + b", "    return a + b  # fine") + '@done("fine")',
    ], verify_done_retries=0)
    res = r.run("goal", [])
    assert "attempt 2 of 2 kept" in res.summary
    assert "# fine" in (proj / "pkg" / "calc.py").read_text()
    assert eng.replies == []   # no judge call was needed


def test_dirty_worktree_runs_once(proj):
    (proj / "scratch.txt").write_text("user's uncommitted work")
    events = []
    r, eng = _runner(['@done("once")'], events)
    res = r.run("goal", [])
    assert res.summary == "once"
    assert (proj / "scratch.txt").read_text() == "user's uncommitted work"
    assert any(e.kind == "attempts" for e in events)


def test_identical_attempts_need_no_judge(proj):
    same = _edit("    return a + b", "    return b + a") + '@done("same")'
    r, eng = _runner([same, same], auto_verify=False)
    res = r.run("goal", [])
    assert res.status == "done" and eng.replies == []
