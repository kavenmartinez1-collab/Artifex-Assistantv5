"""core.repo_map — the per-run map of source files and top-level symbols."""

from core.agent_loop import AgentRunner, AutonomyLevel, RunConfig
from core.repo_map import build_repo_map


def _proj(root):
    (root / "pkg").mkdir()
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "base.py").write_text(
        "class Engine:\n    def load(self):\n        pass\n    def _private(self):\n        pass\n\n"
        "def helper():\n    pass\n")
    (root / "pkg" / "a.py").write_text("from pkg.base import Engine\n\ndef run():\n    pass\n")
    (root / "pkg" / "b.py").write_text("from pkg.base import helper\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_a.py").write_text("def test_x():\n    pass\n")


def test_map_ranks_by_importers_and_lists_symbols(tmp_path):
    _proj(tmp_path)
    m = build_repo_map(str(tmp_path))
    lines = m.splitlines()
    assert lines[0].startswith("REPO MAP")
    assert lines[1].startswith("pkg/base.py [imported by 2]: class Engine(load, _private)")
    assert "def helper" in lines[1]
    assert any(l.startswith("pkg/a.py: def run") for l in lines)
    assert "tests/test_a.py" not in m and "1 test files" in m


def test_not_a_codebase(tmp_path):
    (tmp_path / "one.py").write_text("x = 1\n")
    assert build_repo_map(str(tmp_path)) == ""


def test_budget_drops_detail_then_files(tmp_path):
    for i in range(60):
        (tmp_path / f"mod{i:02}.py").write_text(
            "".join(f"def function_number_{j}():\n    pass\n" for j in range(20)))
    m = build_repo_map(str(tmp_path), budget_chars=1500)
    assert len(m) <= 1500 and "more files" in m


class _Engine:
    def __init__(self):
        self.seen = []

    def generate_streaming(self, messages, max_tokens, temperature, on_token=None, **kw):
        self.seen.append(messages[0]["content"])
        return '@done("ok")'


def test_runner_puts_map_in_system_prompt(tmp_path, monkeypatch):
    _proj(tmp_path)
    monkeypatch.chdir(tmp_path)
    eng = _Engine()
    AgentRunner(eng, build_system_prompt=lambda: "sys",
                config=RunConfig(autonomy=AutonomyLevel.FULL_AUTO)).run("goal", [])
    assert "REPO MAP" in eng.seen[0]
    assert eng.seen[0].index("REPO MAP") < eng.seen[0].index("GOAL:")
    eng = _Engine()
    AgentRunner(eng, build_system_prompt=lambda: "sys",
                config=RunConfig(autonomy=AutonomyLevel.FULL_AUTO, repo_map=False)).run("goal", [])
    assert "REPO MAP" not in eng.seen[0]
