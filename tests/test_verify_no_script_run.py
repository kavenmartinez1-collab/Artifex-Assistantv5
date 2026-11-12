"""
The auto-check after an agent edit imported the edited .py file, and importing
a script runs it. Agent bench 2026-11-11 (multi_chain): the final check re-ran
merge.py and load_db.py after the model had archived their input, and the run
ended "all verified" over an empty merged.csv and an empty database.
A script is now syntax-checked only; modules are still imported.
"""
import sys

from core.verify import EditVerifier, runs_on_import

SCRIPT = '''import csv, glob
rows = []
for path in sorted(glob.glob("raw/*.csv")):
    rows.append(path)
with open("merged.csv", "w") as f:
    f.write("date,region,amount\n")
print("merged", len(rows))
'''

MODULE = '''"""A module."""
import os
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)
HERE = os.path.dirname(os.path.abspath(__file__))
LIMITS = {"a": 1, "b": [2, 3]}
try:
    import json
except ImportError:
    json = None


@dataclass
class Thing:
    names: list = field(default_factory=list)


def main():
    open("ran.txt", "w").write("x")


if __name__ == "__main__":
    main()
'''


def test_script_and_module_are_told_apart():
    assert runs_on_import(SCRIPT)
    assert not runs_on_import(MODULE)
    assert runs_on_import("import os\nos.remove('x.txt')\n")
    assert runs_on_import("conn = sqlite3.connect('sales.db')\n")
    assert runs_on_import("def f(:\n")              # unparseable: never imported
    assert not runs_on_import("")
    assert not runs_on_import("X = 1\nif X:\n    Y = 2\nelse:\n    Y = 3\n")


def test_edited_script_is_not_executed(tmp_path):
    (tmp_path / "merge.py").write_text(SCRIPT, encoding="utf-8")
    v = EditVerifier(str(tmp_path), sys.executable)
    v.before_edit("merge.py")
    rep = v.after_edits(["merge.py"])
    final = v.final_check()
    assert not (tmp_path / "merged.csv").exists()      # the harness ran nothing
    assert rep.clean and final.clean
    assert "NOT run" in rep.render() and "merge.py" in rep.render()


def test_cleanup_script_does_not_delete_on_write(tmp_path):
    (tmp_path / "keep.txt").write_text("data")
    (tmp_path / "cleanup.py").write_text("import os\nos.remove('keep.txt')\n", encoding="utf-8")
    v = EditVerifier(str(tmp_path), sys.executable)
    v.after_edits(["cleanup.py"])
    v.final_check()
    assert (tmp_path / "keep.txt").exists()


def test_module_is_still_imported_and_a_broken_import_reported(tmp_path):
    (tmp_path / "mod.py").write_text(MODULE, encoding="utf-8")
    v = EditVerifier(str(tmp_path), sys.executable)
    v.before_edit("mod.py")
    assert v.after_edits(["mod.py"]).render() == "module imports OK; no related tests found"
    assert not (tmp_path / "ran.txt").exists()
    (tmp_path / "mod.py").write_text("import no_such_module_xyz\n" + MODULE, encoding="utf-8")
    rep = v.after_edits(["mod.py"])
    assert "mod.py" in rep.broken_imports
