"""
Artifex Assistant V5 — automatic checks after agent edits.

A small model rarely runs the tests on its own, and when it does it reads a
wall of output it cannot attribute. This module runs the checks FOR it after
every round that edited a Python file, and reports only what the edits
changed:

  * import check  — does the edited module still import? Measured against
    the module's own state before the run's first edit to it, so a module
    that never imported (missing optional dependency) is not blamed on the
    edit.
  * related tests — tests/test_<module>.py plus test files that import the
    module, capped. Each test file is baselined BEFORE the first edit that
    affects it; afterwards only tests that passed then and fail now are
    regressions. Tests that were already failing, or that the model added,
    are reported separately.

Everything runs in the workspace with the agent's own interpreter, bounded
by a timeout. Nothing here edits files; reverting is the caller's decision.
"""

import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional

MAX_TEST_FILES = 4
TEST_TIMEOUT_S = 180
IMPORT_TIMEOUT_S = 60
_FAIL_LINES = 12


@dataclass
class CheckReport:
    regressions: List[str] = field(default_factory=list)   # passed before, fail now
    still_failing: List[str] = field(default_factory=list)  # failed before too
    new_failing: List[str] = field(default_factory=list)    # tests that did not exist before
    passed: int = 0
    broken_imports: Dict[str, str] = field(default_factory=dict)  # rel path -> error
    details: str = ""
    ran: bool = False

    @property
    def clean(self) -> bool:
        return not (self.regressions or self.new_failing or self.broken_imports)

    def render(self) -> str:
        if not self.ran:
            return ""
        lines = []
        for path, err in self.broken_imports.items():
            lines.append(f"IMPORT BROKEN: {path} no longer imports: {err}")
        total = self.passed + len(self.regressions) + len(self.still_failing) + len(self.new_failing)
        if total:
            if self.clean and not self.still_failing:
                lines.append(f"related tests: all {total} passed")
            else:
                lines.append(f"related tests: {self.passed}/{total} passed")
            if self.regressions:
                lines.append("REGRESSIONS (passed before your edits, fail now): "
                             + ", ".join(self.regressions[:10]))
            if self.new_failing:
                lines.append("new tests failing: " + ", ".join(self.new_failing[:10]))
            if self.still_failing:
                lines.append("already failing before your edits (not caused by you): "
                             + ", ".join(self.still_failing[:10]))
        elif not self.broken_imports:
            lines.append("module imports OK; no related tests found")
        if self.details:
            lines.append(self.details)
        return "\n".join(lines)


def _module_name(rel: str) -> Optional[str]:
    if not rel.endswith(".py"):
        return None
    parts = rel[:-3].replace("\\", "/").split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts or not all(p.isidentifier() for p in parts):
        return None
    return ".".join(parts)


class EditVerifier:
    """Per-run state: baselines of imports and test results."""

    def __init__(self, root: str, python: str, env: Optional[dict] = None):
        self.root = os.path.abspath(root)
        self.python = python
        self.env = env
        self._import_base: Dict[str, Optional[str]] = {}   # rel -> error or None
        self._test_base: Dict[str, Dict[str, bool]] = {}   # test rel -> {id: passed}
        self.edited: List[str] = []                         # rel paths, run order

    # ── discovery ──────────────────────────────────────────────────────────

    def rel(self, path: str) -> Optional[str]:
        ap = os.path.abspath(path if os.path.isabs(path) else os.path.join(self.root, path))
        try:
            r = os.path.relpath(ap, self.root)
        except ValueError:
            return None
        if r.startswith(".."):
            return None
        return r.replace("\\", "/")

    def related_tests(self, rel: str) -> List[str]:
        """Test files for `rel`: itself if it is a test, test_<stem>.py, then
        test files that import the module. Capped at MAX_TEST_FILES."""
        name = os.path.basename(rel)
        if name.startswith("test_") and name.endswith(".py"):
            return [rel]
        mod = _module_name(rel)
        if not mod:
            return []
        stem = mod.rsplit(".", 1)[-1]
        found: List[str] = []
        test_dirs = [d for d in ("tests", "test") if os.path.isdir(os.path.join(self.root, d))]
        for d in test_dirs:
            cand = f"{d}/test_{stem}.py"
            if os.path.isfile(os.path.join(self.root, cand)):
                found.append(cand)
        pat = re.compile(r"^\s*(from\s+" + re.escape(mod) + r"\s+import|import\s+"
                         + re.escape(mod) + r"\b)", re.M)
        for d in test_dirs:
            for fn in sorted(os.listdir(os.path.join(self.root, d))):
                if len(found) >= MAX_TEST_FILES:
                    break
                cand = f"{d}/{fn}"
                if not (fn.startswith("test_") and fn.endswith(".py")) or cand in found:
                    continue
                try:
                    with open(os.path.join(self.root, cand), encoding="utf-8", errors="replace") as f:
                        if pat.search(f.read()):
                            found.append(cand)
                except OSError:
                    continue
        return found[:MAX_TEST_FILES]

    # ── running ────────────────────────────────────────────────────────────

    def _run(self, args, timeout):
        # -B and no bytecode writes: a .pyc records its source's mtime in whole
        # seconds plus its size, so a baseline import followed within the same
        # second by a same-length edit ("a + b" -> "a - b") would otherwise be
        # served the stale bytecode and the check would see the old code.
        env = dict(self.env if self.env is not None else os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run([self.python, "-B", *args], cwd=self.root, env=env,
                              capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")

    def _import_error(self, rel: str) -> Optional[str]:
        mod = _module_name(rel)
        if not mod or os.path.basename(rel).startswith("test_"):
            return None
        try:
            r = self._run(["-c", f"import importlib; importlib.import_module({mod!r})"],
                          IMPORT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return None   # slow import is not a broken import
        if r.returncode == 0:
            return None
        tail = [l for l in (r.stderr or "").strip().splitlines() if l.strip()]
        return tail[-1][:300] if tail else f"exit {r.returncode}"

    def _pytest(self, files: List[str]) -> Dict[str, bool]:
        fd, xml = tempfile.mkstemp(suffix=".xml", prefix="artifex-verify-")
        os.close(fd)
        try:
            try:
                self._run(["-m", "pytest", *files, "-q", "-p", "no:cacheprovider",
                           "--tb=no", f"--junitxml={xml}"], TEST_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                return {}
            try:
                root = ET.parse(xml).getroot()
            except Exception:
                return {}
            out = {}
            for c in root.iter("testcase"):
                tid = f"{c.get('classname', '')}::{c.get('name', '')}"
                out[tid] = not any(ch.tag in ("failure", "error") for ch in c)
            return out
        finally:
            try:
                os.remove(xml)
            except OSError:
                pass

    def _failure_details(self, files: List[str], ids: List[str]) -> str:
        """Short tracebacks for up to 3 failing tests."""
        names = [i.split("::")[-1] for i in ids[:3]]
        if not names:
            return ""
        try:
            r = self._run(["-m", "pytest", *files, "-q", "-p", "no:cacheprovider",
                           "--tb=short", "-k", " or ".join(names)], TEST_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return ""
        text = r.stdout or ""
        m = re.search(r"=+ FAILURES =+\n(.*?)(?:\n=+ short test summary|\Z)", text, re.S)
        body = m.group(1) if m else text
        keep = []
        for block in re.split(r"\n_{3,} ", body)[:3]:
            bl = [l for l in block.splitlines() if l.strip()]
            keep.extend(bl[:2] + (["    ..."] if len(bl) > _FAIL_LINES else []) + bl[-(_FAIL_LINES - 2):])
        return "\n".join(keep)[:3000]

    # ── public ─────────────────────────────────────────────────────────────

    def before_edit(self, path: str) -> None:
        """Record the pre-edit state of `path`'s import and related tests (once)."""
        rel = self.rel(path)
        if not rel or not rel.endswith(".py"):
            return
        if rel not in self._import_base:
            exists = os.path.isfile(os.path.join(self.root, rel))
            self._import_base[rel] = self._import_error(rel) if exists else "new file"
        todo = [t for t in self.related_tests(rel) if t not in self._test_base]
        if todo:
            res = self._pytest(todo)
            for t in todo:
                self._test_base[t] = {k: v for k, v in res.items() if _belongs(k, t)}

    def after_edits(self, paths: List[str]) -> CheckReport:
        """Check the files edited this round against their baselines."""
        rep = CheckReport()
        rels = []
        for p in paths:
            r = self.rel(p)
            if r and r.endswith(".py") and r not in rels:
                rels.append(r)
                if r not in self.edited:
                    self.edited.append(r)
        if not rels:
            return rep
        return self._check(rels)

    def final_check(self) -> CheckReport:
        """Check every file edited during the run (before accepting @done)."""
        return self._check(list(self.edited)) if self.edited else CheckReport()

    def _check(self, rels: List[str]) -> CheckReport:
        rep = CheckReport(ran=True)
        for rel in rels:
            if not os.path.isfile(os.path.join(self.root, rel)):
                continue
            err = self._import_error(rel)
            if err and self._import_base.get(rel) is None:
                rep.broken_imports[rel] = err
        files: List[str] = []
        for rel in rels:
            for t in self.related_tests(rel):
                if t not in files and os.path.isfile(os.path.join(self.root, t)):
                    files.append(t)
        if files:
            res = self._pytest(files)
            for tid, ok in sorted(res.items()):
                base = next((self._test_base[t] for t in files
                             if t in self._test_base and _belongs(tid, t)), {})
                if ok:
                    rep.passed += 1
                elif tid not in base:
                    rep.new_failing.append(tid.split("::")[-1])
                elif base[tid]:
                    rep.regressions.append(tid.split("::")[-1])
                else:
                    rep.still_failing.append(tid.split("::")[-1])
            failing = [t for t, ok in res.items() if not ok
                       and t.split("::")[-1] in rep.regressions + rep.new_failing]
            if failing:
                rep.details = _indent(self._failure_details(files, failing))
        return rep


def _belongs(test_id: str, test_file: str) -> bool:
    """junit classname is dotted ('tests.test_x.TestY'); match on the file's module."""
    mod = test_file[:-3].replace("/", ".")
    return test_id.split("::")[0] == mod or test_id.split("::")[0].startswith(mod + ".")


def _indent(text: str) -> str:
    return "\n".join("  " + l for l in text.splitlines()) if text else ""
