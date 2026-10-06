"""
Artifex Assistant V5 — compact repository map for agent runs.

A small model works through a codebase by grepping for the names in the
goal, so it finds the file the report blames and stops there. A map of every
source file with its top-level symbols, ranked by how many other files
import it, shows the rest of the code it could be looking at, for a few
thousand tokens, once per run.

Built on tools.codebase_tools.CodebaseIndex (Python via AST, other languages
via regex). The map is computed once at run start and placed in the system
prompt, so it never changes mid-run and the server's prompt cache stays warm.
"""

import os
from typing import Dict, List

from tools.codebase_tools import CodebaseIndex

DEFAULT_BUDGET_CHARS = 14000   # ~3.5k tokens
MIN_FILES = 3          # fewer source files than this: not a codebase, no map
MAX_FILES = 3000       # more than this: indexing is too slow for run start


def _is_test(rel: str) -> bool:
    parts = rel.replace("\\", "/").split("/")
    name = parts[-1]
    return (name.startswith("test_") or name.endswith("_test.py")
            or any(p in ("tests", "test", "__tests__") for p in parts[:-1]))


def _module_of(rel: str) -> str:
    mod = os.path.splitext(rel.replace("\\", "/"))[0].replace("/", ".")
    return mod[:-len(".__init__")] if mod.endswith(".__init__") else mod


_SOURCE_EXTS = (".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".cs")
_cache: Dict[tuple, str] = {}


def _fingerprint(root: str, limit: int):
    """(source file count, newest mtime) — cheap, and changes on any edit."""
    n, newest = 0, 0.0
    skip = {"venv", ".venv", "node_modules", "__pycache__", "dist", "build", ".git"}
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in skip and not x.startswith(".")]
        for f in files:
            if f.endswith(_SOURCE_EXTS):
                n += 1
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(d, f)))
                except OSError:
                    pass
        if n > limit:
            break
    return n, newest


def build_repo_map(root: str, budget_chars: int = DEFAULT_BUDGET_CHARS) -> str:
    """The map text, or '' when `root` is not a codebase worth mapping."""
    n, newest = _fingerprint(root, MAX_FILES)
    if n < MIN_FILES or n > MAX_FILES:
        return ""
    key = (os.path.abspath(root), n, newest, budget_chars)
    if key not in _cache:
        if len(_cache) > 16:
            _cache.clear()
        _cache[key] = _build(root, budget_chars)
    return _cache[key]


def _build(root: str, budget_chars: int) -> str:
    idx = CodebaseIndex(root)
    idx.index()

    by_file: Dict[str, list] = {}
    for s in idx._symbols:
        by_file.setdefault(s.file, []).append(s)

    # Rank: how many other files import this one (Python import graph).
    mod_to_file = {_module_of(f): f for f in idx._files}
    indeg: Dict[str, int] = {f: 0 for f in idx._files}
    for f, imps in idx._imports.items():
        seen = set()
        for imp in imps or []:
            m = imp.get("module") or ""
            for cand in (m, *(f"{m}.{n}" for n in imp.get("names") or [])):
                tgt = mod_to_file.get(cand)
                if tgt and tgt != f and tgt not in seen:
                    seen.add(tgt)
                    indeg[tgt] += 1

    src = sorted((f for f in idx._files if not _is_test(f)),
                 key=lambda f: (-indeg.get(f, 0), f.replace("\\", "/")))
    tests = [f for f in idx._files if _is_test(f)]

    def line_for(f: str, methods: int, names: int) -> str:
        syms = by_file.get(f, [])
        parts: List[str] = []
        for s in syms:
            if s.kind == "class":
                ms = [m.name for m in syms if m.kind == "method" and m.parent == s.name
                      and not m.name.startswith("__")]
                body = f"({', '.join(ms[:methods])}{', ...' if len(ms) > methods else ''})" \
                    if methods and ms else ""
                parts.append(f"class {s.name}{body}")
        funcs = [s.name for s in syms if s.kind == "function"]
        if funcs:
            parts.append("def " + ", ".join(funcs[:names]) + (", ..." if len(funcs) > names else ""))
        tag = f" [imported by {indeg[f]}]" if indeg.get(f) else ""
        return f"{f.replace(chr(92), '/')}{tag}: " + ("; ".join(parts) if parts else "-")

    header = ("REPO MAP — every source file with its top-level symbols, most-imported "
              "first. Use it to find code beyond the files the GOAL names.")
    footer = (f"tests: {len(tests)} test files (tests/...)" if tests else "")
    for methods, names in ((8, 12), (4, 8), (0, 6), (0, 3)):
        lines = [line_for(f, methods, names) for f in src]
        text = "\n".join([header, *lines, footer]).strip()
        if len(text) <= budget_chars:
            return text
    # Still too big: keep the highest-ranked files that fit.
    out, used = [header], len(header)
    for i, line in enumerate(lines):
        if used + len(line) + 60 > budget_chars:
            out.append(f"... {len(lines) - i} more files (use @glob / @grep)")
            break
        out.append(line)
        used += len(line) + 1
    if footer:
        out.append(footer)
    return "\n".join(out)
