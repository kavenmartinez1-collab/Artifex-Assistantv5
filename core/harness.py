"""
Artifex Assistant V5 — Harness ingestion ("Agent / Harness Mode").

When Artifex is pointed at a folder that a *different* coding agent has already
worked in — Claude Code, OpenAI Codex, Cursor, Gemini CLI, Qwen Code, Aider,
GitHub Copilot, Cline/Roo, Windsurf, Continue, Zed — this module lets any local
LLM "absorb" that folder's context the way Claude Code does:

  1. DETECT   which harness left behind instruction / memory / rules artifacts.
  2. ADOPT    them — copy the originals verbatim AND synthesize one normalized
              ``.artifex/ARTIFEX.md`` — into a ``.artifex/`` bundle in the folder.
  3. INJECT   the bundle into the model's system prompt (see core/prompts.py).

Canonical bundle written into ``<folder>/.artifex/``::

    .artifex/
      ARTIFEX.md         auto-generated, normalized, provenance-headed  (INJECTED)
      ARTIFEX.local.md   YOUR notes — created once, NEVER overwritten    (INJECTED)
      manifest.json      detected tools, source hashes, adopt timestamp
      sources/<tool>/    verbatim copies of every original artifact
      .gitignore         ignores the derived bundle by default

Design contract: stdlib-only, idempotent, and non-destructive. Re-adopting
refreshes only the auto-generated half, never touches ARTIFEX.local.md, and
never reads, copies, or inlines anything outside the workspace root.
"""

import fnmatch
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from core.logging_config import get_logger

_log = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

BUNDLE_DIR = ".artifex"
AUTO_FILE = "ARTIFEX.md"
LOCAL_FILE = "ARTIFEX.local.md"
MANIFEST_FILE = "manifest.json"
SOURCES_DIR = "sources"

_BEGIN = "<!-- ARTIFEX:BEGIN auto-generated — do not edit; edit ARTIFEX.local.md instead -->"
_END = "<!-- ARTIFEX:END auto-generated -->"

# Directories we never descend into while scanning a workspace.
_SKIP_DIRS = frozenset({
    ".git", BUNDLE_DIR, "node_modules", "__pycache__", ".venv", "venv", "env",
    ".idea", ".vs", ".vscode-test", "dist", "build", ".next", ".nuxt", ".cache",
    "target", "vendor", ".pytest_cache", ".mypy_cache", ".tox", ".gradle",
    "site-packages", ".svn", ".hg", "bin", "obj",
})

_DEFAULT_DEPTH = 4
_MAX_INJECT_BYTES = 64 * 1024       # per-file cap when folding into ARTIFEX.md
_MAX_SOURCE_BYTES = 2 * 1024 * 1024  # per-file cap when copying into sources/
_MAX_IMPORT_DEPTH = 5                # transitive @import resolution depth
_LOCAL_TEMPLATE = (
    "# ARTIFEX.local.md\n\n"
    "Your own notes for this workspace. Artifex injects this verbatim **after** "
    "the auto-generated ARTIFEX.md and never overwrites it. Add project rules, "
    "gotchas, or corrections here.\n"
)


# ═══════════════════════════════════════════════════════════════════════════
# REGISTRY — one spec per known harness
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HarnessSpec:
    """Footprint of one coding agent's on-disk context format.

    inject   — glob patterns whose *content* is folded into ARTIFEX.md.
    copy     — extra patterns copied verbatim into sources/ but not injected.
    imports  — "at" if the format supports ``@path`` transclusion (Claude/Gemini/Qwen).
    frontmatter — True if instruction files carry ``---`` YAML frontmatter (Cursor/Copilot).
    globals  — home-relative paths detected but NOT injected unless include_global=True.
    priority — lower sorts earlier in the synthesized ARTIFEX.md.
    """
    id: str
    name: str
    inject: Tuple[str, ...]
    copy: Tuple[str, ...] = ()
    imports: str = ""
    frontmatter: bool = False
    globals: Tuple[str, ...] = ()
    priority: int = 50
    note: str = ""


REGISTRY: Tuple[HarnessSpec, ...] = (
    HarnessSpec(
        id="claude", name="Claude Code", priority=10, imports="at",
        inject=("CLAUDE.md", "CLAUDE.local.md", "**/CLAUDE.md",
                ".claude/CLAUDE.md", ".claude/agents/*.md", ".claude/commands/*.md"),
        copy=(".claude/settings.json", ".claude/settings.local.json",
              ".mcp.json", ".claude/skills/**"),
        globals=("~/.claude/CLAUDE.md",),
        note="CLAUDE.md + .claude/ (agents, commands, settings, MCP)",
    ),
    HarnessSpec(
        id="agents_md", name="AGENTS.md (universal)", priority=12,
        inject=("AGENTS.md", "AGENT.md", "**/AGENTS.md", "**/AGENT.md"),
        globals=("~/.codex/AGENTS.md",),
        note="Cross-tool standard read by Codex, Cursor, Aider, Gemini, Zed, Jules…",
    ),
    HarnessSpec(
        id="gemini", name="Gemini CLI", priority=15, imports="at",
        inject=("GEMINI.md", "**/GEMINI.md", ".gemini/GEMINI.md"),
        copy=(".gemini/settings.json",), globals=("~/.gemini/GEMINI.md",),
        note="GEMINI.md hierarchy + .gemini/settings.json",
    ),
    HarnessSpec(
        id="qwen", name="Qwen Code", priority=16, imports="at",
        inject=("QWEN.md", "**/QWEN.md", ".qwen/QWEN.md"),
        copy=(".qwen/settings.json",), globals=("~/.qwen/QWEN.md",),
        note="QWEN.md hierarchy + .qwen/settings.json (Gemini-CLI fork)",
    ),
    HarnessSpec(
        id="cursor", name="Cursor", priority=20, frontmatter=True,
        inject=(".cursorrules", ".cursor/rules/*.mdc", "**/.cursor/rules/*.mdc"),
        note=".cursor/rules/*.mdc (frontmatter: description/globs/alwaysApply) + legacy .cursorrules",
    ),
    HarnessSpec(
        id="copilot", name="GitHub Copilot", priority=22, frontmatter=True,
        inject=(".github/copilot-instructions.md",
                ".github/instructions/*.instructions.md"),
        note=".github/copilot-instructions.md + instructions/*.instructions.md (applyTo)",
    ),
    HarnessSpec(
        id="aider", name="Aider", priority=24,
        inject=("CONVENTIONS.md",),
        copy=(".aider.conf.yml", ".aider.conf.yaml"),
        note="CONVENTIONS.md + .aider.conf.yml",
    ),
    HarnessSpec(
        id="cline", name="Cline", priority=26,
        inject=(".clinerules", ".clinerules/*.md", "memory-bank/*.md"),
        note=".clinerules (file or dir) + Memory Bank",
    ),
    HarnessSpec(
        id="roo", name="Roo Code", priority=28,
        inject=(".roorules", ".roo/rules/*.md", ".roo/rules/**/*.md"),
        note=".roorules + .roo/rules/",
    ),
    HarnessSpec(
        id="windsurf", name="Windsurf", priority=30,
        inject=(".windsurfrules", ".windsurf/rules/*.md"),
        note=".windsurfrules + .windsurf/rules/",
    ),
    HarnessSpec(
        id="continue", name="Continue.dev", priority=32,
        inject=(".continuerules", ".continue/rules/*.md"),
        copy=(".continue/config.yaml", ".continue/config.json"),
        note=".continuerules + .continue/rules/",
    ),
    HarnessSpec(
        id="zed", name="Zed", priority=34,
        inject=(".rules",),
        note=".rules",
    ),
)


def get_spec(spec_id: str) -> Optional[HarnessSpec]:
    return next((s for s in REGISTRY if s.id == spec_id), None)


# ═══════════════════════════════════════════════════════════════════════════
# REPORT TYPES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DetectedFile:
    relpath: str            # posix, relative to workspace root
    abspath: str
    role: str               # "instruction" | "artifact"
    sha256: str
    bytes: int
    oversize: bool = False
    frontmatter: Dict[str, str] = field(default_factory=dict)


@dataclass
class HarnessHit:
    spec: HarnessSpec
    files: List[DetectedFile]

    @property
    def instruction_files(self) -> List[DetectedFile]:
        return [f for f in self.files if f.role == "instruction" and not f.oversize]


@dataclass
class HarnessReport:
    folder: str
    hits: List[HarnessHit] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.hits

    @property
    def tool_ids(self) -> List[str]:
        return [h.spec.id for h in self.hits]

    def injectable(self) -> List[HarnessHit]:
        """Hits that contribute at least one instruction file to ARTIFEX.md."""
        return [h for h in self.hits if h.instruction_files]

    def summary(self) -> str:
        if self.is_empty:
            return "No known harness config detected."
        parts = []
        for h in self.hits:
            parts.append(f"{h.spec.name} ({len(h.files)})")
        return "Detected: " + ", ".join(parts)


@dataclass
class AdoptResult:
    folder: str
    bundle_dir: str
    artifex_path: str
    tools: List[Tuple[str, str, int]]   # (id, name, n_files)
    injected_tool_ids: List[str]
    injected_token_estimate: int
    files_copied: int
    files_skipped: int
    created_local: bool


# ═══════════════════════════════════════════════════════════════════════════
# LOW-LEVEL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _posix(p: str) -> str:
    return p.replace("\\", "/")


def _walk(folder: str, max_depth: int):
    """Yield absolute file paths under folder, pruning heavy/irrelevant dirs."""
    folder = os.path.abspath(folder)
    base = folder.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(folder):
        depth = root.count(os.sep) - base
        if depth >= max_depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for f in files:
            yield os.path.join(root, f)


def _match(rel: str, pattern: str) -> bool:
    """Match a posix relpath against a glob. ``**/`` means 'at any depth'."""
    rel = _posix(rel)
    if pattern.startswith("**/"):
        tail = pattern[3:]
        parts = rel.split("/")
        return any(fnmatch.fnmatch("/".join(parts[i:]), tail)
                   for i in range(len(parts)))
    return fnmatch.fnmatch(rel, pattern)


def _is_within(root: str, path: str) -> bool:
    """True if path resolves inside root (path-traversal guard)."""
    try:
        root_r = os.path.realpath(root)
        path_r = os.path.realpath(path)
        return os.path.commonpath([root_r, path_r]) == root_r
    except (ValueError, OSError):
        return False


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _read_text(path: str, cap: int = _MAX_INJECT_BYTES) -> Optional[str]:
    """Read up to `cap` bytes of a text file. None if binary/unreadable."""
    try:
        with open(path, "rb") as f:
            raw = f.read(cap + 1)
    except OSError:
        return None
    if b"\x00" in raw:
        return None  # binary
    truncated = len(raw) > cap
    raw = raw[:cap]
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return None
    if truncated:
        text += "\n\n[...truncated by Artifex harness ingest...]"
    return text


_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _split_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Split leading ``---`` YAML frontmatter into a flat dict + body.

    Intentionally minimal (no pyyaml dep): captures top-level ``key: value``
    lines, which is all the rules formats use (description, globs, alwaysApply,
    applyTo). Nested YAML is ignored.
    """
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    meta: Dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, _, val = line.partition(":")
            key = key.strip()
            if key and not key.startswith(("-", " ")):
                meta[key] = val.strip().strip("'\"")
    return meta, text[m.end():]


_IMPORT_RE = re.compile(r"(?<![\w@`])@([\w./\-]+\.[A-Za-z0-9]+)")


def _resolve_imports(text: str, base_dir: str, root: str,
                     depth: int = 0, visited: Optional[set] = None) -> str:
    """Inline ``@relative/path`` transclusions (Claude/Gemini/Qwen).

    Resolves relative to the importing file's directory, bounded in depth and
    cycle-guarded, and refuses to escape the workspace root. ``@~/...`` (home /
    global) references are deliberately left untouched.
    """
    if depth >= _MAX_IMPORT_DEPTH:
        return text
    visited = visited if visited is not None else set()

    def _repl(m: re.Match) -> str:
        token = m.group(1)
        if token.startswith("~"):
            return m.group(0)
        target = os.path.normpath(os.path.join(base_dir, token))
        if not _is_within(root, target) or not os.path.isfile(target):
            return m.group(0)
        real = os.path.realpath(target)
        if real in visited:
            return m.group(0)
        visited.add(real)
        body = _read_text(target)
        if body is None:
            return m.group(0)
        body = _resolve_imports(body, os.path.dirname(target), root,
                                depth + 1, visited)
        rel = _posix(os.path.relpath(target, root))
        return f"\n<!-- imported: @{rel} -->\n{body.strip()}\n"

    return _IMPORT_RE.sub(_repl, text)


def _file_info(absf: str, rel: str, role: str, spec: HarnessSpec) -> Optional[DetectedFile]:
    try:
        size = os.path.getsize(absf)
    except OSError:
        return None
    info = DetectedFile(
        relpath=rel, abspath=absf, role=role,
        sha256=_sha256_file(absf), bytes=size,
        oversize=size > _MAX_INJECT_BYTES,
    )
    if role == "instruction" and spec.frontmatter and not info.oversize:
        text = _read_text(absf)
        if text:
            info.frontmatter, _ = _split_frontmatter(text)
    return info


# ═══════════════════════════════════════════════════════════════════════════
# DETECT
# ═══════════════════════════════════════════════════════════════════════════

def detect(folder: str, max_depth: int = _DEFAULT_DEPTH) -> HarnessReport:
    """Scan `folder` for known harness artifacts. Read-only; writes nothing."""
    folder = os.path.abspath(os.path.expanduser(folder))
    report = HarnessReport(folder=folder)
    if not os.path.isdir(folder):
        return report

    rels = [(_posix(os.path.relpath(f, folder)), f) for f in _walk(folder, max_depth)]

    for spec in REGISTRY:
        seen: set = set()
        hit_files: List[DetectedFile] = []
        for role, patterns in (("instruction", spec.inject), ("artifact", spec.copy)):
            for rel, absf in rels:
                if absf in seen:
                    continue
                if any(_match(rel, p) for p in patterns):
                    info = _file_info(absf, rel, role, spec)
                    if info:
                        hit_files.append(info)
                        seen.add(absf)
        if hit_files:
            hit_files.sort(key=lambda f: (f.relpath.count("/"), f.relpath))
            report.hits.append(HarnessHit(spec=spec, files=hit_files))

    report.hits.sort(key=lambda h: h.spec.priority)
    return report


def detect_globals() -> List[Tuple[str, str]]:
    """Return [(tool_name, abspath)] for global/home configs that exist."""
    found = []
    for spec in REGISTRY:
        for g in spec.globals:
            p = os.path.expanduser(g)
            if os.path.isfile(p):
                found.append((spec.name, p))
    return found


# ═══════════════════════════════════════════════════════════════════════════
# SYNTHESIZE  (build ARTIFEX.md text)
# ═══════════════════════════════════════════════════════════════════════════

def _section_header(spec: HarnessSpec, f: DetectedFile) -> str:
    extra = ""
    if f.frontmatter:
        keys = ("description", "globs", "alwaysApply", "applyTo")
        bits = [f"{k}: {f.frontmatter[k]}" for k in keys if k in f.frontmatter]
        if bits:
            extra = "  (" + "; ".join(bits) + ")"
    return f"## {spec.name} — `{f.relpath}`{extra}"


def synthesize(report: HarnessReport, include: Optional[set] = None) -> str:
    """Build the normalized ARTIFEX.md body (between BEGIN/END markers)."""
    root = report.folder
    hits = [h for h in report.injectable()
            if include is None or h.spec.id in include]

    detected = ", ".join(h.spec.name for h in report.hits) or "none"
    lines = [
        _BEGIN,
        "# Artifex Harness Context",
        "",
        f"_Absorbed from `{root}` on {datetime.now().isoformat(timespec='seconds')}._",
        f"_Detected harnesses: {detected}._",
        "",
        "The sections below were left in this folder by other AI coding agents. "
        "Treat them as authoritative project instructions and memory.",
        "",
    ]

    seen_hashes: set = set()
    emitted = 0
    for hit in hits:
        for f in hit.instruction_files:
            if f.sha256 and f.sha256 in seen_hashes:
                continue  # identical file already folded in (e.g. shared AGENTS.md)
            text = _read_text(f.abspath)
            if not text:
                continue
            if hit.spec.frontmatter:
                _, text = _split_frontmatter(text)
            if hit.spec.imports == "at":
                text = _resolve_imports(text, os.path.dirname(f.abspath), root)
            text = text.strip()
            if not text:
                continue
            seen_hashes.add(f.sha256)
            lines.append(_section_header(hit.spec, f))
            lines.append("")
            lines.append(text)
            lines.append("")
            emitted += 1

    if emitted == 0:
        lines.append("_(No instruction content found — only config/settings files were present.)_")
        lines.append("")
    lines.append(_END)
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════
# ADOPT  (write the .artifex/ bundle)
# ═══════════════════════════════════════════════════════════════════════════

def adopt(folder: str, include: Optional[set] = None,
          include_global: bool = False) -> AdoptResult:
    """Detect + write/refresh ``<folder>/.artifex/``. Idempotent, non-destructive.

    `include` — set of tool ids to fold into ARTIFEX.md (None = all detected).
                Sources for *all* detected tools are copied regardless.
    """
    folder = os.path.abspath(os.path.expanduser(folder))
    report = detect(folder)

    bundle = os.path.join(folder, BUNDLE_DIR)
    sources_root = os.path.join(bundle, SOURCES_DIR)
    os.makedirs(bundle, exist_ok=True)

    # .gitignore — treat the derived bundle as a cache by default.
    gi = os.path.join(bundle, ".gitignore")
    if not os.path.exists(gi):
        _safe_write(gi,
                    "# Artifex harness bundle — derived from this folder's agent config.\n"
                    "# Ignored by default; delete a line (or this file) to commit it.\n"
                    "*\n")

    # Verbatim source copies (provenance). Rebuilt each adopt for cleanliness.
    if os.path.isdir(sources_root):
        shutil.rmtree(sources_root, ignore_errors=True)
    copied = skipped = 0
    manifest_tools = []
    for hit in report.hits:
        tool_files = []
        for f in hit.files:
            dest = os.path.join(sources_root, hit.spec.id, f.relpath)
            ok = False
            if f.bytes <= _MAX_SOURCE_BYTES and _is_within(folder, f.abspath):
                try:
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(f.abspath, dest)
                    ok = True
                    copied += 1
                except OSError as e:
                    _log.warning("harness: copy failed %s: %s", f.relpath, e)
            if not ok:
                skipped += 1
            tool_files.append({
                "path": f.relpath, "role": f.role, "sha256": f.sha256,
                "bytes": f.bytes, "copied": ok,
                "frontmatter": f.frontmatter or None,
            })
        manifest_tools.append({
            "id": hit.spec.id, "name": hit.spec.name,
            "injected": include is None or hit.spec.id in include,
            "files": tool_files,
        })

    # Synthesized, normalized ARTIFEX.md (always regenerated — safe to overwrite).
    artifex_md = synthesize(report, include)
    artifex_path = os.path.join(bundle, AUTO_FILE)
    _safe_write(artifex_path, artifex_md)

    # ARTIFEX.local.md — created once, never overwritten.
    local_path = os.path.join(bundle, LOCAL_FILE)
    created_local = False
    if not os.path.exists(local_path):
        _safe_write(local_path, _LOCAL_TEMPLATE)
        created_local = True

    globals_found = detect_globals() if include_global else []
    manifest = {
        "version": 1,
        "tool": "artifex-harness",
        "adopted_at": datetime.now().isoformat(timespec="seconds"),
        "workspace": folder,
        "include_global": include_global,
        "globals": [{"name": n, "path": p} for n, p in globals_found],
        "tools": manifest_tools,
    }
    _safe_write(os.path.join(bundle, MANIFEST_FILE),
                json.dumps(manifest, indent=2, ensure_ascii=False))

    injected_ids = [t["id"] for t in manifest_tools if t["injected"]]
    _log.info("harness: adopted %s — %d tools, %d files copied (%d skipped)",
              folder, len(report.hits), copied, skipped)

    return AdoptResult(
        folder=folder, bundle_dir=bundle, artifex_path=artifex_path,
        tools=[(h.spec.id, h.spec.name, len(h.files)) for h in report.hits],
        injected_tool_ids=injected_ids,
        injected_token_estimate=len(artifex_md) // 4,
        files_copied=copied, files_skipped=skipped, created_local=created_local,
    )


def _safe_write(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ═══════════════════════════════════════════════════════════════════════════
# LOAD / INJECT
# ═══════════════════════════════════════════════════════════════════════════

def preview(report: HarnessReport, include: Optional[set] = None) -> str:
    """Marker-stripped synthesized ARTIFEX.md for UI preview (no disk writes)."""
    return synthesize(report, include).replace(_BEGIN, "").replace(_END, "").strip()


def is_adopted(folder: str) -> bool:
    return os.path.isfile(os.path.join(os.path.abspath(os.path.expanduser(folder)),
                                       BUNDLE_DIR, AUTO_FILE))


def read_manifest(folder: str) -> Optional[dict]:
    path = os.path.join(os.path.abspath(os.path.expanduser(folder)),
                        BUNDLE_DIR, MANIFEST_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_injection(folder: str, token_budget: int = 2000,
                   include_local: bool = True) -> str:
    """Return the budget-trimmed text to inject into the system prompt.

    Empty string if the folder has not been adopted. Strips the internal
    BEGIN/END markers; appends ARTIFEX.local.md if it has real content.
    """
    folder = os.path.abspath(os.path.expanduser(folder))
    bundle = os.path.join(folder, BUNDLE_DIR)
    auto = _read_text(os.path.join(bundle, AUTO_FILE), cap=512 * 1024)
    if not auto:
        return ""
    auto = auto.replace(_BEGIN, "").replace(_END, "").strip()

    parts = [auto]
    if include_local:
        local = _read_text(os.path.join(bundle, LOCAL_FILE), cap=128 * 1024)
        if local:
            # Inject only the user's delta beyond the boilerplate template, so
            # an *appended* note is picked up but an untouched file adds nothing.
            extra = local
            if extra.startswith(_LOCAL_TEMPLATE):
                extra = extra[len(_LOCAL_TEMPLATE):]
            extra = extra.strip()
            if extra:
                parts.append("## Workspace notes (ARTIFEX.local.md)\n\n" + extra)

    text = "\n\n".join(parts)
    char_budget = max(0, token_budget) * 4
    if char_budget and len(text) > char_budget:
        text = text[:char_budget].rstrip() + "\n\n[...harness context truncated to fit budget...]"
    return text
