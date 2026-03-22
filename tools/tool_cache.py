"""
Artifex-Lite v2 — Tool Output Cache.

Large tool outputs bloat conversation history and exhaust the model's context window.
This module intercepts large outputs, saves them to disk, and returns a compact
summary + file path. The model can drill into the cached output via @read_file()
whenever it needs the full details.

Usage:
    from tools.tool_cache import maybe_cache_output

    success, output = run_agent_action(action)
    output = maybe_cache_output(action.type, action.display, output)
    # output is now either the original (if small) or a summary + cache path
"""

import os
import re
import time
from core.config import BASE_DIR

# ===== CONFIGURATION =====

# Outputs shorter than this (chars) stay inline — no caching needed.
CACHE_THRESHOLD = 800

# Directory for cached tool outputs (auto-created).
CACHE_DIR = os.path.join(BASE_DIR, ".tool_cache")

# Rolling counter for unique filenames within a session.
_cache_counter = 0


def _ensure_cache_dir():
    """Create the cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def _next_cache_path(tool_type):
    """Generate a unique cache file path."""
    global _cache_counter
    _cache_counter += 1
    filename = f"{tool_type}_{_cache_counter:03d}.txt"
    return os.path.join(CACHE_DIR, filename)


def _save(path, content):
    """Write content to cache file."""
    _ensure_cache_dir()
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ===== TYPE-SPECIFIC SUMMARIZERS =====

def _summarize_architecture(output):
    """Summarize architecture map: stats + directory listing."""
    lines = output.strip().split("\n")
    summary_parts = []

    # Grab the stats line (e.g., "Files: 28 | Symbols: 295")
    for line in lines[:5]:
        if "Files:" in line or "Project:" in line:
            summary_parts.append(line.strip())

    # Extract top-level directories
    dirs = []
    for line in lines:
        stripped = line.strip()
        if stripped.endswith("/") and not stripped.startswith("venv"):
            dirs.append(stripped)
        # Also catch "  core/" style lines
        m = re.match(r"^\s+(\w+)/", line)
        if m and m.group(1) not in ("venv", "__pycache__"):
            if m.group(1) + "/" not in dirs:
                dirs.append(m.group(1) + "/")

    if dirs:
        summary_parts.append(f"Directories: {', '.join(dirs[:10])}")

    return "\n".join(summary_parts) if summary_parts else lines[0] if lines else "Architecture map generated."


def _summarize_grep(output):
    """Summarize grep results: match count + top files."""
    lines = output.strip().split("\n")
    header = lines[0] if lines else "Grep results"

    # Count matches per file
    file_counts = {}
    for line in lines[1:]:
        m = re.match(r"^(.+?):\d+:", line)
        if m:
            fname = m.group(1)
            file_counts[fname] = file_counts.get(fname, 0) + 1

    top_files = sorted(file_counts.items(), key=lambda x: -x[1])[:5]
    if top_files:
        file_summary = ", ".join(f"{f} ({n})" for f, n in top_files)
        return f"{header}\nTop files: {file_summary}"

    return header


def _summarize_skeleton(output):
    """Summarize a skeleton view: file info + key symbols."""
    lines = output.strip().split("\n")
    summary_parts = []

    # Grab "File:" header
    for line in lines[:5]:
        if line.startswith("File:") or "SKELETON VIEW" in line:
            summary_parts.append(line.strip())

    # Count classes and functions
    classes = []
    functions = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("L") and ": class " in stripped:
            m = re.search(r"class (\w+)", stripped)
            if m:
                classes.append(m.group(1))
        elif stripped.startswith("L") and ": def " in stripped:
            m = re.search(r"def (\w+)", stripped)
            if m:
                functions.append(m.group(1))

    if classes:
        summary_parts.append(f"Classes: {', '.join(classes[:8])}")
    if functions:
        shown = functions[:10]
        extra = f" +{len(functions) - 10} more" if len(functions) > 10 else ""
        summary_parts.append(f"Functions: {', '.join(shown)}{extra}")

    return "\n".join(summary_parts) if summary_parts else lines[0] if lines else "Skeleton view."


def _summarize_find(output):
    """Summarize find_symbol/find_references results."""
    lines = output.strip().split("\n")
    header = lines[0] if lines else "Find results"

    # Show first 3-4 results as preview
    results = [l.strip() for l in lines[1:6] if l.strip()]
    total_match = re.search(r"Found (\d+)", header)
    total = int(total_match.group(1)) if total_match else len(lines) - 1

    preview = "\n".join(f"  {r}" for r in results[:4])
    if total > 4:
        preview += f"\n  ... and {total - 4} more"

    return f"{header}\n{preview}" if preview else header


def _summarize_glob(output):
    """Summarize glob results: count + first few matches."""
    lines = output.strip().split("\n")
    header = lines[0] if lines else "Glob results"

    matches = [l.strip() for l in lines[1:6] if l.strip()]
    preview = "\n".join(f"  {m}" for m in matches)

    total_match = re.search(r"Found (\d+)", header)
    total = int(total_match.group(1)) if total_match else len(lines) - 1
    if total > 5:
        preview += f"\n  ... and {total - 5} more"

    return f"{header}\n{preview}" if preview else header


def _summarize_generic(output):
    """Fallback: first few lines + line count."""
    lines = output.strip().split("\n")
    total = len(lines)
    preview = "\n".join(lines[:5])
    if total > 5:
        preview += f"\n... ({total} lines total)"
    return preview


# Map tool types to their summarizers
_SUMMARIZERS = {
    "architecture": _summarize_architecture,
    "grep": _summarize_grep,
    "find_symbol": _summarize_find,
    "find_references": _summarize_find,
    "glob": _summarize_glob,
}


def _detect_skeleton(output):
    """Check if this output is a skeleton view (from read_file on large Python files)."""
    return "SKELETON VIEW" in output[:200]


# ===== PUBLIC API =====

def maybe_cache_output(tool_type, display, output):
    """Cache large tool output to disk and return a summary.

    If the output is small enough, returns it unchanged.
    If large, saves to .tool_cache/ and returns a compact summary
    with a path the model can use to drill in.

    Args:
        tool_type: action type string (e.g., "architecture", "grep")
        display: action display string (for labeling)
        output: the full tool output string

    Returns:
        The output string — either original or replaced with summary + path.
    """
    if not output or len(output) <= CACHE_THRESHOLD:
        return output

    # These tool types should NEVER be cached:
    # - read_function, find_symbol: precise lookups the model needs immediately
    # - python, shell: work products from the model's own code — caching these
    #   causes infinite read loops (model reads cache, read gets cached, repeat)
    # - read_file when reading from .tool_cache/: prevents cache-of-cache chains
    _NEVER_CACHE = {"read_function", "find_symbol", "python", "shell"}
    if tool_type in _NEVER_CACHE:
        return output

    # Never cache reads of already-cached files (prevents infinite chain)
    if tool_type == "read_file" and ".tool_cache" in display:
        return output

    # For read_file: only cache skeleton views (large Python files).
    # Normal file reads (small, single chunk) stay inline.
    if tool_type == "read_file" and not _detect_skeleton(output):
        return output

    # Pick the right summarizer.
    # architecture and skeleton views are now cached because the SessionMap
    # captures their structural data (files, symbols, line numbers).
    if _detect_skeleton(output):
        summarizer = _summarize_skeleton
    elif tool_type == "architecture":
        summarizer = _summarize_architecture
    else:
        summarizer = _SUMMARIZERS.get(tool_type, _summarize_generic)

    summary = summarizer(output)

    # Save full output to cache
    cache_path = _next_cache_path(tool_type)
    _save(cache_path, output)

    # Build the compact replacement.
    # For skeletons and architecture: the SessionMap already has the structural data
    # (files, symbols, line numbers). The model should use @read_function() to drill
    # into specific code, NOT re-read the cached skeleton.
    rel_path = os.path.relpath(cache_path)
    if _detect_skeleton(output) or tool_type == "architecture":
        return (
            f"{summary}\n"
            f"[Symbols captured in SESSION MAP — use @read_function() to drill into specific code]"
        )
    return (
        f"{summary}\n"
        f"[Cached: {rel_path} — use @read_file(\"{rel_path}\") ONLY if you need a specific detail not shown above]"
    )


def clear_cache():
    """Remove all cached tool outputs (call between sessions or on /clear)."""
    import shutil
    if os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR, ignore_errors=True)


# =============================================================================
# SESSION MAP — Persistent navigation index for the current session.
#
# Tracks every file explored, every symbol found (with line numbers), every
# search result cached, and the current task. Rendered as a compact ASCII tree
# injected into the system prompt so the model can navigate without re-reading.
# =============================================================================

from dataclasses import dataclass, field


@dataclass
class SymbolEntry:
    """One symbol within a file."""
    name: str
    kind: str           # "function", "class", "method"
    line: int
    signature: str      # "def run_grep(content)"
    parent: str = ""    # enclosing class name (for methods)
    annotation: str = ""  # finding attached to this symbol


@dataclass
class FileNode:
    """One file the model has examined."""
    path: str
    line_count: int = 0
    symbols: list = field(default_factory=list)   # [SymbolEntry, ...]
    last_accessed: float = 0.0
    access_count: int = 0
    annotations: list = field(default_factory=list)  # file-level findings


@dataclass
class SearchEntry:
    """One cached search result."""
    label: str          # 'grep "run_" in tools/'
    match_count: int
    cache_file: str     # ".tool_cache/grep_003.txt"
    timestamp: float = 0.0


class SessionMap:
    """Persistent navigation index for the current session.

    Tracks files explored (with symbol-level detail), search results,
    and the current task. Renders as a compact ASCII tree for prompt injection.
    """

    MAX_FILES = 15
    MAX_SYMBOLS_PER_FILE = 12
    MAX_SEARCHES = 5

    def __init__(self):
        self._files = {}           # path -> FileNode
        self._searches = []        # [SearchEntry, ...]
        self._task_summary = ""

    # ===== UPDATE METHODS =====

    def set_task(self, text):
        """Set the task summary (from first user message)."""
        self._task_summary = text.strip()[:80]

    def update_from_architecture(self, output):
        """Parse architecture map output — extract files + symbol info.

        Tracks current directory context from indented 'dir/' lines so file
        paths include their directory prefix (e.g., 'core/config.py' not just 'config.py').
        """
        current_dir = ""
        for line in output.split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("=") or stripped.startswith("Project:"):
                continue

            # Detect directory context from indented lines like "  core/"
            indent = len(line) - len(line.lstrip())
            dir_m = re.match(r"(\w[\w.-]*)/$", stripped)
            if dir_m:
                if indent <= 2:
                    current_dir = dir_m.group(1) + "/"
                else:
                    # Nested dir — append to current
                    current_dir = current_dir.split("/")[0] + "/" + dir_m.group(1) + "/"
                continue

            # Reset dir context for top-level files (no indent)
            if indent <= 2 and not stripped.endswith("/"):
                current_dir = ""

            # Match: "filename.py (1.2KB) -funcs: a, b; classes: C"
            m = re.match(r"(\S+\.py)\s+\(([^)]+)\)\s*(?:-(.+))?", stripped)
            if m:
                filename = m.group(1)
                # Prepend directory context if file doesn't already have a path
                path = filename if "/" in filename or "\\" in filename else current_dir + filename
                info = m.group(3) or ""
                # Extract symbol names from info string
                symbols = []
                for kind_match in re.finditer(r"(funcs|classes|methods):\s*([^;]+)", info):
                    kind_word = kind_match.group(1)
                    kind = {"funcs": "function", "classes": "class", "methods": "method"}.get(kind_word, kind_word)
                    names = [n.strip() for n in kind_match.group(2).split(",") if n.strip()]
                    for name in names:
                        symbols.append(SymbolEntry(name=name, kind=kind, line=0, signature=""))
                self._upsert_file(path, 0, symbols)

    def update_from_skeleton(self, display, output):
        """Parse skeleton view — extract file info + all symbols with line numbers."""
        # Parse file header: "File: name.py (N lines) — SKELETON VIEW"
        file_m = re.search(r"File:\s*(\S+)\s*\((\d+)\s+lines\)", output)
        if not file_m:
            return
        filename = file_m.group(1)
        line_count = int(file_m.group(2))

        # Extract filepath from display: read_file: "path/to/file.py"
        path_m = re.search(r'"([^"]+)"', display)
        filepath = path_m.group(1) if path_m else filename

        # Parse symbols: "  L{n}: def name(args)" or "  L{n}: class Name"
        symbols = []
        for sym_m in re.finditer(r"L(\d+):\s*(def\s+\w+\([^)]*\)|class\s+\w+(?:\([^)]*\))?)", output):
            line = int(sym_m.group(1))
            sig = sym_m.group(2).strip()
            if sig.startswith("class "):
                name = sig.split()[1].split("(")[0]
                symbols.append(SymbolEntry(name=name, kind="class", line=line, signature=sig))
            elif sig.startswith("def "):
                name = sig.split("(")[0].replace("def ", "")
                symbols.append(SymbolEntry(name=name, kind="function", line=line, signature=sig))

        # Also parse method lines: "  L{n}: ClassName.method_name()"
        for meth_m in re.finditer(r"L(\d+):\s*(\w+)\.(\w+)\(\)", output):
            line = int(meth_m.group(1))
            parent = meth_m.group(2)
            name = meth_m.group(3)
            symbols.append(SymbolEntry(name=name, kind="method", line=line,
                                       signature=f"{parent}.{name}()", parent=parent))

        self._upsert_file(filepath, line_count, symbols)

    def update_from_read_function(self, display, output):
        """Parse read_function output — record the specific function read."""
        # Parse: "Function: name (lines X-Y of Z in filename)"
        m = re.search(r"Function:\s*(\w+)\s*\(lines\s+(\d+)-(\d+)\s+of\s+(\d+)\s+in\s+(\S+)\)", output)
        if not m:
            return
        func_name = m.group(1)
        start_line = int(m.group(2))
        total_lines = int(m.group(4))
        filename = m.group(5)

        # Get filepath from display
        path_m = re.search(r'"([^"]+)"', display)
        filepath = path_m.group(1) if path_m else filename

        sym = SymbolEntry(name=func_name, kind="function", line=start_line,
                          signature=f"def {func_name}(...)")
        self._upsert_symbol(filepath, total_lines, sym)

    def update_from_grep(self, display, output, cache_path=None):
        """Parse grep results — record search entry + register top files."""
        lines = output.strip().split("\n")
        # Count matches per file
        file_counts = {}
        for line in lines[1:]:
            m = re.match(r"^(.+?):(\d+):", line)
            if m:
                file_counts[m.group(1)] = file_counts.get(m.group(1), 0) + 1

        total = sum(file_counts.values())
        label = display  # e.g., grep: "run_" in tools/
        cache_rel = os.path.relpath(cache_path) if cache_path else ""

        self._searches.append(SearchEntry(
            label=label, match_count=total,
            cache_file=cache_rel, timestamp=time.time()
        ))
        if len(self._searches) > self.MAX_SEARCHES:
            self._searches = self._searches[-self.MAX_SEARCHES:]

        # Register top matched files
        for fpath, _ in sorted(file_counts.items(), key=lambda x: -x[1])[:5]:
            self._touch_file(fpath)

    def update_from_find_symbol(self, display, output):
        """Parse find_symbol results — record symbol definitions."""
        for line in output.strip().split("\n"):
            m = re.match(r"\s*(\S+):(\d+)\s+-(\w+)\s+(\S+)", line)
            if m:
                filepath = m.group(1)
                lineno = int(m.group(2))
                kind = m.group(3)
                name = m.group(4).split("(")[0]  # strip signature
                sym = SymbolEntry(name=name, kind=kind, line=lineno, signature="")
                self._upsert_symbol(filepath, 0, sym)

    def update_from_find_references(self, display, output):
        """Parse find_references — register files that contain references."""
        for line in output.strip().split("\n"):
            m = re.match(r"\s*(\S+):(\d+):", line)
            if m:
                self._touch_file(m.group(1))

    def update_from_glob(self, display, output):
        """Parse glob results — register discovered files."""
        for line in output.strip().split("\n"):
            line = line.strip()
            if line.startswith("Found "):
                continue
            m = re.match(r"(\S+)\s+\(", line)
            if m:
                self._touch_file(m.group(1))

    def annotate(self, filepath, line, note):
        """Attach a finding to a symbol or file."""
        if filepath in self._files:
            node = self._files[filepath]
            # Try to find the closest symbol
            for sym in node.symbols:
                if sym.line == line or (not sym.annotation and abs(sym.line - line) < 10):
                    sym.annotation = note
                    return
            node.annotations.append(f"L{line}: {note}")

    # ===== INTERNAL HELPERS =====

    def _upsert_file(self, filepath, line_count, symbols=None):
        """Add or update a file node. Merges symbols, bumps access."""
        if filepath in self._files:
            node = self._files[filepath]
            node.last_accessed = time.time()
            node.access_count += 1
            if line_count:
                node.line_count = line_count
            if symbols:
                existing = {(s.name, s.line) for s in node.symbols}
                for sym in symbols:
                    if (sym.name, sym.line) not in existing:
                        node.symbols.append(sym)
                node.symbols.sort(key=lambda s: s.line)
        else:
            self._evict_if_needed()
            self._files[filepath] = FileNode(
                path=filepath, line_count=line_count or 0,
                symbols=symbols or [], last_accessed=time.time(),
                access_count=1
            )

    def _upsert_symbol(self, filepath, line_count, sym):
        """Add or update a single symbol in a file."""
        if filepath not in self._files:
            self._upsert_file(filepath, line_count, [sym])
        else:
            node = self._files[filepath]
            node.last_accessed = time.time()
            node.access_count += 1
            for existing in node.symbols:
                if existing.name == sym.name and existing.line == sym.line:
                    return  # already tracked
            node.symbols.append(sym)
            node.symbols.sort(key=lambda s: s.line)

    def _touch_file(self, filepath):
        """Register a file was seen, without adding symbols."""
        if filepath in self._files:
            self._files[filepath].last_accessed = time.time()
            self._files[filepath].access_count += 1
        else:
            self._evict_if_needed()
            self._files[filepath] = FileNode(
                path=filepath, last_accessed=time.time(), access_count=1
            )

    def _evict_if_needed(self):
        """Remove lowest-priority file if at capacity."""
        if len(self._files) < self.MAX_FILES:
            return
        # Never evict files with annotations
        candidates = [
            (p, n) for p, n in self._files.items()
            if not n.annotations and not any(s.annotation for s in n.symbols)
        ]
        if not candidates:
            return
        victim = min(candidates, key=lambda x: (x[1].access_count, x[1].last_accessed))
        del self._files[victim[0]]

    # ===== RENDERING =====

    def render(self, token_budget=300):
        """Render the session map as a compact ASCII tree for prompt injection."""
        char_budget = token_budget * 4
        lines = []
        used = 0

        # Task line
        if self._task_summary:
            task_line = f'Task: "{self._task_summary}"'
            lines.append(task_line)
            used += len(task_line) + 1

        # Sort files: annotated first, then access count, then recency
        ranked = sorted(
            self._files.values(),
            key=lambda f: (
                bool(f.annotations) or any(s.annotation for s in f.symbols),
                f.access_count,
                f.last_accessed,
            ),
            reverse=True
        )

        files_shown = 0
        for i, fnode in enumerate(ranked):
            is_last = (i == len(ranked) - 1) or (files_shown >= self.MAX_FILES - 1)
            prefix = "\\--" if is_last else "+--"
            child_prefix = "   " if is_last else "|  "

            size_info = f" [{fnode.line_count} lines]" if fnode.line_count else ""
            file_line = f"{prefix} {fnode.path}{size_info}"
            cost = len(file_line) + 1

            # Build symbol lines
            sym_lines = []
            visible_syms = fnode.symbols[:self.MAX_SYMBOLS_PER_FILE]
            for j, sym in enumerate(visible_syms):
                is_last_sym = (j == len(visible_syms) - 1)
                sym_prefix = "\\--" if is_last_sym else "+--"
                ann = f" <- {sym.annotation}" if sym.annotation else ""
                if sym.line > 0:
                    sym_text = f"{child_prefix}{sym_prefix} L{sym.line}: {sym.signature or sym.name}{ann}"
                else:
                    sym_text = f"{child_prefix}{sym_prefix} {sym.name} ({sym.kind}){ann}"
                sym_lines.append(sym_text)
                cost += len(sym_text) + 1

            # Extra symbols indicator
            if len(fnode.symbols) > self.MAX_SYMBOLS_PER_FILE:
                extra = f"{child_prefix}   +{len(fnode.symbols) - self.MAX_SYMBOLS_PER_FILE} more"
                sym_lines.append(extra)
                cost += len(extra) + 1

            if used + cost > char_budget:
                break

            lines.append(file_line)
            lines.extend(sym_lines)
            used += cost
            files_shown += 1

            if is_last:
                break

        # Search entries
        for sentry in self._searches[-self.MAX_SEARCHES:]:
            ref = f" -> {sentry.cache_file}" if sentry.cache_file else ""
            s_line = f"+-- {sentry.label} [{sentry.match_count} matches]{ref}"
            if used + len(s_line) + 1 > char_budget:
                break
            lines.append(s_line)
            used += len(s_line) + 1

        # Overflow
        remaining = len(ranked) - files_shown
        if remaining > 0:
            overflow = f"[+{remaining} more files explored]"
            if used + len(overflow) + 1 <= char_budget:
                lines.append(overflow)

        return "\n".join(lines) if lines else "(no files explored yet)"

    def clear(self):
        """Reset the session map."""
        self._files.clear()
        self._searches.clear()
        self._task_summary = ""


# ===== DISPATCHER =====

def update_session_map(smap, action_type, display, output, cache_path=None):
    """Route tool output to the appropriate SessionMap updater.

    Called BEFORE maybe_cache_output() so we get the full output for extraction.
    """
    if not smap or not output:
        return

    if action_type == "architecture":
        smap.update_from_architecture(output)
    elif action_type == "read_file":
        if "SKELETON VIEW" in output[:300]:
            smap.update_from_skeleton(display, output)
        # Non-skeleton reads: just touch the file
        else:
            path_m = re.search(r'"([^"]+)"', display)
            if path_m:
                smap._touch_file(path_m.group(1))
    elif action_type == "read_function":
        smap.update_from_read_function(display, output)
    elif action_type == "grep":
        smap.update_from_grep(display, output, cache_path)
    elif action_type == "find_symbol":
        smap.update_from_find_symbol(display, output)
    elif action_type == "find_references":
        smap.update_from_find_references(display, output)
    elif action_type == "glob":
        smap.update_from_glob(display, output)
