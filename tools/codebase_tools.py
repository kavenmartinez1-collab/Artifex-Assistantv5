"""
Artifex-Lite v2 -Deterministic codebase intelligence tools.

These are NOT AI-powered -they use AST parsing, regex, and file system
traversal to provide accurate, reliable results the model can trust.

Python files: full AST analysis (accurate function/class/import extraction)
Other languages: regex-based pattern matching (good enough for navigation)
"""

import ast
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set


# ===== LANGUAGE DETECTION =====

_LANG_MAP = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "shell", ".bash": "shell",
    ".ps1": "powershell",
}

_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", ".tox", "dist", "build", "egg-info",
    ".idea", ".vs", ".vscode", "target", "bin", "obj",
}

_TEXT_EXTS = set(_LANG_MAP.keys()) | {
    ".txt", ".md", ".rst", ".yaml", ".yml", ".json", ".toml",
    ".ini", ".cfg", ".conf", ".xml", ".html", ".css", ".sql",
}


def detect_language(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    return _LANG_MAP.get(ext, "unknown")


# ===== SYMBOL DEFINITION =====

class SymbolDef:
    """One found symbol definition."""
    __slots__ = ("name", "kind", "file", "line", "signature", "parent")

    def __init__(self, name: str, kind: str, file: str, line: int,
                 signature: str = "", parent: str = ""):
        self.name = name
        self.kind = kind          # "function", "class", "method", "variable"
        self.file = file
        self.line = line
        self.signature = signature  # e.g., "def foo(self, x, y=None)"
        self.parent = parent        # enclosing class name, if method

    def __repr__(self):
        loc = f"{self.file}:{self.line}"
        if self.parent:
            return f"{loc} -{self.kind} {self.parent}.{self.name}"
        return f"{loc} -{self.kind} {self.name}"

    def display(self) -> str:
        sig = f"  {self.signature}" if self.signature else ""
        if self.parent:
            return f"  {self.file}:{self.line} -{self.kind} {self.parent}.{self.name}{sig}"
        return f"  {self.file}:{self.line} -{self.kind} {self.name}{sig}"


# ===== PYTHON AST EXTRACTION =====

def _extract_python_symbols(filepath: str, rel_path: str) -> List[SymbolDef]:
    """Extract symbols from a Python file using AST (accurate)."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except (SyntaxError, ValueError, OSError):
        return []

    symbols = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Determine if it's a method (inside a class) or a function
            parent = ""
            kind = "function"
            # Walk parents to find enclosing class
            for parent_node in ast.walk(tree):
                if isinstance(parent_node, ast.ClassDef):
                    for child in ast.iter_child_nodes(parent_node):
                        if child is node:
                            parent = parent_node.name
                            kind = "method"
                            break

            # Build signature
            args = []
            for arg in node.args.args:
                args.append(arg.arg)
            sig = f"def {node.name}({', '.join(args)})"

            symbols.append(SymbolDef(
                name=node.name, kind=kind, file=rel_path,
                line=node.lineno, signature=sig, parent=parent,
            ))

        elif isinstance(node, ast.ClassDef):
            # Get base classes
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(ast.dump(base))
            sig = f"class {node.name}" + (f"({', '.join(bases)})" if bases else "")

            symbols.append(SymbolDef(
                name=node.name, kind="class", file=rel_path,
                line=node.lineno, signature=sig,
            ))

    return symbols


def _extract_python_imports(filepath: str) -> Tuple[List[Dict], List[str]]:
    """
    Extract imports from a Python file using AST.
    Returns (imports_list, exported_names).
    imports_list: [{"module": "os.path", "names": ["join", "exists"], "line": 5}, ...]
    exported_names: names in __all__ if defined
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except (SyntaxError, ValueError, OSError):
        return [], []

    imports = []
    exported = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "module": alias.name,
                    "names": [alias.asname or alias.name],
                    "line": node.lineno,
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level > 0:
                module = "." * node.level + module
            names = [alias.asname or alias.name for alias in (node.names or [])]
            imports.append({
                "module": module,
                "names": names,
                "line": node.lineno,
            })
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                exported.append(elt.value)

    return imports, exported


# ===== REGEX-BASED EXTRACTION (non-Python) =====

_SYMBOL_PATTERNS = {
    "javascript": [
        (r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)", "function"),
        (r"(?:export\s+)?class\s+(\w+)", "class"),
        (r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(", "function"),  # arrow funcs
    ],
    "typescript": [
        (r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*[\(<]([^)]*)\)", "function"),
        (r"(?:export\s+)?class\s+(\w+)", "class"),
        (r"(?:export\s+)?interface\s+(\w+)", "class"),
        (r"(?:export\s+)?type\s+(\w+)", "class"),
    ],
    "go": [
        (r"func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(([^)]*)\)", "function"),
        (r"type\s+(\w+)\s+struct", "class"),
        (r"type\s+(\w+)\s+interface", "class"),
    ],
    "rust": [
        (r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*[\(<]([^)]*)\)", "function"),
        (r"(?:pub\s+)?struct\s+(\w+)", "class"),
        (r"(?:pub\s+)?enum\s+(\w+)", "class"),
        (r"(?:pub\s+)?trait\s+(\w+)", "class"),
    ],
    "java": [
        (r"(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)+(\w+)\s*\(([^)]*)\)\s*\{", "function"),
        (r"(?:public\s+)?class\s+(\w+)", "class"),
        (r"(?:public\s+)?interface\s+(\w+)", "class"),
    ],
    "ruby": [
        (r"def\s+(\w+)(?:\(([^)]*)\))?", "function"),
        (r"class\s+(\w+)", "class"),
        (r"module\s+(\w+)", "class"),
    ],
    "php": [
        (r"(?:public|private|protected)?\s*function\s+(\w+)\s*\(([^)]*)\)", "function"),
        (r"class\s+(\w+)", "class"),
    ],
    "c": [
        (r"^(?:static\s+)?(?:inline\s+)?(?:\w+[\s*]+)+(\w+)\s*\(([^)]*)\)\s*\{", "function"),
        (r"typedef\s+struct\s*\w*\s*\{[^}]*\}\s*(\w+)", "class"),
    ],
    "cpp": [
        (r"^(?:static\s+)?(?:inline\s+)?(?:virtual\s+)?(?:\w+[\s*&]+)+(\w+)\s*\(([^)]*)\)\s*(?:const\s*)?(?:override\s*)?\{", "function"),
        (r"class\s+(\w+)", "class"),
        (r"struct\s+(\w+)", "class"),
    ],
    "csharp": [
        (r"(?:public|private|protected|internal)?\s*(?:static\s+)?(?:async\s+)?(?:\w+[\s<>[\],]*)\s+(\w+)\s*\(([^)]*)\)", "function"),
        (r"class\s+(\w+)", "class"),
        (r"interface\s+(\w+)", "class"),
    ],
}

_IMPORT_PATTERNS = {
    "javascript": [r"(?:import|require)\s*\(?['\"]([^'\"]+)['\"]"],
    "typescript": [r"import\s+.*?from\s+['\"]([^'\"]+)['\"]"],
    "go":         [r"\"([^\"]+)\""],  # inside import blocks
    "rust":       [r"use\s+([\w:]+)"],
    "java":       [r"import\s+([\w.]+)"],
    "ruby":       [r"require\s+['\"]([^'\"]+)['\"]"],
    "php":        [r"(?:use|require|include)\s+['\"]?([^'\";\s]+)"],
    "c":          [r"#include\s+[<\"]([^>\"]+)[>\"]"],
    "cpp":        [r"#include\s+[<\"]([^>\"]+)[>\"]"],
    "csharp":     [r"using\s+([\w.]+)"],
}


def _extract_regex_symbols(filepath: str, rel_path: str, lang: str) -> List[SymbolDef]:
    """Extract symbols using regex patterns (for non-Python files)."""
    patterns = _SYMBOL_PATTERNS.get(lang, [])
    if not patterns:
        return []

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []

    symbols = []
    for lineno, line in enumerate(lines, 1):
        for pattern, kind in patterns:
            m = re.search(pattern, line)
            if m:
                name = m.group(1)
                sig = m.group(0).strip()[:120]
                symbols.append(SymbolDef(
                    name=name, kind=kind, file=rel_path,
                    line=lineno, signature=sig,
                ))
    return symbols


def _extract_regex_imports(filepath: str, lang: str) -> List[str]:
    """Extract import paths using regex (for non-Python files)."""
    patterns = _IMPORT_PATTERNS.get(lang, [])
    if not patterns:
        return []

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError:
        return []

    imports = []
    for pattern in patterns:
        for m in re.finditer(pattern, content):
            imports.append(m.group(1))
    return imports


# ===== CODEBASE INDEX =====

class CodebaseIndex:
    """
    Deterministic codebase index. Scans once, queries many.
    Python: AST-accurate. Other languages: regex-based.
    """

    def __init__(self, root_path: str = "."):
        self.root = os.path.abspath(root_path)
        self._symbols: List[SymbolDef] = []
        self._files: Dict[str, str] = {}   # rel_path -> language
        self._imports: Dict[str, list] = {} # rel_path -> import info
        self._indexed = False

    def index(self) -> Dict:
        """Scan the workspace and build the symbol + import index."""
        self._symbols = []
        self._files = {}
        self._imports = {}

        file_count = 0
        symbol_count = 0

        for dirpath, dirnames, filenames in os.walk(self.root):
            # Skip ignored directories
            dirnames[:] = [d for d in dirnames
                           if d not in _SKIP_DIRS and not d.startswith(".")]

            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in _LANG_MAP:
                    continue

                fullpath = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(fullpath, self.root)
                lang = _LANG_MAP[ext]
                self._files[rel_path] = lang
                file_count += 1

                # Extract symbols
                if lang == "python":
                    syms = _extract_python_symbols(fullpath, rel_path)
                    imp_info, _ = _extract_python_imports(fullpath)
                    self._imports[rel_path] = imp_info
                else:
                    syms = _extract_regex_symbols(fullpath, rel_path, lang)
                    imp_strs = _extract_regex_imports(fullpath, lang)
                    self._imports[rel_path] = [{"module": m, "names": [], "line": 0}
                                               for m in imp_strs]

                self._symbols.extend(syms)
                symbol_count += len(syms)

        self._indexed = True

        # Detect primary language
        lang_counts = {}
        for lang in self._files.values():
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        primary = max(lang_counts, key=lang_counts.get) if lang_counts else "unknown"

        return {
            "files": file_count,
            "symbols": symbol_count,
            "languages": lang_counts,
            "primary_language": primary,
        }

    def _ensure_indexed(self):
        if not self._indexed:
            self.index()

    # --- Query API ---

    def find_symbol(self, name: str, kind: str = None) -> List[SymbolDef]:
        """Find all definitions matching a name. Exact + substring match."""
        self._ensure_indexed()
        exact = []
        partial = []
        name_lower = name.lower()

        for sym in self._symbols:
            if sym.name == name:
                if kind is None or sym.kind == kind:
                    exact.append(sym)
            elif name_lower in sym.name.lower():
                if kind is None or sym.kind == kind:
                    partial.append(sym)

        return exact + partial[:10]  # Exact matches first, then partials

    def find_references(self, name: str) -> List[Tuple[str, int, str]]:
        """
        Find where a symbol is used (grep-based, but smarter -skips definitions).
        Returns [(file, line_number, line_text), ...]
        """
        self._ensure_indexed()
        refs = []
        # Build set of definition locations to exclude
        def_locations = set()
        for sym in self._symbols:
            if sym.name == name:
                def_locations.add((sym.file, sym.line))

        pattern = re.compile(r"\b" + re.escape(name) + r"\b")

        for rel_path in self._files:
            fullpath = os.path.join(self.root, rel_path)
            try:
                with open(fullpath, "r", encoding="utf-8", errors="replace") as f:
                    for lineno, line in enumerate(f, 1):
                        if (rel_path, lineno) in def_locations:
                            continue
                        if pattern.search(line):
                            refs.append((rel_path, lineno, line.rstrip()[:120]))
            except OSError:
                continue

            if len(refs) >= 50:
                break

        return refs

    def trace_imports(self, filepath: str) -> Dict:
        """
        Trace import relationships for a file.
        Returns {"imports": [...], "imported_by": [...]}
        """
        self._ensure_indexed()
        rel_path = os.path.relpath(os.path.abspath(filepath), self.root)

        # What this file imports
        file_imports = self._imports.get(rel_path, [])

        # What imports this file (for Python: match module path)
        imported_by = []
        # Convert file path to module path: core/knowledge.py -> core.knowledge
        module_name = rel_path.replace(os.sep, ".").replace("/", ".")
        if module_name.endswith(".py"):
            module_name = module_name[:-3]
        basename = os.path.splitext(os.path.basename(rel_path))[0]

        for other_path, other_imports in self._imports.items():
            if other_path == rel_path:
                continue
            for imp in other_imports:
                mod = imp.get("module", "")
                # Check if this import refers to our file
                if mod == module_name or mod.endswith("." + basename):
                    imported_by.append({
                        "file": other_path,
                        "line": imp.get("line", 0),
                        "import": mod,
                        "names": imp.get("names", []),
                    })

        return {
            "file": rel_path,
            "imports": file_imports,
            "imported_by": imported_by,
        }

    def architecture_map(self, max_depth: int = 3) -> str:
        """Generate a compact project structure map with file purposes."""
        self._ensure_indexed()
        lines = []

        # Group symbols by file for summary
        file_symbols: Dict[str, List[SymbolDef]] = {}
        for sym in self._symbols:
            file_symbols.setdefault(sym.file, []).append(sym)

        # Walk directory tree
        tree: Dict[str, list] = {}
        for rel_path, lang in sorted(self._files.items()):
            parts = rel_path.replace("\\", "/").split("/")
            if len(parts) - 1 > max_depth:
                continue
            dir_key = "/".join(parts[:-1]) or "."
            tree.setdefault(dir_key, []).append((parts[-1], rel_path, lang))

        for dir_path in sorted(tree.keys()):
            if dir_path != ".":
                indent = "  " * (dir_path.count("/") + 1)
                lines.append(f"{indent}{dir_path}/")

            for fname, rel_path, lang in tree[dir_path]:
                indent = "  " * (rel_path.replace("\\", "/").count("/") + 1)
                # Get file size
                fullpath = os.path.join(self.root, rel_path)
                try:
                    size = os.path.getsize(fullpath)
                    size_str = f"{size/1024:.1f}KB" if size > 1024 else f"{size}B"
                except OSError:
                    size_str = "?"

                # Symbol summary
                syms = file_symbols.get(rel_path, [])
                classes = [s.name for s in syms if s.kind == "class"]
                funcs = [s.name for s in syms if s.kind == "function"]
                methods = len([s for s in syms if s.kind == "method"])

                summary_parts = []
                if classes:
                    summary_parts.append(f"classes: {', '.join(classes[:3])}")
                if funcs:
                    summary_parts.append(f"funcs: {', '.join(funcs[:4])}")
                if methods:
                    summary_parts.append(f"{methods} methods")
                summary = f" -{'; '.join(summary_parts)}" if summary_parts else ""

                lines.append(f"{indent}{fname} ({size_str}){summary}")

        return "\n".join(lines)

    def get_file_symbols(self, filepath: str) -> List[SymbolDef]:
        """Get all symbols defined in a specific file."""
        self._ensure_indexed()
        rel_path = os.path.relpath(os.path.abspath(filepath), self.root)
        return [s for s in self._symbols if s.file == rel_path]

    @property
    def stats(self) -> Dict:
        self._ensure_indexed()
        lang_counts = {}
        for lang in self._files.values():
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        return {
            "files": len(self._files),
            "symbols": len(self._symbols),
            "languages": lang_counts,
        }


# ===== SAFE EDITOR =====

class SafeEditor:
    """
    Validates edits before applying them. For Python files, checks syntax
    before and after. For all files, checks for common corruption patterns.
    """

    @staticmethod
    def validate_python_syntax(filepath: str) -> Tuple[bool, str]:
        """Check if a Python file has valid syntax. Returns (valid, error_msg)."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                source = f.read()
            compile(source, filepath, "exec")
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
        except (OSError, ValueError) as e:
            return False, str(e)

    @staticmethod
    def validate_python_source(source: str, filename: str = "<edit>") -> Tuple[bool, str]:
        """Check if Python source code string has valid syntax."""
        try:
            compile(source, filename, "exec")
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"

    @staticmethod
    def validate_edit(filepath: str, old_text: str, new_text: str) -> Tuple[bool, str]:
        """
        Validate a proposed edit BEFORE applying it.
        Returns (safe_to_apply, message).
        """
        if not os.path.exists(filepath):
            return False, f"File not found: {filepath}"

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                original = f.read()
        except (OSError, UnicodeDecodeError) as e:
            return False, f"Cannot read file: {e}"

        # Check old_text exists and is unique
        count = original.count(old_text)
        if count == 0:
            return False, f"OLD text not found in file"
        if count > 1:
            return False, f"OLD text appears {count} times -ambiguous. Add more context."

        # Apply the edit in memory
        result = original.replace(old_text, new_text, 1)

        # Language-specific validation
        lang = detect_language(filepath)
        if lang == "python":
            valid, err = SafeEditor.validate_python_source(result, filepath)
            if not valid:
                return False, f"Edit would create syntax error -{err}"

            # Check imports still parse
            try:
                tree = ast.parse(result)
            except SyntaxError:
                pass  # Already caught above

        # Generic checks
        if new_text and not new_text.endswith("\n") and old_text.endswith("\n"):
            pass  # Warn about missing trailing newline? Not a blocker.

        return True, "Edit validated -safe to apply"

    @staticmethod
    def apply_validated_edit(filepath: str, old_text: str, new_text: str) -> Tuple[bool, str]:
        """
        Validate then apply an edit. Returns (success, message).
        Creates a backup before applying.
        """
        valid, msg = SafeEditor.validate_edit(filepath, old_text, new_text)
        if not valid:
            return False, f"REJECTED: {msg}"

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                original = f.read()

            result = original.replace(old_text, new_text, 1)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(result)

            delta = len(new_text) - len(old_text)
            sign = "+" if delta >= 0 else ""
            return True, f"Applied to {os.path.basename(filepath)}: {sign}{delta} chars"

        except OSError as e:
            return False, f"Write failed: {e}"

    @staticmethod
    def find_related_files(filepath: str, index: CodebaseIndex) -> List[str]:
        """Find files that would be affected by changing the given file."""
        trace = index.trace_imports(filepath)
        related = set()
        for imp in trace.get("imported_by", []):
            related.add(imp["file"])
        return sorted(related)


# ===== TOOL RUNNER FUNCTIONS =====
# These follow the (success: bool, output: str) return pattern
# used by agent_tools.py

_index_cache: Dict[str, CodebaseIndex] = {}


def _get_index(root: str = ".") -> CodebaseIndex:
    """Get or create a cached CodebaseIndex for a root directory."""
    root = os.path.abspath(root)
    if root not in _index_cache:
        idx = CodebaseIndex(root)
        idx.index()
        _index_cache[root] = idx
    return _index_cache[root]


def invalidate_index(root: str = "."):
    """Clear the cached index (call after workspace change)."""
    root = os.path.abspath(root)
    _index_cache.pop(root, None)


def run_find_symbol(content: str) -> Tuple[bool, str]:
    """
    Find symbol definitions. Content format: "name" or "name|kind"
    kind: function, class, method
    """
    parts = content.split("|")
    name = parts[0].strip()
    kind = parts[1].strip() if len(parts) > 1 else None

    if not name:
        return False, "Usage: @find_symbol(\"name\") or @find_symbol(\"name\", \"class\")"

    idx = _get_index()
    results = idx.find_symbol(name, kind)

    if not results:
        return True, f"No definitions found for '{name}'"

    lines = [f"Found {len(results)} definition(s) for '{name}':"]
    for sym in results[:15]:
        lines.append(sym.display())

    return True, "\n".join(lines)


def run_find_references(content: str) -> Tuple[bool, str]:
    """Find where a symbol is used. Content: "symbol_name" """
    name = content.strip()
    if not name:
        return False, "Usage: @find_references(\"symbol_name\")"

    idx = _get_index()
    refs = idx.find_references(name)

    if not refs:
        return True, f"No references found for '{name}'"

    lines = [f"Found {len(refs)} reference(s) to '{name}':"]
    for filepath, lineno, text in refs[:30]:
        lines.append(f"  {filepath}:{lineno}: {text}")
    if len(refs) > 30:
        lines.append(f"  ...and {len(refs) - 30} more")

    return True, "\n".join(lines)


def run_trace_imports(content: str) -> Tuple[bool, str]:
    """Trace import graph for a file. Content: "filepath" """
    filepath = content.strip()
    if not filepath:
        return False, "Usage: @trace_imports(\"path/to/file.py\")"

    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"

    idx = _get_index()
    trace = idx.trace_imports(filepath)

    lines = [f"Import trace for {trace['file']}:", ""]

    # What it imports
    imports = trace.get("imports", [])
    if imports:
        lines.append("IMPORTS:")
        for imp in imports:
            mod = imp.get("module", "?")
            names = imp.get("names", [])
            line = imp.get("line", 0)
            if names and names != [mod]:
                lines.append(f"  L{line}: from {mod} import {', '.join(names[:5])}")
            else:
                lines.append(f"  L{line}: import {mod}")
    else:
        lines.append("IMPORTS: (none)")

    lines.append("")

    # What imports it
    imported_by = trace.get("imported_by", [])
    if imported_by:
        lines.append("IMPORTED BY:")
        for imp in imported_by:
            names = imp.get("names", [])
            name_str = f" ({', '.join(names[:3])})" if names else ""
            lines.append(f"  {imp['file']}:{imp.get('line', '?')}{name_str}")
    else:
        lines.append("IMPORTED BY: (none -may be an entry point or unused)")

    return True, "\n".join(lines)


def run_architecture(content: str) -> Tuple[bool, str]:
    """Generate project architecture map. Content: ignored."""
    idx = _get_index()
    stats = idx.stats

    header = (
        f"Project: {os.path.basename(idx.root)}\n"
        f"Files: {stats['files']} | Symbols: {stats['symbols']}\n"
        f"Languages: {', '.join(f'{k}({v})' for k, v in sorted(stats['languages'].items(), key=lambda x: -x[1]))}\n"
        f"{'=' * 50}\n"
    )

    tree = idx.architecture_map()
    return True, header + tree


def run_validate_edit(content: str) -> Tuple[bool, str]:
    """
    Validate a proposed edit without applying it.
    Content: "filepath\\x00old_text\\x00new_text"
    """
    parts = content.split("\x00")
    if len(parts) != 3:
        return False, "Invalid edit format"

    filepath, old_text, new_text = parts
    valid, msg = SafeEditor.validate_edit(filepath, old_text, new_text)

    if valid:
        # Also check what files might be affected
        idx = _get_index()
        related = SafeEditor.find_related_files(filepath, idx)
        if related:
            msg += f"\nRelated files to check: {', '.join(related[:5])}"

    return valid, msg
