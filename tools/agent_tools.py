"""
Artifex-Assistant v5 — General-purpose tool execution for ASSISTANT mode.
Enables shell commands, Python execution, web search, file reading, and web page reading.
"""

import logging
import os
import io
import re
import sys
import shutil
import tempfile
import subprocess
import math
import time
import random

_log = logging.getLogger(__name__)
from collections import namedtuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode

from core.config import IS_WINDOWS, ASSISTANT_DANGEROUS_PATTERNS, WEB_GATEWAY_URL, GATEWAY_AUTH_TOKEN
from core.hardware import sense_system
from core.sandbox.proc_sandbox import (
    MAX_COMMAND_TIMEOUT,
    MAX_PYTHON_TIMEOUT,
    scrub_env,
)

# HTML parsing: lxml for quality, stdlib HTMLParser always available as fallback
from html.parser import HTMLParser
try:
    import lxml.html
    _HAS_LXML = True
except ImportError:
    _HAS_LXML = False

# JSON support for gateway communication
import json
from urllib.parse import urljoin


# ─────────────────────────────────────────────────────────────────────────────
# Web Gateway helpers (route through Docker proxy when available)
# ─────────────────────────────────────────────────────────────────────────────

_gateway_available = None  # None = not checked, True/False = cached result
_gateway_checked_at = 0.0
_GATEWAY_CACHE_TTL = 30  # re-check every 30s if previously unavailable


def _check_gateway():
    """Check if the web gateway is reachable. Caches success permanently,
    retries failures every 30s so a late-starting Docker container is found."""
    global _gateway_available, _gateway_checked_at
    if _gateway_available:
        return True
    if not WEB_GATEWAY_URL:
        return False
    now = time.monotonic()
    if _gateway_available is not None and now - _gateway_checked_at < _GATEWAY_CACHE_TTL:
        return _gateway_available
    try:
        req = Request(f"{WEB_GATEWAY_URL}/health", method="GET")
        with urlopen(req, timeout=3) as resp:
            _gateway_available = resp.status == 200
    except Exception as e:
        _log.debug("Web gateway health check failed: %s", e)
        _gateway_available = False
    _gateway_checked_at = now
    return _gateway_available


def _gateway_post(endpoint, payload):
    """POST JSON to the web gateway. Returns (success, data_dict) or (False, error_str)."""
    url = f"{WEB_GATEWAY_URL}{endpoint}"
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if GATEWAY_AUTH_TOKEN:
        headers["X-Gateway-Token"] = GATEWAY_AUTH_TOKEN
    req = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return True, data
    except HTTPError as e:
        try:
            err_body = json.loads(e.read().decode("utf-8"))
            return False, err_body.get("detail", f"HTTP {e.code}")
        except Exception:
            return False, f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        return False, str(e)


def _gateway_delete(endpoint):
    """DELETE request to the web gateway."""
    url = f"{WEB_GATEWAY_URL}{endpoint}"
    headers = {}
    if GATEWAY_AUTH_TOKEN:
        headers["X-Gateway-Token"] = GATEWAY_AUTH_TOKEN
    req = Request(url, method="DELETE", headers=headers)
    try:
        with urlopen(req, timeout=10) as resp:
            return True, json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return False, str(e)


def reset_gateway_cache():
    """Reset the gateway availability cache (e.g., after Docker starts)."""
    global _gateway_available
    _gateway_available = None


# ─────────────────────────────────────────────────────────────────────────────
# Platform-aware shell + Python detection (resolved once at import time)
# ─────────────────────────────────────────────────────────────────────────────

def _find_git_bash():
    """Find Git Bash on Windows (not WSL bash)."""
    if not IS_WINDOWS:
        return None
    bash = shutil.which("bash")
    if bash and "git" in bash.lower():
        return bash
    # Common install paths
    for candidate in (
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


_GIT_BASH = _find_git_bash()

# Common Unix commands that signal bash syntax (first word of command)
_BASH_COMMANDS = frozenset({
    "ls", "cat", "grep", "awk", "sed", "find", "head", "tail", "wc",
    "chmod", "chown", "mkdir", "cp", "mv", "rm", "touch", "tar",
    "curl", "wget", "ssh", "scp", "rsync", "df", "du", "ps", "kill",
    "sort", "uniq", "cut", "tr", "xargs", "tee", "diff", "which",
    "export", "source", "echo",
})


def _is_bash_syntax(command):
    """Detect if a command uses bash/Unix syntax vs PowerShell."""
    # Bash operators
    if "&&" in command or "||" in command:
        return True
    # Heredocs are bash-only — PowerShell has no equivalent syntax
    if re.search(r"<<-?\s*['\"]?[A-Za-z_]\w*", command):
        return True
    # Bash redirects with fd: 2>&1, 2>/dev/null
    if re.search(r"\d+>&\d+|/dev/null", command):
        return True
    # Check first word of command (strip leading env vars like VAR=val)
    stripped = re.sub(r"^(\w+=\S+\s+)*", "", command.strip())
    first_word = stripped.split()[0] if stripped.split() else ""
    if first_word in _BASH_COMMANDS:
        return True
    # Pipe to Unix commands: | grep, | awk, | sort, etc.
    pipe_cmds = re.findall(r"\|\s*(\w+)", command)
    if any(c in _BASH_COMMANDS for c in pipe_cmds):
        return True
    return False


def _bash_to_powershell(command):
    """Quick-translate the most common bash patterns to PowerShell."""
    cmd = command
    # && → ; (sequential execution)
    cmd = cmd.replace(" && ", " ; ")
    # || → ; (best-effort, PS doesn't have direct equivalent)
    cmd = cmd.replace(" || ", " ; ")
    # 2>&1 → *>&1 (redirect all streams)
    cmd = cmd.replace("2>&1", "*>&1")
    # /dev/null → $null
    cmd = cmd.replace("/dev/null", "$null")
    return cmd


def _find_venv_python():
    """Find the best Python interpreter — prefer the venv running Artifex."""
    # sys.executable is the interpreter that launched us (venv-aware)
    if sys.executable and os.path.isfile(sys.executable):
        return sys.executable
    # Fallback
    return "python" if IS_WINDOWS else "python3"


_PYTHON_BIN = _find_venv_python()


def _is_likely_command(line):
    """Heuristic: does this stripped line look like a shell command (not prose)?"""
    if not line or line.startswith("#") or line.startswith("\\"):
        return False
    if (re.match(r"^[A-Z]", line)
            and re.search(r"[.!?:]\s*$", line)
            and not re.search(r"[|><;$`\\=]", line)):
        return False
    if re.match(
        r"^(Output|Example|Note|Result|Expected|Usage|Then|Next|First|"
        r"Now|After|Before|Step|This|The|You|If|Or|And)\b",
        line, re.IGNORECASE,
    ) and not re.search(r"[|><;$`\\=\-]", line):
        return False
    return True


# ===== ACTION TYPES =====
AgentAction = namedtuple("AgentAction", ["type", "content", "display"])
# type: "shell" | "python" | "search" | "read_file" | "web_read" | "download"
#      | "glob" | "grep" | "edit_file"
#      | "find_symbol" | "find_references" | "trace_imports" | "architecture"
# content: the raw code/query/path string
# display: human-friendly summary for confirmation prompt

# Per-tool output limits (chars fed back to the model).
# Shared by CLI and GUI so they stay in sync.
# These are the STANDARD baseline values — actual limits scale with the active
# context profile via get_tool_output_limit().
TOOL_OUTPUT_LIMITS = {
    "shell":           4000,
    "python":          4000,
    "search":          5000,
    "read_file":       3500,
    "web_read":        4000,
    "download":        2000,
    "glob":            3000,
    "grep":            3500,
    "edit_file":       1000,
    "find_symbol":     3000,
    "find_references": 3500,
    "trace_imports":   3000,
    "architecture":    4000,
    "read_function":   3500,
}
DEFAULT_OUTPUT_LIMIT = 4000

# Ratios relative to the profile's base tool_output_limit (4000 for STANDARD).
# Tools not listed here get the base limit (ratio = 1.0).
_TOOL_LIMIT_RATIOS = {
    "edit_file": 0.25,    # 1000 @ base=4000
    "download":  0.50,    # 2000 @ base=4000
    "search":    1.25,    # 5000 @ base=4000
}


def get_tool_output_limit(tool_type=None):
    """Get the output char limit for a tool type, scaled by context profile."""
    from core.config import get_context_profile
    base = get_context_profile().tool_output_limit
    if tool_type and tool_type in _TOOL_LIMIT_RATIOS:
        return int(base * _TOOL_LIMIT_RATIOS[tool_type])
    return base

# Cache the last search results so @web_read(N) can reference them.
_last_search_results = []


def _extract_fenced_blocks(text, languages, validate=None):
    """Extract content from fenced code blocks using string scanning.

    Safe for arbitrarily large input — O(n) with no regex backtracking.

    validate: optional callable(content) -> bool used to pick the CLOSING
    fence. When the block's content itself contains ``` (a python block
    writing a markdown file with code fences inside a string literal — a
    real Qwen3.6 agent behavior that used to truncate the code mid-string),
    the first close fence is wrong. With a validator, each candidate close
    is tried in order and the first whose content validates wins; if none
    validates, the first fence is used (legacy behavior, so genuinely
    broken code still surfaces its own error).
    """
    blocks = []
    pos = 0
    while pos < len(text):
        fence_start = text.find("```", pos)
        if fence_start == -1:
            break
        line_end = text.find("\n", fence_start)
        if line_end == -1:
            break
        tag = text[fence_start + 3:line_end].strip().lower()
        if tag not in languages:
            pos = fence_start + 3
            continue
        content_start = line_end + 1

        candidates = []
        search = content_start
        while True:
            c = text.find("```", search)
            if c == -1:
                break
            candidates.append(c)
            search = c + 3
        if not candidates:
            break

        close = None
        if validate is not None:
            for c in candidates:
                try:
                    if validate(text[content_start:c]):
                        close = c
                        break
                except Exception:
                    continue
        if close is None:
            close = candidates[0]

        blocks.append(text[content_start:close])
        pos = close + 3
    return blocks


def _python_parses(code):
    """AST-validity check used to disambiguate nested fences in ```python```."""
    import ast
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


_MARKER_TOOL_NAMES = (
    "search|read_file|read_function|find_symbol|find_references"
    "|grep|glob|web_read|download|trace_imports|architecture"
)

# Hybrid syntax some chat-template-trained models (Qwen3.x under --jinja)
# emit: the marker wrapped in their native tool-call tag, e.g.
#   <tool_call>:glob("config.ini")   or   <tool_call>@grep("x", ".")
# Normalized back to plain @marker form before marker matching.
_HYBRID_TOOL_CALL_RE = re.compile(
    rf'<tool_call>\s*[:@]?\s*({_MARKER_TOOL_NAMES}|done|finish)\s*\(',
)

# Fully-native JSON tool calls: <tool_call>{"name": ..., "arguments": ...}</tool_call>
_JSON_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Argument-name aliases seen in the wild for the JSON form.
_JSON_ARG_ALIASES = {
    "path": ("path", "file", "filepath", "file_path", "filename"),
    "pattern": ("pattern", "glob", "query"),
    "name": ("name", "symbol", "function", "function_name"),
    "query": ("query", "q", "search", "text"),
    "url": ("url", "link", "href"),
    "command": ("command", "cmd", "shell"),
    "code": ("code", "script", "source"),
}


def _json_arg(args, canonical):
    for key in _JSON_ARG_ALIASES.get(canonical, (canonical,)):
        if key in args and args[key] is not None:
            return str(args[key])
    return None


def _strip_inline_code(text):
    """Blank out `inline code` spans and **bold-wrapped markers** OUTSIDE
    fenced blocks.

    The system prompt promises that backticked tool markers are inert
    ("when listing tools, use prose or backticks") — without this, a model
    politely listing its tools in a table fires every one of them. Models
    also render tool listings as **@tool("path")** bold items (observed on
    Qwen3.6 via agent_bench); a marker fully enclosed in bold is
    documentation, not a call. Fenced ``` blocks are preserved untouched
    (shell/python/edit extraction and the marker-recovery path depend on
    their contents).
    """
    out = []
    pos = 0
    while True:
        fence = text.find("```", pos)
        segment = text[pos: fence if fence != -1 else len(text)]
        segment = re.sub(r"`[^`\n]*`", "``", segment)
        segment = re.sub(r"\*\*@[^*\n]{0,200}\*\*", "**", segment)
        out.append(segment)
        if fence == -1:
            break
        close = text.find("```", fence + 3)
        if close == -1:
            out.append(text[fence:])
            break
        out.append(text[fence:close + 3])
        pos = close + 3
    return "".join(out)


def _extract_json_tool_calls(response):
    """Parse native <tool_call>{JSON}</tool_call> calls into AgentActions.

    Qwen3.x models running under llama-server --jinja are trained on this
    format and occasionally fall back to it despite the @marker prompt.
    Only tools with an existing executor are mapped; unknown names are
    ignored (better a stall than a hallucinated capability).
    """
    actions = []
    for m in _JSON_TOOL_CALL_RE.finditer(response):
        try:
            obj = json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name", "")).lower().strip()
        args = obj.get("arguments") or obj.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        if not isinstance(args, dict):
            args = {}

        if name == "read_file":
            path = _json_arg(args, "path")
            if path:
                actions.append(AgentAction("read_file", f"{path}|1",
                                           f'read_file: "{path}"'))
        elif name == "read_function":
            path, fn = _json_arg(args, "path"), _json_arg(args, "name")
            if path and fn:
                actions.append(AgentAction("read_function", f"{path}|{fn}",
                                           f'read_function: "{fn}" in {os.path.basename(path)}'))
        elif name == "glob":
            pattern = _json_arg(args, "pattern")
            if pattern:
                actions.append(AgentAction("glob", pattern, f'glob: "{pattern}"'))
        elif name == "grep":
            pattern, path = _json_arg(args, "pattern"), _json_arg(args, "path") or "."
            if pattern:
                actions.append(AgentAction("grep", f"{pattern}|{path}|",
                                           f'grep: "{pattern}" in {path}'))
        elif name == "search":
            query = _json_arg(args, "query")
            if query:
                actions.append(AgentAction("search", query, f'search: "{query}"'))
        elif name == "web_read":
            ref = _json_arg(args, "url")
            if ref:
                actions.append(AgentAction("web_read", ref, f"web_read: {ref}"))
        elif name == "find_symbol":
            sym = _json_arg(args, "name")
            if sym:
                actions.append(AgentAction("find_symbol", sym,
                                           f'find_symbol: "{sym}"'))
        elif name == "find_references":
            sym = _json_arg(args, "name")
            if sym:
                actions.append(AgentAction("find_references", sym,
                                           f'find_references: "{sym}"'))
        elif name == "trace_imports":
            path = _json_arg(args, "path")
            if path:
                actions.append(AgentAction("trace_imports", path,
                                           f'trace_imports: "{path}"'))
        elif name == "architecture":
            actions.append(AgentAction("architecture", "", "architecture: project map"))
        elif name in ("shell", "bash", "run_shell", "execute"):
            cmd = _json_arg(args, "command")
            if cmd:
                actions.append(AgentAction("shell", cmd, cmd[:80]))
        elif name in ("python", "run_python"):
            code = _json_arg(args, "code")
            if code:
                first = code.strip().split("\n")[0]
                actions.append(AgentAction("python", code, first[:80]))
    return actions


def extract_agent_actions(response):
    """
    Extract executable actions from an ASSISTANT mode response.

    Detects:
      1. ```bash / ```sh / ```shell / ```cmd / ```powershell blocks -> shell actions
      2. ```python / ```py blocks -> python actions
      3. @search("query") markers -> search actions
      4. Native <tool_call> forms (hybrid @marker and JSON) from
         chat-template-trained models

    Markers inside `inline code` spans are documentation, not calls, and are
    ignored (fenced blocks keep their existing semantics).

    Returns list of AgentAction tuples.
    """
    actions = []

    # Marker regexes run against a normalized copy: inline code spans blanked
    # (backticked mentions stay inert) and native tool-call wrappers folded
    # back into @marker form. Fenced-block extraction uses the raw response.
    marker_text = _strip_inline_code(response)
    marker_text = _HYBRID_TOOL_CALL_RE.sub(r"@\1(", marker_text)
    marker_text = marker_text.replace("</tool_call>", "")

    # Tool marker pattern — lines that are @tool(...) calls, not shell commands.
    # Some models (especially Ollama) mistakenly wrap these in code blocks.
    _TOOL_MARKER_RE = re.compile(
        r'^\s*@(?:search|read_file|read_function|find_symbol|find_references'
        r'|grep|glob|web_read|download|trace_imports|architecture)\s*\(',
    )

    # --- Shell code blocks (treat entire block as one command for powershell) ---
    shell_blocks = _extract_fenced_blocks(
        response, {"bash", "sh", "shell", "console", "cmd", "powershell", "zsh"},
    )
    for block in shell_blocks:
        block = block.strip()
        if not block:
            continue

        lines = block.split("\n")

        # Filter out lines that are tool markers (not shell commands)
        lines = [l for l in lines if not _TOOL_MARKER_RE.match(l)]
        if not lines:
            continue

        # Heredocs must survive as ONE command — line-splitting executes the
        # document BODY as commands (observed: python source fed line-by-line
        # into PowerShell). The whole block goes to bash, which understands it.
        if re.search(r"<<-?\s*['\"]?[A-Za-z_]\w*", lines[0]) or any(
                re.search(r"\bcat\b.*<<-?\s*['\"]?[A-Za-z_]\w*", l) for l in lines):
            heredoc_block = "\n".join(lines).strip()
            if heredoc_block:
                display = lines[0].strip()[:80]
                actions.append(AgentAction("shell", heredoc_block, display))
            continue

        # If it's a multi-line pipeline (PowerShell piped command), keep as one
        # Detect by checking if it's a single logical command with line continuations
        # or pipes, or if it has multiple independent commands
        joined = " ".join(l.rstrip("\\").strip() for l in lines)

        # Check if it looks like one piped/continued command
        if len(lines) > 1 and ("|" in block or "\\" in block):
            # Multi-line piped command — keep as single command
            if _is_likely_command(joined):
                display = joined[:80] + ("..." if len(joined) > 80 else "")
                actions.append(AgentAction("shell", joined, display))
        else:
            # Multiple separate commands
            for line in lines:
                line = re.sub(r"^[$#>]\s+", "", line.strip())
                if _is_likely_command(line):
                    actions.append(AgentAction("shell", line, line))

    # --- Python code blocks (AST-validated close: content may embed ```) ---
    python_blocks = _extract_fenced_blocks(response, {"python", "py"},
                                           validate=_python_parses)
    for block in python_blocks:
        code = block.strip()
        if not code:
            continue
        lines = code.split("\n")
        if len(lines) == 1:
            display = code[:80]
        else:
            display = f"{lines[0]}  ({len(lines)} lines)"
        actions.append(AgentAction("python", code, display))

    # --- Web search markers ---
    search_matches = re.findall(r'@search\(["\'](.+?)["\']\)', marker_text)
    for query in search_matches:
        actions.append(AgentAction("search", query, f'search: "{query}"'))

    # --- File read markers: @read_file("path") or @read_file("path", chunk=N) ---
    read_file_matches = re.findall(
        r'@read_file\(["\'](.+?)["\']\s*(?:,\s*chunk\s*=\s*(\d+))?\)',
        marker_text,
    )
    for filepath, chunk in read_file_matches:
        chunk_num = chunk if chunk else "1"
        content = f"{filepath}|{chunk_num}"
        display = f'read_file: "{filepath}"'
        if chunk:
            display += f" (chunk {chunk})"
        actions.append(AgentAction("read_file", content, display))

    # --- Web read markers: @web_read("url") or @web_read(N) ---
    web_read_matches = re.findall(
        r'@web_read\((?:["\'](.+?)["\']|(\d+))\)',
        marker_text,
    )
    for url, num in web_read_matches:
        ref = url if url else num
        display = f"web_read: {ref}"
        actions.append(AgentAction("web_read", ref, display))

    # --- Download markers: @download("url") or @download("url", "filename") ---
    download_matches = re.findall(
        r'@download\(["\'](.+?)["\']\s*(?:,\s*["\'](.+?)["\'])?\)',
        marker_text,
    )
    for dl_url, dl_name in download_matches:
        content = f"{dl_url}|{dl_name}" if dl_name else dl_url
        fname = dl_name or dl_url.split("/")[-1].split("?")[0] or "file"
        display = f'download: "{fname}"'
        actions.append(AgentAction("download", content, display))

    # --- Glob markers: @glob("pattern") or @glob("pattern", "base_dir") or @glob("pattern", "+all") ---
    glob_matches = re.findall(
        r'@glob\(["\'](.+?)["\']\s*(?:,\s*["\'](.+?)["\'])?\s*(?:,\s*["\'](.+?)["\'])?\)',
        marker_text,
    )
    for pattern, arg2, arg3 in glob_matches:
        # arg2 can be a base_dir or "+all" flag; arg3 is optional "+all" if arg2 was a dir
        if arg2.strip().lower() == "+all":
            content = f"{pattern}||+all"
            display = f'glob: "{pattern}" (+all)'
        elif arg3.strip().lower() == "+all":
            content = f"{pattern}|{arg2}|+all"
            display = f'glob: "{pattern}" in {arg2} (+all)'
        elif arg2:
            content = f"{pattern}|{arg2}"
            display = f'glob: "{pattern}" in {arg2}'
        else:
            content = pattern
            display = f'glob: "{pattern}"'
        actions.append(AgentAction("glob", content, display))

    # --- Grep markers: @grep("pattern", "path") or @grep("pattern", "path", "flags") ---
    grep_matches = re.findall(
        r'@grep\(["\'](.+?)["\']\s*,\s*["\'](.+?)["\']\s*(?:,\s*["\']([^"\']*)["\'])?\)',
        marker_text,
    )
    for pattern, path, flags in grep_matches:
        content = f"{pattern}|{path}|{flags}"
        display = f'grep: "{pattern}" in {path}'
        actions.append(AgentAction("grep", content, display))

    # --- Edit file blocks: ```edit ... ``` ---
    # Format:
    #   FILE: path/to/file
    #   OLD:
    #   exact text to replace
    #   NEW:
    #   replacement text
    edit_blocks = _extract_fenced_blocks(response, {"edit"})
    for block in edit_blocks:
        file_m = re.search(r"^FILE:[ \t]*(.+)$", block, re.MULTILINE)
        # \n? before NEW: — an empty OLD (create-file form) is written as
        # "OLD:\nNEW:" with no blank line between them.
        old_m = re.search(r"^OLD:[ \t]*\n?(.*?)\n?NEW:", block, re.DOTALL | re.MULTILINE)
        # \Z (absolute end), NOT $: with MULTILINE, a lazy (.*?)$ stops at the
        # FIRST newline, silently truncating every multi-line NEW replacement
        # to its first line (found via agent_bench edit_block_format probe).
        new_m = re.search(r"^NEW:[ \t]*\n(.*?)\s*\Z", block, re.DOTALL | re.MULTILINE)
        if not (file_m and old_m and new_m):
            continue
        path = file_m.group(1).strip()
        old_str = old_m.group(1)
        new_str = new_m.group(1)
        # Use a safe internal delimiter unlikely to appear in code
        content = "\x00".join([path, old_str, new_str])
        preview = old_str.strip()[:50].replace("\n", "↵")
        display = f'edit: {path} ("{preview}...")'
        actions.append(AgentAction("edit_file", content, display))

    # --- Codebase intelligence markers ---

    # @find_symbol("name") or @find_symbol("name", "class")
    sym_matches = re.findall(
        r'@find_symbol\(["\'](.+?)["\']\s*(?:,\s*["\'](\w+)["\'])?\)',
        marker_text,
    )
    for name, kind in sym_matches:
        content = f"{name}|{kind}" if kind else name
        display = f'find_symbol: "{name}"' + (f" ({kind})" if kind else "")
        actions.append(AgentAction("find_symbol", content, display))

    # @find_references("symbol")
    ref_matches = re.findall(r'@find_references\(["\'](.+?)["\']\)', marker_text)
    for name in ref_matches:
        actions.append(AgentAction("find_references", name, f'find_references: "{name}"'))

    # @trace_imports("filepath")
    imp_matches = re.findall(r'@trace_imports\(["\'](.+?)["\']\)', marker_text)
    for path in imp_matches:
        actions.append(AgentAction("trace_imports", path, f'trace_imports: "{path}"'))

    # @architecture()
    if re.search(r'@architecture\(\s*\)', marker_text):
        actions.append(AgentAction("architecture", "", "architecture: project map"))

    # @sysinfo()
    if re.search(r'@sysinfo\(\s*\)', marker_text):
        actions.append(AgentAction("sysinfo", "", "sysinfo: machine specs"))

    # @read_function("filepath", "function_name")
    read_fn_matches = re.findall(
        r'@read_function\(["\'](.+?)["\']\s*,\s*["\'](.+?)["\']\)',
        marker_text,
    )
    for fpath, fname in read_fn_matches:
        content = f"{fpath}|{fname}"
        display = f'read_function: "{fname}" in {os.path.basename(fpath)}'
        actions.append(AgentAction("read_function", content, display))

    # --- Gemma 4 tool call format: <|tool_call>call:func{k:v,...}<tool_call|> ---
    gemma_tool_calls = re.findall(
        r'<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>',
        marker_text, re.DOTALL,
    )
    for func_name, args_str in gemma_tool_calls:
        # Parse Gemma 4 arguments: key:<|"|>value<|"|> or key:plain_value
        args = {}
        for key, quoted_val, plain_val in re.findall(
            r'(\w+):(?:<\|"\|>(.*?)<\|"\|>|([^,}]*))', args_str
        ):
            args[key] = (quoted_val or plain_val).strip()

        # Map to existing action types where possible
        if func_name == "search" and "query" in args:
            actions.append(AgentAction("search", args["query"],
                                       f'search: "{args["query"]}"'))
        elif func_name == "read_file" and "path" in args:
            actions.append(AgentAction("read_file", args["path"],
                                       f'read_file: "{args["path"]}"'))
        else:
            # Generic tool call — display as-is
            display = f'{func_name}({", ".join(f"{k}={v}" for k, v in args.items())})'
            actions.append(AgentAction("shell", f"# Gemma tool: {display}",
                                       f"tool: {display}"))

    # --- Native JSON tool calls: <tool_call>{"name": ...}</tool_call> ---
    actions.extend(_extract_json_tool_calls(response))

    return actions


def _check_dangerous(command):
    """Soft guard against obvious LLM hallucinations (rm -rf /, format c:, etc.).

    NOT a security boundary — commands that pass are executed via shell=True with
    full filesystem access. Do not expose the LLM session to untrusted input.
    """
    cmd_lower = command.lower()

    # Standard pattern matching
    for pattern in ASSISTANT_DANGEROUS_PATTERNS:
        if pattern.lower() in cmd_lower:
            return f"Blocked dangerous pattern: {pattern}"

    # Detect command substitution that could hide dangerous commands
    # e.g., $(echo 'rm -rf /') or `echo 'rm -rf /'`
    subst_contents = re.findall(r'\$\((.+?)\)', command, re.DOTALL)
    subst_contents += re.findall(r'`(.+?)`', command)
    for inner in subst_contents:
        inner_check = _check_dangerous(inner)
        if inner_check:
            return f"Blocked dangerous pattern in command substitution: {inner.strip()[:60]}"

    # Detect eval/exec wrappers: eval "cmd", bash -c "cmd", sh -c "cmd"
    eval_match = re.search(
        r'(?:eval|bash\s+-c|sh\s+-c|cmd\s+/c)\s+["\'](.+?)["\']',
        command, re.IGNORECASE,
    )
    if eval_match:
        inner_check = _check_dangerous(eval_match.group(1))
        if inner_check:
            return f"Blocked dangerous pattern in eval/exec wrapper"

    # Detect quote-splitting bypasses: r''m, r""m -> rm
    # Remove all single and double quotes and re-check
    stripped = re.sub(r"['\"]", "", command)
    if stripped != command:
        for pattern in ASSISTANT_DANGEROUS_PATTERNS:
            if pattern.lower() in stripped.lower():
                return f"Blocked dangerous pattern (quote bypass): {pattern}"

    return None


def _get_clean_env():
    """Get a clean environment for command execution.

    Strips secrets via scrub_env() so child processes never see API keys,
    cloud credentials, or the agent key. Then layers on the encoding hints
    we actually want children to inherit.

    Also puts the venv's Scripts/bin dir at the front of PATH so shell
    commands resolve `python`/`pip`/`pytest` to the SAME interpreter that
    ```python``` blocks use. Without this the agent lives in a split-brain
    world: python blocks ran in the venv while `python -m pytest` in a
    shell hit the bare system Python (no pytest) and `pip` wasn't found
    at all (agent_bench s3 failure mode).
    """
    env = scrub_env()
    env["PYTHONIOENCODING"] = "utf-8"
    if IS_WINDOWS:
        env["PYTHONUTF8"] = "1"
    bin_dir = os.path.dirname(_PYTHON_BIN) if os.path.isabs(_PYTHON_BIN) else ""
    if bin_dir:
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
        venv_root = os.path.dirname(bin_dir)
        if os.path.isfile(os.path.join(venv_root, "pyvenv.cfg")):
            env["VIRTUAL_ENV"] = venv_root
    return env


def run_shell_command(command, timeout=300, cwd=None):
    """
    Execute a shell command.
    On Windows: routes bash-style commands to Git Bash if available,
    otherwise translates common patterns to PowerShell.
    On Linux/Mac: uses /bin/sh.
    Returns (success, output) tuple.

    Timeout is capped at MAX_COMMAND_TIMEOUT from the sandbox so a misbehaving
    caller can't pin a child process indefinitely.
    """
    timeout = min(timeout, MAX_COMMAND_TIMEOUT)
    blocked = _check_dangerous(command)
    if blocked:
        return False, blocked

    try:
        if IS_WINDOWS:
            use_bash = _is_bash_syntax(command)

            if use_bash and _GIT_BASH:
                # Route bash-style commands through Git Bash
                result = subprocess.run(
                    [_GIT_BASH, "-c", command],
                    capture_output=True, text=True, timeout=timeout,
                    cwd=cwd, env=_get_clean_env(),
                    encoding="utf-8", errors="replace",
                )
            else:
                # PowerShell path — translate bash-isms if no Git Bash
                ps_cmd = _bash_to_powershell(command) if use_bash else command
                ps_bin = shutil.which("pwsh") or shutil.which("powershell")
                if ps_bin:
                    # Windows PowerShell 5.1 writes UTF-16 for `>` redirects,
                    # producing NUL-riddled files no downstream tool can read
                    # (agent_bench s5: result.txt was UTF-16 "57"). Force
                    # UTF-8 for redirects and console output.
                    # python3/pip3 don't exist in a Windows venv Scripts dir —
                    # python3 resolves to the Microsoft Store alias stub.
                    # Shim both onto the venv interpreter.
                    ps_prelude = (
                        "$PSDefaultParameterValues['Out-File:Encoding']='utf8'; "
                        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
                        "function python3 { & python @args }; "
                        "function pip3 { & pip @args }; "
                    )
                    result = subprocess.run(
                        [ps_bin, "-NoProfile", "-NonInteractive", "-Command",
                         ps_prelude + ps_cmd],
                        capture_output=True, text=True, timeout=timeout,
                        cwd=cwd, env=_get_clean_env(),
                        encoding="utf-8", errors="replace",
                    )
                else:
                    # Fallback to cmd.exe
                    result = subprocess.run(
                        ps_cmd, shell=True,
                        capture_output=True, text=True, timeout=timeout,
                        cwd=cwd, env=_get_clean_env(),
                        encoding="utf-8", errors="replace",
                    )
        else:
            result = subprocess.run(
                command, shell=True,
                capture_output=True, text=True, timeout=timeout,
                cwd=cwd, env=_get_clean_env(),
                encoding="utf-8", errors="replace",
            )

        output = result.stdout
        if result.stderr:
            output += result.stderr
        return (result.returncode == 0), output.strip()

    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout}s"
    except FileNotFoundError:
        tool_name = command.split()[0]
        return False, f"Tool not found: {tool_name}. Is it installed and in PATH?"
    except Exception as e:
        return False, f"Execution error: {e}"


def run_python_snippet(code, timeout=30):
    """
    Execute a Python code snippet using the venv's interpreter.
    Single-line: python -c. Multi-line: temp file.
    Returns (success, output) tuple.

    Timeout is capped at MAX_PYTHON_TIMEOUT from the sandbox.
    """
    timeout = min(timeout, MAX_PYTHON_TIMEOUT)
    blocked = _check_dangerous(code)
    if blocked:
        return False, blocked

    try:
        lines = code.strip().split("\n")

        if len(lines) == 1:
            result = subprocess.run(
                [_PYTHON_BIN, "-c", code],
                capture_output=True, text=True, timeout=timeout,
                env=_get_clean_env(),
                encoding="utf-8", errors="replace",
            )
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                tmp_path = f.name
            try:
                result = subprocess.run(
                    [_PYTHON_BIN, tmp_path],
                    capture_output=True, text=True, timeout=timeout,
                    env=_get_clean_env(),
                    encoding="utf-8", errors="replace",
                )
            finally:
                os.unlink(tmp_path)

        output = result.stdout
        if result.stderr:
            output += result.stderr
        return (result.returncode == 0), output.strip()

    except subprocess.TimeoutExpired:
        return False, f"Python snippet timed out after {timeout}s"
    except Exception as e:
        return False, f"Python execution error: {e}"


def _ddgs_search_with_retry(query, max_results, retries=3):
    """Try DDGS library up to `retries` times with exponential backoff.
    Returns list of result dicts or None."""
    from duckduckgo_search import DDGS  # caller already checked importability

    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if results:
                return results
        except Exception as e:
            err = str(e).lower()
            # Known rate-limit indicators
            if "ratelimit" in err or "202" in err or "429" in err:
                pass  # expected, retry after sleep
            else:
                pass  # unknown error, still worth retrying

        # Exponential backoff with jitter
        if attempt < retries - 1:
            delay = (2 ** attempt) + random.uniform(0.5, 1.5)
            time.sleep(delay)

    return None


def _search_ddg_lite(query, max_results=8):
    """Fallback: scrape DuckDuckGo Lite HTML page using only stdlib.
    Returns list of {"title":..., "href":..., "body":...} dicts or None."""
    try:
        data = urlencode({"q": query}).encode("utf-8")
        req = Request(
            "https://lite.duckduckgo.com/lite/",
            data=data,
            headers={
                "User-Agent": _USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urlopen(req, timeout=10) as resp:
            if resp.status in (202, 403):
                return None  # rate limited
            body = resp.read(512_000)  # 512 KB max

        html_str = body.decode("utf-8", errors="replace")
        parser = _DDGLiteParser(max_results=max_results)
        parser.feed(html_str)

        if parser.results:
            return parser.results
        return None

    except Exception as e:
        _log.debug("DDG lite search failed: %s", e)
        return None


def run_web_search(query, max_results=8):
    """
    Run a web search with retry and fallback.

    Layer 0: Web Gateway (SearXNG via Docker proxy, sanitized).
    Layer 1: duckduckgo-search library with 3 retries + backoff.
    Layer 2: DuckDuckGo Lite HTML scrape (stdlib only).
    Layer 3: Informative error message.

    Results are cached in _last_search_results so the model can
    follow up with @web_read(N) to read a specific result page.
    """
    global _last_search_results

    # --- Layer 0: Web Gateway (preferred — sanitized, SearXNG-backed) ---
    if _check_gateway():
        ok, data = _gateway_post("/search", {"query": query, "max_results": max_results})
        if ok and data.get("results"):
            gw_results = data["results"]
            _last_search_results = [
                {"title": r.get("title", ""), "url": r.get("url", ""), "body": r.get("snippet", "")}
                for r in gw_results
            ]
            lines = []
            for i, r in enumerate(gw_results, 1):
                lines.append(f"[{i}] {r.get('title', 'No title')}")
                lines.append(f"    {r.get('url', '')}")
                snippet = r.get("snippet", "")
                if snippet:
                    lines.append(f"    {snippet[:400]}")
                lines.append("")
            lines.append("Use @web_read(N) to read the full page for result N.")
            return True, "\n".join(lines)

    # --- Layer 1: DDGS library with retry (fallback for non-Docker use) ---
    results = None
    try:
        from duckduckgo_search import DDGS  # noqa: F401 — import test
        results = _ddgs_search_with_retry(query, max_results)
    except ImportError:
        pass  # library not installed, skip to fallback

    # --- Layer 2: Lite HTML fallback ---
    if not results:
        results = _search_ddg_lite(query, max_results)

    # --- Layer 3: Both failed ---
    if not results:
        _last_search_results = []
        return True, (
            "No results found. DuckDuckGo may be rate-limiting requests.\n"
            "Try again in a minute, or try a different/simpler query."
        )

    # --- Format results (same as before) ---
    _last_search_results = [
        {"title": r.get("title", ""), "url": r.get("href", ""), "body": r.get("body", "")}
        for r in results
    ]

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.get('title', 'No title')}")
        lines.append(f"    {r.get('href', '')}")
        body = r.get("body", "")
        if body:
            lines.append(f"    {body[:400]}")
        lines.append("")

    lines.append("Use @web_read(N) to read the full page for result N.")
    return True, "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Chunked file reader
# ─────────────────────────────────────────────────────────────────────────────

_READ_CHUNK_SIZE = 3000   # chars per chunk (legacy default — use _get_read_chunk_size())


def _get_read_chunk_size():
    """Get chunk size from the active context profile."""
    from core.config import get_context_profile
    return get_context_profile().read_chunk_size
_MAX_FILE_SIZE = 1_048_576  # 1 MB hard cap


def run_read_file(content):
    """
    Read a file in chunks that fit the model's context.

    content format: "filepath|chunk_num"  (chunk_num defaults to 1)
    If the filepath is a URL, redirect to run_web_read() automatically.
    Returns (success, output) tuple.
    """
    parts = content.rsplit("|", 1)
    filepath = parts[0].strip()
    chunk_num = int(parts[1]) if len(parts) > 1 else 1

    # Auto-detect URLs and redirect to web_read
    if filepath.startswith(("http://", "https://", "www.")):
        return run_web_read(filepath)

    # Resolve relative paths from cwd
    if not os.path.isabs(filepath):
        filepath = os.path.join(os.getcwd(), filepath)

    if not os.path.isfile(filepath):
        return False, f"File not found: {filepath}"

    # Size check
    file_size = os.path.getsize(filepath)
    if file_size > _MAX_FILE_SIZE:
        return False, (
            f"File too large: {file_size / 1_048_576:.1f} MB "
            f"(max {_MAX_FILE_SIZE // 1_048_576} MB). "
            "Use a shell command to read specific sections instead."
        )

    # Binary detection (check first 512 bytes for null bytes)
    try:
        with open(filepath, "rb") as f:
            sample = f.read(512)
        if b"\x00" in sample:
            return False, (
                f"Binary file detected: {os.path.basename(filepath)}. "
                "Cannot display binary content. Use a shell command to inspect it."
            )
    except Exception as e:
        return False, f"Cannot read file: {e}"

    # Read with encoding fallback
    text = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(filepath, "r", encoding=encoding) as f:
                text = f.read()
            break
        except (UnicodeDecodeError, ValueError):
            continue

    if text is None:
        return False, "Cannot decode file — unsupported encoding."

    lines = text.split("\n")
    total_lines = len(lines)

    # Smart skeleton for large Python files (chunk 1 only)
    if filepath.endswith(".py") and total_lines > 200 and chunk_num == 1:
        skeleton = _python_file_skeleton(filepath, text, lines)
        if skeleton is not None:
            return True, skeleton

    # Build chunks on line boundaries
    chunk_size = _get_read_chunk_size()
    chunks = []
    current_chunk = []
    current_size = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for the newline
        # If a single line exceeds chunk size, force-split it
        if line_len > chunk_size and not current_chunk:
            for i in range(0, len(line), chunk_size):
                chunks.append(line[i:i + chunk_size])
            continue

        if current_size + line_len > chunk_size and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_size = 0

        current_chunk.append(line)
        current_size += line_len

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    total_chunks = len(chunks)

    if chunk_num < 1 or chunk_num > total_chunks:
        return False, f"Invalid chunk {chunk_num} — file has {total_chunks} chunk(s)."

    chunk_text = chunks[chunk_num - 1]

    # Calculate line range for this chunk
    lines_before = sum(c.count("\n") + 1 for c in chunks[:chunk_num - 1])
    chunk_lines = chunk_text.count("\n") + 1
    start_line = lines_before + 1
    end_line = lines_before + chunk_lines

    # Build output
    fname = os.path.basename(filepath)
    header = f"File: {fname} (chunk {chunk_num}/{total_chunks}, lines {start_line}-{end_line} of {total_lines})"

    result = [header]

    if chunk_num < total_chunks:
        result.append(f'[Next: @read_file("{filepath}", chunk={chunk_num + 1})]')

    result.extend(["---", chunk_text, "---"])

    return True, "\n".join(result)


# ─────────────────────────────────────────────────────────────────────────────
# Smart file reading — skeleton view + function-level drill-down
# ─────────────────────────────────────────────────────────────────────────────

def _python_file_skeleton(filepath, text, lines):
    """Generate a skeleton view for a large Python file.

    Returns a structured overview (imports + symbol listing) instead of raw
    source.  Returns None if AST extraction fails so the caller can fall back
    to chunked reading.
    """
    try:
        from tools.codebase_tools import _extract_python_symbols
    except ImportError:
        return None

    rel_path = os.path.relpath(filepath)
    try:
        symbols = _extract_python_symbols(filepath, rel_path)
    except Exception:
        return None

    total_lines = len(lines)

    # Header: imports and module docstring (up to first def/class, max 40 lines)
    header_end = 0
    for i, line in enumerate(lines[:50]):
        stripped = line.lstrip()
        if stripped.startswith(("def ", "class ", "async def ")):
            header_end = i
            break
    else:
        header_end = min(40, total_lines)

    header = "\n".join(lines[:header_end]).rstrip()

    # Symbol listing grouped by kind
    classes = [s for s in symbols if s.kind == "class"]
    functions = [s for s in symbols if s.kind == "function"]
    methods = [s for s in symbols if s.kind == "method"]

    sym_lines = []
    if classes:
        sym_lines.append("CLASSES:")
        for s in classes:
            sym_lines.append(f"  L{s.line}: {s.signature}")
    if functions:
        sym_lines.append("FUNCTIONS:")
        for s in functions:
            sym_lines.append(f"  L{s.line}: {s.signature}")
    if methods:
        sym_lines.append(f"METHODS ({len(methods)}):")
        for s in methods:
            sym_lines.append(f"  L{s.line}: {s.parent}.{s.name}()")

    chunk_size = _get_read_chunk_size()
    total_chunks = max(1, len(text) // chunk_size + 1)

    result = [
        f"File: {os.path.basename(filepath)} ({total_lines} lines) — SKELETON VIEW",
        f"Large file. Showing structure + imports. "
        f'Use @read_function("{rel_path}", "name") to read a specific function/class.',
        "---",
        "IMPORTS + HEADER:",
        header,
        "---",
        f"SYMBOLS ({len(symbols)} definitions):",
        "\n".join(sym_lines) if sym_lines else "  (no symbols found)",
        "---",
        f'Drill into a function: @read_function("{rel_path}", "function_name")',
        f'Read raw chunks: @read_file("{rel_path}|N") (1-{total_chunks})',
    ]
    return "\n".join(result)


def run_read_function(content):
    """Read a specific function/class from a Python file using AST line ranges.

    content format: "filepath|function_name"
    Returns (success, output) tuple.
    """
    import ast as _ast

    parts = content.split("|", 1)
    if len(parts) != 2:
        return False, 'Usage: @read_function("path/to/file.py", "function_name")'

    filepath, name = parts[0].strip(), parts[1].strip()

    if not os.path.isabs(filepath):
        filepath = os.path.join(os.getcwd(), filepath)

    if not os.path.isfile(filepath):
        return False, f"File not found: {filepath}"

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except OSError as e:
        return False, f"Cannot read file: {e}"

    file_lines = source.split("\n")

    try:
        tree = _ast.parse(source, filename=filepath)
    except SyntaxError as e:
        return False, f"Syntax error in {os.path.basename(filepath)}: {e}"

    # Find the target node — check top-level and class methods
    target = None
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            if node.name == name:
                target = node
                break

    if target is None:
        # Try class.method format: "ClassName.method_name"
        if "." in name:
            cls_name, method_name = name.rsplit(".", 1)
            for node in _ast.walk(tree):
                if isinstance(node, _ast.ClassDef) and node.name == cls_name:
                    for child in _ast.iter_child_nodes(node):
                        if isinstance(child, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                            if child.name == method_name:
                                target = child
                                break

    if target is None:
        return False, f"Symbol '{name}' not found in {os.path.basename(filepath)}"

    start = target.lineno - 1   # 0-indexed
    end = target.end_lineno     # end_lineno is 1-indexed, correct for slicing

    func_lines = file_lines[start:end]
    func_text = "\n".join(func_lines)

    header = (
        f"Function: {name} "
        f"(lines {start + 1}-{end} of {len(file_lines)} in {os.path.basename(filepath)})"
    )

    return True, f"{header}\n---\n{func_text}\n---"


# ─────────────────────────────────────────────────────────────────────────────
# Web page reader / condenser
# ─────────────────────────────────────────────────────────────────────────────

_WEB_READ_MAX_BODY = 5_242_880   # 5 MB max download (PDFs can be larger)
_WEB_READ_TIMEOUT = 20           # seconds (PDFs need more time)
# chars of clean text to return; 3500 was the 8K-ctx-era value — at modern
# ctx windows a fuller page beats a lossy cut.
_WEB_READ_TRUNCATE = int(os.environ.get("ARTIFEX_WEB_READ_TRUNCATE", "12000"))

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

# Tags that contain boilerplate, not article content
_STRIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "noscript", "iframe"}


def _is_pdf_url(url):
    """Check if a URL points to a PDF file."""
    # Strip query params and fragments for extension check
    path = url.split("?")[0].split("#")[0].lower()
    return path.endswith(".pdf")


def _extract_pdf_text(pdf_bytes, max_pages=30):
    """
    Extract text from PDF bytes using pypdf (preferred) or PyPDF2 (fallback).
    Returns extracted text or None if no PDF library is available.
    """
    # Try pypdf first (modern, actively maintained)
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for i, page in enumerate(reader.pages[:max_pages]):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"--- Page {i + 1} ---\n{text.strip()}")
        if pages:
            result = "\n\n".join(pages)
            if len(reader.pages) > max_pages:
                result += f"\n\n[...showing first {max_pages} of {len(reader.pages)} pages]"
            return result
        return "(PDF contained no extractable text — may be scanned/image-based.)"
    except ImportError:
        pass
    except Exception as e:
        _log.debug("pypdf extraction failed: %s", e)

    # Fallback: PyPDF2 (older but commonly installed)
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for i, page in enumerate(reader.pages[:max_pages]):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"--- Page {i + 1} ---\n{text.strip()}")
        if pages:
            result = "\n\n".join(pages)
            if len(reader.pages) > max_pages:
                result += f"\n\n[...showing first {max_pages} of {len(reader.pages)} pages]"
            return result
        return "(PDF contained no extractable text — may be scanned/image-based.)"
    except ImportError:
        pass
    except Exception as e:
        _log.debug("PyPDF2 extraction failed: %s", e)

    return None  # No PDF library available


def _extract_page_text_lxml(html_bytes, encoding="utf-8"):
    """Use lxml to parse HTML and extract clean text."""
    try:
        doc = lxml.html.fromstring(html_bytes)
    except Exception as e:
        _log.debug("lxml HTML parse failed, falling back to raw decode: %s", e)
        return html_bytes.decode(encoding, errors="replace")

    # Remove boilerplate tags
    for tag in _STRIP_TAGS:
        for el in doc.iter(tag):
            el.getparent().remove(el)

    text = doc.text_content()
    # Collapse runs of whitespace / blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


class _SimpleHTMLStripper(HTMLParser):
    """Stdlib fallback: strips tags and extracts text."""

    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip = False
        self._skip_tags = _STRIP_TAGS

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip = True

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self):
        text = " ".join(self._parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()


class _DDGLiteParser(HTMLParser):
    """Parse DuckDuckGo Lite HTML results into structured dicts."""

    def __init__(self, max_results=8):
        super().__init__()
        self.results = []
        self._max = max_results
        self._state = "IDLE"       # IDLE -> TITLE -> SNIPPET_WAIT -> SNIPPET
        self._current = {}
        self._text_buf = []

    def handle_starttag(self, tag, attrs):
        if len(self.results) >= self._max:
            return
        attrs_dict = dict(attrs)

        if tag == "a" and self._state == "IDLE":
            cls = attrs_dict.get("class", "")
            href = attrs_dict.get("href", "")
            # DDG Lite uses class="result-link" for result links
            if "result-link" in cls or (href.startswith("http") and "duckduckgo" not in href):
                self._state = "TITLE"
                self._current = {"href": href}
                self._text_buf = []

        elif tag == "td" and self._state == "SNIPPET_WAIT":
            cls = attrs_dict.get("class", "")
            if "result-snippet" in cls:
                self._state = "SNIPPET"
                self._text_buf = []

    def handle_endtag(self, tag):
        if len(self.results) >= self._max:
            return

        if tag == "a" and self._state == "TITLE":
            self._current["title"] = " ".join(self._text_buf).strip()
            self._state = "SNIPPET_WAIT"
            self._text_buf = []

        elif tag == "td" and self._state == "SNIPPET":
            self._current["body"] = " ".join(self._text_buf).strip()
            if self._current.get("href"):
                self.results.append(self._current)
            self._current = {}
            self._state = "IDLE"

    def handle_data(self, data):
        if self._state in ("TITLE", "SNIPPET"):
            self._text_buf.append(data.strip())


def _extract_page_text_stdlib(html_bytes, encoding="utf-8"):
    """Stdlib fallback for HTML → text."""
    html_str = html_bytes.decode(encoding, errors="replace")
    parser = _SimpleHTMLStripper()
    try:
        parser.feed(html_str)
    except Exception:
        return html_str
    return parser.get_text()


def run_web_read(ref):
    """
    Fetch a web page and extract its main text content.

    ref: a URL string, or a number N referencing the Nth result from the last @search().
    Returns (success, output) tuple.

    When the web gateway is available, content is fetched and sanitized through
    the gateway (trafilatura extraction, prompt injection detection).
    Falls back to direct fetch for non-Docker use.
    """
    global _last_search_results

    # Resolve reference — number means search result index
    url = None
    title = None
    if ref.isdigit():
        idx = int(ref) - 1
        if idx < 0 or idx >= len(_last_search_results):
            if not _last_search_results:
                return False, "No search results cached. Run @search(\"query\") first."
            return False, f"Invalid result number {ref}. Last search had {len(_last_search_results)} results."
        url = _last_search_results[idx]["url"]
        title = _last_search_results[idx]["title"]
    else:
        url = ref.strip()

    if not url:
        return False, "No URL to fetch."

    # Ensure URL has a scheme
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # --- Gateway path (preferred — sanitized + prompt injection detection) ---
    if _check_gateway():
        ok, data = _gateway_post("/fetch", {"url": url})
        if ok:
            text = data.get("text", "")
            gw_title = data.get("title") or title or url
            warnings = data.get("warnings", [])

            if not text:
                return True, "(Page returned no readable text content.)"

            header = f"Source: {gw_title}\nURL: {data.get('url', url)}\n---"

            if warnings:
                header += f"\n[Security warnings: {'; '.join(warnings)}]"

            if len(text) > _WEB_READ_TRUNCATE:
                text = text[:_WEB_READ_TRUNCATE]
                return True, f"{header}\n[Truncated — content is partial.]\n{text}"

            return True, f"{header}\n{text}"
        else:
            # Gateway returned an error — include it
            return False, f"Web gateway error: {data}"

    # --- Direct fallback (non-Docker use) ---

    # Detect if this is a PDF URL
    is_pdf = _is_pdf_url(url)

    # Fetch
    try:
        req = Request(url, headers={"User-Agent": _USER_AGENT})
        with urlopen(req, timeout=_WEB_READ_TIMEOUT) as resp:
            # Check content length if available
            cl = resp.headers.get("Content-Length")
            if cl and int(cl) > _WEB_READ_MAX_BODY:
                return False, f"Page too large ({int(cl) // 1024} KB). Cannot fetch."

            body = resp.read(_WEB_READ_MAX_BODY)

            # Detect content type
            ct = resp.headers.get("Content-Type", "")
            if "application/pdf" in ct:
                is_pdf = True

            # Detect encoding from Content-Type header
            encoding = "utf-8"
            if "charset=" in ct:
                encoding = ct.split("charset=")[-1].split(";")[0].strip()

    except HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason} — {url}"
    except URLError as e:
        return False, f"Network error: {e.reason}"
    except TimeoutError:
        return False, f"Request timed out after {_WEB_READ_TIMEOUT}s — {url}"
    except Exception as e:
        return False, f"Fetch error: {e}"

    # --- PDF handling ---
    if is_pdf:
        header = f"Source: {title or url}\nURL: {url}\nType: PDF\n---"
        text = _extract_pdf_text(body)
        if text is None:
            return False, (
                f"{header}\n"
                "Cannot extract PDF text — no PDF library installed.\n"
                "Install one with: pip install pypdf\n"
                "Then try @web_read again."
            )

        if len(text) > _WEB_READ_TRUNCATE:
            text = text[:_WEB_READ_TRUNCATE]
            return True, f"{header}\n[PDF truncated. Use @download to save full PDF locally.]\n{text}"

        return True, f"{header}\n{text}"

    # --- HTML handling ---
    if _HAS_LXML:
        text = _extract_page_text_lxml(body, encoding)
    else:
        text = _extract_page_text_stdlib(body, encoding)

    if not text:
        return True, "(Page returned no readable text content.)"

    # Build output with source info
    header = f"Source: {title or url}\nURL: {url}\n---"

    if len(text) > _WEB_READ_TRUNCATE:
        text = text[:_WEB_READ_TRUNCATE]
        return True, f"{header}\n[Page truncated. Content is partial.]\n{text}"

    return True, f"{header}\n{text}"


def run_download(content):
    """
    Download a file from a URL.

    content format: "url" or "url|filename"
    Returns (success, output) tuple.

    When the web gateway is available, files are downloaded to a quarantined
    tmpfs directory (RAM-backed, auto-deleted on session end). Falls back to
    direct download to ./output/ for non-Docker use.
    """
    parts = content.split("|", 1)
    url = parts[0].strip()
    custom_name = parts[1].strip() if len(parts) > 1 else None

    # Ensure URL has a scheme
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # --- Gateway path (preferred — quarantined in tmpfs) ---
    if _check_gateway():
        payload = {"url": url}
        if custom_name:
            payload["filename"] = custom_name
        ok, data = _gateway_post("/download", payload)
        if ok:
            qid = data.get("quarantine_id", "unknown")
            fname = data.get("filename", "unknown")
            size = data.get("size_bytes", 0)
            if size > 1024 * 1024:
                size_str = f"{size / (1024*1024):.1f} MB"
            else:
                size_str = f"{size // 1024} KB"
            return True, (
                f"Downloaded: {fname} ({size_str})\n"
                f"Quarantine ID: {qid}\n"
                f"Status: Held in secure quarantine (RAM-backed, auto-deleted on session end).\n"
                f"Content type: {data.get('content_type', 'unknown')}"
            )
        else:
            return False, f"Download blocked by web gateway: {data}"

    # --- Direct fallback (non-Docker use) ---

    # Determine filename
    if custom_name:
        filename = os.path.basename(custom_name)  # strip path traversal
    else:
        # Extract from URL path
        from urllib.parse import urlparse
        path = urlparse(url).path
        filename = os.path.basename(path) or "downloaded_file"

    # Sanitize filename: remove dangerous chars, .., and cap length
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = filename.replace("..", "")
    filename = filename.strip(". ")  # no leading/trailing dots or spaces
    if not filename:
        filename = "downloaded_file"
    if len(filename) > 200:
        ext = os.path.splitext(filename)[1][:20]
        filename = filename[:200 - len(ext)] + ext
    dest = os.path.join(os.getcwd(), filename)

    # Download
    try:
        req = Request(url, headers={"User-Agent": _USER_AGENT})
        with urlopen(req, timeout=30) as resp:
            cl = resp.headers.get("Content-Length")
            if cl and int(cl) > 50_000_000:  # 50 MB hard limit
                return False, f"File too large ({int(cl) // 1_048_576} MB). Max 50 MB."

            with open(dest, "wb") as f:
                total = 0
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > 50_000_000:
                        f.close()
                        os.unlink(dest)
                        return False, "Download exceeded 50 MB limit. Aborted."
                    f.write(chunk)

        size_kb = os.path.getsize(dest) / 1024
        if size_kb > 1024:
            size_str = f"{size_kb / 1024:.1f} MB"
        else:
            size_str = f"{size_kb:.0f} KB"

        return True, (
            f"Downloaded: {filename} ({size_str})\n"
            f"Saved to: {dest}\n"
            f"Use @read_file(\"{filename}\") to read its contents."
        )

    except HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason} — {url}"
    except URLError as e:
        return False, f"Network error: {e.reason}"
    except TimeoutError:
        return False, f"Download timed out after 30s — {url}"
    except Exception as e:
        return False, f"Download error: {e}"


def run_glob(content):
    """
    Find files matching a glob pattern.

    content format: "pattern" or "pattern|base_dir" or "pattern|base_dir|+all"
    The +all flag disables venv/node_modules filtering (for troubleshooting).
    Returns (success, output) tuple.
    """
    import glob as _glob

    parts = content.split("|")
    pattern = parts[0].strip()
    base_dir = parts[1].strip() if len(parts) > 1 and parts[1].strip() else os.getcwd()
    include_all = any(p.strip().lower() == "+all" for p in parts[2:])

    if not os.path.isabs(base_dir):
        base_dir = os.path.join(os.getcwd(), base_dir)

    # Directories excluded by default (override with +all flag)
    _GLOB_SKIP = {"venv", ".venv", "env", "node_modules", "__pycache__",
                   ".git", ".mypy_cache", ".pytest_cache", ".tox",
                   "dist", "build", "egg-info", ".idea", ".vs", ".vscode"}

    full_pattern = os.path.join(base_dir, pattern)
    raw_matches = sorted(_glob.glob(full_pattern, recursive=True))

    if include_all:
        matches = raw_matches
    else:
        # Filter out matches inside skipped directories
        matches = []
        for m in raw_matches:
            rel = os.path.relpath(m, base_dir)
            seg = rel.replace("\\", "/").split("/")
            if not any(p in _GLOB_SKIP for p in seg):
                matches.append(m)

    if not matches:
        skipped = len(raw_matches) - len(matches) if raw_matches else 0
        msg = f"No files matched: {pattern}"
        if skipped:
            msg += f" ({skipped} matches in venv/node_modules/etc excluded — use +all flag to include)"
        return True, msg

    lines = [f"Found {len(matches)} match(es) for: {pattern}"]
    for m in matches[:100]:
        try:
            rel = os.path.relpath(m, base_dir)
            if os.path.isfile(m):
                size = os.path.getsize(m)
                size_str = f"{size:,}B" if size < 1024 else f"{size//1024}KB"
                lines.append(f"  {rel} ({size_str})")
            else:
                lines.append(f"  {rel}/")
        except ValueError:
            lines.append(f"  {m}")

    if len(matches) > 100:
        lines.append(f"  ... and {len(matches) - 100} more")

    return True, "\n".join(lines)


def run_grep(content):
    """
    Search file contents with a regex pattern.

    content format: "pattern|path|flags"
    path can be a file, directory, or glob pattern.
    flags: "i" for case-insensitive (optional).
    Returns (success, output) tuple.
    """
    import glob as _glob

    parts = content.split("|", 2)
    pattern = parts[0].strip()
    path = parts[1].strip() if len(parts) > 1 else os.getcwd()
    flags_str = parts[2].strip().lower() if len(parts) > 2 else ""

    re_flags = re.IGNORECASE if "i" in flags_str else 0

    try:
        regex = re.compile(pattern, re_flags)
    except re.error as e:
        return False, f"Invalid regex: {e}"

    # Resolve path
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)

    # Collect files to search
    _TEXT_EXTS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".c", ".cpp", ".h",
        ".cs", ".rb", ".php", ".sh", ".bat", ".ps1", ".yaml", ".yml", ".json", ".toml",
        ".ini", ".cfg", ".conf", ".txt", ".md", ".rst", ".html", ".css", ".xml", ".sql",
    }
    _SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache"}

    files = []
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        for root, dirs, fnames in os.walk(path):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            for fname in fnames:
                if os.path.splitext(fname)[1].lower() in _TEXT_EXTS:
                    files.append(os.path.join(root, fname))
    else:
        # Treat as glob
        files = _glob.glob(path, recursive=True)

    results = []
    files_searched = 0
    for filepath in files[:200]:
        if not os.path.isfile(filepath):
            continue
        files_searched += 1
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    if regex.search(line):
                        try:
                            rel = os.path.relpath(filepath)
                        except ValueError:
                            rel = filepath
                        results.append(f"{rel}:{lineno}: {line.rstrip()}")
                        if len(results) >= 100:
                            break
        except Exception as e:
            _log.debug("File grep skipped %s: %s", filepath, e)
        if len(results) >= 100:
            results.append("... (limit reached — refine your pattern)")
            break

    if not results:
        return True, f"No matches for '{pattern}' in {files_searched} file(s) searched."

    header = f"Found {len(results)} match(es) for '{pattern}':\n"
    return True, header + "\n".join(results)


def run_edit_file(content):
    """
    Make a targeted string replacement in a file.

    content format (NUL-delimited): "path\x00old_string\x00new_string"
    Fails if old_string is not found or appears more than once.
    Returns (success, output) tuple.
    """
    parts = content.split("\x00", 2)
    if len(parts) != 3:
        return False, "Edit format error."

    path, old_str, new_str = parts
    path = path.strip()

    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)

    if not os.path.isfile(path):
        # Create-file form: empty OLD + nonexistent target. Models reach for
        # this constantly (observed across Qwen3.6 bench runs); it is safe
        # and unambiguous, so support it instead of erroring.
        if not old_str.strip():
            if path.endswith(".py"):
                try:
                    compile(new_str, path, "exec")
                except SyntaxError as e:
                    return False, f"New file has a Python syntax error: {e}"
            try:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(new_str if new_str.endswith("\n") or not new_str
                            else new_str + "\n")
            except OSError as e:
                return False, f"Cannot create {path}: {e}"
            n_lines = new_str.count("\n") + 1
            return True, f"Created {os.path.basename(path)} ({n_lines} lines)"
        return False, f"File not found: {path}"

    # Read with encoding fallback
    original = None
    used_encoding = "utf-8"
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                original = f.read()
            used_encoding = "utf-8" if enc == "utf-8-sig" else enc
            break
        except (UnicodeDecodeError, ValueError):
            continue

    if original is None:
        return False, "Cannot decode file — unsupported encoding."

    if not old_str.strip():
        # "".count() is len+1, which used to surface as a baffling
        # "appears N times" error. Teach instead.
        return False, (
            "OLD is empty — provide the exact existing text to replace. "
            "To create a new file, use a python code block instead."
        )

    count = original.count(old_str)
    if count == 0:
        # Give the model a hint about what was found nearby
        snippet = old_str.strip()[:60]
        return False, (
            f"String not found in {os.path.basename(path)}.\n"
            f"Looking for: {snippet!r}\n"
            "Make sure the old_string matches exactly (whitespace, indentation)."
        )
    if count > 1:
        return False, (
            f"Ambiguous: '{old_str.strip()[:60]}' appears {count} times in "
            f"{os.path.basename(path)}. Add more surrounding context to make it unique."
        )

    new_content = original.replace(old_str, new_str, 1)

    # Validate Python syntax before applying (catches model errors)
    if path.endswith(".py"):
        try:
            compile(new_content, path, "exec")
        except SyntaxError as e:
            return False, (
                f"REJECTED — edit would create syntax error in {os.path.basename(path)}:\n"
                f"  Line {e.lineno}: {e.msg}\n"
                "Fix the NEW text and try again."
            )

    with open(path, "w", encoding=used_encoding) as f:
        f.write(new_content)

    fname = os.path.basename(path)
    delta = len(new_str) - len(old_str)
    sign = "+" if delta >= 0 else ""
    return True, f"Edited {fname}: {sign}{delta} chars. Change applied successfully."


# ─── Git commit-per-edit (P3-T15) ──────────────────────────────────────────

def _find_git_root(path: str) -> str | None:
    """Walk up from path to find the .git directory. Returns repo root or None."""
    d = os.path.abspath(path)
    for _ in range(50):
        if os.path.isdir(os.path.join(d, ".git")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def git_commit_edit(file_path: str, summary: str) -> tuple[bool, str]:
    """Stage and commit a single file edit. Returns (success, message).

    Only commits if the file is inside a git repo and has changes.
    Commit message includes [agent] prefix for easy filtering.

    The commit is pathspec-limited to this one file. A bare `git commit`
    commits everything already in the index, so an agent edit inside a repo
    that had unrelated staged work (the API's default workspace root lives
    under the repo's output/) would sweep that work into an [agent] commit —
    and git_revert_last would then revert it too.
    """
    repo = _find_git_root(file_path)
    if not repo:
        return False, "Not inside a git repository."
    abs_path = os.path.abspath(file_path)
    try:
        subprocess.run(
            ["git", "add", "--", abs_path],
            cwd=repo, capture_output=True, text=True, timeout=10,
        )
        result = subprocess.run(
            ["git", "commit", "-m", f"[agent] {summary}", "--", abs_path],
            cwd=repo, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            short_sha = result.stdout.strip().split()[1] if result.stdout else "?"
            return True, f"Committed: {short_sha}"
        return False, result.stderr.strip() or "git commit failed"
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return False, str(e)


def git_revert_last(file_path: str) -> tuple[bool, str]:
    """Revert the last commit if it was an agent edit. Safety check: only
    reverts commits with [agent] prefix in the message."""
    repo = _find_git_root(file_path)
    if not repo:
        return False, "Not inside a git repository."
    try:
        log = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            cwd=repo, capture_output=True, text=True, timeout=10,
        )
        msg = log.stdout.strip()
        if not msg.startswith("[agent]"):
            return False, f"Last commit is not an agent edit: {msg!r}"
        result = subprocess.run(
            ["git", "revert", "--no-edit", "HEAD"],
            cwd=repo, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return True, "Reverted last agent commit."
        return False, result.stderr.strip() or "git revert failed"
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return False, str(e)


# ─── Agent loop config (P3-T16 / P3-T17) ──────────────────────────────────

MAX_AGENT_ROUNDS = int(os.environ.get("ARTIFEX_MAX_AGENT_ROUNDS", "10"))
AGENT_KEY = os.environ.get("ARTIFEX_AGENT_KEY", "")

# Explicit autonomous-completion marker: a line that is exactly @done(...) or
# @finish(...) (message optional). Anchored to a line so prose mentioning the
# marker doesn't trip it.
_DONE_RE = re.compile(r'(?mi)^\s*@(?:done|finish)\s*(?:\((?P<msg>.*?)\))?\s*$')


def detect_done(response):
    """Return the completion summary if the response declares it is done.

    Returns the message string ("" if the marker carried none) when an @done /
    @finish line is present, or None when absent — so callers can branch on
    `detect_done(r) is not None`.
    """
    if not response:
        return None
    m = _DONE_RE.search(response)
    if not m:
        return None
    return (m.group("msg") or "").strip().strip("'\"")


def agent_auto_exec_enabled() -> bool:
    """True if auto-execution mode is unlocked (ARTIFEX_AGENT_KEY is set)."""
    return bool(AGENT_KEY)


def run_sysinfo() -> tuple[bool, str]:
    """Machine snapshot: OS, CPU, RAM, GPUs, disks. Read-only, no shell.

    Exists because "what are this machine's specs" is a common ask that
    otherwise costs several risk-budgeted shell rounds and tempts the model
    into tools the platform may not ship (wmic). Sources: platform/psutil,
    plus the Windows registry for the CPU name and per-GPU VRAM —
    HardwareInformation.qwMemorySize is the accurate figure; WMI's
    AdapterRAM is a 32-bit field that caps at 4 GB.
    """
    import platform
    lines = [f"OS: {platform.system()} {platform.release()} "
             f"build {platform.version()} ({platform.machine()})"]

    cpu = platform.processor() or "unknown"
    if IS_WINDOWS:
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        import psutil
        lines.append(f"CPU: {cpu} — {psutil.cpu_count(logical=False)} cores / "
                     f"{psutil.cpu_count()} threads")
        vm = psutil.virtual_memory()
        lines.append(f"RAM: {vm.total / 2**30:.1f} GB total, "
                     f"{vm.available / 2**30:.1f} GB available")
    except ImportError:
        lines.append(f"CPU: {cpu}")

    if IS_WINDOWS:
        try:
            import winreg
            base = (r"SYSTEM\CurrentControlSet\Control\Class"
                    r"\{4d36e968-e325-11ce-bfc1-08002be10318}")
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as cls:
                for i in range(32):
                    try:
                        sub = winreg.EnumKey(cls, i)
                    except OSError:
                        break
                    if not sub.isdigit():
                        continue
                    try:
                        with winreg.OpenKey(cls, sub) as k:
                            name = winreg.QueryValueEx(k, "DriverDesc")[0]
                            try:
                                raw = winreg.QueryValueEx(
                                    k, "HardwareInformation.qwMemorySize")[0]
                                vram = int.from_bytes(raw, "little") \
                                    if isinstance(raw, bytes) else int(raw)
                                lines.append(f"GPU: {name} — "
                                             f"{vram / 2**30:.0f} GB VRAM")
                            except OSError:
                                lines.append(f"GPU: {name}")
                    except OSError:
                        continue
        except OSError:
            pass

    try:
        import psutil
        for part in psutil.disk_partitions(all=False):
            try:
                u = psutil.disk_usage(part.mountpoint)
            except OSError:
                continue
            lines.append(f"Disk {part.mountpoint}: {u.free / 2**30:.1f} GB free "
                         f"of {u.total / 2**30:.1f} GB")
    except ImportError:
        pass

    return True, "\n".join(lines)


def run_agent_action(action, confirm_cb=None, policy_check=True):
    """
    Dispatch an AgentAction to the appropriate executor.
    Returns (success, output) tuple.

    Policy enforcement happens HERE, not only at call sites. The sandbox
    policy engine used to be consulted solely by callers that opted in
    (the AgentRunner loop did; the voice pipeline and the Qt action
    worker did not), which left those surfaces running model-authored
    actions with no injection screening, blocklist, or egress checks.
    Centralizing the gate means every current and future caller is
    screened by default.

    Args:
        action: AgentAction to execute.
        confirm_cb: callable(action, decision) -> bool for actions the
            policy marks requires_confirmation. A surface where the human
            already approved the action (e.g. the Qt click-to-run panel)
            passes a callback returning True. When None, confirmation-
            required actions are refused rather than silently executed.
        policy_check: set False ONLY by callers that already ran
            check_policy and obtained approval for this exact action
            (the AgentRunner loop) — re-checking would double-count the
            audit-log and circuit-breaker hooks.
    """
    if policy_check:
        from core.sandbox import check_policy
        decision = check_policy(action.type, action.content)
        if not decision.allowed:
            return False, (
                f"[BLOCKED by sandbox policy — {decision.risk_level.name}] "
                f"{decision.reason}"
            )
        if decision.requires_confirmation:
            if confirm_cb is None:
                return False, (
                    f"[NOT RUN — requires confirmation ({decision.risk_level.name})] "
                    f"This surface has no confirmation flow; action refused: "
                    f"{action.display if hasattr(action, 'display') else action.type}"
                )
            if not confirm_cb(action, decision):
                return False, "[NOT RUN — confirmation declined]"

    if action.type == "sysinfo":
        return run_sysinfo()
    elif action.type == "shell":
        return run_shell_command(action.content, cwd=os.getcwd())
    elif action.type == "python":
        return run_python_snippet(action.content)
    elif action.type == "search":
        return run_web_search(action.content)
    elif action.type == "read_file":
        return run_read_file(action.content)
    elif action.type == "web_read":
        return run_web_read(action.content)
    elif action.type == "download":
        return run_download(action.content)
    elif action.type == "glob":
        return run_glob(action.content)
    elif action.type == "grep":
        return run_grep(action.content)
    elif action.type == "edit_file":
        return run_edit_file(action.content)
    elif action.type == "read_function":
        return run_read_function(action.content)
    elif action.type in ("find_symbol", "find_references", "trace_imports", "architecture"):
        from tools.codebase_tools import (
            run_find_symbol, run_find_references, run_trace_imports, run_architecture,
        )
        dispatch = {
            "find_symbol": run_find_symbol,
            "find_references": run_find_references,
            "trace_imports": run_trace_imports,
            "architecture": run_architecture,
        }
        return dispatch[action.type](action.content)
    else:
        return False, f"Unknown action type: {action.type}"


def get_assistant_tools_prompt():
    """
    Build environment description for the ASSISTANT agent prompt.
    Reuses sense_system() from core/hardware.py.
    """
    sys_info = sense_system()
    lines = []

    lines.append(f"OS: {sys_info.get('os_display', sys_info['os'])} ({sys_info['arch']})")

    # Report the ACTUAL execution shell, not just what's installed
    if IS_WINDOWS:
        if shutil.which("pwsh"):
            lines.append("Command shell: PowerShell Core (pwsh) — all commands run via PowerShell")
        elif shutil.which("powershell"):
            lines.append("Command shell: PowerShell — all commands run via PowerShell")
        else:
            lines.append("Command shell: cmd.exe")
    else:
        lines.append(f"Command shell: {sys_info.get('shell', '/bin/sh')}")

    if sys_info.get("has_wsl"):
        lines.append("WSL: Available")
    if sys_info.get("has_bash") and IS_WINDOWS:
        lines.append("Bash: Available (Git Bash)")

    # Runtimes
    if sys_info.get("runtimes"):
        rt = ", ".join(sorted(sys_info["runtimes"].keys()))
        lines.append(f"Runtimes: {rt}")

    # Useful utilities
    if sys_info.get("utilities"):
        utils = ", ".join(sorted(sys_info["utilities"].keys()))
        lines.append(f"Utilities: {utils}")

    return "\n".join(lines)
