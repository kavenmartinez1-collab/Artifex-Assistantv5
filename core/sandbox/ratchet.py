"""
Artifex Assistant V5 — Self-modification ratchet (P6-T31).

Prevents the agent from modifying its own sandbox, policy, or safety
infrastructure. Files in the sandbox package and critical config are
protected from agent edits.

The ratchet is one-way: once installed, the agent cannot disable it.
Only a human restart with different env vars can change protected paths.

Configured via:
  ARTIFEX_RATCHET_EXTRA — colon-separated additional protected paths
"""

import logging
import os
import re

from core.sandbox.policy import (
    PolicyDecision,
    RiskLevel,
    register_policy_hook,
    split_shell_segments,
)

_log = logging.getLogger(__name__)

# ── Protected path patterns ──────────────────────────────────────────────────

_PROTECTED_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"core[/\\]sandbox[/\\]", re.IGNORECASE),
     "sandbox infrastructure"),

    (re.compile(r"core[/\\]config\.py$", re.IGNORECASE),
     "core config"),

    (re.compile(r"core[/\\]resilience\.py$", re.IGNORECASE),
     "resilience module"),

    (re.compile(r"\.github[/\\]workflows[/\\]", re.IGNORECASE),
     "CI/CD workflows"),

    (re.compile(r"\.env$", re.IGNORECASE),
     "environment file"),

    (re.compile(r"pyproject\.toml$", re.IGNORECASE),
     "project config"),

    (re.compile(r"setup\.py$", re.IGNORECASE),
     "setup script"),
]


def _load_extra_patterns() -> list[tuple[re.Pattern, str]]:
    raw = os.environ.get("ARTIFEX_RATCHET_EXTRA", "")
    if not raw:
        return []
    extra = []
    for part in raw.split(os.pathsep):
        part = part.strip()
        if part:
            try:
                extra.append((re.compile(re.escape(part), re.IGNORECASE), f"user-protected: {part}"))
            except re.error:
                _log.warning("Invalid ratchet pattern: %s", part)
    return extra


def _path_forms(path: str, resolve: bool = True) -> list[str]:
    """The spellings of `path` the patterns are matched against.

    The patterns used to see only the string as written, so
    "core/./sandbox/policy.py", "core//sandbox//policy.py" and
    "core/config.py." (Windows drops a trailing dot or space when it opens
    the file) all named a protected file without matching. Each path is also
    checked normalized and, for a path that is really going to be opened,
    resolved.
    """
    forms = [path]
    p = path.strip().strip("\"'")
    for cand in (p, p.rstrip(". ")):
        if not cand:
            continue
        try:
            forms.append(os.path.normpath(cand))
            if resolve:
                forms.append(os.path.realpath(
                    os.path.abspath(os.path.expanduser(cand))))
        except (ValueError, OSError):
            pass
    return list(dict.fromkeys(forms))


def is_protected(path: str, resolve: bool = True) -> str | None:
    """Check if a path is protected by the ratchet.

    Returns the protection reason if protected, None otherwise.
    """
    extra = _load_extra_patterns()
    for form in _path_forms(path, resolve):
        for pattern, reason in _PROTECTED_PATTERNS:
            if pattern.search(form):
                return reason
        for pattern, reason in extra:
            if pattern.search(form):
                return reason
    return None


# ── Policy hook ──────────────────────────────────────────────────────────────

_WRITE_ACTIONS = frozenset({"edit_file", "shell", "python"})


def _extract_write_paths(action_type: str, content: str) -> list[str]:
    """Extract paths that would be written/modified by an action."""
    paths = []
    if action_type == "edit_file":
        parts = content.split("\x00", 2)
        if parts:
            paths.append(parts[0].strip())

    elif action_type == "shell":
        redirect = re.findall(r'>\s*([^\s&|;]+)', content)
        paths.extend(redirect)
        git_add = re.findall(r'git\s+add\s+(?:--\s+)?([^\s]+)', content)
        paths.extend(git_add)
        # Writers other than `>`: Set-Content, cp, sed -i, git checkout,
        # --output=... A command that writes or removes something is held to
        # every path it names (which argument is the target is not knowable
        # from here).
        for seg in split_shell_segments(content):
            if _SHELL_WRITE_RE.search(seg):
                paths.extend(t for t in _SHELL_TOKEN_RE.findall(seg)
                             if is_protected(t, resolve=False))

    elif action_type == "python":
        paths.extend(_python_write_paths(content))

    return paths


# Commands (and options) that create, change or remove files.
_SHELL_WRITE_RE = re.compile(
    r"(?<![\w.-])(?:set-content|add-content|out-file|clear-content|copy-item"
    r"|move-item|remove-item|rename-item|new-item|tee-object|export-[a-z]+"
    r"|sc|ac|cpi|copy|cp|mi|move|mv|ri|rm|del|erase|rd|rmdir|ni|ren|rni|tee"
    r"|touch|truncate|sed|perl|dd|ln|robocopy|xcopy"
    r"|git\s+(?:checkout|restore|rm|mv|apply|stash|reset|clean))(?![\w.-])"
    r"|--output\b|\s-outfile\b"
    r"|\b(?:curl|wget|sort|iwr|irm|invoke-webrequest|invoke-restmethod)\b[^|;&\n]*\s-o",
    re.IGNORECASE)
_SHELL_TOKEN_RE = re.compile(r"[^\s\"'`;|&<>(){}=,]+")

_PY_STR = r"""[rRbBuUfF]{0,2}(["'])(.+?)\1"""
_PY_WRITE_RES = [
    # open("path", "w") / "a" / "x" / "r+"
    re.compile(r"\bopen\(\s*" + _PY_STR
               + r"""\s*,\s*(?:mode\s*=\s*)?["'][^"']*[wax+]"""),
    # Path("path").write_text(...) / .unlink() / .open("w")
    re.compile(r"\bPath\(\s*" + _PY_STR + r"\s*\)\s*\.\s*(?:write_text|write_bytes"
               r"""|unlink|rename|replace|touch|open\(\s*["'][^"']*[wax+])"""),
]
_PY_FILE_CALL = re.compile(
    r"\b(?:os\.(?:remove|unlink|rename|replace|truncate)"
    r"|shutil\.(?:copy\w*|move|rmtree))\(([^)]*)\)")
_PY_ANY_STR = re.compile(r"""(["'])(.+?)\1""")


def _python_write_paths(code: str) -> list[str]:
    """Paths a python snippet plainly writes, renames or deletes.

    Covers the direct spellings only (a literal path handed to open(..., "w"),
    Path(...).write_text, os.remove, shutil.copy ...). A path built at run
    time is invisible here; this is a ratchet, not a jail.
    """
    paths = []
    for rx in _PY_WRITE_RES:
        paths.extend(m.group(2) for m in rx.finditer(code))
    for m in _PY_FILE_CALL.finditer(code):
        paths.extend(s.group(2) for s in _PY_ANY_STR.finditer(m.group(1)))
    return paths


def _ratchet_hook(action_type: str, content: str, risk: RiskLevel):
    """Policy hook that blocks writes to protected paths."""
    if action_type not in _WRITE_ACTIONS:
        return None

    paths = _extract_write_paths(action_type, content)
    for path in paths:
        reason = is_protected(path)
        if reason:
            return PolicyDecision(
                allowed=False,
                requires_confirmation=False,
                risk_level=RiskLevel.CRITICAL,
                reason=f"ratchet: {reason} ({path})",
                matched_rule="self_mod_ratchet",
            )
    return None


def install():
    """Register the self-modification ratchet with the policy engine."""
    register_policy_hook(_ratchet_hook)
    _log.debug("Self-modification ratchet installed")
