"""
Artifex Assistant V5 — Filesystem sandbox (P6-T23).

Restricts agent file access to an allowed directory tree. By default the
project directory and system temp dirs are allowed; everything else is denied.

Hooks into the policy engine so file-touching actions (read_file, edit_file,
shell, python, download, glob, grep) are path-checked before execution.

Configured via:
  ARTIFEX_SANDBOX_ROOTS  — colon-separated additional allowed roots
  ARTIFEX_SANDBOX_DENY   — colon-separated deny patterns (checked first)
"""

import logging
import os
import re
import tempfile
from pathlib import PurePosixPath, PureWindowsPath

from core.sandbox.policy import (
    PolicyDecision,
    RiskLevel,
    register_policy_hook,
)

_log = logging.getLogger(__name__)

# ── Sensitive paths that are always denied ───────────────────────────────────

_ALWAYS_DENY = [
    re.compile(r"[/\\]\.ssh[/\\]?", re.IGNORECASE),
    re.compile(r"[/\\]\.gnupg[/\\]?", re.IGNORECASE),
    re.compile(r"[/\\]\.aws[/\\]?", re.IGNORECASE),
    re.compile(r"[/\\]\.azure[/\\]?", re.IGNORECASE),
    re.compile(r"[/\\]\.gcloud[/\\]?", re.IGNORECASE),
    re.compile(r"[/\\]\.kube[/\\]?", re.IGNORECASE),
    re.compile(r"[/\\]\.docker[/\\]config\.json", re.IGNORECASE),
    re.compile(r"[/\\]\.npmrc$", re.IGNORECASE),
    re.compile(r"[/\\]\.pypirc$", re.IGNORECASE),
    re.compile(r"[/\\]\.netrc$", re.IGNORECASE),
    re.compile(r"[/\\]\.env$", re.IGNORECASE),
    re.compile(r"[/\\]\.env\.local$", re.IGNORECASE),
    re.compile(r"[/\\]credentials\.json$", re.IGNORECASE),
    re.compile(r"[/\\]service.account\.json$", re.IGNORECASE),
    re.compile(r"[/\\]id_rsa$", re.IGNORECASE),
    re.compile(r"[/\\]id_ed25519$", re.IGNORECASE),
    re.compile(r"[/\\]\.git[/\\]config$", re.IGNORECASE),
    re.compile(r"\\Windows\\System32\\config\\", re.IGNORECASE),
    re.compile(r"[/\\]etc[/\\]shadow$"),
    re.compile(r"[/\\]etc[/\\]passwd$"),
]

_USER_DENY: list[re.Pattern] = []


def _load_user_deny():
    """Load user-specified deny patterns from ARTIFEX_SANDBOX_DENY."""
    raw = os.environ.get("ARTIFEX_SANDBOX_DENY", "")
    if not raw:
        return
    _USER_DENY.clear()
    for part in raw.split(os.pathsep):
        part = part.strip()
        if part:
            try:
                _USER_DENY.append(re.compile(re.escape(part), re.IGNORECASE))
            except re.error:
                _log.warning("Invalid deny pattern: %s", part)


_load_user_deny()


def is_path_denied(path: str) -> str | None:
    """Check if a path matches any deny pattern.

    Returns the matched pattern description, or None if allowed.
    """
    for pat in _ALWAYS_DENY:
        if pat.search(path):
            return f"sensitive path: {pat.pattern}"
    for pat in _USER_DENY:
        if pat.search(path):
            return f"user-denied: {pat.pattern}"
    return None


# ── Allowed root directories ─────────────────────────────────────────────────

def _get_allowed_roots() -> list[str]:
    """Build the list of allowed root directories."""
    roots = []

    from core.config import BASE_DIR
    roots.append(os.path.abspath(BASE_DIR))

    roots.append(os.path.abspath(tempfile.gettempdir()))

    extra = os.environ.get("ARTIFEX_SANDBOX_ROOTS", "")
    if extra:
        for part in extra.split(os.pathsep):
            part = part.strip()
            if part and os.path.isabs(part):
                roots.append(os.path.abspath(part))

    return roots


def _resolve_path(path: str) -> str | None:
    """Expand ~, make absolute, and resolve symlinks.

    Returns None if the path can't be resolved. realpath follows symlinks so
    a link inside the sandbox pointing to /etc/passwd resolves to /etc/passwd
    and gets caught by both the deny list and the sandbox-root check.
    Non-existent path components are kept verbatim, so this also works for
    files we're about to create.
    """
    try:
        expanded = os.path.expanduser(path)
        return os.path.realpath(os.path.abspath(expanded))
    except (ValueError, OSError):
        return None


def _is_under_root(abs_path: str, root: str) -> bool:
    """True iff abs_path == root or abs_path is a child of root.

    Plain startswith() is wrong because /x/projEVIL starts with /x/proj.
    Compare with a trailing separator (or exact equality) to avoid that.
    """
    if abs_path == root:
        return True
    return abs_path.startswith(root + os.sep)


def is_path_within_sandbox(path: str) -> bool:
    """Check if a path is within any allowed root directory.

    Resolves symlinks before the check so a link inside the sandbox pointing
    outside is caught. Uses an exact-or-child check rather than a raw prefix
    match so /x/projEVIL doesn't slip past a /x/proj root.
    """
    abs_path = _resolve_path(path)
    if abs_path is None:
        return False

    for root in _get_allowed_roots():
        resolved_root = os.path.realpath(root)
        if _is_under_root(abs_path, resolved_root):
            return True
    return False


def check_path(path: str) -> str | None:
    """Full path check: deny list first, then sandbox roots.

    Returns an error message if denied, None if allowed. The deny list is
    checked against both the raw input (so it catches obvious cases like
    "/etc/passwd") and the symlink-resolved path (so it catches a sandboxed
    symlink that points at a sensitive target).
    """
    deny_reason = is_path_denied(path)
    if deny_reason:
        return deny_reason

    resolved = _resolve_path(path)
    if resolved is not None and resolved != path:
        deny_reason = is_path_denied(resolved)
        if deny_reason:
            return f"{deny_reason} (via symlink)"

    if not is_path_within_sandbox(path):
        return f"outside sandbox: {path}"
    return None


# ── Path extraction from action content ──────────────────────────────────────

_PATH_LIKE = re.compile(
    r'(?:[A-Za-z]:[/\\]|[/\\]|\.{1,2}[/\\])'
    r'[^\s:*?"<>|\x00-\x1f]+'
)

# Leading escape sequences in shell/python string literals ("hi\n") produce
# bogus candidates: `printf 'hi\n' > out.txt` extracts "\n'" as a path, which
# abspath()s to <drive root>\n' and is rejected as "outside sandbox" —
# blocking the single most common way an agent writes a file. Only candidates
# BEGINNING with a bare-backslash escape are affected; drive-prefixed and
# dot-relative candidates are kept verbatim, since stripping escapes out of
# those breaks the drive prefix and lets real paths whose components start
# with escape letters (C:\temp\notes.txt) evade the check entirely.
_LEADING_ESCAPES = re.compile(
    r'^(?:\\(?:[ntrbfva0]|x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}))+'
)


def _filter_escape_candidates(candidates: list[str]) -> list[str]:
    """Drop or trim path candidates that are really string-literal escapes.

    A candidate starting with an escape sequence ("\\n'") is junk unless a
    genuine path tail follows the escapes ("\\n\\.ssh\\id_rsa" keeps
    "\\.ssh\\id_rsa" so the deny list still sees it).
    """
    kept = []
    for cand in candidates:
        if cand.startswith("\\"):
            remainder = _LEADING_ESCAPES.sub("", cand)
            if remainder != cand:
                if remainder.startswith(("\\", "/")):
                    kept.append(remainder)
                continue
        kept.append(cand)
    return kept


def extract_paths_from_content(action_type: str, content: str) -> list[str]:
    """Extract file paths from action content for sandbox checking."""
    paths = []

    if action_type == "read_file":
        path = content.rsplit("|", 1)[0].strip()
        if path and not path.startswith(("http://", "https://")):
            paths.append(path)

    elif action_type == "read_function":
        path = content.split("|", 1)[0].strip()
        if path:
            paths.append(path)

    elif action_type == "edit_file":
        parts = content.split("\x00", 2)
        if parts:
            paths.append(parts[0].strip())

    elif action_type == "glob":
        parts = content.split("|")
        if len(parts) > 1 and parts[1].strip():
            paths.append(parts[1].strip())

    elif action_type == "grep":
        parts = content.split("|", 2)
        if len(parts) > 1:
            paths.append(parts[1].strip())

    elif action_type in ("shell", "python"):
        paths.extend(_filter_escape_candidates(_PATH_LIKE.findall(content)))

    elif action_type == "download":
        parts = content.split("|", 1)
        if len(parts) > 1:
            paths.append(parts[1].strip())

    return paths


# ── Policy hook ──────────────────────────────────────────────────────────────

def _fs_sandbox_hook(action_type: str, content: str, risk: RiskLevel):
    """Policy hook that checks file paths against the sandbox.

    Two distinct rule tags so hosts can treat them differently:
    - "fs_sandbox"       — deny-list hit (credentials, keys, system files).
      Absolute; never promptable.
    - "fs_sandbox_scope" — path is merely outside the allowed roots. A
      supervised loop (guided/manual) may surface this as an approval
      prompt instead of a wall; unattended (full-auto) hosts keep it hard.
    """
    paths = extract_paths_from_content(action_type, content)
    for path in paths:
        reason = check_path(path)
        if reason:
            scope_only = reason.startswith("outside sandbox:")
            return PolicyDecision(
                allowed=False,
                requires_confirmation=False,
                risk_level=risk,
                reason=f"filesystem sandbox: {reason}",
                matched_rule="fs_sandbox_scope" if scope_only else "fs_sandbox",
            )
    return None


def install():
    """Register the filesystem sandbox hook with the policy engine."""
    register_policy_hook(_fs_sandbox_hook)
    _log.debug("Filesystem sandbox installed")
