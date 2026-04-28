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


def is_path_within_sandbox(path: str) -> bool:
    """Check if a path is within any allowed root directory."""
    try:
        abs_path = os.path.abspath(os.path.expanduser(path))
    except (ValueError, OSError):
        return False

    for root in _get_allowed_roots():
        try:
            os.path.relpath(abs_path, root)
            if abs_path.startswith(root):
                return True
        except ValueError:
            continue
    return False


def check_path(path: str) -> str | None:
    """Full path check: deny list first, then sandbox roots.

    Returns an error message if denied, None if allowed.
    """
    deny_reason = is_path_denied(path)
    if deny_reason:
        return deny_reason
    if not is_path_within_sandbox(path):
        return f"outside sandbox: {path}"
    return None


# ── Path extraction from action content ──────────────────────────────────────

_PATH_LIKE = re.compile(
    r'(?:[A-Za-z]:[/\\]|[/\\]|\.{1,2}[/\\])'
    r'[^\s:*?"<>|\x00-\x1f]+'
)


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
        paths.extend(_PATH_LIKE.findall(content))

    elif action_type == "download":
        parts = content.split("|", 1)
        if len(parts) > 1:
            paths.append(parts[1].strip())

    return paths


# ── Policy hook ──────────────────────────────────────────────────────────────

def _fs_sandbox_hook(action_type: str, content: str, risk: RiskLevel):
    """Policy hook that checks file paths against the sandbox."""
    paths = extract_paths_from_content(action_type, content)
    for path in paths:
        reason = check_path(path)
        if reason:
            return PolicyDecision(
                allowed=False,
                requires_confirmation=False,
                risk_level=risk,
                reason=f"filesystem sandbox: {reason}",
                matched_rule="fs_sandbox",
            )
    return None


def install():
    """Register the filesystem sandbox hook with the policy engine."""
    register_policy_hook(_fs_sandbox_hook)
    _log.debug("Filesystem sandbox installed")
