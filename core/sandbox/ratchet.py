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


def is_protected(path: str) -> str | None:
    """Check if a path is protected by the ratchet.

    Returns the protection reason if protected, None otherwise.
    """
    for pattern, reason in _PROTECTED_PATTERNS:
        if pattern.search(path):
            return reason
    for pattern, reason in _load_extra_patterns():
        if pattern.search(path):
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
