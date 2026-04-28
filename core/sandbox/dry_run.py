"""
Artifex Assistant V5 — Dry-run mode (P6-T27).

When enabled, all actions are evaluated by the policy engine and audit-logged
but NOT executed. The agent sees "[DRY RUN] would execute: <action>" instead
of real output. Useful for testing policy configurations and reviewing what
an agent run would do before enabling it for real.

Configured via:
  ARTIFEX_DRY_RUN=1  — enable dry-run mode
  Or programmatically via enable() / disable().
"""

import logging
import os
import threading

_log = logging.getLogger(__name__)

_lock = threading.Lock()
_enabled: bool | None = None


def _load_from_env() -> bool:
    raw = os.environ.get("ARTIFEX_DRY_RUN", "").strip().lower()
    return raw in ("1", "true", "yes")


def is_enabled() -> bool:
    """Check if dry-run mode is active."""
    global _enabled
    with _lock:
        if _enabled is None:
            _enabled = _load_from_env()
        return _enabled


def enable():
    """Enable dry-run mode."""
    global _enabled
    with _lock:
        _enabled = True
    _log.info("Dry-run mode enabled")


def disable():
    """Disable dry-run mode."""
    global _enabled
    with _lock:
        _enabled = False
    _log.info("Dry-run mode disabled")


def reset():
    """Reset to env-var default. For testing."""
    global _enabled
    with _lock:
        _enabled = None


def dry_run_result(action_type: str, content: str) -> str:
    """Generate a dry-run result message."""
    preview = content[:150] if content else "(empty)"
    return f"[DRY RUN] Would execute {action_type}: {preview}"
