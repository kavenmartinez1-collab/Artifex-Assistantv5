"""
Artifex Assistant V5 — Capability-based permissions (P6-T25).

Each agent session holds a set of capabilities (action types it may use).
By default all capabilities are granted. Capabilities can be narrowed at
session start for restricted agent runs.

Configured via:
  ARTIFEX_CAPABILITIES — comma-separated list of allowed action types.
                         Empty or unset means "all allowed".

Hooks into the policy engine to deny actions outside the capability set.
"""

import logging
import os
import threading

from core.sandbox.policy import (
    PolicyDecision,
    RiskLevel,
    ACTION_RISK,
    register_policy_hook,
)

_log = logging.getLogger(__name__)

ALL_CAPABILITIES = frozenset(ACTION_RISK.keys())

# ── Session capability store (thread-safe) ───────────────────────────────────

_lock = threading.Lock()
_session_caps: set[str] = set()
_initialized = False


def _load_from_env():
    """Load capabilities from ARTIFEX_CAPABILITIES env var."""
    raw = os.environ.get("ARTIFEX_CAPABILITIES", "").strip()
    if not raw:
        return set(ALL_CAPABILITIES)
    caps = set()
    for part in raw.split(","):
        part = part.strip().lower()
        if part in ALL_CAPABILITIES:
            caps.add(part)
        elif part:
            _log.warning("Unknown capability: %r (ignored)", part)
    return caps if caps else set(ALL_CAPABILITIES)


def get_capabilities() -> frozenset[str]:
    """Get the current session's capability set."""
    global _initialized
    with _lock:
        if not _initialized:
            _session_caps.update(_load_from_env())
            _initialized = True
        return frozenset(_session_caps)


def set_capabilities(caps: set[str]) -> None:
    """Override capabilities for the current session."""
    global _initialized
    with _lock:
        _session_caps.clear()
        _session_caps.update(caps & ALL_CAPABILITIES)
        _initialized = True
        _log.info("Capabilities set: %s", _session_caps)


def reset_capabilities() -> None:
    """Reset to default (all allowed). For testing."""
    global _initialized
    with _lock:
        _session_caps.clear()
        _initialized = False


def has_capability(action_type: str) -> bool:
    """Check if the current session has a specific capability."""
    return action_type in get_capabilities()


def grant(action_type: str) -> None:
    """Grant a single capability."""
    global _initialized
    with _lock:
        if not _initialized:
            _session_caps.update(_load_from_env())
            _initialized = True
        if action_type in ALL_CAPABILITIES:
            _session_caps.add(action_type)


def revoke(action_type: str) -> None:
    """Revoke a single capability."""
    global _initialized
    with _lock:
        if not _initialized:
            _session_caps.update(_load_from_env())
            _initialized = True
        _session_caps.discard(action_type)


# ── Policy hook ──────────────────────────────────────────────────────────────

def _capabilities_hook(action_type: str, content: str, risk: RiskLevel):
    """Policy hook that checks action against session capabilities."""
    if not has_capability(action_type):
        return PolicyDecision(
            allowed=False,
            requires_confirmation=False,
            risk_level=risk,
            reason=f"capability not granted: {action_type}",
            matched_rule="capability_denied",
        )
    return None


def install():
    """Register the capabilities hook with the policy engine."""
    register_policy_hook(_capabilities_hook)
    _log.debug("Capability-based permissions installed")
