"""
Artifex Assistant V5 — Auto-exec policy engine (P6-T22).

Central gatekeeper for agent action execution. Classifies actions by risk,
evaluates them against the active policy level, and returns allow/deny/confirm
decisions. All subsequent sandbox components (T23-T33) hook into this module.

Policy levels:
  STRICT     — Every action requires human confirmation (default).
  MODERATE   — Read-only actions auto-allowed; writes/exec need confirmation.
  PERMISSIVE — Only shell commands and downloads need confirmation.
  AUTO       — All actions auto-allowed. Requires ARTIFEX_AGENT_KEY.

Configured via ARTIFEX_POLICY env var. Falls back to STRICT.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

_log = logging.getLogger(__name__)


# ── Risk levels (ordered by severity) ────────────────────────────────────────

class RiskLevel(IntEnum):
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# ── Action type → default risk classification ────────────────────────────────

ACTION_RISK: dict[str, RiskLevel] = {
    "read_file":        RiskLevel.SAFE,
    "read_function":    RiskLevel.SAFE,
    "glob":             RiskLevel.SAFE,
    "grep":             RiskLevel.SAFE,
    "find_symbol":      RiskLevel.SAFE,
    "find_references":  RiskLevel.SAFE,
    "trace_imports":    RiskLevel.SAFE,
    "architecture":     RiskLevel.SAFE,
    "sysinfo":          RiskLevel.SAFE,
    "search":           RiskLevel.LOW,
    "web_read":         RiskLevel.LOW,
    "edit_file":        RiskLevel.MEDIUM,
    "python":           RiskLevel.MEDIUM,
    "download":         RiskLevel.HIGH,
    "shell":            RiskLevel.HIGH,
}


# ── Policy levels ────────────────────────────────────────────────────────────

class PolicyLevel:
    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"
    AUTO = "auto"

    _VALID = frozenset({"strict", "moderate", "permissive", "auto"})

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value.lower() in cls._VALID


# ── Policy decision (returned by check_policy) ──────────────────────────────

@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    requires_confirmation: bool
    risk_level: RiskLevel
    reason: str
    matched_rule: str = ""


# ── Shell command risk escalation ────────────────────────────────────────────
# Commands that escalate shell actions from HIGH to CRITICAL.

_CRITICAL_SHELL_PATTERNS: list[re.Pattern] = [
    re.compile(r"\brm\s+(-[a-z]*r[a-z]*\s+|.*--recursive)", re.IGNORECASE),
    re.compile(r"\brm\s+-[a-z]*f", re.IGNORECASE),
    re.compile(r"\bformat\b.*[a-z]:", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bdd\s+.*of=/dev/", re.IGNORECASE),
    re.compile(r"\b(shutdown|reboot|halt|poweroff)\b", re.IGNORECASE),
    re.compile(r"\bgit\s+push\s+.*--force", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\s+--hard", re.IGNORECASE),
    re.compile(r"\bdrop\s+(table|database)\b", re.IGNORECASE),
    re.compile(r"\btruncate\s+table\b", re.IGNORECASE),
    re.compile(r"\bcurl\b.*\|\s*(bash|sh)\b", re.IGNORECASE),
    re.compile(r"\bwget\b.*\|\s*(bash|sh)\b", re.IGNORECASE),
    re.compile(r"\breg\s+(delete|add)\b", re.IGNORECASE),
    re.compile(r"\bnet\s+user\b", re.IGNORECASE),
]

_MEDIUM_SHELL_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bgit\s+(add|commit|stash|checkout|branch|merge|rebase)\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+(install|uninstall|update)\b", re.IGNORECASE),
    re.compile(r"\bpip\s+install\b", re.IGNORECASE),
    re.compile(r"\bmkdir\b", re.IGNORECASE),
    re.compile(r"\btouch\b", re.IGNORECASE),
    re.compile(r"\bcp\b", re.IGNORECASE),
    re.compile(r"\bmv\b", re.IGNORECASE),
]

_SAFE_SHELL_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*(ls|dir|pwd|cd|echo|cat|head|tail|type|wc|find|which|where)\b"),
    re.compile(r"^\s*(git\s+(status|log|diff|show|branch\s*$))\b"),
    re.compile(r"^\s*python\s+--version"),
    re.compile(r"^\s*(node|npm|pip)\s+--version"),
]


def classify_shell_risk(command: str) -> RiskLevel:
    """Classify a shell command into a risk level based on content analysis."""
    for pat in _CRITICAL_SHELL_PATTERNS:
        if pat.search(command):
            return RiskLevel.CRITICAL
    for pat in _SAFE_SHELL_PATTERNS:
        if pat.search(command):
            return RiskLevel.SAFE
    for pat in _MEDIUM_SHELL_PATTERNS:
        if pat.search(command):
            return RiskLevel.MEDIUM
    return RiskLevel.HIGH


def classify_action(action_type: str, content: str = "") -> RiskLevel:
    """Classify an action into a risk level.

    For shell commands, does content-based analysis to refine the default
    HIGH classification (e.g., `ls` is SAFE, `rm -rf` is CRITICAL).
    """
    if action_type == "shell" and content:
        return classify_shell_risk(content)
    return ACTION_RISK.get(action_type, RiskLevel.HIGH)


# ── Policy auto-allow thresholds ─────────────────────────────────────────────
# Maps policy level → maximum risk level that's auto-allowed (no confirmation).
# Anything above the threshold requires confirmation or is denied.

_STRICT_SENTINEL = -1

_POLICY_AUTO_THRESHOLD: dict[str, int] = {
    PolicyLevel.STRICT:     _STRICT_SENTINEL,    # nothing auto-allowed
    PolicyLevel.MODERATE:   RiskLevel.SAFE,       # read-only auto-allowed
    PolicyLevel.PERMISSIVE: RiskLevel.MEDIUM,     # up to edits/python auto-allowed
    PolicyLevel.AUTO:       RiskLevel.CRITICAL,   # everything auto-allowed
}


# ── Hook registry (for T23-T33 to register additional checks) ───────────────

_policy_hooks: list[Callable] = []


def register_policy_hook(hook: Callable) -> None:
    """Register an additional policy check.

    Hooks are called with (action_type, content, risk_level) and should return
    a PolicyDecision to override the default, or None to defer.
    Hooks are evaluated in registration order; first non-None wins.
    """
    _policy_hooks.append(hook)
    _log.debug("Registered policy hook: %s", hook.__name__)


def clear_policy_hooks() -> None:
    """Remove all registered policy hooks (for testing)."""
    _policy_hooks.clear()


# ── Core policy check ────────────────────────────────────────────────────────

def get_policy_level() -> str:
    """Read the active policy level from environment.

    Returns one of: strict, moderate, permissive, auto.
    Defaults to strict. AUTO requires ARTIFEX_AGENT_KEY to be set.
    """
    raw = os.environ.get("ARTIFEX_POLICY", "strict").lower().strip()
    if not PolicyLevel.is_valid(raw):
        _log.warning("Invalid ARTIFEX_POLICY=%r, falling back to strict", raw)
        return PolicyLevel.STRICT

    if raw == PolicyLevel.AUTO:
        agent_key = os.environ.get("ARTIFEX_AGENT_KEY", "")
        if not agent_key:
            _log.warning("ARTIFEX_POLICY=auto requires ARTIFEX_AGENT_KEY; falling back to strict")
            return PolicyLevel.STRICT

    return raw


def check_policy(action_type: str, content: str = "") -> PolicyDecision:
    """Evaluate an action against the active policy.

    Returns a PolicyDecision indicating whether the action is allowed,
    requires confirmation, or is denied.
    """
    risk = classify_action(action_type, content)
    policy = get_policy_level()

    for hook in _policy_hooks:
        try:
            decision = hook(action_type, content, risk)
            if decision is not None:
                return decision
        except Exception as e:
            _log.warning("Policy hook %s raised: %s", hook.__name__, e)

    threshold = _POLICY_AUTO_THRESHOLD[policy]

    if policy == PolicyLevel.STRICT:
        return PolicyDecision(
            allowed=True,
            requires_confirmation=True,
            risk_level=risk,
            reason="strict policy: all actions require confirmation",
            matched_rule="strict",
        )

    if risk == RiskLevel.CRITICAL and policy != PolicyLevel.AUTO:
        return PolicyDecision(
            allowed=True,
            requires_confirmation=True,
            risk_level=risk,
            reason=f"critical risk: always requires confirmation (policy={policy})",
            matched_rule="critical_always_confirm",
        )

    if risk.value <= threshold.value:
        return PolicyDecision(
            allowed=True,
            requires_confirmation=False,
            risk_level=risk,
            reason=f"auto-allowed: {action_type} is {risk.name} (policy={policy})",
            matched_rule=f"auto_{policy}",
        )

    return PolicyDecision(
        allowed=True,
        requires_confirmation=True,
        risk_level=risk,
        reason=f"{risk.name} risk exceeds {policy} auto-threshold",
        matched_rule=f"confirm_{policy}",
    )
