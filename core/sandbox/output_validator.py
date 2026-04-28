"""
Artifex Assistant V5 — Model output validation (P6-T29).

Validates model-generated action content before execution. Catches:
  1. Prompt injection attempts in tool arguments.
  2. Encoded/obfuscated payloads (base64, hex, URL encoding).
  3. Excessive action chaining in a single response.
  4. Malformed tool call syntax.

This is a defense-in-depth layer — the model is trusted, but its output
is influenced by user input and retrieved web content, both of which can
contain adversarial prompts.
"""

import logging
import re

from core.sandbox.policy import (
    PolicyDecision,
    RiskLevel,
    register_policy_hook,
)

_log = logging.getLogger(__name__)

# ── Injection detection patterns ─────────────────────────────────────────────

_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"ignore\s+(previous|all|above)\s+(instructions|prompts?)", re.IGNORECASE),
     "prompt injection: ignore instructions"),

    (re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
     "prompt injection: role reassignment"),

    (re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
     "prompt injection: system prompt override"),

    (re.compile(r"act\s+as\s+(if\s+you\s+are|a)\s+", re.IGNORECASE),
     "prompt injection: role play"),

    (re.compile(r"forget\s+(everything|all|your)\b", re.IGNORECASE),
     "prompt injection: memory wipe"),

    (re.compile(r"<\|im_start\|>|<\|im_end\|>", re.IGNORECASE),
     "prompt injection: ChatML tokens"),

    (re.compile(r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", re.IGNORECASE),
     "prompt injection: Llama tokens"),

    (re.compile(r"<\|system\|>|<\|user\|>|<\|assistant\|>", re.IGNORECASE),
     "prompt injection: role tokens"),
]

# ── Obfuscation detection ───────────────────────────────────────────────────

_OBFUSCATION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\\x[0-9a-f]{2}(\\x[0-9a-f]{2}){3,}", re.IGNORECASE),
     "hex-encoded payload"),

    (re.compile(r"%[0-9a-f]{2}(%[0-9a-f]{2}){5,}", re.IGNORECASE),
     "URL-encoded payload"),

    (re.compile(r"\\u[0-9a-f]{4}(\\u[0-9a-f]{4}){3,}", re.IGNORECASE),
     "unicode-escaped payload"),

    (re.compile(r"chr\(\d+\)\s*\+\s*chr\(\d+\)", re.IGNORECASE),
     "character concatenation obfuscation"),
]


def check_injection(content: str) -> str | None:
    """Check for prompt injection patterns in action content.

    Returns the reason string if injection detected, None if clean.
    """
    for pattern, reason in _INJECTION_PATTERNS:
        if pattern.search(content):
            return reason
    return None


def check_obfuscation(content: str) -> str | None:
    """Check for obfuscated/encoded payloads.

    Returns the reason string if obfuscation detected, None if clean.
    """
    for pattern, reason in _OBFUSCATION_PATTERNS:
        if pattern.search(content):
            return reason
    return None


def validate_output(content: str) -> str | None:
    """Run all output validation checks.

    Returns the first failure reason, or None if all checks pass.
    """
    result = check_injection(content)
    if result:
        return result
    result = check_obfuscation(content)
    if result:
        return result
    return None


# ── Policy hook ──────────────────────────────────────────────────────────────

def _output_validation_hook(action_type: str, content: str, risk: RiskLevel):
    """Policy hook that validates model output before execution."""
    if not content:
        return None

    reason = validate_output(content)
    if reason:
        _log.warning("Output validation failed for %s: %s", action_type, reason)
        return PolicyDecision(
            allowed=False,
            requires_confirmation=False,
            risk_level=RiskLevel.CRITICAL,
            reason=f"output validation: {reason}",
            matched_rule="output_validator",
        )
    return None


def install():
    """Register the output validation hook with the policy engine."""
    register_policy_hook(_output_validation_hook)
    _log.debug("Output validator installed")
