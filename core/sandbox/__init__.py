"""
Artifex Assistant V5 — Sandbox subsystem.

Full autonomous execution sandbox stack:
  - Policy engine (T22): Central gatekeeper, risk classification, policy levels.
  - Filesystem sandbox (T23): Path allowlist/denylist, sensitive file protection.
  - Subprocess sandbox (T24): Command blocklist, env scrubbing, resource limits.
  - Capabilities (T25): Per-session action type permissions.
  - Audit log (T26): Append-only JSONL action log with replay.
  - Dry-run mode (T27): Evaluate without executing.
  - Human gates (T28): Round/risk/count-based pause checkpoints.
  - Output validator (T29): Prompt injection and obfuscation detection.
  - Egress policy (T30): URL/domain allowlist/denylist for network access.
  - Self-modification ratchet (T31): Protects sandbox and config from agent writes.
  - Circuit breakers (T32): Error rate, repetition, and rate-limit trips.

Usage:
  from core.sandbox import install_all_hooks, check_policy
  install_all_hooks()  # call once at startup
"""

import logging

from core.sandbox.policy import (
    RiskLevel,
    PolicyLevel,
    PolicyDecision,
    ACTION_RISK,
    get_policy_level,
    check_policy,
    classify_action,
    clear_policy_hooks,
)

_log = logging.getLogger(__name__)


def install_all_hooks():
    """Install all sandbox policy hooks. Call once at application startup.

    Hook order matters — hooks are evaluated first-to-last, first non-None wins:
      1. Output validator (catches injection before anything else)
      2. Filesystem sandbox (blocks sensitive path access)
      3. Subprocess sandbox (blocks dangerous commands)
      4. Self-modification ratchet (blocks writes to protected files)
      5. Egress policy (blocks disallowed network access)
      6. Capabilities (blocks actions outside capability set)
    """
    from core.sandbox.output_validator import install as install_output_validator
    from core.sandbox.fs_sandbox import install as install_fs
    from core.sandbox.proc_sandbox import install as install_proc
    from core.sandbox.ratchet import install as install_ratchet
    from core.sandbox.egress import install as install_egress
    from core.sandbox.capabilities import install as install_caps

    install_output_validator()
    install_fs()
    install_proc()
    install_ratchet()
    install_egress()
    install_caps()
    _log.info("All sandbox hooks installed")


__all__ = [
    "RiskLevel",
    "PolicyLevel",
    "PolicyDecision",
    "ACTION_RISK",
    "get_policy_level",
    "check_policy",
    "classify_action",
    "clear_policy_hooks",
    "install_all_hooks",
]
