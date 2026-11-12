"""
Artifex Assistant V5 — Subprocess sandbox (P6-T24).

Tiered subprocess isolation for agent-executed commands:
  1. Resource limits: timeout cap, output size cap, process count.
  2. Command blocklist: patterns that should never execute.
  3. Environment scrubbing: strips secrets from child process env.

On Windows, full OS-level sandboxing (bubblewrap/firejail) isn't available,
so this module focuses on the portable layer: env scrubbing, resource caps,
and command analysis. Docker-based sandboxing is deferred to the gateway.

Hooks into the policy engine via register_policy_hook.
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

# ── Resource limits ──────────────────────────────────────────────────────────

MAX_COMMAND_TIMEOUT = int(os.environ.get("ARTIFEX_MAX_CMD_TIMEOUT", "300"))
MAX_OUTPUT_BYTES = int(os.environ.get("ARTIFEX_MAX_OUTPUT_BYTES", str(5 * 1024 * 1024)))
MAX_PYTHON_TIMEOUT = int(os.environ.get("ARTIFEX_MAX_PYTHON_TIMEOUT", "30"))

# ── Command blocklist (beyond what _check_dangerous covers) ──────────────────
# These patterns catch evasion techniques: base64-encoded payloads,
# environment variable expansion tricks, network exfiltration.

_BLOCKED_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"base64\s+(-d|--decode)", re.IGNORECASE),
     "base64 decode (payload obfuscation)"),

    (re.compile(r"\beval\b.*\$\(", re.IGNORECASE),
     "eval with command substitution"),

    (re.compile(r"python[23]?\s+-c\s+.*import\s+(os|subprocess|shutil)", re.IGNORECASE),
     "inline python with dangerous imports"),

    (re.compile(r"\bnc\b.*-[le]", re.IGNORECASE),
     "netcat listener/exec"),

    (re.compile(r"\bsocat\b", re.IGNORECASE),
     "socat (network relay)"),

    (re.compile(r"\bnmap\b", re.IGNORECASE),
     "nmap (port scanner)"),

    (re.compile(r"\bchmod\s+[0-7]*[sS]", re.IGNORECASE),
     "setuid/setgid chmod"),

    (re.compile(r"\bchmod\s+u\+s", re.IGNORECASE),
     "setuid chmod"),

    (re.compile(r"\bcrontab\b", re.IGNORECASE),
     "crontab modification"),

    # Only where a command can start (line start, after ; | & ( ` or a
    # quote, after cmd /c or -c). As a bare `\bat\s+\d` it also blocked
    # ordinary text: echo "failed at 3 places".
    (re.compile(r"(?:^|[;&|(`\"']|/[ck]\s|\s-(?:c|command)\s)\s*at(?:\.exe)?\s+\d",
                re.IGNORECASE | re.MULTILINE),
     "at job scheduling"),

    (re.compile(r">\s*/etc/", re.IGNORECASE),
     "write to /etc/"),

    (re.compile(r"\bsudo\b", re.IGNORECASE),
     "privilege escalation (sudo)"),

    (re.compile(r"\bsu\s+-", re.IGNORECASE),
     "privilege escalation (su)"),

    (re.compile(r"\brunas\b", re.IGNORECASE),
     "privilege escalation (runas)"),

    (re.compile(r"Invoke-WebRequest.*\|\s*Invoke-Expression", re.IGNORECASE),
     "PowerShell download-and-execute"),

    (re.compile(r"iex\s*\(.*Net\.WebClient", re.IGNORECASE),
     "PowerShell IEX download"),
]


def check_command_blocked(command: str) -> str | None:
    """Check if a command matches any blocked pattern.

    Returns the reason string if blocked, None if allowed.
    """
    for pattern, reason in _BLOCKED_PATTERNS:
        if pattern.search(command):
            return reason
    return None


# ── Environment scrubbing ────────────────────────────────────────────────────
# Keys to strip from child process environments.

_SCRUB_ENV_KEYS = frozenset({
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GH_TOKEN", "GITHUB_TOKEN",
    "GITLAB_TOKEN",
    "NPM_TOKEN",
    "PYPI_TOKEN",
    "ARTIFEX_AGENT_KEY",
    "ARTIFEX_API_KEY",      # the API's bearer key: a command could call the API with it
    "GATEWAY_AUTH_TOKEN",
    "DATABASE_URL",
    "REDIS_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "HF_TOKEN",
})

_SCRUB_PREFIXES = ("SECRET_", "TOKEN_", "PRIVATE_KEY_")
# Name endings that mark a credential whatever the prefix (FOO_API_KEY,
# BAR_SECRET, ...): a new key added to the launcher is stripped without
# anyone remembering this list. Endings, not substrings, so settings such as
# HF_HUB_DISABLE_IMPLICIT_TOKEN or TOKENIZERS_PARALLELISM pass through.
_SCRUB_SUFFIXES = ("_API_KEY", "_SECRET", "_SECRET_KEY", "_PASSWORD",
                   "_ACCESS_KEY", "_PRIVATE_KEY", "_SIGNING_KEY", "_HMAC_KEY",
                   "_AUTH_TOKEN")


def scrub_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return a copy of the environment with secrets removed."""
    base = env if env is not None else os.environ.copy()
    clean = {}
    for key, value in base.items():
        if key in _SCRUB_ENV_KEYS:
            continue
        if any(key.startswith(p) for p in _SCRUB_PREFIXES):
            continue
        if key.upper().endswith(_SCRUB_SUFFIXES):
            continue
        clean[key] = value
    return clean


# ── Policy hook ──────────────────────────────────────────────────────────────

def _proc_sandbox_hook(action_type: str, content: str, risk: RiskLevel):
    """Policy hook that checks shell/python commands against the blocklist."""
    if action_type not in ("shell", "python"):
        return None

    reason = check_command_blocked(content)
    if reason:
        return PolicyDecision(
            allowed=False,
            requires_confirmation=False,
            risk_level=RiskLevel.CRITICAL,
            reason=f"subprocess sandbox: blocked ({reason})",
            matched_rule="proc_blocklist",
        )
    return None


def install():
    """Register the subprocess sandbox hook with the policy engine."""
    register_policy_hook(_proc_sandbox_hook)
    _log.debug("Subprocess sandbox installed")
