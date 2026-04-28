"""
Artifex Assistant V5 — Network egress policy (P6-T30).

Controls which URLs/domains the agent can access via web_read, download,
search, and shell commands containing network operations.

Modes:
  OPEN       — all external network access allowed (default).
  ALLOWLIST  — only explicitly allowed domains.
  DENYLIST   — all except explicitly denied domains.

Configured via:
  ARTIFEX_EGRESS_MODE      — open, allowlist, or denylist
  ARTIFEX_EGRESS_ALLOW     — comma-separated allowed domains (for allowlist)
  ARTIFEX_EGRESS_DENY      — comma-separated denied domains (for denylist)
"""

import logging
import os
import re
from urllib.parse import urlparse

from core.sandbox.policy import (
    PolicyDecision,
    RiskLevel,
    register_policy_hook,
)

_log = logging.getLogger(__name__)

# ── Egress modes ─────────────────────────────────────────────────────────────

class EgressMode:
    OPEN = "open"
    ALLOWLIST = "allowlist"
    DENYLIST = "denylist"

    _VALID = frozenset({"open", "allowlist", "denylist"})

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value.lower() in cls._VALID


def get_egress_mode() -> str:
    raw = os.environ.get("ARTIFEX_EGRESS_MODE", "open").lower().strip()
    if not EgressMode.is_valid(raw):
        _log.warning("Invalid ARTIFEX_EGRESS_MODE=%r, falling back to open", raw)
        return EgressMode.OPEN
    return raw


def _load_domain_list(env_var: str) -> set[str]:
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return set()
    return {d.strip().lower() for d in raw.split(",") if d.strip()}


def _domain_matches(hostname: str, domain_set: set[str]) -> bool:
    """Check if hostname matches any domain in the set (supports subdomain matching)."""
    hostname = hostname.lower()
    for domain in domain_set:
        if hostname == domain or hostname.endswith("." + domain):
            return True
    return False


# ── URL extraction ───────────────────────────────────────────────────────────

_URL_RE = re.compile(r'https?://[^\s<>"\']+', re.IGNORECASE)


def extract_urls(action_type: str, content: str) -> list[str]:
    """Extract URLs from action content."""
    urls = []

    if action_type == "web_read":
        if content and not content.isdigit():
            if not content.startswith(("http://", "https://")):
                content = "https://" + content
            urls.append(content)

    elif action_type == "download":
        url_part = content.split("|", 1)[0].strip()
        if url_part:
            if not url_part.startswith(("http://", "https://")):
                url_part = "https://" + url_part
            urls.append(url_part)

    elif action_type in ("shell", "python"):
        urls.extend(_URL_RE.findall(content))

    return urls


def check_url_egress(url: str) -> str | None:
    """Check if a URL is allowed by the egress policy.

    Returns error message if denied, None if allowed.
    """
    mode = get_egress_mode()

    if mode == EgressMode.OPEN:
        return None

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
    except Exception:
        return f"cannot parse URL: {url[:80]}"

    if not hostname:
        return f"no hostname in URL: {url[:80]}"

    if mode == EgressMode.ALLOWLIST:
        allowed = _load_domain_list("ARTIFEX_EGRESS_ALLOW")
        if not allowed:
            return "egress allowlist is empty — all URLs denied"
        if not _domain_matches(hostname, allowed):
            return f"domain not in allowlist: {hostname}"

    elif mode == EgressMode.DENYLIST:
        denied = _load_domain_list("ARTIFEX_EGRESS_DENY")
        if _domain_matches(hostname, denied):
            return f"domain in denylist: {hostname}"

    return None


# ── Policy hook ──────────────────────────────────────────────────────────────

def _egress_hook(action_type: str, content: str, risk: RiskLevel):
    """Policy hook that checks URLs against the egress policy."""
    urls = extract_urls(action_type, content)
    for url in urls:
        reason = check_url_egress(url)
        if reason:
            return PolicyDecision(
                allowed=False,
                requires_confirmation=False,
                risk_level=risk,
                reason=f"egress policy: {reason}",
                matched_rule="egress_policy",
            )
    return None


def install():
    """Register the egress policy hook with the policy engine."""
    register_policy_hook(_egress_hook)
    _log.debug("Network egress policy installed")
