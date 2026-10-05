"""
Artifex Assistant V5 — Remembered approvals ("Always" for the rest of a run).

When the user answers an approval prompt with "always", the action is turned
into a rule and later actions the rule covers run without asking again —
modeled on Claude Code's "Yes, and don't ask again", but scoped to ONE agent
run and held in memory only (nothing is written to disk; the rules go away
with the run).

What a rule covers:
  shell     — a command PREFIX for ordinary commands (`git commit`, `Set-Volume`,
              `docker`), matched per simple command, so `git commit ... ; rm x`
              is not covered just because it starts with `git commit`. Wrappers
              that run arbitrary other code (powershell, python, Start-Process,
              iex, ...) and destructive heads (Remove-*, rm, del, kill, ...) only
              ever remember the EXACT command. Multi-line scripts likewise.
  edit_file — that one file path.
  download  — that host.
  anything else — the exact same action.

Never covered: CRITICAL-risk actions (they ask every time, whatever was
approved before), and commands containing substitutions or redirects that
could smuggle a second command past a prefix (`$(...)`, backticks, `>`).
"""

import hashlib
import os
import re
import threading
from urllib.parse import urlparse

from core.sandbox.policy import (
    RiskLevel, classify_shell_risk, inner_regions_safe, split_shell_segments,
)

# Heads that run other code given as an argument: approving one invocation
# must never approve the next, different one.
_WRAPPER_HEADS = frozenset({
    "powershell", "pwsh", "cmd", "bash", "sh", "zsh", "wsl", "python",
    "python3", "py", "node", "deno", "ruby", "perl", "start-process", "saps",
    "start", "invoke-expression", "iex", "invoke-command", "icm", "&", ".",
    "sudo", "runas", "env", "xargs", "call", "npx", "uvx", "cscript",
    "wscript", "mshta", "rundll32", "regsvr32", "schtasks", "at",
})
# Heads that destroy or kill: remembered exactly, never by prefix.
_DESTRUCTIVE_HEAD_RE = re.compile(
    r"^(?:(?:remove|clear|stop|uninstall|disable|reset|unregister|format)-"
    r"|(?:rm|del|erase|ri|rd|rmdir|kill|taskkill|spps|mv|move|mi|reg|diskpart"
    r"|shutdown)$)",
    re.IGNORECASE)
# Tools whose second word is a subcommand worth keeping in the prefix.
_SUBCOMMAND_TOOLS = frozenset({
    "git", "npm", "pnpm", "yarn", "pip", "uv", "docker", "kubectl", "gh",
    "winget", "choco", "scoop", "cargo", "dotnet", "go", "claude", "conda",
    "az", "gcloud", "aws", "tailscale", "ollama", "systemctl", "brew",
})
# Things that can hide a second command inside one that starts innocently.
_SMUGGLE_RE = re.compile(r"\$\(|`|>|::")


def _tokens(segment: str) -> list[str]:
    return segment.strip().split()


def _head(segment: str) -> str:
    toks = _tokens(segment)
    if not toks:
        return ""
    head = toks[0].lower().strip("\"'")
    head = os.path.basename(head.replace("\\", "/"))
    return head[:-4] if head.endswith(".exe") else head


def _prefix(segment: str) -> str:
    toks = _tokens(segment)
    head = _head(segment)
    if (head in _SUBCOMMAND_TOOLS and len(toks) > 1
            and re.match(r"^[a-z][a-z0-9_.:-]*$", toks[1], re.IGNORECASE)):
        return f"{head} {toks[1].lower()}"
    return head


def _prefix_ok(segment: str) -> bool:
    """Whether this simple command may be remembered/matched by prefix."""
    head = _head(segment)
    return bool(head) and head not in _WRAPPER_HEADS \
        and not _DESTRUCTIVE_HEAD_RE.match(head)


def _exact_key(action) -> str:
    digest = hashlib.sha1(action.content.encode("utf-8", "replace")).hexdigest()
    return f"{action.type}:exact:{digest}"


class ApprovalMemory:
    """Approval rules the user granted during one run. Thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self._rules: dict[str, str] = {}   # key -> human-readable description

    # ── public ────────────────────────────────────────────────────────────
    def remember(self, action) -> list[str]:
        """Record rules covering `action`; returns their descriptions."""
        added = {}
        if action.type == "shell":
            added = self._shell_rules(action)
        elif action.type == "edit_file":
            path = self._edit_path(action)
            if path:
                added = {f"edit:{path}": f"edits to {path}"}
        elif action.type == "download":
            host = self._host(action)
            if host:
                added = {f"download:{host}": f"downloads from {host}"}
        if not added:
            added = {_exact_key(action): f"this exact {action.type} action"}
        with self._lock:
            self._rules.update(added)
        return list(added.values())

    def covers(self, action, decision) -> str | None:
        """Description of the rule that pre-approves `action`, or None."""
        risk = getattr(decision, "risk_level", None)
        if risk is None or risk >= RiskLevel.CRITICAL:
            return None
        with self._lock:
            rules = dict(self._rules)
        if not rules:
            return None
        exact = rules.get(_exact_key(action))
        if exact:
            return exact
        if action.type == "shell":
            return self._shell_covered(action.content, rules)
        if action.type == "edit_file":
            path = self._edit_path(action)
            return rules.get(f"edit:{path}") if path else None
        if action.type == "download":
            host = self._host(action)
            return rules.get(f"download:{host}") if host else None
        return None

    def descriptions(self) -> list[str]:
        with self._lock:
            return list(self._rules.values())

    # ── shell ─────────────────────────────────────────────────────────────
    @staticmethod
    def _prefixable(command: str) -> bool:
        return ("\n" not in command.strip() and not _SMUGGLE_RE.search(command))

    def _shell_rules(self, action) -> dict:
        command = action.content
        if not self._prefixable(command):
            return {}
        rules = {}
        for seg in split_shell_segments(command):
            if classify_shell_risk(seg) == RiskLevel.SAFE:
                continue
            if not _prefix_ok(seg) or not inner_regions_safe(seg):
                return {}   # fall back to the exact command
            p = _prefix(seg)
            rules[f"shell:prefix:{p}"] = f"commands starting `{p}`"
        return rules

    def _shell_covered(self, command: str, rules: dict) -> str | None:
        if not self._prefixable(command):
            return None
        segments = split_shell_segments(command)
        if not segments:
            return None
        matched = []
        for seg in segments:
            if classify_shell_risk(seg) == RiskLevel.SAFE:
                continue
            if not _prefix_ok(seg) or not inner_regions_safe(seg):
                return None
            rule = rules.get(f"shell:prefix:{_prefix(seg)}")
            if not rule:
                return None
            matched.append(rule)
        return ", ".join(dict.fromkeys(matched)) if matched else None

    # ── helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _edit_path(action) -> str:
        path = action.content.split("\x00", 1)[0].strip()
        return os.path.normcase(os.path.abspath(path)) if path else ""

    @staticmethod
    def _host(action) -> str:
        url = action.content.split("|", 1)[0].strip()
        return (urlparse(url).hostname or "").lower()
