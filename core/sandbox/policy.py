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
    "view_image":       RiskLevel.SAFE,
    "search":           RiskLevel.LOW,
    "web_read":         RiskLevel.LOW,
    "edit_file":        RiskLevel.MEDIUM,
    "describe_images":  RiskLevel.MEDIUM,   # appends to a catalog file
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
    # Windows equivalents. Shell actions on Windows run in PowerShell (or cmd
    # via `cmd /c`), and the Unix-only patterns above let the commands a
    # Windows model actually writes through as merely HIGH.
    # PowerShell accepts any unambiguous parameter prefix: -r = -Recurse,
    # -fo = -Force (-f alone is ambiguous with -Filter, so it never runs).
    # (?<!-) keeps the `ri` alias from matching flags like `grep -ri`.
    re.compile(r"(?<!-)\b(remove-item|ri|rmdir|rd|del|erase)\b.*\s-r[a-z]*\b", re.IGNORECASE),
    re.compile(r"(?<!-)\b(remove-item|ri|del|erase)\b.*\s-fo[a-z]*\b", re.IGNORECASE),
    re.compile(r"(?<!-)\b(rmdir|rd|del|erase)\b.*\s/[sfq]\b", re.IGNORECASE),
    re.compile(r"\b(format-volume|clear-disk|initialize-disk|diskpart)\b", re.IGNORECASE),
    re.compile(r"\b(stop|restart)-computer\b", re.IGNORECASE),
    re.compile(r"\b(remove-item|remove-itemproperty|set-itemproperty|new-itemproperty)\b.*\bhk(lm|cu|cr|u|cc):", re.IGNORECASE),
    re.compile(r"\b(iwr|irm|invoke-webrequest|invoke-restmethod|downloadstring)\b.*\|\s*(iex|invoke-expression)\b", re.IGNORECASE),
    re.compile(r"\b(iex|invoke-expression)\b.*\b(iwr|irm|invoke-webrequest|invoke-restmethod|downloadstring)\b", re.IGNORECASE),
    re.compile(r"\bset-mppreference\b.*\s-disable", re.IGNORECASE),
    # Destructive git spellings the --force / --hard patterns above miss.
    re.compile(r"\bgit\s+push\b.*\s-f\b", re.IGNORECASE),
    re.compile(r"\bgit\s+push\b.*\s\+\S", re.IGNORECASE),
    re.compile(r"\bgit\s+clean\b.*\s-[a-z]*f", re.IGNORECASE),
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

# PowerShell reads. Shell blocks on Windows run in PowerShell, so before
# this existed every command the model actually wrote — Get-Command,
# Test-Path, Get-ChildItem | Select-Object — classified HIGH and needed a
# phone approval (run 6024eea3124b: approve `Get-Command claude`).
# Verb-based: Get/Test/Select/... cmdlets read and never write. ForEach-
# Object is deliberately absent — `gci | % Delete` deletes every file.
_PS_SAFE_HEAD = re.compile(
    r"^\s*(?:(?:get|test|select|where|sort|measure|format|group|compare"
    r"|resolve|split|join|convertto|convertfrom)-[a-z]+"
    r"|out-string|write-(?:output|host)"
    r"|gci|gc|gi|gp|gps|gsv|gcm|gm|gl|gal|gv|gdr|gcim|gwmi|sls|select"
    r"|sort|measure|ft|fl|fw|group|compare|rvpa|\?)(?=\s|$)",
    re.IGNORECASE)

_SAFE_SHELL_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*(ls|dir|pwd|cd|echo|cat|head|tail|type|wc|find|which|where)\b"),
    re.compile(r"^\s*(git\s+(status|log|diff|show|branch\s*$))\b"),
    re.compile(r"^\s*python\s+--version"),
    re.compile(r"^\s*(node|npm|pip)\s+--version"),
    _PS_SAFE_HEAD,
]

# Read-verb cmdlets that still are not safe to auto-run.
_UNSAFE_READ_CMDLETS = re.compile(r"^\s*get-credential\b", re.IGNORECASE)

# The SAFE patterns only look at how each command starts, so anything that
# can run or write something else, from inside an otherwise-safe command,
# disqualifies it: `echo x > file`, `cat $(...)`, `find . -delete`,
# `[IO.File]::Delete(...)`, `(gi x).Delete()`. Chains (`;`, `&&`, `|`,
# newlines) are split by split_shell_segments and every piece is judged on
# its own, so `ls && python x.py` is unsafe because `python x.py` is.
_SAFE_DISQUALIFIERS = re.compile(
    r"[`>]|\$\(|::|\.[A-Za-z_]\w*\s*\(|\s-(exec|execdir|ok|delete)\b",
    re.IGNORECASE)

# Words that run or change things when they appear inside a script block or
# subexpression of an otherwise-safe command (`gci | ? { rm $_ }`).
_DESTRUCTIVE_WORDS = frozenset({
    "rm", "del", "erase", "rd", "rmdir", "ri", "mv", "move", "mi", "cp", "copy",
    "cpi", "ni", "md", "mkdir", "sc", "ac", "si", "sp", "set", "sv", "nv", "rv",
    "kill", "spps", "saps", "start", "iex", "icm", "ii", "rni", "ren", "clc",
    "cli", "clp", "tee", "curl", "wget", "iwr", "irm", "ipmo", "sajb", "nal",
    "epal", "ipal", "powershell", "pwsh", "cmd", "python", "python3", "py",
    "node", "bash", "sh", "foreach", "%", "taskkill", "shutdown", "reg",
})
_VERB_NOUN_RE = re.compile(r"^[A-Za-z]+-[A-Za-z]+$")
_QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"")


def split_shell_segments(command: str) -> list[str]:
    """Split a command line or script into the simple commands it runs.

    Separators are newlines, `;`, `|`, `||`, `&&` and `&`, but only outside
    quotes and outside {...}/(...) — a script block or subexpression stays
    inside its command and is judged with it. Blank lines and `#` comment
    lines are dropped. A bare `&` (PowerShell's call operator, or bash's
    background) also splits, which leaves the called thing as its own
    segment for the caller to judge. Unbalanced quotes are not an error:
    the remainder just lands in one segment, which then fails any
    allow-check on its own, so a bad split errs toward asking.
    """
    segs: list[str] = []
    buf: list[str] = []
    depth = 0
    quote = ""
    i, n = 0, len(command)
    while i < n:
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch in "{(":
            depth += 1
            buf.append(ch)
        elif ch in "})":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif depth == 0 and ch in "\n;|&":
            segs.append("".join(buf))
            buf = []
            if ch in "|&" and i + 1 < n and command[i + 1] == ch:
                i += 1  # || or &&
        else:
            buf.append(ch)
        i += 1
    segs.append("".join(buf))
    return [s.strip() for s in segs
            if s.strip() and not s.strip().startswith("#")]


def inner_regions_safe(segment: str) -> bool:
    """True if nothing inside the segment's {...} / (...) runs or writes.

    Every Verb-Noun cmdlet in there must itself be a safe read, and no
    destructive alias may appear as a bare word. Quoted strings are ignored
    (text, not commands).
    """
    if "{" not in segment and "(" not in segment:
        return True
    text = _QUOTED_RE.sub(" ", segment)
    for region in re.findall(r"[{(]([^{}()]*)", text):
        for tok in re.findall(r"(?<![\w$.\-:\\/])([A-Za-z%][\w-]*)", region):
            if _VERB_NOUN_RE.match(tok):
                if (not _PS_SAFE_HEAD.match(tok)
                        or _UNSAFE_READ_CMDLETS.match(tok)):
                    return False
            elif tok.lower() in _DESTRUCTIVE_WORDS:
                return False
    return True


# `$files = Get-ChildItem`: a variable assignment is as safe as what it
# assigns. Variables die with the one-shot process, so even `$env:X = ...`
# changes nothing outside it.
_PS_ASSIGN_RE = re.compile(r"^\s*\$[A-Za-z_][\w:]*\s*[+\-]?=\s*")
# Also a whole line on its own: PowerShell prints a bare value, so
# `"Count: $n"` is an output statement. `$var` interpolation inside "..."
# only reads; a `$(command)` in there was already disqualified upstream.
_LITERAL_RE = re.compile(r"""^\s*(?:-?\d+(?:\.\d+)?|'[^']*'|"[^"]*"|\$(?:true|false|null))\s*$""",
                         re.IGNORECASE)


# A line that only outputs a variable or its properties: `$files.Name`,
# `$items[0].FullName`. No parentheses allowed, so no method calls.
_VAR_EXPR_RE = re.compile(r"^\s*\$[A-Za-z_][\w:]*(?:\.[A-Za-z_]\w*|\[[^\]()]*\])*\s*$")
# `"Count: $($files.Count)"` — a subexpression that only READS a variable
# or property. Neutralized before the `$(` disqualifier sees it; anything
# else inside `$(...)` (a command, a method call) still disqualifies.
_PROPERTY_SUBEXPR_RE = re.compile(
    r"\$\(\s*\$[A-Za-z_][\w:]*(?:\.[A-Za-z_]\w*|\[\d+\])*\s*\)")


# Syntax only PowerShell writes. The executor (tools.agent_tools) uses this
# same test to pick the shell — PowerShell wins over the bash heuristics —
# so the PowerShell-only idioms below are judged safe exactly when the
# script really runs in PowerShell. In bash a bare `"..."` or `$VAR` line
# EXECUTES its value as a command.
PS_SYNTAX_RE = re.compile(
    r"\b(?:Get|Set|New|Remove|Add|Start|Stop|Test|Invoke|Select|Where|ForEach"
    r"|Out|Write|Format|Measure|Sort|Resolve|Join|Split|Import|Export"
    r"|ConvertTo|ConvertFrom|Copy|Move|Rename|Clear|Wait|Restart|Register"
    r"|Unregister|Enable|Disable|Install|Uninstall|Update|Expand|Compress"
    r"|Read|Show|Push|Pop|Group|Compare|Tee)-[A-Z][A-Za-z]+\b"
    r"|\[[A-Za-z_][\w.]*\]::"            # [Type]::Member
    r"|@[\"']\s*$"                        # here-string opener
    r"|^\s*\$[A-Za-z_][\w:]*\s*[+\-]?=(?!=)"  # $var = ...
    r"|-ComObject\b|\$env:|\$_\b|\$PSScriptRoot\b",
    re.MULTILINE)


def looks_powershell(command: str) -> bool:
    return bool(PS_SYNTAX_RE.search(command))


def _is_safe_segment(segment: str, powershell: bool = False) -> bool:
    if powershell:
        if _VAR_EXPR_RE.match(segment) or _LITERAL_RE.match(segment):
            return True
    m = _PS_ASSIGN_RE.match(segment) if powershell else None
    if m:
        segment = segment[m.end():]
        if not segment.strip() or _LITERAL_RE.match(segment):
            return True
    if _UNSAFE_READ_CMDLETS.match(segment):
        return False
    if not any(pat.search(segment) for pat in _SAFE_SHELL_PATTERNS):
        return False
    return inner_regions_safe(segment)


def _is_safe_shell(command: str) -> bool:
    powershell = looks_powershell(command)
    if powershell:
        command = _PROPERTY_SUBEXPR_RE.sub("$v", command)
    if _SAFE_DISQUALIFIERS.search(command):
        return False
    segments = split_shell_segments(command)
    return bool(segments) and all(_is_safe_segment(s, powershell) for s in segments)


def classify_shell_risk(command: str) -> RiskLevel:
    """Classify a shell command into a risk level based on content analysis."""
    for pat in _CRITICAL_SHELL_PATTERNS:
        if pat.search(command):
            return RiskLevel.CRITICAL
    if _is_safe_shell(command):
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
