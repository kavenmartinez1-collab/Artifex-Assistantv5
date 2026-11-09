"""
Artifex Assistant V5 — Filesystem sandbox (P6-T23).

Restricts agent file access to an allowed directory tree. By default the
project directory and system temp dirs are allowed; everything else is denied.

Hooks into the policy engine so file-touching actions (read_file, edit_file,
shell, python, download, glob, grep) are path-checked before execution.

Configured via:
  ARTIFEX_SANDBOX_ROOTS  — colon-separated additional allowed roots, or the
                           single value "*" for OPEN mode: every resolvable
                           path is in scope. Open mode is for a single-user
                           box where the assistant is meant to work across
                           the whole machine (read any repo, build anywhere);
                           the deny list below still applies.
  ARTIFEX_SANDBOX_DENY   — colon-separated deny patterns (checked first)

Open mode widens SCOPE only. The sensitive-path deny list below is not
configurable off: ssh/gpg keys, cloud credentials, .env files and the
Windows SAM stay unreadable however wide the roots are. "Let it read the
whole machine" and "let it read the key material on the machine" are
different requests, and only the first one has a good answer.
"""

import logging
import os
import re
import tempfile
from pathlib import PurePosixPath, PureWindowsPath

from core.sandbox.policy import (
    PolicyDecision,
    RiskLevel,
    register_policy_hook,
)

_log = logging.getLogger(__name__)

# ── Sensitive paths that are always denied ───────────────────────────────────

_ALWAYS_DENY = [
    re.compile(r"[/\\]\.ssh[/\\]?", re.IGNORECASE),
    re.compile(r"[/\\]\.gnupg[/\\]?", re.IGNORECASE),
    re.compile(r"[/\\]\.aws[/\\]?", re.IGNORECASE),
    re.compile(r"[/\\]\.azure[/\\]?", re.IGNORECASE),
    re.compile(r"[/\\]\.gcloud[/\\]?", re.IGNORECASE),
    re.compile(r"[/\\]\.kube[/\\]?", re.IGNORECASE),
    re.compile(r"[/\\]\.docker[/\\]config\.json", re.IGNORECASE),
    re.compile(r"[/\\]\.npmrc$", re.IGNORECASE),
    re.compile(r"[/\\]\.pypirc$", re.IGNORECASE),
    re.compile(r"[/\\]\.netrc$", re.IGNORECASE),
    re.compile(r"[/\\]\.env$", re.IGNORECASE),
    re.compile(r"[/\\]\.env\.local$", re.IGNORECASE),
    re.compile(r"[/\\]credentials\.json$", re.IGNORECASE),
    re.compile(r"[/\\]service.account\.json$", re.IGNORECASE),
    re.compile(r"[/\\]id_rsa$", re.IGNORECASE),
    re.compile(r"[/\\]id_ed25519$", re.IGNORECASE),
    re.compile(r"[/\\]\.git[/\\]config$", re.IGNORECASE),
    re.compile(r"\\Windows\\System32\\config\\", re.IGNORECASE),
    re.compile(r"[/\\]etc[/\\]shadow$"),
    re.compile(r"[/\\]etc[/\\]passwd$"),
]

_USER_DENY: list[re.Pattern] = []


def _load_user_deny():
    """Load user-specified deny patterns from ARTIFEX_SANDBOX_DENY."""
    raw = os.environ.get("ARTIFEX_SANDBOX_DENY", "")
    if not raw:
        return
    _USER_DENY.clear()
    for part in raw.split(os.pathsep):
        part = part.strip()
        if part:
            try:
                _USER_DENY.append(re.compile(re.escape(part), re.IGNORECASE))
            except re.error:
                _log.warning("Invalid deny pattern: %s", part)


_load_user_deny()


def is_path_denied(path: str) -> str | None:
    """Check if a path matches any deny pattern.

    Returns the matched pattern description, or None if allowed.
    """
    for pat in _ALWAYS_DENY:
        if pat.search(path):
            return f"sensitive path: {pat.pattern}"
    for pat in _USER_DENY:
        if pat.search(path):
            return f"user-denied: {pat.pattern}"
    return None


def path_is_sensitive(path: str) -> bool:
    """True if `path`, as written or once resolved, is on the deny list.

    For tools that reach files the policy hook never saw by name — the files
    a glob expands to, the files a directory walk finds. Scope is not checked
    here, only the key-material deny list.
    """
    if is_path_denied(path):
        return True
    resolved = _resolve_path(path)
    return bool(resolved and resolved != path and is_path_denied(resolved))


def split_grep_content(content: str) -> tuple[str, str, str]:
    """(pattern, path, flags) from a grep action's "pattern|path|flags".

    Counted from the RIGHT. The pattern is a regex, `a|b` is its commonest
    construct, and every producer writes all three fields; counted from the
    left, "foo|bar|src|" made "bar" the path. The executor and the sandbox
    check must agree on which field is the path, so both use this.
    """
    parts = content.rsplit("|", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return content, "", ""


_GLOB_MAGIC = re.compile(r"[*?\[]")


def _glob_static_prefix(pattern: str) -> str:
    """The directory part of a glob pattern that names a fixed location.

    "../../*" -> "../..", "/srv/data/*.csv" -> "/srv/data", "**/*.py" -> "".
    The last component is the file-name part and is never included.
    """
    comps = re.split(r"[/\\]", pattern)
    lead = []
    for comp in comps[:-1]:
        if _GLOB_MAGIC.search(comp):
            break
        lead.append(comp)
    if not lead:
        return ""
    if lead == [""]:
        return "/"
    return "/".join(lead)


# ── Allowed root directories ─────────────────────────────────────────────────

OPEN_ROOTS_SENTINEL = "*"


def is_open_scope() -> bool:
    """True when ARTIFEX_SANDBOX_ROOTS is the open-mode sentinel.

    Read live rather than cached at import: the API process sets this from
    the launcher's environment, and tests flip it per-case.
    """
    return os.environ.get(
        "ARTIFEX_SANDBOX_ROOTS", "").strip() == OPEN_ROOTS_SENTINEL


def _get_allowed_roots() -> list[str]:
    """Build the list of allowed root directories."""
    roots = []

    from core.config import BASE_DIR
    roots.append(os.path.abspath(BASE_DIR))

    roots.append(os.path.abspath(tempfile.gettempdir()))

    extra = os.environ.get("ARTIFEX_SANDBOX_ROOTS", "")
    if extra and extra.strip() != OPEN_ROOTS_SENTINEL:
        for part in extra.split(os.pathsep):
            part = part.strip()
            if part and os.path.isabs(part):
                roots.append(os.path.abspath(part))

    return roots


def _resolve_path(path: str) -> str | None:
    """Expand ~, make absolute, and resolve symlinks.

    Returns None if the path can't be resolved. realpath follows symlinks so
    a link inside the sandbox pointing to /etc/passwd resolves to /etc/passwd
    and gets caught by both the deny list and the sandbox-root check.
    Non-existent path components are kept verbatim, so this also works for
    files we're about to create.
    """
    try:
        expanded = os.path.expanduser(path)
        return os.path.realpath(os.path.abspath(expanded))
    except (ValueError, OSError):
        return None


def _is_under_root(abs_path: str, root: str) -> bool:
    """True iff abs_path == root or abs_path is a child of root.

    Plain startswith() is wrong because /x/projEVIL starts with /x/proj.
    Compare with a trailing separator (or exact equality) to avoid that.
    """
    if abs_path == root:
        return True
    return abs_path.startswith(root + os.sep)


def is_path_within_sandbox(path: str) -> bool:
    """Check if a path is within any allowed root directory.

    Resolves symlinks before the check so a link inside the sandbox pointing
    outside is caught. Uses an exact-or-child check rather than a raw prefix
    match so /x/projEVIL doesn't slip past a /x/proj root.

    In open scope (ARTIFEX_SANDBOX_ROOTS="*") every resolvable path is in
    scope. A path that will not resolve at all is still rejected — that is a
    malformed path, not a permission question.
    """
    abs_path = _resolve_path(path)
    if abs_path is None:
        return False

    if is_open_scope():
        return True

    for root in _get_allowed_roots():
        resolved_root = os.path.realpath(root)
        if _is_under_root(abs_path, resolved_root):
            return True
    return False


def check_path(path: str) -> str | None:
    """Full path check: deny list first, then sandbox roots.

    Returns an error message if denied, None if allowed. The deny list is
    checked against both the raw input (so it catches obvious cases like
    "/etc/passwd") and the symlink-resolved path (so it catches a sandboxed
    symlink that points at a sensitive target).
    """
    deny_reason = is_path_denied(path)
    if deny_reason:
        return deny_reason

    resolved = _resolve_path(path)
    if resolved is not None and resolved != path:
        deny_reason = is_path_denied(resolved)
        if deny_reason:
            return f"{deny_reason} (via symlink)"

    if not is_path_within_sandbox(path):
        return f"outside sandbox: {path}"
    return None


# ── Path extraction from action content ──────────────────────────────────────

# A candidate starts at the START of a token. The pattern used to take any
# "/" wherever it stood, so `hits/total` gave "/total", `src/app.py` gave
# "/app.py" and an https URL gave "s://host/..." (a path on drive S:) — all
# "outside the sandbox", which blocked ordinary code in full-auto and
# prompted for it in guided. A relative `name/rest` is now kept whole and
# resolved against the cwd like any relative path; rooted forms are found
# as before, including after `=`, quotes and brackets.
_PATH_LIKE = re.compile(
    r'(?:(?<![A-Za-z0-9_])[A-Za-z]:[/\\]'       # C:\ ...
    r'|(?<!\w)[/\\]'                             # /rooted, \rooted
    r'|(?<![\w.])\.{1,2}[/\\]'                   # ./  ../
    r'|(?<![\w.\-~$%{}])[\w.\-~$%{}]+[/\\])'    # name/rest (relative)
    r'[^\s:*?"<>|\x00-\x1f]+'
)

# Remote URLs are not local paths (file: URLs are, and are left in).
_REMOTE_URL = re.compile(r"\b(?:https?|ftp|wss?)://[^\s'\"<>]+", re.IGNORECASE)
_CLOSING_TAG = re.compile(r"/[A-Za-z][\w:-]*")


def _path_candidates(content: str) -> list[str]:
    """Path-like tokens in shell/python text, in order, without repeats."""
    text = _REMOTE_URL.sub(" ", content)
    out, seen = [], set()
    for m in _PATH_LIKE.finditer(text):
        cand = m.group(0)
        # "</div>" is markup, not the path /div. A redirect such as
        # `cat </etc/passwd` has no ">" right after and is kept.
        if (m.start() > 0 and text[m.start() - 1] == "<"
                and text[m.end():m.end() + 1] == ">"
                and _CLOSING_TAG.fullmatch(cand)):
            continue
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out

# Leading escape sequences in shell/python string literals ("hi\n") produce
# bogus candidates: `printf 'hi\n' > out.txt` extracts "\n'" as a path, which
# abspath()s to <drive root>\n' and is rejected as "outside sandbox" —
# blocking the single most common way an agent writes a file. Only candidates
# BEGINNING with a bare-backslash escape are affected; drive-prefixed and
# dot-relative candidates are kept verbatim, since stripping escapes out of
# those breaks the drive prefix and lets real paths whose components start
# with escape letters (C:\temp\notes.txt) evade the check entirely.
_LEADING_ESCAPES = re.compile(
    r'^(?:\\(?:[ntrbfva0]|x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}))+'
)


def _filter_escape_candidates(candidates: list[str]) -> list[str]:
    """Drop or trim path candidates that are really string-literal escapes.

    A candidate starting with an escape sequence ("\\n'") is junk unless a
    genuine path tail follows the escapes ("\\n\\.ssh\\id_rsa" keeps
    "\\.ssh\\id_rsa" so the deny list still sees it).
    """
    kept = []
    for cand in candidates:
        if cand.startswith("\\"):
            remainder = _LEADING_ESCAPES.sub("", cand)
            if remainder != cand:
                if remainder.startswith(("\\", "/")):
                    kept.append(remainder)
                continue
        kept.append(cand)
    return kept


def extract_paths_from_content(action_type: str, content: str) -> list[str]:
    """Extract file paths from action content for sandbox checking."""
    paths = []

    if action_type == "read_file":
        path = content.rsplit("|", 1)[0].strip()
        if path and not path.startswith(("http://", "https://")):
            paths.append(path)

    elif action_type == "read_function":
        path = content.split("|", 1)[0].strip()
        if path:
            paths.append(path)

    elif action_type == "edit_file":
        parts = content.split("\x00", 2)
        if parts:
            paths.append(parts[0].strip())

    elif action_type == "glob":
        parts = content.split("|")
        base = parts[1].strip() if len(parts) > 1 else ""
        if base:
            paths.append(base)
        # The pattern can leave the base on its own ("../../*", an absolute
        # "/srv/*"); only the base used to be checked.
        prefix = _glob_static_prefix(parts[0].strip())
        if prefix:
            paths.append(os.path.join(base, prefix) if base else prefix)

    elif action_type == "grep":
        _pattern, path, _flags = split_grep_content(content)
        if "|" in content:
            paths.append(path.strip())

    elif action_type in ("shell", "python"):
        paths.extend(_filter_escape_candidates(_path_candidates(content)))

    elif action_type == "download":
        parts = content.split("|", 1)
        if len(parts) > 1:
            paths.append(parts[1].strip())

    elif action_type == "view_image":
        path = content.split("|", 1)[0].strip()
        if path:
            paths.append(path)

    elif action_type == "describe_images":
        # "folder|catalog|prompt" — the prompt is free text, not a path.
        parts = content.split("|", 2)
        paths.extend(p.strip() for p in parts[:2] if p.strip())

    return paths


# ── Policy hook ──────────────────────────────────────────────────────────────

def _fs_sandbox_hook(action_type: str, content: str, risk: RiskLevel):
    """Policy hook that checks file paths against the sandbox.

    Two distinct rule tags so hosts can treat them differently:
    - "fs_sandbox"       — deny-list hit (credentials, keys, system files).
      Absolute; never promptable.
    - "fs_sandbox_scope" — path is merely outside the allowed roots. A
      supervised loop (guided/manual) may surface this as an approval
      prompt instead of a wall; unattended (full-auto) hosts keep it hard.
    """
    paths = extract_paths_from_content(action_type, content)
    for path in paths:
        reason = check_path(path)
        if reason:
            scope_only = reason.startswith("outside sandbox:")
            return PolicyDecision(
                allowed=False,
                requires_confirmation=False,
                risk_level=risk,
                reason=f"filesystem sandbox: {reason}",
                matched_rule="fs_sandbox_scope" if scope_only else "fs_sandbox",
            )
    return None


def install():
    """Register the filesystem sandbox hook with the policy engine."""
    register_policy_hook(_fs_sandbox_hook)
    if is_open_scope():
        # WARNING, not debug: "the agent can touch the whole filesystem" is
        # not something anyone should have to infer from an env var they
        # cannot see. One line, at startup, in the log everyone greps first.
        _log.warning(
            "Filesystem sandbox: OPEN SCOPE (ARTIFEX_SANDBOX_ROOTS=%s) — "
            "every path on this machine is in scope for agent and chat "
            "tools. The sensitive-path deny list (ssh/gpg keys, cloud "
            "credentials, .env, SAM) is still enforced.",
            OPEN_ROOTS_SENTINEL)
    else:
        _log.debug("Filesystem sandbox installed (roots: %s)",
                   ", ".join(_get_allowed_roots()))
