"""
Artifex API — Web search tool extraction and execution.

Self-contained module for @search and @web_read tool handling in the API layer.
Only exposes web search capabilities — no shell, python, or filesystem tools.
Communicates with the web-gateway Docker container for content sanitization.
"""

import json
import logging
import os
import re
import urllib.request
import urllib.error

_log = logging.getLogger("artifex.api.web_tools")

# ── Gateway configuration ────────────────────────────────────────────────────

# Default empty — requires explicit configuration to prevent accidentally
# connecting to unknown services on port 8080.
# Set via: $env:WEB_GATEWAY_URL = "http://localhost:8080" (PowerShell)
#     or:  export WEB_GATEWAY_URL=http://localhost:8080    (bash)
WEB_GATEWAY_URL = os.getenv("WEB_GATEWAY_URL", "")
GATEWAY_AUTH_TOKEN = os.getenv("GATEWAY_AUTH_TOKEN", "")

MAX_TOOL_ROUNDS = 10

# Max chars of extracted page text forwarded to the model per @web_read.
# 4000 was the 8K-ctx-era value; at modern ctx windows a fuller page beats
# a lossy cut, and this path DROPS the tail (no cache to drill into).
WEB_READ_MAX_CHARS = int(os.getenv("ARTIFEX_WEB_READ_MAX_CHARS", "12000"))

# ── Per-request guards (2026-10-25) ──────────────────────────────────────────
# Chat chatcmpl-7b4b64da: after a 32K-char CVE search the model degenerated
# into 44 tool calls in one turn (the same five @web_read refs ~10x each plus
# @search(...) with a literal "..." query). Every call ran, the sites answered
# HTTP 429, 304K chars of results went back, and the next prompt hit ~80K
# tokens. The guards mirror Anthropic's server tools: `max_uses` per tool per
# request (the over-limit call gets an error result and the request goes on),
# web_fetch's result cache (a URL is fetched once per request), a cap on what
# one call/turn may add to context, and a bound on parallel calls per turn.
# Duplicate detection is the harness's job there too (not done by the API).
WEB_SEARCH_MAX_USES = int(os.getenv("ARTIFEX_WEB_SEARCH_MAX_USES", "6"))
WEB_READ_MAX_USES = int(os.getenv("ARTIFEX_WEB_READ_MAX_USES", "10"))
WEB_TOOLS_MAX_PER_ROUND = int(os.getenv("ARTIFEX_WEB_TOOLS_MAX_PER_ROUND", "8"))
WEB_ROUND_MAX_CHARS = int(os.getenv("ARTIFEX_WEB_ROUND_MAX_CHARS", "48000"))

# Arguments a model writes when it is describing the syntax rather than
# calling it: @search(...), @search(query), @web_read(N), @web_read(<url>).
_PLACEHOLDER_ARGS = {"", "...", "…", "query", "<query>", "q", "n", "<n>",
                     "url", "<url>", "your query", "search query"}

# ── Tool-call regex patterns ─────────────────────────────────────────────────

# Match @search("query") with regular, smart, or NO quotes
# Models sometimes omit quotes: @search(some query here)
_SEARCH_PATTERN = re.compile(
    r'@search\(\s*["\u201c\u201d]?([^"\u201c\u201d)]+?)["\u201c\u201d]?\s*\)'
)
# Match @web_read(N) or @web_read("url") with regular or smart quotes
_WEB_READ_PATTERN = re.compile(
    r'@web_read\(\s*["\u201c\u201d]?([^"\u201c\u201d)]+)["\u201c\u201d]?\s*\)'
)

# Native function-calling format emitted by Qwen3.x and similar models:
#   <tool_call>{"name": "search", "arguments": {"query": "..."}}</tool_call>
# When llama-server is launched with --jinja, the model's own chat template
# trains it to emit tool calls as this JSON envelope rather than the @marker
# syntax. Parse it so a model that reverts to its native format still
# triggers tool execution.
_TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
    re.DOTALL,
)


def _extract_native_tool_calls(text: str) -> list[dict]:
    """Extract <tool_call>{...}</tool_call> JSON calls for search / web_read.

    Tolerates the naming a model picks for arguments (query/q, url/ref/n).
    Non-web tools are ignored here \u2014 only @search/@web_read run in the API.
    """
    tools = []
    for m in _TOOL_CALL_PATTERN.finditer(text):
        try:
            call = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = str(call.get("name", "")).lower()
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            continue
        if name in ("search", "web_search", "websearch"):
            query = args.get("query") or args.get("q") or args.get("text")
            if query:
                tools.append({"type": "search", "query": str(query),
                              "raw": m.group(0)})
        elif name in ("web_read", "webread", "fetch", "read_url", "open_url"):
            ref = (args.get("url") or args.get("ref")
                   or args.get("n") or args.get("index"))
            if ref is not None:
                tools.append({"type": "web_read", "ref": str(ref).strip(),
                              "raw": m.group(0)})
    return tools


def extract_web_tools(text: str) -> list[dict]:
    """Extract @search and @web_read tool calls from model response text.

    Recognizes both the @marker syntax and the native <tool_call> JSON
    envelope that Qwen3.x and other function-calling models emit.
    """
    tools = []
    for m in _SEARCH_PATTERN.finditer(text):
        tools.append({"type": "search", "query": m.group(1), "raw": m.group(0)})
    for m in _WEB_READ_PATTERN.finditer(text):
        tools.append({"type": "web_read", "ref": m.group(1).strip(), "raw": m.group(0)})
    tools.extend(_extract_native_tool_calls(text))
    return tools


# ── Gateway communication ────────────────────────────────────────────────────

def gateway_available() -> bool:
    """Check if the web gateway is reachable."""
    if not WEB_GATEWAY_URL:
        return False
    try:
        req = urllib.request.Request(f"{WEB_GATEWAY_URL}/health", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            ok = resp.status == 200
            if ok:
                _log.info("Web gateway connected at %s", WEB_GATEWAY_URL)
            return ok
    except Exception as e:
        _log.warning("Web gateway unavailable at %s: %s", WEB_GATEWAY_URL, e)
        return False


def gateway_post(endpoint: str, payload: dict) -> tuple[bool, dict | str]:
    """POST JSON to the web gateway. Returns (success, data) or (False, error)."""
    url = f"{WEB_GATEWAY_URL}{endpoint}"
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if GATEWAY_AUTH_TOKEN:
        headers["X-Gateway-Token"] = GATEWAY_AUTH_TOKEN
    req = urllib.request.Request(
        url, data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return True, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode("utf-8"))
            return False, err.get("detail", f"HTTP {e.code}")
        except Exception:
            return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)


# ── Tool execution ───────────────────────────────────────────────────────────

def _per_search_numbering() -> bool:
    return os.getenv("ARTIFEX_WEB_READ_PER_SEARCH_NUMBERING", "").strip().lower() in ("1", "true", "yes", "on")


def _run_search(tool: dict, search_cache: list[dict]) -> str:
    """One @search through the gateway; appends its results to search_cache."""
    _log.info("Executing web search via gateway: query='%s'", tool["query"])
    ok, data = gateway_post("/search", {"query": tool["query"], "max_results": 8})
    if ok:
        sr = data.get("results", [])
        _log.info("Search returned %d results for '%s'", len(sr), tool["query"])
        # Update search cache for @web_read(N) references.
        # 2026-09-30: append and keep numbering ([9]-[16] for the
        # second search) instead of clearing — a clear made N resolve
        # against whichever search ran last, so two searches in one
        # round fetched the wrong page (see tools/agent_tools.py,
        # _remember_search_results). Revert:
        # ARTIFEX_WEB_READ_PER_SEARCH_NUMBERING=1.
        if _per_search_numbering():
            search_cache.clear()
        first = len(search_cache) + 1
        search_cache.extend(sr)
        lines = []
        for i, r in enumerate(sr, first):
            lines.append(f"[{i}] {r.get('title', 'No title')}")
            lines.append(f"    {r.get('url', '')}")
            snippet = r.get("snippet", "")
            if snippet:
                lines.append(f"    {snippet[:400]}")
            lines.append("")
        lines.append("Use @web_read(N) to read the full page for result N.")
        result = "\n".join(lines)
    else:
        _log.warning("Search FAILED for '%s': %s", tool["query"], data)
        result = f"Search failed: {data}"
    return f"[Search results for: {tool['query']}]\n{result}"


def _resolve_read_url(ref: str, search_cache: list[dict]) -> tuple[str, str | None]:
    """(url, error) for a @web_read ref: a result number or a literal URL."""
    if ref.isdigit():
        idx = int(ref) - 1
        if idx < 0 or idx >= len(search_cache):
            if not search_cache:
                return "", "No search results cached. Use @search(\"query\") first."
            return "", f"Invalid result number {ref}. Valid results are 1-{len(search_cache)}."
        return search_cache[idx].get("url", ""), None
    return ref, None


def _run_web_read(url: str, max_chars: int = WEB_READ_MAX_CHARS) -> str:
    """Fetch one page through the gateway, cut to max_chars."""
    ok, data = gateway_post("/fetch", {"url": url})
    if not ok:
        _log.warning("web_read FAILED for %s: %s", url, data)
        return f"[Web page content]\nFetch failed: {data}"

    text = data.get("text", "")
    title = data.get("title") or url
    warnings = data.get("warnings", [])
    _log.info("web_read OK: %s — %d chars, title='%s'", url, len(text), title[:80])

    if not text:
        return "[Web page content]\n(Page returned no readable text content.)"

    header = f"Source: {title}\nURL: {data.get('url', url)}\n---"
    if warnings:
        header += f"\n[Security warnings: {'; '.join(warnings)}]"

    # Truncate very large pages to leave room in context. The tail
    # is DROPPED on this path (no cache file to drill into), so the
    # limit errs generous and the message is honest about the loss.
    if len(text) > max_chars:
        text = text[:max_chars] + (
            "\n\n[TRUNCATED — the page continues beyond this point "
            "and the rest is NOT available. If the answer may be in "
            "the missing part, say so rather than guessing.]"
        )
    return f"[Web page content]\n{header}\n{text}"


def new_request_state() -> dict:
    """Per-request bookkeeping for the guards in execute_web_tools().

    Create one per chat request (next to its search_cache) and pass it to
    every execute_web_tools() call of that request.
    """
    return {"search_uses": 0, "read_uses": 0, "round": 0,
            "searched": {}, "fetched": {}, "duplicates": 0, "refused": 0}


def _call_key(tool: dict) -> tuple[str, str]:
    arg = (tool.get("query") if tool["type"] == "search" else tool.get("ref")) or ""
    return tool["type"], " ".join(str(arg).split()).lower()


def execute_web_tools(tools: list[dict], search_cache: list[dict],
                      state: dict | None = None) -> str:
    """Execute a list of web tool calls and return combined results.

    Args:
        tools: List of tool dicts from extract_web_tools()
        search_cache: Mutable list of this request's search results, numbered
                      continuously across searches,
                      for @web_read(N) index lookups. Modified in-place.
        state: new_request_state() for this request. With it, the per-request
               guards apply (see the WEB_*_MAX_* constants); without it every
               call runs, as before 2026-10-25.
    """
    if state is None:
        results = []
        for tool in tools:
            if tool["type"] == "search":
                results.append(_run_search(tool, search_cache))
            elif tool["type"] == "web_read":
                _log.info("Executing web_read: ref='%s'", tool["ref"])
                url, err = _resolve_read_url(tool["ref"], search_cache)
                if err or not url:
                    results.append(f"[Web page content]\n{err or 'No URL to fetch.'}")
                else:
                    results.append(_run_web_read(url))
        return "\n\n".join(results)

    state["round"] += 1
    results: list[str] = []
    notes: list[str] = []
    seen_this_round: dict[tuple[str, str], int] = {}
    ran = 0
    used_chars = 0

    for tool in tools:
        key = _call_key(tool)
        label = (f'@search("{tool.get("query", "")}")' if tool["type"] == "search"
                 else f'@web_read({tool.get("ref", "")})')

        # 1. The same call twice in one turn runs once (the degenerate-loop case).
        if key in seen_this_round:
            seen_this_round[key] += 1
            state["duplicates"] += 1
            continue
        seen_this_round[key] = 1

        # 2. Placeholder arguments are a syntax description, not a call.
        if key[1] in _PLACEHOLDER_ARGS or not key[1].strip(".… "):
            state["refused"] += 1
            results.append(f"[Tool error: {label} has no real argument; it was not run. "
                           "Give an actual search query or result number.]")
            continue

        # 3. Bounded parallel calls per turn.
        if ran >= WEB_TOOLS_MAX_PER_ROUND:
            state["refused"] += 1
            notes.append(label)
            continue

        if tool["type"] == "search":
            # 4a. A query already run this request: point back instead of re-running.
            if key[1] in state["searched"]:
                results.append(f"[{label} was already run in turn {state['searched'][key[1]]}; "
                               "its results are earlier in this conversation. Not repeated.]")
                continue
            # 5a. max_uses, as Anthropic's web_search: an error result, the request goes on.
            if state["search_uses"] >= WEB_SEARCH_MAX_USES:
                state["refused"] += 1
                results.append(f"[Tool error: max_uses_exceeded — this request already ran "
                               f"{WEB_SEARCH_MAX_USES} searches, the limit. {label} was not run. "
                               "Answer with what you have and say what is missing.]")
                continue
            state["search_uses"] += 1
            state["searched"][key[1]] = state["round"]
            ran += 1
            out = _run_search(tool, search_cache)
        elif tool["type"] == "web_read":
            _log.info("Executing web_read: ref='%s'", tool["ref"])
            url, err = _resolve_read_url(tool["ref"], search_cache)
            if err or not url:
                results.append(f"[Web page content]\n{err or 'No URL to fetch.'}")
                continue
            # 4b. web_fetch-style cache: a page is fetched once per request.
            if url in state["fetched"]:
                results.append(f"[{label} = {url} was already read in turn {state['fetched'][url]}; "
                               "its text is earlier in this conversation. Not fetched again.]")
                continue
            # 5b. max_uses for page reads.
            if state["read_uses"] >= WEB_READ_MAX_USES:
                state["refused"] += 1
                results.append(f"[Tool error: max_uses_exceeded — this request already read "
                               f"{WEB_READ_MAX_USES} pages, the limit. {label} was not run. "
                               "Answer with what you have and say what is missing.]")
                continue
            # 6. What one turn may add to context (max_content_tokens per fetch is
            # WEB_READ_MAX_CHARS; this caps the turn). ~400 chars cover the header
            # and truncation note. A read that no longer fits waits for the next
            # turn and costs nothing against max_uses.
            room = min(WEB_READ_MAX_CHARS, WEB_ROUND_MAX_CHARS - used_chars - 400)
            if room < 1500:
                notes.append(label)
                continue
            state["read_uses"] += 1
            state["fetched"][url] = state["round"]
            ran += 1
            out = _run_web_read(url, max_chars=room)
        else:
            continue
        used_chars += len(out)
        results.append(out)

    dupes = sum(n - 1 for n in seen_this_round.values() if n > 1)
    if dupes:
        results.append(f"[Note: {dupes} repeated tool call(s) in this turn were skipped — each "
                       "distinct call ran once. Do not repeat calls; use the results above.]")
    if notes:
        results.append(f"[Not run: {len(notes)} call(s) over this turn's limits ({WEB_TOOLS_MAX_PER_ROUND} "
                       f"tool calls, ~{WEB_ROUND_MAX_CHARS // 1000}K chars of results): "
                       f"{', '.join(notes[:6])}{' ...' if len(notes) > 6 else ''}. "
                       "Request them in your next turn only if still needed.]")
    _log.info("web tools round %d: %d call(s) in, %d run, %d duplicate(s) skipped, "
              "%d over limit, uses %d/%d searches %d/%d reads, %d chars",
              state["round"], len(tools), ran, dupes, len(notes), state["search_uses"],
              WEB_SEARCH_MAX_USES, state["read_uses"], WEB_READ_MAX_USES, used_chars)
    return "\n\n".join(results)


# A tool-call line repeated in the model's own answer (the loop it was in).
_CALL_LINE = re.compile(r'^\s*(@search\(.*?\)|@web_read\(.*?\))\s*$')


def collapse_repeated_tool_calls(text: str) -> str:
    """Drop repeated tool-call lines from a model answer before it goes back
    into the conversation, so the model does not read its own loop and keep
    going. The first occurrence of each distinct call line stays."""
    seen: set[str] = set()
    out: list[str] = []
    removed = 0
    for line in text.splitlines():
        m = _CALL_LINE.match(line)
        if m:
            k = " ".join(m.group(1).split()).lower()
            if k in seen:
                removed += 1
                continue
            seen.add(k)
        out.append(line)
    if not removed:
        return text
    out.append(f"[{removed} repeated tool-call line(s) removed]")
    return "\n".join(out)


TOOL_RESULTS_PREFIX = "[Tool results — use this data to answer the original question]"


def clear_old_tool_results(messages: list[dict], keep: int = 2,
                           trigger_chars: int = 60000) -> int:
    """Context editing in the spirit of Anthropic's clear_tool_uses: once the
    tool-result messages of this request total more than trigger_chars,
    replace all but the newest `keep` of them with a short placeholder.
    Returns the number of chars cleared (0 = nothing done). In place."""
    idx = [i for i, m in enumerate(messages)
           if m.get("role") == "user" and isinstance(m.get("content"), str)
           and m["content"].startswith(TOOL_RESULTS_PREFIX)]
    total = sum(len(messages[i]["content"]) for i in idx)
    if total <= trigger_chars or len(idx) <= keep:
        return 0
    cleared = 0
    for i in idx[:-keep] if keep else idx:
        n = len(messages[i]["content"])
        if n <= 300:
            continue
        messages[i] = {**messages[i], "content": (
            f"{TOOL_RESULTS_PREFIX}\n\n[Earlier tool results cleared to save context "
            f"({n} chars). Re-run a tool if you need that data again.]")}
        cleared += n
    if cleared:
        _log.info("cleared %d chars of older tool results (kept the newest %d)", cleared, keep)
    return cleared


def tool_status_labels(tools: list[dict]) -> list[str]:
    """Generate human-readable labels for tool status display."""
    return [
        f"{'Searching' if t['type'] == 'search' else 'Reading'}: "
        f"{t.get('query', '') or t.get('ref', '')}"
        for t in tools
    ]
