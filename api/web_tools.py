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

# ── Tool-call regex patterns ─────────────────────────────────────────────────

# Match @search("query") with regular or smart quotes
_SEARCH_PATTERN = re.compile(
    r'@search\(\s*["\u201c\u201d]([^"\u201c\u201d]+)["\u201c\u201d]\s*\)'
)
# Match @web_read(N) or @web_read("url") with regular or smart quotes
_WEB_READ_PATTERN = re.compile(
    r'@web_read\(\s*["\u201c\u201d]?([^"\u201c\u201d)]+)["\u201c\u201d]?\s*\)'
)


def extract_web_tools(text: str) -> list[dict]:
    """Extract @search and @web_read tool calls from model response text."""
    tools = []
    for m in _SEARCH_PATTERN.finditer(text):
        tools.append({"type": "search", "query": m.group(1), "raw": m.group(0)})
    for m in _WEB_READ_PATTERN.finditer(text):
        tools.append({"type": "web_read", "ref": m.group(1).strip(), "raw": m.group(0)})
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

def execute_web_tools(tools: list[dict], search_cache: list[dict]) -> str:
    """Execute a list of web tool calls and return combined results.

    Args:
        tools: List of tool dicts from extract_web_tools()
        search_cache: Mutable list that stores the last search results
                      for @web_read(N) index lookups. Modified in-place.
    """
    results = []

    for tool in tools:
        if tool["type"] == "search":
            _log.info("Executing web search: %s", tool["query"])
            ok, data = gateway_post("/search", {"query": tool["query"], "max_results": 8})
            if ok:
                sr = data.get("results", [])
                # Update search cache for @web_read(N) references
                search_cache.clear()
                search_cache.extend(sr)
                lines = []
                for i, r in enumerate(sr, 1):
                    lines.append(f"[{i}] {r.get('title', 'No title')}")
                    lines.append(f"    {r.get('url', '')}")
                    snippet = r.get("snippet", "")
                    if snippet:
                        lines.append(f"    {snippet[:400]}")
                    lines.append("")
                lines.append("Use @web_read(N) to read the full page for result N.")
                result = "\n".join(lines)
            else:
                result = f"Search failed: {data}"
            results.append(f"[Search results for: {tool['query']}]\n{result}")

        elif tool["type"] == "web_read":
            ref = tool["ref"]
            _log.info("Executing web read: %s", ref)

            url = ref
            if ref.isdigit():
                idx = int(ref) - 1
                if idx < 0 or idx >= len(search_cache):
                    if not search_cache:
                        result = "No search results cached. Use @search(\"query\") first."
                    else:
                        result = f"Invalid result number {ref}. Last search had {len(search_cache)} results."
                    results.append(f"[Web page content]\n{result}")
                    continue
                url = search_cache[idx].get("url", "")

            if not url:
                results.append("[Web page content]\nNo URL to fetch.")
                continue

            ok, data = gateway_post("/fetch", {"url": url})
            if not ok:
                results.append(f"[Web page content]\nFetch failed: {data}")
                continue

            text = data.get("text", "")
            title = data.get("title") or url
            warnings = data.get("warnings", [])

            if not text:
                results.append("[Web page content]\n(Page returned no readable text content.)")
                continue

            header = f"Source: {title}\nURL: {data.get('url', url)}\n---"
            if warnings:
                header += f"\n[Security warnings: {'; '.join(warnings)}]"

            # Truncate to ~4000 chars to leave room in context
            if len(text) > 4000:
                text = text[:4000] + "\n\n[...content truncated. Key content above.]"

            results.append(f"[Web page content]\n{header}\n{text}")

    return "\n\n".join(results)


def tool_status_labels(tools: list[dict]) -> list[str]:
    """Generate human-readable labels for tool status display."""
    return [
        f"{'Searching' if t['type'] == 'search' else 'Reading'}: "
        f"{t.get('query', '') or t.get('ref', '')}"
        for t in tools
    ]
