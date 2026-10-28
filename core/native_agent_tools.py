"""Structured (native) tool calling for agent runs on small models.

Agent runs normally teach the model a text protocol — @read_file("path"),
```bash``` / ```edit``` blocks, @done(...) — through a long system prompt.
Large models learn it from the prompt; small ones do not. LFM3-1.7B tried its
trained XML call format first, then mixed the two, and needed five rounds for
"write hello.txt and read it back" (2026-10-27). Its chat template already
knows how to call OpenAI-style tools, so a model entry marked
"native_tools": true gets these definitions instead, and every call is turned
into the same AgentAction the text parser would have produced. Policy checks,
approvals, the sandbox, the circuit breaker and the audit log are untouched:
they only ever see AgentActions.

The set is deliberately small. A 2B model picks better from nine tools than
from the text protocol's twenty-odd markers.
"""

import json

from tools.agent_tools import AgentAction

FINISH_TOOL = "finish"


def _fn(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required}}}


_S = {"type": "string"}
_I = {"type": "integer"}

AGENT_NATIVE_TOOLS = [
    _fn("run_shell", "Run a shell command or a short script on the user's PC "
        "and return its output. PowerShell syntax is native here.",
        {"command": dict(_S, description="The command or script to run.")},
        ["command"]),
    _fn("run_python", "Run a Python script and return what it prints.",
        {"code": dict(_S, description="Python source code.")}, ["code"]),
    _fn("read_file", "Read a text file. Optionally only lines start_line..end_line.",
        {"path": _S, "start_line": _I, "end_line": _I}, ["path"]),
    _fn("write_file", "Create a NEW file with the given content. To change an "
        "existing file use edit_file.",
        {"path": _S, "content": _S}, ["path", "content"]),
    _fn("edit_file", "Replace one exact piece of text in an existing file.",
        {"path": _S,
         "old_text": dict(_S, description="Exact text currently in the file."),
         "new_text": dict(_S, description="Text to put in its place.")},
        ["path", "old_text", "new_text"]),
    _fn("list_files", "List files matching a glob pattern such as *.py or **/*.txt.",
        {"pattern": _S,
         "directory": dict(_S, description="Folder to search; default is the "
                                           "current folder.")},
        ["pattern"]),
    _fn("search_files", "Search file contents for a regular expression.",
        {"pattern": _S, "path": dict(_S, description="File or folder to search.")},
        ["pattern", "path"]),
    _fn("web_search", "Search the web. Returns numbered results.",
        {"query": _S}, ["query"]),
    _fn("web_read", "Read a web page: a result number from web_search, or a URL.",
        {"ref": _S}, ["ref"]),
    _fn(FINISH_TOOL, "Call when the goal is fully done (or you are blocked), "
        "with a short summary for the user.",
        {"summary": _S}, ["summary"]),
]

NATIVE_AGENT_PREAMBLE = """\
You are Artifex, an agent working on the user's own Windows PC in a loop, not a chat.
- Use the provided tools to do the work. Each tool call really runs, and its
  result comes back to you; then decide the next step.
- Never ask the user to run something: call the tool yourself.
- A file counts only once it is on disk: write it with write_file or edit_file.
- When the goal is fully done, call finish with a one-line summary. If you are
  blocked and need the user, call finish and say what you need."""


def build_native_agent_prompt(env_note: str, workspace: str, goal: str) -> str:
    out = NATIVE_AGENT_PREAMBLE
    if env_note:
        out += "\n\n" + env_note
    if workspace:
        out += (f"\n\nWORKSPACE: {workspace}\nYou are already in this folder; "
                "relative paths resolve here.")
    if goal:
        out += f"\n\nGOAL:\n{goal.strip()}"
    return out


def _args(call) -> dict:
    a = call.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a) if a.strip() else {}
        except json.JSONDecodeError:
            return {}
    return a if isinstance(a, dict) else {}


def call_to_action(call):
    """One OpenAI tool call -> (AgentAction | None, error | None, finish | None).

    Exactly one of the three is set. `error` is fed back to the model as that
    call's result; `finish` is the summary of a finish call.
    """
    name = str(call.get("name", "")).strip().lstrip("@").lower()
    a = _args(call)

    def need(*keys):
        missing = [k for k in keys if not str(a.get(k, "")).strip()]
        return f"{name}: missing {', '.join(missing)}" if missing else None

    if name == FINISH_TOOL:
        return None, None, str(a.get("summary") or "done").strip()
    if name == "run_shell":
        err = need("command")
        if err:
            return None, err, None
        cmd = str(a["command"])
        first = cmd.strip().splitlines()[0][:80]
        more = cmd.strip().count("\n")
        return AgentAction("shell", cmd, first + (f"  (+{more} more lines)" if more else "")), None, None
    if name == "run_python":
        err = need("code")
        if err:
            return None, err, None
        code = str(a["code"]).strip()
        lines = code.split("\n")
        return AgentAction("python", code, lines[0][:80] if len(lines) == 1
                           else f"{lines[0][:60]}  ({len(lines)} lines)"), None, None
    if name == "read_file":
        err = need("path")
        if err:
            return None, err, None
        path = str(a["path"]).strip()
        try:
            start = int(a["start_line"]) if a.get("start_line") not in (None, "") else None
            end = int(a["end_line"]) if a.get("end_line") not in (None, "") else None
        except (TypeError, ValueError):
            start = end = None
        if start:
            end = end or start + 199
            return AgentAction("read_file", f"{path}|L{start}-{end}",
                               f'read_file: "{path}" (lines {start}-{end})'), None, None
        return AgentAction("read_file", f"{path}|1", f'read_file: "{path}"'), None, None
    if name == "write_file":
        err = need("path")
        if err:
            return None, err, None
        path = str(a["path"]).strip()
        return AgentAction("edit_file", "\x00".join([path, "", str(a.get("content", ""))]),
                           f'write: {path}'), None, None
    if name == "edit_file":
        err = need("path", "old_text")
        if err:
            return None, err, None
        path = str(a["path"]).strip()
        old = str(a["old_text"])
        preview = old.strip()[:50].replace("\n", "↵")
        return AgentAction("edit_file", "\x00".join([path, old, str(a.get("new_text", ""))]),
                           f'edit: {path} ("{preview}...")'), None, None
    if name == "list_files":
        err = need("pattern")
        if err:
            return None, err, None
        pattern, base = str(a["pattern"]).strip(), str(a.get("directory") or "").strip()
        return AgentAction("glob", f"{pattern}|{base}" if base else pattern,
                           f'glob: "{pattern}"' + (f" in {base}" if base else "")), None, None
    if name == "search_files":
        err = need("pattern", "path")
        if err:
            return None, err, None
        pattern, path = str(a["pattern"]), str(a["path"]).strip()
        return AgentAction("grep", f"{pattern}|{path}|", f'grep: "{pattern}" in {path}'), None, None
    if name == "web_search":
        err = need("query")
        if err:
            return None, err, None
        q = str(a["query"]).strip()
        return AgentAction("search", q, f'search: "{q}"'), None, None
    if name == "web_read":
        ref = a.get("ref") or a.get("url") or a.get("n")
        if ref in (None, ""):
            return None, "web_read: missing ref", None
        ref = str(ref).strip()
        return AgentAction("web_read", ref, f"web_read: {ref}"), None, None
    known = ", ".join(t["function"]["name"] for t in AGENT_NATIVE_TOOLS)
    return None, f"unknown tool {name!r}; available: {known}", None


def sanitize_tool_pairs(messages: list) -> list:
    """Make tool-call history safe after compaction or trimming.

    Those steps can drop an assistant turn but keep its tool results, or copy
    messages without the tool_calls / tool_call_id keys. A tool result with
    no matching call can break a chat template, so orphans become plain user
    messages and calls with no surviving results lose their tool_calls.
    """
    out, open_ids = [], set()
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            open_ids = {tc.get("id") for tc in m["tool_calls"]}
            out.append(m)
            continue
        if role == "tool":
            if m.get("tool_call_id") in open_ids:
                out.append(m)
                continue
            out.append({"role": "user",
                        "content": "[TOOL OUTPUT]\n" + str(m.get("content", ""))})
            continue
        open_ids = set()
        out.append(m)
    # A trailing assistant turn whose calls have no results: drop the calls.
    answered = {m.get("tool_call_id") for m in out if m.get("role") == "tool"}
    for i, m in enumerate(out):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            ids = {tc.get("id") for tc in m["tool_calls"]}
            if not ids & answered:
                out[i] = {"role": "assistant", "content": m.get("content") or ""}
    return out
