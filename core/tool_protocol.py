"""
Artifex Assistant V5 — Structured tool calling protocol.
Formalizes the existing @tool("arg") marker system for models that support
native function calling (e.g., Qwen3.5).
"""

import json
import re
from typing import List, Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Tool Definitions
# ─────────────────────────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "architecture",
        "description": "Get a full project map showing all files, classes, and functions.",
        "parameters": {},
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file. Large Python files return a skeleton view.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_function",
        "description": "Read the exact source code of a function, class, or method.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "name": {"type": "string", "description": "Function/class/method name"},
            },
            "required": ["path", "name"],
        },
    },
    {
        "name": "find_symbol",
        "description": "Find where a symbol (function, class, variable) is defined.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Symbol name to find"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "grep",
        "description": "Search file contents using a regex pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Directory or file path to search"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g., **/*.py)"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "search",
        "description": "Perform a web search and return results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "shell",
        "description": "Execute a shell command on the user's machine.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "python",
        "description": "Execute Python code for computation or file writing.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"],
        },
    },
]


def format_tools_for_model(tools=None):
    """Format tool definitions for injection into the chat template.

    Supports Qwen3.5 native function calling format.

    Args:
        tools: List of tool dicts. If None, uses TOOL_DEFINITIONS.

    Returns:
        Formatted string for the system prompt.
    """
    tools = tools or TOOL_DEFINITIONS

    lines = ["Available tools:"]
    for tool in tools:
        name = tool["name"]
        desc = tool["description"]
        params = tool.get("parameters", {})
        props = params.get("properties", {})

        if props:
            param_str = ", ".join(
                f"{k}: {v.get('type', 'string')}"
                for k, v in props.items()
            )
            lines.append(f"- {name}({param_str}): {desc}")
        else:
            lines.append(f"- {name}(): {desc}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tool Call Parsing
# ─────────────────────────────────────────────────────────────────────────────

# Match patterns like: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
_TOOL_CALL_RE = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
    re.DOTALL,
)

# Also match Qwen's function call format
_QWEN_CALL_RE = re.compile(
    r'(\w+)\((.*?)\)',
    re.DOTALL,
)


def parse_tool_calls(response: str) -> List[Dict]:
    """Extract structured tool calls from model output.

    Supports multiple formats:
    - <tool_call>{"name": "func", "arguments": {...}}</tool_call>
    - Native @tool("arg") markers (already handled by agent_tools.py)

    Args:
        response: Model output text

    Returns:
        List of dicts with "name" and "arguments" keys
    """
    calls = []

    # Try structured format first
    for match in _TOOL_CALL_RE.finditer(response):
        try:
            call = json.loads(match.group(1))
            if "name" in call:
                calls.append(call)
        except json.JSONDecodeError:
            continue

    return calls


def execute_tool_call(call: Dict) -> str:
    """Dispatch a parsed tool call to the agent tools system.

    Args:
        call: Dict with "name" and "arguments" keys

    Returns:
        Tool output string
    """
    from tools.agent_tools import run_agent_action
    from collections import namedtuple

    name = call.get("name", "")
    args = call.get("arguments", {})

    # Map to the agent_tools action format
    AgentAction = namedtuple("AgentAction", ["type", "display", "content"])

    if name in ("shell", "bash"):
        action = AgentAction("shell", args.get("command", ""), args.get("command", ""))
    elif name == "python":
        action = AgentAction("python", "python code", args.get("code", ""))
    elif name == "read_file":
        action = AgentAction("read_file", args.get("path", ""), args.get("path", ""))
    elif name == "search":
        action = AgentAction("websearch", args.get("query", ""), args.get("query", ""))
    elif name == "grep":
        display = f'grep "{args.get("pattern", "")}" {args.get("path", ".")}'
        action = AgentAction("grep", display, display)
    elif name == "glob":
        action = AgentAction("glob", args.get("pattern", ""), args.get("pattern", ""))
    else:
        return f"Unknown tool: {name}"

    success, output = run_agent_action(action)
    return output if success else f"ERROR: {output}"
