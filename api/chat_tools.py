"""
Artifex API — full tool surface for the CHAT path.

*** NOT REACHABLE AS SHIPPED (2026-09-20). ***
_stream_with_tools() in api/server.py understands tool_mode="full" and will
drive everything below, but ChatCompletionRequest has no `tool_mode` or
`workspace` field, so no caller can ask for it and every request still
normalizes to "web" or "off". Chat behaves exactly as it always has.

To turn it on, one of:
  - add `tool_mode` + `workspace` to ChatCompletionRequest (api/server.py)
    and thread them through api/chat_jobs.py to the /v1/chat/jobs path the
    phone actually uses; then add the Off/Web/Full segment in
    api/static/app.html; or
  - read a server-side default (e.g. ARTIFEX_CHAT_TOOL_MODE) in
    _stream_with_tools instead of taking it per request.
Either way it stays behind ARTIFEX_PHONE_FULL_TOOLS.

Deliberately left in place rather than deleted: the design, the CRITICAL
screen and the workspace handling are the hard parts and they are done and
unit-tested. Nothing below runs until the wiring above exists.


api/web_tools.py deliberately exposes web search and nothing else. This
module is the other half: it lends the chat endpoint the SAME tool set the
Agent tab already has — shell, python, file read/edit, glob/grep, the
codebase tools — by adapting tools/agent_tools.py rather than
reimplementing any of it. Extraction, execution, sandbox policy, output
limits and the tool cache all stay in one place; chat and agent runs can
never drift apart in what a marker means.

Three differences from the agent loop, all forced by chat being chat:

  1. No approval UI. The agent tab can stop mid-run and ask; a streaming
     chat turn cannot. So CRITICAL-classified actions are refused here with
     a pointer to the Agent tab instead of being run unattended. Override
     with ARTIFEX_CHAT_ALLOW_CRITICAL=1 if you want chat to have the whole
     blade.
  2. No per-run folder, and no chdir. The agent API chdirs the whole
     process into the run's workspace; a chat turn can overlap a run that
     owns that chdir, so chat passes an explicit cwd instead
     (ARTIFEX_CHAT_WORKSPACE).
  3. Bounded rounds and output. A chat turn has to terminate and fit in a
     context window someone is also holding a conversation in.

Gated by ARTIFEX_PHONE_FULL_TOOLS — the same master switch that gates the
agent and files tabs. Off in a fresh clone.
"""

import logging
import os

_log = logging.getLogger("artifex.api.chat_tools")

# ── Tool modes ───────────────────────────────────────────────────────────

MODE_OFF = "off"
MODE_WEB = "web"
MODE_FULL = "full"
VALID_MODES = (MODE_OFF, MODE_WEB, MODE_FULL)

# Tool rounds per chat turn. Each round is a full generation, so this is a
# latency budget as much as a safety one.
MAX_TOOL_ROUNDS = int(os.getenv("ARTIFEX_CHAT_MAX_TOOL_ROUNDS", "8"))

# Actions honoured per round. A model that emits fifteen markers in one
# breath has lost the plot; run the first few and tell it so.
MAX_ACTIONS_PER_ROUND = int(os.getenv("ARTIFEX_CHAT_MAX_ACTIONS", "5"))

# Total chars of tool output fed back per round, across all actions. The
# per-tool limits below this come from agent_tools.get_tool_output_limit(),
# which already scales with the active context profile.
MAX_ROUND_OUTPUT_CHARS = int(os.getenv("ARTIFEX_CHAT_MAX_TOOL_CHARS", "24000"))


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def allow_critical() -> bool:
    """Whether chat may run CRITICAL-classified actions unattended."""
    return _env_flag("ARTIFEX_CHAT_ALLOW_CRITICAL")


def normalize_mode(tool_mode, web_tools_flag=False) -> str:
    """Settle the request's tool mode.

    `tool_mode` wins when present and valid. Otherwise the legacy boolean
    `web_tools` is honoured, so phone builds that predate tool_mode keep
    working unchanged.
    """
    if tool_mode:
        m = str(tool_mode).strip().lower()
        if m in VALID_MODES:
            return m
        _log.warning("Unknown tool_mode %r — falling back to web_tools flag", tool_mode)
    return MODE_WEB if web_tools_flag else MODE_OFF


# ── Workspace ────────────────────────────────────────────────────────────

def chat_workspace() -> str:
    """Directory chat-run commands execute in.

    Chat has no per-run folder, and the API's own cwd is the repo root —
    building there would scribble build output straight into git. Defaults
    to a gitignored folder under output/; point ARTIFEX_CHAT_WORKSPACE at a
    real project to work on one directly.
    """
    configured = os.getenv("ARTIFEX_CHAT_WORKSPACE", "").strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))
    from core.config import BASE_DIR
    return os.path.join(os.path.abspath(BASE_DIR), "output", "chat_workspace")


def resolve_workspace(requested: str | None = None) -> tuple[str, str | None]:
    """Return (workspace, error). Creates it if missing.

    A per-request workspace lets the phone point one conversation at a repo
    without restarting the API. It is still subject to the filesystem
    sandbox — in open scope that means anywhere, in the default scope it
    means under the project root.
    """
    target = (os.path.abspath(os.path.expanduser(requested.strip()))
              if requested and requested.strip() else chat_workspace())

    from core.sandbox.fs_sandbox import check_path
    denied = check_path(target)
    if denied:
        return chat_workspace(), f"workspace refused ({denied}) — using the default"

    try:
        os.makedirs(target, exist_ok=True)
    except OSError as e:
        return chat_workspace(), f"workspace not usable ({e}) — using the default"
    return target, None


# ── Extraction ───────────────────────────────────────────────────────────

def extract_chat_tools(text: str) -> list:
    """Extract executable actions from a model response.

    Delegates wholesale to agent_tools.extract_agent_actions, so chat sees
    exactly what an agent run would: @markers, ```bash/python/edit``` fences,
    the hybrid <tool_call>@marker form and native JSON tool calls. That
    includes the inert-marker rules — a marker in `backticks` or **bold** is
    documentation and does not fire.
    """
    from tools.agent_tools import extract_agent_actions
    return extract_agent_actions(text)


def tool_status_labels(actions: list) -> list[str]:
    """Human-readable 'what is running' labels for the SSE status event."""
    verbs = {
        "shell": "Running", "python": "Running python", "search": "Searching",
        "read_file": "Reading", "read_function": "Reading", "web_read": "Reading",
        "download": "Downloading", "glob": "Finding", "grep": "Searching",
        "edit_file": "Editing", "find_symbol": "Finding", "sysinfo": "Checking",
        "find_references": "Finding", "trace_imports": "Tracing",
        "architecture": "Mapping", "view_image": "Looking at",
        "describe_images": "Describing images in",
    }
    out = []
    for a in actions:
        display = str(getattr(a, "display", "") or getattr(a, "content", ""))[:80]
        out.append(f"{verbs.get(a.type, 'Running')}: {display}".strip())
    return out


# ── Execution ────────────────────────────────────────────────────────────

_CRITICAL_REFUSAL = (
    "[NOT RUN — {risk} risk]\n"
    "Chat runs tools without an approval prompt, so it will not run this "
    "one unattended: {display}\n"
    "Start it from the Agent tab instead — that surface pauses and asks "
    "before anything at this risk level. (Or set "
    "ARTIFEX_CHAT_ALLOW_CRITICAL=1 on the server to lift this.)"
)


def _screen(action) -> str | None:
    """Pre-execution screen. Returns a refusal message, or None to proceed.

    This is ON TOP of the sandbox policy engine, not instead of it —
    run_agent_action() still runs the full hook chain. The point is that
    ARTIFEX_POLICY=auto (what this box runs, so that chat CAN act at all)
    auto-allows CRITICAL too, and "rm -rf" arriving from a phone with no
    confirmation step is not what anyone meant by giving chat tools.
    """
    if allow_critical():
        return None
    from core.sandbox import classify_action, RiskLevel
    risk = classify_action(action.type, action.content)
    if risk >= RiskLevel.CRITICAL:
        display = str(getattr(action, "display", "") or action.content)[:200]
        _log.warning("chat refused %s action (%s): %s",
                     action.type, risk.name, display)
        return _CRITICAL_REFUSAL.format(risk=risk.name, display=display)
    return None


def execute_chat_tools(actions: list, cwd: str) -> str:
    """Run extracted actions and return the combined, capped feedback block.

    Never raises: a tool that blows up returns its traceback as text, the
    same way it would inside an agent run. A chat turn dying because a
    grep pattern was malformed would be a worse failure than the grep's.
    """
    from tools.agent_tools import run_agent_action, get_tool_output_limit

    results: list[str] = []
    budget = MAX_ROUND_OUTPUT_CHARS

    honoured = actions[:MAX_ACTIONS_PER_ROUND]
    dropped = len(actions) - len(honoured)

    for action in honoured:
        header = f"[{action.type}] {str(getattr(action, 'display', ''))[:120]}"

        refusal = _screen(action)
        if refusal:
            results.append(f"{header}\n{refusal}")
            continue

        try:
            # confirm_cb=None: under ARTIFEX_POLICY=auto nothing reaches it,
            # and under a stricter policy "refuse and say so" is the right
            # answer for a surface with no way to ask.
            ok, output = run_agent_action(action, confirm_cb=None,
                                          policy_check=True, cwd=cwd)
        except Exception as e:
            _log.exception("chat tool %s crashed", action.type)
            ok, output = False, f"Tool crashed: {type(e).__name__}: {e}"

        output = output or ("(no output)" if ok else "(failed, no output)")
        limit = min(get_tool_output_limit(action.type), max(budget, 500))
        if len(output) > limit:
            output = output[:limit] + (
                f"\n[TRUNCATED at {limit} chars — {len(output) - limit} more. "
                "Narrow the query or read a specific part.]")
        budget -= len(output)

        status = "ok" if ok else "FAILED"
        results.append(f"{header} -> {status}\n{output}")

        if budget <= 0:
            remaining = len(honoured) - len(results)
            if remaining > 0:
                results.append(
                    f"[{remaining} further action(s) not run — this round's "
                    "output budget is spent. Ask for them next turn.]")
            break

    if dropped > 0:
        results.append(
            f"[{dropped} further action(s) ignored — at most "
            f"{MAX_ACTIONS_PER_ROUND} run per round. Take them one at a time.]")

    return "\n\n".join(results)


# ── System prompt ────────────────────────────────────────────────────────

def full_tool_system_prompt(workspace: str) -> str:
    """The full tool catalog + sensed environment, same stack agent runs use.

    Built fresh per turn rather than cached: the workspace can differ per
    request, and sense_system() is what makes @sysinfo's absence obvious to
    the model when it would otherwise reach for shell commands.
    """
    from core.prompts import build_assistant_prompt
    from tools.agent_tools import get_assistant_tools_prompt
    try:
        base = build_assistant_prompt(
            get_assistant_tools_prompt(), workspace, workspace_text=workspace)
    except Exception:
        _log.exception("full tool prompt build failed — using the minimal form")
        base = (
            "You are Artifex, running on the user's own PC with tools that "
            f"really execute.\nWORKSPACE: {workspace}\n"
            'Use @read_file("path"), @glob("pattern"), @grep("pattern", "path"), '
            "```bash``` and ```python``` blocks. Relative paths resolve in the "
            "workspace."
        )
    try:
        from core.prompts import shell_execution_note
        base += "\n\n" + shell_execution_note()
    except Exception:
        _log.exception("shell execution note failed — prompt sent without it")
    return base + "\n\n" + _CHAT_TOOL_ADDENDUM.format(
        workspace=workspace,
        rounds=MAX_TOOL_ROUNDS,
        actions=MAX_ACTIONS_PER_ROUND,
        critical=("CRITICAL actions are ALLOWED on this server."
                  if allow_critical() else
                  "CRITICAL actions (rm -rf, git push --force, git reset "
                  "--hard, shutdown, drop table) are REFUSED here — tell the "
                  "user to use the Agent tab for those."),
    )


_CHAT_TOOL_ADDENDUM = """\
CHAT TOOL RULES — this is a conversation, not an unattended agent run:
- You are NOT cd'd anywhere. Relative paths resolve against the workspace
  below; absolute paths work anywhere on this machine.
- WORKSPACE: {workspace}
- At most {actions} actions per response, and {rounds} tool rounds per turn.
  Take one step, read the output, then decide the next one.
- There is no approval prompt in chat. Anything you run, runs. {critical}
- When you have the answer, STOP calling tools and answer in prose.
"""
