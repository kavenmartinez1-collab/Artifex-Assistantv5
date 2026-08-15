"""
Artifex Assistant V5 — Autonomous agent loop.

A UI-agnostic controller that drives the generate → act → observe cycle on its
own, so a local LLM can pursue a goal across many rounds without a human
clicking "Run" each time. Both the Qt GUI (via a QThread) and the CLI drive the
*same* AgentRunner, differing only in their event sink + approval callback.

The loop reuses what already exists:
  - tools.agent_tools.extract_agent_actions / run_agent_action / detect_done
  - core.sandbox.check_policy + the self-modification ratchet (every action)
  - core.sandbox.circuit_breaker.CircuitBreaker (runaway-loop trips)
  - core.sandbox.human_gate.GateState (periodic human checkpoints)
  - auto-commit-on-edit / auto-revert-on-failure (git_commit_edit/revert)

Autonomy is a dial, not a kill-switch for safety: MANUAL prompts on every
action, GUIDED auto-runs only low-risk reads and prompts on writes / MEDIUM+,
FULL_AUTO runs everything policy *allows* without per-action prompts. Policy,
ratchet, circuit breaker, and human gate apply at ALL levels.
"""

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

from core.prompts import build_autonomous_prompt
from core.sandbox import check_policy, RiskLevel
from core.sandbox.circuit_breaker import CircuitBreaker
from core.sandbox.human_gate import GateState
from tools.agent_tools import (
    extract_agent_actions, run_agent_action, detect_done,
    MAX_AGENT_ROUNDS, git_commit_edit, git_revert_last,
)
from tools.tool_cache import maybe_cache_output, update_session_map

_log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS / VALUE TYPES
# ═══════════════════════════════════════════════════════════════════════════

class AutonomyLevel(str, Enum):
    MANUAL = "manual"        # confirm every action (today's behavior)
    GUIDED = "guided"        # auto low-risk reads; confirm writes / MEDIUM+
    FULL_AUTO = "full_auto"  # run everything policy allows, no per-action prompt


AUTONOMY_LEVELS = (AutonomyLevel.MANUAL, AutonomyLevel.GUIDED, AutonomyLevel.FULL_AUTO)


class Decision(str, Enum):
    APPROVE = "approve"   # run this action / continue past the pause
    DENY = "deny"         # skip this action, keep going
    STOP = "stop"         # abort the whole run


@dataclass
class AgentEvent:
    """One observable thing that happened, handed to the host's event sink."""
    kind: str
    round: int = 0
    text: str = ""
    action: object = None        # tools.agent_tools.AgentAction
    decision: object = None      # core.sandbox.PolicyDecision
    success: Optional[bool] = None
    output: str = ""
    reason: str = ""
    summary: str = ""


@dataclass
class RunConfig:
    autonomy: AutonomyLevel = AutonomyLevel.GUIDED
    max_rounds: int = MAX_AGENT_ROUNDS
    wall_clock_s: float = 0.0            # 0 = no time limit
    max_consecutive_failures: int = 3
    context_window: int = 8192
    max_tokens: int = 2048
    temperature: float = 0.7
    enable_thinking: bool = True
    # Sampler selection for the run. `sampling` (explicit dict) wins over
    # `sampler_preset` (name resolved via core.sampling.get_preset). When a
    # resolved dict carries "temperature", it overrides the field above —
    # presets own the full sampler chain. None/None keeps engine defaults.
    sampler_preset: Optional[str] = None
    sampling: Optional[dict] = None
    # Thinking depth for models whose template exposes it (qwen3.8: low /
    # medium / high / xhigh). None keeps the engine/template default — which
    # on qwen3.8 is xhigh, measured to deliberate past the entire completion
    # budget on some prompts and return zero content, so hosts that drive
    # unattended runs should set "medium" explicitly. Passed to the engine
    # only when its generate_streaming signature accepts it.
    reasoning_effort: Optional[str] = None
    always_confirm_types: tuple = ("edit_file", "download")  # GUIDED always asks
    auto_approve_max_risk: str = "LOW"                       # GUIDED auto-runs <= this
    framing: bool = True   # wrap the prompt with the autonomous preamble + GOAL

    @classmethod
    def default(cls, autonomy: AutonomyLevel = AutonomyLevel.GUIDED) -> "RunConfig":
        try:
            from core.config import MODES
            m = MODES.get("ASSISTANT")
        except Exception:
            m = None
        return cls(
            autonomy=autonomy,
            context_window=getattr(m, "context_window", 8192),
            max_tokens=getattr(m, "max_tokens", 2048),
            temperature=getattr(m, "temperature", 0.7),
            enable_thinking=getattr(m, "enable_thinking", True),
        )


@dataclass
class RunResult:
    status: str            # "done" | "stopped:<reason>"
    rounds: int = 0
    actions_run: int = 0
    summary: str = ""
    history: list = field(default_factory=list)


class GenerationAborted(Exception):
    """Raised inside the token callback to cut off an in-flight generation.

    Without it a stop request is only honored at the round boundary, so a
    stop during a full-budget round leaves the run "running" for minutes
    while the tokens drain (measured 48 s on a 4096-token round). Raising
    from on_token unwinds the engine's SSE read loop; closing that
    connection also makes llama-server abort the generation server-side.
    """


# ═══════════════════════════════════════════════════════════════════════════
# CONTROL (pause / stop, thread-safe — set from the UI thread)
# ═══════════════════════════════════════════════════════════════════════════

class RunControl:
    def __init__(self):
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._msg_lock = threading.Lock()
        self._messages: list[str] = []

    def inject_message(self, text: str):
        """Queue a user follow-up for the loop to pick up at the next round
        boundary. Safe from any thread; the loop drains via drain_messages()."""
        with self._msg_lock:
            self._messages.append(text)

    def drain_messages(self) -> list[str]:
        with self._msg_lock:
            out, self._messages[:] = list(self._messages), []
            return out

    def request_stop(self):
        self._stop.set()
        self._pause.clear()  # don't deadlock a paused loop on stop

    def request_pause(self):
        self._pause.set()

    def resume(self):
        self._pause.clear()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    @property
    def paused(self) -> bool:
        return self._pause.is_set()

    def wait_if_paused(self, poll: float = 0.1):
        while self._pause.is_set() and not self._stop.is_set():
            time.sleep(poll)

    def reset(self):
        self._stop.clear()
        self._pause.clear()


# ═══════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════

class AgentRunner:
    """Drives the autonomous loop. Synchronous — run it in whatever thread the
    host provides (a QThread for the GUI, the main thread for the CLI).

    build_system_prompt — () -> str: host builds the normal assistant system
        prompt (with workspace/knowledge/.artifex context); the runner wraps it
        with the autonomous preamble + goal each round.
    emit                — (AgentEvent) -> None: display sink.
    request_approval    — (action, decision, reason) -> Decision: blocking. For
        per-action prompts action/decision are set; for breaker/gate pauses they
        are None and `reason` explains. Return APPROVE/DENY/STOP.
    """

    def __init__(self, engine, *, build_system_prompt: Callable[[], str],
                 emit: Optional[Callable[[AgentEvent], None]] = None,
                 request_approval: Optional[Callable] = None,
                 km=None, session_map=None, config: Optional[RunConfig] = None,
                 control: Optional[RunControl] = None,
                 gate: Optional[GateState] = None,
                 breaker: Optional[CircuitBreaker] = None):
        self.engine = engine
        self.build_system_prompt = build_system_prompt
        self.emit = emit or (lambda e: None)
        self.request_approval = request_approval or (lambda a, d, r: Decision.APPROVE)
        self.km = km
        self.session_map = session_map
        self.config = config or RunConfig.default()
        self.control = control or RunControl()
        self.gate = gate if gate is not None else GateState()
        self.breaker = breaker if breaker is not None else CircuitBreaker()
        self._round = 0
        self._actions_run = 0
        self._t0 = 0.0
        self._gen_error = ""
        self.goal = ""

    # ── public ──────────────────────────────────────────────────────────────

    def run(self, goal: str, history: list) -> RunResult:
        """Pursue `goal`, mutating `history` in place. Returns a RunResult."""
        self._t0 = time.monotonic()
        self._round = 0
        self._actions_run = 0
        self.goal = goal or ""
        if goal:
            history.append({"role": "user", "content": goal})
        consecutive_failures = 0
        format_retries = 0

        for rnd in range(1, self.config.max_rounds + 1):
            self._round = rnd
            if self.control.stop_requested:
                return self._finish("stopped:user", history)
            self.control.wait_if_paused()
            if self._over_wall_clock():
                self.emit(AgentEvent("stopped", reason="time limit", round=rnd))
                return self._finish("stopped:timeout", history)

            # Follow-up messages from the host (phone/GUI) land between
            # rounds as ordinary user turns, so the next generation sees
            # them exactly like mid-task steering in a chat.
            for injected in self.control.drain_messages():
                history.append({"role": "user", "content": injected})
                self.emit(AgentEvent("user_message", text=injected, round=rnd))

            self.emit(AgentEvent("round_start", round=rnd))
            self._set_system_prompt(history)
            active = self._active_messages(history)
            resp = self._generate(active)
            if self.control.stop_requested:
                # Mid-generation stop: keep whatever partial text streamed
                # (the prose branch below would misread it as a completed
                # answer and report "done" for a run the user killed).
                if resp:
                    history.append({"role": "assistant", "content": resp})
                    self.emit(AgentEvent("assistant_message", text=resp, round=rnd))
                return self._finish("stopped:user", history)
            if not resp and self._gen_error:
                # Engine failure, not a model choice — don't report a clean
                # "done" (empty resp with no actions would read as one).
                return self._finish(f"stopped:error:{self._gen_error}", history)
            history.append({"role": "assistant", "content": resp})
            if self.km:
                self._safe(lambda: self.km.add_from_ai_response(resp))
            self.emit(AgentEvent("assistant_message", text=resp, round=rnd))

            done = detect_done(resp)
            actions = extract_agent_actions(resp)
            if done is None and not actions:
                # No actions and no @done. Distinguish "model is finished and
                # says so in prose" from "model TRIED to act but the call was
                # malformed" (e.g. Qwen3.6 <tool_call> repetition collapse,
                # empty fences). The latter gets corrective feedback and
                # another round instead of being misread as a final answer.
                if format_retries < 2 and self._looks_like_failed_tool_attempt(resp):
                    format_retries += 1
                    self.emit(AgentEvent("format_retry", round=rnd,
                                         reason="malformed tool invocation"))
                    history.append({"role": "user", "content": self._format_nudge()})
                    continue
                if format_retries < 2 and self._looks_like_stalled_plan(resp):
                    # "Step 1: find the definition." <end of turn> — a chat
                    # reflex: announce, then wait for the user. Nobody is
                    # listening; push it to act (observed on Qwen3.6, s4).
                    format_retries += 1
                    self.emit(AgentEvent("format_retry", round=rnd,
                                         reason="announced a plan, took no action"))
                    history.append({"role": "user", "content": self._stall_nudge()})
                    continue
            if not actions:
                summary = done if done else resp.strip()
                self.emit(AgentEvent("done", summary=summary, round=rnd))
                return self._finish("done", history, summary)

            # NOTE: a response carrying BOTH actions and @done falls through to
            # the executor below. Returning here (the old behavior) discarded
            # the actions *and* reported success — a run recorded as
            # "done: wrote answer.txt" with no answer.txt on disk. Models that
            # batch their final write with @done in one turn (Qwen3.8 does this
            # routinely) silently lost that write. The deferred @done is
            # honored after the actions run, and only if they all succeeded.
            outputs: List[str] = []
            pending_edits: List[str] = []
            deferred_done_ok = True
            for action in actions:
                if self.control.stop_requested:
                    return self._finish("stopped:user", history)
                self.control.wait_if_paused()

                decision = check_policy(action.type, action.content)
                self.emit(AgentEvent("action_proposed", action=action,
                                     decision=decision, round=rnd))

                sandbox_approved = False
                if not decision.allowed:
                    # Outside-sandbox (scope) blocks become approval prompts
                    # when a human is supervising — the protection is the
                    # person in the loop, not the wall. Deny-list hits
                    # ("fs_sandbox") and unattended full-auto runs stay hard.
                    promptable = (
                        decision.matched_rule == "fs_sandbox_scope"
                        and self.config.autonomy != AutonomyLevel.FULL_AUTO
                    )
                    if not promptable:
                        self.emit(AgentEvent("blocked", action=action,
                                             reason=decision.reason, round=rnd))
                        outputs.append(f"[BLOCKED by policy] {action.display}: {decision.reason}")
                        deferred_done_ok = False
                        continue
                    self.emit(AgentEvent("approval_required", action=action,
                                         decision=decision, round=rnd))
                    dec = self.request_approval(action, decision, "")
                    if dec == Decision.STOP:
                        return self._finish("stopped:user", history)
                    if dec == Decision.DENY:
                        outputs.append(f"[skipped by user] {action.display}")
                        deferred_done_ok = False
                        continue
                    sandbox_approved = True

                # Circuit breaker — catch runaway loops before executing.
                trip = self.breaker.check_and_trip()
                if trip:
                    self.emit(AgentEvent("breaker_tripped", reason=trip, round=rnd))
                    if self.request_approval(None, None, f"circuit breaker: {trip}") == Decision.STOP:
                        return self._finish("stopped:breaker", history)
                    self.breaker.acknowledge()

                if not sandbox_approved and self._needs_approval(action, decision):
                    self.emit(AgentEvent("approval_required", action=action,
                                         decision=decision, round=rnd))
                    dec = self.request_approval(action, decision, "")
                    if dec == Decision.STOP:
                        return self._finish("stopped:user", history)
                    if dec == Decision.DENY:
                        outputs.append(f"[skipped by user] {action.display}")
                        deferred_done_ok = False
                        continue

                self.emit(AgentEvent("action_started", action=action, round=rnd))
                try:
                    # policy_check=False: this loop already ran check_policy
                    # (blocked/approval handling above) — re-checking inside
                    # run_agent_action would double-count the audit-log and
                    # circuit-breaker hooks for the same action.
                    ok, out = run_agent_action(action, policy_check=False)
                except Exception as e:  # executor blew up — treat as failure
                    ok, out = False, f"ERROR: {e}"
                out = out or ""
                self._actions_run += 1

                if ok:
                    self.breaker.record_success(action.content)
                    consecutive_failures = 0
                else:
                    self.breaker.record_failure(action.content)
                    consecutive_failures += 1
                    deferred_done_ok = False
                self.gate.record_action(decision.risk_level)

                if self.km:
                    self._safe(lambda: self.km.process_tool_result(
                        action.type, action.display, out))
                if self.session_map is not None:
                    self._safe(lambda: update_session_map(
                        self.session_map, action.type, action.display, out))
                self._on_action_complete(action, ok, pending_edits)

                cached = out
                if out:
                    try:
                        cached = maybe_cache_output(action.type, action.display, out)
                    except Exception:
                        cached = out
                self.emit(AgentEvent("action_result", action=action,
                                     success=ok, output=out, round=rnd))
                outputs.append(f"[{action.type}] `{action.display}`:\n{cached}")

                if consecutive_failures >= self.config.max_consecutive_failures:
                    self.emit(AgentEvent("stopped", reason="too many consecutive failures",
                                         round=rnd))
                    return self._finish("stopped:failures", history)

            # Human gate checkpoint (round interval / action cap / risk budget).
            greason = self.gate.should_gate(rnd)
            if greason:
                self.emit(AgentEvent("gate_pause", reason=greason, round=rnd))
                if self.request_approval(None, None, f"human gate: {greason}") == Decision.STOP:
                    return self._finish("stopped:gate", history)
                self.gate.acknowledge_gate(rnd)

            history.append({"role": "user", "content": self._feedback(outputs)})

            # @done arrived alongside the actions we just ran. Honor it only if
            # every action actually succeeded; otherwise drop back into the
            # loop so the model sees the failure in its feedback and can
            # recover, instead of us accepting a success claim it can't back up.
            if done is not None and deferred_done_ok:
                self.emit(AgentEvent("done", summary=done, round=rnd))
                return self._finish("done", history, done)

        self.emit(AgentEvent("stopped", reason=f"max rounds ({self.config.max_rounds})",
                             round=self._round))
        return self._finish("stopped:max_rounds", history)

    # ── internals ─────────────────────────────────────────────────────────────

    def _needs_approval(self, action, decision) -> bool:
        """Whether to pause for human approval before running `action`.

        Autonomy (the dropdown the user picks) is authoritative here — NOT the
        policy engine's `requires_confirmation`, which under the default STRICT
        policy would force a prompt on everything and defeat Guided. Hard denies
        (allowed=False from the ratchet/proc sandbox) never reach this. CRITICAL
        risk stays gated even in FULL_AUTO as a safety floor.
        """
        lvl = self.config.autonomy
        risk = decision.risk_level
        if lvl == AutonomyLevel.FULL_AUTO:
            return risk == RiskLevel.CRITICAL
        if lvl == AutonomyLevel.MANUAL:
            return True
        # GUIDED — auto-run low-risk reads; confirm writes / MEDIUM+.
        if action.type in self.config.always_confirm_types:
            return True
        try:
            ceiling = RiskLevel[self.config.auto_approve_max_risk]
        except KeyError:
            ceiling = RiskLevel.LOW
        return risk > ceiling   # RiskLevel is an IntEnum

    def _set_system_prompt(self, history):
        try:
            base = self.build_system_prompt()
        except Exception as e:
            _log.warning("build_system_prompt failed: %s", e)
            base = ""
        content = build_autonomous_prompt(base, self.goal) if self.config.framing else base
        if history and history[0].get("role") == "system":
            history[0]["content"] = content
        else:
            history.insert(0, {"role": "system", "content": content})

    def _active_messages(self, history):
        from core.inference import (build_active_messages, auto_compact_if_needed,
                                    trim_messages_to_context)
        cw = self.config.context_window
        ctx = 0
        try:
            if hasattr(self.engine, "get_context_size"):
                ctx = self.engine.get_context_size() or 0
        except Exception:
            ctx = 0
        if ctx > 0:
            new_hist, _ = auto_compact_if_needed(history, ctx, cw)
            history[:] = new_hist
        _, active = build_active_messages(history, cw, engine_ctx=ctx)
        if ctx > 0:
            active = trim_messages_to_context(active, int(ctx * 0.85))
        return active

    def _resolved_sampling(self) -> Optional[dict]:
        """Sampler dict for this run: explicit config.sampling > named preset."""
        if self.config.sampling is not None:
            return dict(self.config.sampling)
        if self.config.sampler_preset:
            from core.sampling import get_preset
            return get_preset(self.config.sampler_preset)
        return None

    def _engine_gen_kwargs(self) -> dict:
        """Optional generate_streaming kwargs the engine's signature accepts.

        Engines predate the sampling/enable_thinking wiring (and tests use
        minimal fakes), so only pass what the callee can take instead of
        blowing up with TypeError on older signatures.
        """
        import inspect
        try:
            params = inspect.signature(self.engine.generate_streaming).parameters
        except (TypeError, ValueError):
            return {}
        has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                         for p in params.values())
        kwargs = {}
        if has_var_kw or "enable_thinking" in params:
            kwargs["enable_thinking"] = self.config.enable_thinking
        samp = self._resolved_sampling()
        if samp is not None and (has_var_kw or "sampling" in params):
            kwargs["sampling"] = samp
        if self.config.reasoning_effort and (has_var_kw or "reasoning_effort" in params):
            kwargs["reasoning_effort"] = self.config.reasoning_effort
        return kwargs

    def _generate(self, active) -> str:
        from core.inference import ThinkFilter
        self._gen_error = ""
        parts: List[str] = []

        def on_resp(t):
            if self.control.stop_requested:
                raise GenerationAborted()
            parts.append(t)
            self.emit(AgentEvent("assistant_chunk", text=t, round=self._round))

        def on_think(t):
            if self.control.stop_requested:
                raise GenerationAborted()
            self.emit(AgentEvent("thinking_chunk", text=t, round=self._round))

        # Engines that emit explicit <think> tags (llama.cpp, ollama) start
        # the stream OUTSIDE a think block; transformers pre-fills <think>.
        # With thinking disabled there is no leading think block either way.
        starts_in_think = (bool(getattr(self.engine, "stream_starts_in_think", True))
                           and self.config.enable_thinking)
        tf = ThinkFilter(on_response=on_resp, on_thinking=on_think,
                         starts_in_think=starts_in_think)
        try:
            resp = self.engine.generate_streaming(
                active, max_tokens=self.config.max_tokens,
                temperature=self.config.temperature, on_token=tf.feed,
                **self._engine_gen_kwargs())
        except GenerationAborted:
            # User stop, not a failure: return the partial text so the
            # transcript keeps what was said; the round loop's stop checks
            # prevent any of its actions from running and end the run.
            self._safe(tf.flush)
            return "".join(parts)
        except Exception as e:
            _log.exception("agent_loop generation failed")
            self._gen_error = type(e).__name__
            self.emit(AgentEvent("error", reason=str(e), round=self._round))
            self._safe(tf.flush)
            return ""
        self._safe(tf.flush)
        return resp if isinstance(resp, str) and resp else "".join(parts)

    def _on_action_complete(self, action, ok, pending_edits):
        """Auto-commit a successful edit; auto-revert this round's edits on a
        failed python/shell run (so a broken test rolls back automatically)."""
        if action.type == "edit_file" and ok:
            path = action.content.split("\x00", 1)[0].strip()
            if path:
                cok, msg = git_commit_edit(path, action.display)
                if cok:
                    pending_edits.append(path)
                self.emit(AgentEvent("git", text=msg, round=self._round))
        elif action.type in ("python", "shell") and not ok and pending_edits:
            for path in reversed(pending_edits):
                rok, msg = git_revert_last(path)
                self.emit(AgentEvent("git", text=msg, round=self._round))
                if not rok:
                    break
            pending_edits.clear()

    @staticmethod
    def _looks_like_failed_tool_attempt(resp: str) -> bool:
        """Did the model clearly TRY to invoke tooling that didn't parse?

        Signals: a native <tool_call> tag that produced no action (Qwen3.6
        repetition collapse emits bare `<tool_call>` lines), or an opened
        code fence with no body. Plain prose returns False — that is a
        legitimate final answer.
        """
        if not resp:
            return False
        if "<tool_call" in resp:
            return True
        # Caller only reaches here when ZERO actions parsed — so any fenced
        # tool block in the text is by definition a failed invocation
        # (malformed edit, empty body, unknown structure).
        if re.search(r"```(?:bash|sh|shell|powershell|cmd|python|py|edit)\b", resp):
            return True
        return False

    _PLAN_ANNOUNCE_RE = re.compile(
        r"(?im)\b(step\s*[1-9]\s*[:.]|first[,:]|let me |i'?ll (?:start|begin|now|first)"
        r"|i will (?:start|begin|now|first)|next[,:] )",
    )

    @classmethod
    def _looks_like_stalled_plan(cls, resp: str) -> bool:
        """Announce-without-acting: plan language, zero actions, no @done.

        Only consulted in the no-actions/no-done branch, so a false positive
        costs one nudge round (capped), while a false negative silently ends
        the run mid-task as a fake 'done'.
        """
        return bool(resp) and bool(cls._PLAN_ANNOUNCE_RE.search(resp))

    @staticmethod
    def _stall_nudge() -> str:
        return (
            "[STALL — automated] You announced a step but executed nothing, "
            "and this is an autonomous loop — no one can reply. EXECUTE the "
            "step NOW in this turn using a live @tool(...) marker or a "
            "```bash```/```python``` block, or emit @done(\"summary\") if the "
            "GOAL is already complete."
        )

    @staticmethod
    def _format_nudge() -> str:
        return (
            "[FORMAT ERROR — automated] Your last message tried to invoke a "
            "tool but no valid action could be parsed. Use EXACTLY this "
            "syntax, each on its own line:\n"
            '@read_file("path")  @glob("pattern")  @grep("pattern", "path")  '
            '@find_symbol("name")\n'
            "```bash\n<command>\n```  or  ```python\n<code>\n```\n"
            "Do NOT write <tool_call> tags. Retry the intended action now, "
            'or emit @done("summary") if the GOAL is already complete.'
        )

    def _feedback(self, outputs: List[str]) -> str:
        body = "\n\n".join(outputs) if outputs else "(no output)"
        if self.config.framing:
            tail = ("Continue toward the GOAL. When it is fully done, give a short "
                    'final summary or emit @done("one-line summary").')
        else:
            tail = "Analyze the output above and tell the user what you found, then continue or stop."
        return f"[TOOL OUTPUT — automated command output, not a human message]\n\n{body}\n\n{tail}"

    def _over_wall_clock(self) -> bool:
        return (self.config.wall_clock_s > 0
                and (time.monotonic() - self._t0) >= self.config.wall_clock_s)

    def _finish(self, status: str, history: list, summary: str = "") -> RunResult:
        return RunResult(status=status, rounds=self._round,
                         actions_run=self._actions_run, summary=summary,
                         history=history)

    @staticmethod
    def _safe(fn):
        try:
            fn()
        except Exception:
            pass
