"""
Artifex Assistant V5 — Agent runs over the REST API.

Exposes core.agent_loop.AgentRunner (the same controller the Qt GUI and CLI
drive) to HTTP clients, so a phone on the tailnet can start goals, watch the
loop act, and answer its approval prompts.

Endpoints (all bearer-authed via the check_auth callable the server passes in):

    POST /v1/agent/runs                    start a run  -> {run_id}
    GET  /v1/agent/runs                    list runs (newest first)
    GET  /v1/agent/runs/{id}               snapshot: status + pending approval
    GET  /v1/agent/runs/{id}/events        SSE: replay buffered events, then live
    POST /v1/agent/runs/{id}/approval      {"decision": "approve"|"always"|"deny"|"stop"}
    POST /v1/agent/runs/{id}/message       follow-up: steer live / revive done
    POST /v1/agent/runs/{id}/stop          abort the run

Design notes:

- ONE run at a time. The GPU serves one generation stream anyway, and a
  second concurrent loop would interleave chdir + engine use for no benefit.
  Starting while another run is live returns 409 with the live run's id.

- The worker thread chdirs into the run's workspace folder for the duration
  (agent tool paths are cwd-relative) and restores the previous cwd after.
  chdir is process-global; that is acceptable on this single-user deployment
  and guarded by the single-run rule above.

- Per-token events (assistant_chunk / thinking_chunk) are forwarded to LIVE
  SSE listeners but not persisted to the replay buffer — a long run would
  otherwise buffer tens of thousands of events, and a phone reconnecting
  after backgrounding would replay them all. Reconnects rebuild from the
  round-complete events (assistant_message, action_result, ...), which carry
  the same content in aggregate.

- request_approval blocks the worker thread on a queue. The client answers
  via POST .../approval; a stop request unblocks it with Decision.STOP.
  "always" approves AND remembers a rule for the rest of the run
  (core.sandbox.approvals) — in memory on the AgentRun, so it survives
  follow-up revivals and disappears with the run; nothing is persisted. No
  timeout: an unanswered prompt holds the run in "awaiting_approval"
  indefinitely, which is visible in the run list and resumable from the
  phone whenever the user comes back.

- The run names its model. Starting (or reviving) a run switches the model
  queue to it BEFORE the worker loads the engine — the same path chat takes —
  so an agent started with the vision entry selected runs on the vision
  entry, not on whatever the API happened to launch with. While a run is
  live the queue reports busy (live_run_busy_reason): the idle shrink leaves
  the engine loaded and a chat naming another model is refused, instead of
  llama-server being unloaded under the run.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.agent_loop import (
    AgentRunner, RunConfig, RunControl, AutonomyLevel, Decision, AgentEvent,
)
from core.sandbox.approvals import ApprovalMemory
from core.logging_config import get_logger

_log = get_logger(__name__)

_MAX_KEPT_RUNS = 20          # finished runs retained for the list/replay
_EVENT_TEXT_CAP = 8000       # chars of any single text field kept per event

_runs: dict[str, "AgentRun"] = {}
_runs_lock = threading.Lock()


# ── Request models ───────────────────────────────────────────────────────

class AgentRunRequest(BaseModel):
    goal: str = Field(..., min_length=1,
                      json_schema_extra={"example": "List the .py files here and write their names to files.txt"})
    model: Optional[str] = Field(
        None,
        description=(
            "Model to run on (a /v1/models id). The queue switches to it "
            "before the run starts. Omit for the currently active model."
        ),
    )
    folder: Optional[str] = Field(None, description="Workspace directory; created if missing. Default: output/agent_runs/<run_id>")
    autonomy: Optional[str] = Field("guided", description="manual | guided | full_auto")
    max_rounds: Optional[int] = Field(None, ge=1, le=100)
    max_tokens: Optional[int] = Field(
        12288, ge=256, le=32768,
        description=(
            "Completion budget per ROUND — what the model may WRITE, "
            "independent of context_window (what it may READ). On a "
            "reasoning model the think block is spent from this first, so a "
            "chat-sized cap is consumed entirely by deliberation and the "
            "round returns no content at all: 4096 was measured producing "
            "three empty rounds in a row on qwen3.8-27B at medium effort. "
            "Raise it rather than lowering effort when runs come back blank."
        ),
    )
    context_window: Optional[int] = Field(
        None, ge=2048,
        description=(
            "History window in tokens. Omit for the engine's FULL loaded "
            "context (the default since 2026-09-20 — it used to be a flat "
            "16384, which left most of a 72k split unused). Clamped to the "
            "loaded window; falls back to 16384 if the engine can't report "
            "its size."
        ),
    )
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    reasoning_effort: Optional[str] = Field(
        "medium",
        description="low|medium|high|xhigh. medium bounds thinking for unattended rounds; the template default (xhigh) can spend the whole budget thinking.",
    )
    plan: Optional[str] = Field(
        None,
        description="off | auto. auto: the model first splits the goal into steps and each step runs in a fresh context (core/agent_plan.py). Default off.",
    )
    attempts: Optional[int] = Field(
        None, ge=1, le=5,
        description="Best-of-N: attempt the goal N times from the same commit and keep the best (core/agent_attempts.py). Needs a clean git worktree. Default 1.",
    )


class ApprovalRequest(BaseModel):
    decision: str = Field(..., description="approve | always | deny | stop")


class RunMessageRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Follow-up user message for the run")


# ── Run state ────────────────────────────────────────────────────────────

def _serialize_event(ev: AgentEvent) -> dict:
    d = {"kind": ev.kind, "round": ev.round, "ts": round(time.time(), 2)}
    for k in ("text", "output", "reason", "summary"):
        v = getattr(ev, k, "")
        if v:
            d[k] = v[: _EVENT_TEXT_CAP]
    if ev.success is not None:
        d["success"] = ev.success
    if ev.action is not None:
        d["action"] = {
            "type": getattr(ev.action, "type", "?"),
            "display": str(getattr(ev.action, "display", ""))[:300],
        }
    if ev.decision is not None:
        risk = getattr(ev.decision, "risk_level", None)
        d["risk"] = getattr(risk, "name", str(risk)) if risk is not None else None
        pr = getattr(ev.decision, "reason", "")
        if pr:
            d["policy_reason"] = pr[:300]
    return d


class AgentRun:
    def __init__(self, goal: str, folder: str, config: RunConfig,
                 requested_ctx: int | None = None, model: str | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.goal = goal
        self.folder = folder
        self.config = config
        # The model the queue was switched to for this run; a revival
        # switches back to it, so a chat on another model in between doesn't
        # silently move the run.
        self.model = model
        # "MODEL: ..." line for the system prompt, and whether the engine can
        # see images. Filled in by the worker once the engine is loaded.
        self.model_info = ""
        self.vision: Optional[bool] = None
        # What the CALLER asked for: None = "the engine's full window".
        # Settled against the loaded engine in the worker, because resolving
        # it here would force a cold model load (30-60 s) inside the POST
        # handler that is supposed to return a run id immediately.
        self.requested_ctx = requested_ctx
        self.status = "starting"          # running | awaiting_approval | done | stopped:* | error:*
        self.summary = ""
        self.created = time.time()
        self.finished_at: Optional[float] = None
        self.control = RunControl()
        # "Always" rules the user granted; kept across revivals of this run.
        self.approvals = ApprovalMemory()
        self.history: list = []
        self.events: list[dict] = []      # persisted (non-chunk) events, indexable
        self._cond = threading.Condition()
        self._live_chunks: list[tuple[int, dict]] = []   # (serial, chunk) ring for live listeners
        self._chunk_serial = 0
        self.pending_approval: Optional[dict] = None
        self._approval_q: "queue.Queue[str]" = queue.Queue()
        self.thread: Optional[threading.Thread] = None

    # ── event sink (called from the worker thread) ──────────────────────
    def emit(self, ev: AgentEvent):
        d = _serialize_event(ev)
        with self._cond:
            if ev.kind in ("assistant_chunk", "thinking_chunk"):
                self._chunk_serial += 1
                self._live_chunks.append((self._chunk_serial, d))
                if len(self._live_chunks) > 400:
                    del self._live_chunks[:200]
            else:
                d["i"] = len(self.events)
                self.events.append(d)
            self._cond.notify_all()

    # ── approval bridge ─────────────────────────────────────────────────
    def request_approval(self, action, decision, reason) -> Decision:
        payload = {
            # The loop passes reason="" for ordinary action approvals; the
            # policy engine's justification lives on the decision. Without
            # this fallback a snapshot-polling client (no SSE) renders an
            # approval prompt with no explanation at all.
            "reason": reason or (getattr(decision, "reason", "") if decision else ""),
            "action": None if action is None else {
                "type": getattr(action, "type", "?"),
                "display": str(getattr(action, "display", ""))[:300],
            },
            "risk": (getattr(getattr(decision, "risk_level", None), "name", None)
                     if decision is not None else None),
        }
        with self._cond:
            self.pending_approval = payload
            self.status = "awaiting_approval"
            self._cond.notify_all()
        try:
            while True:
                try:
                    reply = self._approval_q.get(timeout=1.0)
                    break
                except queue.Empty:
                    if self.control.stop_requested:
                        return Decision.STOP
        finally:
            with self._cond:
                self.pending_approval = None
                if self.status == "awaiting_approval":
                    self.status = "running"
                self._cond.notify_all()
        return {"approve": Decision.APPROVE, "always": Decision.APPROVE_ALWAYS,
                "deny": Decision.DENY}.get(reply, Decision.STOP)

    def answer_approval(self, decision: str) -> bool:
        with self._cond:
            if self.pending_approval is None:
                return False
            # Claim the prompt while still holding the lock: a double-tapped
            # button POSTs twice, and without this the second decision would
            # queue up and be consumed as the answer to the NEXT prompt.
            # Restore status in the same critical section so no snapshot can
            # observe awaiting_approval with a null pending_approval.
            self.pending_approval = None
            if self.status == "awaiting_approval":
                self.status = "running"
            self._cond.notify_all()
        if decision == "stop":
            self.control.request_stop()
        self._approval_q.put(decision)
        return True

    def finish(self, status: str, summary: str = ""):
        with self._cond:
            self.status = status
            self.summary = summary
            self.finished_at = time.time()
            self._cond.notify_all()

    @property
    def terminal(self) -> bool:
        return self.status.startswith(("done", "stopped", "error"))

    def snapshot(self) -> dict:
        with self._cond:
            return {
                "run_id": self.id,
                "goal": self.goal,
                "folder": self.folder,
                "status": self.status,
                "summary": self.summary,
                "created": self.created,
                "finished_at": self.finished_at,
                "event_count": len(self.events),
                "pending_approval": self.pending_approval,
                # What the run actually got, once the engine reported its
                # loaded window. None until the worker has resolved it.
                "context_window": self.config.context_window,
                "max_tokens": self.config.max_tokens,
                "model": self.model,
                "vision": self.vision,
            }


# ── Worker ───────────────────────────────────────────────────────────────

# Used when the request names no context_window AND the engine cannot report
# its loaded size. Matches the old hard-coded request default.
_CTX_FALLBACK = 16384


def _resolve_context_window(engine, requested: int | None) -> int:
    """Settle the run's history window against the engine's loaded context.

    Omitted (None) means "all of it" — the point of the 72k split is that the
    window is usable, and a flat default silently threw two thirds of it
    away. An explicit request is honoured but clamped: asking for more than
    is loaded just means the pre-flight trimmer does the clamping later,
    noisily and per round.
    """
    try:
        loaded = engine.get_context_size() or 0
    except Exception as e:
        _log.warning("engine did not report a context size (%s)", e)
        loaded = 0

    if loaded <= 0:
        return requested or _CTX_FALLBACK
    if requested is None:
        return loaded
    return min(requested, loaded)


def live_run_busy_reason() -> str | None:
    """Busy check for the model queue: why the engine must stay put, or None."""
    with _runs_lock:
        live = next((r for r in _runs.values() if not r.terminal), None)
    if live is None:
        return None
    return f"agent run {live.id} is using the model"


def _engine_vision(engine) -> Optional[bool]:
    """True/False from the llama-server the engine talks to; None = can't tell.

    Asks the server (GET /props modalities.vision) rather than trusting the
    config name: the flag reflects the --mmproj the process was actually
    launched with, which is what decides whether an image gets seen.
    """
    base = getattr(engine, "_base_url", None)
    if not base:
        return None
    from tools.image_tools import check_vision_server
    ok, msg = check_vision_server(base)
    if ok:
        return True
    return False if "without vision" in msg else None


def _model_info_text(model: str | None, ctx: int, vision: Optional[bool]) -> str:
    """The MODEL section of the agent's system prompt.

    Without it the model has no idea what it is running as, and answers
    "which model are you / can you see images" with a guess — and it cannot
    tell whether @view_image will work before trying it.
    """
    name = model or "(unknown)"
    if vision is True:
        seeing = ("Vision: YES. You can look at images with @view_image and "
                  "@describe_images.")
    elif vision is False:
        seeing = ("Vision: NO. This model cannot see images, so @view_image and "
                  "@describe_images will fail. If the user asks about images, "
                  "tell them to pick a vision model (an entry with 'vision' in "
                  "its name) in the model menu and start a new run.")
    else:
        seeing = "Vision: unknown."
    return f"You are running as the model {name}, context window {ctx} tokens.\n{seeing}"


def _worker(run: AgentRun, get_engine, goal: str | None = None):
    goal = run.goal if goal is None else goal
    old_cwd = os.getcwd()
    try:
        engine = get_engine()
        resolved_ctx = _resolve_context_window(engine, run.requested_ctx)
        if resolved_ctx != run.config.context_window:
            _log.info("Agent run %s context window: requested=%s -> %d "
                      "(engine loaded ctx)", run.id,
                      run.requested_ctx if run.requested_ctx else "full",
                      resolved_ctx)
        run.config.context_window = resolved_ctx

        if not run.model:
            from core.config import get_active_model_name
            run.model = get_active_model_name()
        run.vision = _engine_vision(engine)
        run.model_info = _model_info_text(run.model, resolved_ctx, run.vision)
        seeing = {True: "can see images", False: "text only"}.get(run.vision, "vision unknown")
        run.emit(AgentEvent("model", text=f"Model: {run.model} ({seeing}, ctx {resolved_ctx})"))

        def build_system_prompt() -> str:
            # Same prompt stack the Qt GUI uses — the full tool catalog
            # (@read_file/@glob/@sysinfo/edit blocks/...) plus the sensed
            # environment. The earlier 4-line stub documented NO tools, so
            # phone-started runs guessed at their own capabilities.
            from core.prompts import build_assistant_prompt
            from tools.agent_tools import get_assistant_tools_prompt
            try:
                return build_assistant_prompt(
                    get_assistant_tools_prompt(),
                    os.getcwd(),
                    workspace_text=run.folder,
                    model_text=run.model_info,
                )
            except Exception:
                _log.exception("full prompt build failed; using minimal prompt")
                return (
                    "You are Artifex, an autonomous local agent running on "
                    f"the user's own PC.\nWORKSPACE: {run.folder}\n"
                    "You are already cd'd into the workspace; relative paths "
                    "resolve there."
                )

        os.chdir(run.folder)
        with run._cond:
            run.status = "running"
            run._cond.notify_all()
        runner = AgentRunner(
            engine,
            build_system_prompt=build_system_prompt,
            emit=run.emit,
            request_approval=run.request_approval,
            config=run.config,
            control=run.control,
            approvals=run.approvals,
        )
        result = runner.run(goal, run.history)
        run.finish(result.status, result.summary)
        _log.info("Agent run %s finished: %s (%d rounds, %d actions)",
                  run.id, result.status, result.rounds, result.actions_run)
    except Exception as e:
        _log.exception("Agent run %s crashed", run.id)
        run.finish(f"error:{type(e).__name__}", str(e)[:500])
    finally:
        try:
            os.chdir(old_cwd)
        except OSError:
            pass


def _prune_finished():
    """Keep the newest _MAX_KEPT_RUNS finished runs (call with _runs_lock held)."""
    finished = sorted((r for r in _runs.values() if r.terminal),
                      key=lambda r: r.created)
    for r in finished[: max(0, len(finished) - _MAX_KEPT_RUNS)]:
        _runs.pop(r.id, None)


# ── Route registration ───────────────────────────────────────────────────

def register_agent_routes(app, check_auth, get_engine, default_workspace_root: str,
                          prepare_model=None):
    """Wire the agent endpoints onto `app`.

    check_auth              — (Request) -> bool, the server's bearer check.
    get_engine              — () -> BaseEngine, loads/adopts the active engine.
    default_workspace_root  — directory under which per-run folders are made
                              when the request names none (gitignored output/).
    prepare_model           — async (requested: str | None) -> str. Resolves the
                              run's model and switches the model queue to it
                              (unloading a different model; the worker's
                              get_engine() then loads the right one). None =
                              run on whatever get_engine() returns.
    """

    def _auth(request: Request):
        if not check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

    def _live_conflict():
        with _runs_lock:
            live = next((r for r in _runs.values() if not r.terminal), None)
        if live is not None:
            raise HTTPException(
                status_code=409,
                detail={"message": "A run is already active", "run_id": live.id},
            )

    async def _prepare(requested: str | None) -> str | None:
        if prepare_model is None:
            return requested
        from core.model_queue import ModelBusyError
        try:
            return await prepare_model(requested)
        except HTTPException:
            raise
        except ModelBusyError as e:
            raise HTTPException(status_code=409, detail={"message": str(e)})
        except Exception as e:
            _log.exception("Agent model prepare failed for %r", requested)
            raise HTTPException(status_code=503,
                                detail=f"Could not switch to model {requested!r}: {e}")

    def _get_run(run_id: str) -> AgentRun:
        with _runs_lock:
            run = _runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="No such run")
        return run

    @app.post("/v1/agent/runs")
    async def start_run(body: AgentRunRequest, request: Request):
        _auth(request)

        try:
            autonomy = AutonomyLevel(body.autonomy or "guided")
        except ValueError:
            raise HTTPException(status_code=422,
                                detail="autonomy must be manual | guided | full_auto")
        effort = (body.reasoning_effort or "").strip().lower()
        if effort and effort not in ("low", "medium", "high", "xhigh"):
            raise HTTPException(status_code=422, detail="bad reasoning_effort")

        # Check for a live run BEFORE switching: switching would unload the
        # model under it. The lock can't be held across the await, so the
        # registration below re-checks.
        _live_conflict()
        model = await _prepare(body.model)

        with _runs_lock:
            live = next((r for r in _runs.values() if not r.terminal), None)
            if live is not None:
                raise HTTPException(
                    status_code=409,
                    detail={"message": "A run is already active", "run_id": live.id},
                )

            config = RunConfig.default(autonomy)
            if body.max_rounds:
                config.max_rounds = body.max_rounds
            config.max_tokens = body.max_tokens or 12288
            # Provisional: the worker replaces this with the engine's loaded
            # window (or the clamped request) before the loop starts.
            config.context_window = body.context_window or _CTX_FALLBACK
            config.temperature = body.temperature if body.temperature is not None else 0.7
            config.reasoning_effort = effort or "medium"
            if (body.plan or "").strip().lower() == "auto":
                config.plan = "auto"
            if body.attempts:
                config.attempts = body.attempts

            if body.folder:
                folder = os.path.abspath(os.path.expanduser(body.folder))
                if not os.path.isdir(folder):
                    # Documented as "created if missing" — a phone client has
                    # no way to mkdir on this box first.
                    try:
                        os.makedirs(folder, exist_ok=True)
                    except OSError as e:
                        raise HTTPException(
                            status_code=422,
                            detail=f"folder is not usable: {folder} ({e})")
                run = AgentRun(body.goal, folder, config,
                               requested_ctx=body.context_window, model=model)
            else:
                run = AgentRun(body.goal, "", config,
                               requested_ctx=body.context_window, model=model)
                run.folder = os.path.join(default_workspace_root, run.id)
                os.makedirs(run.folder, exist_ok=True)

            _runs[run.id] = run
            _prune_finished()

        run.thread = threading.Thread(
            target=_worker, args=(run, get_engine),
            name=f"agent-run-{run.id}", daemon=True,
        )
        run.thread.start()
        _log.info("Agent run %s started: model=%s autonomy=%s folder=%s ctx=%s goal=%r",
                  run.id, model, autonomy.value, run.folder,
                  body.context_window or "full", body.goal[:120])
        # context_window here is still provisional — the worker settles it
        # against the loaded engine. Clients wanting the real figure read it
        # from the snapshot (GET /v1/agent/runs/{id}) once status is running.
        return {"run_id": run.id, "folder": run.folder, "status": run.status,
                "model": model,
                "requested_context_window": body.context_window}

    @app.get("/v1/agent/runs")
    async def list_runs(request: Request):
        _auth(request)
        with _runs_lock:
            snaps = [r.snapshot() for r in _runs.values()]
        snaps.sort(key=lambda s: s["created"], reverse=True)
        return {"runs": snaps}

    @app.get("/v1/agent/runs/{run_id}")
    async def run_snapshot(run_id: str, request: Request):
        _auth(request)
        return _get_run(run_id).snapshot()

    @app.post("/v1/agent/runs/{run_id}/approval")
    async def answer_approval(run_id: str, body: ApprovalRequest, request: Request):
        _auth(request)
        run = _get_run(run_id)
        decision = body.decision.strip().lower()
        if decision not in ("approve", "always", "deny", "stop"):
            raise HTTPException(status_code=422,
                                detail="decision must be approve | always | deny | stop")
        if not run.answer_approval(decision):
            raise HTTPException(status_code=409, detail="Run is not awaiting approval")
        return {"ok": True, "decision": decision}

    @app.post("/v1/agent/runs/{run_id}/message")
    async def send_message(run_id: str, body: RunMessageRequest, request: Request):
        """Follow-up user message: steer a live run, or continue a finished one.

        Live run: the text is queued and picked up at the next round boundary
        (the loop emits a user_message event at the actual pickup point). A
        message queued in the run's final round is dropped when the run ends —
        the client sees the terminal status and can resend, which revives.

        Terminal run: the run restarts in place — same workspace, same history,
        same event stream (indices continue) — with the text as the new goal.
        Counts against the single-live-run rule like any other start.
        """
        _auth(request)
        text = body.text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="Empty message")
        run = _get_run(run_id)

        with _runs_lock:
            if not run.terminal:
                run.control.inject_message(text)
                return {"queued": True, "run_id": run.id, "status": run.status}

        # Reviving: put the run's own model back first — a chat on another
        # model may have swapped it out since the run finished.
        _live_conflict()
        model = await _prepare(run.model)

        with _runs_lock:
            if not run.terminal:
                # Another request revived it while we were switching.
                run.control.inject_message(text)
                return {"queued": True, "run_id": run.id, "status": run.status}
            live = next((r for r in _runs.values() if not r.terminal), None)
            if live is not None:
                raise HTTPException(
                    status_code=409,
                    detail={"message": "A run is already active", "run_id": live.id},
                )
            # Revive: reset control state and re-enter "starting" while the
            # registry lock is held so _prune_finished can't reap us mid-flip.
            run.control.reset()
            run.summary = ""
            run.finished_at = None
            run.status = "starting"
            run.model = model or run.model

        run.emit(AgentEvent("user_message", text=text, round=0))
        run.thread = threading.Thread(
            target=_worker, args=(run, get_engine), kwargs={"goal": text},
            name=f"agent-run-{run.id}", daemon=True,
        )
        run.thread.start()
        _log.info("Agent run %s revived with follow-up: %r", run.id, text[:120])
        return {"revived": True, "run_id": run.id, "status": run.status}

    @app.post("/v1/agent/runs/{run_id}/stop")
    async def stop_run(run_id: str, request: Request):
        _auth(request)
        run = _get_run(run_id)
        run.control.request_stop()
        # Unblock a pending approval wait, if any (harmless otherwise).
        run.answer_approval("stop")
        # The worker honors the stop asynchronously; echoing run.status here
        # would say "running" and read like the stop failed.
        return {"ok": True, "status": run.status if run.terminal else "stopping"}

    @app.get("/v1/agent/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request, since: int = 0):
        """SSE feed: replay persisted events from `since`, then stream live.

        Chunk events (assistant/thinking tokens) are live-only; a client
        that reconnects sees consolidated round events instead. The stream
        ends with an `end` event once the run is terminal and fully
        delivered. Comment lines every 15 s keep intermediaries from
        closing an idle stream.
        """
        _auth(request)
        run = _get_run(run_id)

        def gen():
            next_idx = max(0, since)
            last_chunk_serial = None  # skip pre-connect chunk backlog
            while True:
                out, done_now = [], False
                with run._cond:
                    if last_chunk_serial is None:
                        last_chunk_serial = run._chunk_serial
                    if next_idx >= len(run.events) and not run.terminal:
                        run._cond.wait(timeout=15.0)
                    while next_idx < len(run.events):
                        out.append(run.events[next_idx])
                        next_idx += 1
                    for serial, chunk in run._live_chunks:
                        if serial > last_chunk_serial:
                            out.append(chunk)
                            last_chunk_serial = serial
                    if run.terminal and next_idx >= len(run.events):
                        done_now = True
                if out:
                    for d in out:
                        yield f"data: {json.dumps(d)}\n\n"
                else:
                    yield ": ping\n\n"
                if done_now:
                    yield "data: " + json.dumps({
                        "kind": "end", "status": run.status,
                        "summary": run.summary,
                    }) + "\n\n"
                    return

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
