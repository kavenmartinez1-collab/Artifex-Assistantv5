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
    POST /v1/agent/runs/{id}/approval      {"decision": "approve"|"deny"|"stop"}
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
  via POST .../approval; a stop request unblocks it with Decision.STOP. No
  timeout: an unanswered prompt holds the run in "awaiting_approval"
  indefinitely, which is visible in the run list and resumable from the
  phone whenever the user comes back.
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
    folder: Optional[str] = Field(None, description="Workspace directory; created if missing. Default: output/agent_runs/<run_id>")
    autonomy: Optional[str] = Field("guided", description="manual | guided | full_auto")
    max_rounds: Optional[int] = Field(None, ge=1, le=100)
    max_tokens: Optional[int] = Field(4096, ge=256, le=32768)
    context_window: Optional[int] = Field(16384, ge=2048)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    reasoning_effort: Optional[str] = Field(
        "medium",
        description="low|medium|high|xhigh. medium bounds thinking for unattended rounds; the template default (xhigh) can spend the whole budget thinking.",
    )


class ApprovalRequest(BaseModel):
    decision: str = Field(..., description="approve | deny | stop")


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
    def __init__(self, goal: str, folder: str, config: RunConfig):
        self.id = uuid.uuid4().hex[:12]
        self.goal = goal
        self.folder = folder
        self.config = config
        self.status = "starting"          # running | awaiting_approval | done | stopped:* | error:*
        self.summary = ""
        self.created = time.time()
        self.finished_at: Optional[float] = None
        self.control = RunControl()
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
        return {"approve": Decision.APPROVE, "deny": Decision.DENY}.get(reply, Decision.STOP)

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
            }


# ── Worker ───────────────────────────────────────────────────────────────

def _worker(run: AgentRun, get_engine):
    old_cwd = os.getcwd()
    try:
        engine = get_engine()

        def build_system_prompt() -> str:
            return (
                "You are Artifex, an autonomous local agent running on the "
                "user's own Windows PC.\n"
                f"WORKSPACE: {run.folder}\n"
                "You are already cd'd into the workspace; relative paths "
                "resolve there. Prefer relative paths for files you create."
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
        )
        result = runner.run(run.goal, run.history)
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

def register_agent_routes(app, check_auth, get_engine, default_workspace_root: str):
    """Wire the agent endpoints onto `app`.

    check_auth              — (Request) -> bool, the server's bearer check.
    get_engine              — () -> BaseEngine, loads/adopts the active engine.
    default_workspace_root  — directory under which per-run folders are made
                              when the request names none (gitignored output/).
    """

    def _auth(request: Request):
        if not check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

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
            config.max_tokens = body.max_tokens or 4096
            config.context_window = body.context_window or 16384
            config.temperature = body.temperature if body.temperature is not None else 0.7
            config.reasoning_effort = effort or "medium"

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
                run = AgentRun(body.goal, folder, config)
            else:
                run = AgentRun(body.goal, "", config)
                run.folder = os.path.join(default_workspace_root, run.id)
                os.makedirs(run.folder, exist_ok=True)

            _runs[run.id] = run
            _prune_finished()

        run.thread = threading.Thread(
            target=_worker, args=(run, get_engine),
            name=f"agent-run-{run.id}", daemon=True,
        )
        run.thread.start()
        _log.info("Agent run %s started: autonomy=%s folder=%s goal=%r",
                  run.id, autonomy.value, run.folder, body.goal[:120])
        return {"run_id": run.id, "folder": run.folder, "status": run.status}

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
        if decision not in ("approve", "deny", "stop"):
            raise HTTPException(status_code=422, detail="decision must be approve | deny | stop")
        if not run.answer_approval(decision):
            raise HTTPException(status_code=409, detail="Run is not awaiting approval")
        return {"ok": True, "decision": decision}

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
