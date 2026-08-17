"""
Artifex Assistant V5 — Persistent chat jobs.

Chat generation that survives the client: the phone fires a message, locks
its screen, and the answer is still there later. Same reattach pattern the
agent runs use, applied to plain chat.

    POST /v1/chat/jobs                  start a job -> {job_id}
    GET  /v1/chat/jobs/{id}             snapshot: status + accumulated text
    GET  /v1/chat/jobs/{id}/events      SSE: snapshot first, then live deltas
    POST /v1/chat/jobs/{id}/stop        stop delivering (see caveat below)

Design notes:

- The worker thread runs the SAME _stream_with_tools generator the
  streaming endpoint uses (tool rounds, x_ extensions, usage) inside a
  private asyncio loop, and parses its SSE lines back into payload dicts.
  Slightly inelegant, but zero duplication of the tool-loop logic and
  zero risk to the existing streaming path.

- Reattach is snapshot-then-live: a joining listener first gets one
  {"x_snapshot": {...}} event carrying ALL accumulated response/thinking/
  tool-status text, then live passthrough payloads in the normal chat
  format. No replay buffer of thousands of deltas.

- On completion the job writes the finished exchange into the session
  store server-side (session_api.write_session), so the conversation
  persists even if no client ever reattaches. A client that does
  reattach and PUTs the same history is harmless (full-replace, same
  content).

- ONE live job at a time (409 with the live id), mirroring agent runs —
  the GPU serves one stream anyway.

- Stop caveat: _stream_with_tools has no cancel hook, so stop marks the
  job stopped and detaches from the generator; the engine drains its
  current round in the background. Delivered text stops immediately.

- Engine/queue: like the agent worker, jobs use _get_engine() directly
  (the model-queue asyncio lock belongs to the server's main loop and
  cannot be taken from a private one). Model/tier switching therefore
  follows whatever is or gets loaded — on this single-model deployment
  that is the same behavior the agent runs have always had.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.logging_config import get_logger

_log = get_logger(__name__)

_MAX_KEPT_JOBS = 10

_jobs: dict[str, "ChatJob"] = {}
_jobs_lock = threading.Lock()


class ChatJobRequest(BaseModel):
    messages: list = Field(..., min_length=1)
    model: Optional[str] = None
    session_id: Optional[str] = Field(None, description="Session to append the finished exchange to")
    web_tools: Optional[bool] = False
    max_tokens: Optional[int] = Field(12288, ge=64, le=32768)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    reasoning_effort: Optional[str] = None


def _flatten_image_content(m: dict) -> dict:
    """Collapse an OpenAI multimodal message to a text+marker string before
    it reaches the session store — base64 images belong in the live turn,
    never persisted (mirrors the client's flattenContent; keeps sessions
    under the 2 MB cap). Plain-string messages pass through untouched.
    """
    c = m.get("content")
    if not isinstance(c, list):
        return {"role": m.get("role"), "content": c}
    parts = []
    for item in c:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            parts.append(item.get("text", ""))
        elif item.get("type") == "image_url":
            parts.append("[image]")
    return {"role": m.get("role"), "content": " ".join(p for p in parts if p).strip()}


class ChatJob:
    def __init__(self, req: ChatJobRequest):
        self.id = uuid.uuid4().hex[:12]
        self.created = time.time()
        self.status = "running"          # running | done | error | stopped
        self.req = req
        self.text = ""                   # accumulated visible response
        self.thinking = ""               # accumulated thinking text
        self.tool_status: list[str] = []
        self.usage: dict = {}
        self.error = ""
        self.finished_at: Optional[float] = None
        self._cond = threading.Condition()
        self._live: list[tuple[int, dict]] = []   # (serial, payload) ring
        self._serial = 0
        self._stop = threading.Event()

    @property
    def terminal(self) -> bool:
        return self.status in ("done", "error", "stopped")

    def push(self, payload: dict):
        """Absorb one parsed stream payload; buffer it for live listeners."""
        with self._cond:
            try:
                # x_ extensions appear top-level for some kinds
                # (x_tool_status) and inside choices[0].delta for others
                # (x_thinking) — accept both everywhere.
                choices = payload.get("choices") or []
                delta = (choices[0].get("delta") or {}) if choices else {}
                th = payload.get("x_thinking") or delta.get("x_thinking")
                if th:
                    self.thinking += th
                ts = payload.get("x_tool_status") or delta.get("x_tool_status")
                if ts:
                    self.tool_status.extend(ts)
                us = payload.get("x_usage") or delta.get("x_usage")
                if us:
                    self.usage = us
                content = delta.get("content")
                if content:
                    self.text += content
            except Exception:
                pass
            self._serial += 1
            self._live.append((self._serial, payload))
            if len(self._live) > 400:
                del self._live[:200]
            self._cond.notify_all()

    def finish(self, status: str, error: str = ""):
        with self._cond:
            self.status = status
            self.error = error
            self.finished_at = time.time()
            self._cond.notify_all()

    def snapshot(self) -> dict:
        with self._cond:
            return {
                "job_id": self.id,
                "status": self.status,
                "created": self.created,
                "finished_at": self.finished_at,
                "session_id": self.req.session_id,
                "model": self.req.model,
                "text": self.text,
                "thinking": self.thinking,
                "tool_status": list(self.tool_status),
                "usage": self.usage,
                "error": self.error,
            }


def _worker(job: ChatJob, stream_factory, model: str, session_dir: str):
    """Consume the chat stream inside a private event loop."""
    async def consume():
        gen = stream_factory(job.req, model)
        try:
            async for chunk in gen:
                if job._stop.is_set():
                    break
                for line in str(chunk).splitlines():
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        continue
                    try:
                        job.push(json.loads(data))
                    except json.JSONDecodeError:
                        continue
        finally:
            try:
                await gen.aclose()
            except Exception:
                pass

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(consume())
        job.finish("stopped" if job._stop.is_set() else "done")
    except Exception as e:
        _log.exception("Chat job %s crashed", job.id)
        job.finish("error", str(e)[:500])
    finally:
        loop.close()

    # Persist the finished exchange server-side so the conversation exists
    # even if no client ever reattaches.
    if job.status == "done" and job.req.session_id and job.text:
        try:
            from api.session_api import write_session
            history = [_flatten_image_content(m) for m in job.req.messages
                       if isinstance(m, dict) and m.get("role") != "system"]
            history = history + [{"role": "assistant", "content": job.text}]
            ok = write_session(session_dir, job.req.session_id, history,
                               model=job.req.model)
            _log.info("Chat job %s: session %s write %s",
                      job.id, job.req.session_id, "ok" if ok else "FAILED")
        except Exception:
            _log.exception("Chat job %s: session write failed", job.id)

    with _jobs_lock:
        finished = sorted((j for j in _jobs.values() if j.terminal),
                          key=lambda j: j.created)
        for j in finished[: max(0, len(finished) - _MAX_KEPT_JOBS)]:
            _jobs.pop(j.id, None)


def register_chat_job_routes(app, check_auth, prepare, stream_factory, session_dir: str):
    """Wire the chat-job endpoints.

    prepare        — async (ChatJobRequest) -> str, run on the MAIN event
        loop before the worker spawns: resolves the model (image-bearing
        requests route to a vision model), switches the model queue, and
        loads the engine. Returns the resolved model id. This MUST happen
        on the main loop because the queue's lock is bound to it — the
        worker's private loop can't take it, which is why a vision job
        previously ran against whatever model was already loaded.
    stream_factory — (ChatJobRequest, model) -> async generator of SSE
        chunk strings; shares the server's _stream_with_tools pipeline.
    """

    def _auth(request: Request):
        if not check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

    def _get(job_id: str) -> ChatJob:
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job")
        return job

    @app.post("/v1/chat/jobs")
    async def start_job(body: ChatJobRequest, request: Request):
        _auth(request)
        with _jobs_lock:
            live = next((j for j in _jobs.values() if not j.terminal), None)
            if live is not None:
                raise HTTPException(
                    status_code=409,
                    detail={"message": "A chat job is already running",
                            "job_id": live.id})
            job = ChatJob(body)
            _jobs[job.id] = job
        # Switch/load the right engine on the MAIN loop before the worker
        # runs (image requests route to a vision model here). A failure
        # marks the job errored rather than silently generating on the
        # wrong model.
        try:
            resolved_model = await prepare(body)
        except Exception as e:
            _log.exception("Chat job %s prepare failed", job.id)
            job.finish("error", f"engine prepare failed: {e}"[:500])
            return {"job_id": job.id, "status": job.status}
        threading.Thread(target=_worker,
                         args=(job, stream_factory, resolved_model, session_dir),
                         name=f"chat-job-{job.id}", daemon=True).start()
        _log.info("Chat job %s started (model=%s, session=%s, web=%s)",
                  job.id, resolved_model, body.session_id, bool(body.web_tools))
        return {"job_id": job.id, "status": job.status}

    @app.get("/v1/chat/jobs/{job_id}")
    async def job_snapshot(job_id: str, request: Request):
        _auth(request)
        return _get(job_id).snapshot()

    @app.post("/v1/chat/jobs/{job_id}/stop")
    async def stop_job(job_id: str, request: Request):
        _auth(request)
        job = _get(job_id)
        job._stop.set()
        with job._cond:
            job._cond.notify_all()
        return {"ok": True, "status": job.status if job.terminal else "stopping"}

    @app.get("/v1/chat/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request):
        """SSE: one x_snapshot event with everything so far, then live."""
        _auth(request)
        job = _get(job_id)

        def gen():
            with job._cond:
                snap = job.snapshot()
                last_serial = job._serial
            yield "data: " + json.dumps({"x_snapshot": snap}) + "\n\n"
            if snap["status"] != "running":
                yield "data: [DONE]\n\n"
                return
            while True:
                out = []
                with job._cond:
                    if job._serial == last_serial and not job.terminal:
                        job._cond.wait(timeout=15.0)
                    for serial, payload in job._live:
                        if serial > last_serial:
                            out.append(payload)
                            last_serial = serial
                    done_now = job.terminal and job._serial == last_serial
                if out:
                    for p in out:
                        yield "data: " + json.dumps(p) + "\n\n"
                else:
                    yield ": ping\n\n"
                if done_now:
                    yield "data: " + json.dumps(
                        {"x_job_end": {"status": job.status, "error": job.error,
                                       "usage": job.usage}}) + "\n\n"
                    yield "data: [DONE]\n\n"
                    return

        return StreamingResponse(
            gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
