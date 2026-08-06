"""
Artifex Assistant V5 — WebGPU browser bridge.

A tiny stdlib HTTP server that lets the browser-side WebGPU engine
(webgpu/ app, usually on the second GPU) serve generation requests from
the Python framework. The browser cannot listen on TCP, so the direction
is inverted: the page long-polls this server for jobs and POSTs streamed
results back.

    Python (engine_webgpu)                Browser (webgpu/src/bridge.ts)
    ----------------------                ------------------------------
    submit_job(...)            <--GET---  /bridge/job?wait=25   (long-poll)
    events queue  <-----------POST------  /bridge/event  {token|done|error}
    session_info  <-----------POST------  /bridge/hello  {model, ctx, ...}

Design constraints honored:
  * stdlib only (http.server + threading) — no new dependencies.
  * binds 127.0.0.1 exclusively; CORS is wide-open because the socket
    itself is loopback-only (the browser page runs on a different local
    origin, e.g. 127.0.0.1:5173, so preflight must succeed).
  * event POST responses carry {"cancel": true} when Python wants the
    browser to abort the running generation.
"""

import json
import logging
import os
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

_log = logging.getLogger(__name__)

DEFAULT_BRIDGE_PORT = int(os.environ.get("ARTIFEX_WEBGPU_BRIDGE_PORT", "8790"))

# The browser is considered attached if it polled within this window.
CLIENT_STALE_S = 40.0
# Cap on how long a single /bridge/job long-poll may hold its thread.
MAX_POLL_WAIT_S = 30.0


class _Job:
    __slots__ = ("id", "payload", "events", "cancelled", "created")

    def __init__(self, payload: dict):
        self.id = uuid.uuid4().hex[:12]
        self.payload = payload
        self.events: "queue.Queue[dict]" = queue.Queue()
        self.cancelled = False
        self.created = time.monotonic()


class BridgeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.pending: "queue.Queue[_Job]" = queue.Queue()
        self.jobs: dict[str, _Job] = {}
        self.session_info: dict | None = None
        self.client_last_seen = 0.0

    # ── Python side ────────────────────────────────────────────────────

    def submit(self, payload: dict) -> _Job:
        job = _Job(payload)
        with self.lock:
            self.jobs[job.id] = job
        self.pending.put(job)
        return job

    def cancel(self, job_id: str):
        with self.lock:
            job = self.jobs.get(job_id)
            if job:
                job.cancelled = True

    def finish(self, job_id: str):
        with self.lock:
            self.jobs.pop(job_id, None)

    def client_attached(self) -> bool:
        return (time.monotonic() - self.client_last_seen) < CLIENT_STALE_S

    def is_ready(self) -> bool:
        info = self.session_info
        return self.client_attached() and bool(info) and bool(info.get("ready"))

    # ── Browser side (called from handler threads) ─────────────────────

    def touch(self):
        self.client_last_seen = time.monotonic()

    def next_job(self, wait_s: float) -> _Job | None:
        try:
            return self.pending.get(timeout=min(wait_s, MAX_POLL_WAIT_S))
        except queue.Empty:
            return None

    def push_event(self, event: dict) -> bool:
        """Route an event to its job's queue. Returns cancel flag."""
        job_id = event.get("id", "")
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            return True  # unknown/finished job — tell the browser to stop
        job.events.put(event)
        return job.cancelled


class _Handler(BaseHTTPRequestHandler):
    state: BridgeState = None  # set by server factory

    # Silence per-request stderr logging.
    def log_message(self, fmt, *args):  # noqa: D102
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _reply(self, code: int, body: dict | None = None):
        data = json.dumps(body).encode("utf-8") if body is not None else b""
        self.send_response(code)
        self._cors()
        if data:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)

    def _read_json(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > 64 * 1024 * 1024:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, OSError):
            return {}

    def do_OPTIONS(self):  # noqa: N802 — CORS preflight
        self._reply(204)

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        if url.path == "/bridge/job":
            self.state.touch()
            wait = 25.0
            try:
                wait = float(parse_qs(url.query).get("wait", ["25"])[0])
            except ValueError:
                pass
            job = self.state.next_job(wait)
            self.state.touch()
            if job is None:
                self._reply(204)
            else:
                self._reply(200, {"id": job.id, **job.payload})
        elif url.path == "/bridge/health":
            self._reply(200, {
                "ok": True,
                "client_attached": self.state.client_attached(),
                "session": self.state.session_info,
            })
        else:
            self._reply(404, {"error": "unknown path"})

    def do_POST(self):  # noqa: N802
        url = urlparse(self.path)
        body = self._read_json()
        if url.path == "/bridge/hello":
            self.state.touch()
            self.state.session_info = body or None
            _log.info("WebGPU bridge hello: %s", body)
            self._reply(200, {"ok": True})
        elif url.path == "/bridge/event":
            self.state.touch()
            cancel = self.state.push_event(body)
            self._reply(200, {"cancel": cancel})
        else:
            self._reply(404, {"error": "unknown path"})


class BridgeServer:
    """Owns the ThreadingHTTPServer + its state. One per process."""

    def __init__(self, port: int = DEFAULT_BRIDGE_PORT):
        self.port = port
        self.state = BridgeState()
        handler = type("BoundHandler", (_Handler,), {"state": self.state})
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self.port = self._httpd.server_address[1]  # resolves port=0 for tests
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="webgpu-bridge", daemon=True)
        self._thread.start()
        _log.info("WebGPU bridge listening on 127.0.0.1:%d", self.port)

    def close(self):
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        except Exception:  # noqa: BLE001 — best-effort teardown
            pass


_singleton: BridgeServer | None = None
_singleton_lock = threading.Lock()


def get_bridge(port: int = DEFAULT_BRIDGE_PORT) -> BridgeServer:
    """Process-wide bridge server (GUI chat + agent loop share one)."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = BridgeServer(port)
        return _singleton
