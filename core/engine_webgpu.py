"""
Artifex Assistant V5 — WebGPU browser engine.

Drives the browser-side WebGPU inference engine (webgpu/ app — typically
running the second GPU, e.g. the RX 6700 XT) through core.webgpu_bridge.
The browser page long-polls the bridge for jobs and streams tokens back,
so from the framework's point of view this is a normal BaseEngine: the
Qt GUI chat, the CLI, and the autonomous agent loop can all select it.

The model itself is loaded IN THE BROWSER (the page owns the GPU device
and file handles); load() here just waits for an attached page with a
ready session. Model switching happens in the WebGPU GUI, not here.

Sampling: accepts the core.sampling explicit dicts and maps llama-server
key names onto the WebGPU engine's GenerateOptions names. When sampling
is None, DEFAULT_SAMPLING is mapped — the browser engine's own fallback
defaults never apply, same policy as the llama.cpp engine.
"""

import logging
import os
import queue
import time

from core.engine_base import BaseEngine
from core.inference import _clean_response, strip_think_blocks
from core.webgpu_bridge import get_bridge, DEFAULT_BRIDGE_PORT

_log = logging.getLogger(__name__)

DEFAULT_HANDSHAKE_TIMEOUT = float(os.environ.get("ARTIFEX_WEBGPU_HANDSHAKE_TIMEOUT", "120"))

# Seconds with no event from the browser before the request is declared dead.
EVENT_TIMEOUT_S = 120.0
# Overall per-request wall clock.
REQUEST_TIMEOUT_S = 900.0
# How often the event wait re-checks liveness/timeouts (short in tests).
EVENT_POLL_S = 5.0

# llama-server-style key (core.sampling) -> webgpu GenerateOptions key.
_SAMPLING_KEY_MAP = {
    "temperature": "temperature",
    "top_k": "topK",
    "top_p": "topP",
    "min_p": "minP",
    "repeat_penalty": "repetitionPenalty",
    "dry_multiplier": "dryMultiplier",
    "dry_base": "dryBase",
    "dry_allowed_length": "dryAllowedLength",
    "dry_penalty_last_n": "dryRangeLastN",
}

_FINISH_REASON_MAP = {"eos": "stop", "max_length": "length", "aborted": "abort"}


def map_sampling(sampling: dict | None, temperature: float) -> dict:
    """Translate an explicit core.sampling dict to WebGPU option names."""
    from core.sampling import DEFAULT_SAMPLING
    src = dict(DEFAULT_SAMPLING) if sampling is None else dict(sampling)
    src.setdefault("temperature", temperature)
    out = {}
    for key, value in src.items():
        mapped = _SAMPLING_KEY_MAP.get(key)
        if mapped is not None:
            out[mapped] = value
    return out


class WebGpuEngine(BaseEngine):
    """Browser-hosted WebGPU backend, reached via the local bridge."""

    # WebGPU chat templates pre-fill the assistant turn's <think> opener
    # (same convention as the transformers path), so with thinking enabled
    # the stream begins INSIDE a think block with no opening tag.
    stream_starts_in_think = True

    def __init__(self, port: int = DEFAULT_BRIDGE_PORT,
                 handshake_timeout: float = DEFAULT_HANDSHAKE_TIMEOUT,
                 bridge=None):
        self.port = port
        self._handshake_timeout = handshake_timeout
        self._bridge = bridge  # injectable for tests; else process singleton
        self._last_gen_stats = {}

    # ── lifecycle ──────────────────────────────────────────────────────

    def _ensure_bridge(self):
        if self._bridge is None:
            self._bridge = get_bridge(self.port)
        return self._bridge

    def load(self, status_callback=None):
        bridge = self._ensure_bridge()
        if bridge.state.is_ready():
            return
        if status_callback:
            status_callback(
                "Waiting for the WebGPU browser session — open the WebGPU "
                "GUI (control-center) and load a model. The page attaches "
                f"to the bridge on port {bridge.port} automatically."
            )
        deadline = time.monotonic() + self._handshake_timeout
        while time.monotonic() < deadline:
            if bridge.state.is_ready():
                info = bridge.state.session_info or {}
                _log.info("WebGPU session attached: %s", info)
                if status_callback:
                    status_callback(
                        f"WebGPU ready — {info.get('model', '?')} "
                        f"(ctx={info.get('ctx', '?')})")
                return
            time.sleep(0.5)
        raise TimeoutError(
            "No WebGPU browser session attached within "
            f"{self._handshake_timeout:.0f}s. Open the WebGPU GUI in Chrome "
            "(control-center serves it on 127.0.0.1:5173), load a model, "
            "and try again."
        )

    def unload(self, status_callback=None):
        # The browser owns the model + VRAM; nothing to free here.
        if status_callback:
            status_callback("WebGPU bridge detached (browser keeps the model).")

    def is_loaded(self) -> bool:
        bridge = self._ensure_bridge()
        return bridge.state.is_ready()

    def needs_reload(self) -> bool:
        return False

    def periodic_cleanup(self):
        pass

    def get_context_size(self) -> int:
        bridge = self._ensure_bridge()
        info = bridge.state.session_info or {}
        try:
            return int(info.get("ctx") or 0)
        except (TypeError, ValueError):
            return 0

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    # ── generation ─────────────────────────────────────────────────────

    def generate_streaming(self, messages, max_tokens, temperature,
                           on_token=None, on_complete=None,
                           enable_thinking=True,
                           grammar=None, response_format=None,
                           raw_output=False,
                           web_tools=False,
                           reasoning_effort=None,
                           on_telemetry=None,
                           sampling=None) -> str:
        """Run one chat generation in the attached browser session.

        grammar/response_format are not supported by the WebGPU engine and
        are accepted-and-ignored (logged) per the BaseEngine contract;
        web_tools likewise (handled by the api/server.py post-processor).
        reasoning_effort is also accepted and ignored — the in-browser
        runtime has no per-request effort control.
        """
        self.load()
        if grammar or response_format:
            _log.warning("WebGPU engine ignores grammar/response_format")
        self._last_gen_stats = {}

        bridge = self._ensure_bridge()
        payload = {
            "kind": "chat",
            "messages": list(messages),
            "sampling": {
                **map_sampling(sampling, temperature),
                "maxNewTokens": int(max_tokens) if max_tokens and max_tokens > 0 else 2048,
            },
            "enableThinking": bool(enable_thinking),
        }
        job = bridge.state.submit(payload)

        full_text = ""
        started = time.monotonic()
        last_event = started
        try:
            while True:
                if time.monotonic() - started > REQUEST_TIMEOUT_S:
                    bridge.state.cancel(job.id)
                    raise TimeoutError("WebGPU generation exceeded request timeout")
                try:
                    event = job.events.get(timeout=EVENT_POLL_S)
                except queue.Empty:
                    if not bridge.state.client_attached():
                        raise ConnectionError(
                            "WebGPU browser page detached mid-generation "
                            "(closed tab / crashed page?)")
                    # Liveness keys off events (the page heartbeats every
                    # ~8 s while generating — long prefills are silent
                    # token-wise but not event-wise).
                    if time.monotonic() - last_event > EVENT_TIMEOUT_S:
                        bridge.state.cancel(job.id)
                        raise TimeoutError(
                            "WebGPU page stopped responding mid-generation "
                            "(no events, page still polling?)")
                    continue

                last_event = time.monotonic()
                etype = event.get("type")
                if etype == "ping":
                    continue
                if etype == "token":
                    piece = event.get("text", "")
                    if piece:
                        full_text += piece
                        if on_token:
                            on_token(piece)
                elif etype == "done":
                    stats = event.get("stats") or {}
                    self._last_gen_stats = {
                        "prompt_tokens": stats.get("promptTokens"),
                        "completion_tokens": stats.get("numTokens"),
                        "predicted_per_second": stats.get("tokensPerSecond"),
                        "finish_reason": _FINISH_REASON_MAP.get(
                            stats.get("stopReason", ""), stats.get("stopReason")),
                    }
                    break
                elif etype == "error":
                    raise RuntimeError(
                        f"WebGPU generation failed in browser: "
                        f"{event.get('error', 'unknown error')}")
        finally:
            bridge.state.finish(job.id)

        if raw_output:
            clean = strip_think_blocks(full_text)
        else:
            clean = _clean_response(full_text)
        if on_complete:
            on_complete(clean)
        return clean
