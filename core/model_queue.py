"""
Model-Aware Request Queue for Ollama Backend.

Serializes inference requests so that:
  1. All requests for the same model are batched together
  2. Model switches happen cleanly (unload → load → process)
  3. Requests for a different model wait instead of failing
  4. VRAM is properly freed between model switches

Usage:
    queue = ModelQueue()
    result = await queue.submit("my-model", generate_fn, *args)
"""

import asyncio
import json
import logging
import os
import time
import urllib.request
import urllib.error
from collections import deque
from threading import Lock

_log = logging.getLogger(__name__)

OLLAMA_BASE = os.environ.get("ARTIFEX_OLLAMA_URL", "http://localhost:11434").rstrip("/")

# ── Idle shrink ────────────────────────────────────────────────────────
# Auto-release the loaded engine after this many seconds of inactivity so
# VRAM goes back to the user for whatever else they want to run on the GPU
# (other AI workloads, games, browsers, etc.).  Only acts on llama_cpp;
# Ollama and Transformers each manage their own lifecycles.
IDLE_SHRINK_SEC = 600
IDLE_SHRINK_CHECK_INTERVAL = 60


class ModelQueue:
    """Serialized model-aware request queue for all backends.

    Ensures only one model is loaded at a time. Requests for a different
    model wait until the current model's requests are drained, then the
    queue unloads the old model and loads the new one.

    Works with both Ollama (HTTP unload) and Transformers (engine.unload + reload).
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._current_model: str | None = None
        self._current_backend: str | None = None
        # Live launch ctx for the loaded llama_cpp engine.  Used to detect
        # tier changes that require a relaunch (llama-server's -c is fixed
        # at launch and can't grow without restarting the process).
        self._current_ctx_tier: int | None = None
        # Wall-clock of the last switch_if_needed call.  Drives idle shrink.
        self._last_request_at: float | None = None
        self._idle_shrink_task: asyncio.Task | None = None
        self._engine_unload_fn = None  # registered by api layer
        self._stats = {
            "total_requests": 0,
            "model_switches": 0,
            "queued": 0,
        }

    def register_transformers_unload(self, fn):
        """Register a callback to unload the current engine.

        Despite the legacy name, this callback is used for both
        Transformers and llama.cpp engines — BaseEngine.unload() is
        polymorphic (frees GPU memory or kills the server process).

        Called by the API layer at startup so model_queue doesn't need
        to import api.server directly.
        """
        self._engine_unload_fn = fn

    @property
    def current_model(self) -> str | None:
        return self._current_model

    @property
    def current_ctx_tier(self) -> int | None:
        return self._current_ctx_tier

    @property
    def stats(self) -> dict:
        return {
            **self._stats,
            "current_model": self._current_model,
            "current_backend": self._current_backend,
            "current_ctx_tier": self._current_ctx_tier,
        }

    async def switch_if_needed(
        self, model: str, backend: str, ctx_tier: int | None = None,
    ):
        """Switch model/backend/ctx_tier if the request needs a different config.

        Call this while holding the lock (inside guarded_stream).

        Args:
            model: Target model name.
            backend: Target backend ("ollama", "llama_cpp", "transformers").
            ctx_tier: For llama_cpp only — the launch ctx for llama-server's
                -c flag.  A tier change within the same model triggers a
                relaunch because llama-server's launch ctx is fixed once the
                process starts.  Ignored for other backends.
        """
        needs_switch = (
            self._current_model != model
            or self._current_backend != backend
        )
        # Tier INCREASE within the same llama_cpp model requires a relaunch
        # (you can't grow -c without restarting llama-server). A smaller
        # request after a big one keeps the larger ctx: relaunching downward
        # buys nothing (KV VRAM was already committed and fits) and costs a
        # full cold load — minutes with --no-mmap — on every size ping-pong.
        needs_tier_relaunch = (
            not needs_switch
            and backend == "llama_cpp"
            and ctx_tier is not None
            and self._current_ctx_tier is not None
            and ctx_tier > self._current_ctx_tier
        )

        if needs_switch or needs_tier_relaunch:
            if needs_tier_relaunch:
                _log.info(
                    "Queue: ctx_tier change for %s/%s: %d → %d (relaunch)",
                    backend, model, self._current_ctx_tier, ctx_tier,
                )
            else:
                _log.info(
                    "Queue: switching %s/%s → %s/%s (tier=%s)",
                    self._current_backend, self._current_model,
                    backend, model, ctx_tier,
                )

            if self._current_backend is not None:
                if self._current_backend == "ollama":
                    await self._unload_ollama(self._current_model)
                # Always clear the api layer's cached engine when
                # switching from any active backend.  For
                # transformers/llama_cpp this also frees GPU memory; for
                # ollama it's bookkeeping — without it, the next
                # _get_engine() call returns the stale OllamaEngine
                # instance even after a cross-backend swap, and the
                # request silently runs against Ollama instead of the
                # new backend.
                await self._unload_engine()

            # If backend changed, the engine needs to be recreated
            if self._current_backend != backend:
                from core.config import set_active_backend
                set_active_backend(backend)

            self._stats["model_switches"] += 1

        # Set model for the new request
        if model != self._current_model:
            from core.config import set_active_model
            set_active_model(model)

        self._current_model = model
        self._current_backend = backend
        # On a skipped downgrade the server still runs the LARGER ctx —
        # record that, or the next mid-size request "upgrades" from the
        # smaller recorded tier and relaunches for nothing.
        if (not needs_switch and not needs_tier_relaunch
                and self._current_ctx_tier is not None and ctx_tier is not None):
            self._current_ctx_tier = max(self._current_ctx_tier, ctx_tier)
        else:
            self._current_ctx_tier = ctx_tier
        self._last_request_at = time.time()
        self._stats["total_requests"] += 1

        self._ensure_idle_shrink_task()

    def _ensure_idle_shrink_task(self):
        """Lazy-start the idle shrink loop on first activity.

        We start it from inside an async context (switch_if_needed) so the
        running event loop is guaranteed to exist.  Skipped silently when
        no loop is running (e.g. unit tests that exercise the queue
        without an asyncio.run wrapper).
        """
        if self._idle_shrink_task is not None and not self._idle_shrink_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._idle_shrink_task = loop.create_task(self._idle_shrink_loop())

    async def _idle_shrink_loop(self):
        """Periodically release the loaded engine if it's been idle long enough.

        Only acts on llama_cpp because Ollama and Transformers manage their
        own lifecycles and unloading them through this path would interfere
        with their internal state.  Acquires the queue lock so the shrink
        never races with an in-flight switch_if_needed.
        """
        while True:
            try:
                await asyncio.sleep(IDLE_SHRINK_CHECK_INTERVAL)
                async with self._lock:
                    if (
                        self._current_backend == "llama_cpp"
                        and self._last_request_at is not None
                        and time.time() - self._last_request_at > IDLE_SHRINK_SEC
                    ):
                        idle_for = time.time() - self._last_request_at
                        _log.info(
                            "Queue: idle %.0fs > %ds — releasing %s/%s "
                            "(tier=%s) to free VRAM",
                            idle_for, IDLE_SHRINK_SEC,
                            self._current_backend, self._current_model,
                            self._current_ctx_tier,
                        )
                        await self._unload_engine()
                        self._current_model = None
                        self._current_backend = None
                        self._current_ctx_tier = None
                        self._last_request_at = None
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _log.warning("Idle shrink loop error (continuing): %s", e)

    async def _unload_ollama(self, model: str):
        """Unload a model from Ollama to free VRAM."""
        if not model:
            return
        try:
            payload = json.dumps({
                "model": model,
                "keep_alive": 0,
            }).encode()

            req = urllib.request.Request(
                f"{OLLAMA_BASE}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=30))

            await asyncio.sleep(2)
            _log.info("Unloaded Ollama model %s", model)
        except Exception as e:
            _log.warning("Failed to unload Ollama model %s: %s", model, e)

    async def _unload_engine(self):
        """Tear down the api layer's cached engine instance.

        Uses a registered callback so core/ doesn't import api/.  The
        callback calls BaseEngine.unload() — polymorphic across backends:
        TransformersEngine frees GPU memory, LlamaCppEngine kills the
        llama-server process, OllamaEngine is a no-op (Ollama itself
        already released the model via HTTP).  In all cases it then
        clears the cached engine so the next request creates a fresh
        instance for the now-active backend/model.
        """
        if self._engine_unload_fn is None:
            _log.warning("No engine unload callback registered — skipping")
            return
        try:
            _log.info("Unloading %s engine...", self._current_backend)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._engine_unload_fn)
            await asyncio.sleep(1)
            _log.info("%s engine unloaded", self._current_backend)
        except Exception as e:
            _log.warning("Failed to unload %s engine: %s", self._current_backend, e)

    async def unload_current(self):
        """Manually unload the current model (e.g. on shutdown)."""
        async with self._lock:
            if self._current_model:
                if self._current_backend == "ollama":
                    await self._unload_ollama(self._current_model)
                elif self._current_backend in ("transformers", "llama_cpp"):
                    await self._unload_engine()
                self._current_model = None
                self._current_backend = None
                self._current_ctx_tier = None
                self._last_request_at = None


# Global singleton
_model_queue: ModelQueue | None = None


def get_model_queue() -> ModelQueue:
    """Get or create the global model queue."""
    global _model_queue
    if _model_queue is None:
        _model_queue = ModelQueue()
    return _model_queue
