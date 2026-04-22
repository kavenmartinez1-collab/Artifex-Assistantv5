"""
Model-Aware Request Queue for Ollama Backend.

Serializes inference requests so that:
  1. All requests for the same model are batched together
  2. Model switches happen cleanly (unload → load → process)
  3. Requests for a different model wait instead of failing
  4. VRAM is properly freed between model switches

Usage:
    queue = ModelQueue()
    result = await queue.submit("qwen3.6-27b-q2kxl", generate_fn, *args)
"""

import asyncio
import logging
import time
import urllib.request
import urllib.error
import json
from collections import deque
from threading import Lock

_log = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"


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
        self._transformers_unload_fn = None  # registered by api layer
        self._stats = {
            "total_requests": 0,
            "model_switches": 0,
            "queued": 0,
        }

    def register_transformers_unload(self, fn):
        """Register a callback to unload the transformers engine.

        Called by the API layer at startup so model_queue doesn't need
        to import api.server directly.

        Args:
            fn: callable() that unloads the current transformers engine
        """
        self._transformers_unload_fn = fn

    @property
    def current_model(self) -> str | None:
        return self._current_model

    @property
    def stats(self) -> dict:
        return {
            **self._stats,
            "current_model": self._current_model,
            "current_backend": self._current_backend,
        }

    async def switch_if_needed(self, model: str, backend: str):
        """Switch model/backend if the request targets a different one.

        Call this while holding the lock (inside guarded_stream).
        Handles unloading the old model for both Ollama and Transformers.
        """
        needs_switch = (
            (self._current_model and model != self._current_model)
            or (self._current_backend and backend != self._current_backend)
        )

        if needs_switch:
            _log.info(
                "Queue: switching %s/%s → %s/%s",
                self._current_backend, self._current_model, backend, model,
            )

            if self._current_backend == "ollama":
                await self._unload_ollama(self._current_model)
            elif self._current_backend == "transformers":
                await self._unload_transformers()

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
        self._stats["total_requests"] += 1

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

    async def _unload_transformers(self):
        """Unload the transformers engine to free VRAM.

        Uses a registered callback instead of directly importing api.server,
        keeping core/ decoupled from the API layer.
        """
        if self._transformers_unload_fn is None:
            _log.warning("No transformers unload callback registered — skipping")
            return
        try:
            _log.info("Unloading transformers model...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._transformers_unload_fn)
            await asyncio.sleep(1)
            _log.info("Transformers model unloaded")
        except Exception as e:
            _log.warning("Failed to unload transformers: %s", e)

    async def unload_current(self):
        """Manually unload the current model (e.g. on shutdown)."""
        async with self._lock:
            if self._current_model:
                if self._current_backend == "ollama":
                    await self._unload_ollama(self._current_model)
                elif self._current_backend == "transformers":
                    await self._unload_transformers()
                self._current_model = None
                self._current_backend = None


# Global singleton
_model_queue: ModelQueue | None = None


def get_model_queue() -> ModelQueue:
    """Get or create the global model queue."""
    global _model_queue
    if _model_queue is None:
        _model_queue = ModelQueue()
    return _model_queue
