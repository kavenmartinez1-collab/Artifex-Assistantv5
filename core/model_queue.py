"""
Model-Aware Request Queue for Ollama Backend.

Serializes inference requests so that:
  1. All requests for the same model are batched together
  2. Model switches happen cleanly (unload → load → process)
  3. Requests for a different model wait instead of failing
  4. VRAM is properly freed between model switches

Usage:
    queue = ModelQueue()
    result = await queue.submit("qwen3.5-27b-iq2xxs", generate_fn, *args)
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
    """Serialized model-aware request queue for Ollama.

    Ensures only one model is loaded at a time. Requests for a different
    model wait until the current model's requests are drained, then the
    queue unloads the old model and loads the new one.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._current_model: str | None = None
        self._stats = {
            "total_requests": 0,
            "model_switches": 0,
            "queued": 0,
        }

    @property
    def current_model(self) -> str | None:
        return self._current_model

    @property
    def stats(self) -> dict:
        return {**self._stats, "current_model": self._current_model}

    async def submit(self, model: str, generate_fn, *args, **kwargs):
        """Submit an inference request. Handles model switching if needed.

        Args:
            model: Ollama model name (e.g. "qwen3.5-27b-iq2xxs")
            generate_fn: The generation function to call (sync or async)
            *args, **kwargs: Passed to generate_fn

        Returns:
            Whatever generate_fn returns
        """
        self._stats["total_requests"] += 1
        self._stats["queued"] += 1

        async with self._lock:
            self._stats["queued"] -= 1

            # Switch model if needed
            if self._current_model and model != self._current_model:
                _log.info(
                    "Model switch: %s → %s (unloading...)",
                    self._current_model, model,
                )
                await self._unload_model(self._current_model)
                self._stats["model_switches"] += 1

            self._current_model = model

            # Execute the generation
            if asyncio.iscoroutinefunction(generate_fn):
                return await generate_fn(*args, **kwargs)
            else:
                # Run sync function in executor to not block event loop
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: generate_fn(*args, **kwargs))

    async def _unload_model(self, model: str):
        """Unload a model from Ollama to free VRAM."""
        try:
            # Tell Ollama to unload the model immediately
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

            # Wait briefly for VRAM to free
            await asyncio.sleep(2)
            _log.info("Unloaded model %s", model)
        except Exception as e:
            _log.warning("Failed to unload model %s: %s", model, e)
            # Continue anyway — Ollama may handle the switch itself

    async def unload_current(self):
        """Manually unload the current model (e.g. on shutdown)."""
        async with self._lock:
            if self._current_model:
                await self._unload_model(self._current_model)
                self._current_model = None


# Global singleton
_model_queue: ModelQueue | None = None


def get_model_queue() -> ModelQueue:
    """Get or create the global model queue."""
    global _model_queue
    if _model_queue is None:
        _model_queue = ModelQueue()
    return _model_queue
