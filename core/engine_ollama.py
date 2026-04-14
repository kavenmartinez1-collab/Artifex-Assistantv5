"""
Artifex Assistant V5 — Ollama engine.
HTTP backend that talks to a locally running Ollama server for streaming generation.
All communication stays on localhost — nothing goes over the internet.
Ollama manages its own model lifecycle — this engine controls GPU layer placement
to prevent spilling into shared VRAM on cards with limited memory.
"""

import json
import urllib.request
import urllib.error

from core.engine_base import BaseEngine
from core.inference import STOP_STRINGS, _clean_response


# Default Ollama endpoint — localhost only, never remote
OLLAMA_BASE_URL = "http://localhost:11434"


def _detect_safe_num_gpu(model_size_gb=None):
    """Calculate how many GPU layers Ollama can use without spilling into shared VRAM.

    Strategy: reserve ~1.5 GB for KV cache + OS overhead, give the rest to model layers.
    Ollama models are typically 30-80 layers. We estimate layer size from model total.

    Returns:
        int: number of GPU layers, or -1 if VRAM is abundant (>= 20 GB).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0  # no GPU — CPU only

        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return 0

    # Abundant VRAM — let Ollama use everything
    if total_gb >= 20:
        return -1

    # Reserve headroom for KV cache, CUDA context, OS, and generation buffers.
    if total_gb <= 10:
        usable_gb = total_gb * 0.70 - 1.5  # ~4.1 GB usable on 8 GB card
    else:
        usable_gb = total_gb * 0.75 - 1.5  # ~7.5 GB usable on 12 GB card

    if usable_gb <= 0:
        return 0

    # Estimate: typical GGUF Q4 models use ~0.15-0.20 GB per layer (7-9B),
    # ~0.25-0.35 GB per layer for larger models.
    gb_per_layer = 0.18 if (model_size_gb or 5) <= 6 else 0.30
    safe_layers = int(usable_gb / gb_per_layer)

    return min(safe_layers, 80)


class OllamaEngine(BaseEngine):
    """Ollama HTTP backend — delegates inference to a locally running Ollama server."""

    def __init__(self, model_name: str, base_url: str = OLLAMA_BASE_URL):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._loaded = False
        self._num_gpu = None
        self._model_size_gb = None

    # =========================================================================
    # BaseEngine — LIFECYCLE
    # =========================================================================

    def load(self, status_callback=None):
        """Verify that Ollama is running, the model exists, and resolve GPU layer count."""
        if self._loaded:
            return

        if status_callback:
            status_callback(f"Connecting to Ollama ({self.base_url})...")

        # Ping the server and get model list
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, OSError) as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                f"Is Ollama running? (ollama serve)\n{e}"
            )

        # Check that the model is available + grab its size
        available = data.get("models", [])
        matched_model = None
        for m in available:
            name = m.get("name", "")
            if name == self.model_name or name.startswith(self.model_name + ":"):
                matched_model = m
                break

        if matched_model:
            size_bytes = matched_model.get("size", 0)
            self._model_size_gb = size_bytes / (1024 ** 3) if size_bytes else None
        else:
            if status_callback:
                status_callback(
                    f"Model '{self.model_name}' not found locally. "
                    f"Ollama will attempt to pull it on first use."
                )

        # Resolve num_gpu — prevent shared VRAM spill
        from core.config import OLLAMA_NUM_GPU
        if OLLAMA_NUM_GPU == "auto":
            self._num_gpu = _detect_safe_num_gpu(self._model_size_gb)
        else:
            self._num_gpu = int(OLLAMA_NUM_GPU)

        self._loaded = True

        gpu_msg = f"all layers" if self._num_gpu == -1 else f"{self._num_gpu} layers"
        if self._num_gpu == 0:
            gpu_msg = "CPU only"
        if status_callback:
            status_callback(f"Ollama ready — model: {self.model_name} (GPU: {gpu_msg})")

    def unload(self, status_callback=None):
        """No-op — Ollama manages its own model memory."""
        self._loaded = False
        if status_callback:
            status_callback("Ollama connection closed.")

    def is_loaded(self) -> bool:
        return self._loaded

    def needs_reload(self) -> bool:
        return False

    def periodic_cleanup(self):
        """No-op — Ollama handles its own resource management."""
        pass

    # =========================================================================
    # BaseEngine — TOKEN COUNTING
    # =========================================================================

    def count_tokens(self, text: str) -> int:
        """Approximate token count (~4 chars/token heuristic)."""
        return len(text) // 4

    # =========================================================================
    # BaseEngine — STREAMING GENERATION
    # =========================================================================

    def generate_streaming(self, messages, max_tokens, temperature,
                           on_token=None, on_complete=None,
                           enable_thinking=True) -> str:
        """Stream a response from the Ollama /api/chat endpoint (localhost only)."""
        self.load()

        # Get per-model config (num_ctx, etc.)
        from core.config import get_ollama_model_config
        model_config = get_ollama_model_config(self.model_name)

        options = {
            "num_predict": max_tokens,
            "temperature": temperature,
            "stop": STOP_STRINGS,
            "repeat_penalty": 1.15,
            "repeat_last_n": 128,
            "num_ctx": model_config.get("num_ctx", 8192),
        }

        if self._num_gpu is not None:
            options["num_gpu"] = self._num_gpu

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": options,
        }

        payload["think"] = bool(enable_thinking)

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        full_response = ""
        thinking_response = ""
        in_thinking = False

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for line in resp:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg = chunk.get("message", {})

                    # Wrap Ollama thinking in <think> tags for ThinkFilter
                    thinking = msg.get("thinking", "")
                    if thinking:
                        thinking_response += thinking
                        if not in_thinking:
                            if on_token:
                                on_token("<think>")
                            in_thinking = True
                        if on_token:
                            on_token(thinking)

                    content = msg.get("content", "")
                    if content:
                        if in_thinking:
                            if on_token:
                                on_token("</think>")
                            in_thinking = False
                        full_response += content
                        if on_token:
                            on_token(content)

                    if chunk.get("done", False):
                        if in_thinking:
                            if on_token:
                                on_token("</think>")
                        break

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama API error ({e.code}): {error_body}"
            )
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Lost connection to Ollama: {e}"
            )

        # If model produced only thinking and no content, use thinking as response
        if not full_response.strip() and thinking_response.strip():
            full_response = thinking_response

        clean_response = _clean_response(full_response)

        if on_complete:
            on_complete(clean_response)

        return clean_response
