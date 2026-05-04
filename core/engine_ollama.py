"""
Artifex Assistant V5 — Ollama engine.
HTTP backend that talks to an Ollama server for streaming generation.
Defaults to localhost:11434. Set ARTIFEX_OLLAMA_URL for remote servers
(e.g. a Tailscale peer running Ollama on a headless GPU machine).
"""

import json
import logging
import os
import urllib.request
import urllib.error

from core.engine_base import BaseEngine
from core.inference import STOP_STRINGS, _clean_response, strip_think_blocks


_log = logging.getLogger(__name__)
OLLAMA_BASE_URL = os.environ.get("ARTIFEX_OLLAMA_URL", "http://localhost:11434").rstrip("/")


def _detect_safe_num_gpu(model_size_gb=None):
    """Calculate how many GPU layers Ollama can use without spilling into shared VRAM.

    Returns -1 (full offload) only when the model clearly fits in VRAM with
    headroom for KV cache and compute buffers.  Otherwise returns a safe
    layer count for partial offload.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 0

        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return 0

    model_loaded_gb = (model_size_gb or 5) * 1.1

    # Full offload when the model fits with headroom for KV cache.
    # Ollama's own loader uses ~1.05–1.1x overhead — anything more conservative
    # forces partial offload and CPU-side layers, which tanks throughput
    # (observed 5 tok/s at 26% CPU vs 35 tok/s at full GPU offload).
    if total_gb >= 20 and model_loaded_gb < total_gb * 0.92:
        return -1

    # Partial offload: budget VRAM for layers + KV cache + overhead
    if total_gb <= 10:
        usable_gb = total_gb * 0.70 - 1.5
    elif total_gb <= 16:
        usable_gb = total_gb * 0.75 - 1.5
    else:
        usable_gb = total_gb * 0.80 - 2.0

    if usable_gb <= 0:
        return 0

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
        self._thinking_supported = True

    # =========================================================================
    # BaseEngine — LIFECYCLE
    # =========================================================================

    def load(self, status_callback=None):
        """Verify that Ollama is running, the model exists, and resolve GPU layer count."""
        if self._loaded:
            return
        self._thinking_supported = True

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
        """Always False — Ollama is a persistent service; model switching is an API call."""
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
                           enable_thinking=True,
                           grammar=None, response_format=None,
                           raw_output=False) -> str:
        """Stream a response from the Ollama /api/chat endpoint (localhost only)."""
        self.load()

        from core.config import get_ollama_model_config
        from core.ollama_ctx import compute_safe_ctx, estimate_prompt_tokens, should_disable_mmap
        model_config = get_ollama_model_config(self.model_name)

        options = {
            "num_predict": max_tokens if max_tokens and max_tokens > 0 else -1,
            "temperature": temperature,
            "stop": STOP_STRINGS,
            "repeat_penalty": 1.15,
            "repeat_last_n": 128,
            "num_batch": 1024,
        }
        est_tokens = estimate_prompt_tokens(messages)
        options["num_ctx"] = compute_safe_ctx(self.model_name, est_tokens, model_config)

        if self._num_gpu is not None:
            options["num_gpu"] = self._num_gpu

        if should_disable_mmap(self.model_name, model_config):
            options["use_mmap"] = False

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": options,
            "keep_alive": "24h",
        }

        payload["think"] = bool(enable_thinking) and self._thinking_supported

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
            import time as _time
            _t0 = _time.perf_counter()
            _first_token_t = None
            _token_count = 0

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
                        _token_count += 1
                        if _first_token_t is None:
                            _first_token_t = _time.perf_counter()
                        if on_token:
                            on_token(content)

                    if chunk.get("done", False):
                        if in_thinking:
                            if on_token:
                                on_token("</think>")
                        _elapsed = _time.perf_counter() - _t0
                        _gen_time = _time.perf_counter() - _first_token_t if _first_token_t else 0
                        _ttft = (_first_token_t - _t0) if _first_token_t else 0
                        _tps = _token_count / _gen_time if _gen_time > 0 else 0
                        _log.info(
                            "Ollama stream done: %d chunks in %.1fs "
                            "(TTFT=%.2fs, gen=%.1fs, ~%.1f tok/s) "
                            "num_ctx=%s num_batch=%s",
                            _token_count, _elapsed, _ttft, _gen_time, _tps,
                            options.get("num_ctx"), options.get("num_batch"),
                        )
                        break

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            if e.code == 400 and "does not support thinking" in error_body:
                self._thinking_supported = False
                return self.generate_streaming(
                    messages, max_tokens=max_tokens, temperature=temperature,
                    on_token=on_token, on_complete=on_complete,
                    enable_thinking=False,
                )
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

        if raw_output:
            clean_response = strip_think_blocks(full_response)
        else:
            clean_response = _clean_response(full_response)

        if on_complete:
            on_complete(clean_response)

        return clean_response
