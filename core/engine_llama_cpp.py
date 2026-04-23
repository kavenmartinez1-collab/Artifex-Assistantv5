"""
Artifex Assistant V5 — llama.cpp server engine.
Manages a llama-server process and streams via its OpenAI-compatible API.
Supports custom forks (TurboQuant, etc.) and fine-grained launch flags.

Unlike Ollama (a persistent system service), this engine owns the server
process lifecycle: load() starts it, unload() kills it.  The server speaks
OpenAI-compatible API natively, so no message format conversion is needed.
"""

import json
import logging
import os
import subprocess
import time
import urllib.request
import urllib.error

from core.engine_base import BaseEngine
from core.inference import _clean_response

_log = logging.getLogger(__name__)

HEALTH_TIMEOUT = 120  # seconds to wait for server startup (large models are slow)
HEALTH_POLL_INTERVAL = 0.5


class LlamaCppEngine(BaseEngine):
    """llama.cpp server backend — manages a llama-server process per model."""

    def __init__(self, model_name: str, model_config: dict):
        self.model_name = model_name
        self.model_path = model_config["path"]
        self.port = model_config.get("port", 8081)
        self.num_gpu_layers = model_config.get("num_gpu_layers", 99)
        self.extra_flags = list(model_config.get("extra_flags", []))
        self.server_path = model_config.get("server_path", "llama-server")
        self._configured_num_ctx = model_config.get("num_ctx")
        self._num_ctx = None
        self._process = None
        self._loaded = False
        self._base_url = f"http://localhost:{self.port}"
        self._last_gen_stats = {}

    # =====================================================================
    # BaseEngine — LIFECYCLE
    # =====================================================================

    def _is_server_healthy(self) -> bool:
        try:
            req = urllib.request.Request(f"{self._base_url}/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                return data.get("status") == "ok"
        except Exception:
            return False

    def _compute_num_ctx(self) -> int:
        if self._configured_num_ctx:
            return self._configured_num_ctx
        from core.ollama_ctx import (
            _get_total_vram_mb, SYSTEM_RESERVE_MB, MODEL_OVERHEAD_FACTOR,
            STANDARD_BUCKETS, MIN_CTX,
        )
        total_mb = _get_total_vram_mb()
        if not total_mb:
            return 8192
        try:
            file_mb = os.path.getsize(self.model_path) / (1024 ** 2)
        except OSError:
            return 8192
        model_loaded_mb = file_mb * MODEL_OVERHEAD_FACTOR
        available_mb = total_mb - SYSTEM_RESERVE_MB - model_loaded_mb
        if available_mb <= 0:
            return MIN_CTX
        # Without architecture info from the GGUF, use a conservative
        # 128 KB/token estimate (covers most dense transformer architectures).
        # For hybrid or MoE models, set num_ctx explicitly in the config.
        kv_per_token_mb = 128 / 1024
        max_tokens = int((available_mb * 0.90) / kv_per_token_mb)
        for bucket in reversed(STANDARD_BUCKETS):
            if bucket <= max_tokens:
                return bucket
        return MIN_CTX

    def load(self, status_callback=None):
        if self._loaded and self._is_server_healthy():
            return

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model GGUF not found: {self.model_path}")

        self._num_ctx = self._compute_num_ctx()

        if status_callback:
            status_callback(
                f"Starting llama-server: {self.model_name} "
                f"(port {self.port}, ctx {self._num_ctx})..."
            )

        cmd = [
            self.server_path,
            "-m", self.model_path,
            "--port", str(self.port),
            "-ngl", str(self.num_gpu_layers),
            "-c", str(self._num_ctx),
        ]
        cmd.extend(self.extra_flags)
        _log.info("llama-server cmd: %s", " ".join(cmd))

        try:
            self._process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"'{self.server_path}' not found. Install llama.cpp or set "
                f"server_path in llama_cpp_config.json to the full path."
            )

        start = time.monotonic()
        while time.monotonic() - start < HEALTH_TIMEOUT:
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"llama-server exited with code {self._process.returncode}:\n"
                    f"{stderr[-1000:]}"
                )
            if self._is_server_healthy():
                self._loaded = True
                _log.info(
                    "llama-server ready: %s (port %d, ctx %d, ngl %d)",
                    self.model_name, self.port, self._num_ctx, self.num_gpu_layers,
                )
                if status_callback:
                    status_callback(
                        f"llama-server ready — {self.model_name} (ctx={self._num_ctx})"
                    )
                return
            time.sleep(HEALTH_POLL_INTERVAL)

        self._kill_process()
        raise TimeoutError(
            f"llama-server did not become healthy within {HEALTH_TIMEOUT}s. "
            f"Check that the model fits in VRAM and the GGUF is valid."
        )

    def _kill_process(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        self._process = None

    def unload(self, status_callback=None):
        self._kill_process()
        self._loaded = False
        if status_callback:
            status_callback("llama-server stopped.")

    def is_loaded(self) -> bool:
        return self._loaded and (self._process is not None and self._process.poll() is None)

    def needs_reload(self) -> bool:
        return False

    def periodic_cleanup(self):
        pass

    # =====================================================================
    # BaseEngine — TOKEN COUNTING
    # =====================================================================

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    # =====================================================================
    # BaseEngine — STREAMING GENERATION
    # =====================================================================

    def generate_streaming(self, messages, max_tokens, temperature,
                           on_token=None, on_complete=None,
                           enable_thinking=True) -> str:
        """Stream from llama-server's OpenAI-compatible /v1/chat/completions.

        Thinking is handled via <think> tags in the content stream (same as
        Transformers path).  The server-side streaming layer applies ThinkFilter.
        """
        self.load()
        self._last_gen_stats = {}

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
        }

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        full_text = ""
        in_thinking = False

        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})

                    reasoning = delta.get("reasoning_content", "")
                    if reasoning:
                        if not in_thinking:
                            in_thinking = True
                            full_text += "<think>"
                            if on_token:
                                on_token("<think>")
                        full_text += reasoning
                        if on_token:
                            on_token(reasoning)
                        continue

                    content = delta.get("content", "")
                    if content:
                        if in_thinking:
                            in_thinking = False
                            full_text += "</think>"
                            if on_token:
                                on_token("</think>")
                        full_text += content
                        if on_token:
                            on_token(content)

                    finish = choices[0].get("finish_reason")
                    if finish:
                        self._last_gen_stats["finish_reason"] = finish

                    usage = chunk.get("usage")
                    if usage:
                        self._last_gen_stats["prompt_tokens"] = usage.get("prompt_tokens", 0)
                        self._last_gen_stats["completion_tokens"] = usage.get("completion_tokens", 0)

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server error ({e.code}): {error_body}")
        except urllib.error.URLError as e:
            raise ConnectionError(f"Lost connection to llama-server: {e}")

        if in_thinking:
            full_text += "</think>"

        clean = _clean_response(full_text)
        if on_complete:
            on_complete(clean)
        return clean
