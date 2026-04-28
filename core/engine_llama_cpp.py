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
import struct
import subprocess
import sys
import time
import urllib.request
import urllib.error

from core.engine_base import BaseEngine
from core.inference import _clean_response

_log = logging.getLogger(__name__)

DEFAULT_HEALTH_TIMEOUT = 120
HEALTH_TIMEOUT = int(os.environ.get("ARTIFEX_HEALTH_TIMEOUT", DEFAULT_HEALTH_TIMEOUT))
HEALTH_POLL_INTERVAL = 0.5

_KV_QUANT_BPE = {
    "f16": 2.0, "f32": 4.0,
    "q8_0": 1.0625, "q4_0": 0.5625, "q4_1": 0.625,
    "q5_0": 0.6875, "q5_1": 0.75,
}


def _read_gguf_kv_params(gguf_path: str) -> dict | None:
    """Read KV-cache-relevant architecture params from a GGUF file header.

    Returns dict with head_count, head_count_kv, key_dim, val_dim,
    block_count — or None if the file can't be parsed.
    """
    all_wanted = {
        "attention.head_count", "attention.head_count_kv",
        "embedding_length", "block_count",
        "attention.key_length", "attention.value_length",
    }
    found = {}
    try:
        with open(gguf_path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != 0x46554747:
                return None
            version = struct.unpack("<I", f.read(4))[0]
            if version < 2:
                return None
            struct.unpack("<Q", f.read(8))  # tensor_count
            kv_count = struct.unpack("<Q", f.read(8))[0]

            def _str():
                n = struct.unpack("<Q", f.read(8))[0]
                return f.read(n).decode("utf-8")

            def _val(t):
                if t in (0, 7): return struct.unpack("<B", f.read(1))[0]
                if t == 1:  return struct.unpack("<b", f.read(1))[0]
                if t == 2:  return struct.unpack("<H", f.read(2))[0]
                if t == 3:  return struct.unpack("<h", f.read(2))[0]
                if t == 4:  return struct.unpack("<I", f.read(4))[0]
                if t == 5:  return struct.unpack("<i", f.read(4))[0]
                if t == 6:  return struct.unpack("<f", f.read(4))[0]
                if t == 8:  return _str()
                if t == 9:
                    at = struct.unpack("<I", f.read(4))[0]
                    n = struct.unpack("<Q", f.read(8))[0]
                    return [_val(at) for _ in range(n)]
                if t == 10: return struct.unpack("<Q", f.read(8))[0]
                if t == 11: return struct.unpack("<q", f.read(8))[0]
                if t == 12: return struct.unpack("<d", f.read(8))[0]
                raise ValueError(t)

            for _ in range(kv_count):
                key = _str()
                vtype = struct.unpack("<I", f.read(4))[0]
                value = _val(vtype)
                for suffix in all_wanted:
                    if key.endswith(suffix):
                        found[suffix] = value
                        break
                if len(found) == len(all_wanted):
                    break
    except Exception:
        return None

    if not all(k in found for k in ("attention.head_count", "embedding_length", "block_count")):
        return None

    hc = found["attention.head_count"]
    default_dim = found["embedding_length"] / hc
    key_dim = found.get("attention.key_length", default_dim)
    return {
        "head_count": hc,
        "head_count_kv": found.get("attention.head_count_kv", hc),
        "key_dim": key_dim,
        "val_dim": found.get("attention.value_length", key_dim),
        "block_count": found["block_count"],
    }


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
        self._health_timeout = model_config.get("health_timeout", HEALTH_TIMEOUT)
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

    def _get_kv_quant_bpe(self):
        """(bpe_k, bpe_v) from -ctk/-ctv in extra_flags.  Defaults to f16."""
        bpe_k = bpe_v = 2.0
        flags = self.extra_flags
        for i, flag in enumerate(flags):
            if flag == "-ctk" and i + 1 < len(flags):
                bpe_k = _KV_QUANT_BPE.get(flags[i + 1], 2.0)
            elif flag == "-ctv" and i + 1 < len(flags):
                bpe_v = _KV_QUANT_BPE.get(flags[i + 1], 2.0)
        return bpe_k, bpe_v

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

        kv_per_token_mb = None
        params = _read_gguf_kv_params(self.model_path)
        if params:
            bpe_k, bpe_v = self._get_kv_quant_bpe()
            kv_bytes = (
                params["block_count"] * params["head_count_kv"]
                * (params["key_dim"] * bpe_k + params["val_dim"] * bpe_v)
            )
            kv_per_token_mb = kv_bytes / (1024 ** 2)
            _log.info(
                "GGUF KV sizing: %d layers × %d kv_heads × "
                "(%.0f×%.3f + %.0f×%.3f) → %.4f MB/tok",
                params["block_count"], params["head_count_kv"],
                params["key_dim"], bpe_k, params["val_dim"], bpe_v,
                kv_per_token_mb,
            )

        if kv_per_token_mb is None:
            kv_per_token_mb = 128 / 1024

        max_tokens = int((available_mb * 0.90) / kv_per_token_mb)
        for bucket in reversed(STANDARD_BUCKETS):
            if bucket <= max_tokens:
                return bucket
        return MIN_CTX

    def load(self, status_callback=None):
        if self._loaded and self._is_server_healthy():
            return

        # Another process (API server or GUI) may already have a healthy
        # server on our port — adopt it instead of launching a duplicate.
        if self._is_server_healthy():
            self._loaded = True
            _log.info("Adopting existing llama-server on port %d", self.port)
            if status_callback:
                status_callback(f"llama-server already running — {self.model_name}")
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

        # On Windows, mmap keeps the full GGUF mapped in virtual memory even
        # with full GPU offload, eating commit charge (pagefile) for nothing.
        # --no-mmap lets the loader read→transfer→free instead.
        if sys.platform == "win32" and "--no-mmap" not in cmd:
            params = _read_gguf_kv_params(self.model_path)
            if params and self.num_gpu_layers >= params["block_count"]:
                cmd.append("--no-mmap")
                _log.info(
                    "Auto-adding --no-mmap: Windows + full GPU offload "
                    "(%d ngl >= %d layers)",
                    self.num_gpu_layers, params["block_count"],
                )

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

        timeout = self._health_timeout
        start = time.monotonic()
        while time.monotonic() - start < timeout:
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
            f"llama-server did not become healthy within {timeout}s. "
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
        if not self._loaded:
            return False
        if self._process is not None:
            return self._process.poll() is None
        return self._is_server_healthy()

    def get_context_size(self) -> int:
        return self._num_ctx or self._configured_num_ctx or 0

    def needs_reload(self) -> bool:
        """Always False — model switching kills/restarts the server process."""
        return False

    def periodic_cleanup(self):
        pass

    # =====================================================================
    # BaseEngine — TOKEN COUNTING
    # =====================================================================

    def count_tokens(self, text: str) -> int:
        if self._loaded:
            try:
                body = json.dumps({"content": text}).encode("utf-8")
                req = urllib.request.Request(
                    f"{self._base_url}/tokenize",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())
                    tokens = data.get("tokens", [])
                    return len(tokens)
            except Exception:
                pass
        return len(text) // 4

    # =====================================================================
    # BaseEngine — STREAMING GENERATION
    # =====================================================================

    def generate_streaming(self, messages, max_tokens, temperature,
                           on_token=None, on_complete=None,
                           enable_thinking=True,
                           grammar=None, response_format=None) -> str:
        """Stream from llama-server's OpenAI-compatible /v1/chat/completions.

        Thinking is handled via <think> tags in the content stream (same as
        Transformers path).  The server-side streaming layer applies ThinkFilter.
        """
        self.load()
        self._last_gen_stats = {}

        req_messages = list(messages)
        if not enable_thinking and req_messages:
            if req_messages[0].get("role") == "system":
                content = req_messages[0].get("content", "")
                if "/no_think" not in content:
                    req_messages[0] = dict(req_messages[0])
                    req_messages[0]["content"] = "/no_think\n" + content
            else:
                req_messages.insert(0, {"role": "system", "content": "/no_think"})

        payload = {
            "model": self.model_name,
            "messages": req_messages,
            "stream": True,
            "temperature": temperature,
            "cache_prompt": True,
        }
        if max_tokens and max_tokens > 0:
            payload["max_tokens"] = max_tokens
        if grammar:
            payload["grammar"] = grammar
        if response_format:
            payload["response_format"] = response_format

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
