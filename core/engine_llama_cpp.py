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
from core.inference import _clean_response, strip_think_blocks

_log = logging.getLogger(__name__)

DEFAULT_HEALTH_TIMEOUT = 120
HEALTH_TIMEOUT = int(os.environ.get("ARTIFEX_HEALTH_TIMEOUT", DEFAULT_HEALTH_TIMEOUT))
HEALTH_POLL_INTERVAL = 0.5

_KV_QUANT_BPE = {
    "f16": 2.0, "f32": 4.0,
    "q8_0": 1.0625, "q4_0": 0.5625, "q4_1": 0.625,
    "q5_0": 0.6875, "q5_1": 0.75,
}

# ── Context tier configuration ─────────────────────────────────────────
# Discrete launch-context buckets for llama-server.  Smaller tiers reserve
# much less KV cache but require relaunch when context grows past the tier;
# finer tiers are right for tight-VRAM cards where wasted KV is the binding
# constraint.  On a 24 GB card with q4_0 KV on a 27B model, the cost per
# tier is roughly 576 / 1152 / 2304 / 4608 MB.
CTX_TIERS = (32_000, 64_000, 128_000, 256_000)

# Snap-up margin: when the request need is within this many tokens of the
# next tier, choose the next tier.  Buffers in-flight context growth from
# tool rounds and follow-up turns within a session.
TIER_HEADROOM_TOK = 8_000

# Pressure margin used while a tier is live: if the next request would
# bring the engine within this many tokens of its tier cap, the request
# router should grow to the next tier instead of risking mid-stream
# overflow.  Looser than the snap-up margin because the engine is already
# carrying a session's worth of context that may itself grow.
ACTIVE_TIER_PRESSURE_TOK = 12_000


def pick_ctx_tier(needed_tokens: int, max_cap: int | None = None) -> int:
    """Pick the smallest CTX_TIERS bucket that fits needed_tokens with
    TIER_HEADROOM_TOK of headroom, optionally capped at max_cap.

    Returns the largest selectable tier when the need exceeds even that;
    the VRAM gate downstream handles the truly-too-big case.

    Args:
        needed_tokens: Estimated total context (prompt + completion + buffers).
        max_cap: Optional ceiling — never returns a tier above this (typically
                 model_config['num_ctx'] from llama_cpp_config.json).

    Returns:
        A tier from CTX_TIERS (or the cap-snapped equivalent).
    """
    target = max(0, int(needed_tokens)) + TIER_HEADROOM_TOK
    candidates = CTX_TIERS
    if max_cap and max_cap > 0:
        capped = tuple(t for t in CTX_TIERS if t <= max_cap)
        # Keep at least one tier — degenerate caps below the smallest tier
        # are clamped to that tier so callers always get a usable value.
        candidates = capped or (CTX_TIERS[0],)
    for tier in candidates:
        if tier >= target:
            return tier
    return candidates[-1]


def _read_gguf_kv_params(gguf_path: str) -> dict | None:
    """Read KV-cache-relevant architecture params from a GGUF file header.

    Returns dict with head_count, head_count_kv, key_dim, val_dim,
    block_count, and attn_layer_count (for hybrid models) — or None if
    the file can't be parsed.
    """
    all_wanted = {
        "attention.head_count", "attention.head_count_kv",
        "embedding_length", "block_count",
        "attention.key_length", "attention.value_length",
        "full_attention_interval",
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
    block_count = found["block_count"]
    full_attention_interval = found.get("full_attention_interval", 1)

    if full_attention_interval > 1:
        attn_layer_count = block_count // full_attention_interval
    else:
        attn_layer_count = block_count

    return {
        "head_count": hc,
        "head_count_kv": found.get("attention.head_count_kv", hc),
        "key_dim": key_dim,
        "val_dim": found.get("attention.value_length", key_dim),
        "block_count": block_count,
        "attn_layer_count": attn_layer_count,
        "full_attention_interval": full_attention_interval,
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
        # Cap from config — the engine will never launch above this, even if
        # set_target_tier is called with a larger value.  Semantics changed
        # with tier-aware launch: this is the upper bound, not the value.
        self._configured_num_ctx = model_config.get("num_ctx")
        self._health_timeout = model_config.get("health_timeout", HEALTH_TIMEOUT)
        # Optional explicit GPU pin. None ⇒ auto-pick the GPU with the most
        # free VRAM at load time (correct on multi-GPU rigs where the lower
        # PCI index is a display card, e.g. 1080 Ti for screens + 4090 for
        # inference).  Per-model override > env var > auto.
        self._configured_gpu_index = model_config.get("gpu_index")
        # Tier-driven launch ctx, set by set_target_tier() before load().
        # When None, _compute_num_ctx falls back to the configured cap or the
        # legacy VRAM-fit heuristic.
        self._target_ctx: int | None = None
        self._num_ctx = None
        self._active_gpu_index: int | None = None
        self._process = None
        self._loaded = False
        self._base_url = f"http://localhost:{self.port}"
        self._last_gen_stats = {}

    def set_target_tier(self, tier: int) -> None:
        """Set the launch ctx for the next load(), capped at the configured cap.

        Called by the request router (api/server.py) before engine.load() so
        the engine launches with a ctx sized for the actual workload instead
        of the model's max context.  Has no effect on a currently-loaded
        engine; a relaunch (unload + load) is required for the change to
        take effect — the queue handles that on a tier change.
        """
        if not isinstance(tier, int) or tier <= 0:
            return
        if self._configured_num_ctx and tier > self._configured_num_ctx:
            tier = self._configured_num_ctx
        self._target_ctx = tier

    def current_tier(self) -> int:
        """The live engine's loaded ctx, or 0 if unloaded."""
        return self._num_ctx or 0

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

    def _get_kv_quant_str(self) -> str:
        """KV quant type name from -ctk in extra_flags. Defaults to 'f16'."""
        flags = self.extra_flags
        for i, flag in enumerate(flags):
            if flag == "-ctk" and i + 1 < len(flags):
                return flags[i + 1]
        return "f16"

    def _resolve_gpu_index(self, pool) -> int:
        """Pick the GPU to load on.

        Priority: model_config['gpu_index'] > $ARTIFEX_GPU_INDEX > auto-pick
        (the GPU with the most free VRAM).  Falls back to 0 when nvidia-smi
        is unavailable so single-GPU and headless boxes keep working.
        """
        if self._configured_gpu_index is not None:
            return int(self._configured_gpu_index)

        env_idx = os.environ.get("ARTIFEX_GPU_INDEX")
        if env_idx not in (None, ""):
            try:
                return int(env_idx)
            except ValueError:
                _log.warning(
                    "ARTIFEX_GPU_INDEX=%r is not an int, ignoring", env_idx,
                )

        # find_best_device(0) returns the device with max free VRAM, or None
        # if no devices were enumerated (no nvidia-smi).
        picked = pool.find_best_device(0)
        if picked is None:
            _log.info("No GPUs enumerated; defaulting to device 0")
            return 0
        return picked

    def _compute_num_ctx(self) -> int:
        # Tier picker takes priority — set by request router via set_target_tier
        if self._target_ctx:
            return self._target_ctx
        # Configured cap from llama_cpp_config.json — legacy default
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
            attn_layers = params["attn_layer_count"]
            kv_bytes = (
                attn_layers * params["head_count_kv"]
                * (params["key_dim"] * bpe_k + params["val_dim"] * bpe_v)
            )
            kv_per_token_mb = kv_bytes / (1024 ** 2)
            _log.info(
                "GGUF KV sizing: %d attn_layers (of %d total, interval=%d) × %d kv_heads × "
                "(%.0f×%.3f + %.0f×%.3f) → %.4f MB/tok",
                attn_layers, params["block_count"], params["full_attention_interval"],
                params["head_count_kv"],
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

        # ── VRAM gate ──
        from core.gpu_pool import get_pool
        pool = get_pool()

        gpu_index = self._resolve_gpu_index(pool)
        self._active_gpu_index = gpu_index

        kv_quant_str = self._get_kv_quant_str()
        allocation = pool.estimate_allocation_mb(
            self.model_path, self._num_ctx, kv_quant=kv_quant_str,
            device_index=gpu_index,
        )
        needed_mb = (
            allocation["model_weight_mb"]
            + allocation["kv_cache_mb"]
            + allocation["compute_buffer_mb"]
        )
        _log.info(
            "VRAM gate: need %.0f MB free on GPU %d (weight=%.0f + kv=%.0f + compute=%.0f)",
            needed_mb, gpu_index, allocation["model_weight_mb"],
            allocation["kv_cache_mb"], allocation["compute_buffer_mb"],
        )
        if not pool.wait_for_vram(needed_mb, device_index=gpu_index):
            raise RuntimeError(
                f"VRAM not available: need {needed_mb:.0f} MB free on GPU {gpu_index}. "
                f"Previous process may still be releasing memory. "
                f"Try again in a few seconds or reduce context size."
            )

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

        # PCI_BUS_ID ordering aligns CUDA's view with nvidia-smi's, so the
        # index we resolved against the pool also points at the same device
        # inside llama-server.  CUDA's default FASTEST_FIRST would otherwise
        # reorder by compute capability and silently flip the indices.
        launch_env = os.environ.copy()
        launch_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        launch_env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        _log.info("llama-server cmd: %s  (CUDA_VISIBLE_DEVICES=%d)",
                  " ".join(cmd), gpu_index)

        # Capture llama-server's combined stdout+stderr to a file. Two reasons:
        # (1) llama-server prints model-loading details (and load failures)
        # to STDOUT, not stderr — DEVNULL'ing stdout used to hide the real
        # cause of every launch crash. (2) Python pipe buffers drop the tail
        # of output on abnormal exit, so error messages near the crash often
        # never reached us. A file has no buffer limit and outlives the
        # process for forensic reads.
        launch_log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs",
        )
        os.makedirs(launch_log_dir, exist_ok=True)
        launch_log_path = os.path.join(
            launch_log_dir, f"llama-server-port{self.port}.log"
        )

        max_launch_attempts = 2
        launch_retry_delay = 5.0

        for attempt in range(max_launch_attempts):
            launch_log_fh = open(launch_log_path, "wb")
            try:
                self._process = subprocess.Popen(
                    cmd, stdout=launch_log_fh, stderr=subprocess.STDOUT,
                    env=launch_env,
                )
            except FileNotFoundError:
                launch_log_fh.close()
                raise FileNotFoundError(
                    f"'{self.server_path}' not found. Install llama.cpp or set "
                    f"server_path in llama_cpp_config.json to the full path."
                )
            # Parent doesn't need its handle once the child has inherited it.
            launch_log_fh.close()

            timeout = self._health_timeout
            start = time.monotonic()
            launch_failed = False

            while time.monotonic() - start < timeout:
                if self._process.poll() is not None:
                    try:
                        with open(launch_log_path, "rb") as f:
                            launch_output = f.read().decode("utf-8", errors="replace")
                    except OSError as e:
                        launch_output = f"(could not read {launch_log_path}: {e})"
                    _log.error(
                        "llama-server crashed (attempt %d, code %d). Launch log: %s\n%s",
                        attempt + 1, self._process.returncode,
                        launch_log_path, launch_output,
                    )
                    if attempt < max_launch_attempts - 1:
                        _log.warning(
                            "llama-server crashed on startup (attempt %d/%d, "
                            "code %d) — retrying in %.0fs...",
                            attempt + 1, max_launch_attempts,
                            self._process.returncode, launch_retry_delay,
                        )
                        self._process = None
                        time.sleep(launch_retry_delay)
                        pool.wait_for_vram(needed_mb, device_index=gpu_index, timeout=10)
                        launch_failed = True
                        break
                    raise RuntimeError(
                        f"llama-server exited with code {self._process.returncode}.\n"
                        f"Full launch log: {launch_log_path}\n"
                        f"Last 4000 chars:\n{launch_output[-4000:]}"
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
            else:
                self._kill_process()
                raise TimeoutError(
                    f"llama-server did not become healthy within {timeout}s.\n"
                    f"Launch log: {launch_log_path}\n"
                    f"Check that the model fits in VRAM and the GGUF is valid."
                )

            if not launch_failed:
                break

    def _kill_process(self):
        """Stop the llama-server bound to this engine.

        Two paths because load() has two paths: when we spawned the
        process ourselves we have a Popen handle and terminate via that;
        when load() adopted an already-running server we have no Popen
        handle (self._process is None) and must look up the listener by
        port and terminate the OS process directly.

        Without the second path, unload() silently leaves an adopted
        server running and the next load() re-adopts the same orphan,
        which is how a model/ctx-tier "switch" can be a no-op at the
        server level even though the queue thinks it succeeded.
        """
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        elif self._is_server_healthy():
            self._kill_listener_on_port()
        self._process = None

    def _kill_listener_on_port(self):
        """Find and terminate a llama-server listening on self.port.

        Used by _kill_process() when this engine adopted an existing
        server — without this, unload would be a no-op for adopted
        servers.  Scopes the kill to processes whose executable name
        contains 'llama-server' so an unrelated service that happens to
        bind the port is left alone.
        """
        import psutil
        matches: list[psutil.Process] = []
        for proc in psutil.process_iter(["pid", "name", "exe"]):
            try:
                name = (proc.info.get("name") or "").lower()
                exe = (proc.info.get("exe") or "").lower()
                if "llama-server" not in name and "llama-server" not in exe:
                    continue
                conns = proc.net_connections(kind="tcp")
                for conn in conns:
                    if (
                        conn.laddr is not None
                        and conn.laddr.port == self.port
                        and conn.status == psutil.CONN_LISTEN
                    ):
                        matches.append(proc)
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if not matches:
            _log.warning(
                "Adopted server on port %d but no matching llama-server "
                "process found to terminate; orphan may persist",
                self.port,
            )
            return

        for proc in matches:
            try:
                _log.info(
                    "Terminating adopted llama-server pid=%d on port %d",
                    proc.pid, self.port,
                )
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except psutil.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                _log.warning("Could not terminate pid=%d: %s", proc.pid, e)

        # Brief settle so the next load()'s health check doesn't race the
        # OS releasing the listening socket.
        time.sleep(0.5)

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
                           grammar=None, response_format=None,
                           raw_output=False,
                           web_tools=False) -> str:
        """Stream from llama-server's OpenAI-compatible /v1/chat/completions.

        Thinking is handled via <think> tags in the content stream (same as
        Transformers path).  The server-side streaming layer applies ThinkFilter.

        web_tools is accepted-and-ignored — local llama.cpp models don't
        have native tool execution; Artifex's @search/@web_read
        post-processor in api/server.py handles tools for this backend.
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
        if not enable_thinking:
            payload["reasoning_format"] = "none"
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

        max_retries = 2
        for attempt in range(max_retries):
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

                break  # success — exit retry loop

            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"llama-server error ({e.code}): {error_body}")
            except (urllib.error.URLError, ConnectionResetError, OSError) as e:
                if attempt < max_retries - 1 and self._is_server_healthy():
                    _log.warning("Connection lost mid-request, retrying: %s", e)
                    time.sleep(2)
                    full_text = ""
                    in_thinking = False
                    continue
                if self._is_server_healthy():
                    raise ConnectionError(f"Lost connection to llama-server: {e}")
                raise ConnectionError(
                    "llama-server is not responding — it may have crashed (OOM). "
                    "Restart with a smaller context or check VRAM usage."
                )

        if in_thinking:
            full_text += "</think>"

        if raw_output:
            clean = strip_think_blocks(full_text)
        else:
            clean = _clean_response(full_text)
        if on_complete:
            on_complete(clean)
        return clean
