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
# 224_000 added 2026-08-14 for qwen3.8-27b: probe-measured on the RTX 4090
# at 229376 ctx → 1.26 GB free (vs 0.48 GB at 262144, too thin for prod).
# Without this rung, any num_ctx cap between 128K and 256K silently topped
# out at the 128_000 tier.
CTX_TIERS = (32_000, 64_000, 128_000, 224_000, 256_000)

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

    # llama-server thinking arrives as a separate reasoning_content SSE field
    # that generate_streaming re-wraps in explicit <think>...</think> tags, so
    # the stream does NOT begin inside a thinking block.
    stream_starts_in_think = False

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
        # True when load() adopted a server some other process launched
        # (GUI, another API instance, or a machine-level scheduled task
        # that keeps a model warm).  Adopted servers are not ours to
        # terminate on unload.
        self._adopted = False
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

    def _served_props(self) -> tuple[str | None, int | None]:
        """(model_path, n_ctx) reported by the server on our port (GET /props).

        Either element is None when it can't be determined — endpoint
        missing, timeout, unexpected shape.  Callers must treat None as
        "unknown", not as a match.
        """
        try:
            req = urllib.request.Request(f"{self._base_url}/props")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
            n_ctx = (data.get("default_generation_settings") or {}).get("n_ctx") \
                or data.get("n_ctx")
            return (data.get("model_path") or None,
                    int(n_ctx) if n_ctx else None)
        except Exception:
            return None, None

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

        # Another process (API server, GUI, or a machine-level scheduled
        # task that keeps a model warm) may already have a healthy server
        # on our port — adopt it instead of launching a duplicate.  Verify
        # it serves OUR model first: a stale orphan from a different model
        # config answers /health just as happily, and adopting it blindly
        # would turn a model switch into a no-op that returns wrong-model
        # completions.
        if self._is_server_healthy():
            served, served_ctx = self._served_props()
            ours = os.path.basename(self.model_path).casefold()
            ctx_ok = not (self._target_ctx and served_ctx
                          and served_ctx < self._target_ctx)
            if served and os.path.basename(served).casefold() == ours and ctx_ok:
                self._loaded = True
                self._adopted = True
                # Record the ADOPTED server's real ctx: current_tier() must
                # report what actually serves, and the queue's tier-upgrade
                # comparison is meaningless against a stale/unset value.
                if served_ctx:
                    self._num_ctx = served_ctx
                _log.info(
                    "Adopting existing llama-server on port %d (verified: %s, ctx=%s)",
                    self.port, os.path.basename(served), served_ctx,
                )
                if status_callback:
                    status_callback(f"llama-server already running — {self.model_name}")
                return
            if served and os.path.basename(served).casefold() == ours and not ctx_ok:
                _log.warning(
                    "Healthy llama-server on port %d serves ctx=%s but %d is "
                    "needed — terminating it and relaunching bigger",
                    self.port, served_ctx, self._target_ctx,
                )
                self._kill_listener_on_port()
            else:
                # Wrong model, or a build too old to report one (served is
                # None): reclaim the port and launch our own.  This is the
                # stale-orphan path that adopted-server termination used to
                # cover from the unload side.
                _log.warning(
                    "Healthy llama-server on port %d serves %r, not %r — "
                    "terminating it and launching our own",
                    self.port, served, os.path.basename(self.model_path),
                )
                self._kill_listener_on_port()

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
            device_index=gpu_index, extra_flags=self.extra_flags,
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

        # llama-server.exe depends on ggml-cuda.dll, which in turn depends on
        # cudart64_12.dll / cublas64_12.dll / cublasLt64_12.dll. If the CUDA
        # Toolkit's bin dir is not on PATH, Windows can't resolve them and the
        # process dies at the loader with exit code 0xC0000135 (DLL_NOT_FOUND)
        # before producing any output. Prepend the newest installed CUDA bin
        # dir to PATH for the child so the launch is self-contained.
        if sys.platform == "win32":
            cuda_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
            if os.path.isdir(cuda_root):
                versions = sorted(
                    (d for d in os.listdir(cuda_root) if d.startswith("v")),
                    reverse=True,
                )
                for v in versions:
                    cuda_bin = os.path.join(cuda_root, v, "bin")
                    if os.path.isfile(os.path.join(cuda_bin, "cudart64_12.dll")):
                        launch_env["PATH"] = cuda_bin + os.pathsep + launch_env.get("PATH", "")
                        break

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

        Three paths, mirroring how load() acquired the server:

        - We spawned it: terminate via our Popen handle.
        - We adopted it: leave it RUNNING and just drop our reference.
          Adopted means another process owns it (the GUI, or a scheduled
          task that keeps a model warm for tailnet clients); the idle
          unload exists to free VRAM *this engine* allocated, and killing
          an externally-owned server turns "release to free VRAM" into a
          machine-wide outage until someone relaunches it.  The stale-
          orphan hazard that adopted-kill used to cover is now handled at
          adoption time: load() verifies the served model via /props and
          reclaims the port on mismatch, so a wrong-model orphan can't be
          silently re-adopted.
        - Neither (state lost, e.g. a previous API run crashed after
          spawning): look up the listener by port and terminate it.
        """
        if self._adopted:
            _log.info(
                "Releasing adopted llama-server on port %d without "
                "terminating it (externally managed)", self.port,
            )
        elif self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        elif self._is_server_healthy():
            self._kill_listener_on_port()
        self._process = None
        self._adopted = False

    def _kill_listener_on_port(self):
        """Find and terminate a llama-server listening on self.port.

        Used when the port must be reclaimed from a server we did not
        spawn and cannot adopt: a wrong-model orphan found at adoption
        time, or a leftover from a crashed previous run.  Scopes the
        kill to processes whose executable name contains 'llama-server'
        so an unrelated service that happens to bind the port is left
        alone.
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
                    "Terminating llama-server pid=%d holding port %d",
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
                           web_tools=False,
                           reasoning_effort=None,
                           on_telemetry=None,
                           sampling=None) -> str:
        """Stream from llama-server's OpenAI-compatible /v1/chat/completions.

        Thinking is handled via <think> tags in the content stream (same as
        Transformers path).  The server-side streaming layer applies ThinkFilter.

        sampling: optional dict of llama-server sampler params (core.sampling
        preset or hand-built).  When None, DEFAULT_SAMPLING is sent so the
        full sampler chain is always explicit — llama-server's compiled-in
        request defaults (min_p=0.05, top_k=40, ...) never apply silently.
        A "temperature" key inside sampling overrides the positional arg
        (presets carry their own temperature; plain callers keep theirs).

        web_tools is accepted-and-ignored — local llama.cpp models don't
        have native tool execution; Artifex's @search/@web_read
        post-processor in api/server.py handles tools for this backend.

        reasoning_effort ("low"/"medium"/"high"/"xhigh") rides along in
        chat_template_kwargs for templates that read it.  It only applies
        when thinking is on — with enable_thinking=False there is no think
        block to budget.
        """
        from core.sampling import DEFAULT_SAMPLING, SAMPLING_PAYLOAD_KEYS

        self.load()
        self._last_gen_stats = {}

        req_messages = list(messages)

        samp = dict(DEFAULT_SAMPLING) if sampling is None else dict(sampling)
        payload = {
            "model": self.model_name,
            "messages": req_messages,
            "stream": True,
            "temperature": samp.pop("temperature", temperature),
            "cache_prompt": True,
            # llama-server omits token usage from the stream unless asked.
            "stream_options": {"include_usage": True},
        }
        for key in SAMPLING_PAYLOAD_KEYS:
            if key in samp:
                payload[key] = samp[key]

        # dry_penalty_last_n = -1 is llama.cpp's "scan the whole context"
        # sentinel and is what core.sampling ships as the neutral value.
        # Current llama-server builds validate the field as 0 <= n and reject
        # -1 outright with HTTP 400, which fails EVERY request — including
        # ones that never touch DRY, since the neutral base always sends the
        # key.  Resolve the sentinel to the launch ctx here, at the transport
        # boundary, rather than in core.sampling: the platform contract keeps
        # -1 (the WebGPU engine maps it to dryRangeLastN and honours it), and
        # flattening it to 0 in the shared base would silently disarm DRY's
        # range for the one preset that arms it — "creative" runs
        # dry_multiplier 0.8 and inherits this value.
        if payload.get("dry_penalty_last_n") == -1:
            payload["dry_penalty_last_n"] = self._num_ctx or 0
        if not enable_thinking:
            # Qwen3.x chat templates gate the think block on an
            # `enable_thinking` template variable — not a `/no_think` text
            # directive. Pass it via chat_template_kwargs so the model
            # actually skips the block instead of thinking anyway and
            # leaking raw <think>...</think> into the response.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload["reasoning_format"] = "none"
        elif reasoning_effort:
            # Qwen3.8's template defaults reasoning effort to "xhigh", which
            # can deliberate unboundedly on hard structured tasks (measured:
            # 16K reasoning tokens with no answer emitted).  Callers bound it
            # per request with options.reasoning_effort.
            payload["chat_template_kwargs"] = {
                "reasoning_effort": reasoning_effort,
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

                        # usage/timings arrive in a final chunk whose choices
                        # array is EMPTY — read them before the choices gate
                        # or they are silently skipped.
                        usage = chunk.get("usage")
                        if usage:
                            self._last_gen_stats["prompt_tokens"] = usage.get("prompt_tokens", 0)
                            self._last_gen_stats["completion_tokens"] = usage.get("completion_tokens", 0)
                        timings = chunk.get("timings")
                        if timings:
                            self._last_gen_stats["prompt_per_second"] = timings.get("prompt_per_second")
                            self._last_gen_stats["predicted_per_second"] = timings.get("predicted_per_second")
                            self._last_gen_stats["prompt_n"] = timings.get("prompt_n")
                            self._last_gen_stats["cache_n"] = timings.get("cache_n")

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
