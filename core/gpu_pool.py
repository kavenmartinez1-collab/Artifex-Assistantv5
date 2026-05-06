"""
Artifex Assistant V5 — GPU Resource Pool.

Manages GPU device enumeration, VRAM monitoring, and allocation gating.
Primary purpose: prevent crash-loops when llama-server dies and WDDM hasn't
reclaimed VRAM yet.  The wait_for_vram() gate polls nvidia-smi until the
device actually has enough free memory before allowing a restart.

Also provides VRAM estimation for model configs and device selection for
future multi-GPU rigs.

Design:
  - Singleton pattern (like ModelQueue)
  - Thread-safe via threading.Lock on mutable state
  - Degrades gracefully if nvidia-smi is unavailable
  - All nvidia-smi calls have timeout=5s
  - Logs extensively for diagnostics
"""

import logging
import os
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

_log = logging.getLogger(__name__)

# ── KV Quantization bytes-per-element map ──────────────────────────────

KV_QUANT_BPE = {
    "f16": 2.0,
    "f32": 4.0,
    "q8_0": 1.0625,    # 32 int8 + 1 f16 scale per block of 32
    "q4_0": 0.5625,    # 16 int4-pairs + 1 f16 scale per block of 32
    "q4_1": 0.625,     # 16 int4-pairs + 1 f16 scale + 1 f16 min per block of 32
    "q5_0": 0.6875,    # 20 int5-pairs + 1 f16 scale per block of 32
    "q5_1": 0.75,      # 20 int5-pairs + 1 f16 scale + 1 f16 min per block of 32
}

# ── Constants ──────────────────────────────────────────────────────────

MODEL_OVERHEAD_FACTOR = 1.0     # GGUF file size ≈ GPU weight (embeddings/vocab offloaded to CPU)
COMPUTE_BUFFER_MB = 1000        # Flat estimate for llama.cpp compute buffers
SYSTEM_RESERVE_MB = 2048        # Windows WDDM + compositor + misc
NVIDIA_SMI_TIMEOUT = 5          # seconds

DEFAULT_VRAM_WAIT_TIMEOUT = 30  # seconds
DEFAULT_VRAM_POLL_INTERVAL = 1.5  # seconds


# ── GPU Device Data ────────────────────────────────────────────────────

@dataclass
class GPUDevice:
    """Represents one physical GPU."""
    index: int
    name: str
    memory_total_mb: float
    memory_free_mb: float
    memory_used_mb: float
    compute_cap: str
    last_refresh: float = field(default_factory=time.time)

    @property
    def utilization_pct(self) -> float:
        if self.memory_total_mb <= 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100.0


# ── GGUF Header Reading ────────────────────────────────────────────────

def read_gguf_kv_params(gguf_path: str) -> Optional[dict]:
    """Read KV-cache-relevant architecture params from a GGUF file header.

    Returns dict with:
      - head_count: total attention heads
      - head_count_kv: KV heads (GQA)
      - key_dim: key vector dimension per head
      - val_dim: value vector dimension per head
      - block_count: total layers
      - full_attention_interval: hybrid model attention spacing (1 = all layers)
      - attn_layer_count: actual number of layers with KV cache

    Returns None if the file can't be parsed or required keys are missing.
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
            if magic != 0x46554747:  # "GGUF"
                _log.warning("GGUF magic mismatch in %s", gguf_path)
                return None
            version = struct.unpack("<I", f.read(4))[0]
            if version < 2:
                _log.warning("GGUF version %d unsupported in %s", version, gguf_path)
                return None
            struct.unpack("<Q", f.read(8))  # tensor_count (unused)
            kv_count = struct.unpack("<Q", f.read(8))[0]

            def _str():
                n = struct.unpack("<Q", f.read(8))[0]
                return f.read(n).decode("utf-8")

            def _val(t):
                if t in (0, 7):
                    return struct.unpack("<B", f.read(1))[0]
                if t == 1:
                    return struct.unpack("<b", f.read(1))[0]
                if t == 2:
                    return struct.unpack("<H", f.read(2))[0]
                if t == 3:
                    return struct.unpack("<h", f.read(2))[0]
                if t == 4:
                    return struct.unpack("<I", f.read(4))[0]
                if t == 5:
                    return struct.unpack("<i", f.read(4))[0]
                if t == 6:
                    return struct.unpack("<f", f.read(4))[0]
                if t == 8:
                    return _str()
                if t == 9:
                    at = struct.unpack("<I", f.read(4))[0]
                    n = struct.unpack("<Q", f.read(8))[0]
                    return [_val(at) for _ in range(n)]
                if t == 10:
                    return struct.unpack("<Q", f.read(8))[0]
                if t == 11:
                    return struct.unpack("<q", f.read(8))[0]
                if t == 12:
                    return struct.unpack("<d", f.read(8))[0]
                raise ValueError(f"Unknown GGUF value type: {t}")

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
    except FileNotFoundError:
        _log.error("GGUF file not found: %s", gguf_path)
        return None
    except Exception as e:
        _log.error("Failed to parse GGUF header from %s: %s", gguf_path, e)
        return None

    # Validate minimum required fields
    if not all(k in found for k in ("attention.head_count", "embedding_length", "block_count")):
        _log.warning("GGUF %s missing required architecture keys (found: %s)",
                     gguf_path, list(found.keys()))
        return None

    hc = found["attention.head_count"]
    default_dim = found["embedding_length"] / hc
    key_dim = found.get("attention.key_length", default_dim)
    block_count = found["block_count"]
    full_attention_interval = found.get("full_attention_interval", 1)

    # Hybrid models: only every Nth layer has attention KV cache
    if full_attention_interval > 1:
        attn_layer_count = block_count // full_attention_interval
        _log.debug("Hybrid model: %d/%d layers have attention KV (interval=%d)",
                   attn_layer_count, block_count, full_attention_interval)
    else:
        attn_layer_count = block_count

    return {
        "head_count": hc,
        "head_count_kv": found.get("attention.head_count_kv", hc),
        "key_dim": key_dim,
        "val_dim": found.get("attention.value_length", key_dim),
        "block_count": block_count,
        "full_attention_interval": full_attention_interval,
        "attn_layer_count": attn_layer_count,
    }


# ── GPU Pool Singleton ─────────────────────────────────────────────────

class GPUPool:
    """Singleton GPU resource pool.

    Manages device enumeration, VRAM monitoring, and allocation gating.
    Thread-safe for concurrent API handler access.
    """

    _instance: Optional["GPUPool"] = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "GPUPool":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._lock = threading.Lock()
        self._devices: list[GPUDevice] = []
        self._nvidia_smi_available: Optional[bool] = None
        self._initialized = True
        # Initial enumeration
        self._enumerate_devices()

    # ── Device Enumeration ─────────────────────────────────────────────

    def _run_nvidia_smi(self, query: str) -> Optional[str]:
        """Run nvidia-smi with a query and return stdout, or None on failure."""
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT,
            )
            if result.returncode == 0:
                self._nvidia_smi_available = True
                return result.stdout.strip()
            else:
                _log.warning("nvidia-smi returned code %d: %s",
                             result.returncode, result.stderr.strip())
                return None
        except FileNotFoundError:
            if self._nvidia_smi_available is not False:
                _log.warning("nvidia-smi not found — GPU pool will assume enough VRAM")
                self._nvidia_smi_available = False
            return None
        except subprocess.TimeoutExpired:
            _log.error("nvidia-smi timed out after %ds", NVIDIA_SMI_TIMEOUT)
            return None
        except Exception as e:
            _log.error("nvidia-smi query failed: %s", e)
            return None

    def _enumerate_devices(self) -> None:
        """Query all GPUs and populate the device list."""
        output = self._run_nvidia_smi(
            "index,name,memory.total,memory.free,memory.used,compute_cap"
        )
        if output is None:
            _log.info("No GPU devices enumerated (nvidia-smi unavailable)")
            return

        devices = []
        now = time.time()
        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                _log.warning("Unexpected nvidia-smi line format: %s", line)
                continue
            try:
                dev = GPUDevice(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_total_mb=float(parts[2]),
                    memory_free_mb=float(parts[3]),
                    memory_used_mb=float(parts[4]),
                    compute_cap=parts[5],
                    last_refresh=now,
                )
                devices.append(dev)
            except (ValueError, IndexError) as e:
                _log.warning("Failed to parse GPU line '%s': %s", line, e)
                continue

        with self._lock:
            self._devices = devices

        if devices:
            _log.info("Enumerated %d GPU device(s):", len(devices))
            for d in devices:
                _log.info("  [%d] %s — %0.f MB total, %.0f MB free, compute %s",
                          d.index, d.name, d.memory_total_mb, d.memory_free_mb,
                          d.compute_cap)
        else:
            _log.warning("nvidia-smi ran but returned no devices")

    def refresh_device(self, device_index: int = 0) -> Optional[GPUDevice]:
        """Refresh dynamic VRAM info for a specific device."""
        output = self._run_nvidia_smi("index,memory.free,memory.used")
        if output is None:
            return None

        now = time.time()
        with self._lock:
            for line in output.split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    idx = int(parts[0])
                    free = float(parts[1])
                    used = float(parts[2])
                except (ValueError, IndexError):
                    continue
                # Update existing device or skip
                for dev in self._devices:
                    if dev.index == idx:
                        dev.memory_free_mb = free
                        dev.memory_used_mb = used
                        dev.last_refresh = now
                        break

            # Return the requested device
            for dev in self._devices:
                if dev.index == device_index:
                    return dev
        return None

    # ── CUDA Context Flush ──────────────────────────────────────────────

    def flush_gpu(self, device_index: int = 0, kill_server: bool = True) -> bool:
        """Reset the CUDA primary context to clear stale driver state.

        Call this before launching llama-server when the previous process
        died uncleanly.  Resets the CUDA context via the driver API
        (cuDevicePrimaryCtxReset), which forces the driver to release all
        resources and start fresh — fixes the transient segfaults caused
        by corrupted CUDA state after crashes.

        Args:
            device_index: GPU to reset.
            kill_server: If True, kill any lingering llama-server processes first.

        Returns:
            True if flush succeeded (or was unnecessary), False on error.
        """
        if kill_server:
            self._kill_stale_servers()

        try:
            import ctypes
            if sys.platform == "win32":
                cuda = ctypes.CDLL("nvcuda.dll")
            else:
                cuda = ctypes.CDLL("libcuda.so.1")

            ret = cuda.cuInit(0)
            if ret != 0:
                _log.warning("flush_gpu: cuInit failed (code %d)", ret)
                return False

            device = ctypes.c_int()
            ret = cuda.cuDeviceGet(ctypes.byref(device), device_index)
            if ret != 0:
                _log.warning("flush_gpu: cuDeviceGet(%d) failed (code %d)",
                             device_index, ret)
                return False

            ret = cuda.cuDevicePrimaryCtxReset_v2(device)
            if ret != 0:
                _log.warning("flush_gpu: cuDevicePrimaryCtxReset failed (code %d)", ret)
                return False

            _log.info("flush_gpu: CUDA primary context reset on device %d", device_index)
            return True

        except OSError as e:
            _log.warning("flush_gpu: CUDA driver library not found: %s", e)
            return False
        except Exception as e:
            _log.error("flush_gpu: unexpected error: %s", e)
            return False

    def _kill_stale_servers(self):
        """Kill any lingering llama-server processes."""
        try:
            if sys.platform == "win32":
                result = subprocess.run(
                    ["taskkill", "/F", "/IM", "llama-server.exe"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    _log.info("flush_gpu: killed stale llama-server process(es)")
                    time.sleep(1)
            else:
                subprocess.run(
                    ["pkill", "-f", "llama-server"],
                    capture_output=True, timeout=5,
                )
                time.sleep(1)
        except Exception as e:
            _log.debug("flush_gpu: no stale servers to kill (%s)", e)

    # ── VRAM-Ready Gate (crash-recovery fix) ───────────────────────────

    def wait_for_vram(
        self,
        needed_mb: float,
        device_index: int = 0,
        timeout: float = DEFAULT_VRAM_WAIT_TIMEOUT,
        poll_interval: float = DEFAULT_VRAM_POLL_INTERVAL,
    ) -> bool:
        """Block until the specified device has enough free VRAM.

        This is the critical crash-recovery gate: after llama-server dies,
        WDDM may take several seconds to reclaim VRAM.  Polling here prevents
        the restart-fail-restart crash loop.

        Args:
            needed_mb: Minimum free VRAM required (MiB).
            device_index: GPU device index to check.
            timeout: Maximum seconds to wait before giving up.
            poll_interval: Seconds between nvidia-smi polls.

        Returns:
            True if VRAM became available within timeout.
            False if timeout was exceeded or nvidia-smi is unavailable.
        """
        if self._nvidia_smi_available is False:
            _log.warning("wait_for_vram: nvidia-smi unavailable, assuming VRAM is ready")
            return True

        _log.info("wait_for_vram: need %.0f MB free on GPU %d (timeout=%.1fs, poll=%.1fs)",
                  needed_mb, device_index, timeout, poll_interval)

        deadline = time.time() + timeout
        iteration = 0

        while True:
            iteration += 1
            dev = self.refresh_device(device_index)

            if dev is None:
                # Can't query device — if nvidia-smi is supposed to work, wait and retry
                if time.time() >= deadline:
                    _log.error("wait_for_vram: timeout after %d polls, device %d unreachable",
                               iteration, device_index)
                    return False
                _log.debug("wait_for_vram: poll %d — device query failed, retrying...",
                           iteration)
                time.sleep(poll_interval)
                continue

            free_mb = dev.memory_free_mb

            if free_mb >= needed_mb:
                _log.info("wait_for_vram: READY after %d poll(s) — "
                          "%.0f MB free >= %.0f MB needed on GPU %d",
                          iteration, free_mb, needed_mb, device_index)
                return True

            if time.time() >= deadline:
                _log.error("wait_for_vram: TIMEOUT after %d polls — "
                           "only %.0f MB free (need %.0f MB) on GPU %d",
                           iteration, free_mb, needed_mb, device_index)
                return False

            _log.debug("wait_for_vram: poll %d — %.0f MB free / %.0f MB needed "
                       "(deficit: %.0f MB), waiting %.1fs...",
                       iteration, free_mb, needed_mb, needed_mb - free_mb, poll_interval)
            time.sleep(poll_interval)

    # ── VRAM Estimation ────────────────────────────────────────────────

    def estimate_allocation_mb(
        self,
        model_path: str,
        num_ctx: int = 8192,
        kv_quant: str = "f16",
        extra_flags: Optional[list] = None,
    ) -> dict:
        """Estimate total VRAM needed to run a model config.

        Components:
          - model_weight_mb: file_size * MODEL_OVERHEAD_FACTOR
          - kv_cache_mb: architecture-derived KV cache for num_ctx tokens
          - compute_buffer_mb: flat estimate for llama.cpp compute graph
          - system_reserve_mb: Windows WDDM + compositor headroom

        Args:
            model_path: Path to the GGUF model file.
            num_ctx: Context length in tokens.
            kv_quant: KV cache quantization type (key in KV_QUANT_BPE).
            extra_flags: Unused currently, reserved for future flags that
                         affect VRAM (e.g., speculative decoding draft model).

        Returns:
            Dict with component breakdown and total_mb.
            Returns a conservative estimate if GGUF can't be read.
        """
        result = {
            "model_weight_mb": 0.0,
            "kv_cache_mb": 0.0,
            "compute_buffer_mb": COMPUTE_BUFFER_MB,
            "system_reserve_mb": SYSTEM_RESERVE_MB,
            "total_mb": 0.0,
            "estimation_method": "exact",
        }

        # Model weight from file size
        try:
            file_size_bytes = os.path.getsize(model_path)
            model_weight_mb = (file_size_bytes / (1024 * 1024)) * MODEL_OVERHEAD_FACTOR
            result["model_weight_mb"] = model_weight_mb
        except OSError as e:
            _log.warning("Cannot stat model file %s: %s — using 0 for weight", model_path, e)
            result["estimation_method"] = "degraded_no_file"

        # KV cache from GGUF architecture
        arch = read_gguf_kv_params(model_path)
        if arch is not None:
            bpe_k = KV_QUANT_BPE.get(kv_quant, KV_QUANT_BPE["f16"])
            bpe_v = KV_QUANT_BPE.get(kv_quant, KV_QUANT_BPE["f16"])

            attn_layers = arch["attn_layer_count"]
            head_count_kv = arch["head_count_kv"]
            key_dim = arch["key_dim"]
            val_dim = arch["val_dim"]

            # KV per token in bytes
            kv_per_token_bytes = attn_layers * head_count_kv * (
                key_dim * bpe_k + val_dim * bpe_v
            )
            kv_cache_bytes = kv_per_token_bytes * num_ctx
            kv_cache_mb = kv_cache_bytes / (1024 * 1024)
            result["kv_cache_mb"] = kv_cache_mb

            _log.debug("VRAM estimate KV: %d attn_layers * %d kv_heads * "
                       "(%.0f key_dim * %.4f bpe + %.0f val_dim * %.4f bpe) * %d ctx "
                       "= %.0f MB",
                       attn_layers, head_count_kv, key_dim, bpe_k, val_dim, bpe_v,
                       num_ctx, kv_cache_mb)
        else:
            # Fallback: rough estimate based on file size ratio
            # Typical KV is ~15-25% of model weight at 8K ctx
            fallback_kv = result["model_weight_mb"] * 0.2 * (num_ctx / 8192)
            result["kv_cache_mb"] = fallback_kv
            result["estimation_method"] = "degraded_no_gguf"
            _log.warning("Cannot read GGUF architecture from %s, "
                         "using fallback KV estimate: %.0f MB", model_path, fallback_kv)

        result["total_mb"] = (
            result["model_weight_mb"]
            + result["kv_cache_mb"]
            + result["compute_buffer_mb"]
            + result["system_reserve_mb"]
        )

        _log.info("VRAM estimate for %s @ %d ctx (%s KV): "
                  "weight=%.0f + kv=%.0f + compute=%d + reserve=%d = %.0f MB total",
                  os.path.basename(model_path), num_ctx, kv_quant,
                  result["model_weight_mb"], result["kv_cache_mb"],
                  result["compute_buffer_mb"], result["system_reserve_mb"],
                  result["total_mb"])

        return result

    # ── Device Selection (multi-GPU) ───────────────────────────────────

    def find_best_device(self, needed_mb: float) -> Optional[int]:
        """Find the GPU with the most free VRAM that exceeds needed_mb.

        Refreshes all devices before selecting.

        Args:
            needed_mb: Minimum free VRAM required.

        Returns:
            Device index of the best candidate, or None if no device qualifies.
        """
        # Refresh all devices
        self._enumerate_devices()

        with self._lock:
            candidates = [
                d for d in self._devices if d.memory_free_mb >= needed_mb
            ]

        if not candidates:
            _log.warning("find_best_device: no GPU has %.0f MB free", needed_mb)
            if self._devices:
                with self._lock:
                    best_available = max(self._devices, key=lambda d: d.memory_free_mb)
                _log.info("  Best available: GPU %d (%s) with %.0f MB free",
                          best_available.index, best_available.name,
                          best_available.memory_free_mb)
            return None

        # Pick the one with the most free VRAM (greedy — avoids fragmentation)
        best = max(candidates, key=lambda d: d.memory_free_mb)
        _log.info("find_best_device: selected GPU %d (%s) — %.0f MB free (need %.0f MB)",
                  best.index, best.name, best.memory_free_mb, needed_mb)
        return best.index

    def get_device_status(self) -> list[dict]:
        """Return status of all devices for health/monitoring endpoints.

        Refreshes VRAM info before returning.
        """
        self._enumerate_devices()

        with self._lock:
            status = []
            for dev in self._devices:
                status.append({
                    "index": dev.index,
                    "name": dev.name,
                    "memory_total_mb": dev.memory_total_mb,
                    "memory_free_mb": dev.memory_free_mb,
                    "memory_used_mb": dev.memory_used_mb,
                    "utilization_pct": round(dev.utilization_pct, 1),
                    "compute_cap": dev.compute_cap,
                    "last_refresh": dev.last_refresh,
                })
        return status

    @property
    def device_count(self) -> int:
        with self._lock:
            return len(self._devices)

    @property
    def nvidia_available(self) -> bool:
        return self._nvidia_smi_available is not False


# ── Module-level convenience ───────────────────────────────────────────

def get_pool() -> GPUPool:
    """Get or create the singleton GPUPool instance."""
    return GPUPool()
