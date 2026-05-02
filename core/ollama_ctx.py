"""
Auto-size num_ctx for Ollama requests based on VRAM budget and model architecture.

Strategy:
  1. Query total VRAM via nvidia-smi (cached — never changes at runtime).
  2. Fetch model architecture from Ollama /api/show: layers, KV heads,
     key/value dims, and attention interval (hybrid attention+SSM models).
  3. Compute KV cache cost per context token from architecture.
  4. Budget: total_vram − 2.5 GB reserve − model_loaded(file×1.3) = available for KV.
  5. max_ctx = available / kv_per_token, rounded down to standard bucket.
  6. Per-model override via ollama_config.json:
     - num_ctx:      fixed value — bypasses auto-sizing entirely
     - max_ctx:      upper bound for the auto-sized value
     - kv_quant:     KV cache quantization (f16/q8_0/q4_0) for sizing math
     - num_kv_heads: override GQA head count when GGUF metadata is missing

Hybrid models (e.g. Qwen3.5/3.6):
  full_attention_interval=4 means only every 4th layer has an attention KV
  cache.  SSM layers (Mamba-2) have fixed-size state that doesn't scale with
  context length.  The KV math accounts for this automatically.

The 1.1× model overhead factor covers weight tensor padding, CUDA context,
and compute buffers.  Empirical: Qwen3.6-27B Q4_K_M (16.2 GB file) loads
at ~17–18 GB VRAM at minimal ctx on a 24 GB card.
"""

import json
import logging
import os
import subprocess
import sys
import urllib.request
import urllib.error

_log = logging.getLogger(__name__)

OLLAMA_BASE = os.environ.get("ARTIFEX_OLLAMA_URL", "http://localhost:11434").rstrip("/")
STANDARD_BUCKETS = [4096, 8192, 16384, 32768, 65536, 131072]
MIN_CTX = 4096
MODEL_OVERHEAD_FACTOR = 1.1
SYSTEM_RESERVE_MB = 2560  # 2.5 GB for display compositor, other GPU processes

_total_vram_mb: float | None = None
_model_meta_cache: dict = {}

_KV_QUANT_BYTES = {
    "f16": 2.0,
    "q8_0": 1.0625,   # 32 int8 + 1 f16 scale per block of 32
    "q4_0": 0.5625,   # 16 int4-pairs + 1 f16 scale per block of 32
}


def estimate_prompt_tokens(messages) -> int:
    """Rough upper bound on prompt tokens.  chars/3 is conservative
    (English ~4 chars/tok, JSON/code denser ~3 chars/tok)."""
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content) // 3
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(part.get("text", "")) // 3
    return total


# ── VRAM detection ──────────────────────────────────────────────────────

def _get_total_vram_mb() -> float:
    """Total GPU VRAM in MiB via nvidia-smi.  Cached after first successful call."""
    global _total_vram_mb
    if _total_vram_mb is not None:
        return _total_vram_mb
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            _total_vram_mb = float(result.stdout.strip().split("\n")[0])
            _log.info("Detected total VRAM: %.0f MiB", _total_vram_mb)
            return _total_vram_mb
    except Exception as e:
        _log.debug("nvidia-smi unavailable: %s", e)
    _total_vram_mb = 0.0
    return 0.0


def _get_free_vram_mb() -> float:
    """Current free GPU VRAM in MiB via nvidia-smi.

    Unlike _get_total_vram_mb(), NOT cached — free VRAM changes as
    processes allocate/release memory.  Called once per sizing decision.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except Exception as e:
        _log.debug("nvidia-smi free query failed: %s", e)
    return 0.0


# ── Model metadata ──────────────────────────────────────────────────────

def _parse_modelfile_num_ctx(parameters_text: str) -> int | None:
    """Extract num_ctx from Ollama's 'parameters' text block."""
    if not parameters_text:
        return None
    for line in parameters_text.splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and parts[0] == "num_ctx":
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


_SUFFIX_TO_FIELD = {
    ".context_length": "trained_ctx",
    ".block_count": "num_layers",
    ".attention.head_count_kv": "num_kv_heads",
    ".attention.head_count": "num_heads",
    ".attention.key_length": "key_length",
    ".attention.value_length": "value_length",
    ".embedding_length": "embedding_length",
}


def _get_model_meta(model: str) -> dict:
    """Fetch and cache model metadata from Ollama /api/show and /api/tags.

    Extracts: file_size_mb, trained_ctx, num_layers, num_kv_heads,
    key_length, value_length, attn_layer_count (accounting for hybrid models),
    and modelfile_num_ctx.
    """
    if model in _model_meta_cache:
        return _model_meta_cache[model]

    meta: dict = {}
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            tags = json.loads(resp.read())
        for m in tags.get("models", []):
            name = m.get("name", "")
            if name == model or name.startswith(model + ":"):
                meta["file_size_mb"] = m.get("size", 0) / (1024 ** 2)
                break

        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/show",
            data=json.dumps({"name": model}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            show = json.loads(resp.read())

        model_info = show.get("model_info", {})
        full_attn_interval = None

        # Multimodal models (Gemma 4, etc.) expose sub-component keys
        # (e.g. ".vision.block_count") that share suffixes with the main
        # text model.  Pick the shortest matching key per suffix — the
        # main model key is always shorter than sub-component keys.
        suffix_winners: dict[str, tuple[str, int]] = {}
        for key, val in model_info.items():
            if val is None:
                continue
            if key.endswith(".full_attention_interval"):
                if full_attn_interval is None or len(key) < full_attn_interval[1]:
                    full_attn_interval = (int(val), len(key))
                continue
            for suffix, field in _SUFFIX_TO_FIELD.items():
                if key.endswith(suffix):
                    if suffix not in suffix_winners or len(key) < len(suffix_winners[suffix][0]):
                        suffix_winners[suffix] = (key, int(val))
                    break

        for suffix, (_key, val) in suffix_winners.items():
            meta[_SUFFIX_TO_FIELD[suffix]] = val

        if full_attn_interval is not None:
            full_attn_interval = full_attn_interval[0]

        # How many layers actually have attention KV caches?
        # Hybrid models (attention+SSM) only run attention every N layers.
        num_layers = meta.get("num_layers", 0)
        if full_attn_interval and full_attn_interval > 1 and num_layers:
            meta["attn_layer_count"] = num_layers // full_attn_interval
        else:
            meta["attn_layer_count"] = num_layers

        # Head dimensions: prefer explicit key_length/value_length,
        # fall back to embedding_length / num_heads for pure transformers.
        if "key_length" not in meta and "embedding_length" in meta and "num_heads" in meta:
            dim = meta["embedding_length"] // meta["num_heads"]
            meta["key_length"] = dim
            meta["value_length"] = dim

        mf_ctx = _parse_modelfile_num_ctx(show.get("parameters", ""))
        if mf_ctx:
            meta["modelfile_num_ctx"] = mf_ctx

    except Exception as e:
        _log.debug("Could not fetch model meta for %s: %s", model, e)

    _model_meta_cache[model] = meta
    _log.info("Model meta for %s: %s", model,
              {k: v for k, v in meta.items() if k != "file_size_mb"})
    return meta


# ── KV cache math ───────────────────────────────────────────────────────

def _kv_bytes_per_token(meta: dict, kv_quant: str = "f16",
                        kv_heads_override: int | None = None) -> float | None:
    """Bytes of KV cache per context token, derived from model architecture.

    For hybrid attention+SSM models, only attention layers contribute to
    per-token KV cache growth.  SSM state is fixed-size and folded into
    the model overhead factor.

    Returns None when architecture info is missing.
    """
    attn_layers = meta.get("attn_layer_count")
    num_kv_heads = (kv_heads_override
                    or meta.get("num_kv_heads")
                    or meta.get("num_heads"))
    key_len = meta.get("key_length")
    val_len = meta.get("value_length")

    if not all((attn_layers, num_kv_heads, key_len)):
        return None

    if not val_len:
        val_len = key_len

    bytes_per_elem = _KV_QUANT_BYTES.get(kv_quant, 2.0)
    return attn_layers * num_kv_heads * (key_len + val_len) * bytes_per_elem


def _vram_max_ctx(model: str, kv_quant: str = "f16",
                  kv_heads_override: int | None = None) -> int:
    """Largest context bucket that fits in VRAM after model + system reserve."""
    total_mb = _get_total_vram_mb()
    if not total_mb:
        return 8192

    meta = _get_model_meta(model)
    file_mb = meta.get("file_size_mb", 0)
    if not file_mb:
        return 8192

    model_loaded_mb = file_mb * MODEL_OVERHEAD_FACTOR
    available_mb = total_mb - SYSTEM_RESERVE_MB - model_loaded_mb

    if available_mb <= 0:
        _log.warning(
            "Model %s (est. %.0f MiB loaded) exceeds VRAM budget "
            "(%.0f MiB total − %.0f MiB reserve)",
            model, model_loaded_mb, total_mb, SYSTEM_RESERVE_MB,
        )
        return MIN_CTX

    kv_per_token = _kv_bytes_per_token(meta, kv_quant, kv_heads_override)
    if not kv_per_token:
        _log.debug("No architecture info for %s — falling back to 8K", model)
        return 8192

    kv_per_token_mb = kv_per_token / (1024 * 1024)
    # 90% utilization — reserve 10% for compute buffers that scale with
    # context (attention scratch, graph intermediates, allocator fragmentation).
    max_tokens = int((available_mb * 0.90) / kv_per_token_mb)

    free_mb = _get_free_vram_mb()
    if free_mb > 0:
        if free_mb > model_loaded_mb:
            # Model not yet loaded — subtract its size from free VRAM.
            free_available = free_mb - model_loaded_mb
        else:
            # Model likely already loaded — free VRAM is what's left for KV.
            free_available = free_mb - SYSTEM_RESERVE_MB
        if free_available > 0:
            free_max = int((free_available * 0.90) / kv_per_token_mb)
            if free_max < max_tokens:
                _log.info(
                    "VRAM ceiling: free=%.0f MiB → max %d tokens "
                    "(down from %d based on total VRAM)",
                    free_mb, free_max, max_tokens,
                )
                max_tokens = free_max

    _log.info(
        "VRAM budget for %s: %.0f MiB total (%.0f MiB free), −%.0f MiB reserve, "
        "−%.0f MiB model ≈ %.0f MiB for KV → max ~%d tokens "
        "(%.1f KB/tok, kv_quant=%s, attn_layers=%d/%d)",
        model, total_mb, free_mb, SYSTEM_RESERVE_MB, model_loaded_mb,
        available_mb, max_tokens, kv_per_token / 1024, kv_quant,
        meta.get("attn_layer_count", 0), meta.get("num_layers", 0),
    )

    for bucket in reversed(STANDARD_BUCKETS):
        if bucket <= max_tokens:
            return bucket
    return MIN_CTX


# ── Public API ──────────────────────────────────────────────────────────

def compute_safe_ctx(model: str, estimated_tokens: int,
                     config: dict | None = None) -> int:
    """Pick num_ctx for an Ollama request.

    Precedence:
      1. config['num_ctx']   — fixed override, bypasses auto-sizing
      2. config['max_ctx']   — upper bound for auto-sized value
      3. VRAM budget         — from nvidia-smi + model architecture

    Floor: the Modelfile's num_ctx (passing a smaller value to the API
    gets silently ignored by Ollama and can cause empty responses).

    Always returns a standard bucket (>= 4K), never above trained ctx.
    """
    config = config or {}

    if "num_ctx" in config:
        return int(config["num_ctx"])

    kv_quant = config.get("kv_quant", os.environ.get("OLLAMA_KV_CACHE_TYPE", "f16"))
    kv_heads = config.get("num_kv_heads")

    target = max(int(estimated_tokens * 1.3), MIN_CTX)

    meta = _get_model_meta(model)
    ceiling = _vram_max_ctx(model, kv_quant, kv_heads)
    if "trained_ctx" in meta:
        ceiling = min(ceiling, meta["trained_ctx"])
    if "max_ctx" in config:
        ceiling = min(ceiling, int(config["max_ctx"]))

    floor = max(MIN_CTX, meta.get("modelfile_num_ctx", 0))
    if floor > ceiling:
        ceiling = floor

    for bucket in STANDARD_BUCKETS:
        if bucket >= target and bucket >= floor:
            return min(bucket, ceiling)
    return min(STANDARD_BUCKETS[-1], ceiling)


def should_disable_mmap(model: str | None = None,
                        model_config: dict | None = None) -> bool:
    """Whether to pass use_mmap=false to Ollama.

    On Windows, mmap backs GGUF files with pagefile commit charge even when
    weights are fully offloaded to GPU — can exhaust the pagefile and OOM.
    Only auto-disabled when the model fits entirely in VRAM (full offload
    viable).  For partial offload, CPU-resident layers need accessible
    weights and disabling mmap would just shift the allocation pressure
    to system RAM instead.

    Per-model override: set "use_mmap": true/false in ollama_config.json.
    """
    if model_config and "use_mmap" in model_config:
        return not model_config["use_mmap"]
    if sys.platform != "win32":
        return False

    total_vram = _get_total_vram_mb()
    if total_vram <= 0:
        return False

    if model:
        meta = _get_model_meta(model)
        file_mb = meta.get("file_size_mb", 0)
        if file_mb > 0:
            loaded_mb = file_mb * MODEL_OVERHEAD_FACTOR
            if loaded_mb > total_vram * 0.85:
                _log.info(
                    "Keeping mmap for %s: loaded estimate %.0f MiB "
                    "exceeds 85%% of %.0f MiB VRAM — partial offload path",
                    model, loaded_mb, total_vram,
                )
                return False

    return True
