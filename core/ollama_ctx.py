"""
Auto-size num_ctx for Ollama requests based on prompt size and VRAM budget.

Strategy:
  1. Estimate prompt tokens from messages (chars/3, conservative).
  2. Pick smallest standard bucket (4K/8K/16K/32K/64K/131K) that fits prompt.
  3. Cap by a model-specific ceiling derived from file_size / free_vram.
  4. Allow per-model override in ollama_config.json:
     - num_ctx:  fixed value — bypasses auto-sizing entirely
     - max_ctx:  upper bound for the auto-sized value

Design notes:
- Buckets (powers of 2) matter: Ollama reloads the model when num_ctx
  changes between requests. Same-size-class requests stay in one bucket
  and avoid reload thrash.
- file_size/free_vram ratio is the safety heuristic — measurements on a
  24 GB card showed KV-cache growth varies 8× across architectures (dense
  text GQA vs vision), so a bytes-per-token formula would be unsafe.
  The ratio-based cap is monotonic and conservative.
- torch.cuda.mem_get_info() gives true free VRAM in-process, accounting
  for display + other CUDA processes — no subprocess cost.
"""

import json
import logging
import urllib.request
import urllib.error

_log = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
STANDARD_BUCKETS = [4096, 8192, 16384, 32768, 65536, 131072]
SAFETY_FACTOR = 1.3
MIN_CTX = 4096

_model_meta_cache: dict = {}


def estimate_prompt_tokens(messages) -> int:
    """Rough upper bound on prompt tokens. chars/3 is conservative
    (English ~4/tok, JSON/code denser ~3/tok)."""
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


def _get_free_vram_gb() -> float:
    """Free VRAM in GB via torch.cuda.mem_get_info. 0 if no GPU or unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free_bytes, _total = torch.cuda.mem_get_info(0)
        return free_bytes / (1024 ** 3)
    except Exception:
        return 0.0


def _parse_modelfile_num_ctx(parameters_text: str) -> int | None:
    """Extract num_ctx from /api/show's 'parameters' text (whitespace-separated
    'key value' pairs). Returns None if not set.

    Ollama only honors an API-passed num_ctx cleanly when it's >= the Modelfile's
    num_ctx. Passing a smaller value can get silently ignored or cause empty
    responses. This is used as a floor in compute_safe_ctx.
    """
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


def _get_model_meta(model: str) -> dict:
    """Fetch and cache file_size_gb, trained_ctx, and modelfile_num_ctx.
    Returns {} on error (callers must handle missing keys)."""
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
                meta["file_size_gb"] = m.get("size", 0) / (1024 ** 3)
                break

        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/show",
            data=json.dumps({"name": model}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            show = json.loads(resp.read())
        for key, val in show.get("model_info", {}).items():
            if key.endswith(".context_length"):
                meta["trained_ctx"] = int(val)
                break
        mf_ctx = _parse_modelfile_num_ctx(show.get("parameters", ""))
        if mf_ctx:
            meta["modelfile_num_ctx"] = mf_ctx
    except Exception as e:
        _log.debug("ollama_ctx: could not fetch meta for %s: %s", model, e)

    _model_meta_cache[model] = meta
    return meta


def _heuristic_ceiling(model: str) -> int:
    """Conservative ctx ceiling from loaded-VRAM / free-VRAM ratio.

    A model's loaded VRAM is ~1.3x its file size at minimal ctx (weights +
    CUDA context + minimal KV + buffers). Using raw file_size as the ratio
    was too permissive — e.g. qwen3.5:27B Q4 (16.2 GB file, 21.1 GB loaded
    at 4K ctx on a 24 GB card) looked "safe at 8K" but measurement showed
    Ollama silently spilled to CPU above 4K.

    Thresholds also bake in worst-case KV growth observed across models
    (qwen3-vl grew ~150 KB/ctx-token vs gemma4 ~18 KB/ctx-token).
    """
    meta = _get_model_meta(model)
    file_gb = meta.get("file_size_gb", 0.0)
    free_gb = _get_free_vram_gb()

    if not file_gb or not free_gb:
        return 32768  # safe default when info unavailable

    effective_ratio = (file_gb * 1.3) / free_gb
    if effective_ratio > 0.90:
        return MIN_CTX     # barely fits — no room for ctx growth
    if effective_ratio > 0.75:
        return 8192
    if effective_ratio > 0.60:
        return 16384
    if effective_ratio > 0.40:
        return 32768
    return 65536


def compute_safe_ctx(model: str, estimated_tokens: int, config: dict | None = None) -> int:
    """Pick num_ctx for an Ollama request.

    Precedence:
      1. config['num_ctx']   — fixed, bypass auto-sizing
      2. config['max_ctx']   — upper bound for the auto-sized value
      3. heuristic ceiling   — model_file / free_vram

    Floor: the Modelfile's num_ctx (if any). Passing an API num_ctx below the
    Modelfile value is what historically caused empty-content responses, since
    Ollama may silently ignore the smaller value and reload at the Modelfile
    default.

    Always a standard bucket (>=4K), never above the model's trained ctx.
    """
    config = config or {}

    if "num_ctx" in config:
        return int(config["num_ctx"])

    target = max(int(estimated_tokens * SAFETY_FACTOR), MIN_CTX)

    meta = _get_model_meta(model)
    ceiling = _heuristic_ceiling(model)
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
