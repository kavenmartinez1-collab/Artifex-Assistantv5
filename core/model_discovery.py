"""
Artifex Assistant V5 — Centralized model discovery.

Single source of truth for available models across all backends.
Discovers dynamically at runtime from:
  - Ollama API (localhost:11434/api/tags)
  - llama.cpp config (llama_cpp_config.json)
  - Transformers model directories (models/)

No model names are hardcoded — everything is discovered from what's
actually installed on the system.
"""

import json
import os
import re
import time
import urllib.request
import urllib.error
import logging

_log = logging.getLogger(__name__)

_CACHE_TTL = 10  # seconds — models don't change every request

_cached_models: list[dict] | None = None
_cache_time: float = 0.0

_VISION_PATTERNS = [
    re.compile(r"vl", re.I),
    re.compile(r"vision", re.I),
    re.compile(r"pixtral", re.I),
]

_CODE_PATTERNS = [
    re.compile(r"coder", re.I),
    re.compile(r"starcoder", re.I),
    re.compile(r"codellama", re.I),
    re.compile(r"deepseek.*code", re.I),
]


def _classify_capabilities(
    name: str, *, llama_cpp_config: dict | None = None,
) -> list[str]:
    caps = ["text"]
    has_vision = False
    # llama_cpp publishes vision capability via --mmproj on the launch
    # command — that's the authoritative signal the loaded server can
    # accept image inputs.  Trust it before falling back to name guesses.
    if llama_cpp_config:
        flags = llama_cpp_config.get("extra_flags") or []
        if "--mmproj" in flags:
            has_vision = True
    if not has_vision:
        for p in _VISION_PATTERNS:
            if p.search(name):
                has_vision = True
                break
    if has_vision:
        caps.append("vision")
    for p in _CODE_PATTERNS:
        if p.search(name):
            caps.append("code")
            break
    return caps


# Cap for the recommended_max_completion derivation.  16384 is the
# largest output budget that's still reasonable on a 128K-ctx model
# without leaving the prompt budget too thin; on smaller-ctx models the
# ctx//2 floor takes over before this kicks in.
_RECOMMENDED_MAX_COMPLETION_CAP = 16384


def _compute_recommended_max_completion(context_length: int) -> int:
    """Derive a sensible default cap for client-requested max_tokens.

    Returns half of context_length, capped at 16384, so the prompt
    always has at least half the window to work with.  Returns 0 when
    context_length is unknown so callers can fall back to their own
    defaults rather than picking from a fabricated number.
    """
    if context_length <= 0:
        return 0
    return min(context_length // 2, _RECOMMENDED_MAX_COMPLETION_CAP)


def _get_ollama_context_length(model_name: str) -> int:
    """Best-effort context length for an Ollama model.

    Pulls trained_ctx from /api/show via the cached helper in
    core.ollama_ctx.  This is the model's architectural max, which is
    what clients need for batching decisions — not the Modelfile's
    num_ctx (which is a runtime floor, not a cap).  Returns 0 when
    Ollama is unreachable or the metadata lacks the field.
    """
    try:
        from core.ollama_ctx import _get_model_meta
        meta = _get_model_meta(model_name)
        return int(meta.get("trained_ctx") or 0)
    except Exception as e:
        _log.debug("Could not get context_length for %s: %s", model_name, e)
        return 0


def _get_transformers_context_length(model_path: str) -> int:
    """Best-effort context length for a transformers model.

    Reads max_position_embeddings from the model's config.json.  This is
    the architectural maximum — runtime caps from quantization or
    rope_scaling are applied at load time, not here.  Returns 0 when
    config.json is missing or unparseable.
    """
    config_path = os.path.join(model_path, "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return 0
    val = cfg.get("max_position_embeddings")
    if val is None:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _discover_ollama() -> list[dict]:
    try:
        _ollama_url = os.environ.get("ARTIFEX_OLLAMA_URL", "http://localhost:11434").rstrip("/")
        req = urllib.request.Request(f"{_ollama_url}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return []

    models = []
    for m in data.get("models", []):
        name = m.get("name", "")
        if not name:
            continue
        display = name.rsplit(":latest", 1)[0] if name.endswith(":latest") else name
        ctx_len = _get_ollama_context_length(display)
        models.append({
            "id": display,
            "backend": "ollama",
            "size": m.get("size", 0),
            "modified": m.get("modified_at", ""),
            "capabilities": _classify_capabilities(display),
            "context_length": ctx_len,
            "recommended_max_completion": _compute_recommended_max_completion(ctx_len),
            # Image-token cost is mmproj/projector-specific and not
            # discoverable from /api/show today; null lets clients fall
            # back to their own per-family heuristic.
            "image_token_cost_per_megapixel": None,
        })
    return models


def _discover_llama_cpp() -> list[dict]:
    from core.config import LLAMA_CPP_CONFIG_PATH
    if not os.path.isfile(LLAMA_CPP_CONFIG_PATH):
        return []
    try:
        with open(LLAMA_CPP_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    models = []
    for name, mcfg in cfg.get("models", {}).items():
        ctx_len = int(mcfg.get("num_ctx") or 0)
        # Optional per-model field — set in llama_cpp_config.json when
        # the user knows the projector's token cost (varies by mmproj).
        # Null when not declared.
        img_cost = mcfg.get("image_token_cost_per_megapixel")
        models.append({
            "id": name,
            "backend": "llama_cpp",
            "size": 0,
            "capabilities": _classify_capabilities(name, llama_cpp_config=mcfg),
            "context_length": ctx_len,
            "recommended_max_completion": _compute_recommended_max_completion(ctx_len),
            "image_token_cost_per_megapixel": img_cost,
        })
    return models


def _discover_transformers() -> list[dict]:
    from core.config import MODELS
    models = []
    for name, path in MODELS.items():
        size = 0
        try:
            for root, _, files in os.walk(path):
                for f in files:
                    if f.endswith((".safetensors", ".bin", ".pt")):
                        size += os.path.getsize(os.path.join(root, f))
        except OSError:
            pass
        ctx_len = _get_transformers_context_length(path)
        models.append({
            "id": name,
            "backend": "transformers",
            "size": size,
            "capabilities": _classify_capabilities(name),
            "context_length": ctx_len,
            "recommended_max_completion": _compute_recommended_max_completion(ctx_len),
            "image_token_cost_per_megapixel": None,
        })
    return models


def discover_all(force_refresh: bool = False) -> list[dict]:
    """Discover models from all backends. Cached with short TTL."""
    global _cached_models, _cache_time
    now = time.monotonic()
    if not force_refresh and _cached_models is not None and (now - _cache_time) < _CACHE_TTL:
        return _cached_models

    all_models = []
    all_models.extend(_discover_ollama())
    all_models.extend(_discover_llama_cpp())
    all_models.extend(_discover_transformers())

    _cached_models = all_models
    _cache_time = now
    return all_models


def discover_by_backend(backend: str, force_refresh: bool = False) -> list[dict]:
    """Discover models for a specific backend."""
    return [m for m in discover_all(force_refresh) if m["backend"] == backend]


def get_model_names_for_backend(backend: str) -> list[str]:
    """Get just the model names for a backend. Drop-in replacement for config.get_model_names()."""
    return [m["id"] for m in discover_by_backend(backend)]


def get_default_model(backend: str = None) -> str | None:
    """Pick the best default model for the given backend.

    For Ollama: picks the first available text model (skips VL models).
    For llama.cpp: picks the first configured model.
    For Transformers: picks the first scanned model directory.
    Returns None if no models are available.
    """
    if backend is None:
        from core.config import get_active_backend
        backend = get_active_backend()

    models = discover_by_backend(backend)
    if not models:
        return None

    text_only = [m for m in models if "vision" not in m["capabilities"]]
    if text_only:
        return text_only[0]["id"]
    return models[0]["id"]


def invalidate_cache():
    """Force next discover_all() call to re-scan."""
    global _cached_models
    _cached_models = None
