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


def _classify_capabilities(name: str) -> list[str]:
    caps = ["text"]
    for p in _VISION_PATTERNS:
        if p.search(name):
            caps.append("vision")
            break
    for p in _CODE_PATTERNS:
        if p.search(name):
            caps.append("code")
            break
    return caps


def _discover_ollama() -> list[dict]:
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
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
        models.append({
            "id": display,
            "backend": "ollama",
            "size": m.get("size", 0),
            "modified": m.get("modified_at", ""),
            "capabilities": _classify_capabilities(display),
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
        models.append({
            "id": name,
            "backend": "llama_cpp",
            "size": 0,
            "capabilities": _classify_capabilities(name),
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
        models.append({
            "id": name,
            "backend": "transformers",
            "size": size,
            "capabilities": _classify_capabilities(name),
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
