"""
Artifex Assistant V5 — Centralized configuration.
Auto-detects GPU tier and scales token limits / context windows accordingly.
Supports runtime model/backend switching and context profile toggling.
"""

import json
import os
import platform
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass

# ===== PLATFORM =====
IS_WINDOWS = platform.system() == "Windows"
PLATFORM_STRING = f"{platform.system()} ({platform.machine()})"

# ===== PATHS =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = None  # resolved lazily from MODELS dict on first access
SESSION_DIR = os.path.join(BASE_DIR, "sessions")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
KNOWLEDGE_REFERENCE_DIR = os.path.join(KNOWLEDGE_DIR, "reference")

# ===== WEB GATEWAY =====
# URL of the web gateway proxy (set automatically in Docker, empty for local dev).
# When set, web search/fetch/download are routed through the gateway for
# content sanitization, prompt injection detection, and quarantine storage.
WEB_GATEWAY_URL = os.getenv("WEB_GATEWAY_URL", "")
GATEWAY_AUTH_TOKEN = os.getenv("GATEWAY_AUTH_TOKEN", "")

# ===== MODEL REGISTRY =====
# Auto-discovers model directories in models/. Skip quantized cache dirs.
# Also checks shared model dirs (artifex-lite-v3/models, Artifex-lite/models).
# Add new models by downloading to models/ — they appear in the GUI automatically.
MODELS = {}
_model_dirs = [
    os.path.join(BASE_DIR, "models"),
    os.path.join(os.path.dirname(BASE_DIR), "artifex-lite-v3", "models"),
    os.path.join(os.path.dirname(BASE_DIR), "Artifex-lite", "models"),
]
for _models_dir in _model_dirs:
    if os.path.isdir(_models_dir):
        for _name in sorted(os.listdir(_models_dir)):
            _full = os.path.join(_models_dir, _name)
            if (os.path.isdir(_full) and not _name.endswith("-nf4-cached")
                    and not _name.startswith(".") and _name not in MODELS):
                MODELS[_name] = _full

# ===== THREAD-SAFE MUTABLE STATE =====
# All mutable runtime state is protected by this lock.
# Reads are safe without the lock (Python GIL guarantees atomic reads of
# simple attributes), but writes that depend on reads must hold _config_lock.
_config_lock = threading.Lock()

_active_model = None  # resolved lazily on first call

# ===== BACKEND SWITCHING =====
_active_backend = "transformers"  # "transformers", "ollama", or "llama_cpp"


def get_active_backend():
    """Get the currently active backend name."""
    return _active_backend


def set_active_backend(name):
    """Switch active backend. Returns True on success."""
    global _active_backend
    need_refresh = False
    with _config_lock:
        if name in ("transformers", "ollama", "llama_cpp"):
            _active_backend = name
            need_refresh = (name == "ollama" and not OLLAMA_MODELS)
        else:
            return False
    if need_refresh:
        refresh_ollama_models()
    return True


# ===== INFERENCE OPTIMIZATIONS =====
_torch_compile_enabled = False
_turboquant_kv_enabled = False


def get_torch_compile():
    """Whether torch.compile() is applied to the model for faster inference."""
    return _torch_compile_enabled


def set_torch_compile(enabled):
    """Toggle torch.compile(). Requires model reload to take effect."""
    global _torch_compile_enabled
    with _config_lock:
        _torch_compile_enabled = bool(enabled)
    return True


def get_turboquant_kv():
    """Whether TurboQuant KV cache compression is active."""
    return _turboquant_kv_enabled


def set_turboquant_kv(enabled):
    """Toggle TurboQuant KV cache. Takes effect on next generation."""
    global _turboquant_kv_enabled
    with _config_lock:
        _turboquant_kv_enabled = bool(enabled)
    return True


# ===== OLLAMA CONFIGURATION =====
OLLAMA_MODELS = {}  # auto-discovered from running Ollama server
OLLAMA_NUM_GPU = "auto"  # "auto", 0, -1, or specific int
OLLAMA_CONFIG_PATH = os.path.join(BASE_DIR, "ollama_config.json")

# Per-model Ollama settings (num_ctx, etc.) — persisted to ollama_config.json
_ollama_model_config: dict = {}
_ollama_config_mtime: float = 0.0


def _load_ollama_config():
    """Load per-model Ollama config from disk.  Re-reads when file mtime changes."""
    global _ollama_model_config, _ollama_config_mtime
    if not os.path.isfile(OLLAMA_CONFIG_PATH):
        return _ollama_model_config
    try:
        mtime = os.path.getmtime(OLLAMA_CONFIG_PATH)
        if mtime == _ollama_config_mtime and _ollama_model_config:
            return _ollama_model_config
        with open(OLLAMA_CONFIG_PATH, "r", encoding="utf-8") as f:
            _ollama_model_config = json.load(f)
        _ollama_config_mtime = mtime
    except (json.JSONDecodeError, OSError):
        _ollama_model_config = {}
    return _ollama_model_config


def _save_ollama_config():
    """Persist per-model Ollama config to disk."""
    try:
        with open(OLLAMA_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(_ollama_model_config, f, indent=2)
    except OSError:
        pass  # non-critical


def _estimate_default_ctx(model_size_gb: float, gpu_tier: str = None) -> int:
    """Estimate a reasonable num_ctx default based on model size and GPU tier.

    Strategy:
    - Smaller models can afford larger context (more VRAM headroom)
    - Larger models need smaller context to fit in VRAM
    - GPU tier caps the maximum regardless of model size

    Args:
        model_size_gb: Model file size in GB (from Ollama)
        gpu_tier: "TIGHT", "COMFORTABLE", or "ABUNDANT"

    Returns:
        Recommended num_ctx value
    """
    tier = gpu_tier or GPU_TIER

    # Tier-based caps (based on Ollama's VRAM-based defaults)
    tier_caps = {
        "TIGHT": 8192,       # <=12GB VRAM — conservative
        "COMFORTABLE": 16384,  # 13-20GB VRAM — moderate
        "ABUNDANT": 32768,   # >20GB VRAM — generous
    }
    max_ctx = tier_caps.get(tier, 8192)

    # Model size heuristics (Q4 quantized sizes)
    # Smaller models = more room for larger context
    if model_size_gb <= 3:
        base_ctx = 16384  # <3GB models (0.5B-2B params)
    elif model_size_gb <= 6:
        base_ctx = 12288  # 3-6GB models (~4B-9B params)
    elif model_size_gb <= 10:
        base_ctx = 8192   # 6-10GB models (~9B-14B params)
    elif model_size_gb <= 18:
        base_ctx = 6144   # 10-18GB models (~27B params)
    else:
        base_ctx = 4096   # >18GB models (70B+ params)

    return min(base_ctx, max_ctx)


def get_ollama_model_config(model_name: str) -> dict:
    """Get per-model Ollama config (num_ctx, etc.).

    Returns config dict. Only includes num_ctx if explicitly configured —
    passing num_ctx to Ollama can trigger bugs where content returns empty.
    When not configured, returns empty dict so Ollama uses its own defaults.
    """
    _load_ollama_config()

    # Normalize model name (strip :latest suffix)
    normalized = model_name.rsplit(":latest", 1)[0] if model_name.endswith(":latest") else model_name

    # Check explicit config (user set via /ctx or edited config file)
    if normalized in _ollama_model_config:
        return _ollama_model_config[normalized]

    # Check _default fallback — but skip "_comment" entries
    if "_default" in _ollama_model_config:
        return _ollama_model_config["_default"]

    # No explicit config — return empty dict, let Ollama use its defaults
    # (Passing num_ctx explicitly can cause empty content bug in some models)
    return {}


def set_ollama_model_config(model_name: str, config: dict, persist: bool = True):
    """Set per-model Ollama config.

    Args:
        model_name: Ollama model name (e.g., "qwen3.5:9b")
        config: Dict with settings like {"num_ctx": 8192}
        persist: Whether to save to disk immediately
    """
    global _ollama_model_config
    _load_ollama_config()

    normalized = model_name.rsplit(":latest", 1)[0] if model_name.endswith(":latest") else model_name
    _ollama_model_config[normalized] = config

    if persist:
        _save_ollama_config()


def get_all_ollama_model_configs() -> dict:
    """Get all per-model Ollama configs (for UI/inspection)."""
    if not _ollama_model_config:
        _load_ollama_config()
    return dict(_ollama_model_config)


def _discover_ollama_models():
    """Query locally running Ollama server for installed models.

    Returns:
        (dict, error_str or None) — model_name -> {name, size, ...}
    """
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return {}, "Ollama not reachable at localhost:11434"

    models = {}
    for m in data.get("models", []):
        name = m.get("name", "")
        if name:
            # Strip :latest suffix for cleaner display
            display = name.rsplit(":latest", 1)[0] if name.endswith(":latest") else name
            models[display] = {
                "name": name,
                "size": m.get("size", 0),
                "modified": m.get("modified_at", ""),
            }
    return models, None


def refresh_ollama_models(retries=3, delay=1.0):
    """Refresh OLLAMA_MODELS dict. Retries to handle slow Ollama startup.

    Returns:
        (success: bool, error: str or None)
    """
    global OLLAMA_MODELS
    for attempt in range(retries):
        discovered, err = _discover_ollama_models()
        if discovered:
            OLLAMA_MODELS.update(discovered)
            return True, None
        if attempt < retries - 1:
            time.sleep(delay)
    return False, err


# ===== LLAMA.CPP SERVER CONFIGURATION =====
LLAMA_CPP_CONFIG_PATH = os.path.join(BASE_DIR, "llama_cpp_config.json")
_llama_cpp_config: dict = {}
_llama_cpp_config_mtime: float = 0.0


def _load_llama_cpp_config():
    """Load llama.cpp config from disk.  Re-reads when file mtime changes."""
    global _llama_cpp_config, _llama_cpp_config_mtime
    if not os.path.isfile(LLAMA_CPP_CONFIG_PATH):
        return _llama_cpp_config
    try:
        mtime = os.path.getmtime(LLAMA_CPP_CONFIG_PATH)
        if mtime == _llama_cpp_config_mtime and _llama_cpp_config:
            return _llama_cpp_config
        with open(LLAMA_CPP_CONFIG_PATH, "r", encoding="utf-8") as f:
            _llama_cpp_config = json.load(f)
        _llama_cpp_config_mtime = mtime
    except (json.JSONDecodeError, OSError):
        _llama_cpp_config = {}
    return _llama_cpp_config


def get_llama_cpp_models() -> dict:
    """Get available llama.cpp models from config. Returns {name: config_dict}."""
    cfg = _load_llama_cpp_config()
    return cfg.get("models", {})


def get_llama_cpp_model_config(model_name: str) -> dict:
    """Get config for a specific llama.cpp model. Merges global defaults."""
    cfg = _load_llama_cpp_config()
    models = cfg.get("models", {})
    if model_name not in models:
        return {}
    model_cfg = dict(models[model_name])
    if "server_path" not in model_cfg:
        model_cfg["server_path"] = cfg.get("server_path", "llama-server")
    if "port" not in model_cfg:
        model_cfg["port"] = cfg.get("default_port", 8081)
    return model_cfg


# ===== MODEL SELECTION (all backends) =====

def get_active_model_path():
    """Get the currently selected transformers model's directory path."""
    global _active_model, MODEL_PATH
    with _config_lock:
        if _active_model is None:
            if MODEL_PATH is None and MODELS:
                MODEL_PATH = next(iter(MODELS.values()))
            _active_model = MODEL_PATH
        return _active_model


def set_active_model(name):
    """Switch active model by registry name. Works for all backends."""
    global _active_model
    with _config_lock:
        if _active_backend == "ollama":
            if name in OLLAMA_MODELS:
                _active_model = name
                return True
            _active_model = name
            return True
        if _active_backend == "llama_cpp":
            if name in get_llama_cpp_models():
                _active_model = name
                return True
            return False
        if name in MODELS:
            _active_model = MODELS[name]
            return True
        return False


def get_active_model_name():
    """Get display name of the currently active model."""
    if _active_backend == "ollama":
        return get_active_ollama_model()
    if _active_backend == "llama_cpp":
        return get_active_llama_cpp_model()
    path = get_active_model_path()
    for name, p in MODELS.items():
        if p == path:
            return name
    return os.path.basename(path)


def get_active_ollama_model():
    """Get the active Ollama model name.  Returns None if no models available."""
    global _active_model
    if _active_model and isinstance(_active_model, str):
        if not os.sep in _active_model and not _active_model.startswith("/"):
            return _active_model
    if OLLAMA_MODELS:
        return next(iter(OLLAMA_MODELS.keys()))
    return None


def get_active_llama_cpp_model():
    """Get the active llama.cpp model name."""
    global _active_model
    models = get_llama_cpp_models()
    if _active_model and _active_model in models:
        return _active_model
    if models:
        return next(iter(models.keys()))
    return None


def get_model_names():
    """Get list of available model names for current backend."""
    from core.model_discovery import get_model_names_for_backend
    names = get_model_names_for_backend(_active_backend)
    if not names and _active_backend == "ollama":
        refresh_ollama_models()
        names = get_model_names_for_backend(_active_backend)
    return names

# ===== QUANTIZATION =====
# Note: compute dtype (fp16 vs bf16) is determined at runtime by the GPU's
# compute capability — see model_loader.py.
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}

# ===== TQDM / HF SILENCING =====
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ===== CUDA MEMORY OPTIMIZATIONS =====
# Lazy module loading: defers CUDA kernel JIT compilation until first use.
# Saves ~300-400 MB VRAM on the CUDA context — critical on 12 GB cards.
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

# Lets PyTorch reuse freed memory blocks, preventing the
# "Tried to allocate X MiB" OOM when plenty of VRAM is free but fragmented.
# garbage_collection_threshold:0.8 — proactively reclaims cached VRAM when
# usage exceeds 80%, reducing peak fragmentation on tight cards.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)


# ===== GPU TIER DETECTION =====
def _detect_gpu_tier() -> str:
    """
    Classify GPU by dedicated VRAM.

    Tiers:
      TIGHT       <= 12 GB  — RTX 3060 12GB, RTX 4060 8GB
      COMFORTABLE  13-20 GB — RTX 4070 Ti 16GB, RTX 3070 Ti
      ABUNDANT    > 20 GB   — RTX 3090 24GB, 4090 24GB, 5090 32GB

    Falls back to TIGHT (most conservative) when CUDA is unavailable.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return "TIGHT"
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_gb <= 12:
            return "TIGHT"
        elif total_gb <= 20:
            return "COMFORTABLE"
        return "ABUNDANT"
    except Exception:
        return "TIGHT"


GPU_TIER = _detect_gpu_tier()

# ===== MULTI-GPU SUPPORT =====
_active_device = 0

def get_gpu_count():
    """Get number of available GPUs."""
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0

def get_active_device():
    """Get the currently active CUDA device index."""
    return _active_device

def set_active_device(idx):
    """Set the active CUDA device by index."""
    global _active_device
    with _config_lock:
        count = get_gpu_count()
        if 0 <= idx < count:
            _active_device = idx
            return True
        return False


# ===== MODE CONFIGS =====
@dataclass
class ModeConfig:
    max_tokens: int
    temperature: float
    context_window: int  # sliding window for conversation history
    enable_thinking: bool = True  # whether model generates <think> blocks


# Token / context limits per GPU tier.
# TIGHT keeps things small to leave room for KV cache in <=12 GB VRAM.
# With enable_thinking=True, the model generates a <think> block PLUS the
# response — so effective output can be 2-3x the visible response length.
# TIGHT limits account for this overhead.
_TIER_LIMITS = {
    "TIGHT":       {"mt": 2048,     "cw": 6},
    "COMFORTABLE": {"mt": 3072,     "cw": 10},
    "ABUNDANT":    {"mt": 4096,     "cw": 15},
}

_L = _TIER_LIMITS[GPU_TIER]

MODES = {
    "ASSISTANT": ModeConfig(
        max_tokens=_L["mt"],
        temperature=0.7,
        context_window=_L["cw"],
        enable_thinking=True,
    ),
    "CODE": ModeConfig(
        max_tokens=min(_L["mt"] * 2, 8192),
        temperature=0.2,
        context_window=_L["cw"],
        enable_thinking=True,
    ),
}


# ===== CONTEXT PROFILES =====
# Token budget profiles — STANDARD preserves current behavior, HIGH expands limits.
# Independent from ModeConfig: modes define behavior, profiles define budgets.
# GPU_TIER maps to default profile: TIGHT/COMFORTABLE → STANDARD, ABUNDANT → HIGH.
@dataclass
class ContextProfile:
    """Token budget profile — controls how much context the model ingests/outputs."""
    max_total_input_tokens: int  # HARD CAP on everything sent to model
    max_output_tokens: int       # max generation length
    max_history_tokens: int      # soft cap for history alone
    knowledge_token_budget: int  # passed to KnowledgeManager.render_for_prompt()
    workspace_token_budget: int  # passed to KnowledgeManager.get_workspace_summary()
    session_map_token_budget: int  # passed to SessionMap.render()
    tool_output_limit: int       # unified per-tool truncation limit (chars)
    read_chunk_size: int         # chars per chunk for run_read_file


CONTEXT_PROFILES = {
    "STANDARD": ContextProfile(
        max_total_input_tokens=10000,
        max_output_tokens=1536,
        max_history_tokens=6000,
        knowledge_token_budget=550,
        workspace_token_budget=300,
        session_map_token_budget=250,
        tool_output_limit=5000,
        read_chunk_size=4000,
    ),
    "HIGH": ContextProfile(
        max_total_input_tokens=14000,
        max_output_tokens=2048,
        max_history_tokens=10000,
        knowledge_token_budget=750,
        workspace_token_budget=400,
        session_map_token_budget=400,
        tool_output_limit=6000,
        read_chunk_size=6000,
    ),
}

# Map GPU tier to default context profile
_DEFAULT_PROFILES = {"TIGHT": "STANDARD", "COMFORTABLE": "STANDARD", "ABUNDANT": "HIGH"}
_active_profile = _DEFAULT_PROFILES.get(GPU_TIER, "STANDARD")


def get_context_profile():
    """Get the currently active context profile."""
    return CONTEXT_PROFILES[_active_profile]


def set_context_profile(name):
    """Switch context profile. Returns True on success."""
    global _active_profile
    with _config_lock:
        if name in CONTEXT_PROFILES:
            _active_profile = name
            return True
        return False


def get_context_profile_name():
    """Get the name of the currently active context profile."""
    return _active_profile


# ===== MODEL-SPECIFIC SAMPLING DEFAULTS =====
# Google's recommended sampling for Gemma 4 models
GEMMA4_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
}

# ===== SAFETY =====
DANGEROUS_PATTERNS = [
    "rm -rf /",
    "mkfs",
    "dd if=/dev/zero",
    "> /dev/sda",
    ":(){ :|:& };:",
]

# Expanded safety patterns for general-purpose ASSISTANT mode
ASSISTANT_DANGEROUS_PATTERNS = DANGEROUS_PATTERNS + [
    "rm -rf ~",
    "rm -rf *",
    "format c:",
    "format d:",
    "del /s /q c:\\",
    "del /s /q d:\\",
    "remove-item -recurse c:\\",
    "remove-item -recurse d:\\",
    "reg delete hklm",
    "reg delete hkcu",
    "shutdown",
    "restart-computer",
    "stop-computer",
    "> /dev/null 2>&1 &",
    "chmod -r 000",
    "chown -r nobody",
    "iptables -f",
    "passwd root",
]

# ===== PHASES (used by knowledge.py) =====
PHASES = ["recon", "enum", "exploit", "privesc", "post"]
