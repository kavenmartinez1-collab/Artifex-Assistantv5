"""
Artifex Assistant V5 — Centralized configuration.
Auto-detects GPU tier and scales token limits / context windows accordingly.
Supports runtime model/backend switching and context profile toggling.
"""

import json
import os
import platform
import time
import urllib.request
import urllib.error
from dataclasses import dataclass

# ===== PLATFORM =====
IS_WINDOWS = platform.system() == "Windows"
PLATFORM_STRING = f"{platform.system()} ({platform.machine()})"

# ===== PATHS =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_MODEL = os.path.join(BASE_DIR, "models", "qwen3.5-9b")
_SHARED_MODEL = os.path.join(os.path.dirname(BASE_DIR), "Artifex-lite", "models", "qwen3.5-9b")
_SHARED_MODEL_V3 = os.path.join(os.path.dirname(BASE_DIR), "artifex-lite-v3", "models", "qwen3.5-9b")
MODEL_PATH = (
    _LOCAL_MODEL if os.path.isdir(_LOCAL_MODEL)
    else _SHARED_MODEL_V3 if os.path.isdir(_SHARED_MODEL_V3)
    else _SHARED_MODEL
)
SESSION_DIR = os.path.join(BASE_DIR, "sessions")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
KNOWLEDGE_REFERENCE_DIR = os.path.join(KNOWLEDGE_DIR, "reference")

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

_active_model = None  # resolved lazily on first call

# ===== BACKEND SWITCHING =====
_active_backend = "transformers"  # "transformers" or "ollama"


def get_active_backend():
    """Get the currently active backend name."""
    return _active_backend


def set_active_backend(name):
    """Switch active backend. Returns True on success."""
    global _active_backend
    if name in ("transformers", "ollama"):
        _active_backend = name
        return True
    return False


# ===== OLLAMA CONFIGURATION =====
OLLAMA_MODELS = {}  # auto-discovered from running Ollama server
OLLAMA_NUM_GPU = "auto"  # "auto", 0, -1, or specific int


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


# ===== MODEL SELECTION (both backends) =====

def get_active_model_path():
    """Get the currently selected transformers model's directory path."""
    global _active_model
    if _active_model is None:
        _active_model = MODEL_PATH  # default
    return _active_model


def set_active_model(name):
    """Switch active model by registry name. Works for both backends."""
    global _active_model
    if _active_backend == "ollama":
        if name in OLLAMA_MODELS:
            _active_model = name
            return True
        # Also accept raw model names not in cache
        _active_model = name
        return True
    if name in MODELS:
        _active_model = MODELS[name]
        return True
    return False


def get_active_model_name():
    """Get display name of the currently active model."""
    if _active_backend == "ollama":
        return get_active_ollama_model()
    path = get_active_model_path()
    for name, p in MODELS.items():
        if p == path:
            return name
    return os.path.basename(path)


def get_active_ollama_model():
    """Get the active Ollama model name."""
    global _active_model
    if _active_model and isinstance(_active_model, str):
        # If it looks like an Ollama model name (not a filesystem path)
        if not os.sep in _active_model and not _active_model.startswith("/"):
            return _active_model
    # Default: first discovered model, or generic
    if OLLAMA_MODELS:
        return next(iter(OLLAMA_MODELS.keys()))
    return "qwen3.5:latest"


def get_model_names():
    """Get list of available model names for current backend."""
    if _active_backend == "ollama":
        if not OLLAMA_MODELS:
            refresh_ollama_models()
        return list(OLLAMA_MODELS.keys())
    return list(MODELS.keys())

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
    if name in CONTEXT_PROFILES:
        _active_profile = name
        return True
    return False


def get_context_profile_name():
    """Get the name of the currently active context profile."""
    return _active_profile


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
