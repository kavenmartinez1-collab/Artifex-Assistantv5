"""
Artifex Assistant V5 — Universal Model Registry.
Auto-detects model types from HuggingFace configs, scans local models,
and provides VRAM estimates for any model type.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional


# Pipeline types that Artifex Assistant V5 supports
PIPELINE_TYPES = {
    "text-generation": "Text Generation (Chat / Completion)",
    "text-to-image": "Image Generation (Stable Diffusion, SDXL, FLUX)",
    "image-to-image": "Image Editing (img2img / Inpaint / Upscale)",
    "shap-e": "3D Generation (ShapE)",
    "image-text-to-text": "Vision (Image Understanding)",
    "text-to-audio": "Audio Generation (TTS)",
    "text-to-music": "Music Generation (MusicGen)",
    "automatic-speech-recognition": "Speech Recognition (STT)",
    "text-to-video": "Video Generation",
    "feature-extraction": "Embedding / RAG",
}

# Map HuggingFace model_type to our pipeline type
_MODEL_TYPE_MAP = {
    # Text generation models
    "qwen2": "text-generation",
    "qwen2_5": "text-generation",
    "qwen3": "text-generation",
    "qwen3_5": "text-generation",
    "qwen3_5_text": "text-generation",
    "qwen3_6": "text-generation",
    "qwen3_6_text": "text-generation",
    "llama": "text-generation",
    "mistral": "text-generation",
    "gemma": "text-generation",
    "gemma2": "text-generation",
    "gemma3": "text-generation",
    # Gemma 4 — multimodal (uses AutoModelForMultimodalLM)
    "gemma3n": "image-text-to-text",
    "gemma3n_text": "image-text-to-text",
    "gemma3n_vision": "image-text-to-text",
    "gemma3n_audio": "image-text-to-text",
    "phi": "text-generation",
    "phi3": "text-generation",
    "starcoder2": "text-generation",
    "deepseek_v2": "text-generation",
    "deepseek_v3": "text-generation",
    "gpt2": "text-generation",
    "gpt_neo": "text-generation",
    "gpt_neox": "text-generation",
    "falcon": "text-generation",
    "mpt": "text-generation",
    "bloom": "text-generation",
    "opt": "text-generation",
    "cohere": "text-generation",
    "command-r": "text-generation",
    "internlm2": "text-generation",

    # Diffusion / image generation
    "stable-diffusion": "text-to-image",
    "stable-diffusion-xl": "text-to-image",
    "flux": "text-to-image",
    "kandinsky": "text-to-image",
    "pixart": "text-to-image",

    # 3D generation
    "shap-e": "shap-e",

    # Vision / multimodal
    "llava": "image-text-to-text",
    "llava_next": "image-text-to-text",
    "qwen2_vl": "image-text-to-text",
    "internvl": "image-text-to-text",

    # Audio
    "bark": "text-to-audio",
    "whisper": "automatic-speech-recognition",
    "wav2vec2": "automatic-speech-recognition",

    # Embedding
    "bert": "feature-extraction",
    "roberta": "feature-extraction",
    "sentence-transformers": "feature-extraction",

    # Music generation
    "musicgen": "text-to-music",
    "musicgen_melody": "text-to-music",

    # Video generation
    "text-to-video": "text-to-video",
    "cogvideox": "text-to-video",
}

# Backend mapping: which backend handles each pipeline type
PIPELINE_BACKEND_MAP = {
    "text-generation": "transformers",  # or "ollama"
    "text-to-image": "diffusers",
    "image-to-image": "diffusers",
    "shap-e": "diffusers",
    "image-text-to-text": "transformers",
    "text-to-audio": "transformers",
    "text-to-music": "transformers",
    "automatic-speech-recognition": "transformers",
    "text-to-video": "diffusers",
    "feature-extraction": "transformers",
}


@dataclass
class ModelInfo:
    """Information about a discovered model."""
    name: str
    path: str                              # local path or HF repo ID
    source: str = "local"                  # "local", "huggingface", "ollama"
    pipeline_type: str = "text-generation"  # from PIPELINE_TYPES keys
    size_gb: float = 0.0                   # estimated model size on disk
    backend: str = "transformers"          # "transformers", "ollama", "diffusers"
    quantization: str = "none"             # "none", "nf4", "int8", "fp16"
    vram_required_gb: float = 0.0          # estimated VRAM needed
    metadata: dict = field(default_factory=dict)


def detect_model_type(model_path: str) -> str:
    """Auto-detect pipeline type from a local model's config.json.

    Checks (in priority order):
      1. pipeline_tag field (most reliable, set by model authors)
      2. model_type field (architecture name)
      3. _class_name field (diffusers models)
      4. Filename heuristics

    Args:
        model_path: Path to the model directory

    Returns:
        Pipeline type string (key from PIPELINE_TYPES)
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        # Check for diffusers model_index.json
        model_index = os.path.join(model_path, "model_index.json")
        if os.path.isfile(model_index):
            return _detect_diffusers_type(model_index)
        return "text-generation"  # default

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "text-generation"

    # 1. Check pipeline_tag (most explicit)
    pipeline_tag = cfg.get("pipeline_tag", "")
    if pipeline_tag in PIPELINE_TYPES:
        return pipeline_tag

    # 2. Check model_type
    model_type = cfg.get("model_type", "").lower().replace("-", "_")
    if model_type in _MODEL_TYPE_MAP:
        return _MODEL_TYPE_MAP[model_type]

    # 3. Check architectures list
    architectures = cfg.get("architectures", [])
    for arch in architectures:
        arch_lower = arch.lower()
        if "causal" in arch_lower or "lm" in arch_lower:
            return "text-generation"
        if "seq2seq" in arch_lower or "conditional" in arch_lower:
            return "text-generation"
        if "imagetext" in arch_lower or "vision" in arch_lower:
            return "image-text-to-text"
        if "whisper" in arch_lower:
            return "automatic-speech-recognition"

    # 4. Check _class_name for diffusers
    class_name = cfg.get("_class_name", "")
    if "Pipeline" in class_name:
        if "Img2Img" in class_name or "Txt2Img" in class_name or "StableDiffusion" in class_name:
            return "text-to-image"
        if "ShapE" in class_name:
            return "shap-e"

    return "text-generation"


def _detect_diffusers_type(model_index_path: str) -> str:
    """Detect pipeline type from a diffusers model_index.json."""
    try:
        with open(model_index_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "text-to-image"

    class_name = cfg.get("_class_name", "")
    if "ShapE" in class_name:
        return "shap-e"
    if "StableDiffusion" in class_name or "SDXL" in class_name or "Flux" in class_name:
        return "text-to-image"
    if "Audio" in class_name or "Music" in class_name:
        return "text-to-audio"

    return "text-to-image"


def _estimate_model_size(model_path: str) -> float:
    """Estimate model size in GB by summing weight files."""
    total_bytes = 0
    weight_extensions = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}

    for root, _dirs, files in os.walk(model_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in weight_extensions:
                total_bytes += os.path.getsize(os.path.join(root, f))

    return total_bytes / (1024 ** 3)


def estimate_vram(model_info: ModelInfo, gpu_gb: float) -> dict:
    """Estimate VRAM requirements and recommend quantization.

    Args:
        model_info: ModelInfo for the model
        gpu_gb: Available GPU VRAM in GB

    Returns:
        Dict with keys: fits, recommended_quantization, estimated_vram_gb, notes
    """
    size_gb = model_info.size_gb or 1.0
    pipeline = model_info.pipeline_type

    if pipeline == "text-generation":
        # LLM VRAM estimation
        # fp16: ~2x model size on disk (weights + overhead)
        # 4-bit: ~0.6x model size
        # 8-bit: ~1.1x model size
        fp16_vram = size_gb * 2.0
        int8_vram = size_gb * 1.1
        nf4_vram = size_gb * 0.6

        if nf4_vram + 2.0 <= gpu_gb:  # +2 GB for KV cache
            return {
                "fits": True,
                "recommended_quantization": "nf4",
                "estimated_vram_gb": round(nf4_vram + 2.0, 1),
                "notes": "4-bit quantization recommended",
            }
        elif int8_vram + 2.0 <= gpu_gb:
            return {
                "fits": True,
                "recommended_quantization": "int8",
                "estimated_vram_gb": round(int8_vram + 2.0, 1),
                "notes": "8-bit quantization needed for this GPU",
            }
        elif fp16_vram + 2.0 <= gpu_gb:
            return {
                "fits": True,
                "recommended_quantization": "fp16",
                "estimated_vram_gb": round(fp16_vram + 2.0, 1),
                "notes": "Full precision fits",
            }
        else:
            return {
                "fits": False,
                "recommended_quantization": "nf4",
                "estimated_vram_gb": round(nf4_vram + 2.0, 1),
                "notes": f"Model needs ~{nf4_vram + 2.0:.0f} GB even quantized. "
                         f"Consider a smaller model or Ollama with CPU offload.",
            }

    elif pipeline in ("text-to-image", "shap-e"):
        # Diffusion models: typically run in fp16
        fp16_vram = size_gb * 1.5  # weights + intermediate activations
        if fp16_vram <= gpu_gb:
            return {
                "fits": True,
                "recommended_quantization": "fp16",
                "estimated_vram_gb": round(fp16_vram, 1),
                "notes": "fp16 recommended for diffusion models",
            }
        else:
            return {
                "fits": False,
                "recommended_quantization": "fp16",
                "estimated_vram_gb": round(fp16_vram, 1),
                "notes": f"Model needs ~{fp16_vram:.0f} GB. "
                         f"Try a smaller variant or enable CPU offload.",
            }

    else:
        # Generic estimate
        return {
            "fits": size_gb * 1.5 <= gpu_gb,
            "recommended_quantization": "fp16",
            "estimated_vram_gb": round(size_gb * 1.5, 1),
            "notes": "Generic estimate — actual usage may vary",
        }


def scan_local_models(models_dir: str) -> list:
    """Scan the models/ directory and classify each model.

    Args:
        models_dir: Path to the models directory

    Returns:
        List of ModelInfo objects
    """
    results = []

    if not os.path.isdir(models_dir):
        return results

    for name in sorted(os.listdir(models_dir)):
        full_path = os.path.join(models_dir, name)
        if not os.path.isdir(full_path):
            continue
        if name.endswith("-nf4-cached") or name.startswith("."):
            continue

        pipeline_type = detect_model_type(full_path)
        size_gb = _estimate_model_size(full_path)
        backend = PIPELINE_BACKEND_MAP.get(pipeline_type, "transformers")

        info = ModelInfo(
            name=name,
            path=full_path,
            source="local",
            pipeline_type=pipeline_type,
            size_gb=round(size_gb, 2),
            backend=backend,
        )
        results.append(info)

    return results


def scan_ollama_models() -> list:
    """Query Ollama for installed models and return ModelInfo list."""
    from core.config import OLLAMA_MODELS, refresh_ollama_models

    if not OLLAMA_MODELS:
        refresh_ollama_models()

    results = []
    for name, data in OLLAMA_MODELS.items():
        size_gb = data.get("size", 0) / (1024 ** 3) if data.get("size") else 0
        info = ModelInfo(
            name=name,
            path=data.get("name", name),
            source="ollama",
            pipeline_type="text-generation",
            size_gb=round(size_gb, 2),
            backend="ollama",
        )
        results.append(info)

    return results


def get_all_models(models_dir: Optional[str] = None) -> list:
    """Get all available models from all sources.

    Args:
        models_dir: Path to local models directory (auto-detected if None)

    Returns:
        List of ModelInfo objects from local + Ollama sources
    """
    if models_dir is None:
        from core.config import BASE_DIR
        models_dir = os.path.join(BASE_DIR, "models")

    models = scan_local_models(models_dir)

    try:
        ollama_models = scan_ollama_models()
        models.extend(ollama_models)
    except Exception:
        pass  # Ollama not available

    return models
