"""
Artifex Assistant V5 — Pydantic input schemas for pipeline validation.
Each schema mirrors the kwargs accepted by its pipeline's run() method.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Text Generation ────────────────────────────────────────────────────

class TextGenerationInput(BaseModel):
    messages: list = Field(default_factory=list)
    max_tokens: int = 1024
    temperature: float = 0.7
    on_token: Any = None
    on_complete: Any = None
    enable_thinking: bool = True


# ── Image Generation ───────────────────────────────────────────────────

class ImageGenerationInput(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    num_steps: int = 30
    guidance_scale: float = 7.5
    seed: int = -1
    output_path: Optional[str] = None


# ── Vision ─────────────────────────────────────────────────────────────

class VisionInput(BaseModel):
    image_path: str = ""
    prompt: str = "Describe what is happening in detail."
    max_tokens: int = 512
    # Video-specific overrides (blank = auto-plan based on video length + RAM)
    nframes: Optional[int] = None
    fps: Optional[float] = None
    max_pixels: int = 360 * 420


# ── 3D Generation (ShapE) ─────────────────────────────────────────────

class Shape3DInput(BaseModel):
    prompt: str = ""
    num_steps: int = 64
    frame_size: int = 64
    output_path: Optional[str] = None
    output_format: str = "ply"


# ── Audio (TTS / STT) ─────────────────────────────────────────────────

class AudioInput(BaseModel):
    # TTS fields
    text: str = ""
    output_path: Optional[str] = None
    # STT fields
    audio_path: str = ""


# ── Embedding ──────────────────────────────────────────────────────────

class EmbeddingInput(BaseModel):
    texts: list = Field(default_factory=list)
    query: str = ""
    corpus_embeddings: Any = None
    top_k: int = 5


# ── Music Generation ──────────────────────────────────────────────────

class MusicInput(BaseModel):
    prompt: str = ""
    duration_seconds: int = 10
    output_path: Optional[str] = None


# ── Video Generation ──────────────────────────────────────────────────

class VideoGenerationInput(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    num_frames: int = 16
    fps: int = 8
    num_steps: int = 25
    output_path: Optional[str] = None


# ── Image Editing ──────────────────────────────────────────────────────

class ImageEditInput(BaseModel):
    image_path: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    mask_path: str = ""
    strength: float = 0.75
    num_steps: int = 30
    guidance_scale: float = 7.5
    output_path: Optional[str] = None


# ── Photo Restoration ──────────────────────────────────────────────────

class PhotoRestoreInput(BaseModel):
    image_path: str = ""
    prompt: str = ""
    # Stage 2 (AI restore)
    strength: float = 0.3          # low = repair, high = reinvent
    num_steps: int = 30
    guidance_scale: float = 7.5
    # Stage 1 (classical clean)
    denoise: float = 7.0           # non-local means h, 0 disables
    smooth_radius: int = 4         # bilateral circle-of-interest radius
    auto_levels: bool = True
    # Stage 3 (upscale)
    upscale: int = 2               # 2 or 4
    tile: int = 512
    output_path: Optional[str] = None


# ── Voice Assistant ────────────────────────────────────────────────────

class VoiceAssistantInput(BaseModel):
    # Routing flags
    stt_only: bool = False
    tts_only: bool = False

    # Conversation mode
    text: Optional[str] = None
    audio_data: Any = None
    on_token: Any = None
    play_audio: bool = True
    progress_callback: Any = None
    max_tokens: int = 512
    temperature: float = 0.7

    # TTS-only mode
    tts_text: str = ""

    model_config = {"arbitrary_types_allowed": True}
