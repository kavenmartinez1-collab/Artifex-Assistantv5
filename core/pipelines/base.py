"""
Artifex Assistant V5 — Base pipeline interface.
All model pipelines (text, image, 3D, audio) implement this contract.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PipelineResult:
    """Result from running a pipeline."""
    success: bool
    output_type: str          # "text", "image", "mesh", "audio"
    content: Any = None       # The actual output (text string, PIL Image, file path, etc.)
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None
    # Which engine served the request — "transformers", "ollama", or "" if N/A
    # (e.g. pure diffusers image gen doesn't flow through a text engine).
    # API handlers surface this to callers as `x_backend` so clients can
    # tell which runtime actually handled their request.
    backend: str = ""


class BasePipeline(ABC):
    """Interface for all model pipelines.

    Each pipeline handles a specific type of AI model:
    - TextGenerationPipeline: LLM chat/completion
    - ImageGenerationPipeline: Text-to-image (SD, SDXL, FLUX)
    - Shape3DPipeline: Text-to-3D (ShapE)
    - VisionPipeline: Image understanding
    - AudioPipeline: TTS/STT
    """

    # Subclasses must set these
    pipeline_type: str = ""           # e.g. "text-generation", "text-to-image"
    display_name: str = ""            # e.g. "Text Generation"
    output_type: str = "text"         # "text", "image", "mesh", "audio"

    @abstractmethod
    def load(self, model_path: str, status_callback=None, **kwargs):
        """Load a model for this pipeline.

        Args:
            model_path: Local path to model directory, or HF repo ID
            status_callback: Optional callback(str) for progress updates
            **kwargs: Pipeline-specific loading options (quantization, device, etc.)
        """

    @abstractmethod
    def unload(self, status_callback=None):
        """Unload the model and free resources."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if a model is loaded and ready."""

    @abstractmethod
    def run(self, **kwargs) -> PipelineResult:
        """Run inference.

        Args vary by pipeline type:
        - Text: messages=[], max_tokens=..., temperature=..., on_token=...
        - Image: prompt="...", negative_prompt="...", width=..., height=..., steps=...
        - 3D: prompt="...", num_steps=..., output_path="..."
        - Audio: text="..." or audio_path="..."

        Returns:
            PipelineResult with the output
        """

    @abstractmethod
    def get_vram_estimate(self, model_path: str) -> float:
        """Estimate VRAM required to load this model (in GB)."""

    def get_capabilities(self) -> dict:
        """Return a dict describing this pipeline's capabilities.

        Override to provide pipeline-specific info (supported formats,
        max resolution, etc.)
        """
        return {
            "pipeline_type": self.pipeline_type,
            "display_name": self.display_name,
            "output_type": self.output_type,
        }
