"""
Artifex Assistant V5 — Text Generation Pipeline.
Wraps the existing engine abstraction (Transformers/Ollama) for chat and completion.
"""

from core.pipelines.base import BasePipeline, PipelineResult
from core.engine_factory import create_engine
from core.model_registry import _estimate_model_size


class TextGenerationPipeline(BasePipeline):
    """Text generation (chat/completion) using Transformers or Ollama backend."""

    pipeline_type = "text-generation"
    display_name = "Text Generation"
    output_type = "text"

    def __init__(self):
        self.engine = None

    def load(self, model_path: str, status_callback=None, **kwargs):
        """Load a text generation model.

        Uses the engine factory to create the right backend (Transformers/Ollama)
        based on the active backend setting.
        """
        backend = kwargs.get("backend", None)
        if backend:
            from core.config import set_active_backend
            set_active_backend(backend)

        model_name = kwargs.get("model_name", None)
        if model_name:
            from core.config import set_active_model
            set_active_model(model_name)

        self.engine = create_engine()
        self.engine.load(status_callback=status_callback)

    def unload(self, status_callback=None):
        if self.engine is not None:
            self.engine.unload(status_callback=status_callback)
            self.engine = None

    def is_loaded(self) -> bool:
        return self.engine is not None and self.engine.is_loaded()

    def run(self, **kwargs) -> PipelineResult:
        """Run text generation.

        Args:
            messages: list of {role, content} dicts
            max_tokens: max new tokens (default 1024)
            temperature: sampling temperature (default 0.7)
            on_token: streaming callback (optional)
            on_complete: completion callback (optional)
            enable_thinking: enable think blocks (default True)

        Returns:
            PipelineResult with text content
        """
        from core.pipelines.schemas import TextGenerationInput
        from pydantic import ValidationError
        try:
            params = TextGenerationInput(**kwargs)
        except ValidationError as e:
            return PipelineResult(success=False, output_type="text", error=f"Invalid input: {e}")

        if not self.is_loaded():
            return PipelineResult(
                success=False, output_type="text",
                error="No model loaded. Call load() first."
            )

        try:
            response = self.engine.generate_streaming(
                messages=params.messages,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                on_token=params.on_token,
                on_complete=params.on_complete,
                enable_thinking=params.enable_thinking,
            )
            backend = "transformers" if "Transformers" in type(self.engine).__name__ else "ollama"
            return PipelineResult(
                success=True,
                output_type="text",
                content=response,
                backend=backend,
            )
        except Exception as e:
            return PipelineResult(
                success=False, output_type="text",
                error=str(e),
            )

    def get_vram_estimate(self, model_path: str) -> float:
        """Estimate VRAM for a text generation model (4-bit quantized)."""
        size_gb = _estimate_model_size(model_path)
        return round(size_gb * 0.6 + 2.0, 1)  # 4-bit + KV cache

    def get_capabilities(self) -> dict:
        caps = super().get_capabilities()
        caps["supports_streaming"] = True
        caps["supports_thinking"] = True
        caps["backends"] = ["transformers", "ollama"]
        return caps
