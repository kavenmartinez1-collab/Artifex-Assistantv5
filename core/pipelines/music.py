"""
Artifex Assistant V5 — Music Generation Pipeline.
Text-to-music using Meta's MusicGen via transformers.
"""

import gc
import os

import torch

from core.pipelines.base import BasePipeline, PipelineResult


class MusicPipeline(BasePipeline):
    """Text-to-music generation using MusicGen."""

    pipeline_type = "text-to-music"
    display_name = "Music Generation"
    output_type = "audio"

    # Model variants by GPU tier
    MODELS = {
        "small": "facebook/musicgen-small",    # ~2 GB
        "medium": "facebook/musicgen-medium",   # ~4 GB
        "large": "facebook/musicgen-large",     # ~8 GB
    }

    def __init__(self):
        self.model = None
        self.processor = None
        self._model_path = None
        self._sampling_rate = 32000

    def load(self, model_path: str, status_callback=None, **kwargs):
        """Load a MusicGen model.

        Args:
            model_path: Local path or HuggingFace repo ID
                        (default: auto-selects by GPU tier)
            status_callback: Progress callback
        """
        from transformers import MusicgenForConditionalGeneration, AutoProcessor

        if not model_path:
            model_path = self._auto_select_model()

        if status_callback:
            status_callback(f"Loading music model: {os.path.basename(model_path)}...")

        dtype = kwargs.get("dtype", torch.float16)

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
        )

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        # Get sampling rate from model config
        if hasattr(self.model.config, "audio_encoder"):
            sr = getattr(self.model.config.audio_encoder, "sampling_rate", None)
            if sr:
                self._sampling_rate = sr

        self._model_path = model_path

        if status_callback:
            status_callback(f"Music model loaded: {os.path.basename(model_path)}")

    def _auto_select_model(self):
        """Select model variant based on available VRAM."""
        gpu_gb = 0
        if torch.cuda.is_available():
            gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        if gpu_gb > 20:
            return self.MODELS["large"]
        elif gpu_gb > 12:
            return self.MODELS["medium"]
        return self.MODELS["small"]

    def unload(self, status_callback=None):
        for attr in ("model", "processor"):
            if getattr(self, attr, None) is not None:
                delattr(self, attr)
                setattr(self, attr, None)
        self._model_path = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if status_callback:
            status_callback("Music model unloaded.")

    def is_loaded(self) -> bool:
        return self.model is not None

    def run(self, **kwargs) -> PipelineResult:
        """Generate music from a text description.

        Args:
            prompt: Text description of the music
            duration_seconds: Length of audio to generate (default 10, max 30)
            output_path: Where to save the .wav file

        Returns:
            PipelineResult with audio file path
        """
        if not self.is_loaded():
            return PipelineResult(
                success=False, output_type="audio",
                error="No model loaded."
            )

        prompt = kwargs.get("prompt", "")
        duration = min(kwargs.get("duration_seconds", 10), 30)
        output_path = kwargs.get("output_path", None)

        if not prompt:
            return PipelineResult(
                success=False, output_type="audio",
                error="Prompt is required."
            )

        try:
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )

            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Calculate max_new_tokens from duration
            tokens_per_second = self._sampling_rate / 320  # approximate
            max_new_tokens = int(duration * tokens_per_second)

            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
            )

            audio = audio_values[0, 0].cpu().numpy()

            # Save
            if output_path is None:
                os.makedirs("output", exist_ok=True)
                safe = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt)[:30]
                output_path = os.path.join("output", f"music_{safe}.wav")

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            import scipy.io.wavfile
            scipy.io.wavfile.write(output_path, self._sampling_rate, audio)

            return PipelineResult(
                success=True,
                output_type="audio",
                content=output_path,
                metadata={
                    "prompt": prompt,
                    "duration_seconds": duration,
                    "sampling_rate": self._sampling_rate,
                },
            )

        except Exception as e:
            return PipelineResult(
                success=False, output_type="audio",
                error=str(e),
            )

    def get_vram_estimate(self, model_path: str) -> float:
        """Estimate VRAM: small=2GB, medium=4GB, large=8GB."""
        name = os.path.basename(model_path or "").lower()
        if "large" in name:
            return 8.0
        elif "medium" in name:
            return 4.0
        return 2.0

    def get_capabilities(self) -> dict:
        caps = super().get_capabilities()
        caps["models"] = self.MODELS
        caps["max_duration_seconds"] = 30
        caps["output_formats"] = ["wav"]
        return caps
