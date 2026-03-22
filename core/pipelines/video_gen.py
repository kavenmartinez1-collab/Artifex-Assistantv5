"""
Artifex Assistant V5 — Video Generation Pipeline.
Text-to-video using diffusers. Requires ABUNDANT GPU tier (>20 GB VRAM).
"""

import gc
import os

import torch

from core.pipelines.base import BasePipeline, PipelineResult
from core.config import GPU_TIER


class VideoGenerationPipeline(BasePipeline):
    """Text-to-video generation. ABUNDANT tier only."""

    pipeline_type = "text-to-video"
    display_name = "Video Generation"
    output_type = "video"

    DEFAULT_MODEL = "damo-vilab/text-to-video-ms-1.7b"

    def __init__(self):
        self.pipe = None
        self._model_path = None

    def load(self, model_path: str, status_callback=None, **kwargs):
        """Load a text-to-video model.

        Args:
            model_path: Local path or HuggingFace repo ID
            status_callback: Progress callback

        Raises:
            RuntimeError: If GPU tier is not ABUNDANT
        """
        try:
            from diffusers import DiffusionPipeline
            from diffusers.utils import export_to_video
        except ImportError:
            raise ImportError(
                "diffusers is required for video generation.\n"
                "Install with: pip install diffusers>=0.30.0"
            )

        # Check GPU tier
        gpu_gb = 0
        if torch.cuda.is_available():
            gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        if gpu_gb < 16:
            raise RuntimeError(
                f"Video generation requires >16 GB VRAM. "
                f"Detected: {gpu_gb:.1f} GB ({GPU_TIER} tier). "
                f"Minimum recommended: ABUNDANT tier (>20 GB)."
            )

        if not model_path:
            model_path = self.DEFAULT_MODEL

        if status_callback:
            status_callback(f"Loading video model: {os.path.basename(model_path)}...")

        dtype = kwargs.get("dtype", torch.float16)

        self.pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
        )

        # Always use CPU offload — video models are VRAM-heavy
        self.pipe.enable_model_cpu_offload()

        try:
            self.pipe.enable_vae_slicing()
        except Exception:
            pass

        self._model_path = model_path

        if status_callback:
            status_callback(f"Video model loaded: {os.path.basename(model_path)}")

    def unload(self, status_callback=None):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self._model_path = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if status_callback:
                status_callback("Video model unloaded.")

    def is_loaded(self) -> bool:
        return self.pipe is not None

    def run(self, **kwargs) -> PipelineResult:
        """Generate a short video from text.

        Args:
            prompt: Text description of the video
            negative_prompt: What to avoid
            num_frames: Number of frames (default 16, max 32)
            fps: Frames per second for output (default 8)
            num_steps: Inference steps (default 25)
            output_path: Where to save the .mp4 file

        Returns:
            PipelineResult with video file path
        """
        if not self.is_loaded():
            return PipelineResult(
                success=False, output_type="video",
                error="No model loaded."
            )

        prompt = kwargs.get("prompt", "")
        negative_prompt = kwargs.get("negative_prompt", "")
        num_frames = min(kwargs.get("num_frames", 16), 32)
        fps = kwargs.get("fps", 8)
        num_steps = kwargs.get("num_steps", 25)
        output_path = kwargs.get("output_path", None)

        if not prompt:
            return PipelineResult(
                success=False, output_type="video",
                error="Prompt is required."
            )

        try:
            gen_kwargs = {
                "prompt": prompt,
                "num_frames": num_frames,
                "num_inference_steps": num_steps,
            }
            if negative_prompt:
                gen_kwargs["negative_prompt"] = negative_prompt

            result = self.pipe(**gen_kwargs)
            frames = result.frames[0]  # list of PIL Images

            # Save
            if output_path is None:
                os.makedirs("output", exist_ok=True)
                safe = "".join(c if c.isalnum() or c in " -_" else "" for c in prompt)[:30]
                output_path = os.path.join("output", f"video_{safe}.mp4")

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            from diffusers.utils import export_to_video
            export_to_video(frames, output_path, fps=fps)

            return PipelineResult(
                success=True,
                output_type="video",
                content=output_path,
                metadata={
                    "prompt": prompt,
                    "num_frames": num_frames,
                    "fps": fps,
                    "steps": num_steps,
                },
            )

        except Exception as e:
            return PipelineResult(
                success=False, output_type="video",
                error=str(e),
            )

    def get_vram_estimate(self, model_path: str) -> float:
        """Video models need significant VRAM: 8-16 GB."""
        return 12.0

    def get_capabilities(self) -> dict:
        caps = super().get_capabilities()
        caps["default_model"] = self.DEFAULT_MODEL
        caps["max_frames"] = 32
        caps["min_vram_gb"] = 16
        caps["output_formats"] = ["mp4"]
        return caps
