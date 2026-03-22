"""
Artifex Assistant V5 — Image Generation Pipeline.
Text-to-image using diffusers (Stable Diffusion, SDXL, FLUX, etc.)
"""

import gc
import os

import torch

from core.pipelines.base import BasePipeline, PipelineResult


class ImageGenerationPipeline(BasePipeline):
    """Text-to-image generation using HuggingFace diffusers."""

    pipeline_type = "text-to-image"
    display_name = "Image Generation"
    output_type = "image"

    def __init__(self):
        self.pipe = None
        self._model_path = None

    def load(self, model_path: str, status_callback=None, **kwargs):
        """Load a diffusion model for image generation.

        Automatically detects model type (SD 1.5, SDXL, FLUX) and loads
        with appropriate settings for the available VRAM.

        Args:
            model_path: Local path or HuggingFace repo ID
            status_callback: Progress callback
            **kwargs: dtype (default fp16), enable_cpu_offload (default auto)
        """
        try:
            from diffusers import AutoPipelineForText2Image
        except ImportError:
            raise ImportError(
                "diffusers is required for image generation.\n"
                "Install with: pip install diffusers>=0.30.0"
            )

        if status_callback:
            status_callback(f"Loading image model: {os.path.basename(model_path)}...")

        dtype = kwargs.get("dtype", torch.float16)
        enable_cpu_offload = kwargs.get("enable_cpu_offload", "auto")

        # Detect available VRAM
        gpu_gb = 0
        if torch.cuda.is_available():
            gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        try:
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                model_path,
                torch_dtype=dtype,
                use_safetensors=True,
            )
        except Exception as e:
            # Try without safetensors flag
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                model_path,
                torch_dtype=dtype,
            )

        # VRAM management: choose offload strategy based on GPU
        if enable_cpu_offload == "auto":
            if gpu_gb < 8:
                # Very tight: sequential CPU offload (slowest but fits)
                self.pipe.enable_sequential_cpu_offload()
                if status_callback:
                    status_callback("Using sequential CPU offload (low VRAM)")
            elif gpu_gb < 16:
                # Moderate: model CPU offload (balanced)
                self.pipe.enable_model_cpu_offload()
                if status_callback:
                    status_callback("Using model CPU offload (moderate VRAM)")
            else:
                # Abundant: keep everything on GPU
                self.pipe = self.pipe.to("cuda")
                if status_callback:
                    status_callback("Full GPU mode (abundant VRAM)")
        elif enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to("cuda")

        # Enable memory-efficient attention if available
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers not available, use default attention

        self._model_path = model_path

        if status_callback:
            status_callback(f"Image model loaded: {os.path.basename(model_path)}")

    def unload(self, status_callback=None):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self._model_path = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if status_callback:
                status_callback("Image model unloaded.")

    def is_loaded(self) -> bool:
        return self.pipe is not None

    def run(self, **kwargs) -> PipelineResult:
        """Generate an image from text.

        Args:
            prompt: Text description of the image
            negative_prompt: What to avoid (optional)
            width: Image width (default 512)
            height: Image height (default 512)
            num_steps: Inference steps (default 30)
            guidance_scale: CFG scale (default 7.5)
            seed: Random seed (optional, -1 for random)
            output_path: Save path (optional)

        Returns:
            PipelineResult with PIL Image as content
        """
        if not self.is_loaded():
            return PipelineResult(
                success=False, output_type="image",
                error="No model loaded."
            )

        prompt = kwargs.get("prompt", "")
        negative_prompt = kwargs.get("negative_prompt", "")
        width = kwargs.get("width", 512)
        height = kwargs.get("height", 512)
        num_steps = kwargs.get("num_steps", 30)
        guidance_scale = kwargs.get("guidance_scale", 7.5)
        seed = kwargs.get("seed", -1)
        output_path = kwargs.get("output_path", None)

        if not prompt:
            return PipelineResult(
                success=False, output_type="image",
                error="Prompt is required."
            )

        try:
            generator = None
            if seed >= 0:
                generator = torch.Generator(device="cpu").manual_seed(seed)

            gen_kwargs = {
                "prompt": prompt,
                "num_inference_steps": num_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
            }
            if negative_prompt:
                gen_kwargs["negative_prompt"] = negative_prompt
            if generator:
                gen_kwargs["generator"] = generator

            result = self.pipe(**gen_kwargs)
            image = result.images[0]

            # Save if path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                image.save(output_path)

            return PipelineResult(
                success=True,
                output_type="image",
                content=image,
                metadata={
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "steps": num_steps,
                    "seed": seed,
                    "saved_to": output_path,
                },
            )

        except Exception as e:
            return PipelineResult(
                success=False, output_type="image",
                error=str(e),
            )

    def get_vram_estimate(self, model_path: str) -> float:
        """Estimate VRAM for a diffusion model in fp16."""
        from core.model_registry import _estimate_model_size
        size_gb = _estimate_model_size(model_path)
        return round(size_gb * 1.5, 1)  # fp16 + activations

    def get_capabilities(self) -> dict:
        caps = super().get_capabilities()
        caps["supports_negative_prompt"] = True
        caps["supports_resolution"] = True
        caps["supports_seed"] = True
        caps["default_resolution"] = "512x512"
        return caps
