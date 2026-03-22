"""
Artifex Assistant V5 — Image Editing Pipeline.
Image-to-image, inpainting, and upscaling using diffusers.
"""

import gc
import os

import torch

from core.pipelines.base import BasePipeline, PipelineResult


class ImageEditPipeline(BasePipeline):
    """Image editing: img2img, inpainting, and upscaling."""

    pipeline_type = "image-to-image"
    display_name = "Image Editing (img2img / Inpaint / Upscale)"
    output_type = "image"

    def __init__(self):
        self.pipe = None
        self._model_path = None
        self._mode = None  # "img2img", "inpaint", "upscale"

    def load(self, model_path: str, status_callback=None, **kwargs):
        """Load an image editing model.

        Args:
            model_path: Local path or HuggingFace repo ID
            status_callback: Progress callback
            mode: "img2img", "inpaint", or "upscale" (default: img2img)
        """
        try:
            import diffusers
        except ImportError:
            raise ImportError(
                "diffusers is required for image editing.\n"
                "Install with: pip install diffusers>=0.30.0"
            )

        mode = kwargs.get("mode", "img2img")

        if status_callback:
            status_callback(f"Loading image edit model ({mode}): {os.path.basename(model_path)}...")

        dtype = kwargs.get("dtype", torch.float16)

        # VRAM management
        gpu_gb = 0
        if torch.cuda.is_available():
            gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        if mode == "upscale":
            from diffusers import StableDiffusionUpscalePipeline
            self.pipe = StableDiffusionUpscalePipeline.from_pretrained(
                model_path or "stabilityai/stable-diffusion-x4-upscaler",
                torch_dtype=dtype,
            )
        elif mode == "inpaint":
            from diffusers import AutoPipelineForInpainting
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                model_path,
                torch_dtype=dtype,
            )
        else:
            from diffusers import AutoPipelineForImage2Image
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                model_path,
                torch_dtype=dtype,
            )

        # VRAM-aware placement
        if gpu_gb < 8:
            self.pipe.enable_model_cpu_offload()
            if status_callback:
                status_callback("Using CPU offload (low VRAM)")
        elif gpu_gb < 12:
            self.pipe.enable_model_cpu_offload()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        else:
            self.pipe = self.pipe.to("cuda")

        self._model_path = model_path
        self._mode = mode

        if status_callback:
            status_callback(f"Image edit model loaded ({mode}): {os.path.basename(model_path)}")

    def unload(self, status_callback=None):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self._model_path = None
            self._mode = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if status_callback:
                status_callback("Image edit model unloaded.")

    def is_loaded(self) -> bool:
        return self.pipe is not None

    def run(self, **kwargs) -> PipelineResult:
        """Run image editing.

        Args:
            image_path: Source image path (required)
            prompt: Text prompt for guidance
            negative_prompt: Negative prompt
            mask_path: Mask image path (inpaint mode)
            strength: Denoising strength 0.0-1.0 (img2img, default 0.75)
            num_steps: Inference steps (default 30)
            guidance_scale: CFG scale (default 7.5)
            output_path: Where to save result

        Returns:
            PipelineResult with output image path
        """
        if not self.is_loaded():
            return PipelineResult(
                success=False, output_type="image",
                error="No model loaded."
            )

        image_path = kwargs.get("image_path", "")
        prompt = kwargs.get("prompt", "")
        negative_prompt = kwargs.get("negative_prompt", "")
        mask_path = kwargs.get("mask_path", "")
        strength = kwargs.get("strength", 0.75)
        num_steps = kwargs.get("num_steps", 30)
        guidance_scale = kwargs.get("guidance_scale", 7.5)
        output_path = kwargs.get("output_path", None)

        if not image_path or not os.path.isfile(image_path):
            return PipelineResult(
                success=False, output_type="image",
                error=f"Image not found: {image_path}"
            )

        try:
            from PIL import Image
            source = Image.open(image_path).convert("RGB")

            gen_kwargs = {
                "prompt": prompt,
                "image": source,
                "num_inference_steps": num_steps,
                "guidance_scale": guidance_scale,
            }

            if negative_prompt:
                gen_kwargs["negative_prompt"] = negative_prompt

            if self._mode == "inpaint" and mask_path and os.path.isfile(mask_path):
                gen_kwargs["mask_image"] = Image.open(mask_path).convert("L")
            elif self._mode != "upscale":
                gen_kwargs["strength"] = strength

            result = self.pipe(**gen_kwargs)
            image = result.images[0]

            # Save output
            if output_path is None:
                os.makedirs("output", exist_ok=True)
                base = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join("output", f"{base}_{self._mode}.png")

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            image.save(output_path)

            return PipelineResult(
                success=True,
                output_type="image",
                content=output_path,
                metadata={
                    "mode": self._mode,
                    "prompt": prompt,
                    "strength": strength,
                    "steps": num_steps,
                },
            )

        except Exception as e:
            return PipelineResult(
                success=False, output_type="image",
                error=str(e),
            )

    def get_vram_estimate(self, model_path: str) -> float:
        from core.model_registry import _estimate_model_size
        size_gb = _estimate_model_size(model_path)
        return round(size_gb * 1.5 + 2.0, 1)

    def get_capabilities(self) -> dict:
        caps = super().get_capabilities()
        caps["modes"] = ["img2img", "inpaint", "upscale"]
        caps["input_types"] = ["image + text"]
        caps["supported_formats"] = ["jpg", "png", "webp"]
        return caps
