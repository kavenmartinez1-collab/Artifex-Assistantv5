"""
Artifex Assistant V5 — 3D Generation Pipeline (ShapE).
Text-to-3D mesh generation using OpenAI's ShapE model via diffusers.
"""

import gc
import os

import torch

from core.pipelines.base import BasePipeline, PipelineResult


class Shape3DPipeline(BasePipeline):
    """Text-to-3D generation using ShapE (diffusers)."""

    pipeline_type = "shap-e"
    display_name = "3D Generation (ShapE)"
    output_type = "mesh"

    def __init__(self):
        self.pipe = None
        self._model_path = None

    def load(self, model_path: str, status_callback=None, **kwargs):
        """Load the ShapE model for 3D generation.

        Args:
            model_path: Local path or HF repo ID (default: openai/shap-e)
            status_callback: Progress callback
        """
        try:
            from diffusers import ShapEPipeline
        except ImportError:
            raise ImportError(
                "diffusers is required for 3D generation.\n"
                "Install with: pip install diffusers>=0.30.0"
            )

        # Check mesh export deps — trimesh needed for PLY/OBJ export
        try:
            import trimesh  # noqa: F401
        except ImportError:
            raise ImportError(
                "trimesh is required for 3D mesh export (PLY/OBJ).\n"
                "Install with: pip install trimesh"
            )

        if not model_path:
            model_path = "openai/shap-e"

        if status_callback:
            status_callback(f"Loading ShapE model: {model_path}...")

        dtype = kwargs.get("dtype", torch.float16)

        self.pipe = ShapEPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
        )

        # VRAM management
        gpu_gb = 0
        if torch.cuda.is_available():
            gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        if gpu_gb < 12:
            self.pipe.enable_model_cpu_offload()
            if status_callback:
                status_callback("Using CPU offload for ShapE (low VRAM)")
        else:
            self.pipe = self.pipe.to("cuda")

        self._model_path = model_path

        if status_callback:
            status_callback("ShapE model loaded.")

    def unload(self, status_callback=None):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self._model_path = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if status_callback:
                status_callback("ShapE model unloaded.")

    def is_loaded(self) -> bool:
        return self.pipe is not None

    def run(self, **kwargs) -> PipelineResult:
        """Generate a 3D mesh from text.

        Args:
            prompt: Text description of the 3D object
            num_steps: Inference steps (default 64)
            frame_size: Resolution of the implicit representation (default 64)
            output_path: Path to save .ply file (optional)
            output_format: "ply" or "obj" (default "ply")

        Returns:
            PipelineResult with mesh file path as content
        """
        from core.pipelines.schemas import Shape3DInput
        from pydantic import ValidationError
        try:
            params = Shape3DInput(**kwargs)
        except ValidationError as e:
            return PipelineResult(success=False, output_type="mesh", error=f"Invalid input: {e}")

        if not self.is_loaded():
            return PipelineResult(
                success=False, output_type="mesh",
                error="No model loaded."
            )

        if not params.prompt:
            return PipelineResult(
                success=False, output_type="mesh",
                error="Prompt is required."
            )

        output_path = params.output_path

        try:
            images = self.pipe(
                params.prompt,
                guidance_scale=15.0,
                num_inference_steps=params.num_steps,
                frame_size=params.frame_size,
            ).images

            # Export mesh
            if output_path is None:
                os.makedirs("output", exist_ok=True)
                safe_name = "".join(c if c.isalnum() or c in " -_" else "" for c in params.prompt)[:50]
                output_path = os.path.join("output", f"{safe_name}.{params.output_format}")

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            # Convert ShapE output to mesh
            try:
                from diffusers.utils import export_to_ply, export_to_obj
                if params.output_format == "obj":
                    export_to_obj(images[0], output_path)
                else:
                    export_to_ply(images[0], output_path)
            except ImportError:
                # Fallback: save raw output
                torch.save(images[0], output_path + ".pt")
                output_path = output_path + ".pt"

            return PipelineResult(
                success=True,
                output_type="mesh",
                content=output_path,
                metadata={
                    "prompt": params.prompt,
                    "steps": params.num_steps,
                    "format": params.output_format,
                },
            )

        except Exception as e:
            return PipelineResult(
                success=False, output_type="mesh",
                error=str(e),
            )

    def get_vram_estimate(self, model_path: str) -> float:
        """ShapE requires ~4-6 GB VRAM in fp16."""
        return 6.0

    def get_capabilities(self) -> dict:
        caps = super().get_capabilities()
        caps["output_formats"] = ["ply", "obj"]
        caps["default_model"] = "openai/shap-e"
        return caps
