"""
Artifex Assistant V5 — Vision Pipeline.
Image understanding using multimodal models (LLaVA, Qwen-VL, etc.)
"""

import gc
import os

import torch

from core.pipelines.base import BasePipeline, PipelineResult


class VisionPipeline(BasePipeline):
    """Image understanding using multimodal vision-language models."""

    pipeline_type = "image-text-to-text"
    display_name = "Vision (Image Understanding)"
    output_type = "text"

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._model_path = None

    def load(self, model_path: str, status_callback=None, **kwargs):
        """Load a vision-language model.

        Args:
            model_path: Local path or HuggingFace repo ID
            status_callback: Progress callback
        """
        from transformers import AutoTokenizer, AutoProcessor
        from core.engine_transformers import _get_auto_model_class

        if status_callback:
            status_callback(f"Loading vision model: {os.path.basename(model_path)}...")

        dtype = kwargs.get("dtype", torch.float16)
        quantize = kwargs.get("quantize", True)

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "torch_dtype": dtype,
        }

        # Quantize on tight GPUs
        gpu_gb = 0
        if torch.cuda.is_available():
            gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        if quantize and gpu_gb < 20:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["device_map"] = "auto"
            except ImportError:
                load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = "auto"

        # Use the correct AutoModel class — Gemma 4 needs AutoModelForMultimodalLM
        ModelClass = _get_auto_model_class(model_path)
        self.model = ModelClass.from_pretrained(model_path, **load_kwargs)

        try:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        except Exception:
            self.processor = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._model_path = model_path

        if status_callback:
            status_callback(f"Vision model loaded: {os.path.basename(model_path)}")

    def unload(self, status_callback=None):
        for attr in ("model", "tokenizer", "processor"):
            if getattr(self, attr, None) is not None:
                delattr(self, attr)
                setattr(self, attr, None)
        self._model_path = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if status_callback:
            status_callback("Vision model unloaded.")

    def is_loaded(self) -> bool:
        return self.model is not None

    def run(self, **kwargs) -> PipelineResult:
        """Analyze an image with text prompt.

        Args:
            image_path: Path to the image file
            prompt: Question about the image (default: "Describe this image")
            max_tokens: Max output tokens (default 512)

        Returns:
            PipelineResult with text description
        """
        if not self.is_loaded():
            return PipelineResult(
                success=False, output_type="text",
                error="No model loaded."
            )

        image_path = kwargs.get("image_path", "")
        prompt = kwargs.get("prompt", "Describe this image in detail.")
        max_tokens = kwargs.get("max_tokens", 512)

        if not image_path or not os.path.isfile(image_path):
            return PipelineResult(
                success=False, output_type="text",
                error=f"Image not found: {image_path}"
            )

        try:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")

            if self.processor is not None:
                inputs = self.processor(
                    text=prompt, images=image, return_tensors="pt"
                ).to(self.model.device)
            else:
                # Fallback for models without a processor
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )

            response = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            return PipelineResult(
                success=True,
                output_type="text",
                content=response.strip(),
                metadata={"image_path": image_path, "prompt": prompt},
            )

        except Exception as e:
            return PipelineResult(
                success=False, output_type="text",
                error=str(e),
            )

    def get_vram_estimate(self, model_path: str) -> float:
        from core.model_registry import _estimate_model_size
        size_gb = _estimate_model_size(model_path)
        return round(size_gb * 0.7 + 2.0, 1)

    def get_capabilities(self) -> dict:
        caps = super().get_capabilities()
        caps["input_types"] = ["image + text"]
        caps["supported_formats"] = ["jpg", "png", "webp", "bmp"]
        return caps
