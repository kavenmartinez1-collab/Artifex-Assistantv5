"""
Artifex Assistant V5 — Vision Pipeline.
Image understanding using multimodal models (LLaVA, Qwen-VL, etc.)
"""

import gc
import os

import torch

from core.pipelines.base import BasePipeline, PipelineResult

_VIDEO_EXTS = {".mp4", ".avi", ".webm", ".mov", ".mkv", ".m4v", ".flv", ".wmv"}


def _plan_video_sampling(
    video_path: str,
    max_pixels: int,
    target_interval_s: float = 20.0,
    min_frames: int = 8,
    max_frames: int = 128,
) -> tuple[int, str]:
    """Pick a frame count that fits host RAM and covers the video meaningfully.

    Returns (nframes, explanation). Falls back to a safe default if probing fails.
    """
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        n_total = len(vr)
        avg_fps = vr.get_avg_fps() or 30.0
        duration_s = n_total / avg_fps
        h, w = vr[0].shape[:2]
        del vr
    except Exception:
        return 32, "default (could not probe video)"

    try:
        import psutil
        free_bytes = psutil.virtual_memory().available
    except ImportError:
        free_bytes = 8 * 1024 ** 3

    # Budget: 25% of free RAM, clamped to [512 MB, 2 GB] so we don't starve
    # the model or over-commit on huge machines.
    budget = max(512 * 1024 ** 2, min(2 * 1024 ** 3, free_bytes // 4))

    # Raw decoded frame (uint8 RGB at native resolution) dominates cost.
    # Processed tensor (resized to max_pixels, bfloat16) is secondary.
    raw_per_frame = w * h * 3
    processed_per_frame = max_pixels * 3 * 2
    per_frame = raw_per_frame + processed_per_frame

    max_affordable = int(budget // per_frame)
    target = int(duration_s / target_interval_s)
    nframes = max(min_frames, min(max_frames, max_affordable, target or min_frames))

    interval = duration_s / nframes if nframes else 0
    explanation = (
        f"{nframes} frames from {duration_s:.0f}s {w}x{h} video "
        f"(~1/{interval:.1f}s, budget {budget/1024**2:.0f} MB)"
    )
    return nframes, explanation


def _resolve_vlm_class(model_path: str):
    """Pick the right AutoModel class for a vision-language model.

    Modern VLMs (Qwen-VL, LLaVA, Llama-3.2-Vision, etc.) register under
    AutoModelForImageTextToText. Gemma 3n/4 still uses AutoModelForMultimodalLM
    in some transformers builds. Fall back to CausalLM only if neither works.
    """
    import json
    cfg_path = os.path.join(model_path, "config.json")
    model_type = ""
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            model_type = (json.load(f).get("model_type") or "").lower()
    except (OSError, ValueError):
        pass

    # Gemma-4 family prefers AutoModelForMultimodalLM
    if model_type.startswith(("gemma3n", "gemma4", "gemma_4")):
        try:
            from transformers import AutoModelForMultimodalLM
            return AutoModelForMultimodalLM
        except ImportError:
            pass

    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
    except ImportError:
        pass

    try:
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq
    except ImportError:
        pass

    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM


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

        if status_callback:
            status_callback(f"Loading vision model: {os.path.basename(model_path)}...")

        dtype = kwargs.get("dtype", torch.bfloat16)
        quantize = kwargs.get("quantize", True)

        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }

        gpu_gb = 0
        if torch.cuda.is_available():
            gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        # Load strategy:
        # - Quantize (<20 GB GPUs): bnb + device_map="auto" (standard big-model path).
        # - >=20 GB GPUs: classic path (no device_map, no low_cpu_mem_usage). Every
        #   meta-tensor device_map ("auto" and explicit single-device) crashes in
        #   transformers 5.2.0's _materialize_copy for Qwen3-VL on Windows. Classic
        #   path needs ~19 GB host commit briefly; we free everything else first
        #   and move to GPU immediately after.
        use_classic_path = True
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
                load_kwargs["low_cpu_mem_usage"] = True
                use_classic_path = False
            except ImportError:
                pass

        ModelClass = _resolve_vlm_class(model_path)

        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

        # Free as much host RAM / VRAM as possible before the 17 GB staging spike.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model = ModelClass.from_pretrained(model_path, **load_kwargs)

        if use_classic_path and torch.cuda.is_available():
            self.model = self.model.to("cuda")
            gc.collect()
            torch.cuda.empty_cache()

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
        """Analyze an image or video with a text prompt.

        Args:
            image_path: Path to an image OR video file (auto-detected by extension)
            prompt: Question about the media (default: "Describe this image/video")
            max_tokens: Max output tokens (default 512)
            fps: Video sampling rate (default 1.0)
            max_frames: Cap on sampled frames for a video (default 32)
            max_pixels: Per-frame max pixels for video (default 360*420)

        Returns:
            PipelineResult with text description
        """
        from core.pipelines.schemas import VisionInput
        from pydantic import ValidationError
        try:
            params = VisionInput(**kwargs)
        except ValidationError as e:
            return PipelineResult(success=False, output_type="text", error=f"Invalid input: {e}")

        if not self.is_loaded():
            return PipelineResult(
                success=False, output_type="text",
                error="No model loaded."
            )

        media_path = params.image_path

        if not media_path or not os.path.isfile(media_path):
            return PipelineResult(
                success=False, output_type="text",
                error=f"Media not found: {media_path}"
            )

        is_video = os.path.splitext(media_path)[1].lower() in _VIDEO_EXTS

        try:
            if is_video:
                return self._run_video(
                    media_path, params.prompt, params.max_tokens,
                    nframes=params.nframes, fps=params.fps, max_pixels=params.max_pixels,
                )
            return self._run_image(media_path, params.prompt, params.max_tokens)
        except Exception as e:
            return PipelineResult(
                success=False, output_type="text",
                error=str(e),
            )

    def _run_image(self, image_path: str, prompt: str, max_tokens: int) -> PipelineResult:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

        if self.processor is not None:
            inputs = self.processor(
                text=prompt, images=image, return_tensors="pt"
            ).to(self.model.device)
        else:
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
            success=True, output_type="text",
            content=response.strip(),
            metadata={"image_path": image_path, "prompt": prompt},
            backend="transformers",
        )

    def _run_video(self, video_path: str, prompt: str, max_tokens: int, **kwargs) -> PipelineResult:
        if self.processor is None:
            raise RuntimeError(
                "Video input requires a processor. The loaded model has none — "
                "use a Qwen-VL / LLaVA-style model with an AutoProcessor."
            )

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError(
                "Video input requires qwen-vl-utils.\n"
                "Install with: pip install 'qwen-vl-utils[decord]'"
            )

        # Qwen-VL accepts either `fps` (uniform sampling rate) OR `nframes`
        # (fixed frame count), not both — passing both makes one silently win.
        # When neither is supplied, auto-plan based on video length + free RAM.
        nframes = kwargs.get("nframes")
        fps = kwargs.get("fps")
        max_pixels = int(kwargs.get("max_pixels", 360 * 420))
        if nframes is None and fps is None:
            nframes, plan_note = _plan_video_sampling(video_path, max_pixels)
            print(f"[vision] auto-sample: {plan_note}")

        video_content: dict = {
            "type": "video",
            "video": video_path,
            "max_pixels": max_pixels,
        }
        if nframes is not None:
            video_content["nframes"] = int(nframes)
        else:
            video_content["fps"] = float(fps)

        messages = [{
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        trimmed = generated[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return PipelineResult(
            success=True, output_type="text",
            content=response.strip(),
            metadata={
                "video_path": video_path,
                "prompt": prompt,
                "fps": fps,
                "nframes": nframes,
                "max_pixels": max_pixels,
            },
            backend="transformers",
        )

    def get_vram_estimate(self, model_path: str) -> float:
        from core.model_registry import _estimate_model_size
        size_gb = _estimate_model_size(model_path)
        return round(size_gb * 0.7 + 2.0, 1)

    def get_capabilities(self) -> dict:
        caps = super().get_capabilities()
        caps["input_types"] = ["image + text", "video + text"]
        caps["supported_formats"] = ["jpg", "png", "webp", "bmp"]
        caps["video_formats"] = sorted(e.lstrip(".") for e in _VIDEO_EXTS)
        return caps
