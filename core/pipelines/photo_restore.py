"""
Artifex Assistant V5 — Photo Restoration pipeline.

Three-stage restoration of damaged/old photos, returning every intermediate:

  Stage 1  Classical clean  — edge-preserving denoise (non-local means +
           bilateral "circle of interest" color rounding) + optional
           auto-levels. Pure OpenCV, CPU, sub-second.
  Stage 2  AI restore       — generative repair of the cleaned image:
           diffusers img2img at low denoising strength (keeps identity),
           or GFPGAN face restoration when available and no diffusion
           model is loaded.
  Stage 3  Upscale          — Real-ESRGAN (vendored, basicsr-free) with
           tiled inference so large scans fit in VRAM.

The result contains all three images so the GUI can show a comparison strip.
"""

import os
import time

from core.pipelines.base import BasePipeline, PipelineResult


# Recommended configuration per GPU VRAM tier. Consumed by the GUI model
# guide and by pick_preset(); mirrors what fits without CPU offload.
VRAM_PRESETS = [
    # (min_vram_gb, preset) — thresholds sit below nominal sizes because
    # cards report slightly under (a 24 GB card reports ~23.99 GB)
    (30, {
        "label": "32 GB+",
        "img2img_model": "black-forest-labs/FLUX.1-dev",
        "max_side": 2048, "tile": 1024, "work_res": 1024,
        "sr_model": "realesrgan-x4plus",
        "note": "FLUX.1-dev img2img, everything resident, no offload.",
    }),
    (22, {
        "label": "24 GB",
        "img2img_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "max_side": 1536, "tile": 800, "work_res": 1024,
        "sr_model": "realesrgan-x4plus",
        "note": "SDXL img2img fully resident + x4plus upscaler.",
    }),
    (15, {
        "label": "16 GB",
        "img2img_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "max_side": 1024, "tile": 512, "work_res": 1024,
        "sr_model": "realesrgan-x4plus",
        "note": "SDXL img2img fp16.",
    }),
    (11, {
        "label": "12 GB",
        "img2img_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "max_side": 1024, "tile": 512, "work_res": 640,
        "sr_model": "realesrgan-x4plus",
        "note": "SD 1.5 img2img; SDXL possible with CPU offload.",
    }),
    (7, {
        "label": "8 GB",
        "img2img_model": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "max_side": 768, "tile": 400, "work_res": 512,
        "sr_model": "realesr-general-x4v3",
        "note": "SD 1.5 at 768px + compact upscaler.",
    }),
    (0, {
        "label": "4 GB / CPU",
        "img2img_model": None,
        "max_side": 1024, "tile": 256, "work_res": 512,
        "sr_model": "realesr-general-x4v3",
        "note": "Classical clean + GFPGAN faces + compact upscaler; "
                "skip diffusion.",
    }),
]


REALISM_PROMPT = ("photorealistic photograph of this exact scene, realistic "
                  "materials and textures, natural lighting, sharp detail, "
                  "shot on a DSLR camera, high resolution photo")
REALISM_NEGATIVE = ("pixel art, pixelated, blocky, mosaic, cartoon, "
                    "illustration, painting, drawing, render, low quality, "
                    "jpeg artifacts, watermark, text, blurry, out of focus, "
                    "soft focus, hazy, bokeh")


def _weights_dir() -> str:
    """Absolute cache dir for SR / GFPGAN weights.

    Dot-prefixed so scan_local_models() skips it — these are bare .pth
    checkpoints, not a selectable diffusers model. An older build wrote
    them to models/restoration, which the model list did offer, and
    picking it crashed the loader; migrate anything left there.
    """
    from core.config import BASE_DIR
    dest = os.path.join(BASE_DIR, "models", ".restoration")
    legacy = os.path.join(BASE_DIR, "models", "restoration")
    os.makedirs(dest, exist_ok=True)
    if os.path.isdir(legacy):
        try:
            for name in os.listdir(legacy):
                target = os.path.join(dest, name)
                if not os.path.exists(target):
                    os.replace(os.path.join(legacy, name), target)
            if not os.listdir(legacy):
                os.rmdir(legacy)
        except OSError:
            pass
    return dest


def pick_preset(vram_gb: float) -> dict:
    """Return the recommended preset for a given VRAM size."""
    for min_gb, preset in VRAM_PRESETS:
        if vram_gb >= min_gb:
            return preset
    return VRAM_PRESETS[-1][1]


class PhotoRestorePipeline(BasePipeline):
    """Photo restoration: classical clean -> AI restore -> upscale."""

    pipeline_type = "photo-restoration"
    display_name = "Photo Restoration"
    output_type = "image"

    def __init__(self):
        self._ready = False
        self._model_path = None
        self.img2img = None       # diffusers pipeline (optional)
        self._sr_net = None
        self._sr_name = None
        self._gfpgan = None
        self._weights_dir = _weights_dir()

    # ── Loading ────────────────────────────────────────────────────────

    def load(self, model_path: str, status_callback=None, **kwargs):
        """Load stage-2 diffusion model if one is given.

        model_path may be empty — the pipeline still works (classical clean
        + optional GFPGAN + upscale). The SR net is loaded lazily in run()
        because the scale factor is a run-time parameter.
        """
        try:
            import cv2  # noqa: F401
        except ImportError:
            raise ImportError(
                "opencv-python-headless is required for photo restoration.\n"
                "Install with: pip install opencv-python-headless"
            )

        # A local path with no model_index.json is not a diffusers pipeline
        # (weights cache, LLM folder, ...). Stage 2 is optional, so warn and
        # fall through to classical clean + GFPGAN + upscale rather than
        # failing the whole run.
        if model_path and os.path.isdir(model_path) and not os.path.isfile(
                os.path.join(model_path, "model_index.json")):
            if status_callback:
                status_callback(
                    f"'{os.path.basename(model_path)}' is not a diffusion "
                    "model — skipping the AI restore stage.")
            model_path = ""

        if model_path:
            try:
                import torch
                from diffusers import AutoPipelineForImage2Image
            except ImportError:
                raise ImportError(
                    "diffusers is required for the AI restore stage.\n"
                    "Install with: pip install diffusers>=0.30.0"
                )
            if status_callback:
                status_callback(
                    f"Loading restore model: {os.path.basename(model_path)}...")
            self.img2img = AutoPipelineForImage2Image.from_pretrained(
                model_path, torch_dtype=kwargs.get("dtype", torch.float16),
            )
            from core.device import gpu_info
            if gpu_info.is_available and gpu_info.total_gb >= 12:
                self.img2img = self.img2img.to("cuda")
            else:
                self.img2img.enable_model_cpu_offload()
                if status_callback:
                    status_callback("Using CPU offload (low VRAM)")

        self._model_path = model_path
        self._ready = True
        if status_callback:
            status_callback("Photo restoration ready.")

    def unload(self, status_callback=None):
        import gc
        self.img2img = None
        self._sr_net = None
        self._sr_name = None
        self._gfpgan = None
        self._ready = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        if status_callback:
            status_callback("Photo restoration unloaded.")

    def is_loaded(self) -> bool:
        return self._ready

    # ── Pixelation detection ───────────────────────────────────────────

    @staticmethod
    def estimate_pixel_factor(image, max_factor=16, min_factor=3,
                              tol_db=32.0) -> int:
        """Detect nearest-neighbor pixelation; return the block factor.

        Box-downscale by each candidate factor and reconstruct with nearest
        neighbor; if the reconstruction still matches the original (high
        PSNR), the image only really contains 1/factor of its pixels.
        Returns 1 for genuinely detailed images.
        """
        import numpy as np
        from PIL import Image

        g = image.convert("L")
        src = np.asarray(g, np.float32)
        w, h = g.size
        best = 1
        for f in range(2, max_factor + 1):
            tw, th = max(1, round(w / f)), max(1, round(h / f))
            recon = np.asarray(
                g.resize((tw, th), Image.BOX).resize((w, h), Image.NEAREST),
                np.float32)
            mse = float(((src - recon) ** 2).mean())
            psnr = 99.0 if mse < 1e-3 else 10 * np.log10(255.0 ** 2 / mse)
            if psnr >= tol_db:
                best = f
        return best if best >= min_factor else 1

    # ── Stage 1: classical clean ───────────────────────────────────────

    @staticmethod
    def classical_clean(image, denoise: float = 7.0, smooth_radius: int = 4,
                        color_sigma: float = 30.0, auto_levels: bool = True):
        """Edge-preserving clean of scan grain and color noise.

        Non-local means removes grain by averaging similar patches; the
        bilateral pass then rounds each pixel toward the dominant color of
        its neighborhood circle, weighted by spatial distance within the
        circle and color-gradient similarity — smooths within regions,
        preserves edges.
        """
        import cv2
        import numpy as np
        from PIL import Image

        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        if denoise > 0:
            h = float(denoise)
            bgr = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)

        if smooth_radius > 0:
            d = smooth_radius * 2 + 1
            bgr = cv2.bilateralFilter(bgr, d, color_sigma, smooth_radius * 2.5)

        if auto_levels:
            # Faded old photos: CLAHE on lightness only, colors untouched
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            l_chan, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
            lab = cv2.merge((clahe.apply(l_chan), a, b))
            bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    # ── Stage 2: AI restore ────────────────────────────────────────────

    def _ai_restore(self, image, prompt, strength, num_steps,
                    guidance_scale, status_callback=None,
                    negative_prompt=""):
        """Generative repair. Returns (image, method) — method '' if skipped."""
        if self.img2img is not None:
            if status_callback:
                status_callback("AI restore (img2img)...")
            import inspect
            gen_kwargs = {
                "prompt": prompt,
                "image": image,
                "num_inference_steps": num_steps,
                "guidance_scale": guidance_scale,
            }
            if negative_prompt:
                gen_kwargs["negative_prompt"] = negative_prompt
            if "strength" in inspect.signature(
                    self.img2img.__call__).parameters:
                gen_kwargs["strength"] = strength
            return self.img2img(**gen_kwargs).images[0], "img2img"

        # No diffusion model — try GFPGAN face restoration
        try:
            if self._gfpgan is None:
                from gfpgan import GFPGANer
                from core.pipelines.sr_arch import download_weights
                url = ("https://github.com/TencentARC/GFPGAN/releases/"
                       "download/v1.3.0/GFPGANv1.4.pth")
                os.makedirs(self._weights_dir, exist_ok=True)
                dest = os.path.join(self._weights_dir, "GFPGANv1.4.pth")
                if not os.path.isfile(dest):
                    if status_callback:
                        status_callback("Downloading GFPGAN weights...")
                    import urllib.request
                    urllib.request.urlretrieve(url, dest + ".part")
                    os.replace(dest + ".part", dest)
                self._gfpgan = GFPGANer(model_path=dest, upscale=1,
                                        arch="clean", channel_multiplier=2)
            if status_callback:
                status_callback("AI restore (GFPGAN faces)...")
            import cv2
            import numpy as np
            from PIL import Image
            bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            _, _, restored = self._gfpgan.enhance(
                bgr, has_aligned=False, only_center_face=False,
                paste_back=True)
            return Image.fromarray(
                cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)), "gfpgan"
        except ImportError:
            return image, ""
        except Exception as e:
            if status_callback:
                status_callback(f"GFPGAN skipped: {e}")
            return image, ""

    # ── Stage 3: upscale ───────────────────────────────────────────────

    def _upscale(self, image, upscale, tile, status_callback=None):
        from core.pipelines import sr_arch
        sr_name = ("realesrgan-x2plus" if upscale == 2
                   else "realesrgan-x4plus")
        if self._sr_net is None or self._sr_name != sr_name:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._sr_net = sr_arch.load_sr_model(
                sr_name, self._weights_dir, device=device,
                half=(device == "cuda"), status_callback=status_callback)
            self._sr_name = sr_name
        scale = sr_arch.SR_MODELS[sr_name]["scale"]
        return sr_arch.upscale_tiled(
            self._sr_net, image, scale, tile=tile,
            status_callback=status_callback)

    # ── Run ────────────────────────────────────────────────────────────

    def run(self, **kwargs) -> PipelineResult:
        """Run the full restoration chain.

        Args:
            image_path: Source photo (required)
            prompt: Guidance for the AI restore stage (optional)
            strength: img2img denoising strength (default 0.3 — low keeps
                the photo's identity; high reinvents it)
            num_steps: Diffusion steps (default 30)
            guidance_scale: CFG (default 7.5)
            denoise: Classical denoise strength 0-30 (default 7)
            smooth_radius: Bilateral circle-of-interest radius (default 4)
            auto_levels: CLAHE contrast recovery (default True)
            upscale: Final upscale factor, 2 or 4 (default 2)
            tile: Upscaler tile size in px (default 512)
            output_path: Base path for outputs (optional)

        Returns:
            PipelineResult; content = final upscaled image path,
            metadata["stage_paths"] = [(label, path) x3].
        """
        from core.pipelines.schemas import PhotoRestoreInput
        from pydantic import ValidationError
        try:
            params = PhotoRestoreInput(**kwargs)
        except ValidationError as e:
            return PipelineResult(success=False, output_type="image",
                                  error=f"Invalid input: {e}")

        if not params.image_path or not os.path.isfile(params.image_path):
            return PipelineResult(
                success=False, output_type="image",
                error=f"Image not found: {params.image_path}")

        status_callback = kwargs.get("status_callback")
        try:
            from PIL import Image
            source = Image.open(params.image_path).convert("RGB")

            out_dir = os.path.dirname(params.output_path) if params.output_path \
                else "output"
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(params.image_path))[0]
            stamp = time.strftime("%H%M%S")

            timings = {}

            # Pixelated sources (tiny image blown up into blocks): collapse
            # back to true resolution first, otherwise every later stage
            # just sharpens the blocks.
            denoise, smooth_radius = params.denoise, params.smooth_radius
            strength = params.strength
            realism = params.realism and self.img2img is not None
            if params.realism and self.img2img is None and status_callback:
                status_callback("Realism mode needs a diffusion model "
                                "loaded — running normal restore instead.")
            if realism:
                strength = max(strength, 0.7)
            factor = self.estimate_pixel_factor(source) \
                if params.depixelate else 1
            work = source
            if factor > 1:
                true_size = (max(1, round(source.width / factor)),
                             max(1, round(source.height / factor)))
                if status_callback:
                    status_callback(
                        f"Pixelated source (~{factor}x blocks) — rebuilding "
                        f"from {true_size[0]}x{true_size[1]}")
                work = source.resize(true_size, Image.BOX)
                # Every true pixel is precious now — barely touch them
                denoise = min(denoise, 2.0)
                smooth_radius = min(smooth_radius, 1)
                strength = max(strength, 0.4)

            # Stage 1 — classical clean
            if status_callback:
                status_callback("Stage 1/3: classical clean...")
            t0 = time.time()
            cleaned = self.classical_clean(
                work, denoise=denoise,
                smooth_radius=smooth_radius,
                auto_levels=params.auto_levels)
            timings["clean_s"] = round(time.time() - t0, 2)
            p1 = os.path.join(out_dir, f"{base}_{stamp}_1_cleaned.png")
            cleaned.save(p1)

            # Depixelated images are tiny — SR up to the diffusion model's
            # native working resolution (small inputs make diffusion
            # hallucinate garbage); 320 is enough when no model is loaded.
            if factor > 1:
                if self.img2img is not None:
                    from core.device import gpu_info
                    target = pick_preset(gpu_info.total_gb)["work_res"]
                else:
                    target = 320
                while max(cleaned.size) < target:
                    cleaned = self._upscale(cleaned, 4, params.tile,
                                            status_callback)
                if max(cleaned.size) > target:
                    r = target / max(cleaned.size)
                    cleaned = cleaned.resize(
                        (round(cleaned.width * r), round(cleaned.height * r)),
                        Image.LANCZOS)

            # Realism mode: blur away the source's stylization (pixel
            # blocks, hard cartoon edges) at working resolution so the
            # structure pass re-renders subject and scene instead of
            # imitating the style.
            if realism:
                from PIL import ImageFilter
                from core.device import gpu_info
                work_res = pick_preset(gpu_info.total_gb)["work_res"]
                r = work_res / max(cleaned.size)
                cleaned = cleaned.resize(
                    (round(cleaned.width * r), round(cleaned.height * r)),
                    Image.LANCZOS).filter(ImageFilter.GaussianBlur(5))

            # Stage 2 — AI restore
            if status_callback:
                status_callback("Stage 2/3: AI restore...")
            t0 = time.time()
            prompt = params.prompt or (
                REALISM_PROMPT if realism else
                "restored old photograph, sharp focus, natural colors, "
                "high quality photo")
            negative = REALISM_NEGATIVE if realism else ""
            restored, method = self._ai_restore(
                cleaned, prompt, strength, params.num_steps,
                params.guidance_scale, status_callback,
                negative_prompt=negative)
            if realism and method:
                # Second pass regenerates the fine detail the blurred
                # structure pass could not commit to.
                if status_callback:
                    status_callback("Realism detail pass...")
                restored, method = self._ai_restore(
                    restored, prompt, 0.45, params.num_steps,
                    params.guidance_scale, status_callback,
                    negative_prompt=negative)
                method = "realism"
            timings["restore_s"] = round(time.time() - t0, 2)
            p2 = os.path.join(out_dir, f"{base}_{stamp}_2_restored.png")
            restored.save(p2)

            # Stage 3 — upscale
            if status_callback:
                status_callback("Stage 3/3: upscale...")
            t0 = time.time()
            upscaled = self._upscale(restored, params.upscale, params.tile,
                                     status_callback)
            timings["upscale_s"] = round(time.time() - t0, 2)
            p3 = os.path.join(out_dir, f"{base}_{stamp}_3_upscaled.png")
            upscaled.save(p3)

            return PipelineResult(
                success=True,
                output_type="image",
                content=p3,
                metadata={
                    "saved_to": p3,
                    "stage_paths": [
                        ("Cleaned (classical)", p1),
                        (f"AI restored ({method or 'skipped'})", p2),
                        (f"Upscaled x{params.upscale}", p3),
                    ],
                    "restore_method": method,
                    "depixelate_factor": factor,
                    "timings": timings,
                    "source_size": list(source.size),
                    "final_size": list(upscaled.size),
                },
            )
        except Exception as e:
            return PipelineResult(success=False, output_type="image",
                                  error=str(e))

    # ── Introspection ──────────────────────────────────────────────────

    def get_vram_estimate(self, model_path: str) -> float:
        if not model_path:
            return 2.5  # SR net + GFPGAN headroom
        from core.model_registry import _estimate_model_size
        return round(_estimate_model_size(model_path) * 1.5 + 2.5, 1)

    def get_capabilities(self) -> dict:
        caps = super().get_capabilities()
        caps["stages"] = ["classical-clean", "ai-restore", "upscale"]
        caps["input_types"] = ["image"]
        caps["supported_formats"] = ["jpg", "png", "webp", "tif", "bmp"]
        caps["vram_presets"] = {p["label"]: p["note"]
                                for _, p in VRAM_PRESETS}
        return caps
