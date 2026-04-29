"""
Artifex Assistant V5 — Transformers engine.
Merges ModelManager (load/unload/VRAM) + InferenceEngine (streaming generation)
into a single BaseEngine implementation.

Supports pre-quantized weight caching: first load quantizes from source and
saves NF4 weights to disk. Subsequent loads skip quantization entirely,
cutting startup from ~60-90s to ~20-30s.

GPU tiers (by dedicated VRAM):
  TIGHT       <= 12 GB  — aggressive savings: quantize lm_head, cap VRAM
  COMFORTABLE  13-20 GB — moderate: cap VRAM at 92%, native lm_head
  ABUNDANT    > 20 GB   — unrestricted: no VRAM cap, full precision compute
"""

import gc
import json
import os
import logging
import warnings
from threading import Thread

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    StopStringCriteria,
    StoppingCriteriaList,
)

from core.engine_base import BaseEngine
from core.config import (
    BNB_CONFIG, get_active_model_path,
    get_torch_compile, get_turboquant_kv,
)
from core.inference import STOP_STRINGS, _clean_response

# Model types that historically required AutoModelForMultimodalLM (Gemma 4
# family). Image-text-to-text models (Qwen-VL, LLaVA, etc.) are now resolved
# via the transformers auto-mapping registry so we don't have to maintain
# a per-family allowlist.
_MULTIMODAL_MODEL_TYPES = {
    "gemma3n", "gemma3n_text", "gemma3n_vision", "gemma3n_audio",
}
_MULTIMODAL_MODEL_PREFIXES = ("gemma3n", "gemma4", "gemma_4")


def _detect_model_type_from_config(model_path: str):
    """Read model_type from config.json."""
    config_path = os.path.join(model_path, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f).get("model_type")
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _is_image_text_to_text(model_type: str) -> bool:
    """True if model_type is registered for AutoModelForImageTextToText.

    Covers Qwen-VL (qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_vl_moe), LLaVA,
    InternVL, MiniCPM-V, Pixtral, etc. — anything transformers itself maps
    to image→text. Reading from the live registry means new VL families
    work without code changes here.
    """
    if not model_type:
        return False
    try:
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
        )
    except ImportError:
        return False
    return model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES


def _is_multimodal_model(model_type: str) -> bool:
    """Check if a model_type needs a multimodal AutoModel + AutoProcessor.

    Returns True for image-text-to-text models (registry lookup) AND for
    the Gemma 4 / 3n family which uses AutoModelForMultimodalLM.
    """
    if not model_type:
        return False
    if _is_image_text_to_text(model_type):
        return True
    if model_type in _MULTIMODAL_MODEL_TYPES:
        return True
    model_type_lower = model_type.lower()
    return any(model_type_lower.startswith(p) for p in _MULTIMODAL_MODEL_PREFIXES)


def _matches_gemma_multimodal(model_type: str) -> bool:
    """True for the Gemma 4 / 3n family (vision + audio multimodal)."""
    if not model_type:
        return False
    if model_type in _MULTIMODAL_MODEL_TYPES:
        return True
    return any(model_type.lower().startswith(p) for p in _MULTIMODAL_MODEL_PREFIXES)


def _get_auto_model_class(model_path: str):
    """Return the appropriate AutoModel class for a given model.

    Resolution order:
      1. Gemma 4 / 3n family → AutoModelForMultimodalLM (vision + audio)
      2. Anything registered for image-text-to-text (Qwen-VL, LLaVA,
         InternVL, etc.) → AutoModelForImageTextToText
      3. Everything else → AutoModelForCausalLM

    Step 2 reads the live transformers registry, so new VL families work
    without code changes here.
    """
    model_type = _detect_model_type_from_config(model_path)

    if _matches_gemma_multimodal(model_type):
        try:
            from transformers import AutoModelForMultimodalLM
            return AutoModelForMultimodalLM
        except ImportError:
            logging.getLogger(__name__).warning(
                "AutoModelForMultimodalLM not available (needs transformers >= 5.5.0). "
                "Falling back to AutoModelForCausalLM for %s.", model_type
            )

    if _is_image_text_to_text(model_type):
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText
        except ImportError:
            logging.getLogger(__name__).warning(
                "AutoModelForImageTextToText not available; falling back for %s.",
                model_type,
            )

    return AutoModelForCausalLM

# Suppress harmless warnings
warnings.filterwarnings("ignore", message="expandable_segments not supported")
warnings.filterwarnings("ignore", message="The following layers were not sharded")
warnings.filterwarnings("ignore", message=".*fast path is not available.*")
warnings.filterwarnings("ignore", message=".*required library is not installed.*")
warnings.filterwarnings("ignore", message=".*FP4 quantization state not initialized.*")
warnings.filterwarnings("ignore", message=".*quantization.*", module="bitsandbytes")
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
warnings.filterwarnings("ignore", message=".*bitsandbytes.*")


def _patch_config_json(model_path: str):
    """Promote text_config attributes into the top-level config.json.

    Qwen3.5 is multimodal — text attributes (vocab_size, pad_token_id, etc.)
    live inside text_config.  Some transformers versions don't auto-delegate
    from the parent config to text_config, causing AttributeError during init.

    This patches the JSON once (idempotent) so all subsequent loads work
    natively without passing config=config to from_pretrained.
    """
    config_path = os.path.join(model_path, "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return

    if "text_config" not in cfg:
        return

    patched = False
    for key, value in cfg["text_config"].items():
        if key not in cfg:
            cfg[key] = value
            patched = True

    if patched:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
        logging.getLogger(__name__).info(
            "Patched config.json — promoted text_config attributes"
        )


def _verify_4bit_packing(model) -> bool:
    """Return True if Linear4bit weights were correctly packed.

    On some GPU / CUDA-driver combos bitsandbytes wraps layers in
    Linear4bit but silently fails to pack weights to 4-bit format.
    Checking one quantised layer catches this before the first forward pass.
    """
    try:
        from bitsandbytes.nn import Linear4bit
        for module in model.modules():
            if isinstance(module, Linear4bit):
                return module.weight.shape[1] == 1
    except Exception:
        pass  # bnb not installed or no 4-bit layers — assume packed
    return True


class TransformersEngine(BaseEngine):
    """Local transformers backend with NF4 quantization and streaming generation."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None  # For multimodal models (Gemma 4, etc.)
        self._loaded_path = None
        self._gpu_tier = None
        self._last_gen_stats = None

    def _load_processor(self, model_path: str):
        """Load AutoProcessor for multimodal models (Gemma 4, etc.).

        Falls back silently for text-only models that don't ship a processor.
        """
        model_type = _detect_model_type_from_config(model_path)
        if not _is_multimodal_model(model_type):
            self.processor = None
            return
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            logging.getLogger(__name__).info("Loaded AutoProcessor for %s", model_type)
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Could not load processor for %s: %s", model_type, e
            )
            self.processor = None

    # =========================================================================
    # GPU TIER DETECTION
    # =========================================================================

    def _detect_gpu_tier(self):
        """Detect GPU VRAM and return tier-appropriate settings.

        Tiers:
            TIGHT (<=12 GB):   8-12 GB cards — aggressive savings
            COMFORTABLE (13-20 GB): 12-16 GB cards — balanced
            ABUNDANT (>20 GB): 24+ GB cards — full precision lm_head
        """
        if self._gpu_tier is not None:
            return self._gpu_tier

        if not torch.cuda.is_available():
            self._gpu_tier = {
                "tier": "TIGHT", "name": "CPU", "total_gb": 0,
                "compute_cap": 0, "quantize_lm_head": True,
                "compute_dtype": torch.float16, "mem_fraction": 0.85,
            }
            return self._gpu_tier

        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        compute_cap = props.major

        if total_gb <= 12:
            tier, fraction = "TIGHT", 0.85
        elif total_gb <= 20:
            tier, fraction = "COMFORTABLE", 0.92
        else:
            tier, fraction = "ABUNDANT", None

        self._gpu_tier = {
            "tier": tier,
            "name": props.name,
            "total_gb": round(total_gb, 1),
            "compute_cap": compute_cap,
            "quantize_lm_head": (tier == "TIGHT"),
            "compute_dtype": torch.bfloat16 if compute_cap >= 8 else torch.float16,
            "mem_fraction": fraction,
        }
        return self._gpu_tier

    # =========================================================================
    # QUANTIZATION CACHE
    # =========================================================================

    def _has_cached_quantized(self, quantized_path):
        """Check if pre-quantized weights exist on disk."""
        if not os.path.isdir(quantized_path):
            return False
        if not os.path.isfile(os.path.join(quantized_path, "config.json")):
            return False
        # Must have actual model weights, not just config stubs
        has_weights = any(
            f for f in os.listdir(quantized_path)
            if f.endswith((".safetensors", ".bin"))
        )
        return has_weights

    def _get_vram_constraints(self, allow_cpu_staging=False):
        """Build max_memory dict using GPU tier-aware memory fraction."""
        max_mem = {}
        if torch.cuda.is_available():
            gpu = self._detect_gpu_tier()
            total = torch.cuda.get_device_properties(0).total_memory
            fraction = gpu["mem_fraction"] or 0.95
            max_mem[0] = int(total * fraction)

            if allow_cpu_staging:
                max_mem["cpu"] = 8 * 1024 ** 3
            else:
                max_mem["cpu"] = 0
        return max_mem if max_mem else None

    def _save_quantized_cache(self, quantized_path, status_callback=None):
        """Save quantized weights to disk for fast future loads."""
        from datetime import datetime

        if status_callback:
            status_callback("Caching quantized weights to disk (one-time)...")

        self.model.save_pretrained(quantized_path)
        self.tokenizer.save_pretrained(quantized_path)

        gpu = self._detect_gpu_tier()
        meta = {
            "gpu_tier": gpu["tier"],
            "lm_head_quantized": gpu["quantize_lm_head"],
            "quantized_at": datetime.now().isoformat(),
            "total_vram_gb": round(gpu["total_gb"], 1),
        }
        meta_path = os.path.join(quantized_path, "_quant_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        if status_callback:
            status_callback("Quantized cache saved.")

    def _check_cache_tier_mismatch(self, quantized_path, gpu, status_callback=None):
        """Warn if cached weights were quantized on a different GPU tier."""
        meta_path = os.path.join(quantized_path, "_quant_meta.json")
        if not os.path.isfile(meta_path):
            return

        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        cached_lm_head_quantized = meta.get("lm_head_quantized", False)
        need_lm_head_quantized = gpu["quantize_lm_head"]

        if need_lm_head_quantized and not cached_lm_head_quantized:
            cached_tier = meta.get("gpu_tier", "unknown")
            msg = (
                f"WARNING: Cached weights were quantized on {cached_tier} tier "
                f"(lm_head NOT quantized). This GPU is {gpu['tier']} tier and needs "
                f"lm_head quantized to save ~1.4 GB VRAM.\n"
                f"Consider deleting the -nf4-cached directory to re-quantize."
            )
            if status_callback:
                status_callback(msg)

    # =========================================================================
    # BaseEngine — LIFECYCLE
    # =========================================================================

    def load(self, status_callback=None):
        """Load the quantized model. No-op if already loaded.

        First load: quantizes from source weights and caches to disk.
        Subsequent loads: loads pre-quantized weights directly (2-3x faster).
        """
        if self.is_loaded():
            return

        # ── Pre-flight: CUDA check ───────────────────────────────────────
        if not torch.cuda.is_available():
            raise RuntimeError(
                "\n\n"
                "═══════════════════════════════════════════════════════════\n"
                "  CUDA NOT AVAILABLE — cannot load model on GPU.\n"
                "═══════════════════════════════════════════════════════════\n"
                "\n"
                "  PyTorch was installed without GPU support.\n"
                "  Fix: reinstall PyTorch with CUDA:\n"
                "\n"
                "    pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
                "\n"
                "  Then restart Artifex Assistant V5.\n"
                "═══════════════════════════════════════════════════════════\n"
            )

        # ── Pre-flight: bitsandbytes check ──────────────────────────────
        try:
            import bitsandbytes as _bnb  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "\n\n"
                "═══════════════════════════════════════════════════════════\n"
                "  bitsandbytes NOT INSTALLED — cannot quantize model.\n"
                "═══════════════════════════════════════════════════════════\n"
                "\n"
                "  Fix:  pip install bitsandbytes>=0.49.0\n"
                "  Then restart Artifex Assistant V5.\n"
                "═══════════════════════════════════════════════════════════\n"
            )

        # ── Resolve model path ─────────────────────────────────────────
        model_path = get_active_model_path()
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"\n\n"
                f"Model not found: {model_path}\n\n"
                f"No local model is downloaded. To fix this, either:\n"
                f"  1. Download a model:  python download_model.py\n"
                f"  2. Switch to Ollama:  /backend ollama  (in CLI)\n"
                f"     or select 'ollama' in the GUI backend dropdown\n"
            )
        quantized_path = model_path + "-nf4-cached"
        model_name = os.path.basename(model_path)

        # Suppress noisy logs
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        logging.getLogger("transformers.models").setLevel(logging.ERROR)
        logging.getLogger("accelerate").setLevel(logging.ERROR)

        # ── Detect GPU capabilities ──────────────────────────────────────
        gpu = self._detect_gpu_tier()

        if status_callback:
            status_callback(
                f"GPU: {gpu['name']} — {gpu['total_gb']:.0f} GB, tier={gpu['tier']}"
            )

        # ── FAST PATH: Load pre-quantized cache ──────────────────────────
        if self._has_cached_quantized(quantized_path):
            self._check_cache_tier_mismatch(quantized_path, gpu, status_callback)

            if status_callback:
                status_callback(f"Loading {model_name} (fast path)...")

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._load_processor(model_path)

            try:
                self.model = _get_auto_model_class(model_path).from_pretrained(
                    quantized_path,
                    device_map={"": 0},
                    dtype=gpu["compute_dtype"],
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="sdpa",
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                if status_callback:
                    status_callback("Direct GPU load failed — retrying with memory constraints...")
                torch.cuda.empty_cache()
                max_mem = self._get_vram_constraints(allow_cpu_staging=False)
                self.model = _get_auto_model_class(model_path).from_pretrained(
                    quantized_path,
                    device_map="auto",
                    max_memory=max_mem,
                    dtype=gpu["compute_dtype"],
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    attn_implementation="sdpa",
                )

            # Register dtype correction hooks
            self._register_dtype_hooks()
            self._apply_torch_compile(status_callback)

            self._loaded_path = model_path
            self._report_vram(model_name, gpu, "cached-nf4", status_callback)
            return

        # ── SLOW PATH: First-time quantization ───────────────────────────
        if status_callback:
            status_callback(f"Loading {model_name} (first run — quantizing)...")

        # Dynamic VRAM budget: analyze model weights and decide which layer
        # groups can stay at BF16 vs must be NF4 to fit on this GPU.
        plan = self._plan_load_strategy(model_path, gpu, status_callback)
        _skip_modules = plan["skip_modules"]

        bnb_kwargs = dict(
            load_in_4bit=BNB_CONFIG["load_in_4bit"],
            bnb_4bit_compute_dtype=gpu["compute_dtype"],
            bnb_4bit_quant_type=BNB_CONFIG["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=BNB_CONFIG["bnb_4bit_use_double_quant"],
            llm_int8_skip_modules=_skip_modules,
        )
        bnb_config = BitsAndBytesConfig(**bnb_kwargs)

        # Patch multimodal config (Qwen3.5 specific)
        _patch_config_json(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._load_processor(model_path)

        # GPU-only placement — no CPU offload / shared memory spillover
        self.model = _get_auto_model_class(model_path).from_pretrained(
            model_path,
            device_map={"": 0},
            quantization_config=bnb_config,
            dtype=gpu["compute_dtype"],
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )

        # ── Verify 4-bit packing ────────────────────────────────────────
        _fell_back_to_8bit = False
        if BNB_CONFIG["load_in_4bit"] and not _verify_4bit_packing(self.model):
            logging.getLogger(__name__).warning(
                "NF4 4-bit packing failed — reloading with 8-bit quantization."
            )
            if status_callback:
                status_callback("4-bit quantization failed — reloading with 8-bit...")

            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()

            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=_skip_modules,
            )

            self.model = _get_auto_model_class(model_path).from_pretrained(
                model_path,
                device_map={"": 0},
                quantization_config=bnb_config,
                dtype=gpu["compute_dtype"],
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
            _fell_back_to_8bit = True

        # Register dtype correction hooks
        self._register_dtype_hooks()

        # ── Cache quantized weights (4-bit only) ──────────────────────
        if not _fell_back_to_8bit:
            try:
                self._save_quantized_cache(quantized_path, status_callback)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "Failed to cache quantized weights: %s", e
                )

        self._apply_torch_compile(status_callback)

        self._loaded_path = model_path
        quant_label = "8-bit" if _fell_back_to_8bit else plan["strategy_label"]
        self._report_vram(model_name, gpu, quant_label, status_callback)

    def _apply_torch_compile(self, status_callback=None):
        """Optionally JIT-compile the model for faster inference.

        Gated by config flag. Adds ~30s warmup on first generation but
        20-40% throughput improvement on subsequent ones.
        """
        if not get_torch_compile():
            return
        if self.model is None:
            return

        log = logging.getLogger(__name__)
        try:
            if status_callback:
                status_callback("Applying torch.compile (first generation will be slower)...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            log.info("torch.compile applied (mode=reduce-overhead)")
        except Exception as e:
            log.warning("torch.compile failed (non-fatal): %s", e)
            if status_callback:
                status_callback(f"torch.compile skipped: {e}")

    def _register_dtype_hooks(self):
        """Fix dtype mismatches between quantized and non-quantized layers.

        Only hooks modules whose weight dtype differs from the model's
        compute dtype — avoids adding forward-pass overhead to layers
        that already match.
        """
        try:
            import bitsandbytes as bnb

            compute_dtype = self._detect_gpu_tier()["compute_dtype"]

            def _make_dtype_hook(mod):
                def hook(module, args):
                    if args[0].dtype != module.weight.dtype:
                        return (args[0].to(module.weight.dtype),) + args[1:]
                return hook

            hooked = 0
            for name, module in self.model.named_modules():
                if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
                    continue
                if hasattr(module, "weight") and isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
                    if module.weight.dtype != compute_dtype:
                        module.register_forward_pre_hook(_make_dtype_hook(module))
                        hooked += 1

            if hooked:
                logging.getLogger(__name__).debug(
                    "Registered dtype hooks on %d modules", hooked
                )
        except ImportError:
            pass

    # =========================================================================
    # DYNAMIC VRAM BUDGET PLANNER
    # =========================================================================

    def _plan_load_strategy(self, model_path, gpu, status_callback=None):
        """Analyze model weights and plan quantization to fit VRAM budget.

        Reads the safetensors weight index to calculate per-group sizes,
        then progressively drops groups from the BF16 skip list until the
        estimated footprint fits in the GPU's VRAM budget.

        Priority order (last dropped = highest quality preference):
          1. lm_head / embed_tokens  — large but tolerant of NF4
          2. SSM / recurrent layers  — sensitive but BnB NF4 dequantizes
             to full precision before compute, so recurrence runs at BF16
          3. Everything else          — always NF4

        Returns:
            dict with 'skip_modules', 'estimated_gb', 'strategy_label'
        """
        log = logging.getLogger(__name__)

        # ── VRAM budget ──────────────────────────────────────────────────
        # Reserve for CUDA context (~0.4 GB), KV cache (~0.5 GB),
        # activations / peak overhead (~0.3 GB)
        overhead_gb = 1.2
        vram_budget = gpu["total_gb"] - overhead_gb

        # ── Detect layer groups from weight index ────────────────────────
        groups = self._analyze_weight_groups(model_path)

        if not groups:
            # No weight index found — fall back to empty skip (NF4 everything)
            log.warning("No weight index found — quantizing all layers to NF4")
            return {"skip_modules": [], "estimated_gb": 0, "strategy_label": "nf4-all"}

        # ── Calculate size under each skip configuration ─────────────────
        # "candidates" are groups we COULD keep at BF16 for quality.
        # Ordered from most-expendable to least-expendable.
        candidate_order = ["lm_head", "embed", "ssm"]
        base_nf4_gb = groups.get("other_nf4", 0)

        # Build configs from most aggressive (NF4-all) to least (BF16-all-candidates)
        configs = []

        # Config 0: NF4 everything
        total = base_nf4_gb + sum(groups.get(f"{c}_nf4", 0) for c in candidate_order)
        configs.append(([], total, "nf4-all"))

        # Progressive configs: keep candidates at BF16 one by one
        kept = []
        for cand in reversed(candidate_order):
            kept.insert(0, cand)
            total = base_nf4_gb
            for c in candidate_order:
                if c in kept:
                    total += groups.get(f"{c}_bf16", 0)
                else:
                    total += groups.get(f"{c}_nf4", 0)
            patterns = []
            for c in kept:
                patterns.extend(groups.get(f"{c}_patterns", []))
            label = "nf4+bf16(" + ",".join(kept) + ")"
            configs.append((patterns, total, label))

        # ── Pick the least-aggressive config that fits ───────────────────
        chosen = configs[0]  # fallback: NF4 everything
        for skip_modules, est_gb, label in reversed(configs):
            if est_gb <= vram_budget:
                chosen = (skip_modules, est_gb, label)
                break

        skip_modules, est_gb, label = chosen
        log.info(
            "Load strategy: %s — estimated %.1f / %.1f GB budget (%.1f GB total VRAM)",
            label, est_gb, vram_budget, gpu["total_gb"],
        )
        if status_callback:
            status_callback(f"Strategy: {label} — ~{est_gb:.1f} GB / {vram_budget:.1f} GB budget")

        return {
            "skip_modules": skip_modules,
            "estimated_gb": est_gb,
            "strategy_label": label,
        }

    def _analyze_weight_groups(self, model_path):
        """Read safetensors index and classify params into groups with sizes.

        Returns dict with keys like:
            ssm_nf4, ssm_bf16, ssm_patterns,
            lm_head_nf4, lm_head_bf16, lm_head_patterns,
            embed_nf4, embed_bf16, embed_patterns,
            other_nf4
        Values are sizes in GB.
        """
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if not os.path.isfile(index_path):
            # Single-file model — can't analyze without loading
            return {}

        try:
            with open(index_path, "r") as f:
                index = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

        weight_map = index.get("weight_map", {})
        if not weight_map:
            return {}

        # ── Detect SSM/recurrent patterns from model config ──────────────
        config_path = os.path.join(model_path, "config.json")
        ssm_keywords = set()
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            # Check for hybrid layer_types (e.g., Qwen3.5, Jamba, Zamba)
            layer_types = cfg.get("layer_types", [])
            for lt in layer_types:
                if lt != "full_attention" and "attention" in lt:
                    # e.g., "linear_attention" → "linear_attn" pattern
                    ssm_keywords.add(lt.replace("attention", "attn"))
                    ssm_keywords.add(lt)
            # Check for mamba/ssm indicators
            if cfg.get("mamba_ssm_dtype") or cfg.get("ssm_cfg"):
                ssm_keywords.add("mamba")
                ssm_keywords.add("ssm")
            if "linear_attention" in layer_types:
                # DeltaNet-specific projection names
                ssm_keywords.update(["in_proj_qkv", "in_proj_a", "in_proj_b",
                                     "in_proj_z", "out_proj", "linear_attn"])
        except (json.JSONDecodeError, OSError, KeyError):
            pass

        # ── Classify each parameter ──────────────────────────────────────
        # Estimate param count from name (we don't load tensors here).
        # Use metadata.total_size if available, else estimate from weight count.
        total_size = index.get("metadata", {}).get("total_size", 0)
        param_count = len(weight_map)

        # We need actual tensor sizes. Parse from safetensors metadata if available.
        param_bytes = {}
        try:
            from safetensors import safe_open
            shard_files = set(weight_map.values())
            for shard in shard_files:
                path = os.path.join(model_path, shard)
                with safe_open(path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        t = f.get_slice(key)
                        shape = t.get_shape()
                        # Assume source is BF16/FP16 (2 bytes per element)
                        numel = 1
                        for s in shape:
                            numel *= s
                        param_bytes[key] = numel * 2  # BF16 bytes
        except Exception:
            # Fallback: estimate evenly from total_size
            if total_size and param_count:
                avg = total_size / param_count
                param_bytes = {k: avg for k in weight_map}
            else:
                return {}

        # ── Group params ─────────────────────────────────────────────────
        ssm_bf16 = 0
        lm_head_bf16 = 0
        embed_bf16 = 0
        other_bf16 = 0
        ssm_patterns_found = set()

        for name, bf16_bytes in param_bytes.items():
            if "lm_head" in name:
                lm_head_bf16 += bf16_bytes
            elif "embed" in name and "layer" not in name:
                embed_bf16 += bf16_bytes
            elif ssm_keywords and any(kw in name for kw in ssm_keywords):
                ssm_bf16 += bf16_bytes
                # Extract the module-level pattern for the skip list
                for kw in ssm_keywords:
                    if kw in name:
                        ssm_patterns_found.add(kw)
            else:
                other_bf16 += bf16_bytes

        to_gb = 1 / (1024 ** 3)
        # NF4 = 0.5 bytes per param, double-quant adds ~12.5% overhead
        nf4_ratio = 0.5 * 1.125 / 2  # relative to BF16 bytes

        return {
            "ssm_bf16": ssm_bf16 * to_gb,
            "ssm_nf4": ssm_bf16 * nf4_ratio * to_gb,
            "ssm_patterns": sorted(ssm_patterns_found),
            "lm_head_bf16": lm_head_bf16 * to_gb,
            "lm_head_nf4": lm_head_bf16 * nf4_ratio * to_gb,
            "lm_head_patterns": ["lm_head"],
            "embed_bf16": embed_bf16 * to_gb,
            "embed_nf4": embed_bf16 * nf4_ratio * to_gb,
            "embed_patterns": ["embed"],
            "other_nf4": other_bf16 * nf4_ratio * to_gb,
        }

    def _report_vram(self, model_name, gpu, quant_label, status_callback):
        """Log VRAM usage after model load and warn about shared memory spill."""
        used_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
        free_gb = gpu["total_gb"] - reserved_gb

        log = logging.getLogger(__name__)
        log.info(
            "VRAM after load: %.1f GB allocated, %.1f GB reserved, "
            "%.1f GB free (of %.1f GB total)",
            used_gb, reserved_gb, free_gb, gpu["total_gb"],
        )

        if gpu["tier"] == "TIGHT" and free_gb < 2.0:
            _warn = (
                f"WARNING: Only {free_gb:.1f} GB VRAM free after loading. "
                f"Inference may spill into shared GPU memory (slow). "
                f"Close other GPU apps to free VRAM."
            )
            log.warning(_warn)
            if status_callback:
                status_callback(_warn)

        if status_callback:
            dtype_label = "bf16" if gpu["compute_dtype"] == torch.bfloat16 else "fp16"
            status_callback(
                f"{model_name} loaded — {used_gb:.1f} / {gpu['total_gb']:.0f} GB VRAM "
                f"({free_gb:.1f} GB free) [{gpu['tier']}  {dtype_label}  {quant_label}]"
            )

    def unload(self, status_callback=None):
        """Unload model and free VRAM."""
        if status_callback:
            status_callback("Unloading model...")

        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if status_callback:
            status_callback("VRAM cleared.")

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_context_size(self) -> int:
        if self.model is not None:
            return getattr(self.model.config, "max_position_embeddings", 0)
        return 0

    def needs_reload(self) -> bool:
        """True when active model changed — in-process model must be swapped."""
        return self._loaded_path != get_active_model_path()

    def periodic_cleanup(self):
        """VRAM cleanup between generations."""
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # =========================================================================
    # BaseEngine — TOKEN COUNTING
    # =========================================================================

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(text) // 4

    # =========================================================================
    # BaseEngine — STREAMING GENERATION
    # =========================================================================

    def generate_streaming(self, messages, max_tokens, temperature,
                           on_token=None, on_complete=None,
                           enable_thinking=True,
                           grammar=None, response_format=None) -> str:
        """Run streaming inference on the local transformers model."""
        self.load()

        tokenizer = self.tokenizer
        model = self.model

        # Expose tokenizer for accurate token counting in build_active_messages
        from core import inference as _inf
        if _inf._tokenizer is None:
            _inf._tokenizer = tokenizer

        # Use processor for multimodal models (Gemma 4), tokenizer otherwise
        template_source = self.processor if self.processor is not None else tokenizer

        inputs = template_source.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        # Real prompt token count — input_ids is (batch, seq_len).
        # Used by API handlers to populate usage.prompt_tokens honestly
        # instead of the old len(content)//4 heuristic.
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        streamer = TextIteratorStreamer(
            self.processor or tokenizer,
            skip_prompt=True, skip_special_tokens=True,
        )

        stop_criteria = StopStringCriteria(tokenizer=tokenizer, stop_strings=STOP_STRINGS)

        # Build EOS token list
        eos_ids = set()
        if tokenizer.eos_token_id is not None:
            if isinstance(tokenizer.eos_token_id, list):
                eos_ids.update(tokenizer.eos_token_id)
            else:
                eos_ids.add(tokenizer.eos_token_id)
        for special in ("<|im_end|>", "<|endoftext|>"):
            tid = tokenizer.convert_tokens_to_ids(special)
            if tid is not None and tid != getattr(tokenizer, "unk_token_id", None):
                eos_ids.add(tid)

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
            if isinstance(pad_id, list):
                pad_id = pad_id[0]

        # KV cache strategy: TurboQuant+ wraps native > quanto > none
        past_kv = None
        gpu = self._detect_gpu_tier()
        _use_turboquant = get_turboquant_kv()
        if _use_turboquant:
            try:
                from core.turboquant_cache import wrap_cache_with_turboquant
                _tq_log = logging.getLogger(__name__)
                native_cache = None
                cache_cls = getattr(model, '_get_cache', None)
                if cache_cls:
                    native_cache = cache_cls()
                    _tq_log.debug("TurboQuant: got cache from model._get_cache() → %s", type(native_cache).__name__)
                if native_cache is None:
                    _tq_log.debug("TurboQuant: _get_cache unavailable, scanning class attributes")
                    for cls_name in dir(type(model)):
                        obj = getattr(type(model), cls_name, None)
                        if isinstance(obj, type) and 'Cache' in cls_name:
                            try:
                                native_cache = obj(config=model.config)
                                _tq_log.debug("TurboQuant: instantiated %s from model class", cls_name)
                                break
                            except Exception as e:
                                _tq_log.debug("TurboQuant: %s failed: %s", cls_name, e)
                                continue
                if native_cache is None:
                    _tq_log.debug("TurboQuant: class scan failed, searching model module for DynamicCache variants")
                    model_module = type(model).__module__
                    import importlib
                    mod = importlib.import_module(model_module)
                    for name in dir(mod):
                        if 'DynamicCache' in name and name != 'DynamicCache':
                            cls = getattr(mod, name)
                            if isinstance(cls, type):
                                try:
                                    native_cache = cls(config=model.config)
                                    _tq_log.debug("TurboQuant: instantiated %s from module", name)
                                    break
                                except Exception as e:
                                    _tq_log.debug("TurboQuant: %s failed: %s", name, e)
                                    continue
                if native_cache is None:
                    _tq_log.info("TurboQuant: all discovery failed, falling back to generic DynamicCache")
                    from transformers.cache_utils import DynamicCache
                    native_cache = DynamicCache()

                num_layers = getattr(model.config, "num_hidden_layers", None)
                past_kv = wrap_cache_with_turboquant(
                    native_cache, num_layers=num_layers,
                    key_bits=3, value_bits=2, residual_length=128, boundary_layers=2,
                )
                logging.getLogger(__name__).info(
                    "Using TurboQuant+ KV cache wrapping %s (K=3-bit, V=2-bit, boundary=2)",
                    type(native_cache).__name__,
                )
            except Exception as e:
                logging.getLogger(__name__).warning("TurboQuant KV cache failed: %s", e)
        if past_kv is None and gpu["tier"] == "TIGHT":
            try:
                from transformers.cache_utils import QuantizedCache
                past_kv = QuantizedCache(
                    backend="quanto",
                    config=model.config,
                    nbits=4,
                    residual_length=128,
                )
            except Exception:
                pass  # quanto optional — falls back to native cache

        effective_max = max_tokens if max_tokens and max_tokens > 0 else 8192

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=effective_max,
            temperature=temperature,
            do_sample=True,
            use_cache=True,
            repetition_penalty=1.15,
            eos_token_id=list(eos_ids) if eos_ids else tokenizer.eos_token_id,
            pad_token_id=pad_id,
            stopping_criteria=StoppingCriteriaList([stop_criteria]),
        )
        if past_kv is not None:
            gen_kwargs["past_key_values"] = past_kv

        gen_error = [None]

        def _safe_generate():
            try:
                model.generate(**gen_kwargs)
            except Exception as e:
                gen_error[0] = e
            finally:
                streamer.end()

        thread = Thread(target=_safe_generate, daemon=True)
        thread.start()

        import time as _time
        full_response = ""
        token_count = 0
        t_first = None
        t_start = _time.perf_counter()

        for new_text in streamer:
            full_response += new_text
            if new_text:
                token_count += 1
                if t_first is None:
                    t_first = _time.perf_counter()
            if on_token and new_text:
                on_token(new_text)

        t_end = _time.perf_counter()
        thread.join()

        # Real completion token count from the tokenizer (streamer chunks are
        # text fragments, not tokens — token_count above is fragment count,
        # not a true token count). Fall back to fragment count if encoding
        # fails for any reason.
        try:
            completion_tokens = len(
                tokenizer.encode(full_response, add_special_tokens=False)
            )
        except Exception:
            completion_tokens = token_count

        # finish_reason: "length" if we hit the generation cap, else "stop"
        # (EOS or StopStringCriteria both count as a natural stop). Uses the
        # real token count since max_new_tokens is measured in tokens, not
        # streamer chunks.
        finish_reason = "length" if completion_tokens >= effective_max else "stop"

        # Log throughput stats
        wall_time = t_end - t_start
        ttft = (t_first - t_start) if t_first else 0
        toks_per_sec = completion_tokens / wall_time if wall_time > 0 else 0
        # Decode-only speed (excludes first-token latency)
        decode_tokens = max(completion_tokens - 1, 0)
        decode_time = (t_end - t_first) if t_first else wall_time
        decode_tps = decode_tokens / decode_time if decode_time > 0 else 0

        log = logging.getLogger(__name__)
        log.info(
            "Generation: %d tokens in %.1fs (%.1f tok/s, TTFT=%.2fs, decode=%.1f tok/s, finish=%s)",
            completion_tokens, wall_time, toks_per_sec, ttft, decode_tps, finish_reason,
        )
        # Store for GUI / API access. prompt_tokens, completion_tokens, and
        # finish_reason are how api/server.py fills in honest usage figures
        # for /v1/chat/completions responses.
        self._last_gen_stats = {
            "tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "wall_time": round(wall_time, 2),
            "tok_per_sec": round(toks_per_sec, 1),
            "ttft": round(ttft, 3),
            "decode_tok_per_sec": round(decode_tps, 1),
        }

        # VRAM cleanup
        del inputs, gen_kwargs, streamer, stop_criteria, thread
        if hasattr(model, '_cache'):
            model._cache = None
        if hasattr(model, 'past_key_values'):
            model.past_key_values = None
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if gen_error[0] is not None:
            raise gen_error[0]

        clean_response = _clean_response(full_response)

        if on_complete:
            on_complete(clean_response)

        return clean_response
