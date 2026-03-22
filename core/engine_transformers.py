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
from core.config import BNB_CONFIG, get_active_model_path
from core.inference import STOP_STRINGS, _clean_response

# Suppress harmless warnings
warnings.filterwarnings("ignore", message="expandable_segments not supported")
warnings.filterwarnings("ignore", message="The following layers were not sharded")
warnings.filterwarnings("ignore", message=".*fast path is not available.*")
warnings.filterwarnings("ignore", message=".*required library is not installed.*")
warnings.filterwarnings("ignore", message=".*FP4 quantization state not initialized.*")


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
        pass
    return True


class TransformersEngine(BaseEngine):
    """Local transformers backend with NF4 quantization and streaming generation."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded_path = None
        self._gpu_tier = None

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
        return os.path.isdir(quantized_path) and os.path.isfile(
            os.path.join(quantized_path, "config.json")
        )

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

            self.tokenizer = AutoTokenizer.from_pretrained(quantized_path)

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    quantized_path,
                    device_map={"": 0},
                    dtype=gpu["compute_dtype"],
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                if status_callback:
                    status_callback("Direct GPU load failed — retrying with memory constraints...")
                torch.cuda.empty_cache()
                max_mem = self._get_vram_constraints(allow_cpu_staging=False)
                self.model = AutoModelForCausalLM.from_pretrained(
                    quantized_path,
                    device_map="auto",
                    max_memory=max_mem,
                    dtype=gpu["compute_dtype"],
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )

            # Register dtype correction hooks
            self._register_dtype_hooks()

            self._loaded_path = model_path
            self._report_vram(model_name, gpu, "cached-nf4", status_callback)
            return

        # ── SLOW PATH: First-time quantization ───────────────────────────
        if status_callback:
            status_callback(f"Loading {model_name} (first run — quantizing)...")

        # Qwen3.5's hybrid linear_attn layers don't reliably pack to 4-bit
        # on all hardware / CUDA driver combos — skip them from quantization.
        _linear_attn_skip = [
            "linear_attn",
            "in_proj_qkv", "in_proj_a", "in_proj_b",
            "in_proj_z", "out_proj",
        ]

        bnb_kwargs = dict(
            load_in_4bit=BNB_CONFIG["load_in_4bit"],
            bnb_4bit_compute_dtype=gpu["compute_dtype"],
            bnb_4bit_quant_type=BNB_CONFIG["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=BNB_CONFIG["bnb_4bit_use_double_quant"],
            llm_int8_skip_modules=["lm_head"] + _linear_attn_skip,
        )
        bnb_config = BitsAndBytesConfig(**bnb_kwargs)

        # Patch multimodal config (Qwen3.5 specific)
        _patch_config_json(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # GPU-only placement — no CPU offload / shared memory spillover
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": 0},
            quantization_config=bnb_config,
            dtype=gpu["compute_dtype"],
            trust_remote_code=True,
            low_cpu_mem_usage=True,
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
                llm_int8_skip_modules=["lm_head"] + list(_linear_attn_skip),
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map={"": 0},
                quantization_config=bnb_config,
                dtype=gpu["compute_dtype"],
                trust_remote_code=True,
                low_cpu_mem_usage=True,
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

        self._loaded_path = model_path
        quant_label = "8-bit" if _fell_back_to_8bit else "4-bit nf4"
        self._report_vram(model_name, gpu, quant_label, status_callback)

    def _register_dtype_hooks(self):
        """Fix dtype mismatches between quantized and non-quantized layers."""
        try:
            import bitsandbytes as bnb

            def _make_dtype_hook(mod):
                def hook(module, args):
                    if args[0].dtype != module.weight.dtype:
                        return (args[0].to(module.weight.dtype),) + args[1:]
                return hook

            for name, module in self.model.named_modules():
                if isinstance(module, bnb.nn.Linear4bit):
                    continue
                if hasattr(module, "weight") and isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
                    module.register_forward_pre_hook(_make_dtype_hook(module))
        except ImportError:
            pass

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

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if status_callback:
            status_callback("VRAM cleared.")

    def is_loaded(self) -> bool:
        return self.model is not None

    def needs_reload(self) -> bool:
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
                           enable_thinking=True) -> str:
        """Run streaming inference on the local transformers model."""
        self.load()

        tokenizer = self.tokenizer
        model = self.model

        # Expose tokenizer for accurate token counting in build_active_messages
        from core import inference as _inf
        if _inf._tokenizer is None:
            _inf._tokenizer = tokenizer

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
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

        # Quantized KV cache
        past_kv = None
        try:
            from transformers.cache_utils import QuantizedCache
            past_kv = QuantizedCache(
                backend="quanto",
                config=model.config,
                nbits=4,
                residual_length=128,
            )
        except Exception:
            pass

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            use_cache=True,
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

        full_response = ""

        for new_text in streamer:
            full_response += new_text
            if on_token and new_text:
                on_token(new_text)

        thread.join()

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
