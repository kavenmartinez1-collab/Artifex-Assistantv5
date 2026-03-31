#!/usr/bin/env python3
"""
GPTQ Calibrated INT4 Quantization for WebGPU Inference
=======================================================

Proper Hessian-based GPTQ quantization that produces 5-10x less error than RTN.
Pure PyTorch implementation — no CUDA compilation needed.

Algorithm (Frantar et al., 2022 — "GPTQ: Accurate Post-Training Quantization"):
  1. Run calibration data through model, capture input activations per linear layer
  2. Build Hessian H = X^T @ X for each linear
  3. Cholesky decompose H^{-1}, quantize columns with error propagation
  4. Pack output in GPTQ v2 format matching the WebGPU matmul_q4.wgsl shader

Output format is identical to quantize_mixed_precision.py — drop-in compatible
with the existing WebGPU engine.

Usage:
  python scripts/quantize_gptq.py --model ./models/qwen3.5-9b --output models/qwen3.5-9b-GPTQ-Int4
  python scripts/quantize_gptq.py --model Qwen/Qwen3.5-9B --device cpu --output models/out
"""
import argparse
import copy
import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

# Force unbuffered output so progress is visible in real-time
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


def log(msg: str):
    """Print with immediate flush."""
    print(msg, flush=True)


# ── GPTQ Core Algorithm ────────────────────────────────────────────────────

class GPTQQuantizer:
    """GPTQ quantization for a single weight matrix.

    Reimplemented from auto_gptq/quantization/gptq.py fasterquant() without
    CUDA dependencies. All operations are pure PyTorch (CPU or GPU).
    """

    def __init__(self, bits=4, group_size=128, percdamp=0.01, blocksize=128, sym=False):
        self.bits = bits
        self.group_size = group_size
        self.percdamp = percdamp
        self.blocksize = blocksize
        self.sym = sym
        self.maxq = 2 ** bits - 1  # 15 for INT4

    def quantize_weight(self, W: torch.Tensor, H: torch.Tensor):
        """Run GPTQ column-wise quantization with error propagation.

        Args:
            W: Weight matrix [out_features, in_features] as float32
            H: Hessian matrix [in_features, in_features] as float32

        Returns:
            (Q, scales, zeros, avg_loss) where:
            - Q: quantized weight [out_features, in_features] as int32 (0-15)
            - scales: [out_features, num_groups] as float32
            - zeros: [out_features, num_groups] as int32 (0-15)
            - avg_loss: scalar quantization loss
        """
        W = W.clone().float()
        rows, columns = W.shape
        dev = W.device

        # Handle dead columns (zero Hessian diagonal = weight doesn't affect output)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # Damping: stabilize Hessian by adding small diagonal
        damp = self.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=dev)
        H[diag, diag] += damp

        # Cholesky decomposition: H → H^{-1} → upper Cholesky of H^{-1}
        try:
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
        except torch.linalg.LinAlgError:
            # Fallback: add more damping if Cholesky fails
            log(f"    WARNING: Cholesky failed, adding extra damping")
            extra_damp = 0.1 * torch.mean(torch.diag(H))
            H_orig = self.H_backup.clone() if hasattr(self, 'H_backup') else H.clone()
            H_orig[diag, diag] += damp + extra_damp
            H_orig = torch.linalg.cholesky(H_orig)
            H_orig = torch.cholesky_inverse(H_orig)
            H = torch.linalg.cholesky(H_orig, upper=True)

        Hinv = H
        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        all_scales = []
        all_zeros = []
        group_size = self.group_size if self.group_size > 0 else columns

        # Block-wise quantization with error propagation
        for i1 in range(0, columns, self.blocksize):
            i2 = min(i1 + self.blocksize, columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # Find quantization parameters for this group
                col_idx = i1 + i
                if col_idx % group_size == 0:
                    # New group: compute scale and zero from current (error-adjusted) weights
                    group_end = min(col_idx + group_size, columns)
                    group_w = W[:, col_idx:group_end]

                    if self.sym:
                        wmax = torch.max(torch.abs(group_w), dim=1).values
                        wmin = -wmax
                    else:
                        wmin = torch.min(group_w, dim=1).values
                        wmax = torch.max(group_w, dim=1).values
                        # Ensure range includes zero
                        wmin = torch.minimum(wmin, torch.zeros_like(wmin))
                        wmax = torch.maximum(wmax, torch.zeros_like(wmax))

                    scale = (wmax - wmin) / self.maxq
                    scale = scale.clamp(min=1e-10)
                    if self.sym:
                        zero = torch.full_like(scale, (self.maxq + 1) / 2)
                    else:
                        zero = torch.round(-wmin / scale).clamp(0, self.maxq)

                    all_scales.append(scale)
                    all_zeros.append(zero)

                # Quantize this column
                q = torch.clamp(torch.round(w / scale) + zero, 0, self.maxq)
                Q1[:, i] = q

                # Dequantize for error computation
                dq = scale * (q - zero)
                Losses1[:, i] = (w - dq) ** 2 / d ** 2

                # Error propagation: adjust remaining columns
                err1 = (w - dq) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            # Propagate block error to remaining columns
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        avg_loss = torch.sum(Losses).item()

        # Stack scales and zeros: [out_features, num_groups]
        scales = torch.stack(all_scales, dim=1)  # [rows, num_groups]
        zeros = torch.stack(all_zeros, dim=1).to(torch.int32)  # [rows, num_groups]
        Q = Q.to(torch.int32)

        return Q, scales, zeros, avg_loss


# ── Weight Packing (matches matmul_q4.wgsl format) ─────────────────────────

def pack_qweight(q_int: torch.Tensor) -> torch.Tensor:
    """Pack [N, K] int4 values into [K//8, N] int32.

    Matches quantize_mixed_precision.py lines 104-108.
    The WebGPU shader reads: qweight[(k/8)*N + n], nibble at (k%8)*4.
    """
    N, K = q_int.shape
    assert K % 8 == 0, f"K={K} not divisible by 8"
    qweight = torch.zeros(K // 8, N, dtype=torch.int32)
    for nibble in range(8):
        k_indices = torch.arange(nibble, K, 8)
        qweight[k_indices // 8] |= (q_int[:, k_indices].T << (nibble * 4))
    return qweight


def pack_scales(scales: torch.Tensor) -> torch.Tensor:
    """Convert [N, num_groups] scales to [num_groups, N] float16.

    Matches quantize_mixed_precision.py line 112.
    """
    return scales.T.contiguous().to(torch.float16)


def pack_qzeros(zeros: torch.Tensor) -> torch.Tensor:
    """Pack [N, num_groups] int4 zeros into [num_groups, N//8] int32.

    Matches quantize_mixed_precision.py lines 114-118.
    """
    N, num_groups = zeros.shape
    assert N % 8 == 0, f"N={N} not divisible by 8"
    zeros_t = zeros.T.contiguous().to(torch.int32)  # [num_groups, N]
    qzeros = torch.zeros(num_groups, N // 8, dtype=torch.int32)
    for nibble in range(8):
        n_indices = torch.arange(nibble, N, 8)
        qzeros[:, n_indices // 8] |= (zeros_t[:, n_indices] << (nibble * 4))
    return qzeros


# ── Architecture Detection ─────────────────────────────────────────────────

# Patterns that should NEVER be quantized
SKIP_PATTERNS = ['norm', 'bias', 'A_log', 'dt_bias', 'conv1d', 'mtp.']

# SSM patterns that must stay BF16 (recurrence is too sensitive for INT4)
SSM_PATTERNS = ['linear_attn']

# Patterns that ARE quantized (FFN + attention projections)
QUANT_PATTERNS = [
    'gate_proj', 'up_proj', 'down_proj',          # FFN
    'q_proj', 'k_proj', 'v_proj', 'o_proj',       # Attention
    'dense', 'gate_up_proj',                        # Phi-style names
]


def should_quantize(name: str, keep_bf16_patterns: list[str]) -> bool:
    """Determine if a weight tensor should be quantized.

    Works for any architecture by pattern matching.
    """
    if not name.endswith('.weight'):
        return False

    # Never quantize norms, biases, SSM-specific tensors
    for pat in SKIP_PATTERNS:
        if pat in name:
            return False

    # Keep specified patterns at BF16
    for pat in keep_bf16_patterns:
        if pat in name:
            return False

    # Check if it's a quantizable projection
    for pat in QUANT_PATTERNS:
        if pat in name:
            return True

    return False


# ── Calibration Data ───────────────────────────────────────────────────────

def load_calibration_data(
    dataset_name: str = "wikitext",
    num_samples: int = 128,
    seq_length: int = 2048,
    tokenizer_path: str = None,
) -> list[torch.Tensor]:
    """Load and tokenize calibration data.

    Returns list of [1, seq_length] token ID tensors.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )

    if dataset_name == "wikitext":
        try:
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n\n".join([t for t in dataset["text"] if t.strip()])
        except ImportError:
            log("WARNING: 'datasets' not installed. Using random calibration data.")
            log("  Install with: pip install datasets")
            log("  For best quality, use: --dataset wikitext")
            # Fallback: random tokens (much worse quality but works)
            samples = []
            for _ in range(num_samples):
                ids = torch.randint(100, tokenizer.vocab_size - 100, (1, seq_length))
                samples.append(ids)
            return samples
    elif os.path.isfile(dataset_name):
        with open(dataset_name, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'wikitext' or a .txt file path.")

    # Tokenize and split into chunks
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    all_ids = encoded["input_ids"][0]

    samples = []
    for i in range(0, len(all_ids) - seq_length, seq_length):
        chunk = all_ids[i:i + seq_length].unsqueeze(0)
        samples.append(chunk)
        if len(samples) >= num_samples:
            break

    if len(samples) < num_samples:
        log(f"  WARNING: Only {len(samples)} calibration samples available "
              f"(requested {num_samples})")

    return samples


# ── Layer-by-Layer Model Processing ────────────────────────────────────────

def collect_hessians_and_quantize(
    model_path: str,
    calibration_samples: list[torch.Tensor],
    quantizer_config: dict,
    keep_bf16_patterns: list[str],
    device: str = "cpu",
    report_mse: bool = False,
) -> dict[str, torch.Tensor]:
    """Run calibration and GPTQ quantization layer-by-layer.

    Uses transformers to load the model, hooks to capture activations,
    then quantizes each linear layer with GPTQ.
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    log("\n[1/3] Loading model...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Load in bfloat16 to CPU — ~18 GB for 9B model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Flatten text_config if needed (Qwen3.5 multimodal)
    hf_config = config.to_dict()
    if hasattr(config, 'text_config'):
        text_cfg = config.text_config.to_dict() if hasattr(config.text_config, 'to_dict') else {}
        hf_config.update(text_cfg)

    num_layers = hf_config.get('num_hidden_layers', 32)
    hidden_size = hf_config.get('hidden_size', 4096)

    log(f"  Model: {hf_config.get('model_type', 'unknown')}")
    log(f"  Layers: {num_layers}, Hidden: {hidden_size}")
    log(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

    # ── Step 2: Identify quantizable layers ──────────────────────────────
    log("\n[2/3] Identifying layers to quantize...")
    target_layers = {}  # name → nn.Linear module
    layer_names = {}    # module id → name

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            full_name = name + ".weight"
            if should_quantize(full_name, keep_bf16_patterns):
                target_layers[name] = module
                layer_names[id(module)] = name

    log(f"  Quantizing: {len(target_layers)} linear layers")
    skip_count = sum(1 for n, m in model.named_modules()
                     if isinstance(m, nn.Linear) and n not in target_layers)
    log(f"  Keeping BF16: {skip_count} linear layers")

    # ── Step 3: Run calibration and collect Hessians ─────────────────────
    log("\n[3/3] Running GPTQ calibration + quantization...")
    log(f"  Calibration samples: {len(calibration_samples)}")
    log(f"  Device: {device}")

    # Hooks to capture input activations (Hessians always stored on CPU)
    hessians = {}  # name → Hessian [K, K] on CPU
    nsamples = {}  # name → sample count

    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0]
            if len(x.shape) == 3:
                x = x.reshape(-1, x.shape[-1])
            x = x.float().cpu()  # Always accumulate Hessian on CPU
            K = x.shape[1]

            if name not in hessians:
                hessians[name] = torch.zeros((K, K), dtype=torch.float32)
                nsamples[name] = 0

            n = x.shape[0]
            hessians[name] *= nsamples[name] / (nsamples[name] + n)
            nsamples[name] += n
            x = math.sqrt(2 / nsamples[name]) * x
            hessians[name] += x.t().matmul(x)

        return hook_fn

    # Register hooks
    hooks = []
    for name, module in target_layers.items():
        h = module.register_forward_hook(make_hook(name))
        hooks.append(h)

    use_gpu = device != "cpu" and torch.cuda.is_available()

    # Find the model's internal structure for layer-by-layer processing
    # Support "model.model.layers" (multimodal) and "model.layers" (text-only)
    inner_model = None
    for attr_path in ['model.model', 'model']:
        obj = model
        for part in attr_path.split('.'):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, 'layers'):
            inner_model = obj
            break

    if use_gpu and inner_model is None:
        log("  WARNING: Could not find layer container, falling back to CPU calibration")
        use_gpu = False

    if use_gpu:
        log(f"  Using GPU-accelerated calibration (layer-by-layer on {device})")
    else:
        log("  Using CPU calibration (slow -- use --device cuda for GPU acceleration)")

    # Run calibration samples
    t0 = time.time()
    gpu_dev = torch.device(device) if use_gpu else torch.device('cpu')

    with torch.no_grad():
        for i, sample in enumerate(calibration_samples):
            log(f"  Calibration: {i+1}/{len(calibration_samples)} "
                f"({time.time() - t0:.1f}s)")
            try:
                if use_gpu:
                    # Manual layer-by-layer forward pass on GPU
                    seq_len = sample.shape[1]

                    # 1. Embedding
                    inner_model.embed_tokens.to(gpu_dev)
                    hidden = inner_model.embed_tokens(sample.to(gpu_dev))
                    inner_model.embed_tokens.to('cpu')

                    # 2. Compute rotary position embeddings (needed by attention layers)
                    position_ids = torch.arange(seq_len, device=gpu_dev).unsqueeze(0)
                    rotary_emb = getattr(inner_model, 'rotary_emb', None)
                    pos_embeds = None
                    if rotary_emb is not None:
                        rotary_emb.to(gpu_dev)
                        # rotary_emb.forward(x, position_ids) -> (cos, sin)
                        pos_embeds = rotary_emb(hidden, position_ids)
                        rotary_emb.to('cpu')
                        # Keep pos_embeds on CPU, move per-layer
                        pos_embeds = tuple(p.to('cpu') for p in pos_embeds)

                    # 3. Each transformer layer
                    for li, layer in enumerate(inner_model.layers):
                        layer.to(gpu_dev)
                        hidden = hidden.to(gpu_dev)
                        # Pass position_embeddings for attention layers
                        kwargs = {}
                        if pos_embeds is not None:
                            kwargs['position_embeddings'] = tuple(p.to(gpu_dev) for p in pos_embeds)
                        out = layer(hidden, **kwargs)
                        if isinstance(out, tuple):
                            hidden = out[0]
                        else:
                            hidden = out
                        hidden = hidden.to('cpu')
                        layer.to('cpu')
                        torch.cuda.empty_cache()
                else:
                    model(sample.to("cpu"))
            except Exception as e:
                log(f"  WARNING: Sample {i} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
            gc.collect()
            if use_gpu:
                torch.cuda.empty_cache()

    log(f"  Calibration done in {time.time() - t0:.1f}s")

    # Remove hooks
    for h in hooks:
        h.remove()

    # ── Step 4: GPTQ quantize each layer ────────────────────────────────
    output_tensors = {}
    gptq = GPTQQuantizer(**quantizer_config)
    total_loss = 0
    total_params = 0

    for idx, (name, module) in enumerate(target_layers.items()):
        W = module.weight.data.float()
        N, K = W.shape  # [out_features, in_features]
        H = hessians.get(name)

        if H is None:
            log(f"  SKIP {name}: no Hessian (layer not reached during calibration)")
            continue

        # Ensure dimensions are compatible with packing
        if K % 8 != 0 or N % 8 != 0:
            log(f"  SKIP {name}: K={K} or N={N} not divisible by 8")
            # Keep as BF16
            output_tensors[f"{name}.weight"] = module.weight.data.to(torch.bfloat16)
            continue

        group_size = quantizer_config.get('group_size', 128)
        if K % group_size != 0:
            # Adjust group size for this layer
            for gs in [64, 32, K]:
                if K % gs == 0:
                    gptq_local = GPTQQuantizer(**{**quantizer_config, 'group_size': gs})
                    log(f"  {name}: adjusted group_size {group_size}→{gs} (K={K})")
                    break
            else:
                gptq_local = gptq
        else:
            gptq_local = gptq

        t1 = time.time()
        Q, scales, zeros, loss = gptq_local.quantize_weight(W, H)
        dt = time.time() - t1
        total_loss += loss
        total_params += N * K

        # Report quality
        if report_mse:
            dq = scales.unsqueeze(2) * (Q.reshape(N, -1, gptq_local.group_size).float() -
                                         zeros.unsqueeze(2).float())
            dq = dq.reshape(N, K)
            mse = ((W - dq) ** 2).mean().item()
            rel_mse = mse / ((W ** 2).mean().item() + 1e-10)
            log(f"  [{idx+1}/{len(target_layers)}] {name}: "
                  f"[{N}×{K}] MSE={mse:.8f} Rel={rel_mse:.4%} ({dt:.1f}s)")
        else:
            log(f"  [{idx+1}/{len(target_layers)}] {name}: [{N}×{K}] loss={loss:.4f} ({dt:.1f}s)")

        # Pack into GPTQ v2 format
        qweight = pack_qweight(Q)
        scales_packed = pack_scales(scales)
        qzeros = pack_qzeros(zeros)

        base_name = name
        output_tensors[f"{base_name}.qweight"] = qweight
        output_tensors[f"{base_name}.scales"] = scales_packed
        output_tensors[f"{base_name}.qzeros"] = qzeros

        # Free Hessian memory
        del hessians[name]
        gc.collect()

    # ── Step 5: Collect non-quantized tensors ────────────────────────────
    log("\n  Collecting non-quantized tensors...")
    for name, param in model.named_parameters():
        base = name.rsplit('.', 1)[0] if '.' in name else name
        # Skip if we already quantized this
        if base in target_layers:
            continue
        # Skip MTP layers
        if 'mtp.' in name:
            continue
        output_tensors[name] = param.data.to(torch.bfloat16)

    # Free model memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log(f"\n  Total GPTQ loss: {total_loss:.4f} "
          f"({total_params / 1e6:.1f}M quantized parameters)")

    return output_tensors


# ── Output ─────────────────────────────────────────────────────────────────

def save_quantized_model(
    output_tensors: dict[str, torch.Tensor],
    model_path: str,
    output_path: str,
    quantizer_config: dict,
    keep_bf16_patterns: list[str],
):
    """Save quantized model as SafeTensors with config."""
    out_path = Path(output_path)
    os.makedirs(out_path, exist_ok=True)

    # Calculate total size
    total_bytes = sum(t.numel() * t.element_size() for t in output_tensors.values())
    log(f"\n  Total output: {total_bytes / (1024**3):.2f} GB, "
          f"{len(output_tensors)} tensors")

    # Split into shards (4 GB max per shard)
    SHARD_LIMIT = 4 * 1024 * 1024 * 1024
    sorted_names = sorted(output_tensors.keys())
    shards = []
    current_shard = {}
    current_size = 0

    for name in sorted_names:
        tensor = output_tensors[name]
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > SHARD_LIMIT and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[name] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    weight_map = {}
    for i, shard_tensors in enumerate(shards):
        if len(shards) == 1:
            filename = "model.safetensors"
        else:
            filename = f"model.safetensors-{i+1:05d}-of-{len(shards):05d}.safetensors"
        filepath = out_path / filename
        log(f"  Writing {filename} ({len(shard_tensors)} tensors)...")
        save_file(shard_tensors, str(filepath))
        for name in shard_tensors:
            weight_map[name] = filename

    if len(shards) > 1:
        index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
        with open(out_path / "model.safetensors.index.json", 'w') as f:
            json.dump(index, f, indent=2)

    # Copy tokenizer and metadata
    src_path = Path(model_path)
    for cfg_file in ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                     'merges.txt', 'vocab.json', 'preprocessor_config.json',
                     'special_tokens_map.json', 'generation_config.json']:
        src = src_path / cfg_file
        if src.exists():
            shutil.copy2(str(src), str(out_path / cfg_file))

    # Build config.json with quantization metadata
    config_src = src_path / 'config.json'
    with open(config_src) as f:
        config = json.load(f)

    config['quantization_config'] = {
        "quant_method": "gptq",
        "bits": quantizer_config['bits'],
        "group_size": quantizer_config['group_size'],
        "checkpoint_format": "gptq_v2",
        "mixed_precision": bool(keep_bf16_patterns),
        "calibration": "gptq",
        "sym": quantizer_config.get('sym', False),
        "modules_not_quantized": keep_bf16_patterns,
    }

    with open(out_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    log(f"\n  Model saved to {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPTQ Calibrated INT4 Quantization for WebGPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize local model:
  python scripts/quantize_gptq.py --model ./models/qwen3.5-9b --output models/qwen3.5-9b-GPTQ

  # Quantize from HuggingFace:
  python scripts/quantize_gptq.py --model Qwen/Qwen3.5-9B --output models/qwen3.5-9b-GPTQ

  # With MSE reporting:
  python scripts/quantize_gptq.py --model ./models/qwen3.5-9b --output models/out --report-mse
        """)
    parser.add_argument("--model", required=True, help="HF repo ID or local model path")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits (default: 4)")
    parser.add_argument("--group-size", type=int, default=128, help="Group size (default: 128)")
    parser.add_argument("--num-samples", type=int, default=128, help="Calibration samples (16 is fast, 128 is standard)")
    parser.add_argument("--seq-length", type=int, default=2048, help="Calibration sequence length")
    parser.add_argument("--dataset", default="wikitext", help="'wikitext' or path to .txt file")
    parser.add_argument("--percdamp", type=float, default=0.01, help="GPTQ damping factor")
    parser.add_argument("--blocksize", type=int, default=128, help="GPTQ block size")
    parser.add_argument("--device", default="cuda", help="Device: cpu or cuda (cuda uses layer-by-layer GPU acceleration)")
    parser.add_argument("--keep-bf16", nargs="*",
                        default=["linear_attn", "norm", "embed_tokens", "lm_head"],
                        help="Module patterns to keep at BF16")
    parser.add_argument("--sym", action="store_true", help="Use symmetric quantization")
    parser.add_argument("--report-mse", action="store_true", help="Report per-layer MSE")
    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path(args.model).name) + "-GPTQ-Int4"

    log(f"{'='*60}")
    log(f"GPTQ Calibrated INT4 Quantization")
    log(f"{'='*60}")
    log(f"Model:        {args.model}")
    log(f"Output:       {args.output}")
    log(f"Bits:         {args.bits}")
    log(f"Group size:   {args.group_size}")
    log(f"Calibration:  {args.num_samples} samples × {args.seq_length} tokens")
    log(f"Dataset:      {args.dataset}")
    log(f"Device:       {args.device}")
    log(f"Keep BF16:    {args.keep_bf16}")
    log(f"Symmetric:    {args.sym}")
    log(f"{'='*60}")

    # Load calibration data
    log("\nLoading calibration data...")
    tokenizer_path = args.model
    calibration_samples = load_calibration_data(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        tokenizer_path=tokenizer_path,
    )
    log(f"  Loaded {len(calibration_samples)} calibration samples")

    # Quantize
    quantizer_config = {
        'bits': args.bits,
        'group_size': args.group_size,
        'percdamp': args.percdamp,
        'blocksize': args.blocksize,
        'sym': args.sym,
    }

    t_start = time.time()
    output_tensors = collect_hessians_and_quantize(
        model_path=args.model,
        calibration_samples=calibration_samples,
        quantizer_config=quantizer_config,
        keep_bf16_patterns=args.keep_bf16,
        device=args.device,
        report_mse=args.report_mse,
    )
    t_quant = time.time() - t_start

    # Save
    save_quantized_model(
        output_tensors=output_tensors,
        model_path=args.model,
        output_path=args.output,
        quantizer_config=quantizer_config,
        keep_bf16_patterns=args.keep_bf16,
    )

    log(f"\n{'='*60}")
    log(f"GPTQ quantization complete in {t_quant / 60:.1f} minutes")
    log(f"Output: {args.output}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
