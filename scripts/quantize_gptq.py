#!/usr/bin/env python3
"""
GPTQ Calibrated INT4 Quantization for WebGPU Inference (v2)
=============================================================

Layer-by-layer GPTQ with activation ordering, percentile clipping,
and interleaved calibration/quantization for memory efficiency.

Based on Frantar et al., 2022 — "GPTQ: Accurate Post-Training Quantization"
Pure PyTorch implementation — no CUDA compilation needed.

Key improvements over v1:
  - Layer-by-layer processing: calibrate → quantize → propagate (not all-at-once)
  - Actorder: columns sorted by Hessian diagonal for better error propagation
  - Percentile clipping: outlier-robust scale computation
  - Fixed Hessian accumulation numerics (sample counting, TF32 disabled)
  - True sequential: sublayer groups quantized in dependency order
  - ~37 min on 8GB GPU (vs ~7 hours in v1)

Output format: GPTQ v2 SafeTensors matching matmul_q4.wgsl shader layout.
Adds g_idx tensor per layer when actorder is enabled.

Usage:
  python scripts/quantize_gptq.py --model ./models/qwen3.5-9b --output models/qwen3.5-9b-GPTQ
  python scripts/quantize_gptq.py --model Qwen/Qwen3.5-9B --device cuda --output models/out
  python scripts/quantize_gptq.py --model ./models/qwen3.5-9b --no-actorder --output models/out
"""
import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path

# Force unbuffered output so progress is visible in real-time
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
from safetensors.torch import save_file

# Disable TF32 for full float32 precision in Hessian matmuls
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def log(msg: str):
    """Print with immediate flush."""
    print(msg, flush=True)


# ── GPTQ Core Algorithm ────────────────────────────────────────────────────

class GPTQQuantizer:
    """GPTQ quantization with actorder and percentile clipping.

    Improvements over standard GPTQ:
    - Actorder: sort columns by Hessian diagonal (importance) before quantizing.
      Important columns get quantized first with less accumulated error.
    - Percentile clipping: exclude 0.1% outliers from scale computation,
      matching RTN+'s key quality advantage.
    - Fixed Hessian: accumulate-and-divide (no running-average drift).
    """

    def __init__(self, bits=4, group_size=128, percdamp=0.01, blocksize=128,
                 sym=False, actorder=True, percclip=0.001):
        self.bits = bits
        self.group_size = group_size
        self.percdamp = percdamp
        self.blocksize = blocksize
        self.sym = sym
        self.actorder = actorder
        self.percclip = percclip
        self.maxq = 2 ** bits - 1  # 15 for INT4

    def quantize_weight(self, W: torch.Tensor, H: torch.Tensor):
        """Run GPTQ column-wise quantization with error propagation.

        Args:
            W: Weight matrix [out_features, in_features] as float32
            H: Hessian matrix [in_features, in_features] as float32
                (already normalized as 2/n * X^T @ X)

        Returns:
            (Q_int, scales, zeros, g_idx, Q_deq, avg_loss) where:
            - Q_int: [out_features, in_features] int32 (0-15) in original col order
            - scales: [out_features, num_groups] float32
            - zeros: [out_features, num_groups] int32 (0-15)
            - g_idx: [in_features] int32 — column-to-group mapping
            - Q_deq: [out_features, in_features] float32 dequantized weight
            - avg_loss: scalar quantization loss
        """
        W = W.clone().float()
        rows, columns = W.shape
        dev = W.device

        # Handle dead columns (zero Hessian diagonal = never activated)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # Actorder: sort columns by importance (descending Hessian diagonal)
        perm = None
        if self.actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        # Damping: stabilize Hessian by adding small diagonal
        damp = self.percdamp * torch.mean(torch.diag(H))
        diag_idx = torch.arange(columns, device=dev)
        H[diag_idx, diag_idx] += damp

        # Cholesky: H → chol(H) → H^{-1} → upper chol(H^{-1})
        H_backup = H.clone()
        try:
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
        except torch.linalg.LinAlgError:
            log("    WARNING: Cholesky failed, adding extra damping")
            extra = 0.1 * torch.mean(torch.diag(H_backup))
            H_backup[diag_idx, diag_idx] += extra
            H = torch.linalg.cholesky(H_backup)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
        del H_backup

        Hinv = H
        Losses = torch.zeros_like(W)
        Q = torch.zeros(rows, columns, dtype=torch.int32, device=dev)

        group_size = self.group_size if self.group_size > 0 else columns
        num_groups = (columns + group_size - 1) // group_size
        all_scales = torch.zeros(rows, num_groups, device=dev)
        all_zeros = torch.zeros(rows, num_groups, device=dev)

        scale = None
        zero = None

        # Block-wise quantization with error propagation
        for i1 in range(0, columns, self.blocksize):
            i2 = min(i1 + self.blocksize, columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros(rows, count, dtype=torch.int32, device=dev)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                col_idx = i1 + i
                group_idx = col_idx // group_size

                if col_idx % group_size == 0:
                    # Compute scale/zero for this group
                    group_end = min(col_idx + group_size, columns)
                    group_w = W[:, col_idx:group_end]

                    # Percentile clipping: exclude outliers from range
                    if self.percclip > 0 and group_w.shape[1] > 1:
                        lo = torch.quantile(group_w, self.percclip, dim=1)
                        hi = torch.quantile(group_w, 1.0 - self.percclip, dim=1)
                    else:
                        lo = group_w.min(dim=1).values
                        hi = group_w.max(dim=1).values

                    if self.sym:
                        maxabs = torch.max(lo.abs(), hi.abs())
                        lo = -maxabs
                        hi = maxabs

                    # Ensure range includes zero
                    lo = torch.minimum(lo, torch.zeros_like(lo))
                    hi = torch.maximum(hi, torch.zeros_like(hi))

                    # Handle all-zero groups (avoid division by zero)
                    dead_g = (lo == 0) & (hi == 0)
                    lo[dead_g] = -1.0
                    hi[dead_g] = 1.0

                    scale = (hi - lo) / self.maxq
                    if self.sym:
                        zero = torch.full_like(scale, (self.maxq + 1) / 2)
                    else:
                        zero = torch.round(-lo / scale).clamp(0, self.maxq)

                    all_scales[:, group_idx] = scale
                    all_zeros[:, group_idx] = zero

                # Quantize this column
                q = torch.clamp(torch.round(w / scale) + zero, 0, self.maxq)
                Q1[:, i] = q.to(torch.int32)

                # Dequantize for error computation
                dq = scale * (q - zero)
                Losses1[:, i] = (w - dq) ** 2 / d ** 2

                # Error propagation to remaining columns in block
                err1 = (w - dq) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            # Propagate block error to remaining columns
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        avg_loss = torch.sum(Losses).item()

        # Build g_idx and un-permute if actorder
        if self.actorder:
            invperm = torch.argsort(perm)
            # Un-permute Q back to original column order
            Q = Q[:, invperm]
            # g_idx[original_col] = group in reordered space
            # invperm[j] = reordered position of original column j
            g_idx = (invperm // group_size).to(torch.int32)
        else:
            g_idx = (torch.arange(columns, device=dev) // group_size).to(torch.int32)

        # Dequantized weight (for replacing nn.Linear weight in-place)
        scales_per_col = all_scales[:, g_idx.long()]
        zeros_per_col = all_zeros[:, g_idx.long()]
        Q_deq = scales_per_col * (Q.float() - zeros_per_col)

        all_zeros = all_zeros.to(torch.int32)

        return Q, all_scales, all_zeros, g_idx, Q_deq, avg_loss


# ── Hessian Collection ────────────────────────────────────────────────────

class HessianCollector:
    """Accumulates raw X^T @ X for a single Linear layer.

    Uses accumulate-and-divide: stores raw sum, normalizes once at the end.
    Counts batches (not tokens) to match auto_gptq semantics.

    v1 bug fix: v1 counted tokens (B*seq) instead of batches (B), making
    the Hessian ~2048x too small and degrading GPTQ to plain RTN.
    """

    def __init__(self, layer: nn.Linear, device: torch.device):
        K = layer.in_features
        self.H = torch.zeros((K, K), dtype=torch.float32, device=device)
        self.nsamples = 0

    def hook(self, module, inp, out):
        """Forward hook: accumulate X^T @ X from layer input activations."""
        x = inp[0]
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]  # Count batches BEFORE reshape (v1 bug fix)
        x = x.reshape(-1, x.shape[-1]).float().to(self.H.device)
        self.H += x.t() @ x
        self.nsamples += batch_size

    def get_hessian(self) -> torch.Tensor:
        """Return normalized Hessian: H = (2 / n_samples) * X^T X"""
        if self.nsamples > 0:
            return (2.0 / self.nsamples) * self.H
        return self.H

    def free(self):
        """Release Hessian memory immediately after quantization."""
        del self.H
        self.H = None


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
    zeros_t = zeros.T.contiguous().to(torch.int32)
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
    """Determine if a weight tensor should be quantized."""
    if not name.endswith('.weight'):
        return False
    for pat in SKIP_PATTERNS:
        if pat in name:
            return False
    for pat in keep_bf16_patterns:
        if pat in name:
            return False
    for pat in QUANT_PATTERNS:
        if pat in name:
            return True
    return False


# ── Sublayer Grouping (True Sequential) ────────────────────────────────────

# Sublayers that share the same input are grouped together.
# Groups are processed in order: after quantizing group N, the forward
# pass for group N+1's Hessian sees quantized activations from group N.
SUBLAYER_GROUP_ORDER = [
    ['q_proj', 'k_proj', 'v_proj'],   # Attention inputs (parallel, same input)
    ['o_proj'],                         # Attention output
    ['gate_proj', 'up_proj'],           # FFN inputs (parallel, same input)
    ['down_proj'],                      # FFN output
]


def group_sublayers(sublayer_names: list[str]) -> list[list[str]]:
    """Group sublayer names for true-sequential processing."""
    groups = []
    assigned = set()
    for patterns in SUBLAYER_GROUP_ORDER:
        group = []
        for name in sublayer_names:
            for pat in patterns:
                if pat in name and name not in assigned:
                    group.append(name)
                    assigned.add(name)
                    break
        if group:
            groups.append(group)
    # Catch any unmatched sublayers
    remaining = [n for n in sublayer_names if n not in assigned]
    if remaining:
        groups.append(remaining)
    return groups


# ── Calibration Data ───────────────────────────────────────────────────────

def load_calibration_data(
    dataset_name: str = "wikitext",
    num_samples: int = 128,
    seq_length: int = 2048,
    tokenizer_path: str = None,
) -> list[torch.Tensor]:
    """Load and tokenize calibration data.

    Returns list of [1, seq_length] token ID tensors.
    Uses train split for wikitext (test split only has ~245K tokens,
    not enough for 128 x 2048 = 262K).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )

    if dataset_name == "wikitext":
        try:
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            text = "\n\n".join([t for t in dataset["text"] if t.strip()])
        except ImportError:
            log("WARNING: 'datasets' not installed. Using random calibration data.")
            log("  Install with: pip install datasets")
            log("  For best quality, use: --dataset wikitext")
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


# ── Model Inspection ───────────────────────────────────────────────────────

def find_inner_model(model):
    """Find the inner model containing .layers and .embed_tokens."""
    for attr_path in ['model.model', 'model']:
        obj = model
        for part in attr_path.split('.'):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, 'layers'):
            return obj
    raise RuntimeError("Cannot find model.layers — unsupported architecture")


def find_layer_prefix(model) -> str:
    """Detect the parameter name prefix for transformer layers."""
    for name, _ in model.named_parameters():
        if '.layers.0.' in name:
            return name.split('.layers.0.')[0] + '.layers'
    return 'model.layers'


# ── Layer-by-Layer GPTQ Engine ─────────────────────────────────────────────

def quantize_layer_by_layer(
    model_path: str,
    calibration_samples: list[torch.Tensor],
    quantizer_config: dict,
    keep_bf16_patterns: list[str],
    device: str = "cuda",
    report_mse: bool = False,
) -> dict[str, torch.Tensor]:
    """Layer-by-layer GPTQ with interleaved calibration and quantization.

    For each layer:
      1. Move to GPU
      2. For each sublayer group (true sequential):
         a. Forward all samples — hooks accumulate Hessians
         b. GPTQ quantize, replace weight with dequantized version
         c. Free Hessian immediately
      3. Re-run all samples to get outputs for next layer
      4. Move back to CPU

    This ensures each layer's Hessians reflect the quantized-so-far model,
    not the original BF16 model. Only one layer's Hessians exist at a time.

    Memory: ~2.5 GB peak VRAM for 9B model on 8GB GPU.
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    log("\n[1/4] Loading model...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

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

    inner_model = find_inner_model(model)
    layer_prefix = find_layer_prefix(model)
    layers = inner_model.layers

    use_gpu = device != "cpu" and torch.cuda.is_available()
    gpu = torch.device(device if use_gpu else "cpu")

    # ── Step 2: Compute initial hidden states ──────────────────────────────
    log(f"\n[2/4] Computing initial hidden states "
        f"({len(calibration_samples)} samples)...")
    layer_inputs = []

    inner_model.embed_tokens.to(gpu)
    with torch.no_grad():
        for i, sample in enumerate(calibration_samples):
            hidden = inner_model.embed_tokens(sample.to(gpu))
            layer_inputs.append(hidden.to(torch.bfloat16).cpu())
            if (i + 1) % 32 == 0:
                log(f"  Embedded {i+1}/{len(calibration_samples)}")
    inner_model.embed_tokens.to('cpu')
    if use_gpu:
        torch.cuda.empty_cache()
    log(f"  Done — {len(layer_inputs)} hidden states cached")

    # Compute rotary position embeddings (shared across all layers)
    seq_len = calibration_samples[0].shape[1]
    pos_embeds = None
    rotary_emb = getattr(inner_model, 'rotary_emb', None)
    if rotary_emb is not None:
        rotary_emb.to(gpu)
        position_ids = torch.arange(seq_len, device=gpu).unsqueeze(0)
        with torch.no_grad():
            pos_embeds = rotary_emb(layer_inputs[0].to(gpu), position_ids)
        pos_embeds = tuple(p.cpu() for p in pos_embeds)
        rotary_emb.to('cpu')
        if use_gpu:
            torch.cuda.empty_cache()

    def layer_kwargs():
        kw = {}
        if pos_embeds is not None:
            kw['position_embeddings'] = tuple(p.to(gpu) for p in pos_embeds)
        return kw

    # ── Step 3: Layer-by-layer GPTQ ───────────────────────────────────────
    log(f"\n[3/4] GPTQ quantization ({len(layers)} layers)...")
    output_tensors = {}
    gptq = GPTQQuantizer(**quantizer_config)
    total_loss = 0
    total_params = 0
    quant_count = 0
    t_start = time.time()

    for li in range(len(layers)):
        layer = layers[li]
        layer.to(gpu)
        t_layer = time.time()

        # Find quantizable sublayers in this layer
        sublayers = {}
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"{layer_prefix}.{li}.{name}.weight"
                if should_quantize(full_name, keep_bf16_patterns):
                    sublayers[name] = module

        quantized_modules = set()

        if sublayers:
            groups = group_sublayers(list(sublayers.keys()))

            for group_names in groups:
                group_modules = {n: sublayers[n] for n in group_names
                                 if n in sublayers}
                if not group_modules:
                    continue

                # Register Hessian collection hooks
                collectors = {}
                hooks = []
                for name, module in group_modules.items():
                    col = HessianCollector(module, gpu)
                    collectors[name] = col
                    h = module.register_forward_hook(col.hook)
                    hooks.append(h)

                # Forward all calibration samples through the full layer
                # (hooks accumulate X^T @ X for this group's sublayers)
                kw = layer_kwargs()
                with torch.no_grad():
                    for inp in layer_inputs:
                        try:
                            layer(inp.to(gpu), **kw)
                        except Exception as e:
                            log(f"    WARNING: Forward failed: {e}")
                            continue

                for h in hooks:
                    h.remove()

                # GPTQ quantize each sublayer in the group
                for name, module in group_modules.items():
                    W = module.weight.data.float().to(gpu)
                    N, K = W.shape

                    H = collectors[name].get_hessian()
                    collectors[name].free()

                    # Skip layers with incompatible dimensions
                    if K % 8 != 0 or N % 8 != 0:
                        log(f"    SKIP {name}: K={K}/N={N} not div by 8")
                        del W, H
                        continue

                    # Adjust group size if K not divisible
                    gs = quantizer_config.get('group_size', 128)
                    if K % gs != 0:
                        local_cfg = quantizer_config.copy()
                        for alt_gs in [64, 32, K]:
                            if K % alt_gs == 0:
                                local_cfg['group_size'] = alt_gs
                                log(f"    {name}: group_size {gs}→{alt_gs}")
                                break
                        local_gptq = GPTQQuantizer(**local_cfg)
                    else:
                        local_gptq = gptq

                    Q_int, scales, zeros, g_idx, Q_deq, loss = \
                        local_gptq.quantize_weight(W, H)

                    total_loss += loss
                    total_params += N * K
                    quant_count += 1

                    # Report quality
                    if report_mse:
                        mse = ((W - Q_deq) ** 2).mean().item()
                        rel = mse / ((W ** 2).mean().item() + 1e-10)
                        log(f"    [{quant_count}] {name} [{N}x{K}] "
                            f"MSE={mse:.8f} Rel={rel:.4%}")
                    else:
                        log(f"    [{quant_count}] {name} [{N}x{K}] "
                            f"loss={loss:.4f}")

                    # Pack for output (before replacing weight)
                    base = f"{layer_prefix}.{li}.{name}"
                    output_tensors[f"{base}.qweight"] = \
                        pack_qweight(Q_int.cpu())
                    output_tensors[f"{base}.scales"] = \
                        pack_scales(scales.cpu())
                    output_tensors[f"{base}.qzeros"] = \
                        pack_qzeros(zeros.cpu())
                    output_tensors[f"{base}.g_idx"] = g_idx.cpu()

                    # Replace weight with dequantized approximation
                    # so subsequent forward passes see quantized activations
                    module.weight.data = Q_deq.to(module.weight.dtype)
                    quantized_modules.add(name)

                    del W, H, Q_int, scales, zeros, g_idx, Q_deq
                    if use_gpu:
                        torch.cuda.empty_cache()

        # Collect non-quantized parameters from this layer
        for pname, param in layer.named_parameters():
            # Skip .weight of quantized modules (packed as qweight already)
            module_base = pname.rsplit('.', 1)[0] if '.' in pname else ''
            if module_base in quantized_modules and pname.endswith('.weight'):
                continue
            full_name = f"{layer_prefix}.{li}.{pname}"
            if 'mtp.' not in full_name:
                output_tensors[full_name] = param.data.cpu().to(torch.bfloat16)

        # Re-run all samples through quantized layer to get outputs
        # for the next layer's calibration
        layer_outputs = []
        kw = layer_kwargs()
        with torch.no_grad():
            for inp in layer_inputs:
                try:
                    out = layer(inp.to(gpu), **kw)
                    if isinstance(out, tuple):
                        out = out[0]
                    layer_outputs.append(out.to(torch.bfloat16).cpu())
                except Exception as e:
                    log(f"    WARNING: Re-run failed: {e}")
                    layer_outputs.append(inp)

        layer.to('cpu')
        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()

        layer_inputs = layer_outputs
        dt = time.time() - t_layer
        log(f"  Layer {li}/{len(layers)-1} done ({dt:.1f}s, "
            f"{len(sublayers)} quantized)")

    # ── Step 4: Quantize global weights (embed_tokens, lm_head) ─────────
    log("\n[4/4] Processing global parameters...")
    layer_param_prefix = f"{layer_prefix}."

    # Helper: RTN quantize a weight matrix (for embeddings — no Hessian needed)
    def rtn_quantize(W: torch.Tensor, name: str, gs: int = 128):
        """Round-to-nearest INT4 quantization with percentile clipping."""
        W = W.float()
        N, K = W.shape
        if K % 8 != 0 or N % 8 != 0:
            log(f"    SKIP {name}: K={K}/N={N} not div by 8")
            return None
        if K % gs != 0:
            for alt_gs in [64, 32, K]:
                if K % alt_gs == 0:
                    gs = alt_gs
                    break
        num_groups = K // gs
        maxq = 2 ** quantizer_config['bits'] - 1
        percclip = quantizer_config.get('percclip', 0.001)

        all_scales = torch.zeros(N, num_groups)
        all_zeros = torch.zeros(N, num_groups, dtype=torch.int32)
        Q = torch.zeros(N, K, dtype=torch.int32)

        for g in range(num_groups):
            g_start = g * gs
            g_end = g_start + gs
            group_w = W[:, g_start:g_end]

            if percclip > 0 and gs > 1:
                lo = torch.quantile(group_w, percclip, dim=1)
                hi = torch.quantile(group_w, 1.0 - percclip, dim=1)
            else:
                lo = group_w.min(dim=1).values
                hi = group_w.max(dim=1).values

            lo = torch.minimum(lo, torch.zeros_like(lo))
            hi = torch.maximum(hi, torch.zeros_like(hi))
            dead_g = (lo == 0) & (hi == 0)
            lo[dead_g] = -1.0
            hi[dead_g] = 1.0

            scale = (hi - lo) / maxq
            zero = torch.round(-lo / scale).clamp(0, maxq)
            all_scales[:, g] = scale
            all_zeros[:, g] = zero.to(torch.int32)

            for k in range(gs):
                w = group_w[:, k]
                q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
                Q[:, g_start + k] = q.to(torch.int32)

        g_idx = (torch.arange(K) // gs).to(torch.int32)
        return Q, all_scales, all_zeros, g_idx

    # Helper: GPTQ quantize a global linear layer using cached activations
    def gptq_quantize_global(module: nn.Linear, name: str,
                             activations: list[torch.Tensor]):
        """GPTQ quantize a global weight using provided activations."""
        W = module.weight.data.float().to(gpu)
        N, K = W.shape
        if K % 8 != 0 or N % 8 != 0:
            log(f"    SKIP {name}: K={K}/N={N} not div by 8")
            return None

        # Build Hessian from activations
        H = torch.zeros((K, K), dtype=torch.float32, device=gpu)
        n_samples = 0
        for act in activations:
            x = act.to(gpu).float()
            if len(x.shape) == 3:
                x = x.reshape(-1, x.shape[-1])
            H += x.t() @ x
            n_samples += 1
        H = (2.0 / n_samples) * H

        gs = quantizer_config.get('group_size', 128)
        if K % gs != 0:
            local_cfg = quantizer_config.copy()
            for alt_gs in [64, 32, K]:
                if K % alt_gs == 0:
                    local_cfg['group_size'] = alt_gs
                    break
            local_gptq = GPTQQuantizer(**local_cfg)
        else:
            local_gptq = gptq

        Q_int, scales, zeros, g_idx, Q_deq, loss = \
            local_gptq.quantize_weight(W, H)
        return Q_int, scales, zeros, g_idx, Q_deq, loss

    # Check which global params should be quantized
    keep_embed_bf16 = any(p in 'embed_tokens' for p in keep_bf16_patterns)
    keep_lmhead_bf16 = any(p in 'lm_head' for p in keep_bf16_patterns)

    # Find embed_tokens and lm_head
    embed_module = inner_model.embed_tokens
    lm_head_module = None
    lm_head_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'lm_head' in name:
            lm_head_module = module
            lm_head_name = name

    # Quantize lm_head with GPTQ on CPU (too large for GPU — [248K, 4096]
    # creates ~3.7 GB intermediates during error propagation, but GPTQ's
    # within-matrix error compensation still improves quality over RTN)
    if lm_head_module is not None and not keep_lmhead_bf16:
        log("  Quantizing lm_head with GPTQ (on CPU, large vocab)...")
        # Build Hessian from final hidden states (after norm)
        final_norm = getattr(inner_model, 'norm', None)
        if final_norm is not None:
            final_norm.to(gpu)
            normed_inputs = []
            with torch.no_grad():
                for inp in layer_inputs:
                    normed = final_norm(inp.to(gpu))
                    normed_inputs.append(normed.to(torch.bfloat16).cpu())
            final_norm.to('cpu')
            if use_gpu:
                torch.cuda.empty_cache()
        else:
            normed_inputs = layer_inputs

        # Accumulate Hessian on CPU
        K = lm_head_module.in_features
        H = torch.zeros((K, K), dtype=torch.float32)
        n_samples = 0
        for act in normed_inputs:
            x = act.float()
            if len(x.shape) == 3:
                x = x.reshape(-1, x.shape[-1])
            H += x.t() @ x
            n_samples += 1
        H = (2.0 / n_samples) * H
        del normed_inputs

        # Run GPTQ on CPU (avoids OOM from large vocab intermediates)
        W = lm_head_module.weight.data.float()  # already on CPU
        N = W.shape[0]

        gs = quantizer_config.get('group_size', 128)
        if K % gs != 0:
            local_cfg = quantizer_config.copy()
            for alt_gs in [64, 32, K]:
                if K % alt_gs == 0:
                    local_cfg['group_size'] = alt_gs
                    break
            local_gptq = GPTQQuantizer(**local_cfg)
        else:
            local_gptq = gptq

        t_lm = time.time()
        Q_int, scales, zeros, g_idx, Q_deq, loss = \
            local_gptq.quantize_weight(W, H)
        dt_lm = time.time() - t_lm

        quant_count += 1
        total_params += N * K
        total_loss += loss
        if report_mse:
            mse = ((W - Q_deq) ** 2).mean().item()
            rel = mse / ((W ** 2).mean().item() + 1e-10)
            log(f"    [{quant_count}] lm_head [{N}x{K}] "
                f"MSE={mse:.8f} Rel={rel:.4%} ({dt_lm:.1f}s, CPU)")
        else:
            log(f"    [{quant_count}] lm_head [{N}x{K}] "
                f"loss={loss:.4f} ({dt_lm:.1f}s, CPU)")

        output_tensors[f"{lm_head_name}.qweight"] = pack_qweight(Q_int)
        output_tensors[f"{lm_head_name}.scales"] = pack_scales(scales)
        output_tensors[f"{lm_head_name}.qzeros"] = pack_qzeros(zeros)
        output_tensors[f"{lm_head_name}.g_idx"] = g_idx
        del W, H, Q_int, scales, zeros, g_idx, Q_deq

    # Quantize embed_tokens with RTN (lookup table — no Hessian)
    if not keep_embed_bf16:
        log("  Quantizing embed_tokens with RTN...")
        W = embed_module.weight.data.float()
        N, K = W.shape
        result = rtn_quantize(W, 'embed_tokens',
                              quantizer_config.get('group_size', 128))
        if result is not None:
            Q_int, scales, zeros, g_idx = result
            quant_count += 1
            total_params += N * K
            Q_deq_scales = scales[:, g_idx.long()]
            Q_deq_zeros = zeros[:, g_idx.long()].float()
            Q_deq = Q_deq_scales * (Q_int.float() - Q_deq_zeros)
            if report_mse:
                mse = ((W - Q_deq) ** 2).mean().item()
                rel = mse / ((W ** 2).mean().item() + 1e-10)
                log(f"    [{quant_count}] embed_tokens [{N}x{K}] "
                    f"MSE={mse:.8f} Rel={rel:.4%}")
            else:
                log(f"    [{quant_count}] embed_tokens [{N}x{K}] quantized")

            # Find the actual parameter name for embed_tokens
            embed_name = None
            for pname, _ in model.named_parameters():
                if 'embed_tokens' in pname and pname.endswith('.weight'):
                    embed_name = pname.replace('.weight', '')
                    break
            if embed_name:
                output_tensors[f"{embed_name}.qweight"] = pack_qweight(Q_int)
                output_tensors[f"{embed_name}.scales"] = pack_scales(scales)
                output_tensors[f"{embed_name}.qzeros"] = pack_qzeros(zeros)
                output_tensors[f"{embed_name}.g_idx"] = g_idx
            del Q_int, scales, zeros, g_idx, Q_deq
        else:
            keep_embed_bf16 = True  # fallback

    # Collect remaining non-layer parameters as BF16
    for name, param in model.named_parameters():
        if name in output_tensors:
            continue
        if name.startswith(layer_param_prefix):
            continue
        if 'mtp.' in name:
            continue
        # Skip if we already packed quantized version
        base = name.replace('.weight', '')
        if f"{base}.qweight" in output_tensors:
            continue
        output_tensors[name] = param.data.cpu().to(torch.bfloat16)

    del model, layer_inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log(f"\n  Total GPTQ loss: {total_loss:.4f} "
        f"({total_params / 1e6:.1f}M params, {quant_count} layers)")
    log(f"  Total time: {(time.time() - t_start) / 60:.1f} minutes")

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

    # Clean up stale shard files from previous runs
    for old_shard in out_path.glob("model.safetensors*"):
        old_shard.unlink()

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
        model_config = json.load(f)

    model_config['quantization_config'] = {
        "quant_method": "gptq",
        "bits": quantizer_config['bits'],
        "group_size": quantizer_config['group_size'],
        "checkpoint_format": "gptq_v2",
        "mixed_precision": bool(keep_bf16_patterns),
        "calibration": "gptq",
        "sym": quantizer_config.get('sym', False),
        "actorder": quantizer_config.get('actorder', True),
        "percclip": quantizer_config.get('percclip', 0.001),
        "modules_not_quantized": keep_bf16_patterns,
    }

    with open(out_path / 'config.json', 'w') as f:
        json.dump(model_config, f, indent=2)

    log(f"\n  Model saved to {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPTQ Calibrated INT4 Quantization for WebGPU (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard quantization (actorder enabled by default):
  python scripts/quantize_gptq.py --model ./models/qwen3.5-9b --output models/qwen3.5-9b-GPTQ

  # With MSE reporting:
  python scripts/quantize_gptq.py --model ./models/qwen3.5-9b --output models/out --report-mse

  # Disable actorder (simpler output, no g_idx needed in shader):
  python scripts/quantize_gptq.py --model ./models/qwen3.5-9b --no-actorder --output models/out
        """)
    parser.add_argument("--model", required=True,
                        help="HF repo ID or local model path")
    parser.add_argument("--output", default=None,
                        help="Output directory")
    parser.add_argument("--bits", type=int, default=4,
                        help="Quantization bits (default: 4)")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Group size (default: 128)")
    parser.add_argument("--num-samples", type=int, default=128,
                        help="Calibration samples (default: 128)")
    parser.add_argument("--seq-length", type=int, default=2048,
                        help="Calibration sequence length (default: 2048)")
    parser.add_argument("--dataset", default="wikitext",
                        help="'wikitext' or path to .txt file")
    parser.add_argument("--percdamp", type=float, default=0.01,
                        help="GPTQ damping factor (default: 0.01)")
    parser.add_argument("--blocksize", type=int, default=128,
                        help="GPTQ block size (default: 128)")
    parser.add_argument("--device", default="cuda",
                        help="Device: cpu or cuda")
    parser.add_argument("--keep-bf16", nargs="*",
                        default=["linear_attn", "norm"],
                        help="Module patterns to keep at BF16 (default: SSM + norms only)")
    parser.add_argument("--sym", action="store_true",
                        help="Use symmetric quantization")
    parser.add_argument("--no-actorder", action="store_true",
                        help="Disable activation ordering (simpler but lower quality)")
    parser.add_argument("--percclip", type=float, default=0.001,
                        help="Percentile clipping (0.001 = 0.1%% outlier removal)")
    parser.add_argument("--report-mse", action="store_true",
                        help="Report per-layer MSE")
    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path(args.model).name) + "-GPTQ-Int4"

    actorder = not args.no_actorder

    log(f"{'='*60}")
    log(f"GPTQ Calibrated INT4 Quantization (v2)")
    log(f"{'='*60}")
    log(f"Model:        {args.model}")
    log(f"Output:       {args.output}")
    log(f"Bits:         {args.bits}")
    log(f"Group size:   {args.group_size}")
    log(f"Calibration:  {args.num_samples} samples x {args.seq_length} tokens")
    log(f"Dataset:      {args.dataset}")
    log(f"Device:       {args.device}")
    log(f"Keep BF16:    {args.keep_bf16}")
    log(f"Symmetric:    {args.sym}")
    log(f"Actorder:     {actorder}")
    log(f"Perc clip:    {args.percclip}")
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
        'actorder': actorder,
        'percclip': args.percclip,
    }

    t_start = time.time()
    output_tensors = quantize_layer_by_layer(
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
