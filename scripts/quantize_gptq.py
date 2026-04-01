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

import math

import torch
import torch.nn as nn
from safetensors.torch import save_file

# Disable TF32 for full float32 precision in Hessian matmuls
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def log(msg: str):
    """Print with immediate flush."""
    print(msg, flush=True)


# ── Hadamard Rotation (QuIP#/QuaRot) ─────────────────────────────────────

def fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """In-place-style fast Walsh-Hadamard transform along the last dimension.

    O(n log n) using butterfly operations — no multiplications, only add/sub.
    Dimension must be a power of 2. Returns x @ H / sqrt(n) where H is the
    Walsh-Hadamard matrix.
    """
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"Dimension {n} must be a power of 2"
    x = x.clone()
    h = 1
    while h < n:
        # Butterfly: pair elements h apart
        x_view = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        a = x_view[..., 0, :].clone()
        b = x_view[..., 1, :].clone()
        x_view[..., 0, :] = a + b
        x_view[..., 1, :] = a - b
        h *= 2
    return x / math.sqrt(n)


def apply_hadamard_to_weight(W: torch.Tensor, H_hessian: torch.Tensor) -> tuple:
    """Apply Hadamard rotation to weight and Hessian for better GPTQ.

    Rotates the INPUT dimension (K) of weight W [N, K]:
      W_rot = W @ Had_K
      H_rot = Had_K @ H @ Had_K

    The rotation makes W more "incoherent" (uniform magnitude), reducing GPTQ
    quantization error by spreading outlier energy across all channels.

    Uses pure Hadamard (no random signs) so all projections sharing the same
    input can use one online FWHT at inference.

    Returns: (W_rot, H_rot)
    """
    N, K = W.shape
    if K & (K - 1) != 0:
        return W, H_hessian  # skip non-power-of-2

    W_rot = fast_hadamard_transform(W)       # W @ Had (operates on last dim = K)
    H_rot = fast_hadamard_transform(H_hessian)      # Had @ H (rows)
    H_rot = fast_hadamard_transform(H_rot.T).T       # H @ Had (cols)

    return W_rot, H_rot


def compute_klt_rotation(calibration_acts: list[torch.Tensor],
                         device: torch.device) -> torch.Tensor:
    """Compute KLT rotation matrix from calibration activations (MambaQuant).

    Eigendecomposes the activation covariance matrix to find an orthogonal
    rotation that equalizes channel variances — data-adaptive alternative to
    fixed Hadamard rotation.

    Args:
        calibration_acts: list of activation tensors [batch, seq, hidden] or [tokens, hidden]
        device: computation device

    Returns: R [hidden, hidden] orthogonal rotation matrix
    """
    # Flatten and concatenate all activations
    all_acts = []
    for x in calibration_acts:
        all_acts.append(x.reshape(-1, x.shape[-1]).float())
    X = torch.cat(all_acts, dim=0).to(device)

    # Compute covariance
    mean = X.mean(dim=0, keepdim=True)
    X_centered = X - mean
    C = (X_centered.T @ X_centered) / X_centered.shape[0]

    # Eigendecompose: C = V @ Lambda @ V^T
    eigenvalues, V = torch.linalg.eigh(C)

    # Combined rotation: Hadamard @ eigenvectors (if dimension is power of 2)
    K = C.shape[0]
    if K & (K - 1) == 0:
        # Combine with Hadamard for extra incoherence
        R = fast_hadamard_transform(V.T).T  # Had @ V
    else:
        R = V  # KLT only (non-power-of-2)

    return R


def apply_rotation_to_weight(W: torch.Tensor, H_hessian: torch.Tensor,
                             R: torch.Tensor) -> tuple:
    """Apply an arbitrary orthogonal rotation to weight and Hessian.

    W_rot = W @ R^T,  H_rot = R @ H @ R^T

    Returns: (W_rot, H_rot)
    """
    W_rot = W @ R.T
    H_rot = R @ H_hessian @ R.T
    return W_rot, H_rot


def compute_snc_permutation(calibration_acts: list[torch.Tensor],
                            group_size: int = 128) -> torch.Tensor:
    """Compute Sort-and-Cluster channel permutation (Quamba2).

    Sorts channels by calibration activation magnitude so that channels
    with similar ranges end up in the same GPTQ quantization group.
    This makes per-group scale/zero more uniform, reducing clipping loss.

    Args:
        calibration_acts: list of activation tensors
        group_size: GPTQ group size (channels in same group should have similar range)

    Returns: perm [hidden] — permutation indices for channel reordering
    """
    all_acts = []
    for x in calibration_acts:
        all_acts.append(x.reshape(-1, x.shape[-1]).float())
    X = torch.cat(all_acts, dim=0)

    # Per-channel maximum absolute value across all tokens
    channel_max = X.abs().max(dim=0).values  # [hidden]

    # Sort by magnitude — channels with similar ranges become adjacent
    perm = torch.argsort(channel_max)

    return perm


def apply_snc_to_weight(W: torch.Tensor, H_hessian: torch.Tensor,
                        perm: torch.Tensor) -> tuple:
    """Apply Sort-and-Cluster permutation to weight and Hessian.

    Reorders the INPUT dimension (columns) of W [N, K] and both dims of H [K, K].

    Returns: (W_permuted, H_permuted)
    """
    W_perm = W[:, perm]
    H_perm = H_hessian[perm][:, perm]
    return W_perm, H_perm


# ── Global Offline KLT Fusion ──────────────────────────────────────────────

# Weight fusion rules for global KLT rotation R [H, H]:
#   Weights reading from residual stream (input dim = H):  W_new = W @ R.T
#   Weights writing to residual stream (output dim = H):   W_new = R @ W
#   RMSNorm: unchanged (commutes with orthogonal R)
#   A_log, dt_bias, conv1d: unchanged (internal to SSM, not on residual stream)

# Projections that WRITE to the residual stream (output dim = hidden_size)
OUTPUT_PROJECTIONS = {'o_proj', 'out_proj', 'down_proj'}


def compute_global_klt(
    inner_model,
    calibration_samples: list[torch.Tensor],
    device: torch.device,
    num_samples: int = 32,
) -> torch.Tensor:
    """Compute a global KLT rotation matrix from post-embedding activations.

    Uses the covariance of the residual stream (post-embed hidden states)
    to find an orthogonal rotation R that equalizes channel variances.
    R is shared across ALL layers — fused into weights offline.

    Returns: R [hidden_size, hidden_size] orthogonal matrix
    """
    samples = calibration_samples[:num_samples]

    log(f"[KLT] Computing global rotation from {len(samples)} samples...")

    inner_model.embed_tokens.to(device)
    activations = []
    with torch.no_grad():
        for sample in samples:
            hidden = inner_model.embed_tokens(sample.to(device))
            activations.append(hidden.cpu())
    inner_model.embed_tokens.to('cpu')
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    R = compute_klt_rotation(activations, device)

    # Verify orthogonality
    I = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
    ortho_err = (R @ R.T - I).abs().max().item()
    log(f"[KLT] R [{R.shape[0]}x{R.shape[1]}], orthogonality error: {ortho_err:.2e}")
    if ortho_err > 1e-4:
        log(f"[KLT] WARNING: R may not be orthogonal! Error {ortho_err:.2e} > 1e-4")

    return R


def absorb_norm_scales(model, device: torch.device):
    """Absorb RMSNorm learnable scales into adjacent projection weights.

    RMSNorm: y = scale * x / ||x||_rms
    The scale does NOT commute with orthogonal rotation R.
    Fix: absorb scale into projections, set norm to identity.
    Must be called BEFORE fuse_klt_into_model().
    """
    inner_model = find_inner_model(model)
    layers = inner_model.layers

    # Detect (1+weight) vs (weight) convention
    sample_w = None
    for name, param in layers[0].named_parameters():
        if 'input_layernorm' in name and 'weight' in name:
            sample_w = param.data.float().mean().item()
            break

    use_residual = sample_w is not None and abs(sample_w) < 0.5
    log(f"[KLT] Norm convention: {'(1+w)' if use_residual else '(w)'}, "
        f"sample mean={sample_w:.4f}")

    absorbed = 0

    # Projections that read from input_layernorm
    INPUT_NORM_READERS = {'q_proj', 'k_proj', 'v_proj',
                          'in_proj_qkv', 'in_proj_a', 'in_proj_b', 'in_proj_z'}
    # Projections that read from post_attention_layernorm
    POST_NORM_READERS = {'gate_proj', 'up_proj'}

    for li, layer in enumerate(layers):
        layer.to(device)

        norms = {}
        projs_by_norm = {'input': [], 'post': []}

        for name, module in layer.named_modules():
            if 'input_layernorm' in name and hasattr(module, 'weight'):
                norms['input'] = module
            elif 'post_attention_layernorm' in name and hasattr(module, 'weight'):
                norms['post'] = module
            elif isinstance(module, nn.Linear):
                pname = name.split('.')[-1]
                if pname in INPUT_NORM_READERS:
                    projs_by_norm['input'].append(module)
                elif pname in POST_NORM_READERS:
                    projs_by_norm['post'].append(module)

        for norm_key in ['input', 'post']:
            norm = norms.get(norm_key)
            projs = projs_by_norm[norm_key]
            if norm is None or not projs:
                continue

            w = norm.weight.data.float().to(device)
            scale = (1.0 + w) if use_residual else w

            for proj in projs:
                # W_new[i, j] = W[i, j] * scale[j]
                proj.weight.data = (proj.weight.data.float().to(device) *
                                    scale.unsqueeze(0)).to(proj.weight.dtype)
                absorbed += 1

            # Neutralize norm: set to identity
            if use_residual:
                norm.weight.data.zero_()
            else:
                norm.weight.data.fill_(1.0)

        layer.to('cpu')
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Final norm -> lm_head
    final_norm = getattr(inner_model, 'norm', None)
    lm_head = getattr(model, 'lm_head', None)
    if final_norm is not None and lm_head is not None and hasattr(final_norm, 'weight'):
        final_norm.to(device)
        lm_head.to(device)
        w = final_norm.weight.data.float().to(device)
        scale = (1.0 + w) if use_residual else w
        lm_head.weight.data = (lm_head.weight.data.float().to(device) *
                                scale.unsqueeze(0)).to(lm_head.weight.dtype)
        if use_residual:
            final_norm.weight.data.zero_()
        else:
            final_norm.weight.data.fill_(1.0)
        final_norm.to('cpu')
        lm_head.to('cpu')
        absorbed += 1
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    log(f"[KLT] Absorbed norm scales into {absorbed} projection weights")


def fuse_klt_into_model(model, R: torch.Tensor, device: torch.device):
    """Fuse global KLT rotation R into ALL model weights offline.

    After this, the entire model operates in the KLT-rotated basis.
    IMPORTANT: call absorb_norm_scales() FIRST so RMSNorm commutes with R.
    SiLU is transparent for W4A16 (activations stay full precision).

    Modifies model weights IN-PLACE. Zero inference cost after fusion.
    """
    inner_model = find_inner_model(model)
    R = R.float().to(device)
    R_T = R.T.contiguous()

    rotated_count = 0

    def rotate_input(module: nn.Module, attr: str = 'weight'):
        """W_new = W @ R.T — for projections reading from residual stream."""
        nonlocal rotated_count
        w = getattr(module, attr)
        if w is None:
            return
        w_float = w.data.float().to(device)
        w_rot = w_float @ R_T
        w.data = w_rot.to(w.dtype).to(w.device)
        rotated_count += 1

    def rotate_output(module: nn.Module, attr: str = 'weight'):
        """W_new = R @ W — for projections writing to residual stream."""
        nonlocal rotated_count
        w = getattr(module, attr)
        if w is None:
            return
        w_float = w.data.float().to(device)
        w_rot = R @ w_float
        w.data = w_rot.to(w.dtype).to(w.device)
        rotated_count += 1

    # 1. embed_tokens: output enters rotated residual stream
    log(f"[KLT] Fusing into embed_tokens...")
    # embed_tokens.weight is [vocab, H] — each row is an embedding vector
    # We need embed output to be in rotated space: x_rot = R @ x = R @ embed[token]
    # This means embed_rot[i] = R @ embed[i] for each row, i.e. embed_rot = embed @ R.T
    rotate_input(inner_model.embed_tokens)

    # 2. lm_head: reads from rotated residual stream
    lm_head = getattr(model, 'lm_head', None)
    if lm_head is not None and hasattr(lm_head, 'weight'):
        log(f"[KLT] Fusing into lm_head...")
        rotate_input(lm_head)

    # 3. Per-layer projections
    layers = inner_model.layers
    for li, layer in enumerate(layers):
        layer.to(device)

        for name, module in layer.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # Skip modules that don't touch the residual stream
            # (A_log, dt_bias, conv1d are not nn.Linear, so won't match)
            # (norm weights are not nn.Linear either)
            name_parts = name.split('.')
            proj_name = name_parts[-1] if name_parts else name

            if proj_name in OUTPUT_PROJECTIONS:
                # Writes to residual stream: R @ W
                rotate_output(module)
            elif any(pat in name for pat in QUANT_PATTERNS):
                # Reads from residual stream: W @ R.T
                rotate_input(module)
            # else: skip (not a recognized projection)

        layer.to('cpu')
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    log(f"[KLT] Fused R into {rotated_count} weight matrices")


# ── SSM Activation Profiling ───────────────────────────────────────────────

class ChannelProfiler:
    """Accumulates per-channel statistics using Welford's online algorithm.

    Memory-efficient: O(channels) per profiler, not O(tokens × channels).
    Tracks: count, mean, M2 (for variance), max_abs, min_abs, kurtosis_acc.
    """

    def __init__(self, channels: int, device: torch.device):
        self.n = 0
        self.sum_x = torch.zeros(channels, dtype=torch.float64, device=device)
        self.sum_x2 = torch.zeros(channels, dtype=torch.float64, device=device)
        self.max_abs = torch.zeros(channels, dtype=torch.float64, device=device)
        # Track per-sample channel maxes for persistence analysis
        self.sample_rankings = []  # list of top-K channel indices per sample

    def update(self, x: torch.Tensor):
        """Update stats with activation tensor x [tokens, channels]. Fully vectorized."""
        x = x.reshape(-1, x.shape[-1]).double()
        batch_n = x.shape[0]

        # Per-channel max abs (across all tokens seen so far)
        batch_max = x.abs().max(dim=0).values
        self.max_abs = torch.maximum(self.max_abs, batch_max)

        # Accumulate sum and sum-of-squares (vectorized, no Python loop)
        self.sum_x += x.sum(dim=0)
        self.sum_x2 += (x * x).sum(dim=0)
        self.n += batch_n

        # Track which channels are hottest in this batch (persistence check)
        top_k = min(32, x.shape[1])
        batch_top = batch_max.topk(top_k).indices.cpu().tolist()
        self.sample_rankings.append(batch_top)

    @property
    def mean(self) -> torch.Tensor:
        if self.n == 0:
            return self.sum_x
        return self.sum_x / self.n

    @property
    def variance(self) -> torch.Tensor:
        if self.n < 2:
            return self.sum_x2
        mean = self.mean
        return self.sum_x2 / self.n - mean * mean

    def report(self) -> dict:
        """Generate per-channel statistics report."""
        var = self.variance
        std = var.clamp(min=0).sqrt()
        mean_abs = self.mean.abs()

        # Variance ratio: how non-uniform are channel variances?
        var_sorted = var.sort().values
        var_nonzero = var_sorted[var_sorted > 1e-12]
        if len(var_nonzero) > 1:
            variance_ratio = (var_nonzero[-1] / var_nonzero[0]).item()
            # 90th/10th percentile ratio (more robust)
            p90 = var_nonzero[int(len(var_nonzero) * 0.9)].item()
            p10 = var_nonzero[int(len(var_nonzero) * 0.1)].item()
            var_p90_p10 = p90 / max(p10, 1e-12)
        else:
            variance_ratio = 1.0
            var_p90_p10 = 1.0

        # Outlier channels: channels where max_abs > 3× median
        median_max = self.max_abs.median().item()
        outlier_mask = self.max_abs > 3 * median_max
        outlier_count = outlier_mask.sum().item()
        outlier_indices = outlier_mask.nonzero(as_tuple=True)[0].cpu().tolist()

        # Channel persistence: do the same channels dominate across samples?
        if len(self.sample_rankings) >= 2:
            from collections import Counter
            all_tops = [ch for ranking in self.sample_rankings for ch in ranking]
            top_freq = Counter(all_tops)
            most_common = top_freq.most_common(10)
            # Persistence score: fraction of samples where top-1 channel is in top-K
            top1_ch = most_common[0][0] if most_common else -1
            persistence = most_common[0][1] / len(self.sample_rankings) if most_common else 0
        else:
            most_common = []
            persistence = 0
            top1_ch = -1

        return {
            'num_tokens': self.n,
            'num_channels': len(self.mean),
            'mean_abs': {
                'min': mean_abs.min().item(),
                'max': mean_abs.max().item(),
                'median': mean_abs.median().item(),
            },
            'max_abs': {
                'min': self.max_abs.min().item(),
                'max': self.max_abs.max().item(),
                'median': median_max,
                'top5': self.max_abs.topk(5).values.cpu().tolist(),
                'top5_indices': self.max_abs.topk(5).indices.cpu().tolist(),
            },
            'variance': {
                'min': var.min().item(),
                'max': var.max().item(),
                'median': var.median().item(),
                'ratio_max_min': variance_ratio,
                'ratio_p90_p10': var_p90_p10,
            },
            'outliers': {
                'count': outlier_count,
                'pct': outlier_count / len(self.mean) * 100,
                'threshold': 3 * median_max,
                'top10_indices': outlier_indices[:10],
            },
            'persistence': {
                'score': persistence,
                'top1_channel': top1_ch,
                'top10_channels': [ch for ch, _ in most_common[:10]],
            },
        }


def profile_ssm_activations(
    model_path: str,
    calibration_samples: list[torch.Tensor],
    device: str = "cuda",
    num_profile_samples: int = 32,
) -> dict:
    """Profile per-channel activation statistics for all SSM layer projections.

    Runs calibration samples through each layer, hooking every Linear sublayer
    in DeltaNet (linear_attn) blocks. Returns a JSON-serializable report with
    per-channel stats for each projection.

    This tells us:
      - Which projections have the worst outliers (need INT8?)
      - How non-uniform channel variances are (KLT potential)
      - Whether outliers are channel-persistent (SnC potential)
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    log("\n[Profile] Loading model...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    hf_config = config.to_dict()
    if hasattr(config, 'text_config'):
        text_cfg = config.text_config.to_dict() if hasattr(config.text_config, 'to_dict') else {}
        hf_config.update(text_cfg)

    num_layers = hf_config.get('num_hidden_layers', 32)
    hidden_size = hf_config.get('hidden_size', 4096)
    layer_types = hf_config.get('layer_types', [])

    log(f"  Model: {hf_config.get('model_type', 'unknown')}")
    log(f"  Layers: {num_layers}, Hidden: {hidden_size}")
    log(f"  Hybrid: {bool(layer_types)}")
    if layer_types:
        n_linear = sum(1 for t in layer_types if t == 'linear_attention')
        n_full = sum(1 for t in layer_types if t == 'full_attention')
        log(f"  Linear attention: {n_linear}, Full attention: {n_full}")

    inner_model = find_inner_model(model)
    layers = inner_model.layers

    use_gpu = device != "cpu" and torch.cuda.is_available()
    gpu = torch.device(device if use_gpu else "cpu")

    # Limit samples for profiling (don't need as many as GPTQ)
    samples = calibration_samples[:num_profile_samples]

    # ── Embed calibration samples ──
    log(f"\n[Profile] Embedding {len(samples)} samples...")
    layer_inputs = []
    inner_model.embed_tokens.to(gpu)
    with torch.no_grad():
        for sample in samples:
            hidden = inner_model.embed_tokens(sample.to(gpu))
            layer_inputs.append(hidden.to(torch.bfloat16).cpu())
    inner_model.embed_tokens.to('cpu')
    if use_gpu:
        torch.cuda.empty_cache()

    # Compute rotary embeddings
    seq_len = samples[0].shape[1]
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

    # ── Profile each layer ──
    log(f"\n[Profile] Profiling {num_layers} layers...")
    report = {
        'model': model_path,
        'num_layers': num_layers,
        'hidden_size': hidden_size,
        'num_samples': len(samples),
        'seq_length': seq_len,
        'layers': {},
    }

    for li in range(num_layers):
        layer = layers[li]
        layer.to(gpu)
        ltype = layer_types[li] if li < len(layer_types) else 'unknown'
        is_ssm = (ltype == 'linear_attention')

        # Find ALL Linear sublayers in this layer
        profilers = {}
        hooks = []
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                channels = module.in_features
                profiler = ChannelProfiler(channels, gpu)
                profilers[name] = profiler

                def make_hook(prof):
                    def hook_fn(mod, inp, out):
                        x = inp[0].detach()
                        prof.update(x)
                    return hook_fn

                h = module.register_forward_hook(make_hook(profiler))
                hooks.append(h)

        # Forward calibration samples
        kw = layer_kwargs()
        with torch.no_grad():
            for inp in layer_inputs:
                try:
                    layer(inp.to(gpu), **kw)
                except Exception as e:
                    log(f"  WARNING: L{li} forward failed: {e}")

        for h in hooks:
            h.remove()

        # Collect reports
        layer_report = {
            'type': ltype,
            'is_ssm': is_ssm,
            'projections': {},
        }
        for name, profiler in profilers.items():
            layer_report['projections'][name] = profiler.report()
            layer_report['projections'][name]['in_features'] = \
                [m for n, m in layer.named_modules() if n == name][0].in_features
            layer_report['projections'][name]['out_features'] = \
                [m for n, m in layer.named_modules() if n == name][0].out_features

        report['layers'][f'layer_{li}'] = layer_report

        # Summary line
        if profilers:
            worst_proj = max(profilers.keys(),
                            key=lambda n: profilers[n].report()['variance']['ratio_p90_p10'])
            worst_ratio = profilers[worst_proj].report()['variance']['ratio_p90_p10']
            n_outlier_projs = sum(
                1 for p in profilers.values()
                if p.report()['outliers']['count'] > 0
            )
            marker = '*SSM' if is_ssm else ' ATN'
            log(f"  L{li:2d} {marker} | {len(profilers)} projs | "
                f"worst var ratio: {worst_ratio:8.1f}x ({worst_proj}) | "
                f"{n_outlier_projs} with outliers")

        # Propagate through layer for next layer's input
        layer_outputs = []
        with torch.no_grad():
            for inp in layer_inputs:
                try:
                    out = layer(inp.to(gpu), **kw)
                    if isinstance(out, tuple):
                        out = out[0]
                    layer_outputs.append(out.to(torch.bfloat16).cpu())
                except Exception:
                    layer_outputs.append(inp)

        layer.to('cpu')
        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()
        layer_inputs = layer_outputs

    del model, layer_inputs
    gc.collect()
    if use_gpu:
        torch.cuda.empty_cache()

    return report


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

# Patterns that ARE quantized (FFN + attention + SSM projections)
QUANT_PATTERNS = [
    'gate_proj', 'up_proj', 'down_proj',          # FFN
    'q_proj', 'k_proj', 'v_proj', 'o_proj',       # Attention
    'dense', 'gate_up_proj',                        # Phi-style names
    'in_proj_qkv', 'in_proj_a', 'in_proj_b',      # SSM input projections
    'in_proj_z', 'out_proj',                        # SSM gate + output
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
# Note: 'out_proj' and 'o_proj' are NOT substrings of each other, so
# SSM out_proj won't collide with attention o_proj in pattern matching.
SUBLAYER_GROUP_ORDER = [
    # SSM input projections (parallel, same input = normed hidden)
    ['in_proj_qkv', 'in_proj_a', 'in_proj_b', 'in_proj_z'],
    # SSM output projection (after SSM step)
    ['out_proj'],
    # Attention inputs (parallel, same input = normed hidden)
    ['q_proj', 'k_proj', 'v_proj'],
    # Attention output (after attention)
    ['o_proj'],
    # FFN inputs (parallel, same input = post-attn normed hidden)
    ['gate_proj', 'up_proj'],
    # FFN output (after SiLU*up)
    ['down_proj'],
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
    use_hadamard: bool = False,
    use_klt: bool = False,
    use_snc: bool = False,
    recipe: dict = None,
) -> dict[str, torch.Tensor]:
    """Layer-by-layer GPTQ with interleaved calibration and quantization.

    Supports mixed-precision via recipe: per-layer bits=2 (E8), 4 (GPTQ), 8 (INT8), 16 (BF16).

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

    # ── Step 1.5: Global KLT Fusion (if enabled) ─────────────────────────
    if use_klt:
        log(f"\n[1.5/4] Global KLT rotation fusion...")
        # Step 1: Absorb norm scales into projection weights
        # This makes RMSNorm commute with rotation (learnable scale removed)
        absorb_norm_scales(model, gpu)
        # Step 2: Compute and fuse rotation into all weights
        R = compute_global_klt(inner_model, calibration_samples, gpu,
                               num_samples=min(32, len(calibration_samples)))
        fuse_klt_into_model(model, R, gpu)
        log(f"  KLT fusion complete -- model now operates in rotated basis")
        log(f"  Hessians will be collected on the rotated model")
        if use_gpu:
            torch.cuda.empty_cache()

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
    log(f"\n[3/4] Quantization ({len(layers)} layers)...")
    output_tensors = {}
    gptq = GPTQQuantizer(**quantizer_config)
    e8_codebook = None  # Lazy-loaded on first E8 layer
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

                # Quantize each sublayer in the group (GPTQ INT4 / E8 2-bit / INT8)
                for name, module in group_modules.items():
                    W = module.weight.data.float().to(gpu)
                    N, K = W.shape

                    H = collectors[name].get_hessian()
                    collectors[name].free()

                    # Resolve per-layer quantization config from recipe
                    layer_recipe = None
                    if recipe is not None:
                        layer_recipe = resolve_layer_recipe(recipe, li, name)
                    layer_bits = layer_recipe['bits'] if layer_recipe else quantizer_config.get('bits', 4)
                    layer_method = layer_recipe.get('method', 'gptq') if layer_recipe else 'gptq'
                    layer_gs = layer_recipe.get('group_size', quantizer_config.get('group_size', 128)) if layer_recipe else quantizer_config.get('group_size', 128)

                    # Skip to BF16 if recipe says 16 bits
                    if layer_bits == 16:
                        log(f"    [{name}] -> BF16 (recipe)")
                        del W, H
                        continue

                    # Skip layers with incompatible dimensions
                    if K % 8 != 0 or N % 8 != 0:
                        log(f"    SKIP {name}: K={K}/N={N} not div by 8")
                        del W, H
                        continue

                    # Adjust group size if K not divisible
                    gs = layer_gs
                    if K % gs != 0:
                        for alt_gs in [64, 32, K]:
                            if K % alt_gs == 0:
                                log(f"    {name}: group_size {gs}->{alt_gs}")
                                gs = alt_gs
                                break

                    # Rotation preprocessing (QuIP#/MambaQuant/Quamba2)
                    rotation_tag = ""
                    if use_hadamard and (K & (K - 1)) == 0:
                        W, H = apply_hadamard_to_weight(W, H)
                        rotation_tag += "+Had"
                    if use_snc and len(layer_inputs) > 0:
                        snc_perm = compute_snc_permutation(
                            [inp.to(gpu) for inp in layer_inputs[:8]],
                            quantizer_config.get('group_size', 128))
                        W, H = apply_snc_to_weight(W, H, snc_perm.to(gpu))
                        rotation_tag += "+SnC"

                    base = f"{layer_prefix}.{li}.{name}"

                    if layer_bits == 2 or layer_method == 'e8':
                        # ── E8 Lattice 2-bit Vector Quantization ──
                        if e8_codebook is None:
                            e8_codebook = load_or_generate_e8_codebook()
                        e8_indices, e8_scales, e8_offsets, e8_cb = \
                            quantize_weight_e8(W, e8_codebook.to(gpu), group_size=gs)
                        Q_deq = dequantize_weight_e8(
                            e8_indices, e8_scales, e8_offsets, e8_cb, group_size=gs)
                        loss = ((W - Q_deq.to(gpu)) ** 2).sum().item()

                        total_loss += loss
                        total_params += N * K
                        quant_count += 1

                        if report_mse:
                            mse = ((W - Q_deq.to(gpu)) ** 2).mean().item()
                            rel = mse / ((W ** 2).mean().item() + 1e-10)
                            log(f"    [{quant_count}] {name} [{N}x{K}] E8 "
                                f"MSE={mse:.8f} Rel={rel:.4%}{rotation_tag}")
                        else:
                            log(f"    [{quant_count}] {name} [{N}x{K}] E8 "
                                f"loss={loss:.4f}{rotation_tag}")

                        # Pack for output
                        output_tensors[f"{base}.e8_indices"] = \
                            pack_e8_indices(e8_indices.cpu())
                        output_tensors[f"{base}.e8_scales"] = \
                            pack_e8_f16(e8_scales.cpu())
                        output_tensors[f"{base}.e8_offsets"] = \
                            pack_e8_f16(e8_offsets.cpu())
                        # Codebook saved once globally (not per-layer)
                        if 'e8_codebook' not in output_tensors:
                            output_tensors['e8_codebook'] = e8_cb.cpu()

                        module.weight.data = Q_deq.to(module.weight.dtype)
                        quantized_modules.add(name)

                        del W, H, e8_indices, e8_scales, e8_offsets, e8_cb, Q_deq

                    elif layer_bits == 8 or layer_method == 'rtn':
                        # ── INT8 RTN Quantization ──
                        Q_int, scales, zeros, g_idx = \
                            quantize_weight_int8(W, group_size=gs,
                                                 percclip=quantizer_config.get('percclip', 0.001))
                        Q_deq = dequantize_weight_int8(Q_int, scales, zeros, g_idx)
                        loss = ((W - Q_deq) ** 2).sum().item()

                        total_loss += loss
                        total_params += N * K
                        quant_count += 1

                        if report_mse:
                            mse = ((W - Q_deq) ** 2).mean().item()
                            rel = mse / ((W ** 2).mean().item() + 1e-10)
                            log(f"    [{quant_count}] {name} [{N}x{K}] INT8 "
                                f"MSE={mse:.8f} Rel={rel:.4%}{rotation_tag}")
                        else:
                            log(f"    [{quant_count}] {name} [{N}x{K}] INT8 "
                                f"loss={loss:.4f}{rotation_tag}")

                        # Pack for output
                        output_tensors[f"{base}.qweight_q8"] = \
                            pack_qweight_int8(Q_int.cpu())
                        output_tensors[f"{base}.scales_q8"] = \
                            pack_scales_int8(scales.cpu())
                        output_tensors[f"{base}.qzeros_q8"] = \
                            pack_qzeros_int8(zeros.cpu())
                        output_tensors[f"{base}.g_idx_q8"] = g_idx.cpu()

                        module.weight.data = Q_deq.to(module.weight.dtype)
                        quantized_modules.add(name)

                        del W, H, Q_int, scales, zeros, g_idx, Q_deq

                    else:
                        # ── GPTQ INT4 Quantization (default) ──
                        local_cfg = quantizer_config.copy()
                        local_cfg['group_size'] = gs
                        if gs != quantizer_config.get('group_size', 128):
                            local_gptq = GPTQQuantizer(**local_cfg)
                        else:
                            local_gptq = gptq

                        Q_int, scales, zeros, g_idx, Q_deq, loss = \
                            local_gptq.quantize_weight(W, H)

                        total_loss += loss
                        total_params += N * K
                        quant_count += 1

                        if report_mse:
                            mse = ((W - Q_deq) ** 2).mean().item()
                            rel = mse / ((W ** 2).mean().item() + 1e-10)
                            log(f"    [{quant_count}] {name} [{N}x{K}] INT4 "
                                f"MSE={mse:.8f} Rel={rel:.4%}{rotation_tag}")
                        else:
                            log(f"    [{quant_count}] {name} [{N}x{K}] INT4 "
                                f"loss={loss:.4f}{rotation_tag}")

                        # Pack for output (before replacing weight)
                        output_tensors[f"{base}.qweight"] = \
                            pack_qweight(Q_int.cpu())
                        output_tensors[f"{base}.scales"] = \
                            pack_scales(scales.cpu())
                        output_tensors[f"{base}.qzeros"] = \
                            pack_qzeros(zeros.cpu())
                        output_tensors[f"{base}.g_idx"] = g_idx.cpu()

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

        # Optional Hadamard rotation for lm_head
        if use_hadamard and (K & (K - 1)) == 0:
            W, H = apply_hadamard_to_weight(W, H)

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


# ── E8 Lattice 2-bit Vector Quantization ──────────────────────────────────

def load_or_generate_e8_codebook(
    calibration_weights: list[torch.Tensor] = None,
    codebook_path: str = "codebook_e8.bin",
) -> torch.Tensor:
    """Load E8 codebook from file, or generate from calibration data.

    If calibration_weights is provided, generates a data-adaptive codebook
    via k-means on groups of 8 weight values. Otherwise loads from file,
    or generates the default E8 lattice codebook.

    Returns: codebook [256, 8] float32 tensor.
    """
    import numpy as np

    if calibration_weights is not None and len(calibration_weights) > 0:
        # Data-adaptive codebook via k-means on weight groups
        log("[E8] Generating trained codebook from calibration weights...")
        all_groups = []
        for W in calibration_weights:
            w = W.float().reshape(-1)
            # Trim to multiple of 8
            n = (w.numel() // 8) * 8
            if n > 0:
                groups = w[:n].reshape(-1, 8)
                # Subsample if too many groups (keep memory reasonable)
                if groups.shape[0] > 500000:
                    idx = torch.randperm(groups.shape[0])[:500000]
                    groups = groups[idx]
                all_groups.append(groups.cpu().numpy())

        if all_groups:
            data = np.concatenate(all_groups, axis=0)
            log(f"[E8] Training codebook on {data.shape[0]} weight groups...")
            try:
                from scripts.e8_codebook import generate_trained_codebook
            except ImportError:
                from e8_codebook import generate_trained_codebook
            codebook_np = generate_trained_codebook(data)
            codebook = torch.from_numpy(codebook_np)
            log(f"[E8] Trained codebook ready: {codebook.shape}")
            return codebook

    # Try loading from file
    codebook_file = Path(codebook_path)
    if codebook_file.exists():
        log(f"[E8] Loading codebook from {codebook_path}")
        try:
            from scripts.e8_codebook import load_codebook
        except ImportError:
            from e8_codebook import load_codebook
        codebook_np = load_codebook(str(codebook_file))
        return torch.from_numpy(codebook_np)

    # Generate default E8 lattice codebook
    log("[E8] Generating default E8 lattice codebook...")
    try:
        from scripts.e8_codebook import generate_e8_codebook
    except ImportError:
        from e8_codebook import generate_e8_codebook
    codebook_np = generate_e8_codebook()
    return torch.from_numpy(codebook_np)


def quantize_weight_e8(
    W: torch.Tensor,
    codebook: torch.Tensor,
    group_size: int = 128,
) -> tuple:
    """Quantize weight matrix W using E8 vector codebook.

    Groups of 8 consecutive elements along the K dimension are quantized
    to the nearest codebook entry. Per-group (group_size) scale and offset
    provide affine rescaling.

    Args:
        W: [N, K] float32 weight matrix
        codebook: [256, 8] float32 unit-norm codebook
        group_size: group size for scale/offset (must be divisible by 8)

    Returns:
        (indices, scales, offsets, codebook) where:
        - indices: [N, K//8] uint8 codebook indices
        - scales: [N, num_groups] float16 per-group scale
        - offsets: [N, num_groups] float16 per-group offset
        - codebook: [256, 8] float32 (passed through for saving)
    """
    N, K = W.shape
    dev = W.device
    assert K % 8 == 0, f"K={K} must be divisible by 8 for E8 quantization"
    assert group_size % 8 == 0, f"group_size={group_size} must be divisible by 8"

    num_groups = K // group_size
    num_vecs_per_group = group_size // 8
    num_vecs = K // 8

    cb = codebook.to(dev).float()  # [256, 8]

    indices = torch.zeros(N, num_vecs, dtype=torch.uint8, device=dev)
    all_scales = torch.zeros(N, num_groups, dtype=torch.float32, device=dev)
    all_offsets = torch.zeros(N, num_groups, dtype=torch.float32, device=dev)

    # Process each group
    for g in range(num_groups):
        k_start = g * group_size
        k_end = k_start + group_size
        group_w = W[:, k_start:k_end].float()  # [N, group_size]

        # Per-row scale and offset for this group
        # offset = mean, scale = max(|group - offset|) (per-row)
        offset = group_w.mean(dim=1)                   # [N]
        centered = group_w - offset.unsqueeze(1)        # [N, group_size]
        scale = centered.abs().amax(dim=1).clamp(min=1e-10)  # [N]

        all_scales[:, g] = scale
        all_offsets[:, g] = offset

        # Normalize: group_norm = (group - offset) / scale
        normalized = centered / scale.unsqueeze(1)      # [N, group_size]

        # Reshape to E8 vector groups: [N * num_vecs_per_group, 8]
        vecs = normalized.reshape(N * num_vecs_per_group, 8)

        # Find nearest codebook entry for each vector
        # distances[i, j] = ||vecs[i] - cb[j]||^2
        # = ||vecs[i]||^2 - 2 * vecs[i] @ cb[j]^T + ||cb[j]||^2
        # cb is unit-norm so ||cb[j]||^2 = 1
        vecs_sq = (vecs ** 2).sum(dim=1, keepdim=True)  # [N*nvpg, 1]
        dots = vecs @ cb.T                                # [N*nvpg, 256]
        dists = vecs_sq - 2.0 * dots + 1.0               # [N*nvpg, 256]
        best_idx = dists.argmin(dim=1).to(torch.uint8)    # [N*nvpg]

        # Store indices
        vec_start = g * num_vecs_per_group
        indices[:, vec_start:vec_start + num_vecs_per_group] = \
            best_idx.reshape(N, num_vecs_per_group)

    return (
        indices,
        all_scales.to(torch.float16),
        all_offsets.to(torch.float16),
        codebook.float(),
    )


def dequantize_weight_e8(
    indices: torch.Tensor,
    scales: torch.Tensor,
    offsets: torch.Tensor,
    codebook: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Dequantize E8-quantized weight matrix for verification.

    Args:
        indices: [N, K//8] uint8 codebook indices
        scales: [N, num_groups] float16 per-group scale
        offsets: [N, num_groups] float16 per-group offset
        codebook: [256, 8] float32 codebook
        group_size: quantization group size

    Returns: [N, K] float32 dequantized weight matrix
    """
    N = indices.shape[0]
    num_vecs = indices.shape[1]
    K = num_vecs * 8
    dev = indices.device

    cb = codebook.to(dev).float()  # [256, 8]
    s = scales.float()             # [N, num_groups]
    o = offsets.float()            # [N, num_groups]

    num_vecs_per_group = group_size // 8
    num_groups = K // group_size

    W = torch.zeros(N, K, dtype=torch.float32, device=dev)

    for g in range(num_groups):
        vec_start = g * num_vecs_per_group
        group_indices = indices[:, vec_start:vec_start + num_vecs_per_group]  # [N, nvpg]

        # Lookup codebook vectors
        flat_idx = group_indices.reshape(-1).long()       # [N * nvpg]
        decoded_vecs = cb[flat_idx]                        # [N * nvpg, 8]
        decoded = decoded_vecs.reshape(N, num_vecs_per_group * 8)  # [N, group_size]

        # Apply affine: weight = codebook_val * scale + offset
        scale = s[:, g].unsqueeze(1)   # [N, 1]
        offset = o[:, g].unsqueeze(1)  # [N, 1]
        k_start = g * group_size
        W[:, k_start:k_start + group_size] = decoded * scale + offset

    return W


def pack_e8_indices(indices: torch.Tensor) -> torch.Tensor:
    """Pack [N, K//8] uint8 indices into [N, K//8//4] uint32.

    4 consecutive uint8 indices packed per u32, little-endian byte order.
    Layout matches matmul_e8.wgsl: B_indices[n, vec_idx/4], byte vec_idx%4.

    packed[n, i] contains indices for vec_idx = 4*i+0, 4*i+1, 4*i+2, 4*i+3
    at byte positions 0, 1, 2, 3 respectively.
    """
    N, num_vecs = indices.shape
    assert num_vecs % 4 == 0, f"num_vecs={num_vecs} must be divisible by 4"
    packed_count = num_vecs // 4
    idx = indices.to(torch.int32).cpu()
    # Reshape to [N, packed_count, 4], then pack 4 consecutive indices per u32
    idx_grouped = idx.reshape(N, packed_count, 4)
    packed = (idx_grouped[:, :, 0] & 0xFF) \
           | ((idx_grouped[:, :, 1] & 0xFF) << 8) \
           | ((idx_grouped[:, :, 2] & 0xFF) << 16) \
           | ((idx_grouped[:, :, 3] & 0xFF) << 24)
    return packed.to(torch.int32)


def pack_e8_f16(data: torch.Tensor) -> torch.Tensor:
    """Pack [num_groups, N] float16 data for WebGPU shader.

    Layout: [num_groups, N] as float16 (stored as-is, shader reads via read_f16).
    The shader indexes as scales_raw[group_id * N + col] with the read_f16 helper.
    """
    return data.T.contiguous().to(torch.float16)


# ── INT8 Uniform Quantization ─────────────────────────────────────────────

def quantize_weight_int8(
    W: torch.Tensor,
    group_size: int = 128,
    percclip: float = 0.001,
) -> tuple:
    """Uniform INT8 quantization with per-group scale and zero point.

    Similar to INT4 GPTQ but with 256 levels (0..255) instead of 16.
    Uses simple RTN (round-to-nearest) — no Hessian-guided error propagation
    needed at 8-bit precision.

    Args:
        W: [N, K] float32 weight matrix
        group_size: quantization group size
        percclip: percentile clipping fraction for outlier removal

    Returns:
        (Q_int, scales, zeros, g_idx) where:
        - Q_int: [N, K] int32 (0..255)
        - scales: [N, num_groups] float32
        - zeros: [N, num_groups] int32 (0..255)
        - g_idx: [K] int32 column-to-group mapping
    """
    N, K = W.shape
    dev = W.device
    maxq = 255

    num_groups = (K + group_size - 1) // group_size
    all_scales = torch.zeros(N, num_groups, device=dev)
    all_zeros = torch.zeros(N, num_groups, dtype=torch.int32, device=dev)
    Q = torch.zeros(N, K, dtype=torch.int32, device=dev)

    for g in range(num_groups):
        k_start = g * group_size
        k_end = min(k_start + group_size, K)
        group_w = W[:, k_start:k_end].float()

        # Percentile clipping for outlier robustness
        if percclip > 0 and group_w.shape[1] > 1:
            lo = torch.quantile(group_w, percclip, dim=1)
            hi = torch.quantile(group_w, 1.0 - percclip, dim=1)
        else:
            lo = group_w.min(dim=1).values
            hi = group_w.max(dim=1).values

        # Ensure range includes zero
        lo = torch.minimum(lo, torch.zeros_like(lo))
        hi = torch.maximum(hi, torch.zeros_like(hi))

        # Handle all-zero groups
        dead = (lo == 0) & (hi == 0)
        lo[dead] = -1.0
        hi[dead] = 1.0

        scale = (hi - lo) / maxq                          # [N]
        zero = torch.round(-lo / scale).clamp(0, maxq)    # [N]

        all_scales[:, g] = scale
        all_zeros[:, g] = zero.to(torch.int32)

        # Quantize
        q = torch.clamp(
            torch.round(group_w / scale.unsqueeze(1)) + zero.unsqueeze(1),
            0, maxq
        ).to(torch.int32)
        Q[:, k_start:k_end] = q

    g_idx = (torch.arange(K, device=dev) // group_size).to(torch.int32)

    return Q, all_scales, all_zeros, g_idx


def dequantize_weight_int8(
    Q_int: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    g_idx: torch.Tensor,
) -> torch.Tensor:
    """Dequantize INT8-quantized weight matrix for verification.

    Returns: [N, K] float32 dequantized weight matrix
    """
    scales_per_col = scales[:, g_idx.long()]
    zeros_per_col = zeros.float()[:, g_idx.long()]
    return scales_per_col * (Q_int.float() - zeros_per_col)


def pack_qweight_int8(q_int: torch.Tensor) -> torch.Tensor:
    """Pack [N, K] int8 values into [K//4, N] int32.

    4 INT8 values packed per i32 along the K dimension.
    The WebGPU shader reads: B_packed[(k/4)*N + n], byte at (k%4)*8.
    """
    N, K = q_int.shape
    assert K % 4 == 0, f"K={K} not divisible by 4"
    qweight = torch.zeros(K // 4, N, dtype=torch.int32)
    for byte_idx in range(4):
        k_indices = torch.arange(byte_idx, K, 4)
        qweight[k_indices // 4] |= ((q_int[:, k_indices].T.to(torch.int32)) & 0xFF) << (byte_idx * 8)
    return qweight


def pack_scales_int8(scales: torch.Tensor) -> torch.Tensor:
    """Convert [N, num_groups] scales to [num_groups, N] float16."""
    return scales.T.contiguous().to(torch.float16)


def pack_qzeros_int8(zeros: torch.Tensor) -> torch.Tensor:
    """Pack [N, num_groups] int8 zeros into [num_groups, N//4] int32.

    4 INT8 values packed per i32, layout [num_groups, N//4].
    """
    N, num_groups = zeros.shape
    assert N % 4 == 0, f"N={N} not divisible by 4"
    zeros_t = zeros.T.contiguous().to(torch.int32)  # [num_groups, N]
    qzeros = torch.zeros(num_groups, N // 4, dtype=torch.int32)
    for byte_idx in range(4):
        n_indices = torch.arange(byte_idx, N, 4)
        qzeros[:, n_indices // 4] |= ((zeros_t[:, n_indices]) & 0xFF) << (byte_idx * 8)
    return qzeros


# ── Recipe System ──────────────────────────────────────────────────────────

VALID_BITS = {2, 4, 8, 16}
VALID_METHODS = {'gptq', 'e8', 'rtn'}
VALID_ROTATIONS = {'none', 'hadamard', 'klt', 'snc', 'klt+snc'}


def load_recipe(path: str) -> dict:
    """Load and validate a quantization recipe from JSON."""
    with open(path) as f:
        recipe = json.load(f)

    # Validate global defaults
    g = recipe.get('global', {})
    if 'bits' in g and g['bits'] not in VALID_BITS:
        raise ValueError(f"Invalid global bits={g['bits']}, must be one of {VALID_BITS}")
    if 'rotation' in g and g['rotation'] not in VALID_ROTATIONS:
        raise ValueError(f"Invalid rotation='{g['rotation']}', must be one of {VALID_ROTATIONS}")
    if 'method' in g and g['method'] not in VALID_METHODS:
        raise ValueError(f"Invalid method='{g['method']}', must be one of {VALID_METHODS}")

    # Validate layer_defaults and layer_overrides
    for section in ['layer_defaults', 'layer_overrides']:
        entries = recipe.get(section, {})
        items = entries.items() if section == 'layer_defaults' else [
            (f"L{li}.{p}", cfg)
            for li, projs in entries.items()
            for p, cfg in projs.items()
        ]
        for key, cfg in items:
            if isinstance(cfg, dict):
                if 'bits' in cfg and cfg['bits'] not in VALID_BITS:
                    raise ValueError(f"Invalid bits={cfg['bits']} in {section}.{key}")
                if 'rotation' in cfg and cfg['rotation'] not in VALID_ROTATIONS:
                    raise ValueError(f"Invalid rotation in {section}.{key}")
                if 'method' in cfg and cfg['method'] not in VALID_METHODS:
                    raise ValueError(f"Invalid method='{cfg['method']}' in {section}.{key}")

    return recipe


def generate_auto_recipe(profile_report: dict) -> dict:
    """Generate an optimal quantization recipe from SSM profiling results.

    Thresholds based on variance ratio (p90/p10):
      >100 -> BF16 (too risky for INT4)
      >30  -> INT4 + KLT + SnC
      >10  -> INT4 + KLT
      >3   -> INT4 + SnC
      else -> INT4 standard
    """
    recipe = {
        'global': {'bits': 4, 'group_size': 128, 'rotation': 'none'},
        'embed_tokens': {'bits': 16},
        'lm_head': {'bits': 16},
        'layer_defaults': {},
        'layer_overrides': {},
    }

    # Aggregate variance ratios per projection type across SSM layers
    proj_stats = {}
    for lname, ldata in profile_report.get('layers', {}).items():
        if not ldata.get('is_ssm', False):
            continue
        for pname, pdata in ldata.get('projections', {}).items():
            vr = pdata.get('variance', {}).get('ratio_p90_p10', 1.0)
            if pname not in proj_stats:
                proj_stats[pname] = []
            proj_stats[pname].append({'layer': lname, 'var_ratio': vr})

    # Set layer_defaults from average variance ratio
    for pname, entries in proj_stats.items():
        avg_vr = sum(e['var_ratio'] for e in entries) / len(entries)
        max_vr = max(e['var_ratio'] for e in entries)

        if avg_vr > 100:
            recipe['layer_defaults'][pname] = {'bits': 16}
        elif avg_vr > 30:
            recipe['layer_defaults'][pname] = {'bits': 4, 'rotation': 'klt+snc'}
        elif avg_vr > 10:
            recipe['layer_defaults'][pname] = {'bits': 4, 'rotation': 'klt'}
        elif avg_vr > 3:
            recipe['layer_defaults'][pname] = {'bits': 4, 'rotation': 'snc'}
        # else: standard INT4, no override needed

        # Per-layer overrides for extreme outliers (>2x avg)
        for entry in entries:
            li = entry['layer'].replace('layer_', '')
            if entry['var_ratio'] > 2 * avg_vr and entry['var_ratio'] > 30:
                if li not in recipe['layer_overrides']:
                    recipe['layer_overrides'][li] = {}
                recipe['layer_overrides'][li][pname] = {'bits': 16}

    return recipe


def resolve_layer_recipe(
    recipe: dict, layer_idx: int, proj_name: str,
) -> dict:
    """Resolve the final quantization config for a specific projection.

    Merge order (most specific wins):
      layer_overrides[layer_idx][proj_name] > layer_defaults[proj_name] > global
    """
    defaults = {'bits': 4, 'group_size': 128, 'rotation': 'none', 'method': 'gptq'}
    result = {**defaults}

    # Level 1: global
    result.update(recipe.get('global', {}))

    # Level 2: layer_defaults — match by substring (same as QUANT_PATTERNS)
    for pat, cfg in recipe.get('layer_defaults', {}).items():
        if pat in proj_name:
            result.update(cfg)
            break

    # Level 3: layer_overrides — most specific
    layer_overrides = recipe.get('layer_overrides', {}).get(str(layer_idx), {})
    for pat, cfg in layer_overrides.items():
        if pat in proj_name:
            result.update(cfg)
            break

    # Auto-select method from bits if not explicitly set
    if 'method' not in result or result.get('method') == 'gptq':
        if result['bits'] == 2:
            result['method'] = 'e8'
        elif result['bits'] == 8:
            result['method'] = 'rtn'

    return result


# ── Output ─────────────────────────────────────────────────────────────────

def save_quantized_model(
    output_tensors: dict[str, torch.Tensor],
    model_path: str,
    output_path: str,
    quantizer_config: dict,
    keep_bf16_patterns: list[str],
    use_hadamard: bool = False,
    use_klt: bool = False,
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

    # Detect if E8 or INT8 tensors are present
    has_e8 = any(k.endswith('.e8_indices') for k in output_tensors)
    has_q8 = any(k.endswith('.qweight_q8') for k in output_tensors)

    model_config['quantization_config'] = {
        "quant_method": "gptq",
        "bits": quantizer_config['bits'],
        "group_size": quantizer_config['group_size'],
        "checkpoint_format": "gptq_v2",
        "mixed_precision": bool(keep_bf16_patterns) or has_e8 or has_q8,
        "calibration": "gptq",
        "sym": quantizer_config.get('sym', False),
        "actorder": quantizer_config.get('actorder', True),
        "percclip": quantizer_config.get('percclip', 0.001),
        "hadamard": use_hadamard,
        "klt": use_klt,
        "rotation_method": "klt" if use_klt else ("hadamard" if use_hadamard else "none"),
        "ssm_quantized": not any('linear_attn' in p for p in keep_bf16_patterns),
        "modules_not_quantized": keep_bf16_patterns,
        "has_e8": has_e8,
        "has_q8": has_q8,
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
                        default=["norm", "embed_tokens", "lm_head"],
                        help="Module patterns to keep at BF16 (default: norms + embed + lm_head; SSM A_log/dt_bias/conv1d always BF16 via SKIP_PATTERNS)")
    parser.add_argument("--sym", action="store_true",
                        help="Use symmetric quantization")
    parser.add_argument("--no-actorder", action="store_true",
                        help="Disable activation ordering (simpler but lower quality)")
    parser.add_argument("--percclip", type=float, default=0.001,
                        help="Percentile clipping (0.001 = 0.1%% outlier removal)")
    parser.add_argument("--report-mse", action="store_true",
                        help="Report per-layer MSE")
    parser.add_argument("--profile-ssm", action="store_true",
                        help="Profile SSM activation statistics instead of quantizing (outputs JSON report)")
    parser.add_argument("--profile-samples", type=int, default=32,
                        help="Number of calibration samples for profiling (default: 32)")
    parser.add_argument("--hadamard", action="store_true",
                        help="Apply Hadamard rotation before GPTQ (QuIP#-style, improves quantization quality)")
    parser.add_argument("--klt", action="store_true",
                        help="Apply KLT rotation before GPTQ for SSM layers (MambaQuant-style, data-adaptive)")
    parser.add_argument("--snc", action="store_true",
                        help="Apply Sort-and-Cluster channel reordering (Quamba2-style)")
    parser.add_argument("--recipe", type=str, default=None,
                        help="Path to quantization recipe JSON (overrides --hadamard/--klt/--snc/--keep-bf16)")
    parser.add_argument("--auto-recipe", action="store_true",
                        help="Auto-generate recipe from SSM profiling, then quantize")
    parser.add_argument("--save-recipe", type=str, default=None,
                        help="Save auto-generated recipe to JSON for review/editing")
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
    log(f"Hadamard:     {args.hadamard}")
    log(f"KLT:          {args.klt}")
    log(f"SnC:          {args.snc}")
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

    # ── Profile mode ──
    if args.profile_ssm:
        log(f"\n{'='*60}")
        log(f"SSM Activation Profiling")
        log(f"{'='*60}")

        t_start = time.time()
        report = profile_ssm_activations(
            model_path=args.model,
            calibration_samples=calibration_samples,
            device=args.device,
            num_profile_samples=args.profile_samples,
        )
        t_profile = time.time() - t_start

        # Print summary
        log(f"\n{'='*60}")
        log(f"PROFILE SUMMARY")
        log(f"{'='*60}")

        ssm_layers = []
        attn_layers = []
        for lname, ldata in report['layers'].items():
            if ldata['is_ssm']:
                ssm_layers.append((lname, ldata))
            else:
                attn_layers.append((lname, ldata))

        log(f"\nSSM layers: {len(ssm_layers)}, Attention layers: {len(attn_layers)}")

        # Aggregate worst-case stats across SSM layers
        if ssm_layers:
            log(f"\n-- SSM Layer Analysis --")
            all_projs = {}
            for lname, ldata in ssm_layers:
                for pname, pdata in ldata['projections'].items():
                    if pname not in all_projs:
                        all_projs[pname] = []
                    all_projs[pname].append({
                        'layer': lname,
                        'var_ratio': pdata['variance']['ratio_p90_p10'],
                        'outlier_pct': pdata['outliers']['pct'],
                        'persistence': pdata['persistence']['score'],
                        'max_abs_max': pdata['max_abs']['max'],
                    })

            log(f"\n{'Projection':<35} {'Avg VarRatio':>12} {'Max VarRatio':>12} "
                f"{'Avg Outlier%':>12} {'Persistence':>11} {'Max |x|':>10}")
            log(f"{'-'*92}")
            for pname, entries in sorted(all_projs.items()):
                avg_vr = sum(e['var_ratio'] for e in entries) / len(entries)
                max_vr = max(e['var_ratio'] for e in entries)
                avg_out = sum(e['outlier_pct'] for e in entries) / len(entries)
                avg_pers = sum(e['persistence'] for e in entries) / len(entries)
                max_abs = max(e['max_abs_max'] for e in entries)
                log(f"{pname:<35} {avg_vr:>12.1f}x {max_vr:>12.1f}x "
                    f"{avg_out:>11.1f}% {avg_pers:>11.2f} {max_abs:>10.1f}")

            # Recommendation
            log(f"\n-- Quantization Recommendations --")
            for pname, entries in sorted(all_projs.items()):
                avg_vr = sum(e['var_ratio'] for e in entries) / len(entries)
                avg_out = sum(e['outlier_pct'] for e in entries) / len(entries)
                dims = None
                for lname, ldata in ssm_layers:
                    if pname in ldata['projections']:
                        dims = (ldata['projections'][pname]['in_features'],
                                ldata['projections'][pname]['out_features'])
                        break
                dim_str = f"[{dims[1]}x{dims[0]}]" if dims else ""

                if avg_vr > 100 or avg_out > 5:
                    rec = "INT8 or BF16 (high outliers, KLT critical)"
                elif avg_vr > 10:
                    rec = "INT4 + KLT rotation (moderate outliers)"
                elif avg_vr > 3:
                    rec = "INT4 + SnC (mild non-uniformity)"
                else:
                    rec = "INT4 standard (uniform channels)"
                log(f"  {pname:<32} {dim_str:<12} -> {rec}")

        # Save full report
        report_path = Path(args.output if args.output else 'ssm-profile.json')
        if report_path.suffix != '.json':
            report_path = report_path.with_suffix('.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        log(f"\nFull report saved to {report_path}")
        log(f"Profiling completed in {t_profile:.0f}s")
        return

    # ── Recipe handling ──
    recipe = None
    if args.recipe:
        recipe = load_recipe(args.recipe)
        log(f"\nLoaded recipe from {args.recipe}")
        log(f"  Recipe: {json.dumps(recipe, indent=2)}")
    elif args.auto_recipe:
        log("\nRunning SSM profiling for auto-recipe...")
        profile_report = profile_ssm_activations(
            model_path=args.model,
            calibration_samples=calibration_samples,
            device=args.device,
            num_profile_samples=min(32, args.num_samples),
        )
        recipe = generate_auto_recipe(profile_report)
        log(f"\nAuto-recipe generated:")
        log(f"  {json.dumps(recipe, indent=2)}")
        if args.save_recipe:
            with open(args.save_recipe, 'w') as f:
                json.dump(recipe, f, indent=2)
            log(f"  Recipe saved to {args.save_recipe}")

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
        use_hadamard=args.hadamard,
        use_klt=args.klt,
        use_snc=args.snc,
        recipe=recipe,
    )
    t_quant = time.time() - t_start

    # Save
    save_quantized_model(
        output_tensors=output_tensors,
        model_path=args.model,
        output_path=args.output,
        quantizer_config=quantizer_config,
        keep_bf16_patterns=args.keep_bf16,
        use_hadamard=args.hadamard,
        use_klt=args.klt,
    )

    log(f"\n{'='*60}")
    log(f"GPTQ quantization complete in {t_quant / 60:.1f} minutes")
    log(f"Output: {args.output}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
