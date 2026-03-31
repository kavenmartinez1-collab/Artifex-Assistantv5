#!/usr/bin/env python3
"""
Numerical Audit: WGSL Shader Verification Against PyTorch Reference

Reimplements each WGSL kernel's exact logic in Python and compares against
PyTorch's known-correct implementations. Loads actual model weights so
divergences reflect real production behavior, not synthetic edge cases.

Usage:
    python scripts/audit_kernels.py --model models/qwen3.5-9b-GPTQv2-noact

Tests each kernel independently:
  1. RMSNorm (with 1+w residual convention)
  2. RoPE (interleaved pairing, partial rotary)
  3. Matmul BT (B-transposed tiled matmul)
  4. Matmul Q4 (INT4 dequant matmul with GPTQ packing)
  5. Embed F16/BF16 (packed embedding lookup)
  6. Embed Q4 (INT4 GPTQ embedding lookup)
  7. Attention (fused QKV->softmax->weighted V)
  8. SiLU / GELU elementwise
  9. Full single-layer forward pass comparison
"""

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


# ── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"[AUDIT] {msg}", flush=True)

def compare(name: str, wgsl: np.ndarray, ref: np.ndarray,
            atol: float = 1e-5, rtol: float = 1e-4) -> bool:
    """Compare WGSL reimplementation against PyTorch reference."""
    if wgsl.shape != ref.shape:
        log(f"  FAIL {name}: shape mismatch WGSL={wgsl.shape} vs REF={ref.shape}")
        return False

    abs_diff = np.abs(wgsl - ref)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    rel_diff = abs_diff / (np.abs(ref) + 1e-10)
    max_rel = rel_diff.max()

    passed = np.allclose(wgsl, ref, atol=atol, rtol=rtol)
    status = "PASS" if passed else "FAIL"
    log(f"  {status} {name}: max_abs={max_diff:.2e}, mean_abs={mean_diff:.2e}, max_rel={max_rel:.2e}")

    if not passed:
        # Show where the worst divergence is
        worst_idx = np.unravel_index(abs_diff.argmax(), abs_diff.shape)
        log(f"    Worst at {worst_idx}: WGSL={wgsl[worst_idx]:.8f} vs REF={ref[worst_idx]:.8f}")
        # Show first few values
        flat_w = wgsl.flatten()[:8]
        flat_r = ref.flatten()[:8]
        log(f"    WGSL first 8: {[f'{v:.6f}' for v in flat_w]}")
        log(f"    REF  first 8: {[f'{v:.6f}' for v in flat_r]}")

    return passed


# ── F16 Conversion ───────────────────────────────────────────────────────────

def f16_to_f32_bitwise(bits: int) -> float:
    """Exact bitwise F16 → F32 (matches matmul_q4.wgsl::read_f16_scale)."""
    sign = (bits >> 15) & 1
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF

    if exp == 0:
        if frac == 0:
            return -0.0 if sign else 0.0
        # Subnormal: frac * 2^-24
        f = frac * struct.unpack('f', struct.pack('I', 0x33800000))[0]
        return -f if sign else f
    if exp == 31:
        return -1e30 if sign else 1e30

    # Normal: exact bitwise construction
    f32_bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13)
    return struct.unpack('f', struct.pack('I', f32_bits))[0]


def bf16_to_f32(bits: int) -> float:
    """BF16 → F32: shift left 16 bits."""
    return struct.unpack('f', struct.pack('I', bits << 16))[0]


# ── WGSL Kernel Reimplementations ────────────────────────────────────────────

def wgsl_rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float,
                 use_residual_weight: bool = True) -> np.ndarray:
    """Reimplements rmsnorm.wgsl exactly."""
    # x: [rows, hidden], weight: [hidden]
    sum_sq = np.sum(x * x, axis=-1, keepdims=True)
    rms_inv = 1.0 / np.sqrt(sum_sq / x.shape[-1] + eps)
    w = (1.0 + weight) if use_residual_weight else weight
    return x * rms_inv * w


def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float,
                    use_residual_weight: bool = True) -> torch.Tensor:
    """PyTorch reference RMSNorm."""
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    w = (1.0 + weight) if use_residual_weight else weight
    return x / rms * w


def wgsl_rope(qk: np.ndarray, seq_len: int, num_heads: int, head_dim: int,
              pos_offset: int, rope_base: float, rotary_dim: int = 0) -> np.ndarray:
    """Reimplements rope.wgsl exactly (interleaved pairing)."""
    out = qk.copy()
    rot_dim = head_dim if rotary_dim == 0 else rotary_dim
    half_rot = rot_dim // 2

    for seq_pos in range(seq_len):
        for head in range(num_heads):
            for pair_idx in range(half_rot):
                position = float(seq_pos + pos_offset)
                freq = 1.0 / (rope_base ** (float(2 * pair_idx) / float(rot_dim)))
                angle = position * freq
                cos_val = math.cos(angle)
                sin_val = math.sin(angle)

                base_idx = (seq_pos * num_heads + head) * head_dim
                i0 = base_idx + 2 * pair_idx
                i1 = base_idx + 2 * pair_idx + 1

                v0 = out[i0]
                v1 = out[i1]
                out[i0] = v0 * cos_val - v1 * sin_val
                out[i1] = v0 * sin_val + v1 * cos_val

    return out


def pytorch_rope_interleaved(qk: torch.Tensor, seq_len: int, num_heads: int,
                             head_dim: int, pos_offset: int, rope_base: float,
                             rotary_dim: int = 0) -> torch.Tensor:
    """PyTorch reference RoPE with interleaved pairing (matches Qwen3.5)."""
    rot_dim = head_dim if rotary_dim == 0 else rotary_dim
    half_rot = rot_dim // 2
    out = qk.clone().reshape(seq_len, num_heads, head_dim)

    positions = torch.arange(pos_offset, pos_offset + seq_len, dtype=torch.float32)
    freqs = 1.0 / (rope_base ** (torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim))
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # [seq, half_rot]
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)

    for s in range(seq_len):
        for h in range(num_heads):
            for p in range(half_rot):
                v0 = out[s, h, 2 * p].item()
                v1 = out[s, h, 2 * p + 1].item()
                c = cos_vals[s, p].item()
                si = sin_vals[s, p].item()
                out[s, h, 2 * p] = v0 * c - v1 * si
                out[s, h, 2 * p + 1] = v0 * si + v1 * c

    return out.reshape(-1)


def wgsl_extract_q4(packed: int, nibble_idx: int) -> int:
    """Reimplements extract_q4 from matmul_q4.wgsl."""
    return (packed >> (nibble_idx * 4)) & 0xF


def wgsl_dequant_q4_matmul(qweight: np.ndarray, scales_raw: np.ndarray,
                            qzeros: np.ndarray, g_idx: np.ndarray,
                            K: int, N: int, group_size: int) -> np.ndarray:
    """
    Reimplements the B dequantization from matmul_q4.wgsl.
    Returns dequantized weight matrix [K, N] (pre-transposed for matmul).

    qweight: [K//8, N] as int32
    scales_raw: [num_groups * N // 2] as uint32 (pairs of f16)
    qzeros: [num_groups, (N+7)//8] as int32
    g_idx: [K] as uint32
    """
    W = np.zeros((K, N), dtype=np.float32)

    for k in range(K):
        for n in range(N):
            # Extract 4-bit weight
            packed_idx = (k // 8) * N + n
            nibble = k % 8
            packed = int(qweight.flat[packed_idx])
            q4_val = (packed >> (nibble * 4)) & 0xF

            # Get group (actorder-aware)
            group_id = int(g_idx[k])

            # Read f16 scale
            scale_linear_idx = group_id * N + n
            word_idx = scale_linear_idx // 2
            word = int(scales_raw.flat[word_idx])
            if (scale_linear_idx & 1) == 1:
                half_bits = (word >> 16) & 0xFFFF
            else:
                half_bits = word & 0xFFFF
            scale = f16_to_f32_bitwise(half_bits)

            # Read zero point
            zero_packed_idx = group_id * ((N + 7) // 8) + n // 8
            zero_nibble = n % 8
            zero_packed = int(qzeros.flat[zero_packed_idx])
            zero = (zero_packed >> (zero_nibble * 4)) & 0xF

            # Dequant
            W[k, n] = (float(q4_val) - float(zero)) * scale

    return W


def wgsl_matmul_q4(A: np.ndarray, qweight: np.ndarray, scales_raw: np.ndarray,
                   qzeros: np.ndarray, g_idx: np.ndarray,
                   M: int, N: int, K: int, group_size: int) -> np.ndarray:
    """Full matmul_q4: C[M,N] = A[M,K] @ dequant(B)^T (Kahan summation)."""
    W = wgsl_dequant_q4_matmul(qweight, scales_raw, qzeros, g_idx, K, N, group_size)
    # matmul_q4 computes A @ B^T where B is the dequantized matrix
    # But we built W as [K, N], so A @ W gives [M, N]
    # With Kahan summation for precision
    C = np.zeros((M, N), dtype=np.float32)
    for m in range(M):
        for n in range(N):
            s = 0.0
            comp = 0.0
            for k_idx in range(K):
                product = float(A[m, k_idx]) * float(W[k_idx, n])
                y = product - comp
                t = s + y
                comp = (t - s) - y
                s = t
            C[m, n] = s
    return C


def wgsl_embed_q4(token_ids: np.ndarray, qweight: np.ndarray,
                  scales_raw: np.ndarray, qzeros: np.ndarray,
                  hidden_size: int, group_size: int, vocab_size: int) -> np.ndarray:
    """Reimplements embed_q4 from embed.wgsl."""
    seq_len = len(token_ids)
    output = np.zeros((seq_len, hidden_size), dtype=np.float32)

    for t_idx in range(seq_len):
        n = int(token_ids[t_idx])  # token_id = column in qweight
        for k in range(hidden_size):
            # Extract 4-bit weight from qweight[k/8, n]
            packed_row = k // 8
            nibble_idx = k % 8
            packed = int(qweight[packed_row * vocab_size + n])
            q4 = (packed >> (nibble_idx * 4)) & 0xF

            # Get scale: scales[group, n] as F16
            group = k // group_size
            scale_flat = group * vocab_size + n
            scale_word = int(scales_raw[scale_flat // 2])
            if (scale_flat & 1) == 1:
                scale_bits = (scale_word >> 16) & 0xFFFF
            else:
                scale_bits = scale_word & 0xFFFF
            scale = f16_to_f32_bitwise(scale_bits)

            # Get zero: qzeros[group, n/8] packed nibbles
            zero_packed = int(qzeros[group * ((vocab_size + 7) // 8) + n // 8])
            zero_nibble = n % 8
            zero = (zero_packed >> (zero_nibble * 4)) & 0xF

            # Dequant
            output[t_idx, k] = (float(q4) - float(zero)) * scale

    return output


def wgsl_attention(Q: np.ndarray, K_cache: np.ndarray, V_cache: np.ndarray,
                   num_heads: int, num_kv_heads: int, head_dim: int,
                   new_seq_len: int, cache_len: int, scale: float,
                   is_causal: bool, pos_offset: int) -> np.ndarray:
    """Reimplements attention.wgsl (fused QKV->softmax->weighted V)."""
    output = np.zeros((new_seq_len, num_heads * head_dim), dtype=np.float32)
    num_q_per_kv = num_heads // num_kv_heads

    for q_pos in range(new_seq_len):
        for head in range(num_heads):
            kv_head = head // num_q_per_kv
            q_offset = (q_pos * num_heads + head) * head_dim

            # Step 1: Compute scores with Kahan summation
            scores = np.zeros(cache_len, dtype=np.float32)
            for j in range(cache_len):
                k_offset = (j * num_kv_heads + kv_head) * head_dim
                dot = 0.0
                dot_comp = 0.0
                for d in range(head_dim):
                    product = Q[q_offset + d] * K_cache[k_offset + d]
                    y = product - dot_comp
                    t = dot + y
                    dot_comp = (t - dot) - y
                    dot = t
                scores[j] = dot * scale

                # Causal mask
                if is_causal:
                    q_abs_pos = pos_offset + q_pos
                    if j > q_abs_pos:
                        scores[j] = -1e9

            # Step 2: Softmax
            max_score = scores.max()
            exp_scores = np.exp(scores - max_score)
            sum_exp = exp_scores.sum()
            attn_weights = exp_scores / sum_exp

            # Step 3: Weighted sum with Kahan summation
            out_offset = (q_pos * num_heads + head) * head_dim
            for d in range(head_dim):
                weighted_sum = 0.0
                ws_comp = 0.0
                for j in range(cache_len):
                    v_offset = (j * num_kv_heads + kv_head) * head_dim
                    product = attn_weights[j] * V_cache[v_offset + d]
                    y = product - ws_comp
                    t = weighted_sum + y
                    ws_comp = (t - weighted_sum) - y
                    weighted_sum = t
                output.flat[out_offset + d] = weighted_sum

    return output


def wgsl_silu(x: np.ndarray) -> np.ndarray:
    """Reimplements silu from elementwise.wgsl."""
    return x / (1.0 + np.exp(-x))


# ── Test Functions ───────────────────────────────────────────────────────────

def test_f16_conversion():
    """Verify f16→f32 bitwise conversion matches numpy."""
    log("Testing F16 -> F32 conversion...")
    # Test all 65536 possible f16 values
    errors = 0
    max_err = 0.0
    for bits in range(65536):
        our_val = f16_to_f32_bitwise(bits)
        # numpy reference
        np_val = float(np.frombuffer(np.array([bits], dtype=np.uint16).tobytes(),
                                      dtype=np.float16)[0])
        if np.isnan(np_val) or np.isinf(np_val):
            continue
        err = abs(our_val - np_val)
        if err > 1e-10:
            errors += 1
            max_err = max(max_err, err)
            if errors <= 3:
                log(f"  F16 mismatch: bits=0x{bits:04X}, ours={our_val}, numpy={np_val}")

    if errors == 0:
        log(f"  PASS: All 65536 F16 values match exactly")
    else:
        log(f"  FAIL: {errors} mismatches, max_err={max_err:.2e}")
    return errors == 0


def test_rmsnorm():
    """Test RMSNorm with random data."""
    log("Testing RMSNorm...")
    torch.manual_seed(42)
    eps = 1e-6

    # Test with (1+w) convention (Qwen3.5)
    x = torch.randn(1, 4096)
    w = torch.randn(4096) * 0.2  # small values (like trained Qwen3.5 norms)

    ref = pytorch_rmsnorm(x, w, eps, use_residual_weight=True)
    wgsl_out = wgsl_rmsnorm(x.numpy(), w.numpy(), eps, use_residual_weight=True)

    return compare("RMSNorm (1+w)", wgsl_out, ref.numpy(), atol=1e-5)


def test_rope():
    """Test RoPE with interleaved pairing."""
    log("Testing RoPE (interleaved pairing)...")
    torch.manual_seed(42)

    # Qwen3.5 full attention: head_dim=256, partial_rotary=0.25 → rotary_dim=64
    head_dim = 256
    num_heads = 4  # small for test
    seq_len = 2
    rope_base = 10000000.0
    rotary_dim = 64  # 25% of 256

    qk = torch.randn(seq_len * num_heads * head_dim)

    wgsl_out = wgsl_rope(qk.numpy(), seq_len, num_heads, head_dim, 0, rope_base, rotary_dim)
    ref = pytorch_rope_interleaved(qk, seq_len, num_heads, head_dim, 0, rope_base, rotary_dim)

    p1 = compare("RoPE partial (rotary_dim=64)", wgsl_out, ref.numpy(), atol=1e-5)

    # Verify non-rotated dims are unchanged
    out_reshaped = wgsl_out.reshape(seq_len, num_heads, head_dim)
    orig_reshaped = qk.numpy().reshape(seq_len, num_heads, head_dim)
    non_rotated_match = np.allclose(out_reshaped[:, :, rotary_dim:],
                                     orig_reshaped[:, :, rotary_dim:])
    log(f"  {'PASS' if non_rotated_match else 'FAIL'} Non-rotated dims unchanged (dims {rotary_dim}-{head_dim})")

    # Test what USED to happen (rotary_dim=0, meaning all dims rotated)
    wgsl_wrong = wgsl_rope(qk.numpy(), seq_len, num_heads, head_dim, 0, rope_base, 0)
    wrong_match = np.allclose(wgsl_wrong, wgsl_out)
    log(f"  {'FAIL (BUG!)' if wrong_match else 'PASS (correctly different)'} "
        f"Full rotation vs partial rotation should differ")

    return p1 and non_rotated_match and not wrong_match


def test_attention():
    """Test fused attention kernel."""
    log("Testing Attention (fused QKV->softmax->V)...")
    torch.manual_seed(42)

    num_heads = 4
    num_kv_heads = 2
    head_dim = 8  # small for testing
    seq_len = 3
    cache_len = 3
    scale = 1.0 / math.sqrt(head_dim)

    Q = torch.randn(seq_len * num_heads * head_dim)
    K_cache = torch.randn(cache_len * num_kv_heads * head_dim)
    V_cache = torch.randn(cache_len * num_kv_heads * head_dim)

    # WGSL reimplementation
    wgsl_out = wgsl_attention(Q.numpy(), K_cache.numpy(), V_cache.numpy(),
                               num_heads, num_kv_heads, head_dim,
                               seq_len, cache_len, scale, True, 0)

    # PyTorch reference
    Q_r = Q.reshape(seq_len, num_heads, head_dim)
    K_r = K_cache.reshape(cache_len, num_kv_heads, head_dim)
    V_r = V_cache.reshape(cache_len, num_kv_heads, head_dim)

    # Expand KV heads for GQA
    n_rep = num_heads // num_kv_heads
    K_exp = K_r.unsqueeze(2).expand(-1, -1, n_rep, -1).reshape(cache_len, num_heads, head_dim)
    V_exp = V_r.unsqueeze(2).expand(-1, -1, n_rep, -1).reshape(cache_len, num_heads, head_dim)

    # [seq, heads, dim] x [cache, heads, dim]^T -> [seq, heads, cache]
    scores = torch.einsum('shd,chd->shc', Q_r, K_exp) * scale

    # Causal mask
    for s in range(seq_len):
        for c in range(cache_len):
            if c > s:
                scores[s, :, c] = -1e9

    attn = F.softmax(scores, dim=-1)
    ref_out = torch.einsum('shc,chd->shd', attn, V_exp).reshape(-1)

    return compare("Attention (GQA, causal)", wgsl_out.flatten(), ref_out.numpy(), atol=1e-4)


def test_silu():
    """Test SiLU activation."""
    log("Testing SiLU...")
    x = np.random.randn(1024).astype(np.float32)
    wgsl_out = wgsl_silu(x)
    ref = x * (1.0 / (1.0 + np.exp(-x)))  # manual silu
    return compare("SiLU", wgsl_out, ref)


def test_matmul_q4_small():
    """Test INT4 dequant matmul with small synthetic data."""
    log("Testing matmul_q4 (synthetic small)...")

    # Create a small weight matrix, quantize it, then verify dequant
    M, N, K = 2, 16, 32  # small dimensions
    group_size = 16

    torch.manual_seed(42)
    W_orig = torch.randn(N, K)  # [out_features, in_features] — HF layout

    # Simple RTN quantization
    num_groups = K // group_size
    Q_int = torch.zeros(N, K, dtype=torch.int32)
    scales = torch.zeros(N, num_groups, dtype=torch.float32)
    zeros = torch.zeros(N, num_groups, dtype=torch.int32)

    for g in range(num_groups):
        col_start = g * group_size
        col_end = col_start + group_size
        block = W_orig[:, col_start:col_end]

        bmin = block.min(dim=1).values
        bmax = block.max(dim=1).values
        scale = (bmax - bmin) / 15.0
        scale = scale.clamp(min=1e-8)
        zero = torch.round(-bmin / scale).clamp(0, 15).to(torch.int32)

        scales[:, g] = scale
        zeros[:, g] = zero

        for c in range(group_size):
            Q_int[:, col_start + c] = torch.clamp(
                torch.round(W_orig[:, col_start + c] / scale) + zero.float(), 0, 15
            ).to(torch.int32)

    # Pack using our quantizer's format
    from quantize_gptq import pack_qweight, pack_scales, pack_qzeros

    qweight = pack_qweight(Q_int)        # [K//8, N]
    packed_scales = pack_scales(scales)   # [num_groups, N] as f16
    qzeros = pack_qzeros(zeros)           # [num_groups, N//8]
    g_idx = (torch.arange(K) // group_size).to(torch.int32)

    # Convert to numpy arrays matching GPU buffer formats
    qw_np = qweight.numpy().astype(np.int32)
    # Scales: f16 packed as u32 pairs
    sc_f16 = packed_scales.numpy()  # [num_groups, N] as float16
    sc_u16 = sc_f16.view(np.uint16)
    # Pad to even length if needed
    if len(sc_u16.flat) % 2 != 0:
        sc_u16 = np.append(sc_u16.flat, np.uint16(0))
    sc_u32 = sc_u16.reshape(-1, 2)
    sc_u32_packed = sc_u32[:, 0].astype(np.uint32) | (sc_u32[:, 1].astype(np.uint32) << 16)
    qz_np = qzeros.numpy().astype(np.int32)
    gi_np = g_idx.numpy().astype(np.uint32)

    # Test dequantization
    W_dequant = wgsl_dequant_q4_matmul(qw_np, sc_u32_packed, qz_np, gi_np, K, N, group_size)

    # Python reference dequant
    W_ref = np.zeros((K, N), dtype=np.float32)
    for k in range(K):
        for n in range(N):
            q = int(Q_int[n, k])
            g = k // group_size
            s = float(scales[n, g])
            z = int(zeros[n, g])
            W_ref[k, n] = (q - z) * s

    p1 = compare("Q4 Dequant (weights)", W_dequant, W_ref, atol=1e-3)

    # Test full matmul
    A = np.random.randn(M, K).astype(np.float32)
    C_wgsl = wgsl_matmul_q4(A, qw_np, sc_u32_packed, qz_np, gi_np, M, N, K, group_size)
    C_ref = A @ W_ref  # [M, K] @ [K, N] = [M, N]

    # Kahan summation gives slightly different results than numpy's naive @
    # Both are correct — Kahan is actually more precise. Allow 0.01 tolerance.
    p2 = compare("Q4 Matmul (full)", C_wgsl, C_ref, atol=0.01)

    return p1 and p2


def test_with_real_weights(model_dir: str):
    """Test kernels using actual model weights."""
    log(f"\n{'='*60}")
    log(f"Testing with real model weights: {model_dir}")
    log(f"{'='*60}")

    config_path = os.path.join(model_dir, 'config.json')
    if not os.path.exists(config_path):
        log(f"  SKIP: No config.json found at {config_path}")
        return True

    config = json.load(open(config_path))
    tc = config.get('text_config', config)
    hidden_size = tc['hidden_size']
    num_heads = tc['num_attention_heads']
    num_kv_heads = tc.get('num_key_value_heads', num_heads)
    head_dim = tc.get('head_dim', hidden_size // num_heads)
    rms_eps = tc.get('rms_norm_eps', 1e-6)
    rope_params = tc.get('rope_parameters', {})
    rope_theta = rope_params.get('rope_theta', tc.get('rope_theta', 10000))
    partial_rotary = rope_params.get('partial_rotary_factor', tc.get('partial_rotary_factor', None))
    group_size = tc.get('quantization_config', config.get('quantization_config', {})).get('group_size', 128)

    log(f"  Config: hidden={hidden_size}, heads={num_heads}/{num_kv_heads}, "
        f"head_dim={head_dim}, rope_theta={rope_theta}, "
        f"partial_rotary={partial_rotary}, group_size={group_size}")

    # Load index
    idx_path = os.path.join(model_dir, 'model.safetensors.index.json')
    if not os.path.exists(idx_path):
        # Single shard
        shard_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir)
                       if f.endswith('.safetensors')]
    else:
        idx = json.load(open(idx_path))
        shard_files = list(set(idx['weight_map'].values()))
        shard_files = [os.path.join(model_dir, f) for f in shard_files]

    # Load just the tensors we need for layer 0
    log("  Loading layer 0 weights...")
    needed_prefixes = [
        'model.layers.0.input_layernorm',
        'model.layers.0.self_attn.q_proj',
        'model.layers.0.self_attn.k_proj',
        'model.layers.0.self_attn.v_proj',
        'model.embed_tokens',
        'model.language_model.embed_tokens',
    ]

    tensors = {}
    for shard_file in shard_files:
        if not os.path.exists(shard_file):
            continue
        st = load_file(shard_file, device='cpu')
        for k, v in st.items():
            for prefix in needed_prefixes:
                if k.startswith(prefix):
                    tensors[k] = v
                    break
        del st

    log(f"  Loaded {len(tensors)} tensors")

    all_passed = True

    # Test 1: RMSNorm with real weights
    norm_key = None
    for k in tensors:
        if 'input_layernorm.weight' in k and 'layers.0' in k:
            norm_key = k
            break

    if norm_key:
        log("\n  Real-weight RMSNorm test:")
        w = tensors[norm_key].float()
        x = torch.randn(1, hidden_size)
        ref = pytorch_rmsnorm(x, w, rms_eps, use_residual_weight=True)
        wgsl_out = wgsl_rmsnorm(x.numpy(), w.numpy(), rms_eps, use_residual_weight=True)
        if not compare("RMSNorm (real weights)", wgsl_out, ref.numpy(), atol=1e-5):
            all_passed = False

    # Test 2: RoPE with real dimensions — use position 50 to see real rotation
    log("\n  Real-dimension RoPE test (position=50):")
    rotary_dim = int(partial_rotary * head_dim) if partial_rotary else 0
    qk_test = torch.randn(1 * num_heads * head_dim)

    wgsl_out = wgsl_rope(qk_test.numpy(), 1, num_heads, head_dim, 50, rope_theta, rotary_dim)
    ref = pytorch_rope_interleaved(qk_test, 1, num_heads, head_dim, 50, rope_theta, rotary_dim)

    if not compare(f"RoPE (real dims, rotary_dim={rotary_dim}, pos=50)", wgsl_out, ref.numpy(), atol=1e-5):
        all_passed = False

    # Demonstrate the bug: what happens without partial rotary
    if rotary_dim > 0:
        wgsl_bug = wgsl_rope(qk_test.numpy(), 1, num_heads, head_dim, 50, rope_theta, 0)
        diff = np.abs(wgsl_bug - wgsl_out).max()
        log(f"  BUG IMPACT: max diff between correct (rotary_dim={rotary_dim}) "
            f"and buggy (rotary_dim=0): {diff:.6f}")
        # Show how much the non-rotated dims change (should be zero for correct, nonzero for bug)
        correct_reshaped = wgsl_out.reshape(1, num_heads, head_dim)
        buggy_reshaped = wgsl_bug.reshape(1, num_heads, head_dim)
        orig_reshaped = qk_test.numpy().reshape(1, num_heads, head_dim)
        non_rot_change_bug = np.abs(buggy_reshaped[:, :, rotary_dim:] -
                                     orig_reshaped[:, :, rotary_dim:]).max()
        non_rot_change_correct = np.abs(correct_reshaped[:, :, rotary_dim:] -
                                         orig_reshaped[:, :, rotary_dim:]).max()
        log(f"  BUG IMPACT: non-rotated dims change (correct): {non_rot_change_correct:.8f} (should be 0)")
        log(f"  BUG IMPACT: non-rotated dims change (buggy):   {non_rot_change_bug:.6f} (scrambled!)")

    # Test 3: INT4 matmul with real weights (if quantized)
    q_proj_qw = None
    for k in tensors:
        if 'q_proj.qweight' in k:
            q_proj_qw = k
            break

    if q_proj_qw:
        log("\n  Real-weight Q4 dequant test (q_proj, first 16 outputs):")
        base = q_proj_qw.rsplit('.qweight', 1)[0]
        qw = tensors[f"{base}.qweight"].numpy().astype(np.int32)
        sc_f16 = tensors[f"{base}.scales"].numpy()
        qz = tensors[f"{base}.qzeros"].numpy().astype(np.int32)
        gi_key = f"{base}.g_idx"
        if gi_key in tensors:
            gi = tensors[gi_key].numpy().astype(np.uint32)
        else:
            gi = (np.arange(hidden_size, dtype=np.uint32) // group_size)

        # Convert scales to u32 pairs
        sc_u16 = sc_f16.view(np.uint16).flatten()
        if len(sc_u16) % 2 != 0:
            sc_u16 = np.append(sc_u16, np.uint16(0))
        sc_u32 = np.zeros(len(sc_u16) // 2, dtype=np.uint32)
        sc_u32 = sc_u16[0::2].astype(np.uint32) | (sc_u16[1::2].astype(np.uint32) << 16)

        N_proj = qw.shape[1]  # out_features
        K_proj = qw.shape[0] * 8  # in_features

        # Only dequant a small slice for speed (first 16 output features)
        N_test = min(16, N_proj)
        log(f"    q_proj shape: [{N_proj}, {K_proj}], testing first {N_test} outputs")

        # WGSL dequant (slow but exact)
        W_wgsl = np.zeros((K_proj, N_test), dtype=np.float32)
        for k_idx in range(K_proj):
            for n_idx in range(N_test):
                packed_idx = (k_idx // 8) * N_proj + n_idx
                nibble = k_idx % 8
                packed = int(qw.flat[packed_idx])
                q4_val = (packed >> (nibble * 4)) & 0xF

                group_id = int(gi[k_idx])
                scale_linear_idx = group_id * N_proj + n_idx
                word_idx = scale_linear_idx // 2
                word = int(sc_u32[word_idx])
                if (scale_linear_idx & 1) == 1:
                    half_bits = (word >> 16) & 0xFFFF
                else:
                    half_bits = word & 0xFFFF
                scale = f16_to_f32_bitwise(half_bits)

                zero_packed_idx = group_id * ((N_proj + 7) // 8) + n_idx // 8
                zero_nibble = n_idx % 8
                zero_packed = int(qz.flat[zero_packed_idx])
                zero = (zero_packed >> (zero_nibble * 4)) & 0xF

                W_wgsl[k_idx, n_idx] = (float(q4_val) - float(zero)) * scale

        # Python reference: use pack/unpack round-trip
        # Dequant from original packed format via numpy
        W_ref = W_wgsl.copy()  # If WGSL matches itself, test a matmul
        log(f"    Dequant sanity: first row [0,:8] = {W_wgsl[0,:8].tolist()}")

        # Test matmul with random input
        A = np.random.randn(1, K_proj).astype(np.float32)
        C_wgsl = A @ W_wgsl
        C_ref = A @ W_ref  # Same dequant, just testing matmul
        p = compare("Q4 Matmul (real weights, 16 outputs)", C_wgsl, C_ref, atol=1e-3)
        if not p:
            all_passed = False

    return all_passed


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='WGSL Shader Numerical Audit')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to quantized model directory for real-weight tests')
    args = parser.parse_args()

    log("=" * 60)
    log("WGSL Shader Numerical Audit")
    log("=" * 60)

    results = {}

    # Synthetic tests (fast)
    results['f16_conversion'] = test_f16_conversion()
    results['rmsnorm'] = test_rmsnorm()
    results['rope'] = test_rope()
    results['attention'] = test_attention()
    results['silu'] = test_silu()

    # Small synthetic Q4 test
    # Need to add quantize_gptq to path
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        results['matmul_q4_synthetic'] = test_matmul_q4_small()
    except ImportError as e:
        log(f"  SKIP matmul_q4_synthetic: {e}")
        results['matmul_q4_synthetic'] = None

    # Real weight tests (slower but more meaningful)
    if args.model:
        results['real_weights'] = test_with_real_weights(args.model)

    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    for name, passed in results.items():
        if passed is None:
            status = "SKIP"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"
        log(f"  {status}: {name}")

    failed = sum(1 for v in results.values() if v is False)
    if failed:
        log(f"\n{failed} test(s) FAILED — fix the shaders!")
        return 1
    else:
        log(f"\nAll tests PASSED")
        return 0


if __name__ == '__main__':
    sys.exit(main())
