"""l21_bias_fingerprint.py — search for a weight-corruption hypothesis that
predicts the exact observed shift.

Observed (engine vs PyTorch ref, L21 mlp.up_proj, prefill-end last token):
  engine:  mean=-0.9258  absMean=1.0285
  ref:     mean=-0.0032  absMean=0.5527

Absolute differences:
  Δmean   = -0.9226
  Δabsmean = +0.4758

We simulate candidate kernel bugs by reconstructing W_up's dequant under each
perturbation, multiplying by a simulated input (drawn from ref layer-20-out
distribution: mean≈0, absMean≈0.52), and checking which candidate reproduces
both Δmean≈-0.92 and absMean≈1.03.

Candidate bugs considered:
  (a) missing `- zero` on all columns (out = sum w_int * scale, no zero-subtract)
  (b) using zero=0 (or zero=1) for all groups/cols (i.e. zeros buffer zeroed)
  (c) using a different tensor's qzeros buffer (row-offset bug)
  (d) kernel is correct; weights in VRAM have a specific corruption pattern
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from audit_weight_tensor import load_tensor  # noqa: E402


def dequant_with_zero_fn(qweight, scales, qzeros, g_idx, zero_fn):
    """Generic dequant where zero_fn(zeros_int) returns modified zeros_int.
    Returns [in_features, out_features] fp32.
    """
    qw = qweight.to(torch.int64)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    w_int = ((qw.unsqueeze(1) >> shifts.view(1, -1, 1)) & 0xF).to(torch.int32)
    in_f = qweight.shape[0] * 8
    out_f = qweight.shape[1]
    w_int = w_int.reshape(in_f, out_f)

    qz = qzeros.to(torch.int64)
    z_int = ((qz.unsqueeze(2) >> shifts.view(1, 1, -1)) & 0xF).to(torch.int32)
    z_int = z_int.reshape(qzeros.shape[0], out_f)
    z_int = zero_fn(z_int)

    sc_f = scales.to(torch.float32)
    z_f = z_int.to(torch.float32)
    gi_long = g_idx.to(torch.long)
    sel_sc = sc_f[gi_long]
    sel_z = z_f[gi_long]
    return (w_int.to(torch.float32) - sel_z) * sel_sc  # [in, out]


def compute_out_stats(W_in_out, input_vec):
    """input_vec [in_f] × W_in_out.T [out_f, in_f] = [out_f]."""
    out = input_vec @ W_in_out  # [in] @ [in, out] = [out]
    return out.mean().item(), out.abs().mean().item(), out.std().item()


def main() -> int:
    qm = Path("models/qwen3.5-9b-HailMary")
    stem = "model.layers.21.mlp.up_proj"

    qw = load_tensor(qm, f"{stem}.qweight")
    sc = load_tensor(qm, f"{stem}.scales")
    qz = load_tensor(qm, f"{stem}.qzeros")
    gi = load_tensor(qm, f"{stem}.g_idx")

    # Synthetic input: draw from distribution with stats ≈ layer-20-out sample.
    # (Use mean≈0, absMean≈0.52, and random signs — this is what RMSnorm output
    # typically looks like.)
    torch.manual_seed(42)
    in_f = qw.shape[0] * 8  # 4096
    # Scale factor so E[|x|] ≈ 0.52 with N(0, σ²) → E[|x|] = σ*√(2/π), so σ ≈ 0.65
    x = torch.randn(in_f, dtype=torch.float32) * 0.65

    targets = {
        "observed engine L21.up-out":  (-0.9258, 1.0285),
        "observed ref    L21.up-out":  (-0.0032, 0.5527),
    }
    print("Targets (mean, absMean):")
    for k, v in targets.items():
        print(f"  {k}: mean={v[0]:+.4f} absMean={v[1]:.4f}")
    print()

    candidates = [
        ("CORRECT (baseline, engine matches ref if kernel+weights fine)",
         lambda z: z),
        ("Bug A: missing `-zero` entirely (zeros forced to 0)",
         lambda z: torch.zeros_like(z)),
        ("Bug A': all zeros = 1 (i.e. re-adds +1 then drops zero)",
         lambda z: torch.ones_like(z)),
        ("Bug B: all zeros = 8 (int4 mid; zeroes decode bias)",
         lambda z: torch.full_like(z, 8)),
        ("Bug C: kernel decodes zero_nibble via wrong bit-layout (shift*col vs shift*(col%8))",
         lambda z: ((z.to(torch.int64).unsqueeze(-1) >> torch.arange(0,32,4)).to(torch.int64) & 0xF).to(torch.int32)[..., 0]),  # placeholder
        ("Bug D: zeros from DIFFERENT tensor (use L21.gate_proj qzeros instead)",
         None),  # handled specially below
        ("Bug E: scales doubled (would flip absMean 2x)",
         None),  # handled by scaling W
    ]

    # Baseline correct
    W = dequant_with_zero_fn(qw, sc, qz, gi, lambda z: z)
    m, am, sd = compute_out_stats(W, x)
    print(f"CORRECT baseline        : mean={m:+.4f} absMean={am:.4f} std={sd:.4f}")

    # Bug A
    W = dequant_with_zero_fn(qw, sc, qz, gi, lambda z: torch.zeros_like(z))
    m, am, sd = compute_out_stats(W, x)
    print(f"Bug A (zeros=0)         : mean={m:+.4f} absMean={am:.4f} std={sd:.4f}")

    # Bug A2
    W = dequant_with_zero_fn(qw, sc, qz, gi, lambda z: torch.full_like(z, 1))
    m, am, sd = compute_out_stats(W, x)
    print(f"Bug A' (zeros=1)        : mean={m:+.4f} absMean={am:.4f} std={sd:.4f}")

    # Bug B
    W = dequant_with_zero_fn(qw, sc, qz, gi, lambda z: torch.full_like(z, 8))
    m, am, sd = compute_out_stats(W, x)
    print(f"Bug B (zeros=8)         : mean={m:+.4f} absMean={am:.4f} std={sd:.4f}")

    # Bug D: swap in gate_proj's qzeros (same layer, sibling tensor)
    gate_stem = "model.layers.21.mlp.gate_proj"
    qz_gate = load_tensor(qm, f"{gate_stem}.qzeros")
    W = dequant_with_zero_fn(qw, sc, qz_gate, gi, lambda z: z)
    m, am, sd = compute_out_stats(W, x)
    print(f"Bug D (use gate qzeros) : mean={m:+.4f} absMean={am:.4f} std={sd:.4f}")

    # Bug D2: swap in L20.up's qzeros
    qz_l20 = load_tensor(qm, "model.layers.20.mlp.up_proj.qzeros")
    W = dequant_with_zero_fn(qw, sc, qz_l20, gi, lambda z: z)
    m, am, sd = compute_out_stats(W, x)
    print(f"Bug D2 (use L20.up qzeros): mean={m:+.4f} absMean={am:.4f} std={sd:.4f}")

    # Bug E: scales doubled
    W = dequant_with_zero_fn(qw, sc * 2.0, qz, gi, lambda z: z)
    m, am, sd = compute_out_stats(W, x)
    print(f"Bug E (scales×2)        : mean={m:+.4f} absMean={am:.4f} std={sd:.4f}")

    # Bug F: scales from different tensor
    sc_gate = load_tensor(qm, f"{gate_stem}.scales")
    W = dequant_with_zero_fn(qw, sc_gate, qz, gi, lambda z: z)
    m, am, sd = compute_out_stats(W, x)
    print(f"Bug F (use gate scales) : mean={m:+.4f} absMean={am:.4f} std={sd:.4f}")

    # Bug G: qweight from different tensor (would produce wildly different output)
    qw_gate = load_tensor(qm, f"{gate_stem}.qweight")
    W = dequant_with_zero_fn(qw_gate, sc, qz, gi, lambda z: z)
    m, am, sd = compute_out_stats(W, x)
    print(f"Bug G (use gate qweight): mean={m:+.4f} absMean={am:.4f} std={sd:.4f}")

    # Bug H: g_idx from different tensor
    gi_l20 = load_tensor(qm, "model.layers.20.mlp.up_proj.g_idx")
    W = dequant_with_zero_fn(qw, sc, qz, gi_l20, lambda z: z)
    m, am, sd = compute_out_stats(W, x)
    print(f"Bug H (use L20 g_idx)   : mean={m:+.4f} absMean={am:.4f} std={sd:.4f}")

    # Bug I: weights sign-flipped (nibble = 15 - w_int)
    class FlipZero:
        def __call__(self, z):
            return z  # keep zeros same, but flip w by returning 15-w elsewhere — handled differently
    # simulate: W_flipped = (15 - w_int - z) * s = -((w_int - (15-z)) * s) = -(correct with zeros=15-z)
    W = dequant_with_zero_fn(qw, sc, qz, gi, lambda z: 15 - z)
    # also flip w by negation: we skip that since dequant_with_zero_fn doesn't touch w
    # Instead, compute (-(w_int) - z)*s effectively — easier: just report what happens
    m, am, sd = compute_out_stats(W, x)
    print(f"Bug I (zeros = 15-z)    : mean={m:+.4f} absMean={am:.4f} std={sd:.4f}")

    print()
    print("Which candidate best matches engine (-0.9258, 1.0285) while baseline matches ref (≈0, 0.55)?")

    return 0


if __name__ == "__main__":
    sys.exit(main())
