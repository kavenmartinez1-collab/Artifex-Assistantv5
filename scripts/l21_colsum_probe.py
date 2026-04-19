"""l21_colsum_probe.py — column-sum analysis.

Hypothesis: input to L21 MLP has a small mean bias c. Gate_out shift = c * E[colsum(W_gate)].
Up_out shift = c * E[colsum(W_up)]. If W_up has systematically negative column
sums (~something × 10 larger than W_gate's), then the same input bias
produces a tiny shift in gate_out but a huge one in up_out.

Observed delta from ref:
  gate: Δmean ≈ -0.02  absmean same
  up:   Δmean ≈ -0.92  absmean +0.48 (near doubled)

Tests:
  1. Per-column sums of dequanted W_gate and W_up.
  2. Distribution mean + variance of colsum.
  3. If column sums differ by ~46×, that proves input-bias-sensitivity alone.
  4. Also compute: for each output col j, bias_coef[j] = sum_k W[k,j]. Predict
     up_out_shift = c * bias_coef. Find c that best fits engine - ref shifts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from audit_weight_tensor import load_tensor  # noqa: E402


def dequant_full(qweight, scales, qzeros, g_idx) -> torch.Tensor:
    qw = qweight.to(torch.int64)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    w_int = ((qw.unsqueeze(1) >> shifts.view(1, -1, 1)) & 0xF).to(torch.int32)
    in_f = qweight.shape[0] * 8
    out_f = qweight.shape[1]
    w_int = w_int.reshape(in_f, out_f)

    qz = qzeros.to(torch.int64)
    z_int = ((qz.unsqueeze(2) >> shifts.view(1, 1, -1)) & 0xF).to(torch.int32)
    z_int = z_int.reshape(qzeros.shape[0], out_f)

    sc_f = scales.to(torch.float32)
    z_f = z_int.to(torch.float32)
    gi_long = g_idx.to(torch.long)
    return (w_int.to(torch.float32) - z_f[gi_long]) * sc_f[gi_long]  # [in, out]


def analyze(label: str, W_in_out: torch.Tensor) -> torch.Tensor:
    colsum = W_in_out.sum(dim=0)  # [out]
    print(f"{label}: shape={list(W_in_out.shape)}")
    print(f"  elementwise: mean={W_in_out.mean().item():+.6e} absMean={W_in_out.abs().mean().item():.6e}")
    print(f"  column sums: mean={colsum.mean().item():+.6f} std={colsum.std().item():.6f} "
          f"min={colsum.min().item():+.4f} max={colsum.max().item():+.4f}")
    print(f"  column-sum absMean={colsum.abs().mean().item():.4f}")
    return colsum


def main() -> int:
    qm = Path("models/qwen3.5-9b-HailMary")

    print("=== L21 MLP projections (on-disk dequant) ===\n")

    specs = [
        ("L21.mlp.gate_proj (correct)", "model.layers.21.mlp.gate_proj"),
        ("L21.mlp.up_proj   (BROKEN)",  "model.layers.21.mlp.up_proj"),
        ("L20.mlp.up_proj   (fine)",    "model.layers.20.mlp.up_proj"),
        ("L22.mlp.up_proj   (fine)",    "model.layers.22.mlp.up_proj"),
        ("L27.mlp.up_proj   (fine)",    "model.layers.27.mlp.up_proj"),
        ("L28.mlp.up_proj   (fine)",    "model.layers.28.mlp.up_proj"),
        ("L31.mlp.up_proj   (fine)",    "model.layers.31.mlp.up_proj"),
    ]

    colsums = {}
    for label, stem in specs:
        try:
            qw = load_tensor(qm, f"{stem}.qweight")
            sc = load_tensor(qm, f"{stem}.scales")
            qz = load_tensor(qm, f"{stem}.qzeros")
            gi = load_tensor(qm, f"{stem}.g_idx")
        except KeyError as e:
            print(f"{label}: MISSING ({e})")
            continue
        W = dequant_full(qw, sc, qz, gi)
        colsums[label] = analyze(label, W)
        print()

    print("=" * 70)
    print("Summary: which tensor has uniquely-negative column sums?")
    print(f"{'tensor':<40} {'colsum mean':>15} {'colsum absMean':>16}")
    for lbl, cs in colsums.items():
        print(f"{lbl:<40} {cs.mean().item():>+15.4f} {cs.abs().mean().item():>16.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
