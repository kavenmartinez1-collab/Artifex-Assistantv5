"""l21_structural_probe.py — compare L21.up_proj vs L21.gate_proj packed-tensor
structure.

Engine output: L21.gate_proj is correct (cos=0.86 vs ref BF16, stats match).
L21.up_proj is broken (mean=-0.93 vs ref ~0, absMean 1.86x inflated). Both take
the same input (post-FFN-RMSnorm), both dispatch through the same code in
forward-pass.ts (dispatchProjection → tryGetQ4 → matmul_q4 GEMM or GEMV).

Before a VRAM readback, sanity-check: does up_proj's packed data have any
structural peculiarity vs gate_proj that would trip a latent kernel edge case?
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from audit_weight_tensor import load_tensor  # noqa: E402


def unpack_i4(packed: torch.Tensor, rows_major: bool) -> torch.Tensor:
    """Unpack a [*, ...] int32 tensor with 8 nibbles per u32.

    rows_major=True means qweight layout (8 input rows packed into 1 u32 along dim0).
    rows_major=False means qzeros layout (8 output cols packed into 1 u32 along dim1).
    """
    qi = packed.to(torch.int64)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    if rows_major:
        vals = ((qi.unsqueeze(1) >> shifts.view(1, -1, 1)) & 0xF).to(torch.int32)
        return vals.reshape(packed.shape[0] * 8, packed.shape[1])
    vals = ((qi.unsqueeze(2) >> shifts.view(1, 1, -1)) & 0xF).to(torch.int32)
    return vals.reshape(packed.shape[0], packed.shape[1] * 8)


def summarize(stem_label: str, qw, sc, qz, gi) -> None:
    print(f"=== {stem_label} ===")
    print(f"  qweight shape={list(qw.shape)} dtype={qw.dtype}")
    print(f"  scales  shape={list(sc.shape)} dtype={sc.dtype}  "
          f"min={sc.float().min().item():.6e} max={sc.float().max().item():.6e} "
          f"mean={sc.float().mean().item():.6e} absMean={sc.float().abs().mean().item():.6e}")
    print(f"  qzeros  shape={list(qz.shape)} dtype={qz.dtype}")
    print(f"  g_idx   shape={list(gi.shape)} dtype={gi.dtype}  "
          f"min={gi.min().item()} max={gi.max().item()} "
          f"unique={gi.unique().numel()}")

    # qweight nibble histogram
    w_nib = unpack_i4(qw, rows_major=True).flatten()
    w_hist = torch.bincount(w_nib, minlength=16)
    total = w_hist.sum().item()
    print(f"  qweight nibble hist (fraction):")
    print(f"    " + " ".join(f"{v}:{(w_hist[v].item()/total*100):.2f}%" for v in range(16)))

    # qzeros nibble histogram
    z_nib = unpack_i4(qz, rows_major=False).flatten()
    z_hist = torch.bincount(z_nib, minlength=16)
    z_tot = z_hist.sum().item()
    print(f"  qzeros nibble hist (fraction):")
    print(f"    " + " ".join(f"{v}:{(z_hist[v].item()/z_tot*100):.2f}%" for v in range(16)))

    # scales per-group stats — per-output-col range & mean
    # scales is [num_groups, out_features]
    sc_f = sc.float()
    col_max = sc_f.max(dim=0).values
    col_min = sc_f.min(dim=0).values
    col_mean = sc_f.mean(dim=0)
    print(f"  per-output-col scale max-over-groups: min={col_max.min().item():.4e} "
          f"max={col_max.max().item():.4e} mean={col_max.mean().item():.4e}")
    print(f"  per-output-col scale range (max/min): mean={(col_max/col_min.clamp(min=1e-30)).mean().item():.4f} "
          f"max={(col_max/col_min.clamp(min=1e-30)).max().item():.4f}")

    # g_idx basic histogram
    gi_f = gi.to(torch.int64)
    gi_hist = torch.bincount(gi_f, minlength=int(gi_f.max().item())+1)
    print(f"  g_idx histogram: min_count_per_group={gi_hist.min().item()} "
          f"max_count_per_group={gi_hist.max().item()} "
          f"(perfectly-uniform would be {gi.numel() // (gi_f.max().item()+1)})")

    # g_idx first 16 values
    print(f"  g_idx[:16] = {gi[:16].tolist()}")
    print(f"  g_idx[128:144] = {gi[128:144].tolist()}")
    print()


def main() -> int:
    qm = Path("models/qwen3.5-9b-HailMary")

    specs = [
        ("L21.mlp.gate_proj (known good)", "model.layers.21.mlp.gate_proj"),
        ("L21.mlp.up_proj   (BROKEN)",     "model.layers.21.mlp.up_proj"),
        ("L21.mlp.down_proj (known bad)",  "model.layers.21.mlp.down_proj"),
        ("L20.mlp.up_proj   (fine)",       "model.layers.20.mlp.up_proj"),
        ("L22.mlp.up_proj   (fine)",       "model.layers.22.mlp.up_proj"),
    ]

    for label, stem in specs:
        try:
            qw = load_tensor(qm, f"{stem}.qweight")
            sc = load_tensor(qm, f"{stem}.scales")
            qz = load_tensor(qm, f"{stem}.qzeros")
            gi = load_tensor(qm, f"{stem}.g_idx")
        except KeyError as e:
            print(f"{label} :: MISSING ({e})")
            continue
        summarize(label, qw, sc, qz, gi)

    return 0


if __name__ == "__main__":
    sys.exit(main())
