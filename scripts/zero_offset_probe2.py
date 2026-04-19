"""zero_offset_probe2.py — deeper check on the zero_offset_probe result.

cos(no+1) = 1.0022 for L21.up_proj is too high. Real INT4 quant should be
cos ≈ 0.97-0.99 at best. Options:
  (a) reference model (models/qwen3.5-9b) is not the BF16 base we assume —
      maybe it's the same quant dequantized, so dequant-vs-ref trivially matches.
  (b) WGSL-style decode is *accidentally* correct and HailMary quant is extremely
      lossless (unlikely for INT4).
  (c) the `> 1.0` cosine is an FP artifact of large dot products in fp32.

Check:
  1. Print dtype, shape, and first-element of ref weight.
  2. Compute deq - ref for L21.up_proj in both decode modes, report distribution.
  3. Compute the no+1 reconstruction error as: for each group g, within rows mapped
     to group g, norm(deq_row - ref_row) / norm(ref_row). If it's near zero for all
     groups, HailMary really is near-lossless. If error correlates with group index,
     we might be hitting a trivial-match edge case.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from audit_weight_tensor import load_tensor  # noqa: E402
from zero_offset_probe import dequant  # noqa: E402


def main() -> int:
    qm = Path("models/qwen3.5-9b-HailMary")
    rm = Path("models/qwen3.5-9b")

    stem = "model.layers.21.mlp.up_proj"
    qw = load_tensor(qm, f"{stem}.qweight")
    sc = load_tensor(qm, f"{stem}.scales")
    qz = load_tensor(qm, f"{stem}.qzeros")
    gi = load_tensor(qm, f"{stem}.g_idx")
    ref = load_tensor(rm, f"{stem}.weight")

    print(f"ref dtype = {ref.dtype}, shape = {list(ref.shape)}")
    print(f"ref first 8 elements flat: {ref.flatten()[:8].tolist()}")
    print(f"ref abs mean = {ref.float().abs().mean().item():.6f}")
    print(f"ref min/max = {ref.float().min().item():.6f} / {ref.float().max().item():.6f}")

    deq_raw = dequant(qw, sc, qz, gi, bits=4, add_one_to_zero=False).T  # [out, in]
    deq_p1  = dequant(qw, sc, qz, gi, bits=4, add_one_to_zero=True).T

    diff_raw = (deq_raw.float() - ref.float())
    diff_p1  = (deq_p1.float() - ref.float())

    print()
    print(f"no+1 decode:  abs mean diff = {diff_raw.abs().mean().item():.6e}  "
          f"max abs diff = {diff_raw.abs().max().item():.6e}  "
          f"rel_L2 = {(diff_raw.norm() / ref.float().norm()).item():.6e}")
    print(f"+1 decode  :  abs mean diff = {diff_p1.abs().mean().item():.6e}  "
          f"max abs diff = {diff_p1.abs().max().item():.6e}  "
          f"rel_L2 = {(diff_p1.norm() / ref.float().norm()).item():.6e}")

    # no+1 per-row max diff histogram
    row_max = diff_raw.abs().max(dim=1).values
    print()
    print(f"no+1 per-output-row max-abs-diff stats over {row_max.numel()} rows:")
    print(f"  min={row_max.min().item():.6e} mean={row_max.mean().item():.6e} "
          f"max={row_max.max().item():.6e}")
    print(f"  rows with max_diff > 1e-3: {int((row_max > 1e-3).sum().item())}")
    print(f"  rows with max_diff > 1e-2: {int((row_max > 1e-2).sum().item())}")

    # If all differences are ≤ ~1e-3 the ref is almost certainly the post-dequant
    # weight, not the true BF16 source. Print a few raw qweight values to verify
    # they really *are* packed int4 (not just 0/1 or something degenerate).
    sample = qw.flatten()[:4]
    print()
    print(f"qweight first 4 i32 (hex): {[hex(x.item() & 0xFFFFFFFF) for x in sample]}")
    print(f"qweight shape = {list(qw.shape)}  scales shape = {list(sc.shape)}")
    print(f"qzeros shape  = {list(qz.shape)}  g_idx shape  = {list(gi.shape)}")

    # Sanity: compare to a different-base reference that we KNOW is BF16 base.
    # We'll fall back to a second tensor as a cross-check. Try L0 up_proj.
    print()
    print("Cross-check: L0 up_proj no+1 decode vs ref")
    stem0 = "model.layers.0.mlp.up_proj"
    qw0 = load_tensor(qm, f"{stem0}.qweight")
    sc0 = load_tensor(qm, f"{stem0}.scales")
    qz0 = load_tensor(qm, f"{stem0}.qzeros")
    gi0 = load_tensor(qm, f"{stem0}.g_idx")
    ref0 = load_tensor(rm, f"{stem0}.weight")
    deq0 = dequant(qw0, sc0, qz0, gi0, bits=4, add_one_to_zero=False).T
    diff0 = deq0.float() - ref0.float()
    print(f"  abs mean diff = {diff0.abs().mean().item():.6e}  "
          f"rel_L2 = {(diff0.norm() / ref0.float().norm()).item():.6e}")

    # Also: what does +1 look like for a DIFFERENT canonical GPTQ quant? Use
    # qwen3.5-9b-GPTQv2-noact (we spotted it in models/). That tells us whether
    # the +1 issue is HailMary-specific or universal to our auto-gptq pipeline.
    other = Path("models/qwen3.5-9b-GPTQv2-noact")
    if (other / "model.safetensors.index.json").exists() or any(other.glob("*.safetensors")):
        print()
        print(f"Cross-check against {other.name} L0 up_proj:")
        try:
            qw2 = load_tensor(other, f"{stem0}.qweight")
            sc2 = load_tensor(other, f"{stem0}.scales")
            qz2 = load_tensor(other, f"{stem0}.qzeros")
            gi2 = load_tensor(other, f"{stem0}.g_idx")
            deq2_raw = dequant(qw2, sc2, qz2, gi2, bits=4, add_one_to_zero=False).T
            deq2_p1  = dequant(qw2, sc2, qz2, gi2, bits=4, add_one_to_zero=True).T
            diff2_raw = deq2_raw.float() - ref0.float()
            diff2_p1  = deq2_p1.float()  - ref0.float()
            print(f"  no+1: rel_L2 = {(diff2_raw.norm() / ref0.float().norm()).item():.6e}")
            print(f"  +1  : rel_L2 = {(diff2_p1.norm()  / ref0.float().norm()).item():.6e}")
        except KeyError as e:
            print(f"  could not load: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
