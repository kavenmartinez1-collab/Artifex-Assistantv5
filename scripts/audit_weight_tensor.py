"""audit_weight_tensor.py — tear apart a single GPTQ-quantized projection.

Compare a HailMary GPTQ INT4 weight (qweight/scales/qzeros/g_idx) against the
same projection in a reference (BF16/FP16) model. Dumps stats on every quant
tensor, dequantizes to a full FP32 matrix, and reports per-row / per-column
divergence so we can see whether a specific row, group, or scale band is
broken.

Generic by design — pass any tensor stem (e.g. `model.layers.21.mlp.up_proj`)
and any pair of model directories. Works for any GPTQ-format model so the
same script will check Gemma4 quants later.

Usage:
  ./venv/Scripts/python.exe scripts/audit_weight_tensor.py \\
      --quant-model models/qwen3.5-9b-HailMary \\
      --ref-model   models/qwen3.5-9b \\
      --tensor      model.layers.21.mlp.up_proj
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def _candidate_keys(key: str) -> list[str]:
    """Generate name candidates to handle wrapper differences across packagings.

    Some models nest the text decoder under 'language_model.' (multimodal
    wrappers); some don't. Try both directions.
    """
    cands = [key]
    if ".language_model." not in key and key.startswith("model."):
        cands.append(key.replace("model.", "model.language_model.", 1))
    if ".language_model." in key:
        cands.append(key.replace("model.language_model.", "model.", 1))
    return cands


def find_shard(model_dir: Path, key: str) -> tuple[Path, str]:
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        for cand in _candidate_keys(key):
            if cand in idx["weight_map"]:
                return model_dir / idx["weight_map"][cand], cand
    for sf in model_dir.glob("*.safetensors"):
        with safe_open(sf, framework="pt") as f:
            ks = set(f.keys())
            for cand in _candidate_keys(key):
                if cand in ks:
                    return sf, cand
    raise KeyError(f"{key} (and aliases) not found in {model_dir}")


def load_tensor(model_dir: Path, key: str) -> torch.Tensor:
    sh, real_key = find_shard(model_dir, key)
    with safe_open(sh, framework="pt") as f:
        return f.get_tensor(real_key)


def fmt_stats(t: torch.Tensor) -> str:
    f = t.detach().float().flatten()
    nan = torch.isnan(f).sum().item()
    inf = (torch.isinf(f) & ~torch.isnan(f)).sum().item()
    valid = f[~torch.isnan(f) & ~torch.isinf(f)]
    if valid.numel() == 0:
        return f"shape={list(t.shape)} dtype={t.dtype} ALL NAN/INF"
    return (
        f"shape={list(t.shape)} dtype={t.dtype} "
        f"min={valid.min():.6f} max={valid.max():.6f} "
        f"mean={valid.mean():.6f} absMean={valid.abs().mean():.6f} "
        f"std={valid.std():.6f} NaN={nan} Inf={inf}"
    )


def dequantize_gptq(qweight: torch.Tensor, scales: torch.Tensor,
                    qzeros: torch.Tensor, g_idx: torch.Tensor,
                    bits: int = 4) -> torch.Tensor:
    """Reconstruct a [in_features, out_features] FP32 weight from GPTQ tensors.

    Standard auto-gptq layout:
      qweight : [in_features // (32 // bits), out_features] int32 packed cols
      qzeros  : [num_groups, out_features // (32 // bits)] int32 packed cols
      scales  : [num_groups, out_features] fp16
      g_idx   : [in_features] int32 (group index per input row)
    Output: [in_features, out_features] fp32 (transpose later if needed).
    """
    pack = 32 // bits
    mask = (1 << bits) - 1
    in_per_packed = pack
    in_features = qweight.shape[0] * in_per_packed
    out_features = qweight.shape[1]

    # Unpack qweight into [in_features, out_features] int values 0..2^bits-1.
    qw = qweight.to(torch.int64)  # avoid sign issues
    shifts = torch.arange(0, 32, bits, dtype=torch.int64, device=qw.device)
    # [in_features // pack, pack, out_features]
    weights_int = ((qw.unsqueeze(1) >> shifts.view(1, -1, 1)) & mask).to(torch.int32)
    weights_int = weights_int.reshape(in_features, out_features)

    # Unpack qzeros into [num_groups, out_features] int values; auto-gptq stores
    # zero - 1, so the conventional decode adds 1 after unpacking.
    qz = qzeros.to(torch.int64)
    zeros_int = ((qz.unsqueeze(2) >> shifts.view(1, 1, -1)) & mask).to(torch.int32)
    zeros_int = zeros_int.reshape(qzeros.shape[0], out_features) + 1

    # Per-row scale + zero via g_idx
    scales_f = scales.to(torch.float32)
    zeros_f = zeros_int.to(torch.float32)
    sel_scales = scales_f[g_idx.to(torch.long)]      # [in_features, out_features]
    sel_zeros = zeros_f[g_idx.to(torch.long)]        # [in_features, out_features]

    return (weights_int.to(torch.float32) - sel_zeros) * sel_scales


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quant-model", required=True)
    ap.add_argument("--ref-model", required=True)
    ap.add_argument("--tensor", required=True,
                    help="weight stem, e.g. 'model.layers.21.mlp.up_proj'")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--neighbors", default="",
                    help="comma-separated list of neighbor stems for sanity "
                         "compare (e.g. 'model.layers.20.mlp.up_proj,model.layers.22.mlp.up_proj')")
    args = ap.parse_args()

    qm = Path(args.quant_model)
    rm = Path(args.ref_model)

    # --- 1. Load all four GPTQ tensors and print raw stats ---
    print(f"=== {args.tensor} (HailMary GPTQ INT{args.bits}) ===")
    qw = load_tensor(qm, f"{args.tensor}.qweight")
    scales = load_tensor(qm, f"{args.tensor}.scales")
    qzeros = load_tensor(qm, f"{args.tensor}.qzeros")
    g_idx = load_tensor(qm, f"{args.tensor}.g_idx")
    print(f"  qweight: {fmt_stats(qw)}")
    print(f"  scales : {fmt_stats(scales)}")
    print(f"  qzeros : {fmt_stats(qzeros)}")
    print(f"  g_idx  : {fmt_stats(g_idx)}")
    print(f"  g_idx unique = {torch.unique(g_idx).numel()} (expect num_groups = {scales.shape[0]})")

    # --- 2. Dequantize and stats ---
    deq = dequantize_gptq(qw, scales, qzeros, g_idx, bits=args.bits)
    print(f"  dequant: {fmt_stats(deq)}")

    # --- 3. Load reference and compare ---
    print(f"\n=== {args.tensor}.weight (reference, full precision) ===")
    ref = load_tensor(rm, f"{args.tensor}.weight")
    print(f"  ref: {fmt_stats(ref)}")

    # auto-gptq stores qweight as [in/pack, out] (each int32 holds `pack` packed
    # values along the IN dim). Our dequant_gptq produces [in, out]. HF stores
    # the Linear weight as [out, in]. So the correct alignment is ALWAYS .T,
    # not whichever happens to match shape (which falsely picks the untransposed
    # orientation for square matrices and silently scores ~0 cosine).
    cand = deq.T
    if cand.shape != ref.shape:
        print(f"  WARNING: dequant.T shape {list(cand.shape)} != ref shape {list(ref.shape)}")
        return 1
    a = cand.float().flatten()
    b = ref.float().flatten()
    cos = (a @ b).item() / ((a.norm() * b.norm()).item() + 1e-30)
    rel_l2 = ((a - b).norm() / b.norm().clamp(min=1e-30)).item()
    print(f"  aligned shape {list(cand.shape)}: cos={cos:.6f} rel_L2={rel_l2:.6f}")
    best = (cos, rel_l2, cand)

    cos, rel_l2, dequant_aligned = best
    print(f"\n[overall] cosine={cos:.6f}  rel_L2={rel_l2:.6f}")

    # --- 4. Per-row diagnosis on aligned dequant ---
    deq_f = dequant_aligned.float()
    ref_f = ref.float()
    err = deq_f - ref_f
    # rows = output features (HF weight is [out, in])
    row_err = err.norm(dim=1)
    row_ref = ref_f.norm(dim=1).clamp(min=1e-30)
    row_rel = row_err / row_ref
    rk = min(20, row_rel.numel())
    bad_rows = torch.topk(row_rel, rk).indices.tolist()
    print(f"\n[rows] worst {rk} output rows by rel_L2:")
    for i, idx in enumerate(bad_rows):
        print(f"  row {idx:5d}  rel_L2={row_rel[idx]:.4f}  ref_norm={row_ref[idx]:.4f}  err_norm={row_err[idx]:.4f}")

    # --- 5. Per-input-column / per-group diagnosis ---
    col_err = err.norm(dim=0)
    col_ref = ref_f.norm(dim=0).clamp(min=1e-30)
    col_rel = col_err / col_ref
    bad_cols = torch.topk(col_rel, min(20, col_rel.numel())).indices.tolist()
    print(f"\n[cols] worst 20 input cols by rel_L2:")
    for idx in bad_cols:
        gi = int(g_idx[idx].item()) if idx < g_idx.numel() else -1
        print(f"  col {idx:5d}  rel_L2={col_rel[idx]:.4f}  group={gi}")

    # --- 6. Group-level scale stats ---
    sf = scales.float()
    print(f"\n[scales] per-group abs-stats over {sf.shape[0]} groups, {sf.shape[1]} out_features")
    grp_abs_max = sf.abs().max(dim=1).values
    print(f"  per-group max-abs:   min={grp_abs_max.min():.6f} max={grp_abs_max.max():.6f} "
          f"mean={grp_abs_max.mean():.6f} std={grp_abs_max.std():.6f}")
    # Flag groups with anomalous scale magnitude
    z = (grp_abs_max - grp_abs_max.mean()) / grp_abs_max.std().clamp(min=1e-9)
    big = (z.abs() > 5).nonzero().flatten().tolist()
    print(f"  groups with |z| > 5 in max-scale: {len(big)} ({big[:20]}{'...' if len(big) > 20 else ''})")

    # --- 7. Neighbor sanity (overall cosine) ---
    if args.neighbors:
        print(f"\n[neighbors] dequant-vs-ref cosine for sanity neighbors:")
        for stem in [s.strip() for s in args.neighbors.split(",") if s.strip()]:
            try:
                qw2 = load_tensor(qm, f"{stem}.qweight")
                sc2 = load_tensor(qm, f"{stem}.scales")
                qz2 = load_tensor(qm, f"{stem}.qzeros")
                gi2 = load_tensor(qm, f"{stem}.g_idx")
                deq2 = dequantize_gptq(qw2, sc2, qz2, gi2, bits=args.bits)
                ref2 = load_tensor(rm, f"{stem}.weight")
                # match orientation
                cand = deq2 if deq2.shape == ref2.shape else deq2.T
                if cand.shape != ref2.shape:
                    print(f"  {stem}: shape mismatch deq={list(deq2.shape)} ref={list(ref2.shape)}")
                    continue
                a = cand.float().flatten()
                b = ref2.float().flatten()
                cn = (a @ b).item() / ((a.norm() * b.norm()).item() + 1e-30)
                rl = ((a - b).norm() / b.norm()).item()
                print(f"  {stem}: cos={cn:.6f} rel_L2={rl:.6f}")
            except Exception as e:
                print(f"  {stem}: FAILED ({e})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
