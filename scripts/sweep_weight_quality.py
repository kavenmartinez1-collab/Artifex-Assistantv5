"""sweep_weight_quality.py — disk-side quant quality across ALL projections.

For every layer, compute dequant-vs-reference cosine and rel_L2 for each
linear projection (gate/up/down for FFN, q/k/v/o for FA, lin-* for SSM).
Sorts the output worst-first so anomalies pop. Generic — pass any GPTQ
quant + any FP reference and it walks the layer count from config.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
from audit_weight_tensor import dequantize_gptq, load_tensor, find_shard  # noqa: E402


# Engine-projection short-name -> HF dotted submodule path inside one layer.
# Mirrors compare_label_map.qwen35.json.
QWEN35_PROJS = {
    "lin-qkv": "linear_attn.in_proj_qkv",
    "lin-a":   "linear_attn.in_proj_a",
    "lin-b":   "linear_attn.in_proj_b",
    "lin-z":   "linear_attn.in_proj_z",
    "lin-out": "linear_attn.out_proj",
    "q":       "self_attn.q_proj",
    "k":       "self_attn.k_proj",
    "v":       "self_attn.v_proj",
    "o":       "self_attn.o_proj",
    "gate":    "mlp.gate_proj",
    "up":      "mlp.up_proj",
    "down":    "mlp.down_proj",
}


def cos_l2(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    af = a.float().flatten()
    bf = b.float().flatten()
    cos = (af @ bf).item() / ((af.norm() * bf.norm()).item() + 1e-30)
    rel = ((af - bf).norm() / bf.norm().clamp(min=1e-30)).item()
    return cos, rel


def try_dequant(qm: Path, stem: str, bits: int) -> torch.Tensor | None:
    try:
        qw = load_tensor(qm, f"{stem}.qweight")
        sc = load_tensor(qm, f"{stem}.scales")
        qz = load_tensor(qm, f"{stem}.qzeros")
        gi = load_tensor(qm, f"{stem}.g_idx")
        return dequantize_gptq(qw, sc, qz, gi, bits=bits)
    except KeyError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quant-model", required=True)
    ap.add_argument("--ref-model", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=0,
                    help="layer count; 0 = read from quant config.json")
    ap.add_argument("--out", default="weight_sweep.json",
                    help="JSON output for full table")
    args = ap.parse_args()

    qm = Path(args.quant_model)
    rm = Path(args.ref_model)

    if args.num_layers > 0:
        nL = args.num_layers
    else:
        cfg = json.loads((qm / "config.json").read_text())
        text = cfg.get("text_config", cfg)
        nL = int(text.get("num_hidden_layers"))
        print(f"[sweep] num_hidden_layers from config: {nL}")

    rows: list[dict] = []
    for li in range(nL):
        for short, dotted in QWEN35_PROJS.items():
            stem = f"model.layers.{li}.{dotted}"
            deq = try_dequant(qm, stem, args.bits)
            if deq is None:
                continue
            try:
                ref = load_tensor(rm, f"{stem}.weight")
            except KeyError:
                continue
            cand = deq.T  # always transpose: dequant produces [in,out]; ref is [out,in]
            if cand.shape != ref.shape:
                continue
            cos, rel = cos_l2(cand, ref)
            rows.append({
                "layer": li, "proj": short, "stem": stem,
                "cos": cos, "rel_L2": rel,
                "ref_absMean": float(ref.float().abs().mean()),
                "deq_absMean": float(cand.float().abs().mean()),
            })
        print(f"  layer {li}: {sum(1 for r in rows if r['layer'] == li)} projections audited")

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\n[sweep] wrote {args.out}  ({len(rows)} rows)")

    # Print the 25 worst by cosine (most divergent on disk).
    print()
    print(f"{'rank':>4}  {'layer':>5}  {'proj':<8}  {'cos':>9}  {'rel_L2':>9}  "
          f"{'deq_abs':>9}  {'ref_abs':>9}")
    print("-" * 80)
    by_cos = sorted(rows, key=lambda r: r["cos"])
    for i, r in enumerate(by_cos[:25]):
        print(f"{i + 1:>4}  L{r['layer']:>3}  {r['proj']:<8}  {r['cos']:>9.4f}  "
              f"{r['rel_L2']:>9.4f}  {r['deq_absMean']:>9.4f}  {r['ref_absMean']:>9.4f}")

    # And the magnitude-anomaly view: deq_absMean / ref_absMean far from 1.
    print()
    print("By |deq_abs / ref_abs - 1| (magnitude anomalies):")
    print(f"{'rank':>4}  {'layer':>5}  {'proj':<8}  {'ratio':>9}  {'cos':>9}")
    print("-" * 60)
    for i, r in enumerate(sorted(rows,
                                  key=lambda r: abs(r["deq_absMean"] / max(r["ref_absMean"], 1e-9) - 1.0),
                                  reverse=True)[:25]):
        ratio = r["deq_absMean"] / max(r["ref_absMean"], 1e-9)
        print(f"{i + 1:>4}  L{r['layer']:>3}  {r['proj']:<8}  {ratio:>9.4f}  {r['cos']:>9.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
