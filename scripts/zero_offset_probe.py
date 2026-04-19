"""zero_offset_probe.py — test whether the WGSL kernel's missing `+1` on the
unpacked zero-point is the cause of L21.up_proj engine-side blow-up.

The auto-gptq on-disk format stores zero-points as `(zero - 1)` packed into the
4-bit slots of qzeros. Conventional decoders (including auto-gptq, exllama,
and this repo's scripts/audit_weight_tensor.py) re-add the +1 after unpacking.
The WGSL kernels (matmul_q4.wgsl, matmul_q4_gemv.wgsl) do NOT add the +1.

If the missing +1 is the bug, then dequantizing the SAME weight two ways —
(a) conventional (+1) and (b) kernel-faithful (no +1) — should produce:
  * conventional:     cos ≈ 0.95  vs BF16 reference (quant noise only)
  * kernel-faithful:  cos << 0.95 for *tensors whose zeros sit near the
                      boundary* — specifically for L21.up_proj.

Why only some tensors? The +1 offset shifts every dequantized weight by
`scale` within its group. On most tensors, zero-points spread across 0..15
and the shift produces a roughly uniform bias absorbable by activations. On
tensors with zeros concentrated near a specific nibble (e.g., 14 meaning
"true zero = 15, clamp at the top of the int4 range"), the missing +1
wedges a +scale error into every weight simultaneously — which *does* show
up catastrophically in cosine and magnitude. L21.up_proj's 2-3× magnitude
inflation is exactly that fingerprint.

This script runs both decodes on a panel of tensors and prints:
  * cos_with_plus1    — PyTorch-style (expected high, ~0.95)
  * cos_without_plus1 — WGSL-style (hypothesis: collapses on L21.up_proj)
  * zero distribution — histogram of unpacked zeros (where mass sits)

Usage:
  ./venv/Scripts/python.exe scripts/zero_offset_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from audit_weight_tensor import load_tensor  # noqa: E402


PANEL = [
    # layer, dotted proj inside one layer
    (21, "mlp.up_proj"),
    (21, "mlp.gate_proj"),
    (21, "mlp.down_proj"),
    (20, "mlp.up_proj"),
    (22, "mlp.up_proj"),
    (21, "self_attn.q_proj"),
    (21, "self_attn.v_proj"),
    (0,  "mlp.up_proj"),
    (15, "mlp.up_proj"),
    (27, "mlp.up_proj"),
]


def dequant(qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor,
            g_idx: torch.Tensor, bits: int, add_one_to_zero: bool) -> torch.Tensor:
    pack = 32 // bits
    mask = (1 << bits) - 1
    in_per_packed = pack
    in_features = qweight.shape[0] * in_per_packed
    out_features = qweight.shape[1]

    qw = qweight.to(torch.int64)
    shifts = torch.arange(0, 32, bits, dtype=torch.int64, device=qw.device)
    weights_int = ((qw.unsqueeze(1) >> shifts.view(1, -1, 1)) & mask).to(torch.int32)
    weights_int = weights_int.reshape(in_features, out_features)

    qz = qzeros.to(torch.int64)
    zeros_int = ((qz.unsqueeze(2) >> shifts.view(1, 1, -1)) & mask).to(torch.int32)
    zeros_int = zeros_int.reshape(qzeros.shape[0], out_features)
    if add_one_to_zero:
        zeros_int = zeros_int + 1

    scales_f = scales.to(torch.float32)
    zeros_f = zeros_int.to(torch.float32)
    sel_scales = scales_f[g_idx.to(torch.long)]
    sel_zeros = zeros_f[g_idx.to(torch.long)]

    return (weights_int.to(torch.float32) - sel_zeros) * sel_scales


def cos_ratio(deq: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    """Returns (cos, rel_L2, abs_mean_ratio)."""
    cand = deq.T  # dequant is [in,out]; HF ref is [out,in]
    if cand.shape != ref.shape:
        raise RuntimeError(f"shape mismatch deq.T={list(cand.shape)} ref={list(ref.shape)}")
    a = cand.float().flatten()
    b = ref.float().flatten()
    cos = (a @ b).item() / ((a.norm() * b.norm()).item() + 1e-30)
    rel = ((a - b).norm() / b.norm().clamp(min=1e-30)).item()
    ratio = (cand.float().abs().mean() / ref.float().abs().mean().clamp(min=1e-30)).item()
    return cos, rel, ratio


def zero_histogram(qzeros: torch.Tensor, bits: int = 4) -> torch.Tensor:
    pack = 32 // bits
    mask = (1 << bits) - 1
    qz = qzeros.to(torch.int64)
    shifts = torch.arange(0, 32, bits, dtype=torch.int64)
    zeros_int = ((qz.unsqueeze(2) >> shifts.view(1, 1, -1)) & mask).to(torch.int32).flatten()
    hist = torch.zeros(16, dtype=torch.int64)
    for v in range(16):
        hist[v] = int((zeros_int == v).sum().item())
    return hist


def main() -> int:
    qm = Path("models/qwen3.5-9b-HailMary")
    rm = Path("models/qwen3.5-9b")

    print(f"{'tensor':<40} {'cos(+1)':>9} {'cos(no+1)':>10} {'ratio(+1)':>10} {'ratio(no+1)':>12} {'zero-mode':>10}")
    print("-" * 100)

    for li, proj in PANEL:
        stem = f"model.layers.{li}.{proj}"
        try:
            qw = load_tensor(qm, f"{stem}.qweight")
            sc = load_tensor(qm, f"{stem}.scales")
            qz = load_tensor(qm, f"{stem}.qzeros")
            gi = load_tensor(qm, f"{stem}.g_idx")
            ref = load_tensor(rm, f"{stem}.weight")
        except KeyError as e:
            print(f"{stem:<40}  MISSING ({e})")
            continue

        deq_plus1 = dequant(qw, sc, qz, gi, bits=4, add_one_to_zero=True)
        deq_raw = dequant(qw, sc, qz, gi, bits=4, add_one_to_zero=False)

        cos_p, _, ratio_p = cos_ratio(deq_plus1, ref)
        cos_r, _, ratio_r = cos_ratio(deq_raw, ref)

        hist = zero_histogram(qz)
        mode_val = int(hist.argmax().item())
        mode_frac = hist[mode_val].item() / max(int(hist.sum().item()), 1)

        label = f"L{li}.{proj}"
        print(f"{label:<40} {cos_p:>9.4f} {cos_r:>10.4f} {ratio_p:>10.4f} {ratio_r:>12.4f}  "
              f"{mode_val:>2}({mode_frac*100:.0f}%)")

    print()
    print("Interpretation:")
    print("  cos(+1)      = PyTorch/auto-gptq conventional decode (should be ~0.92-0.95)")
    print("  cos(no+1)    = WGSL kernel-faithful decode (hypothesis: collapses for L21.up_proj)")
    print("  ratio(...)   = deq_absMean / ref_absMean (engine showed ~2x for L21.up_proj)")
    print("  zero-mode    = most common unpacked zero value and its fraction")
    print()
    print("If cos(no+1) for L21.up_proj is << 0.9 and ratio(no+1) is ~2x while gate_proj")
    print("stays high, the missing +1 in WGSL is the root cause.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
