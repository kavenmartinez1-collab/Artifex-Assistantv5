"""gidx_trivial_probe.py — check whether each layer's gate/up/down g_idx is
trivial (floor(k/group_size)) or a real actorder permutation.

Replicates the engine's isTrivialGIdxBuffer heuristic (weight-loader.ts:82):
  - view[0] must be 0
  - find first k where view[k] != 0, call that gs
  - verify view[k] == floor(k / gs) for all k

If L21.up_proj's g_idx is uniquely trivial (or uniquely non-trivial in a way
that fools the heuristic) compared to its siblings, that explains why engine
applies the wrong scale/zero per column — producing 2x magnitude inflation
and a systematic negative mean shift on L21.up_proj output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from audit_weight_tensor import load_tensor  # noqa: E402


def is_trivial(g_idx: torch.Tensor) -> tuple[bool, int, str]:
    """Mirror webgpu/src/model/weight-loader.ts:isTrivialGIdxBuffer.

    Returns (is_trivial, detected_gs, reason).
    """
    arr = g_idx.to(torch.int32).tolist()
    n = len(arr)
    if n < 2:
        return False, -1, "length<2"
    if arr[0] != 0:
        return False, -1, f"g_idx[0]={arr[0]} (not 0)"
    gs = -1
    for k in range(1, n):
        if arr[k] != 0:
            if arr[k] != 1:
                return False, -1, f"g_idx[{k}]={arr[k]} (jumped past 1)"
            gs = k
            break
    if gs <= 0:
        return False, -1, "all zeros"
    if n % gs != 0:
        return False, gs, f"length {n} not divisible by gs={gs}"
    for k in range(n):
        expected = k // gs
        if arr[k] != expected:
            return False, gs, f"g_idx[{k}]={arr[k]} expected {expected}"
    return True, gs, "trivial"


def main() -> int:
    qm = Path("models/qwen3.5-9b-HailMary")
    layers_to_check = [0, 1, 5, 15, 20, 21, 22, 27, 28, 29, 30, 31]
    projs = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

    print(f"{'tensor':<40} {'trivial?':>10} {'gs':>5} {'reason':<40}")
    print("-" * 100)
    for li in layers_to_check:
        for proj in projs:
            stem = f"model.layers.{li}.{proj}"
            try:
                gi = load_tensor(qm, f"{stem}.g_idx")
            except KeyError:
                continue
            trivial, gs, reason = is_trivial(gi)
            flag = "TRIVIAL" if trivial else "actorder"
            print(f"L{li}.{proj:<25} {flag:>10} {gs:>5} {reason:<40}")
        print()

    # Also check SSM & attention projections for L21 since L21 is SSM layer
    print("L21 SSM + attention projections:")
    for proj in [
        "linear_attn.in_proj_qkv", "linear_attn.in_proj_a", "linear_attn.in_proj_b",
        "linear_attn.in_proj_z", "linear_attn.out_proj",
    ]:
        stem = f"model.layers.21.{proj}"
        try:
            gi = load_tensor(qm, f"{stem}.g_idx")
        except KeyError:
            continue
        trivial, gs, reason = is_trivial(gi)
        flag = "TRIVIAL" if trivial else "actorder"
        print(f"  L21.{proj:<30} {flag:>10} {gs:>5} {reason}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
