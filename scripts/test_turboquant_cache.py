"""TurboQuantCache audit repro: multi-event compression token ordering.

Pushes tokens through TurboQuantCache (wrapping DynamicCache) in several
update() calls so compression fires more than once, then checks that the
reconstructed K/V tokens come back in the right order per head.

If the cross-step flat-concat in _compress_overflow scrambles (head, token)
order, per-token reconstruction error explodes after the second compression
event even though single-event reconstruction is accurate.

Run: ./venv/Scripts/python.exe scripts/test_turboquant_cache.py
"""
import sys
import torch

sys.path.insert(0, ".")
from core.turboquant_cache import TurboQuantCache
from transformers.cache_utils import DynamicCache

torch.manual_seed(0)

B, H, D = 1, 4, 64          # batch, kv heads, head_dim (power of 2)
RESID = 8                    # small residual window so compression fires fast
LAYER = 2                    # non-boundary layer (boundary_layers=2, num_layers=8)

inner = DynamicCache()
cache = TurboQuantCache(inner, key_bits=3, value_bits=2,
                        residual_length=RESID, boundary_layers=2, num_layers=8)

# Distinct per-token vectors so ordering mistakes are visible.
def make_kv(t0, n):
    k = torch.zeros(B, H, n, D)
    v = torch.zeros(B, H, n, D)
    for i in range(n):
        for h in range(H):
            torch.manual_seed(1000 * (t0 + i) + h)
            k[0, h, i] = torch.randn(D)
            v[0, h, i] = torch.randn(D)
    return k, v

all_k, all_v = [], []

def step(t0, n):
    k, v = make_kv(t0, n)
    all_k.append(k)
    all_v.append(v)
    return cache.update(k, v, LAYER)

# Prefill 16 tokens (compression event 1: 8 compressed), then 6 decode steps
# (events 2..7: 1 token each).
fk, fv = step(0, 16)
for t in range(16, 22):
    fk, fv = step(t, 1)

ref_k = torch.cat(all_k, dim=2)
ref_v = torch.cat(all_v, dim=2)
assert fk.shape == ref_k.shape, f"shape {fk.shape} vs {ref_k.shape}"

T = ref_k.shape[2]
n_compressed = T - RESID
print(f"tokens={T} compressed={n_compressed} residual={RESID} heads={H}")

# Per-token cosine similarity of reconstruction vs original (averaged over heads)
def cos_by_token(rec, ref):
    r = torch.nn.functional.cosine_similarity(rec.float(), ref.float(), dim=-1)  # (B,H,T)
    return r.mean(dim=(0, 1))

ck = cos_by_token(fk, ref_k)
cv = cos_by_token(fv, ref_v)
print("\nper-token mean cosine sim (K | V):")
for t in range(T):
    tag = "compressed" if t < n_compressed else "residual  "
    flag = "  <-- BAD" if ck[t] < 0.9 or cv[t] < 0.9 else ""
    print(f"  t={t:2d} {tag}  K={ck[t]:+.4f}  V={cv[t]:+.4f}{flag}")

bad = [(t, float(ck[t]), float(cv[t])) for t in range(T) if ck[t] < 0.9 or cv[t] < 0.9]
if bad:
    print(f"\nFAIL: {len(bad)}/{T} tokens reconstruct wrong (ordering scrambled across compression events)")
    sys.exit(1)
print("\nPASS: all tokens reconstruct in order")

# ── Sliding-window layers (Gemma 3/4 style) must be left alone ──────────────
from transformers.models.gemma3 import Gemma3TextConfig

cfg = Gemma3TextConfig(num_hidden_layers=8, sliding_window=32,
                       layer_types=["sliding_attention"] * 5
                                   + ["full_attention", "sliding_attention", "full_attention"],
                       num_attention_heads=4, num_key_value_heads=2,
                       hidden_size=64, head_dim=16)
inner2 = DynamicCache(config=cfg)
cache2 = TurboQuantCache(inner2, residual_length=RESID, boundary_layers=2, num_layers=8)

k = torch.randn(1, 2, 64, 16)
sk, _ = cache2.update(k, k.clone(), 4)            # sliding, non-boundary
assert sk.shape[2] == 64, sk.shape
sk2, _ = cache2.update(torch.randn(1, 2, 1, 16), torch.randn(1, 2, 1, 16), 4)
assert sk2.shape[2] == 32, f"sliding window not honored: {sk2.shape}"  # window cap
assert 4 not in cache2._compressed, "sliding layer was compressed"
print("PASS: sliding layers skipped (window cap intact, no compressed blob)")

fk2, _ = cache2.update(k, k.clone(), 7)           # full_attention BUT boundary (last 2)
assert 7 not in cache2._compressed and fk2.shape[2] == 64
fk3, _ = cache2.update(k, k.clone(), 5)           # full_attention, non-boundary → compresses
assert 5 in cache2._compressed and fk3.shape[2] == 64
print("PASS: full-attention layer compressed, boundary layer untouched")

# ── Non-power-of-2 head_dim: graceful skip, not crash ───────────────────────
inner3 = DynamicCache()
cache3 = TurboQuantCache(inner3, residual_length=RESID, boundary_layers=2, num_layers=8)
k96 = torch.randn(1, 2, 64, 96)
out, _ = cache3.update(k96, k96.clone(), 3)
assert out.shape[2] == 64 and 3 not in cache3._compressed
print("PASS: head_dim=96 skipped gracefully")
