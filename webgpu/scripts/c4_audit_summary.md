# Path 2 — kernel utilization audit (2026-06-16)

## Measurements (RX 6700 XT, headless Chrome, MAXTOK=64-128, Deterministic)

### Q2_K 27B baseline (`local/qwen3.5-27b-gguf` Q2_K.gguf)
- e2e: **9.79 tok/s**, 102 ms/tok, 91% gpu_wait, 1154 dispatches/token
- Top GPU categories (last profiled forward, 91 ms total):

| category            | n   | total_ms | %     | avg_us |
|---------------------|-----|----------|-------|--------|
| gguf_Q2_K_r         | 336 | 59.864   | 65.7% | 178.2  |
| gguf_Q3_K_v4        |  80 | 19.032   | 20.9% | 237.9  |
| ssm_step_split      |  48 |  3.795   |  4.2% |  79.1  |
| gguf_Q6_K_tiled     |   1 |  3.188   |  3.5% | 3187.8 |
| rmsnorm             | 209 |  2.096   |  2.3% |  10.0  |

### IQ2_XXS 27B baseline (`local/qwen3.6-27b-iq2xxs-gguf`)
- e2e: **8.27 tok/s**, 121 ms/tok, 91% gpu_wait, 1154 dispatches/token
- Top GPU categories (121 ms total):

| category            | n   | total_ms | %     | avg_us |
|---------------------|-----|----------|-------|--------|
| gguf_IQ2_XXS        | 360 | 100.064  | 82.3% | 278.0  |
| gguf_Q4_K_v4        |  64 |   7.822  |  6.4% | 122.2  |
| ssm_step_split      |  48 |   3.353  |  2.8% |  69.9  |
| gguf_Q5_K_v4        |   1 |   3.197  |  2.6% | 3197.5 |
| gguf_Q2_K_r         |   8 |   2.764  |  2.3% | 345.4  |

## Per-kernel bandwidth utilization (peak ≈ 512 GB/s)

| kernel              | bytes moved | time     | GB/s | % peak |
|---------------------|-------------|----------|------|--------|
| Q2_K_r              | 5.9 GB      | 60.0 ms  | 98   | 19%    |
| Q3_K_v4             | 2.7 GB      | 19.0 ms  | 142  | 28%    |
| IQ2_XXS             | 7.5 GB      | 100.1 ms | 75   | **15%**|
| Q6_K_tiled lm_head  | 1.05 GB     | 3.2 ms   | 328  | **64%**|

The Q6_K_tiled lm_head proves the GPU CAN saturate at >60% peak with the tiled
multi-row-per-workgroup pattern (`@workgroup_size(TWG)`, shared activation tile).

## Conclusions

1. **Q2_K is at its ceiling.** `gguf_Q2_K_r` already uses the tiled + repacked
   kernel (`matmul_gguf_q2_k_tiled_r`, matmul_gguf.wgsl:1586). Memory
   `project_lever3_tile_tuning.md` confirms the chapter closed with all 6
   hypotheses falsified at 8.4-8.5 tok/s. No win available here.

2. **IQ2_XXS is the freshest, least-tuned kernel.** `matmul_gguf_iq2_xxs`
   (matmul_gguf.wgsl:497) is a vanilla single-row GEMV: WG_SIZE threads,
   one output row per workgroup, no shared activation staging, no repack.
   Q4_K, Q5_K, Q6_K, and Q2_K_r all have `_tiled` variants — IQ2_XXS does
   not.

3. **Per-byte cost gap:** IQ2_XXS at 278 µs/dispatch is 56% slower per
   dispatch than Q2_K_r at 178 µs despite IQ2_XXS weights being SMALLER
   total (8.65 GB vs 9.75 GB). The codebook indirection (`IQ2XXS_GRID` +
   `IQ2XXS_SIGNS` byte loads inside the inner loop, 64 multiplies per
   unit) is heavier than Q2_K's direct 2-bit decode.

## Recommended first target

**Build `matmul_gguf_iq2_xxs_tiled` mirroring `matmul_gguf_q2_k_tiled` /
`matmul_gguf_q6_k_tiled` (matmul_gguf.wgsl:1448, 1515):**

- TN rows per workgroup (TWG threads, TN output rows)
- Shared activation tile (`a_tile`) — each 256-elem subblock loaded ONCE
  into LDS and reused across all TN rows
- Same IQ2_XXS bit math (codebook + sign) per row in registers
- Decode-only path first (M=1); GEMM variant for spec-decode verify
  rows later if needed

Expected win, by analogy to the Q2_K_r tile gain (took it from ~18% →
28% peak per prior chapter):
- IQ2_XXS: 15% → 28% peak ≈ **~50% faster on the dominant kernel**
- e2e: 100 ms → ~55 ms on the IQ2_XXS slice
- 121 ms/tok → ~76 ms/tok = **~13 tok/s** on IQ2_XXS 27B (vs 8.27
  baseline = **+57% throughput**)

This is much larger than the MTP spec-decode ceiling we just walked away
from (Opt 1 = 0.87× baseline), and it benefits every IQ2_XXS model on
the platform, not only MTP-equipped 27B.

## Verification gates (when implemented)

1. Parity vs reference IQ2_XXS GEMV (existing `matmul_gguf_iq2_xxs`)
   on a fixed activation+weight pair, bit-exact at M=1.
2. Greedy parity vs current engine on the IQ2_XXS 27B prompt
   ("how a refrigerator keeps food cold") — byte-identical 64-tok
   response.
3. GPU bench: tok/s ≥ 10 on the same firstload-27b.mts run.

## Out of scope this audit

- Tree drafting for spec decode (deferred; Path 2 supersedes per user
  directive 2026-06-16).
- Q3_K_v4 tile work — only 20.9% of Q2_K 27B GPU time, and Q2_K is the
  short-term focus for that model.
