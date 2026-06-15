// GGUF k-quant dequantizing GEMV/GEMM — native llama.cpp block formats.
//
// C[M, N] = A[M, K] @ dequant(W)^T
//
// W holds GGUF blocks in llama.cpp row-major weight layout: tensor row n
// (output column n) is K contiguous elements = K/blockSize contiguous
// blocks. Types and per-block u32 strides (after load-time repack, see
// gguf-dequant.ts repack*):
//   Q8_0: 9 u32  (repacked 34→36 B: f16 d @ bytes 0-1, pad, qs[32] @ 4-35)
//   Q4_K: 36 u32 (raw 144 B: f16 d,dmin | scales[12] | qs[128])
//   Q5_K: 44 u32 (raw 176 B: f16 d,dmin | scales[12] | qh[32] | qs[128])
//   Q6_K: 53 u32 (repacked 210→212 B: ql[128] | qh[64] | i8 scales[16] | f16 d)
//   IQ4_NL: 5 u32 (repacked 18→20 B like Q4_0; non-linear codebook)
//   IQ4_XS: 34 u32 (raw 136 B: f16 d | u16 scales_h | scales_l[4] | qs[128])
//
// Bit decoding mirrors gguf-dequant.ts (the CPU reference validated bit-exact
// against the official Python gguf package) — kernel-audit rule.
//
// Execution shape: ONE WORKGROUP per output element (n, m). The WG's 64
// threads stride the row's 32-element units (adjacent threads read adjacent
// blocks → coalesced), then tree-reduce in workgroup memory. Works for both
// decode (M=1) and prefill (M>1, grid.y = M); a tiled GEMM for prefill can
// come later (Phase C4).
//
// Grid: x = min(N, 65535), y = M, z = ceil(N / 65535); n = wid.x + wid.z*65535.

const WG_SIZE: u32 = 64u;

// ── Tiled GEMV (lever 2) ────────────────────────────────────────────────
// TN output rows per workgroup instead of one. The activation slice for a
// chunk is staged once into workgroup memory as vec4s and shared by all TN
// rows, cutting global A traffic by TN× and amortizing weight reads across
// wider workgroups. Decode bit-math is IDENTICAL to the legacy kernels; only
// accumulation order changes (plain f32 + dot(), no inner Kahan), which the
// teacher-forced parity gate covers.
// TPR = TWG/TN lanes per row; each lane owns one 32-elem unit per chunk.
// Grid: x = min(ceil(N/TN), 65535), y = M, z = ceil(ceil(N/TN) / 65535).
// Both constants are pipeline-overridable for per-card tuning (no subgroup
// ops — keeps RDNA2/AMD portability). TPR must be a power of two.

override TN: u32 = 8u;     // output rows per workgroup
override TWG: u32 = 256u;  // threads per workgroup
// Max M (activation rows) the GEMM kernel batches. A plain const, NOT an
// override: WGSL only permits override-sized arrays in <workgroup> space, and
// the GEMM kernel needs function-local acc/partial arrays of this size. 8
// matches the spec-decode cap (MAX_SPEC_ROWS / MAX_ARGMAX_ROWS).
const MAXM: u32 = 8u;

struct Params {
  M: u32,
  N: u32,
  K: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> wg_partial: array<f32, WG_SIZE>;

// Tiled-kernel workgroup storage (override-sized; only allocated for the
// *_tiled entry points). a_tile holds one chunk of activations: TPR units
// × 32 elems = TPR*8 vec4s (4 KB at defaults). row_acc holds per-lane
// partial sums for the per-row tree reduce.
var<workgroup> a_tile: array<vec4<f32>, (TWG / TN) * 8u>;
var<workgroup> row_acc: array<f32, TWG>;

// GEMM (Phase C4) workgroup storage: one chunk of activations for ALL M rows,
// laid out [m][vec], so a decoded weight unit is reused across M outputs.
// MAXM * (TWG/TN) * 8 vec4s (16 KB at MAXM=8, TWG=128, TN=8). Only materialized
// for the *_gemm entry point (which never references a_tile).
var<workgroup> a_tile_m: array<vec4<f32>, MAXM * (TWG / TN) * 8u>;

// ── No-stage decode GEMV (lever 4) ──────────────────────────────────────
// A4 aliases binding 0 with a vec4 view. WGSL allows duplicate
// (group,binding) declarations as long as no single entry point statically
// references both — the *_tiled_ns entries use A4 only (A is only reached
// via tile_stage, which they don't call), and a_tile is not allocated for
// them (workgroup vars are materialized per entry point by static use).
@group(0) @binding(0) var<storage, read> A4: array<vec4<f32>>;

// ── vec4 weight loads (lever 4 phase 2) ─────────────────────────────────
// W4 aliases binding 1 with a vec4 view for the *_tiled_v4 entries (legal
// per-block u32 strides ≡ 0 mod 4: Q3_K 28, Q4_K 36, Q5_K 44 — every qs/
// qh/header word offset lands on a vec4 boundary; asserted by gate 4 in
// scripts/test-gemv-tiled.mts). v4 entries must not reference W, so they
// use scale_min_k4_w below instead of wbyte()-based scale_min_k4.
@group(0) @binding(1) var<storage, read> W4: array<vec4<u32>>;

/** scale_min_k4 with the three packed-scale words passed as values
 *  (w1..w3 = bytes 4..15 of a k-quant superblock header vec4). Byte math
 *  identical to scale_min_k4. */
fn scale_min_k4_w(j: u32, w1: u32, w2: u32, w3: u32) -> vec2<f32> {
  if (j < 4u) {
    return vec2<f32>(
      f32((w1 >> (j * 8u)) & 63u),
      f32((w2 >> (j * 8u)) & 63u),
    );
  }
  let b1 = (w1 >> ((j - 4u) * 8u)) & 0xFFu;
  let b2 = (w2 >> ((j - 4u) * 8u)) & 0xFFu;
  let b3 = (w3 >> ((j - 4u) * 8u)) & 0xFFu;
  return vec2<f32>(
    f32((b3 & 0x0Fu) | ((b1 >> 6u) << 4u)),
    f32((b3 >> 4u) | ((b2 >> 6u) << 4u)),
  );
}

/** Byte `off` of the W buffer (off in bytes; W is u32-backed, little-endian). */
fn wbyte(off: u32) -> u32 {
  return (W[off >> 2u] >> ((off & 3u) * 8u)) & 0xFFu;
}

/** (scale, min) pair j (0..7) from the 12-byte packed scales at byte `sbyte`.
 *  Mirrors get_scale_min_k4 (gguf-dequant.ts / llama.cpp ggml-quants.c). */
fn scale_min_k4(j: u32, sbyte: u32) -> vec2<f32> {
  if (j < 4u) {
    return vec2<f32>(
      f32(wbyte(sbyte + j) & 63u),
      f32(wbyte(sbyte + j + 4u) & 63u),
    );
  }
  let sc = (wbyte(sbyte + j + 4u) & 0x0Fu) | ((wbyte(sbyte + j - 4u) >> 6u) << 4u);
  let mn = (wbyte(sbyte + j + 4u) >> 4u) | ((wbyte(sbyte + j) >> 6u) << 4u);
  return vec2<f32>(f32(sc), f32(mn));
}

/** Sign-extended i8 from byte j (0..3) of a u32 word. */
fn i8_byte(word: u32, j: u32) -> f32 {
  return f32(bitcast<i32>(word << ((3u - j) * 8u)) >> 24u);
}

/** IQ4 non-linear codebook value (kvalues_iq4nl, ggml-quants.c), q in 0..15.
 *  Values: -127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113 —
 *  packed here as i8 bytes in 4 u32 words, extracted via i8_byte. */
fn iq4nl_val(q: u32) -> f32 {
  var kv = array<u32, 4>(0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u);
  return i8_byte(kv[q >> 2u], q & 3u);
}

/** Tree-reduce wg_partial and write C[m*N + n] from thread 0. */
fn reduce_and_store(tid: u32, n: u32, m: u32, sum: f32) {
  wg_partial[tid] = sum;
  workgroupBarrier();
  var stride = WG_SIZE / 2u;
  while (stride > 0u) {
    if (tid < stride) {
      wg_partial[tid] = wg_partial[tid] + wg_partial[tid + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (tid == 0u && n < params.N) {
    C[m * params.N + n] = wg_partial[0];
  }
}

/** Cooperatively stage TPR*32 activation elements [elemBase, elemBase+TPR*32)
 *  of row `aRow` into a_tile as vec4s. K is always a multiple of 32 and
 *  elemBase a multiple of 4, so each vec4 is either fully in range or fully
 *  out (zero-filled). Caller wraps with workgroupBarrier() on both sides. */
fn tile_stage(tid: u32, aRow: u32, elemBase: u32, K: u32) {
  let nVec = (TWG / TN) * 8u;
  for (var i = tid; i < nVec; i = i + TWG) {
    let e = elemBase + i * 4u;
    var v = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (e < K) {
      v = vec4<f32>(A[aRow + e], A[aRow + e + 1u], A[aRow + e + 2u], A[aRow + e + 3u]);
    }
    a_tile[i] = v;
  }
}

/** GEMM variant of tile_stage: stage the same TPR*32-element chunk for ALL M
 *  activation rows into a_tile_m, laid out [m * nVec + i]. Caller wraps with
 *  workgroupBarrier() on both sides. */
fn tile_stage_m(tid: u32, M: u32, elemBase: u32, K: u32) {
  let nVec = (TWG / TN) * 8u;
  for (var i = tid; i < nVec; i = i + TWG) {
    let e = elemBase + i * 4u;
    for (var mi = 0u; mi < M; mi = mi + 1u) {
      var v = vec4<f32>(0.0, 0.0, 0.0, 0.0);
      if (e < K) {
        let aRow = mi * K;
        v = vec4<f32>(A[aRow + e], A[aRow + e + 1u], A[aRow + e + 2u], A[aRow + e + 3u]);
      }
      a_tile_m[mi * nVec + i] = v;
    }
  }
}

/** Tree-reduce each row's TPR-lane segment of row_acc; lane 0 of each row
 *  writes C[m*N + n]. */
fn tile_reduce_store(tid: u32, lane: u32, n: u32, m: u32, acc: f32) {
  row_acc[tid] = acc;
  workgroupBarrier();
  var stride = (TWG / TN) / 2u;
  while (stride > 0u) {
    if (lane < stride) {
      row_acc[tid] = row_acc[tid] + row_acc[tid + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (lane == 0u && n < params.N) {
    C[m * params.N + n] = row_acc[tid];
  }
}

// ── Q4_0: 32-elem blocks, 5 u32 each (f16 d | pad | qs[16]) ────────────
// x[j] = d·((qs[j]&0xF)-8), x[j+16] = d·((qs[j]>>4)-8), j in 0..15.

@compute @workgroup_size(WG_SIZE, 1, 1)
fn matmul_gguf_q4_0(@builtin(local_invocation_id) lid: vec3u,
                    @builtin(workgroup_id) wid: vec3u) {
  let n = wid.x + wid.z * 65535u;
  let m = wid.y;
  let tid = lid.x;
  let K = params.K;
  let nBlocks = K / 32u;

  var sum: f32 = 0.0;
  var comp: f32 = 0.0;
  for (var blk = tid; blk < nBlocks; blk = blk + WG_SIZE) {
    let base = (n * nBlocks + blk) * 5u;          // u32 words
    let d = unpack2x16float(W[base]).x;
    let qsByte = (base + 1u) * 4u;                // qs[16] start, in bytes
    let aBase = m * K + blk * 32u;
    var dot: f32 = 0.0;
    for (var j = 0u; j < 16u; j = j + 1u) {
      let q = wbyte(qsByte + j);
      let lo = f32(i32(q & 0x0Fu) - 8);
      let hi = f32(i32(q >> 4u) - 8);
      dot = dot + A[aBase + j] * lo + A[aBase + 16u + j] * hi;
    }
    let y = d * dot - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  reduce_and_store(tid, n, m, sum);
}

// ── Q5_0: 32-elem blocks, 6 u32 each (f16 d | pad | qh:u32 | qs[16]) ────
// 5th bit per value from qh; x[j] = d·((4-bit | hi-bit) - 16).

@compute @workgroup_size(WG_SIZE, 1, 1)
fn matmul_gguf_q5_0(@builtin(local_invocation_id) lid: vec3u,
                    @builtin(workgroup_id) wid: vec3u) {
  let n = wid.x + wid.z * 65535u;
  let m = wid.y;
  let tid = lid.x;
  let K = params.K;
  let nBlocks = K / 32u;

  var sum: f32 = 0.0;
  var comp: f32 = 0.0;
  for (var blk = tid; blk < nBlocks; blk = blk + WG_SIZE) {
    let base = (n * nBlocks + blk) * 6u;          // u32 words
    let d = unpack2x16float(W[base]).x;
    let qh = W[base + 1u];                         // 32 high bits
    let qsByte = (base + 2u) * 4u;                 // qs[16] start, in bytes
    let aBase = m * K + blk * 32u;
    var dot: f32 = 0.0;
    for (var j = 0u; j < 16u; j = j + 1u) {
      let q = wbyte(qsByte + j);
      let loBit = (qh >> j) & 1u;
      let hiBit = (qh >> (j + 16u)) & 1u;
      let lo = f32(i32((q & 0x0Fu) | (loBit << 4u)) - 16);
      let hi = f32(i32((q >> 4u) | (hiBit << 4u)) - 16);
      dot = dot + A[aBase + j] * lo + A[aBase + 16u + j] * hi;
    }
    let y = d * dot - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  reduce_and_store(tid, n, m, sum);
}

// ── IQ4_NL: 32-elem blocks, 5 u32 each (f16 d | pad | qs[16]) ──────────
// Q4_0-shaped repack with the non-linear iq4nl codebook: x = d·kvalues[q4].

@compute @workgroup_size(WG_SIZE, 1, 1)
fn matmul_gguf_iq4_nl(@builtin(local_invocation_id) lid: vec3u,
                      @builtin(workgroup_id) wid: vec3u) {
  let n = wid.x + wid.z * 65535u;
  let m = wid.y;
  let tid = lid.x;
  let K = params.K;
  let nBlocks = K / 32u;

  var sum: f32 = 0.0;
  var comp: f32 = 0.0;
  for (var blk = tid; blk < nBlocks; blk = blk + WG_SIZE) {
    let base = (n * nBlocks + blk) * 5u;          // u32 words
    let d = unpack2x16float(W[base]).x;
    let qsByte = (base + 1u) * 4u;                // qs[16] start, in bytes
    let aBase = m * K + blk * 32u;
    var dot: f32 = 0.0;
    for (var j = 0u; j < 16u; j = j + 1u) {
      let q = wbyte(qsByte + j);
      dot = dot + A[aBase + j] * iq4nl_val(q & 0x0Fu)
                + A[aBase + 16u + j] * iq4nl_val(q >> 4u);
    }
    let y = d * dot - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  reduce_and_store(tid, n, m, sum);
}

// ── IQ4_XS: 256-elem superblocks, 34 u32 each; unit = 32 elems ─────────
// Layout (raw 136 B): d(f16) @0 | scales_h(u16) @2 | scales_l[4] @4 | qs[128] @8.
// 6-bit sub-scale per 32 elems: low nibble from scales_l, 2 high bits from
// scales_h; dl = d·(ls-32), values via the iq4nl codebook.
// Mirrors dequantize_row_iq4_xs (gguf-dequant.ts dequantIQ4_XS).

@compute @workgroup_size(WG_SIZE, 1, 1)
fn matmul_gguf_iq4_xs(@builtin(local_invocation_id) lid: vec3u,
                      @builtin(workgroup_id) wid: vec3u) {
  let n = wid.x + wid.z * 65535u;
  let m = wid.y;
  let tid = lid.x;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;

  var sum: f32 = 0.0;
  var comp: f32 = 0.0;
  for (var u = tid; u < nUnits; u = u + WG_SIZE) {
    let sb = u / 8u;
    let sub = u % 8u;                        // 32-elem chunk index (ib)
    let wordBase = (n * nSB + sb) * 34u;
    let byteBase = wordBase * 4u;
    let w0 = W[wordBase];
    let d = unpack2x16float(w0).x;           // f16 d @ bytes 0-1
    let scalesH = w0 >> 16u;                 // u16 scales_h @ bytes 2-3
    let slByte = wbyte(byteBase + 4u + (sub >> 1u));
    let ls = ((slByte >> (4u * (sub & 1u))) & 0x0Fu) | (((scalesH >> (2u * sub)) & 3u) << 4u);
    let dl = d * f32(i32(ls) - 32);
    let qBase = byteBase + 8u + sub * 16u;   // 16 qs bytes for this chunk
    let aBase = m * K + sb * 256u + sub * 32u;
    var dot: f32 = 0.0;
    for (var j = 0u; j < 16u; j = j + 1u) {
      let q = wbyte(qBase + j);
      dot = dot + A[aBase + j] * iq4nl_val(q & 0x0Fu)
                + A[aBase + 16u + j] * iq4nl_val(q >> 4u);
    }
    let y = dl * dot - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  reduce_and_store(tid, n, m, sum);
}

// ── Q2_K: 256-elem superblocks, 21 u32 each; unit = 16-elem sub-block ───
// Layout: scales[16] @0 | qs[64] @16 | d(f16) @80 | dmin(f16) @82.
// Mirrors dequantize_row_q2_K: 16 sub-blocks, each with a 4-bit scale + 4-bit
// min byte; 2-bit quants selected by shift within two 128-element groups.

@compute @workgroup_size(WG_SIZE, 1, 1)
fn matmul_gguf_q2_k(@builtin(local_invocation_id) lid: vec3u,
                    @builtin(workgroup_id) wid: vec3u) {
  let n = wid.x + wid.z * 65535u;
  let m = wid.y;
  let tid = lid.x;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 16u;

  var sum: f32 = 0.0;
  var comp: f32 = 0.0;
  for (var u = tid; u < nUnits; u = u + WG_SIZE) {
    let sb = u / 16u;
    let sub = u % 16u;                       // scale index (is)
    let wordBase = (n * nSB + sb) * 21u;
    let byteBase = wordBase * 4u;
    let dm = unpack2x16float(W[wordBase + 20u]);   // d @80, dmin @82
    let sc = wbyte(byteBase + sub);
    let dl = dm.x * f32(sc & 0x0Fu);
    let ml = dm.y * f32(sc >> 4u);
    let group = select(0u, 1u, sub >= 8u);
    let jhalf = sub & 7u;
    let shift = (jhalf >> 1u) * 2u;
    let half = jhalf & 1u;
    let qBase = byteBase + 16u + group * 32u + half * 16u;
    let aBase = m * K + sb * 256u + sub * 16u;
    var dot: f32 = 0.0;
    for (var l = 0u; l < 16u; l = l + 1u) {
      let q2 = (wbyte(qBase + l) >> shift) & 3u;
      dot = dot + A[aBase + l] * (dl * f32(q2) - ml);
    }
    let y = dot - comp; let t = sum + y; comp = (t - sum) - y; sum = t;
  }
  reduce_and_store(tid, n, m, sum);
}

// ── IQ2_XXS: 256-elem superblocks, 17 u32 each; unit = 32-elem ib32 ─────
// Layout (repacked 66→68 B): d(f16) @0 | qs[32] u16 @2 (8 ib32 groups of 8
// bytes). Per ib32: 4 grid indices (aux0 bytes) + aux1 u32 (4×7-bit sign
// idx + 4-bit scale ls in top nibble). db = d·(0.5+ls)·0.25; each of the 4
// grid lookups yields 8 magnitude bytes, sign-flipped per ksigns_iq2xs.
// Tables (iq2xxs_grid[256]×8 bytes, ksigns_iq2xs[128]) and bit math mirror
// dequantize_row_iq2_xxs (gguf-dequant.ts dequantIQ2_XXS). Generated bit-
// exact by scripts/gen-iq2xxs-tables.mjs.
const IQ2XXS_GRID = array<u32, 512>(
  0x08080808u, 0x08080808u, 0x0808082bu, 0x08080808u, 0x08081919u, 0x08080808u, 0x08082b08u, 0x08080808u,
  0x08082b2bu, 0x08080808u, 0x08190819u, 0x08080808u, 0x08191908u, 0x08080808u, 0x082b0808u, 0x08080808u,
  0x082b082bu, 0x08080808u, 0x082b2b08u, 0x08080808u, 0x082b2b2bu, 0x08080808u, 0x19080819u, 0x08080808u,
  0x19081908u, 0x08080808u, 0x19190808u, 0x08080808u, 0x19192b08u, 0x08080808u, 0x192b0819u, 0x08080808u,
  0x192b1908u, 0x08080808u, 0x2b080808u, 0x08080808u, 0x2b08082bu, 0x08080808u, 0x2b082b2bu, 0x08080808u,
  0x2b2b082bu, 0x08080808u, 0x08080819u, 0x08080819u, 0x08081908u, 0x08080819u, 0x08190808u, 0x08080819u,
  0x08191919u, 0x08080819u, 0x19080808u, 0x08080819u, 0x2b081908u, 0x08080819u, 0x2b192b08u, 0x08080819u,
  0x08080808u, 0x0808082bu, 0x0808082bu, 0x0808082bu, 0x082b082bu, 0x0808082bu, 0x2b08082bu, 0x0808082bu,
  0x08080819u, 0x08081908u, 0x08081908u, 0x08081908u, 0x08190808u, 0x08081908u, 0x082b0819u, 0x08081908u,
  0x082b1908u, 0x08081908u, 0x19080808u, 0x08081908u, 0x1908082bu, 0x08081908u, 0x19082b08u, 0x08081908u,
  0x192b0808u, 0x08081908u, 0x2b080819u, 0x08081908u, 0x2b081908u, 0x08081908u, 0x2b190808u, 0x08081908u,
  0x2b2b1908u, 0x08081908u, 0x08080808u, 0x08081919u, 0x0808082bu, 0x08081919u, 0x08082b08u, 0x08081919u,
  0x082b0808u, 0x08081919u, 0x1908192bu, 0x08081919u, 0x192b2b19u, 0x08081919u, 0x2b080808u, 0x08081919u,
  0x2b190819u, 0x08081919u, 0x08082b19u, 0x0808192bu, 0x08190808u, 0x0808192bu, 0x19080808u, 0x0808192bu,
  0x2b081908u, 0x0808192bu, 0x2b2b1908u, 0x0808192bu, 0x08080808u, 0x08082b08u, 0x08081919u, 0x08082b08u,
  0x08082b08u, 0x08082b08u, 0x08191908u, 0x08082b08u, 0x082b2b08u, 0x08082b08u, 0x19080819u, 0x08082b08u,
  0x19081908u, 0x08082b08u, 0x19190808u, 0x08082b08u, 0x1919082bu, 0x08082b08u, 0x2b082b08u, 0x08082b08u,
  0x08081908u, 0x08082b19u, 0x19080808u, 0x08082b19u, 0x0808082bu, 0x08082b2bu, 0x08191908u, 0x08082b2bu,
  0x08080819u, 0x08190808u, 0x08081908u, 0x08190808u, 0x08190808u, 0x08190808u, 0x082b0819u, 0x08190808u,
  0x19080808u, 0x08190808u, 0x192b0808u, 0x08190808u, 0x2b081908u, 0x08190808u, 0x2b190808u, 0x08190808u,
  0x2b191919u, 0x08190808u, 0x08080808u, 0x08190819u, 0x08082b08u, 0x08190819u, 0x082b0808u, 0x08190819u,
  0x19190808u, 0x08190819u, 0x19192b2bu, 0x08190819u, 0x2b080808u, 0x08190819u, 0x082b1908u, 0x0819082bu,
  0x19081919u, 0x0819082bu, 0x08080808u, 0x08191908u, 0x08082b08u, 0x08191908u, 0x082b0808u, 0x08191908u,
  0x082b1919u, 0x08191908u, 0x19082b19u, 0x08191908u, 0x2b080808u, 0x08191908u, 0x08192b08u, 0x08191919u,
  0x192b082bu, 0x08191919u, 0x08080808u, 0x0819192bu, 0x0819192bu, 0x0819192bu, 0x08080819u, 0x08192b08u,
  0x08081908u, 0x08192b08u, 0x08190808u, 0x08192b08u, 0x19080808u, 0x08192b08u, 0x2b080819u, 0x08192b08u,
  0x08080808u, 0x08192b19u, 0x08081919u, 0x08192b19u, 0x2b2b0808u, 0x08192b19u, 0x19190819u, 0x08192b2bu,
  0x08080808u, 0x082b0808u, 0x0808082bu, 0x082b0808u, 0x08082b2bu, 0x082b0808u, 0x19081908u, 0x082b0808u,
  0x192b0819u, 0x082b0808u, 0x2b080808u, 0x082b0808u, 0x2b08082bu, 0x082b0808u, 0x082b2b19u, 0x082b0819u,
  0x19082b08u, 0x082b0819u, 0x08080808u, 0x082b082bu, 0x0808082bu, 0x082b082bu, 0x08080819u, 0x082b1908u,
  0x08081908u, 0x082b1908u, 0x08190808u, 0x082b1908u, 0x19080808u, 0x082b1908u, 0x1919192bu, 0x082b1908u,
  0x08080808u, 0x082b1919u, 0x19080819u, 0x082b1919u, 0x192b1908u, 0x082b1919u, 0x2b190808u, 0x082b192bu,
  0x08082b08u, 0x082b2b08u, 0x082b0808u, 0x082b2b08u, 0x2b191908u, 0x082b2b08u, 0x19081908u, 0x082b2b2bu,
  0x08080819u, 0x19080808u, 0x08081908u, 0x19080808u, 0x08190808u, 0x19080808u, 0x08192b08u, 0x19080808u,
  0x082b0819u, 0x19080808u, 0x082b1908u, 0x19080808u, 0x19080808u, 0x19080808u, 0x19082b08u, 0x19080808u,
  0x1919192bu, 0x19080808u, 0x192b0808u, 0x19080808u, 0x2b080819u, 0x19080808u, 0x2b081908u, 0x19080808u,
  0x2b190808u, 0x19080808u, 0x08080808u, 0x19080819u, 0x082b0808u, 0x19080819u, 0x192b0819u, 0x19080819u,
  0x2b080808u, 0x19080819u, 0x2b081919u, 0x19080819u, 0x08080819u, 0x1908082bu, 0x08190808u, 0x1908082bu,
  0x19082b08u, 0x1908082bu, 0x1919192bu, 0x1908082bu, 0x192b2b08u, 0x1908082bu, 0x08080808u, 0x19081908u,
  0x08082b08u, 0x19081908u, 0x082b0808u, 0x19081908u, 0x2b080808u, 0x19081908u, 0x2b192b19u, 0x19081908u,
  0x0819082bu, 0x19081919u, 0x082b1908u, 0x19081919u, 0x08080808u, 0x1908192bu, 0x08080819u, 0x19082b08u,
  0x08081908u, 0x19082b08u, 0x08190808u, 0x19082b08u, 0x19080808u, 0x19082b08u, 0x19081919u, 0x19082b08u,
  0x08080808u, 0x19082b19u, 0x19192b08u, 0x19082b19u, 0x192b0819u, 0x19082b19u, 0x2b08082bu, 0x19082b19u,
  0x19081919u, 0x19082b2bu, 0x2b190808u, 0x19082b2bu, 0x08080808u, 0x19190808u, 0x08082b08u, 0x19190808u,
  0x08190819u, 0x19190808u, 0x08192b19u, 0x19190808u, 0x082b0808u, 0x19190808u, 0x2b080808u, 0x19190808u,
  0x2b082b08u, 0x19190808u, 0x08081908u, 0x19190819u, 0x1908082bu, 0x19190819u, 0x2b2b1908u, 0x19190819u,
  0x2b190819u, 0x1919082bu, 0x2b190808u, 0x19191908u, 0x2b19082bu, 0x19191908u, 0x08082b2bu, 0x19191919u,
  0x08080819u, 0x1919192bu, 0x19191908u, 0x1919192bu, 0x08080808u, 0x19192b08u, 0x08190819u, 0x19192b08u,
  0x08192b19u, 0x19192b08u, 0x192b1908u, 0x19192b08u, 0x19080808u, 0x19192b19u, 0x08082b08u, 0x19192b2bu,
  0x08081908u, 0x192b0808u, 0x08190808u, 0x192b0808u, 0x19080808u, 0x192b0808u, 0x192b2b08u, 0x192b0808u,
  0x08080808u, 0x192b0819u, 0x19191919u, 0x192b0819u, 0x08192b08u, 0x192b082bu, 0x192b0808u, 0x192b082bu,
  0x08080808u, 0x192b1908u, 0x08081919u, 0x192b1908u, 0x08190808u, 0x192b1919u, 0x0819082bu, 0x192b1919u,
  0x2b081908u, 0x192b1919u, 0x1908082bu, 0x192b2b08u, 0x08080808u, 0x2b080808u, 0x0808082bu, 0x2b080808u,
  0x08082b2bu, 0x2b080808u, 0x19080819u, 0x2b080808u, 0x2b08082bu, 0x2b080808u, 0x08081908u, 0x2b080819u,
  0x08192b08u, 0x2b080819u, 0x19080808u, 0x2b080819u, 0x08190819u, 0x2b08082bu, 0x08080819u, 0x2b081908u,
  0x08081908u, 0x2b081908u, 0x08190808u, 0x2b081908u, 0x08191919u, 0x2b081908u, 0x19080808u, 0x2b081908u,
  0x192b0808u, 0x2b081908u, 0x08080808u, 0x2b081919u, 0x1908192bu, 0x2b081919u, 0x2b191908u, 0x2b081919u,
  0x08082b19u, 0x2b08192bu, 0x19080808u, 0x2b08192bu, 0x192b0808u, 0x2b08192bu, 0x0808082bu, 0x2b082b08u,
  0x08081908u, 0x2b082b19u, 0x08190819u, 0x2b082b2bu, 0x08081908u, 0x2b190808u, 0x08190808u, 0x2b190808u,
  0x082b1908u, 0x2b190808u, 0x19080808u, 0x2b190808u, 0x2b2b0819u, 0x2b190808u, 0x0819192bu, 0x2b190819u,
  0x2b080808u, 0x2b190819u, 0x19081919u, 0x2b19082bu, 0x08080808u, 0x2b191908u, 0x082b082bu, 0x2b191908u,
  0x19081908u, 0x2b191908u, 0x19190819u, 0x2b191919u, 0x2b080819u, 0x2b192b08u, 0x082b0808u, 0x2b192b19u,
  0x0808082bu, 0x2b2b0808u, 0x19190808u, 0x2b2b0808u, 0x2b081919u, 0x2b2b0808u, 0x08082b19u, 0x2b2b0819u,
  0x08080808u, 0x2b2b082bu, 0x08192b08u, 0x2b2b1908u, 0x19190808u, 0x2b2b2b08u, 0x08081908u, 0x2b2b2b19u,
);
const IQ2XXS_SIGNS = array<u32, 32>(
  0x03828100u, 0x87060584u, 0x8b0a0988u, 0x0f8e8d0cu, 0x93121190u, 0x17969514u, 0x1b9a9918u, 0x9f1e1d9cu,
  0xa32221a0u, 0x27a6a524u, 0x2baaa928u, 0xaf2e2dacu, 0x33b2b130u, 0xb73635b4u, 0xbb3a39b8u, 0x3fbebd3cu,
  0xc34241c0u, 0x47c6c544u, 0x4bcac948u, 0xcf4e4dccu, 0x53d2d150u, 0xd75655d4u, 0xdb5a59d8u, 0x5fdedd5cu,
  0x63e2e160u, 0xe76665e4u, 0xeb6a69e8u, 0x6feeed6cu, 0xf37271f0u, 0x77f6f574u, 0x7bfaf978u, 0xff7e7dfcu,
);

/** Magnitude byte j (0..7) of iq2xxs_grid entry `idx` (0..255). */
fn iq2xxs_grid_byte(idx: u32, j: u32) -> f32 {
  let word = IQ2XXS_GRID[idx * 2u + (j >> 2u)];
  return f32((word >> ((j & 3u) * 8u)) & 0xFFu);
}

/** Sign-bit byte for the 7-bit sign index `s` (0..127). */
fn iq2xxs_sign_byte(s: u32) -> u32 {
  let word = IQ2XXS_SIGNS[s >> 2u];
  return (word >> ((s & 3u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(WG_SIZE, 1, 1)
fn matmul_gguf_iq2_xxs(@builtin(local_invocation_id) lid: vec3u,
                       @builtin(workgroup_id) wid: vec3u) {
  let n = wid.x + wid.z * 65535u;
  let m = wid.y;
  let tid = lid.x;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;

  var sum: f32 = 0.0;
  var comp: f32 = 0.0;
  for (var u = tid; u < nUnits; u = u + WG_SIZE) {
    let sb = u / 8u;
    let ib = u % 8u;                         // 32-elem ib32 group
    let wordBase = (n * nSB + sb) * 17u;
    let d = unpack2x16float(W[wordBase]).x;  // f16 d @ bytes 0-1
    let g = wordBase * 4u + 2u + ib * 8u;    // 8 bytes: aux0[4] | aux1(u32)
    let aux1 = wbyte(g + 4u) | (wbyte(g + 5u) << 8u)
             | (wbyte(g + 6u) << 16u) | (wbyte(g + 7u) << 24u);
    let ls = aux1 >> 28u;
    let db = d * (0.5 + f32(ls)) * 0.25;
    let aBase = m * K + sb * 256u + ib * 32u;
    var dot: f32 = 0.0;
    for (var l = 0u; l < 4u; l = l + 1u) {
      let gridIdx = wbyte(g + l);
      let signs = iq2xxs_sign_byte((aux1 >> (7u * l)) & 127u);
      for (var j = 0u; j < 8u; j = j + 1u) {
        let mag = iq2xxs_grid_byte(gridIdx, j);
        let s = select(1.0, -1.0, ((signs >> j) & 1u) == 1u);
        dot = dot + A[aBase + l * 8u + j] * mag * s;
      }
    }
    let y = db * dot - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  reduce_and_store(tid, n, m, sum);
}

// ── Q3_K: 256-elem superblocks, 28 u32 each; unit = 16-elem sub-block ───
// Layout: hmask[32] @0 | qs[64] @32 | scales[12] @96 | d(f16) @108.
// Mirrors dequantize_row_q3_K incl. the 6-bit scale aux-shuffle and the
// per-element high bit from hmask (value = q_low2 - (hbit ? 0 : 4)).

@compute @workgroup_size(WG_SIZE, 1, 1)
fn matmul_gguf_q3_k(@builtin(local_invocation_id) lid: vec3u,
                    @builtin(workgroup_id) wid: vec3u) {
  let n = wid.x + wid.z * 65535u;
  let m = wid.y;
  let tid = lid.x;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 16u;
  let KM1 = 0x03030303u;
  let KM2 = 0x0f0f0f0fu;

  var sum: f32 = 0.0;
  var comp: f32 = 0.0;
  for (var u = tid; u < nUnits; u = u + WG_SIZE) {
    let sb = u / 16u;
    let sub = u % 16u;
    let wordBase = (n * nSB + sb) * 28u;
    let byteBase = wordBase * 4u;
    let dAll = unpack2x16float(W[wordBase + 27u]).x;   // d @108

    // Unpack the 6-bit signed scale for sub-block `sub`.
    let s0 = W[wordBase + 24u];
    let s1 = W[wordBase + 25u];
    let tmp = W[wordBase + 26u];
    var auxWord: u32;
    let widx = sub >> 2u;
    if (widx == 0u) {
      auxWord = (s0 & KM2) | (((tmp >> 0u) & KM1) << 4u);
    } else if (widx == 1u) {
      auxWord = (s1 & KM2) | (((tmp >> 2u) & KM1) << 4u);
    } else if (widx == 2u) {
      auxWord = ((s0 >> 4u) & KM2) | (((tmp >> 4u) & KM1) << 4u);
    } else {
      auxWord = ((s1 >> 4u) & KM2) | (((tmp >> 6u) & KM1) << 4u);
    }
    let scByte = (auxWord >> ((sub & 3u) * 8u)) & 0xFFu;
    let dl = dAll * f32(i32(scByte) - 32);

    let group = select(0u, 1u, sub >= 8u);
    let jhalf = sub & 7u;
    let j = jhalf >> 1u;
    let shift = j * 2u;
    let half = jhalf & 1u;
    let mbit = 1u << (group * 4u + j);
    let qBase = byteBase + 32u + group * 32u + half * 16u;
    let hBase = byteBase + half * 16u;             // hmask @0
    let aBase = m * K + sb * 256u + sub * 16u;
    var dot: f32 = 0.0;
    for (var l = 0u; l < 16u; l = l + 1u) {
      let q3 = (wbyte(qBase + l) >> shift) & 3u;
      let hoff = select(4.0, 0.0, (wbyte(hBase + l) & mbit) != 0u);
      dot = dot + A[aBase + l] * (dl * (f32(q3) - hoff));
    }
    let y = dot - comp; let t = sum + y; comp = (t - sum) - y; sum = t;
  }
  reduce_and_store(tid, n, m, sum);
}

// ── Q8_0: 32-elem blocks, 9 u32 each ──────────────────────────────────

@compute @workgroup_size(WG_SIZE, 1, 1)
fn matmul_gguf_q8_0(@builtin(local_invocation_id) lid: vec3u,
                    @builtin(workgroup_id) wid: vec3u) {
  let n = wid.x + wid.z * 65535u;
  let m = wid.y;
  let tid = lid.x;
  let K = params.K;
  let nBlocks = K / 32u;

  var sum: f32 = 0.0;
  var comp: f32 = 0.0;
  for (var blk = tid; blk < nBlocks; blk = blk + WG_SIZE) {
    let base = (n * nBlocks + blk) * 9u;
    let d = unpack2x16float(W[base]).x;
    let aBase = m * K + blk * 32u;
    var dot: f32 = 0.0;
    for (var w = 0u; w < 8u; w = w + 1u) {
      let word = W[base + 1u + w];
      let ai = aBase + w * 4u;
      dot = dot + A[ai]      * i8_byte(word, 0u)
                + A[ai + 1u] * i8_byte(word, 1u)
                + A[ai + 2u] * i8_byte(word, 2u)
                + A[ai + 3u] * i8_byte(word, 3u);
    }
    // Kahan-compensated accumulation of the block contribution
    let y = d * dot - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  reduce_and_store(tid, n, m, sum);
}

// ── Q4_K: 256-elem superblocks, 36 u32 each; unit = 32 elems ───────────

@compute @workgroup_size(WG_SIZE, 1, 1)
fn matmul_gguf_q4_k(@builtin(local_invocation_id) lid: vec3u,
                    @builtin(workgroup_id) wid: vec3u) {
  let n = wid.x + wid.z * 65535u;
  let m = wid.y;
  let tid = lid.x;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;

  var sum: f32 = 0.0;
  var comp: f32 = 0.0;
  for (var u = tid; u < nUnits; u = u + WG_SIZE) {
    let sb = u / 8u;
    let sub = u % 8u;                       // scale/min index j
    let sbBase = (n * nSB + sb) * 36u;      // words
    let dm = unpack2x16float(W[sbBase]);    // (d, dmin)
    let sm = scale_min_k4(sub, sbBase * 4u + 4u);
    let d1 = dm.x * sm.x;
    let min1 = dm.y * sm.y;
    // chunk c = sub>>1 owns qs bytes [c*32, c*32+32); even sub = low nibbles
    let qsWord = sbBase + 4u + (sub >> 1u) * 8u;
    let hi = (sub & 1u) == 1u;
    let aBase = m * K + sb * 256u + sub * 32u;
    var dot: f32 = 0.0;
    var asum: f32 = 0.0;
    for (var w = 0u; w < 8u; w = w + 1u) {
      let word = W[qsWord + w];
      let ai = aBase + w * 4u;
      for (var j = 0u; j < 4u; j = j + 1u) {
        let b = (word >> (j * 8u)) & 0xFFu;
        let q = select(b & 0x0Fu, b >> 4u, hi);
        let a = A[ai + j];
        dot = dot + a * f32(q);
        asum = asum + a;
      }
    }
    let y = (d1 * dot - min1 * asum) - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  reduce_and_store(tid, n, m, sum);
}

// ── Q5_K: 256-elem superblocks, 44 u32 each; unit = 32 elems ───────────

@compute @workgroup_size(WG_SIZE, 1, 1)
fn matmul_gguf_q5_k(@builtin(local_invocation_id) lid: vec3u,
                    @builtin(workgroup_id) wid: vec3u) {
  let n = wid.x + wid.z * 65535u;
  let m = wid.y;
  let tid = lid.x;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;

  var sum: f32 = 0.0;
  var comp: f32 = 0.0;
  for (var u = tid; u < nUnits; u = u + WG_SIZE) {
    let sb = u / 8u;
    let sub = u % 8u;
    let sbBase = (n * nSB + sb) * 44u;      // words
    let sbByte = sbBase * 4u;
    let dm = unpack2x16float(W[sbBase]);
    let sm = scale_min_k4(sub, sbByte + 4u);
    let d1 = dm.x * sm.x;
    let min1 = dm.y * sm.y;
    let qsWord = sbBase + 12u + (sub >> 1u) * 8u;  // qs after 16B header + 32B qh
    let qhByte = sbByte + 16u;                     // qh[32]
    let hi = (sub & 1u) == 1u;
    let aBase = m * K + sb * 256u + sub * 32u;
    var dot: f32 = 0.0;
    var asum: f32 = 0.0;
    for (var w = 0u; w < 8u; w = w + 1u) {
      let word = W[qsWord + w];
      let ai = aBase + w * 4u;
      for (var j = 0u; j < 4u; j = j + 1u) {
        let l = w * 4u + j;                  // elem index within the 32
        let b = (word >> (j * 8u)) & 0xFFu;
        let q4 = select(b & 0x0Fu, b >> 4u, hi);
        let h = (wbyte(qhByte + l) >> sub) & 1u;   // 5th bit: mask 1<<sub
        let a = A[ai + j];
        dot = dot + a * f32(q4 + h * 16u);
        asum = asum + a;
      }
    }
    let y = (d1 * dot - min1 * asum) - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  reduce_and_store(tid, n, m, sum);
}

// ── Q6_K: 256-elem superblocks, 53 u32 each; unit = 32 elems ───────────

@compute @workgroup_size(WG_SIZE, 1, 1)
fn matmul_gguf_q6_k(@builtin(local_invocation_id) lid: vec3u,
                    @builtin(workgroup_id) wid: vec3u) {
  let n = wid.x + wid.z * 65535u;
  let m = wid.y;
  let tid = lid.x;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;

  var sum: f32 = 0.0;
  var comp: f32 = 0.0;
  for (var u = tid; u < nUnits; u = u + WG_SIZE) {
    let sb = u / 8u;
    let sub = u % 8u;
    let half = sub >> 2u;                   // 0 or 1 (which 128-elem half)
    let s = sub & 3u;                       // q1..q4 group within the half
    let sbBase = (n * nSB + sb) * 53u;      // words
    let sbByte = sbBase * 4u;
    let d = unpack2x16float(W[sbBase + 52u]).x;
    // ql[l] for s∈{0,2}, ql[l+32] for s∈{1,3}; low nibble for s<2
    let qlByte = sbByte + half * 64u + select(0u, 32u, (s & 1u) == 1u);
    let qhByte = sbByte + 128u + half * 32u;
    let scByte = sbByte + 192u + half * 8u + 2u * s;  // + l/16
    let hShift = 2u * s;
    let lowNib = s < 2u;
    let aBase = m * K + sb * 256u + sub * 32u;
    // Scale changes at l=16: accumulate the two 16-elem sub-dots separately.
    var dot0: f32 = 0.0;
    var dot1: f32 = 0.0;
    for (var l = 0u; l < 32u; l = l + 1u) {
      let bl = wbyte(qlByte + l);
      let q4 = select(bl >> 4u, bl & 0x0Fu, lowNib);
      let h = (wbyte(qhByte + l) >> hShift) & 3u;
      let qv = f32(q4 | (h << 4u)) - 32.0;   // 6-bit value, centered
      let av = A[aBase + l] * qv;
      if (l < 16u) { dot0 = dot0 + av; } else { dot1 = dot1 + av; }
    }
    let sc0 = f32(bitcast<i32>(wbyte(scByte) << 24u) >> 24u);        // i8
    let sc1 = f32(bitcast<i32>(wbyte(scByte + 1u) << 24u) >> 24u);   // i8
    let y = d * (sc0 * dot0 + sc1 * dot1) - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  reduce_and_store(tid, n, m, sum);
}

// ── Q4_K tiled: TN rows/workgroup, shared activation staging ───────────
// Same superblock decode as matmul_gguf_q4_k; nibbles extracted 4-at-a-time
// via word masks (each masked byte is 0..15, exactly representable in f32).

@compute @workgroup_size(TWG, 1, 1)
fn matmul_gguf_q4_k_tiled(@builtin(local_invocation_id) lid: vec3u,
                          @builtin(workgroup_id) wid: vec3u) {
  let tid = lid.x;
  let TPR = TWG / TN;
  let lane = tid % TPR;
  let row = tid / TPR;
  let n = (wid.x + wid.z * 65535u) * TN + row;
  let m = wid.y;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;
  let nChunks = (nUnits + TPR - 1u) / TPR;
  let aRow = m * K;
  let valid = n < params.N;

  var acc: f32 = 0.0;
  for (var c = 0u; c < nChunks; c = c + 1u) {
    workgroupBarrier();                       // previous chunk's readers done
    tile_stage(tid, aRow, c * TPR * 32u, K);
    workgroupBarrier();

    let u = c * TPR + lane;
    if (u < nUnits && valid) {
      let sb = u / 8u;
      let sub = u % 8u;                       // scale/min index j
      let sbBase = (n * nSB + sb) * 36u;      // words
      let dm = unpack2x16float(W[sbBase]);    // (d, dmin)
      let sm = scale_min_k4(sub, sbBase * 4u + 4u);
      let d1 = dm.x * sm.x;
      let min1 = dm.y * sm.y;
      let qsWord = sbBase + 4u + (sub >> 1u) * 8u;
      let hi = (sub & 1u) == 1u;
      var dq: f32 = 0.0;
      var asum: f32 = 0.0;
      for (var w = 0u; w < 8u; w = w + 1u) {
        let word = W[qsWord + w];
        let nib = select(word & 0x0F0F0F0Fu, (word >> 4u) & 0x0F0F0F0Fu, hi);
        let q4 = vec4<f32>(
          f32(nib & 0xFFu), f32((nib >> 8u) & 0xFFu),
          f32((nib >> 16u) & 0xFFu), f32(nib >> 24u),
        );
        let a4 = a_tile[lane * 8u + w];
        dq = dq + dot(a4, q4);
        asum = asum + a4.x + a4.y + a4.z + a4.w;
      }
      acc = acc + (d1 * dq - min1 * asum);
    }
  }
  tile_reduce_store(tid, lane, n, m, acc);
}

// ── Q5_K tiled: TN rows/workgroup, shared activation staging ───────────
// Same superblock decode as matmul_gguf_q5_k. qh[32] sits at byte 16
// (word-aligned, words 4..11); the per-byte 5th bit is (qh_byte >> sub) & 1,
// extracted 4-at-a-time: byte j of (hw >> sub) has bit sub of source byte j
// in its low bit (sub ≤ 7, never crosses a byte boundary under the
// 0x01010101 mask).

@compute @workgroup_size(TWG, 1, 1)
fn matmul_gguf_q5_k_tiled(@builtin(local_invocation_id) lid: vec3u,
                          @builtin(workgroup_id) wid: vec3u) {
  let tid = lid.x;
  let TPR = TWG / TN;
  let lane = tid % TPR;
  let row = tid / TPR;
  let n = (wid.x + wid.z * 65535u) * TN + row;
  let m = wid.y;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;
  let nChunks = (nUnits + TPR - 1u) / TPR;
  let aRow = m * K;
  let valid = n < params.N;

  var acc: f32 = 0.0;
  for (var c = 0u; c < nChunks; c = c + 1u) {
    workgroupBarrier();
    tile_stage(tid, aRow, c * TPR * 32u, K);
    workgroupBarrier();

    let u = c * TPR + lane;
    if (u < nUnits && valid) {
      let sb = u / 8u;
      let sub = u % 8u;
      let sbBase = (n * nSB + sb) * 44u;      // words
      let dm = unpack2x16float(W[sbBase]);    // (d, dmin)
      let sm = scale_min_k4(sub, sbBase * 4u + 4u);
      let d1 = dm.x * sm.x;
      let min1 = dm.y * sm.y;
      let qsWord = sbBase + 12u + (sub >> 1u) * 8u;  // qs after 16B header + 32B qh
      let qhWord = sbBase + 4u;                      // qh[32]
      let hi = (sub & 1u) == 1u;
      var dq: f32 = 0.0;
      var asum: f32 = 0.0;
      for (var w = 0u; w < 8u; w = w + 1u) {
        let word = W[qsWord + w];
        let hw = W[qhWord + w];
        let nib = select(word & 0x0F0F0F0Fu, (word >> 4u) & 0x0F0F0F0Fu, hi);
        let q = nib | (((hw >> sub) & 0x01010101u) << 4u);   // 5-bit value 0..31
        let q5 = vec4<f32>(
          f32(q & 0xFFu), f32((q >> 8u) & 0xFFu),
          f32((q >> 16u) & 0xFFu), f32(q >> 24u),
        );
        let a4 = a_tile[lane * 8u + w];
        dq = dq + dot(a4, q5);
        asum = asum + a4.x + a4.y + a4.z + a4.w;
      }
      acc = acc + (d1 * dq - min1 * asum);
    }
  }
  tile_reduce_store(tid, lane, n, m, acc);
}

// ── Q6_K tiled: TN rows/workgroup, shared activation staging ───────────
// Same superblock decode as matmul_gguf_q6_k; ql/qh offsets are word-aligned
// so both are read as words, 4 elements per iteration. The qh shift trick is
// safe: hShift ≤ 6, so bits hShift..hShift+1 of each byte never cross a byte
// boundary under a whole-word shift + 0x03030303 mask.

@compute @workgroup_size(TWG, 1, 1)
fn matmul_gguf_q6_k_tiled(@builtin(local_invocation_id) lid: vec3u,
                          @builtin(workgroup_id) wid: vec3u) {
  let tid = lid.x;
  let TPR = TWG / TN;
  let lane = tid % TPR;
  let row = tid / TPR;
  let n = (wid.x + wid.z * 65535u) * TN + row;
  let m = wid.y;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;
  let nChunks = (nUnits + TPR - 1u) / TPR;
  let aRow = m * K;
  let valid = n < params.N;

  var acc: f32 = 0.0;
  for (var c = 0u; c < nChunks; c = c + 1u) {
    workgroupBarrier();
    tile_stage(tid, aRow, c * TPR * 32u, K);
    workgroupBarrier();

    let u = c * TPR + lane;
    if (u < nUnits && valid) {
      let sb = u / 8u;
      let sub = u % 8u;
      let half = sub >> 2u;                   // 0 or 1 (which 128-elem half)
      let s = sub & 3u;                       // q1..q4 group within the half
      let sbBase = (n * nSB + sb) * 53u;      // words
      let sbByte = sbBase * 4u;
      let d = unpack2x16float(W[sbBase + 52u]).x;
      // ql[l] for s∈{0,2}, ql[l+32] for s∈{1,3}; low nibble for s<2
      let qlWord = sbBase + half * 16u + select(0u, 8u, (s & 1u) == 1u);
      let qhWord = sbBase + 32u + half * 8u;
      let scByte = sbByte + 192u + half * 8u + 2u * s;
      let hShift = 2u * s;
      let lowNib = s < 2u;
      // Scale changes at l=16 (word 4): two separate sub-dots.
      var dq0: f32 = 0.0;
      var dq1: f32 = 0.0;
      for (var w = 0u; w < 8u; w = w + 1u) {
        let lw = W[qlWord + w];
        let hw = W[qhWord + w];
        let l4 = select((lw >> 4u) & 0x0F0F0F0Fu, lw & 0x0F0F0F0Fu, lowNib);
        let q = l4 | (((hw >> hShift) & 0x03030303u) << 4u);
        let qv = vec4<f32>(
          f32(q & 0xFFu), f32((q >> 8u) & 0xFFu),
          f32((q >> 16u) & 0xFFu), f32(q >> 24u),
        ) - vec4<f32>(32.0);
        let a4 = a_tile[lane * 8u + w];
        if (w < 4u) { dq0 = dq0 + dot(a4, qv); } else { dq1 = dq1 + dot(a4, qv); }
      }
      let sc0 = f32(bitcast<i32>(wbyte(scByte) << 24u) >> 24u);        // i8
      let sc1 = f32(bitcast<i32>(wbyte(scByte + 1u) << 24u) >> 24u);   // i8
      acc = acc + d * (sc0 * dq0 + sc1 * dq1);
    }
  }
  tile_reduce_store(tid, lane, n, m, acc);
}

// ── Q2_K tiled: TN rows/workgroup, shared activation staging ───────────
// Same superblock decode as matmul_gguf_q2_k. Q2_K/Q3_K sub-blocks are 16
// elements, so one 32-elem tile unit covers the consecutive PAIR
// (sub0=2p, sub1=2p+1). sub0 is even, so both subs share group/shift and
// their qs bytes are contiguous (8 words: w<4 → sub0, w≥4 → sub1).
// The whole-word 2-bit shift trick is safe: shift ≤ 6, bits shift..shift+1
// of each byte never cross a byte boundary under the 0x03030303 mask.

@compute @workgroup_size(TWG, 1, 1)
fn matmul_gguf_q2_k_tiled(@builtin(local_invocation_id) lid: vec3u,
                          @builtin(workgroup_id) wid: vec3u) {
  let tid = lid.x;
  let TPR = TWG / TN;
  let lane = tid % TPR;
  let row = tid / TPR;
  let n = (wid.x + wid.z * 65535u) * TN + row;
  let m = wid.y;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;
  let nChunks = (nUnits + TPR - 1u) / TPR;
  let aRow = m * K;
  let valid = n < params.N;

  var acc: f32 = 0.0;
  for (var c = 0u; c < nChunks; c = c + 1u) {
    workgroupBarrier();
    tile_stage(tid, aRow, c * TPR * 32u, K);
    workgroupBarrier();

    let u = c * TPR + lane;
    if (u < nUnits && valid) {
      let sb = u / 8u;
      let pair = u % 8u;
      let sub0 = pair * 2u;                   // even; sub1 = sub0 + 1
      let wordBase = (n * nSB + sb) * 21u;
      let byteBase = wordBase * 4u;
      let dm = unpack2x16float(W[wordBase + 20u]);   // d @80, dmin @82
      let sc0 = wbyte(byteBase + sub0);
      let sc1 = wbyte(byteBase + sub0 + 1u);
      let dl0 = dm.x * f32(sc0 & 0x0Fu);
      let ml0 = dm.y * f32(sc0 >> 4u);
      let dl1 = dm.x * f32(sc1 & 0x0Fu);
      let ml1 = dm.y * f32(sc1 >> 4u);
      let group = select(0u, 1u, sub0 >= 8u);
      let shift = ((sub0 & 7u) >> 1u) * 2u;
      let qWord = wordBase + 4u + group * 8u;  // qs @16 + group*32 bytes
      var dq0: f32 = 0.0;
      var as0: f32 = 0.0;
      var dq1: f32 = 0.0;
      var as1: f32 = 0.0;
      for (var w = 0u; w < 8u; w = w + 1u) {
        let q = (W[qWord + w] >> shift) & 0x03030303u;
        let q2 = vec4<f32>(
          f32(q & 0xFFu), f32((q >> 8u) & 0xFFu),
          f32((q >> 16u) & 0xFFu), f32(q >> 24u),
        );
        let a4 = a_tile[lane * 8u + w];
        let asum = a4.x + a4.y + a4.z + a4.w;
        if (w < 4u) { dq0 = dq0 + dot(a4, q2); as0 = as0 + asum; }
        else        { dq1 = dq1 + dot(a4, q2); as1 = as1 + asum; }
      }
      acc = acc + (dl0 * dq0 - ml0 * as0) + (dl1 * dq1 - ml1 * as1);
    }
  }
  tile_reduce_store(tid, lane, n, m, acc);
}

// ── Q2_K tiled on REPACKED data (lever C, repack_q2k.wgsl) ──────────────
// The load-time repack gives each 32-elem unit p = u%8 two contiguous qs
// words at wordBase + 4 + 2p: source shift-plane t = p&3 of group g = p>>2
// is byte-compacted so element l = w*4+k sits at bit (w&3)*8 + 2k of word
// (w<4 ? 0 : 1). Loads per unit drop 8→2 and the 4× cross-lane re-read of
// each qs word disappears; extractBits is one bitfield-extract op/element.
// Scales (words 0-3), d/dmin (word 20), the dl/ml w<4 split, a_tile
// indexing, and the acc expression are IDENTICAL to matmul_gguf_q2_k_tiled
// → bit-exact (gate 5 in scripts/test-gemv-tiled.mts).

@compute @workgroup_size(TWG, 1, 1)
fn matmul_gguf_q2_k_tiled_r(@builtin(local_invocation_id) lid: vec3u,
                            @builtin(workgroup_id) wid: vec3u) {
  let tid = lid.x;
  let TPR = TWG / TN;
  let lane = tid % TPR;
  let row = tid / TPR;
  let n = (wid.x + wid.z * 65535u) * TN + row;
  let m = wid.y;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;
  let nChunks = (nUnits + TPR - 1u) / TPR;
  let aRow = m * K;
  let valid = n < params.N;

  var acc: f32 = 0.0;
  for (var c = 0u; c < nChunks; c = c + 1u) {
    workgroupBarrier();
    tile_stage(tid, aRow, c * TPR * 32u, K);
    workgroupBarrier();

    let u = c * TPR + lane;
    if (u < nUnits && valid) {
      let sb = u / 8u;
      let pair = u % 8u;
      let sub0 = pair * 2u;                   // even; sub1 = sub0 + 1
      let wordBase = (n * nSB + sb) * 21u;
      let byteBase = wordBase * 4u;
      let dm = unpack2x16float(W[wordBase + 20u]);   // d @80, dmin @82
      let sc0 = wbyte(byteBase + sub0);
      let sc1 = wbyte(byteBase + sub0 + 1u);
      let dl0 = dm.x * f32(sc0 & 0x0Fu);
      let ml0 = dm.y * f32(sc0 >> 4u);
      let dl1 = dm.x * f32(sc1 & 0x0Fu);
      let ml1 = dm.y * f32(sc1 >> 4u);
      let qBase = wordBase + 4u + pair * 2u;   // unit-contiguous pair
      let w0 = W[qBase];
      let w1 = W[qBase + 1u];
      var dq0: f32 = 0.0;
      var as0: f32 = 0.0;
      var dq1: f32 = 0.0;
      var as1: f32 = 0.0;
      for (var w = 0u; w < 8u; w = w + 1u) {
        let src = select(w1, w0, w < 4u);
        let off = (w & 3u) * 8u;
        let q2 = vec4<f32>(
          f32(extractBits(src, off, 2u)),
          f32(extractBits(src, off + 2u, 2u)),
          f32(extractBits(src, off + 4u, 2u)),
          f32(extractBits(src, off + 6u, 2u)),
        );
        let a4 = a_tile[lane * 8u + w];
        let asum = a4.x + a4.y + a4.z + a4.w;
        if (w < 4u) { dq0 = dq0 + dot(a4, q2); as0 = as0 + asum; }
        else        { dq1 = dq1 + dot(a4, q2); as1 = as1 + asum; }
      }
      acc = acc + (dl0 * dq0 - ml0 * as0) + (dl1 * dq1 - ml1 * as1);
    }
  }
  tile_reduce_store(tid, lane, n, m, acc);
}

// ── Q2_K tiled GEMM on REPACKED data (Phase C4 — M-reuse) ───────────────
// Same decode bit-math as matmul_gguf_q2_k_tiled_r, but M (the verify-row
// count) is handled INSIDE the workgroup instead of in grid.y. Each lane
// decodes its 32-elem unit's 8 quants ONCE, then loops over the M staged
// activation rows accumulating per-m partials. This amortizes the 2-bit
// decode ALU and the weight loads across all M outputs — the no-reuse GEMV
// path re-decoded every weight once per row (V≈K). Grid.y = 1.
//
// Bit-exact requirement: the per-m accumulation order (dq0/as0/dq1/as1 split,
// then acc += (dl0*dq0 - ml0*as0) + (dl1*dq1 - ml1*as1)) is IDENTICAL to _r,
// so the M=1 column is byte-for-byte equal to matmul_gguf_q2_k_tiled_r.

@compute @workgroup_size(TWG, 1, 1)
fn matmul_gguf_q2_k_tiled_r_gemm(@builtin(local_invocation_id) lid: vec3u,
                                 @builtin(workgroup_id) wid: vec3u) {
  let tid = lid.x;
  let TPR = TWG / TN;
  let lane = tid % TPR;
  let row = tid / TPR;
  let n = (wid.x + wid.z * 65535u) * TN + row;
  let M = params.M;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;
  let nChunks = (nUnits + TPR - 1u) / TPR;
  let nVec = (TWG / TN) * 8u;
  let valid = n < params.N;

  var acc: array<f32, MAXM>;
  for (var mi = 0u; mi < M; mi = mi + 1u) { acc[mi] = 0.0; }

  for (var c = 0u; c < nChunks; c = c + 1u) {
    workgroupBarrier();
    tile_stage_m(tid, M, c * TPR * 32u, K);
    workgroupBarrier();

    let u = c * TPR + lane;
    if (u < nUnits && valid) {
      let sb = u / 8u;
      let pair = u % 8u;
      let sub0 = pair * 2u;                   // even; sub1 = sub0 + 1
      let wordBase = (n * nSB + sb) * 21u;
      let byteBase = wordBase * 4u;
      let dm = unpack2x16float(W[wordBase + 20u]);   // d @80, dmin @82
      let sc0 = wbyte(byteBase + sub0);
      let sc1 = wbyte(byteBase + sub0 + 1u);
      let dl0 = dm.x * f32(sc0 & 0x0Fu);
      let ml0 = dm.y * f32(sc0 >> 4u);
      let dl1 = dm.x * f32(sc1 & 0x0Fu);
      let ml1 = dm.y * f32(sc1 >> 4u);
      let qBase = wordBase + 4u + pair * 2u;   // unit-contiguous pair
      let w0 = W[qBase];
      let w1 = W[qBase + 1u];

      var dq0: array<f32, MAXM>;
      var as0: array<f32, MAXM>;
      var dq1: array<f32, MAXM>;
      var as1: array<f32, MAXM>;
      for (var mi = 0u; mi < M; mi = mi + 1u) {
        dq0[mi] = 0.0; as0[mi] = 0.0; dq1[mi] = 0.0; as1[mi] = 0.0;
      }
      for (var w = 0u; w < 8u; w = w + 1u) {
        let src = select(w1, w0, w < 4u);
        let off = (w & 3u) * 8u;
        let q2 = vec4<f32>(
          f32(extractBits(src, off, 2u)),
          f32(extractBits(src, off + 2u, 2u)),
          f32(extractBits(src, off + 4u, 2u)),
          f32(extractBits(src, off + 6u, 2u)),
        );
        for (var mi = 0u; mi < M; mi = mi + 1u) {
          let a4 = a_tile_m[mi * nVec + lane * 8u + w];
          let asum = a4.x + a4.y + a4.z + a4.w;
          if (w < 4u) { dq0[mi] = dq0[mi] + dot(a4, q2); as0[mi] = as0[mi] + asum; }
          else        { dq1[mi] = dq1[mi] + dot(a4, q2); as1[mi] = as1[mi] + asum; }
        }
      }
      for (var mi = 0u; mi < M; mi = mi + 1u) {
        acc[mi] = acc[mi] + (dl0 * dq0[mi] - ml0 * as0[mi]) + (dl1 * dq1[mi] - ml1 * as1[mi]);
      }
    }
  }
  for (var mi = 0u; mi < M; mi = mi + 1u) {
    tile_reduce_store(tid, lane, n, mi, acc[mi]);
  }
}

// ── Q2_K repacked, 2 units/lane (ILP probe — loads-in-flight) ────────────
// Same math as matmul_gguf_q2_k_tiled_r; the ONLY change is scheduling.
// Hypothesis: _r issues ~5 scalar loads per unit then runs a ~32-op serial
// ALU chain, so memory idles during ALU and ALU idles during loads, with 10
// barrier pairs/dispatch flushing any overlap (K=5120). Here each lane owns
// TWO units per chunk (ua = c*2*TPR + lane, ub = ua + TPR) and ALL global
// loads for both units are issued up front (8 independent words) before any
// decode ALU — double the loads in flight, half the barriers.
// Per-lane unit ORDER is unchanged (lane, lane+TPR, lane+2*TPR, ...) and the
// per-unit acc expression is identical → bit-exact with _r/_tiled (gate:
// byte-identical 2048-token A/B). Inactive units read superblock 0 (always
// present) and their contribution is dropped via select().

var<workgroup> a_tile2: array<vec4<f32>, (TWG / TN) * 16u>;

/** tile_stage for the double-width chunk: 2*TPR units = TPR*16 vec4s. */
fn tile_stage2(tid: u32, aRow: u32, elemBase: u32, K: u32) {
  let nVec = (TWG / TN) * 16u;
  for (var i = tid; i < nVec; i = i + TWG) {
    let e = elemBase + i * 4u;
    var v = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (e < K) {
      v = vec4<f32>(A[aRow + e], A[aRow + e + 1u], A[aRow + e + 2u], A[aRow + e + 3u]);
    }
    a_tile2[i] = v;
  }
}

/** One repacked unit's acc contribution — the exact _r inner block, fed
 *  from pre-loaded words. atBase = a_tile2 vec4 index of the unit's slice. */
fn q2k_unit_acc(dmw: u32, scw: u32, sub: u32, w0: u32, w1: u32, atBase: u32) -> f32 {
  let dm = unpack2x16float(dmw);
  let sc0 = (scw >> ((sub & 3u) * 8u)) & 0xFFu;
  let sc1 = (scw >> (((sub & 3u) + 1u) * 8u)) & 0xFFu; // sub even → same word
  let dl0 = dm.x * f32(sc0 & 0x0Fu);
  let ml0 = dm.y * f32(sc0 >> 4u);
  let dl1 = dm.x * f32(sc1 & 0x0Fu);
  let ml1 = dm.y * f32(sc1 >> 4u);
  var dq0: f32 = 0.0;
  var as0: f32 = 0.0;
  var dq1: f32 = 0.0;
  var as1: f32 = 0.0;
  for (var w = 0u; w < 8u; w = w + 1u) {
    let src = select(w1, w0, w < 4u);
    let off = (w & 3u) * 8u;
    let q2 = vec4<f32>(
      f32(extractBits(src, off, 2u)),
      f32(extractBits(src, off + 2u, 2u)),
      f32(extractBits(src, off + 4u, 2u)),
      f32(extractBits(src, off + 6u, 2u)),
    );
    let a4 = a_tile2[atBase + w];
    let asum = a4.x + a4.y + a4.z + a4.w;
    if (w < 4u) { dq0 = dq0 + dot(a4, q2); as0 = as0 + asum; }
    else        { dq1 = dq1 + dot(a4, q2); as1 = as1 + asum; }
  }
  return (dl0 * dq0 - ml0 * as0) + (dl1 * dq1 - ml1 * as1);
}

@compute @workgroup_size(TWG, 1, 1)
fn matmul_gguf_q2_k_tiled_r2(@builtin(local_invocation_id) lid: vec3u,
                             @builtin(workgroup_id) wid: vec3u) {
  let tid = lid.x;
  let TPR = TWG / TN;
  let lane = tid % TPR;
  let row = tid / TPR;
  let n = (wid.x + wid.z * 65535u) * TN + row;
  let m = wid.y;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;
  let UPC = TPR * 2u;                       // units per chunk (whole row)
  let nChunks = (nUnits + UPC - 1u) / UPC;
  let aRow = m * K;
  let valid = n < params.N;

  var acc: f32 = 0.0;
  for (var c = 0u; c < nChunks; c = c + 1u) {
    workgroupBarrier();
    tile_stage2(tid, aRow, c * UPC * 32u, K);
    workgroupBarrier();

    let ua = c * UPC + lane;
    let ub = ua + TPR;
    let okA = ua < nUnits && valid;
    let okB = ub < nUnits && valid;
    // ── all global loads, both units, before any decode ALU ─────────────
    let pairA = ua % 8u;
    let subA = pairA * 2u;
    let baseA = select(0u, (n * nSB + ua / 8u) * 21u, okA);
    let dmwA = W[baseA + 20u];
    let scwA = W[baseA + (subA >> 2u)];
    let wA0 = W[baseA + 4u + pairA * 2u];
    let wA1 = W[baseA + 4u + pairA * 2u + 1u];
    let pairB = ub % 8u;
    let subB = pairB * 2u;
    let baseB = select(0u, (n * nSB + ub / 8u) * 21u, okB);
    let dmwB = W[baseB + 20u];
    let scwB = W[baseB + (subB >> 2u)];
    let wB0 = W[baseB + 4u + pairB * 2u];
    let wB1 = W[baseB + 4u + pairB * 2u + 1u];
    // ── decode + accumulate (per-lane order matches _r exactly) ─────────
    let cA = q2k_unit_acc(dmwA, scwA, subA, wA0, wA1, lane * 8u);
    acc = acc + select(0.0, cA, okA);
    let cB = q2k_unit_acc(dmwB, scwB, subB, wB0, wB1, (TPR + lane) * 8u);
    acc = acc + select(0.0, cB, okB);
  }
  tile_reduce_store(tid, lane, n, m, acc);
}

// ── Q3_K tiled: TN rows/workgroup, shared activation staging ───────────
// Same superblock decode as matmul_gguf_q3_k (6-bit scale aux-shuffle, high
// bit from hmask). Pair layout as Q2_K tiled: sub0=2p/sub1=2p+1 share
// group/shift/mbit and widx (sub0>>2 == sub1>>2), qs and hmask reads are
// both 8 contiguous words. Per element: value = dl·(q3 + 4·hbit − 4) —
// q3 ≤ 3 and 4·hbit ≤ 4, so the byte-wise add never carries.

@compute @workgroup_size(TWG, 1, 1)
fn matmul_gguf_q3_k_tiled(@builtin(local_invocation_id) lid: vec3u,
                          @builtin(workgroup_id) wid: vec3u) {
  let tid = lid.x;
  let TPR = TWG / TN;
  let lane = tid % TPR;
  let row = tid / TPR;
  let n = (wid.x + wid.z * 65535u) * TN + row;
  let m = wid.y;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;
  let nChunks = (nUnits + TPR - 1u) / TPR;
  let aRow = m * K;
  let valid = n < params.N;
  let KM1 = 0x03030303u;
  let KM2 = 0x0f0f0f0fu;

  var acc: f32 = 0.0;
  for (var c = 0u; c < nChunks; c = c + 1u) {
    workgroupBarrier();
    tile_stage(tid, aRow, c * TPR * 32u, K);
    workgroupBarrier();

    let u = c * TPR + lane;
    if (u < nUnits && valid) {
      let sb = u / 8u;
      let pair = u % 8u;
      let sub0 = pair * 2u;                   // even; sub1 = sub0 + 1
      let wordBase = (n * nSB + sb) * 28u;
      let dAll = unpack2x16float(W[wordBase + 27u]).x;   // d @108

      // 6-bit signed scales for sub0/sub1 — same auxWord (shared widx).
      let s0 = W[wordBase + 24u];
      let s1 = W[wordBase + 25u];
      let tmp = W[wordBase + 26u];
      var auxWord: u32;
      let widx = sub0 >> 2u;
      if (widx == 0u) {
        auxWord = (s0 & KM2) | (((tmp >> 0u) & KM1) << 4u);
      } else if (widx == 1u) {
        auxWord = (s1 & KM2) | (((tmp >> 2u) & KM1) << 4u);
      } else if (widx == 2u) {
        auxWord = ((s0 >> 4u) & KM2) | (((tmp >> 4u) & KM1) << 4u);
      } else {
        auxWord = ((s1 >> 4u) & KM2) | (((tmp >> 6u) & KM1) << 4u);
      }
      let scB0 = (auxWord >> ((sub0 & 3u) * 8u)) & 0xFFu;
      let scB1 = (auxWord >> (((sub0 & 3u) + 1u) * 8u)) & 0xFFu;
      let dl0 = dAll * f32(i32(scB0) - 32);
      let dl1 = dAll * f32(i32(scB1) - 32);

      let group = select(0u, 1u, sub0 >= 8u);
      let j = (sub0 & 7u) >> 1u;
      let shift = j * 2u;
      let hbitpos = group * 4u + j;           // ≤ 7: stays inside each byte
      let qWord = wordBase + 8u + group * 8u; // qs @32 + group*32 bytes
      let hWord = wordBase;                   // hmask @0
      var dq0: f32 = 0.0;
      var dq1: f32 = 0.0;
      for (var w = 0u; w < 8u; w = w + 1u) {
        let q3 = (W[qWord + w] >> shift) & 0x03030303u;
        let hb = (W[hWord + w] >> hbitpos) & 0x01010101u;
        let q = q3 + (hb << 2u);              // 0..7 per byte, no carry
        let qv = vec4<f32>(
          f32(q & 0xFFu), f32((q >> 8u) & 0xFFu),
          f32((q >> 16u) & 0xFFu), f32(q >> 24u),
        ) - vec4<f32>(4.0);
        let a4 = a_tile[lane * 8u + w];
        if (w < 4u) { dq0 = dq0 + dot(a4, qv); } else { dq1 = dq1 + dot(a4, qv); }
      }
      acc = acc + dl0 * dq0 + dl1 * dq1;
    }
  }
  tile_reduce_store(tid, lane, n, m, acc);
}

// ── No-stage tiled GEMV variants (lever 4, decode/M=1 only) ─────────────
// Same lane→unit ownership and identical per-unit math as the *_tiled
// kernels, but each lane reads its activations directly from A via the A4
// vec4 alias instead of staging chunks through a_tile — removing the two
// workgroupBarriers per K-chunk (5-19 chunks/dispatch at 27B shapes). At
// M=1 the A row is 14-75 KB and cache-resident, so the TN× re-read that
// staging avoided is nearly free; at M>1 (prefill) keep the staged tiled
// kernels. The unit stride u = lane, lane+TPR, ... visits exactly the
// chunk sequence c*TPR+lane, so accumulation order is bit-identical to
// *_tiled: a_tile[lane*8+w] == A4[(aRow + u*32)/4 + w] (units never cross
// K; parity gate 3 in scripts/test-gemv-tiled.mts). Also folds in
// bit-exact scale-load micro-fixes: Q2_K sc0/sc1 and Q6_K sc0/sc1 each
// come from one word load instead of two wbyte() word re-reads.

@compute @workgroup_size(TWG, 1, 1)
fn matmul_gguf_q4_k_tiled_ns(@builtin(local_invocation_id) lid: vec3u,
                             @builtin(workgroup_id) wid: vec3u) {
  let tid = lid.x;
  let TPR = TWG / TN;
  let lane = tid % TPR;
  let row = tid / TPR;
  let n = (wid.x + wid.z * 65535u) * TN + row;
  let m = wid.y;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;
  let aRow = m * K;

  var acc: f32 = 0.0;
  if (n < params.N) {
    for (var u = lane; u < nUnits; u = u + TPR) {
      let sb = u / 8u;
      let sub = u % 8u;                       // scale/min index j
      let sbBase = (n * nSB + sb) * 36u;      // words
      let dm = unpack2x16float(W[sbBase]);    // (d, dmin)
      let sm = scale_min_k4(sub, sbBase * 4u + 4u);
      let d1 = dm.x * sm.x;
      let min1 = dm.y * sm.y;
      let qsWord = sbBase + 4u + (sub >> 1u) * 8u;
      let hi = (sub & 1u) == 1u;
      let a4Base = (aRow + u * 32u) / 4u;
      var dq: f32 = 0.0;
      var asum: f32 = 0.0;
      for (var w = 0u; w < 8u; w = w + 1u) {
        let word = W[qsWord + w];
        let nib = select(word & 0x0F0F0F0Fu, (word >> 4u) & 0x0F0F0F0Fu, hi);
        let q4 = vec4<f32>(
          f32(nib & 0xFFu), f32((nib >> 8u) & 0xFFu),
          f32((nib >> 16u) & 0xFFu), f32(nib >> 24u),
        );
        let a4 = A4[a4Base + w];
        dq = dq + dot(a4, q4);
        asum = asum + a4.x + a4.y + a4.z + a4.w;
      }
      acc = acc + (d1 * dq - min1 * asum);
    }
  }
  tile_reduce_store(tid, lane, n, m, acc);
}

@compute @workgroup_size(TWG, 1, 1)
fn matmul_gguf_q5_k_tiled_ns(@builtin(local_invocation_id) lid: vec3u,
                             @builtin(workgroup_id) wid: vec3u) {
  let tid = lid.x;
  let TPR = TWG / TN;
  let lane = tid % TPR;
  let row = tid / TPR;
  let n = (wid.x + wid.z * 65535u) * TN + row;
  let m = wid.y;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;
  let aRow = m * K;

  var acc: f32 = 0.0;
  if (n < params.N) {
    for (var u = lane; u < nUnits; u = u + TPR) {
      let sb = u / 8u;
      let sub = u % 8u;
      let sbBase = (n * nSB + sb) * 44u;      // words
      let dm = unpack2x16float(W[sbBase]);    // (d, dmin)
      let sm = scale_min_k4(sub, sbBase * 4u + 4u);
      let d1 = dm.x * sm.x;
      let min1 = dm.y * sm.y;
      let qsWord = sbBase + 12u + (sub >> 1u) * 8u;  // qs after 16B header + 32B qh
      let qhWord = sbBase + 4u;                      // qh[32]
      let hi = (sub & 1u) == 1u;
      let a4Base = (aRow + u * 32u) / 4u;
      var dq: f32 = 0.0;
      var asum: f32 = 0.0;
      for (var w = 0u; w < 8u; w = w + 1u) {
        let word = W[qsWord + w];
        let hw = W[qhWord + w];
        let nib = select(word & 0x0F0F0F0Fu, (word >> 4u) & 0x0F0F0F0Fu, hi);
        let q = nib | (((hw >> sub) & 0x01010101u) << 4u);   // 5-bit value 0..31
        let q5 = vec4<f32>(
          f32(q & 0xFFu), f32((q >> 8u) & 0xFFu),
          f32((q >> 16u) & 0xFFu), f32(q >> 24u),
        );
        let a4 = A4[a4Base + w];
        dq = dq + dot(a4, q5);
        asum = asum + a4.x + a4.y + a4.z + a4.w;
      }
      acc = acc + (d1 * dq - min1 * asum);
    }
  }
  tile_reduce_store(tid, lane, n, m, acc);
}

@compute @workgroup_size(TWG, 1, 1)
fn matmul_gguf_q6_k_tiled_ns(@builtin(local_invocation_id) lid: vec3u,
                             @builtin(workgroup_id) wid: vec3u) {
  let tid = lid.x;
  let TPR = TWG / TN;
  let lane = tid % TPR;
  let row = tid / TPR;
  let n = (wid.x + wid.z * 65535u) * TN + row;
  let m = wid.y;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;
  let aRow = m * K;

  var acc: f32 = 0.0;
  if (n < params.N) {
    for (var u = lane; u < nUnits; u = u + TPR) {
      let sb = u / 8u;
      let sub = u % 8u;
      let half = sub >> 2u;                   // 0 or 1 (which 128-elem half)
      let s = sub & 3u;                       // q1..q4 group within the half
      let sbBase = (n * nSB + sb) * 53u;      // words
      let d = unpack2x16float(W[sbBase + 52u]).x;
      // ql[l] for s∈{0,2}, ql[l+32] for s∈{1,3}; low nibble for s<2
      let qlWord = sbBase + half * 16u + select(0u, 8u, (s & 1u) == 1u);
      let qhWord = sbBase + 32u + half * 8u;
      let hShift = 2u * s;
      let lowNib = s < 2u;
      let a4Base = (aRow + u * 32u) / 4u;
      // Scale changes at l=16 (word 4): two separate sub-dots.
      var dq0: f32 = 0.0;
      var dq1: f32 = 0.0;
      for (var w = 0u; w < 8u; w = w + 1u) {
        let lw = W[qlWord + w];
        let hw = W[qhWord + w];
        let l4 = select((lw >> 4u) & 0x0F0F0F0Fu, lw & 0x0F0F0F0Fu, lowNib);
        let q = l4 | (((hw >> hShift) & 0x03030303u) << 4u);
        let qv = vec4<f32>(
          f32(q & 0xFFu), f32((q >> 8u) & 0xFFu),
          f32((q >> 16u) & 0xFFu), f32(q >> 24u),
        ) - vec4<f32>(32.0);
        let a4 = A4[a4Base + w];
        if (w < 4u) { dq0 = dq0 + dot(a4, qv); } else { dq1 = dq1 + dot(a4, qv); }
      }
      // i8 scales sc0/sc1 sit in one word: bytes 2(s&1) and 2(s&1)+1 of
      // word 48 + half*2 + s/2 (byte offset 192 + half*8 + 2s) — same
      // bytes the tiled kernel fetches via two wbyte() word re-reads.
      let scw = W[sbBase + 48u + half * 2u + (s >> 1u)];
      let sc0 = i8_byte(scw, (s & 1u) * 2u);
      let sc1 = i8_byte(scw, (s & 1u) * 2u + 1u);
      acc = acc + d * (sc0 * dq0 + sc1 * dq1);
    }
  }
  tile_reduce_store(tid, lane, n, m, acc);
}

@compute @workgroup_size(TWG, 1, 1)
fn matmul_gguf_q2_k_tiled_ns(@builtin(local_invocation_id) lid: vec3u,
                             @builtin(workgroup_id) wid: vec3u) {
  let tid = lid.x;
  let TPR = TWG / TN;
  let lane = tid % TPR;
  let row = tid / TPR;
  let n = (wid.x + wid.z * 65535u) * TN + row;
  let m = wid.y;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;
  let aRow = m * K;

  var acc: f32 = 0.0;
  if (n < params.N) {
    for (var u = lane; u < nUnits; u = u + TPR) {
      let sb = u / 8u;
      let pair = u % 8u;
      let sub0 = pair * 2u;                   // even; sub1 = sub0 + 1
      let wordBase = (n * nSB + sb) * 21u;
      let dm = unpack2x16float(W[wordBase + 20u]);   // d @80, dmin @82
      // scales[sub0] and scales[sub0+1] share one word (sub0 is even).
      let scw = W[wordBase + (sub0 >> 2u)];
      let sc0 = (scw >> ((sub0 & 3u) * 8u)) & 0xFFu;
      let sc1 = (scw >> ((sub0 & 3u) * 8u + 8u)) & 0xFFu;
      let dl0 = dm.x * f32(sc0 & 0x0Fu);
      let ml0 = dm.y * f32(sc0 >> 4u);
      let dl1 = dm.x * f32(sc1 & 0x0Fu);
      let ml1 = dm.y * f32(sc1 >> 4u);
      let group = select(0u, 1u, sub0 >= 8u);
      let shift = ((sub0 & 7u) >> 1u) * 2u;
      let qWord = wordBase + 4u + group * 8u;  // qs @16 + group*32 bytes
      let a4Base = (aRow + u * 32u) / 4u;
      var dq0: f32 = 0.0;
      var as0: f32 = 0.0;
      var dq1: f32 = 0.0;
      var as1: f32 = 0.0;
      for (var w = 0u; w < 8u; w = w + 1u) {
        let q = (W[qWord + w] >> shift) & 0x03030303u;
        let q2 = vec4<f32>(
          f32(q & 0xFFu), f32((q >> 8u) & 0xFFu),
          f32((q >> 16u) & 0xFFu), f32(q >> 24u),
        );
        let a4 = A4[a4Base + w];
        let asum = a4.x + a4.y + a4.z + a4.w;
        if (w < 4u) { dq0 = dq0 + dot(a4, q2); as0 = as0 + asum; }
        else        { dq1 = dq1 + dot(a4, q2); as1 = as1 + asum; }
      }
      acc = acc + (dl0 * dq0 - ml0 * as0) + (dl1 * dq1 - ml1 * as1);
    }
  }
  tile_reduce_store(tid, lane, n, m, acc);
}

@compute @workgroup_size(TWG, 1, 1)
fn matmul_gguf_q3_k_tiled_ns(@builtin(local_invocation_id) lid: vec3u,
                             @builtin(workgroup_id) wid: vec3u) {
  let tid = lid.x;
  let TPR = TWG / TN;
  let lane = tid % TPR;
  let row = tid / TPR;
  let n = (wid.x + wid.z * 65535u) * TN + row;
  let m = wid.y;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;
  let aRow = m * K;
  let KM1 = 0x03030303u;
  let KM2 = 0x0f0f0f0fu;

  var acc: f32 = 0.0;
  if (n < params.N) {
    for (var u = lane; u < nUnits; u = u + TPR) {
      let sb = u / 8u;
      let pair = u % 8u;
      let sub0 = pair * 2u;                   // even; sub1 = sub0 + 1
      let wordBase = (n * nSB + sb) * 28u;
      let dAll = unpack2x16float(W[wordBase + 27u]).x;   // d @108

      // 6-bit signed scales for sub0/sub1 — same auxWord (shared widx).
      let s0 = W[wordBase + 24u];
      let s1 = W[wordBase + 25u];
      let tmp = W[wordBase + 26u];
      var auxWord: u32;
      let widx = sub0 >> 2u;
      if (widx == 0u) {
        auxWord = (s0 & KM2) | (((tmp >> 0u) & KM1) << 4u);
      } else if (widx == 1u) {
        auxWord = (s1 & KM2) | (((tmp >> 2u) & KM1) << 4u);
      } else if (widx == 2u) {
        auxWord = ((s0 >> 4u) & KM2) | (((tmp >> 4u) & KM1) << 4u);
      } else {
        auxWord = ((s1 >> 4u) & KM2) | (((tmp >> 6u) & KM1) << 4u);
      }
      let scB0 = (auxWord >> ((sub0 & 3u) * 8u)) & 0xFFu;
      let scB1 = (auxWord >> (((sub0 & 3u) + 1u) * 8u)) & 0xFFu;
      let dl0 = dAll * f32(i32(scB0) - 32);
      let dl1 = dAll * f32(i32(scB1) - 32);

      let group = select(0u, 1u, sub0 >= 8u);
      let j = (sub0 & 7u) >> 1u;
      let shift = j * 2u;
      let hbitpos = group * 4u + j;           // ≤ 7: stays inside each byte
      let qWord = wordBase + 8u + group * 8u; // qs @32 + group*32 bytes
      let hWord = wordBase;                   // hmask @0
      let a4Base = (aRow + u * 32u) / 4u;
      var dq0: f32 = 0.0;
      var dq1: f32 = 0.0;
      for (var w = 0u; w < 8u; w = w + 1u) {
        let q3 = (W[qWord + w] >> shift) & 0x03030303u;
        let hb = (W[hWord + w] >> hbitpos) & 0x01010101u;
        let q = q3 + (hb << 2u);              // 0..7 per byte, no carry
        let qv = vec4<f32>(
          f32(q & 0xFFu), f32((q >> 8u) & 0xFFu),
          f32((q >> 16u) & 0xFFu), f32(q >> 24u),
        ) - vec4<f32>(4.0);
        let a4 = A4[a4Base + w];
        if (w < 4u) { dq0 = dq0 + dot(a4, qv); } else { dq1 = dq1 + dot(a4, qv); }
      }
      acc = acc + dl0 * dq0 + dl1 * dq1;
    }
  }
  tile_reduce_store(tid, lane, n, m, acc);
}

// ── vec4 W-load tiled GEMV variants (lever 4 phase 2) ───────────────────
// Identical to the staged *_tiled entries (chunk loop, tile_stage/a_tile,
// per-chunk barriers — Phase 1 proved LDS staging beats direct A reads on
// RDNA2) but every weight access is a vec4<u32> load through the W4 alias:
// fewer, wider memory transactions on the W stream that dominates decode
// bandwidth. Per-unit reads collapse 18-20 scalar u32 loads into 5 vec4
// loads. Same words, same bit extraction, same accumulation order →
// bit-identical (gate 4 in scripts/test-gemv-tiled.mts asserts both vec4
// alignment and exact decode equality). Only Q3_K/Q4_K/Q5_K qualify:
// strides 28/36/44 ≡ 0 mod 4 (Q2_K 21 and Q6_K 53 are odd — impossible
// without repack).

@compute @workgroup_size(TWG, 1, 1)
fn matmul_gguf_q4_k_tiled_v4(@builtin(local_invocation_id) lid: vec3u,
                             @builtin(workgroup_id) wid: vec3u) {
  let tid = lid.x;
  let TPR = TWG / TN;
  let lane = tid % TPR;
  let row = tid / TPR;
  let n = (wid.x + wid.z * 65535u) * TN + row;
  let m = wid.y;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;
  let nChunks = (nUnits + TPR - 1u) / TPR;
  let aRow = m * K;
  let valid = n < params.N;

  var acc: f32 = 0.0;
  for (var c = 0u; c < nChunks; c = c + 1u) {
    workgroupBarrier();                       // previous chunk's readers done
    tile_stage(tid, aRow, c * TPR * 32u, K);
    workgroupBarrier();

    let u = c * TPR + lane;
    if (u < nUnits && valid) {
      let sb = u / 8u;
      let sub = u % 8u;                       // scale/min index j
      let sbV4 = (n * nSB + sb) * 9u;         // 36 words / 4
      let hv = W4[sbV4];                      // d/dmin | scales[12]
      let dm = unpack2x16float(hv.x);         // (d, dmin)
      let sm = scale_min_k4_w(sub, hv.y, hv.z, hv.w);
      let d1 = dm.x * sm.x;
      let min1 = dm.y * sm.y;
      let qsV4 = sbV4 + 1u + (sub >> 1u) * 2u;
      var qs: array<vec4<u32>, 2>;
      qs[0] = W4[qsV4];
      qs[1] = W4[qsV4 + 1u];
      let hi = (sub & 1u) == 1u;
      var dq: f32 = 0.0;
      var asum: f32 = 0.0;
      for (var w = 0u; w < 8u; w = w + 1u) {
        let word = qs[w >> 2u][w & 3u];
        let nib = select(word & 0x0F0F0F0Fu, (word >> 4u) & 0x0F0F0F0Fu, hi);
        let q4 = vec4<f32>(
          f32(nib & 0xFFu), f32((nib >> 8u) & 0xFFu),
          f32((nib >> 16u) & 0xFFu), f32(nib >> 24u),
        );
        let a4 = a_tile[lane * 8u + w];
        dq = dq + dot(a4, q4);
        asum = asum + a4.x + a4.y + a4.z + a4.w;
      }
      acc = acc + (d1 * dq - min1 * asum);
    }
  }
  tile_reduce_store(tid, lane, n, m, acc);
}

@compute @workgroup_size(TWG, 1, 1)
fn matmul_gguf_q5_k_tiled_v4(@builtin(local_invocation_id) lid: vec3u,
                             @builtin(workgroup_id) wid: vec3u) {
  let tid = lid.x;
  let TPR = TWG / TN;
  let lane = tid % TPR;
  let row = tid / TPR;
  let n = (wid.x + wid.z * 65535u) * TN + row;
  let m = wid.y;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;
  let nChunks = (nUnits + TPR - 1u) / TPR;
  let aRow = m * K;
  let valid = n < params.N;

  var acc: f32 = 0.0;
  for (var c = 0u; c < nChunks; c = c + 1u) {
    workgroupBarrier();
    tile_stage(tid, aRow, c * TPR * 32u, K);
    workgroupBarrier();

    let u = c * TPR + lane;
    if (u < nUnits && valid) {
      let sb = u / 8u;
      let sub = u % 8u;
      let sbV4 = (n * nSB + sb) * 11u;        // 44 words / 4
      let hv = W4[sbV4];                      // d/dmin | scales[12]
      let dm = unpack2x16float(hv.x);         // (d, dmin)
      let sm = scale_min_k4_w(sub, hv.y, hv.z, hv.w);
      let d1 = dm.x * sm.x;
      let min1 = dm.y * sm.y;
      // qh[32] = words 4..11 (vec4s +1, +2); qs after 16B header + 32B qh
      var qh: array<vec4<u32>, 2>;
      qh[0] = W4[sbV4 + 1u];
      qh[1] = W4[sbV4 + 2u];
      let qsV4 = sbV4 + 3u + (sub >> 1u) * 2u;
      var qs: array<vec4<u32>, 2>;
      qs[0] = W4[qsV4];
      qs[1] = W4[qsV4 + 1u];
      let hi = (sub & 1u) == 1u;
      var dq: f32 = 0.0;
      var asum: f32 = 0.0;
      for (var w = 0u; w < 8u; w = w + 1u) {
        let word = qs[w >> 2u][w & 3u];
        let hw = qh[w >> 2u][w & 3u];
        let nib = select(word & 0x0F0F0F0Fu, (word >> 4u) & 0x0F0F0F0Fu, hi);
        let q = nib | (((hw >> sub) & 0x01010101u) << 4u);   // 5-bit value 0..31
        let q5 = vec4<f32>(
          f32(q & 0xFFu), f32((q >> 8u) & 0xFFu),
          f32((q >> 16u) & 0xFFu), f32(q >> 24u),
        );
        let a4 = a_tile[lane * 8u + w];
        dq = dq + dot(a4, q5);
        asum = asum + a4.x + a4.y + a4.z + a4.w;
      }
      acc = acc + (d1 * dq - min1 * asum);
    }
  }
  tile_reduce_store(tid, lane, n, m, acc);
}

@compute @workgroup_size(TWG, 1, 1)
fn matmul_gguf_q3_k_tiled_v4(@builtin(local_invocation_id) lid: vec3u,
                             @builtin(workgroup_id) wid: vec3u) {
  let tid = lid.x;
  let TPR = TWG / TN;
  let lane = tid % TPR;
  let row = tid / TPR;
  let n = (wid.x + wid.z * 65535u) * TN + row;
  let m = wid.y;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 8u;
  let nChunks = (nUnits + TPR - 1u) / TPR;
  let aRow = m * K;
  let valid = n < params.N;
  let KM1 = 0x03030303u;
  let KM2 = 0x0f0f0f0fu;

  var acc: f32 = 0.0;
  for (var c = 0u; c < nChunks; c = c + 1u) {
    workgroupBarrier();
    tile_stage(tid, aRow, c * TPR * 32u, K);
    workgroupBarrier();

    let u = c * TPR + lane;
    if (u < nUnits && valid) {
      let sb = u / 8u;
      let pair = u % 8u;
      let sub0 = pair * 2u;                   // even; sub1 = sub0 + 1
      let sbV4 = (n * nSB + sb) * 7u;         // 28 words / 4
      // scales[12] + d in one vec4 (words 24..27)
      let sv = W4[sbV4 + 6u];
      let dAll = unpack2x16float(sv.w).x;     // d @108

      // 6-bit signed scales for sub0/sub1 — same auxWord (shared widx).
      let s0 = sv.x;
      let s1 = sv.y;
      let tmp = sv.z;
      var auxWord: u32;
      let widx = sub0 >> 2u;
      if (widx == 0u) {
        auxWord = (s0 & KM2) | (((tmp >> 0u) & KM1) << 4u);
      } else if (widx == 1u) {
        auxWord = (s1 & KM2) | (((tmp >> 2u) & KM1) << 4u);
      } else if (widx == 2u) {
        auxWord = ((s0 >> 4u) & KM2) | (((tmp >> 4u) & KM1) << 4u);
      } else {
        auxWord = ((s1 >> 4u) & KM2) | (((tmp >> 6u) & KM1) << 4u);
      }
      let scB0 = (auxWord >> ((sub0 & 3u) * 8u)) & 0xFFu;
      let scB1 = (auxWord >> (((sub0 & 3u) + 1u) * 8u)) & 0xFFu;
      let dl0 = dAll * f32(i32(scB0) - 32);
      let dl1 = dAll * f32(i32(scB1) - 32);

      let group = select(0u, 1u, sub0 >= 8u);
      let j = (sub0 & 7u) >> 1u;
      let shift = j * 2u;
      let hbitpos = group * 4u + j;           // ≤ 7: stays inside each byte
      // qs @32 + group*32 bytes (vec4s +2/+3 or +4/+5); hmask @0 (vec4s 0/1)
      let qsV4 = sbV4 + 2u + group * 2u;
      var qs: array<vec4<u32>, 2>;
      qs[0] = W4[qsV4];
      qs[1] = W4[qsV4 + 1u];
      var hm: array<vec4<u32>, 2>;
      hm[0] = W4[sbV4];
      hm[1] = W4[sbV4 + 1u];
      var dq0: f32 = 0.0;
      var dq1: f32 = 0.0;
      for (var w = 0u; w < 8u; w = w + 1u) {
        let q3 = (qs[w >> 2u][w & 3u] >> shift) & 0x03030303u;
        let hb = (hm[w >> 2u][w & 3u] >> hbitpos) & 0x01010101u;
        let q = q3 + (hb << 2u);              // 0..7 per byte, no carry
        let qv = vec4<f32>(
          f32(q & 0xFFu), f32((q >> 8u) & 0xFFu),
          f32((q >> 16u) & 0xFFu), f32(q >> 24u),
        ) - vec4<f32>(4.0);
        let a4 = a_tile[lane * 8u + w];
        if (w < 4u) { dq0 = dq0 + dot(a4, qv); } else { dq1 = dq1 + dot(a4, qv); }
      }
      acc = acc + dl0 * dq0 + dl1 * dq1;
    }
  }
  tile_reduce_store(tid, lane, n, m, acc);
}
