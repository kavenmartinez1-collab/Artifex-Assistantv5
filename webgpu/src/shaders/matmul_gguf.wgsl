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

// ── IQ2_XXS tiled: TN rows/workgroup, shared activation staging ────────
// Same superblock decode as matmul_gguf_iq2_xxs above (17 u32/SB, 8 ib32
// groups of 8 bytes = aux0[4] grid indices | aux1 with 4× 7-bit signs +
// 4-bit ls scale at the top). One 32-elem tile unit = one ib32 group.
//
// Lever C5 (2026-07-01) — same recipe that took Q2_K to _r2-level issue
// rates, value-preserving so results stay bit-identical to the previous
// tiled body (which the ?gemvTile=0 legacy A/B already covered):
//  • The unit's 8 aux bytes sit at byte 2+8·ib of the 68 B block — always
//    2 mod 4. Instead of 8 wbyte() byte-extract loads, issue 3 aligned u32
//    loads (w0..w2, hoisted before any decode ALU like q2_k_tiled_r2) and
//    reassemble with a 16-bit skew: aux0=(w0>>16)|(w1<<16), aux1=(w1>>16)|
//    (w2<<16). In-bounds: ib=7 reads word 16, the last word of the block.
//  • Grid magnitudes: 2 table words + unpack4xU8 (exact u8 → f32) instead
//    of 8 per-byte dynamic IQ2XXS_GRID indexings per l-group.
//  • Signs: flip the f32 sign bit with a vector XOR (bit-identical to
//    mag·(±1) for all finite values incl. ±0) instead of 8 selects + muls.

@compute @workgroup_size(TWG, 1, 1)
fn matmul_gguf_iq2_xxs_tiled(@builtin(local_invocation_id) lid: vec3u,
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

  let UPC = TPR * 2u;                          // units per chunk (r2 pattern)
  let nChunks2 = (nUnits + UPC - 1u) / UPC;

  var acc: f32 = 0.0;
  for (var c = 0u; c < nChunks2; c = c + 1u) {
    workgroupBarrier();
    tile_stage2(tid, aRow, c * UPC * 32u, K);
    workgroupBarrier();

    // Two units per lane per chunk: ua then ub = ua + TPR. Per-lane visit
    // order (lane, TPR+lane, 2·TPR+lane, …) is IDENTICAL to the single-unit
    // body, so accumulation stays bit-identical.
    let ua = c * UPC + lane;
    let ub = ua + TPR;
    let okA = ua < nUnits && valid;
    let okB = ub < nUnits && valid;
    // ── all global loads, both units, before any decode ALU ─────────────
    let bA = select(0u, (n * nSB + ua / 8u) * 17u, okA);   // block word 0
    let qA = bA + (ua % 8u) * 2u;              // unit's aux bytes @ qA, byte 2
    let dwA = W[bA];                           // f16 d in low half
    let wA0 = W[qA];
    let wA1 = W[qA + 1u];
    let wA2 = W[qA + 2u];
    let bB = select(0u, (n * nSB + ub / 8u) * 17u, okB);
    let qB = bB + (ub % 8u) * 2u;
    let dwB = W[bB];
    let wB0 = W[qB];
    let wB1 = W[qB + 1u];
    let wB2 = W[qB + 2u];
    // ── decode + accumulate ──────────────────────────────────────────────
    let cA = iq2xxs_unit_acc(dwA, wA0, wA1, wA2, lane * 8u);
    acc = acc + select(0.0, cA, okA);
    let cB = iq2xxs_unit_acc(dwB, wB0, wB1, wB2, (TPR + lane) * 8u);
    acc = acc + select(0.0, cB, okB);
  }
  tile_reduce_store(tid, lane, n, m, acc);
}

/** One IQ2_XXS unit's acc contribution, fed from pre-loaded words.
 *  dw = block word 0 (f16 d in low half); w0..w2 = the three words holding
 *  the unit's 8 aux bytes at 16-bit skew; atBase = a_tile2 vec4 index. */
fn iq2xxs_unit_acc(dw: u32, w0: u32, w1: u32, w2: u32, atBase: u32) -> f32 {
  let d = unpack2x16float(dw).x;
  let aux0 = (w0 >> 16u) | (w1 << 16u);        // 4 grid indices
  let aux1 = (w1 >> 16u) | (w2 << 16u);        // 4× 7-bit sign idx + ls nibble
  let ls = aux1 >> 28u;
  let db = d * (0.5 + f32(ls)) * 0.25;
  // Single accumulation chain, in legacy visit order — bit-identical to the
  // legacy kernel. A lo/hi split (2 independent chains) benched flat
  // (183.8 vs 181.6 us, 2026-07-01), so keep the simpler exact-order form.
  var dot_acc: f32 = 0.0;
  for (var l = 0u; l < 4u; l = l + 1u) {
    let gridIdx = (aux0 >> (l * 8u)) & 0xFFu;
    let signs = iq2xxs_sign_byte((aux1 >> (7u * l)) & 127u);
    let mlo = vec4<f32>(unpack4xU8(IQ2XXS_GRID[gridIdx * 2u]));
    let mhi = vec4<f32>(unpack4xU8(IQ2XXS_GRID[gridIdx * 2u + 1u]));
    let slo = ((vec4<u32>(signs) >> vec4<u32>(0u, 1u, 2u, 3u)) & vec4<u32>(1u)) << vec4<u32>(31u);
    let shi = ((vec4<u32>(signs) >> vec4<u32>(4u, 5u, 6u, 7u)) & vec4<u32>(1u)) << vec4<u32>(31u);
    let alo = a_tile2[atBase + l * 2u];
    let ahi = a_tile2[atBase + l * 2u + 1u];
    dot_acc = dot_acc + dot(alo, bitcast<vec4<f32>>(bitcast<vec4<u32>>(mlo) ^ slo));
    dot_acc = dot_acc + dot(ahi, bitcast<vec4<f32>>(bitcast<vec4<u32>>(mhi) ^ shi));
  }
  return db * dot_acc;
}

// ── IQ2_XXS tiled GEMM (M-reuse, mirrors q2_k_tiled_r_gemm) ─────────────
// The spec-decode verify runs M rows (pending + drafts) through the same
// weights; the GEMV path put M in grid.y so every row re-read and re-decoded
// every weight — an M-row verify cost ~M single forwards, which is exactly
// why MTP Opt-1/Opt-2 lost (falsified hypotheses 1-2, ENGINEERING_LOG.md).
// Here each lane loads + decodes its unit ONCE (C5 recipe: aligned u32 loads,
// unpack4xU8 grid decode, XOR sign flip) and dots it against all M staged
// activation rows. Grid.y = 1. Per-lane unit visit order (lane, TPR+lane, …)
// and the per-unit lo/hi accumulation order match the r2 tiled kernel, so
// the M=1 column is bit-identical to matmul_gguf_iq2_xxs_tiled.

@compute @workgroup_size(TWG, 1, 1)
fn matmul_gguf_iq2_xxs_gemm(@builtin(local_invocation_id) lid: vec3u,
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
      // All global loads before decode ALU (same as the r2 tiled body).
      let b = (n * nSB + u / 8u) * 17u;        // block word 0 (f16 d)
      let q = b + (u % 8u) * 2u;               // unit's aux bytes @ byte 2
      let dw = W[b];
      let w0 = W[q];
      let w1 = W[q + 1u];
      let w2 = W[q + 2u];
      let d = unpack2x16float(dw).x;
      let aux0 = (w0 >> 16u) | (w1 << 16u);    // 4 grid indices
      let aux1 = (w1 >> 16u) | (w2 << 16u);    // 4× 7-bit sign idx + ls nibble
      let ls = aux1 >> 28u;
      let db = d * (0.5 + f32(ls)) * 0.25;
      var usum: array<f32, MAXM>;
      for (var mi = 0u; mi < M; mi = mi + 1u) { usum[mi] = 0.0; }
      for (var l = 0u; l < 4u; l = l + 1u) {
        let gridIdx = (aux0 >> (l * 8u)) & 0xFFu;
        let signs = iq2xxs_sign_byte((aux1 >> (7u * l)) & 127u);
        let mlo = vec4<f32>(unpack4xU8(IQ2XXS_GRID[gridIdx * 2u]));
        let mhi = vec4<f32>(unpack4xU8(IQ2XXS_GRID[gridIdx * 2u + 1u]));
        let slo = ((vec4<u32>(signs) >> vec4<u32>(0u, 1u, 2u, 3u)) & vec4<u32>(1u)) << vec4<u32>(31u);
        let shi = ((vec4<u32>(signs) >> vec4<u32>(4u, 5u, 6u, 7u)) & vec4<u32>(1u)) << vec4<u32>(31u);
        let wlo = bitcast<vec4<f32>>(bitcast<vec4<u32>>(mlo) ^ slo);
        let whi = bitcast<vec4<f32>>(bitcast<vec4<u32>>(mhi) ^ shi);
        for (var mi = 0u; mi < M; mi = mi + 1u) {
          let at = mi * nVec + lane * 8u + l * 2u;
          usum[mi] = usum[mi] + dot(a_tile_m[at], wlo);
          usum[mi] = usum[mi] + dot(a_tile_m[at + 1u], whi);
        }
      }
      for (var mi = 0u; mi < M; mi = mi + 1u) {
        acc[mi] = acc[mi] + db * usum[mi];
      }
    }
  }
  for (var mi = 0u; mi < M; mi = mi + 1u) {
    tile_reduce_store(tid, lane, n, mi, acc[mi]);
  }
}

// ── IQ3_XXS / IQ3_S / IQ2_S grids ───────────────────────────────────────
// iq3xxs_grid[256] / iq3s_grid[512] (u32, 4 magnitude bytes each) and
// iq2s_grid[1024] (u64 → 2 u32 lo/hi, 8 magnitude bytes each). Generated
// bit-exact by scripts/gen-iq3-iq2s-tables.mjs (INJECT=1). Sign bits reuse
// the IQ2XXS_SIGNS (ksigns_iq2xs) table above for IQ3_XXS; IQ3_S/IQ2_S
// carry inline sign bytes. Bit math mirrors dequantize_row_iq3_xxs /
// _iq3_s / _iq2_s (gguf-dequant.ts dequantIQ3_XXS / IQ3_S / IQ2_S).
// <iq3-iq2s-tables-wgsl>
const IQ3XXS_GRID = array<u32, 256>(
  0x04040404u, 0x04040414u, 0x04040424u, 0x04040c0cu, 0x04040c1cu, 0x04040c3eu, 0x04041404u, 0x04041414u,
  0x04041c0cu, 0x04042414u, 0x04043e1cu, 0x04043e2cu, 0x040c040cu, 0x040c041cu, 0x040c0c04u, 0x040c0c14u,
  0x040c140cu, 0x040c142cu, 0x040c1c04u, 0x040c1c14u, 0x040c240cu, 0x040c2c24u, 0x040c3e04u, 0x04140404u,
  0x04140414u, 0x04140424u, 0x04140c0cu, 0x04141404u, 0x04141414u, 0x04141c0cu, 0x04141c1cu, 0x04141c3eu,
  0x04142c0cu, 0x04142c3eu, 0x04143e2cu, 0x041c040cu, 0x041c043eu, 0x041c0c04u, 0x041c0c14u, 0x041c142cu,
  0x041c3e04u, 0x04240c1cu, 0x04241c3eu, 0x04242424u, 0x04242c3eu, 0x04243e1cu, 0x04243e2cu, 0x042c040cu,
  0x042c043eu, 0x042c1c14u, 0x042c2c14u, 0x04341c2cu, 0x04343424u, 0x043e0c04u, 0x043e0c24u, 0x043e0c34u,
  0x043e241cu, 0x043e340cu, 0x0c04040cu, 0x0c04041cu, 0x0c040c04u, 0x0c040c14u, 0x0c04140cu, 0x0c04141cu,
  0x0c041c04u, 0x0c041c14u, 0x0c041c24u, 0x0c04243eu, 0x0c042c04u, 0x0c0c0404u, 0x0c0c0414u, 0x0c0c0c0cu,
  0x0c0c1404u, 0x0c0c1414u, 0x0c14040cu, 0x0c14041cu, 0x0c140c04u, 0x0c140c14u, 0x0c14140cu, 0x0c141c04u,
  0x0c143e14u, 0x0c1c0404u, 0x0c1c0414u, 0x0c1c1404u, 0x0c1c1c0cu, 0x0c1c2434u, 0x0c1c3434u, 0x0c24040cu,
  0x0c24042cu, 0x0c242c04u, 0x0c2c1404u, 0x0c2c1424u, 0x0c2c2434u, 0x0c2c3e0cu, 0x0c34042cu, 0x0c3e1414u,
  0x0c3e2404u, 0x14040404u, 0x14040414u, 0x14040c0cu, 0x14040c1cu, 0x14041404u, 0x14041414u, 0x14041434u,
  0x14041c0cu, 0x14042414u, 0x140c040cu, 0x140c041cu, 0x140c042cu, 0x140c0c04u, 0x140c0c14u, 0x140c140cu,
  0x140c1c04u, 0x140c341cu, 0x140c343eu, 0x140c3e04u, 0x14140404u, 0x14140414u, 0x14140c0cu, 0x14140c3eu,
  0x14141404u, 0x14141414u, 0x14141c3eu, 0x14142404u, 0x14142c2cu, 0x141c040cu, 0x141c0c04u, 0x141c0c24u,
  0x141c3e04u, 0x141c3e24u, 0x14241c2cu, 0x14242c1cu, 0x142c041cu, 0x142c143eu, 0x142c240cu, 0x142c3e24u,
  0x143e040cu, 0x143e041cu, 0x143e0c34u, 0x143e242cu, 0x1c04040cu, 0x1c040c04u, 0x1c040c14u, 0x1c04140cu,
  0x1c04141cu, 0x1c042c04u, 0x1c04342cu, 0x1c043e14u, 0x1c0c0404u, 0x1c0c0414u, 0x1c0c1404u, 0x1c0c1c0cu,
  0x1c0c2424u, 0x1c0c2434u, 0x1c14040cu, 0x1c14041cu, 0x1c140c04u, 0x1c14142cu, 0x1c142c14u, 0x1c143e14u,
  0x1c1c0c0cu, 0x1c1c1c1cu, 0x1c241c04u, 0x1c24243eu, 0x1c243e14u, 0x1c2c0404u, 0x1c2c0434u, 0x1c2c1414u,
  0x1c2c2c2cu, 0x1c340c24u, 0x1c341c34u, 0x1c34341cu, 0x1c3e1c1cu, 0x1c3e3404u, 0x24040424u, 0x24040c3eu,
  0x24041c2cu, 0x24041c3eu, 0x24042c1cu, 0x24042c3eu, 0x240c3e24u, 0x24141404u, 0x24141c3eu, 0x24142404u,
  0x24143404u, 0x24143434u, 0x241c043eu, 0x241c242cu, 0x24240424u, 0x24242c0cu, 0x24243424u, 0x242c142cu,
  0x242c241cu, 0x242c3e04u, 0x243e042cu, 0x243e0c04u, 0x243e0c14u, 0x243e1c04u, 0x2c040c14u, 0x2c04240cu,
  0x2c043e04u, 0x2c0c0404u, 0x2c0c0434u, 0x2c0c1434u, 0x2c0c2c2cu, 0x2c140c24u, 0x2c141c14u, 0x2c143e14u,
  0x2c1c0414u, 0x2c1c2c1cu, 0x2c240c04u, 0x2c24141cu, 0x2c24143eu, 0x2c243e14u, 0x2c2c0414u, 0x2c2c1c0cu,
  0x2c342c04u, 0x2c3e1424u, 0x2c3e2414u, 0x34041424u, 0x34042424u, 0x34042434u, 0x34043424u, 0x340c140cu,
  0x340c340cu, 0x34140c3eu, 0x34143424u, 0x341c1c04u, 0x341c1c34u, 0x34242424u, 0x342c042cu, 0x342c2c14u,
  0x34341c1cu, 0x343e041cu, 0x343e140cu, 0x3e04041cu, 0x3e04042cu, 0x3e04043eu, 0x3e040c04u, 0x3e041c14u,
  0x3e042c14u, 0x3e0c1434u, 0x3e0c2404u, 0x3e140c14u, 0x3e14242cu, 0x3e142c14u, 0x3e1c0404u, 0x3e1c0c2cu,
  0x3e1c1c1cu, 0x3e1c3404u, 0x3e24140cu, 0x3e24240cu, 0x3e2c0404u, 0x3e2c0414u, 0x3e2c1424u, 0x3e341c04u,
);
const IQ3S_GRID = array<u32, 512>(
  0x01010101u, 0x01010103u, 0x01010105u, 0x0101010bu, 0x0101010fu, 0x01010301u, 0x01010303u, 0x01010305u,
  0x01010309u, 0x0101030du, 0x01010501u, 0x01010503u, 0x0101050bu, 0x01010707u, 0x01010901u, 0x01010905u,
  0x0101090bu, 0x0101090fu, 0x01010b03u, 0x01010b07u, 0x01010d01u, 0x01010d05u, 0x01010f03u, 0x01010f09u,
  0x01010f0fu, 0x01030101u, 0x01030103u, 0x01030105u, 0x01030109u, 0x01030301u, 0x01030303u, 0x0103030bu,
  0x01030501u, 0x01030507u, 0x0103050fu, 0x01030703u, 0x0103070bu, 0x01030909u, 0x01030d03u, 0x01030d0bu,
  0x01030f05u, 0x01050101u, 0x01050103u, 0x0105010bu, 0x0105010fu, 0x01050301u, 0x01050307u, 0x0105030du,
  0x01050503u, 0x0105050bu, 0x01050701u, 0x01050709u, 0x01050905u, 0x0105090bu, 0x0105090fu, 0x01050b03u,
  0x01050b07u, 0x01050f01u, 0x01050f07u, 0x01070107u, 0x01070303u, 0x0107030bu, 0x01070501u, 0x01070505u,
  0x01070703u, 0x01070707u, 0x0107070du, 0x01070909u, 0x01070b01u, 0x01070b05u, 0x01070d0fu, 0x01070f03u,
  0x01070f0bu, 0x01090101u, 0x01090307u, 0x0109030fu, 0x01090503u, 0x01090509u, 0x01090705u, 0x01090901u,
  0x01090907u, 0x01090b03u, 0x01090f01u, 0x010b0105u, 0x010b0109u, 0x010b0501u, 0x010b0505u, 0x010b050du,
  0x010b0707u, 0x010b0903u, 0x010b090bu, 0x010b090fu, 0x010b0d0du, 0x010b0f07u, 0x010d010du, 0x010d0303u,
  0x010d0307u, 0x010d0703u, 0x010d0b05u, 0x010d0f03u, 0x010f0101u, 0x010f0105u, 0x010f0109u, 0x010f0501u,
  0x010f0505u, 0x010f050du, 0x010f0707u, 0x010f0b01u, 0x010f0b09u, 0x03010101u, 0x03010103u, 0x03010105u,
  0x03010109u, 0x03010301u, 0x03010303u, 0x03010307u, 0x0301030bu, 0x0301030fu, 0x03010501u, 0x03010505u,
  0x03010703u, 0x03010709u, 0x0301070du, 0x03010b09u, 0x03010b0du, 0x03010d03u, 0x03010f05u, 0x03030101u,
  0x03030103u, 0x03030107u, 0x0303010du, 0x03030301u, 0x03030309u, 0x03030503u, 0x03030701u, 0x03030707u,
  0x03030903u, 0x03030b01u, 0x03030b05u, 0x03030f01u, 0x03030f0du, 0x03050101u, 0x03050305u, 0x0305030bu,
  0x0305030fu, 0x03050501u, 0x03050509u, 0x03050705u, 0x03050901u, 0x03050907u, 0x03050b0bu, 0x03050d01u,
  0x03050f05u, 0x03070103u, 0x03070109u, 0x0307010fu, 0x03070301u, 0x03070307u, 0x03070503u, 0x0307050fu,
  0x03070701u, 0x03070709u, 0x03070903u, 0x03070d05u, 0x03070f01u, 0x03090107u, 0x0309010bu, 0x03090305u,
  0x03090309u, 0x03090703u, 0x03090707u, 0x03090905u, 0x0309090du, 0x03090b01u, 0x03090b09u, 0x030b0103u,
  0x030b0301u, 0x030b0307u, 0x030b0503u, 0x030b0701u, 0x030b0705u, 0x030b0b03u, 0x030d0501u, 0x030d0509u,
  0x030d050fu, 0x030d0909u, 0x030d090du, 0x030f0103u, 0x030f0107u, 0x030f0301u, 0x030f0305u, 0x030f0503u,
  0x030f070bu, 0x030f0903u, 0x030f0d05u, 0x030f0f01u, 0x05010101u, 0x05010103u, 0x05010107u, 0x0501010bu,
  0x0501010fu, 0x05010301u, 0x05010305u, 0x05010309u, 0x0501030du, 0x05010503u, 0x05010507u, 0x0501050fu,
  0x05010701u, 0x05010705u, 0x05010903u, 0x05010907u, 0x0501090bu, 0x05010b01u, 0x05010b05u, 0x05010d0fu,
  0x05010f01u, 0x05010f07u, 0x05010f0bu, 0x05030101u, 0x05030105u, 0x05030301u, 0x05030307u, 0x0503030fu,
  0x05030505u, 0x0503050bu, 0x05030703u, 0x05030709u, 0x05030905u, 0x05030b03u, 0x05050103u, 0x05050109u,
  0x0505010fu, 0x05050503u, 0x05050507u, 0x05050701u, 0x0505070fu, 0x05050903u, 0x05050b07u, 0x05050b0fu,
  0x05050f03u, 0x05050f09u, 0x05070101u, 0x05070105u, 0x0507010bu, 0x05070303u, 0x05070505u, 0x05070509u,
  0x05070703u, 0x05070707u, 0x05070905u, 0x05070b01u, 0x05070d0du, 0x05090103u, 0x0509010fu, 0x05090501u,
  0x05090507u, 0x05090705u, 0x0509070bu, 0x05090903u, 0x05090f05u, 0x05090f0bu, 0x050b0109u, 0x050b0303u,
  0x050b0505u, 0x050b070fu, 0x050b0901u, 0x050b0b07u, 0x050b0f01u, 0x050d0101u, 0x050d0105u, 0x050d010fu,
  0x050d0503u, 0x050d0b0bu, 0x050d0d03u, 0x050f010bu, 0x050f0303u, 0x050f050du, 0x050f0701u, 0x050f0907u,
  0x050f0b01u, 0x07010105u, 0x07010303u, 0x07010307u, 0x0701030bu, 0x0701030fu, 0x07010505u, 0x07010703u,
  0x07010707u, 0x0701070bu, 0x07010905u, 0x07010909u, 0x0701090fu, 0x07010b03u, 0x07010d07u, 0x07010f03u,
  0x07030103u, 0x07030107u, 0x0703010bu, 0x07030309u, 0x07030503u, 0x07030507u, 0x07030901u, 0x07030d01u,
  0x07030f05u, 0x07030f0du, 0x07050101u, 0x07050305u, 0x07050501u, 0x07050705u, 0x07050709u, 0x07050b01u,
  0x07070103u, 0x07070301u, 0x07070309u, 0x07070503u, 0x07070507u, 0x0707050fu, 0x07070701u, 0x07070903u,
  0x07070907u, 0x0707090fu, 0x07070b0bu, 0x07070f07u, 0x07090107u, 0x07090303u, 0x0709030du, 0x07090505u,
  0x07090703u, 0x07090b05u, 0x07090d01u, 0x07090d09u, 0x070b0103u, 0x070b0301u, 0x070b0305u, 0x070b050bu,
  0x070b0705u, 0x070b0909u, 0x070b0b0du, 0x070b0f07u, 0x070d030du, 0x070d0903u, 0x070f0103u, 0x070f0107u,
  0x070f0501u, 0x070f0505u, 0x070f070bu, 0x09010101u, 0x09010109u, 0x09010305u, 0x09010501u, 0x09010509u,
  0x0901050fu, 0x09010705u, 0x09010903u, 0x09010b01u, 0x09010f01u, 0x09030105u, 0x0903010fu, 0x09030303u,
  0x09030307u, 0x09030505u, 0x09030701u, 0x0903070bu, 0x09030907u, 0x09030b03u, 0x09030b0bu, 0x09050103u,
  0x09050107u, 0x09050301u, 0x0905030bu, 0x09050503u, 0x09050707u, 0x09050901u, 0x09050b0fu, 0x09050d05u,
  0x09050f01u, 0x09070109u, 0x09070303u, 0x09070307u, 0x09070501u, 0x09070505u, 0x09070703u, 0x0907070bu,
  0x09090101u, 0x09090105u, 0x09090509u, 0x0909070fu, 0x09090901u, 0x09090f03u, 0x090b010bu, 0x090b010fu,
  0x090b0503u, 0x090b0d05u, 0x090d0307u, 0x090d0709u, 0x090d0d01u, 0x090f0301u, 0x090f030bu, 0x090f0701u,
  0x090f0907u, 0x090f0b03u, 0x0b010105u, 0x0b010301u, 0x0b010309u, 0x0b010505u, 0x0b010901u, 0x0b010909u,
  0x0b01090fu, 0x0b010b05u, 0x0b010d0du, 0x0b010f09u, 0x0b030103u, 0x0b030107u, 0x0b03010bu, 0x0b030305u,
  0x0b030503u, 0x0b030705u, 0x0b030f05u, 0x0b050101u, 0x0b050303u, 0x0b050507u, 0x0b050701u, 0x0b05070du,
  0x0b050b07u, 0x0b070105u, 0x0b07010fu, 0x0b070301u, 0x0b07050fu, 0x0b070909u, 0x0b070b03u, 0x0b070d0bu,
  0x0b070f07u, 0x0b090103u, 0x0b090109u, 0x0b090501u, 0x0b090705u, 0x0b09090du, 0x0b0b0305u, 0x0b0b050du,
  0x0b0b0b03u, 0x0b0b0b07u, 0x0b0d0905u, 0x0b0f0105u, 0x0b0f0109u, 0x0b0f0505u, 0x0d010303u, 0x0d010307u,
  0x0d01030bu, 0x0d010703u, 0x0d010707u, 0x0d010d01u, 0x0d030101u, 0x0d030501u, 0x0d03050fu, 0x0d030d09u,
  0x0d050305u, 0x0d050709u, 0x0d050905u, 0x0d050b0bu, 0x0d050d05u, 0x0d050f01u, 0x0d070101u, 0x0d070309u,
  0x0d070503u, 0x0d070901u, 0x0d09050bu, 0x0d090907u, 0x0d090d05u, 0x0d0b0101u, 0x0d0b0107u, 0x0d0b0709u,
  0x0d0b0d01u, 0x0d0d010bu, 0x0d0d0901u, 0x0d0f0303u, 0x0d0f0307u, 0x0f010101u, 0x0f010109u, 0x0f01010fu,
  0x0f010501u, 0x0f010505u, 0x0f01070du, 0x0f010901u, 0x0f010b09u, 0x0f010d05u, 0x0f030105u, 0x0f030303u,
  0x0f030509u, 0x0f030907u, 0x0f03090bu, 0x0f050103u, 0x0f050109u, 0x0f050301u, 0x0f05030du, 0x0f050503u,
  0x0f050701u, 0x0f050b03u, 0x0f070105u, 0x0f070705u, 0x0f07070bu, 0x0f070b07u, 0x0f090103u, 0x0f09010bu,
  0x0f090307u, 0x0f090501u, 0x0f090b01u, 0x0f0b0505u, 0x0f0b0905u, 0x0f0d0105u, 0x0f0d0703u, 0x0f0f0101u,
);
const IQ2S_GRID = array<u32, 2048>(
  0x08080808u, 0x08080808u, 0x0808082bu, 0x08080808u, 0x08081919u, 0x08080808u, 0x08082b08u, 0x08080808u,
  0x08082b2bu, 0x08080808u, 0x08190819u, 0x08080808u, 0x08191908u, 0x08080808u, 0x0819192bu, 0x08080808u,
  0x08192b19u, 0x08080808u, 0x082b0808u, 0x08080808u, 0x082b082bu, 0x08080808u, 0x082b1919u, 0x08080808u,
  0x082b2b08u, 0x08080808u, 0x19080819u, 0x08080808u, 0x19081908u, 0x08080808u, 0x1908192bu, 0x08080808u,
  0x19082b19u, 0x08080808u, 0x19190808u, 0x08080808u, 0x1919082bu, 0x08080808u, 0x19191919u, 0x08080808u,
  0x19192b08u, 0x08080808u, 0x192b0819u, 0x08080808u, 0x192b1908u, 0x08080808u, 0x192b192bu, 0x08080808u,
  0x192b2b19u, 0x08080808u, 0x2b080808u, 0x08080808u, 0x2b08082bu, 0x08080808u, 0x2b081919u, 0x08080808u,
  0x2b082b08u, 0x08080808u, 0x2b190819u, 0x08080808u, 0x2b191908u, 0x08080808u, 0x2b2b0808u, 0x08080808u,
  0x2b2b1919u, 0x08080808u, 0x2b2b2b2bu, 0x08080808u, 0x08080819u, 0x08080819u, 0x08081908u, 0x08080819u,
  0x0808192bu, 0x08080819u, 0x08082b19u, 0x08080819u, 0x08190808u, 0x08080819u, 0x0819082bu, 0x08080819u,
  0x08191919u, 0x08080819u, 0x08192b08u, 0x08080819u, 0x082b0819u, 0x08080819u, 0x082b1908u, 0x08080819u,
  0x19080808u, 0x08080819u, 0x1908082bu, 0x08080819u, 0x19081919u, 0x08080819u, 0x19082b08u, 0x08080819u,
  0x19190819u, 0x08080819u, 0x19191908u, 0x08080819u, 0x1919192bu, 0x08080819u, 0x19192b19u, 0x08080819u,
  0x192b0808u, 0x08080819u, 0x192b1919u, 0x08080819u, 0x192b2b08u, 0x08080819u, 0x2b080819u, 0x08080819u,
  0x2b081908u, 0x08080819u, 0x2b190808u, 0x08080819u, 0x2b19082bu, 0x08080819u, 0x2b191919u, 0x08080819u,
  0x2b2b0819u, 0x08080819u, 0x2b2b1908u, 0x08080819u, 0x08080808u, 0x0808082bu, 0x0808082bu, 0x0808082bu,
  0x08081919u, 0x0808082bu, 0x08082b08u, 0x0808082bu, 0x08190819u, 0x0808082bu, 0x08191908u, 0x0808082bu,
  0x082b0808u, 0x0808082bu, 0x082b2b2bu, 0x0808082bu, 0x19080819u, 0x0808082bu, 0x19081908u, 0x0808082bu,
  0x1908192bu, 0x0808082bu, 0x19082b19u, 0x0808082bu, 0x19190808u, 0x0808082bu, 0x19191919u, 0x0808082bu,
  0x2b080808u, 0x0808082bu, 0x2b081919u, 0x0808082bu, 0x2b082b2bu, 0x0808082bu, 0x2b191908u, 0x0808082bu,
  0x2b2b082bu, 0x0808082bu, 0x08080819u, 0x08081908u, 0x08081908u, 0x08081908u, 0x0808192bu, 0x08081908u,
  0x08082b19u, 0x08081908u, 0x08190808u, 0x08081908u, 0x0819082bu, 0x08081908u, 0x08191919u, 0x08081908u,
  0x08192b08u, 0x08081908u, 0x082b0819u, 0x08081908u, 0x082b1908u, 0x08081908u, 0x082b192bu, 0x08081908u,
  0x082b2b19u, 0x08081908u, 0x19080808u, 0x08081908u, 0x1908082bu, 0x08081908u, 0x19081919u, 0x08081908u,
  0x19082b08u, 0x08081908u, 0x19082b2bu, 0x08081908u, 0x19190819u, 0x08081908u, 0x19191908u, 0x08081908u,
  0x1919192bu, 0x08081908u, 0x19192b19u, 0x08081908u, 0x192b0808u, 0x08081908u, 0x192b082bu, 0x08081908u,
  0x192b1919u, 0x08081908u, 0x2b080819u, 0x08081908u, 0x2b081908u, 0x08081908u, 0x2b08192bu, 0x08081908u,
  0x2b082b19u, 0x08081908u, 0x2b190808u, 0x08081908u, 0x2b191919u, 0x08081908u, 0x2b192b08u, 0x08081908u,
  0x2b2b0819u, 0x08081908u, 0x2b2b1908u, 0x08081908u, 0x08080808u, 0x08081919u, 0x0808082bu, 0x08081919u,
  0x08081919u, 0x08081919u, 0x08082b08u, 0x08081919u, 0x08082b2bu, 0x08081919u, 0x08190819u, 0x08081919u,
  0x08191908u, 0x08081919u, 0x0819192bu, 0x08081919u, 0x08192b19u, 0x08081919u, 0x082b0808u, 0x08081919u,
  0x082b1919u, 0x08081919u, 0x082b2b08u, 0x08081919u, 0x19080819u, 0x08081919u, 0x19081908u, 0x08081919u,
  0x1908192bu, 0x08081919u, 0x19082b19u, 0x08081919u, 0x19190808u, 0x08081919u, 0x1919082bu, 0x08081919u,
  0x19191919u, 0x08081919u, 0x19192b08u, 0x08081919u, 0x192b0819u, 0x08081919u, 0x192b1908u, 0x08081919u,
  0x2b080808u, 0x08081919u, 0x2b08082bu, 0x08081919u, 0x2b081919u, 0x08081919u, 0x2b082b08u, 0x08081919u,
  0x2b190819u, 0x08081919u, 0x2b191908u, 0x08081919u, 0x2b2b0808u, 0x08081919u, 0x08080819u, 0x0808192bu,
  0x08081908u, 0x0808192bu, 0x0808192bu, 0x0808192bu, 0x08082b19u, 0x0808192bu, 0x08190808u, 0x0808192bu,
  0x08191919u, 0x0808192bu, 0x19080808u, 0x0808192bu, 0x19081919u, 0x0808192bu, 0x19082b08u, 0x0808192bu,
  0x19190819u, 0x0808192bu, 0x19191908u, 0x0808192bu, 0x192b0808u, 0x0808192bu, 0x2b080819u, 0x0808192bu,
  0x2b081908u, 0x0808192bu, 0x2b190808u, 0x0808192bu, 0x08080808u, 0x08082b08u, 0x0808082bu, 0x08082b08u,
  0x08081919u, 0x08082b08u, 0x08082b08u, 0x08082b08u, 0x08190819u, 0x08082b08u, 0x08191908u, 0x08082b08u,
  0x0819192bu, 0x08082b08u, 0x08192b19u, 0x08082b08u, 0x082b0808u, 0x08082b08u, 0x082b1919u, 0x08082b08u,
  0x082b2b2bu, 0x08082b08u, 0x19080819u, 0x08082b08u, 0x19081908u, 0x08082b08u, 0x1908192bu, 0x08082b08u,
  0x19082b19u, 0x08082b08u, 0x19190808u, 0x08082b08u, 0x1919082bu, 0x08082b08u, 0x19191919u, 0x08082b08u,
  0x19192b08u, 0x08082b08u, 0x192b0819u, 0x08082b08u, 0x192b1908u, 0x08082b08u, 0x2b080808u, 0x08082b08u,
  0x2b081919u, 0x08082b08u, 0x2b191908u, 0x08082b08u, 0x2b2b2b2bu, 0x08082b08u, 0x08080819u, 0x08082b19u,
  0x08081908u, 0x08082b19u, 0x08190808u, 0x08082b19u, 0x0819082bu, 0x08082b19u, 0x08191919u, 0x08082b19u,
  0x08192b08u, 0x08082b19u, 0x082b0819u, 0x08082b19u, 0x19080808u, 0x08082b19u, 0x19081919u, 0x08082b19u,
  0x19082b08u, 0x08082b19u, 0x19190819u, 0x08082b19u, 0x19191908u, 0x08082b19u, 0x192b0808u, 0x08082b19u,
  0x2b080819u, 0x08082b19u, 0x2b190808u, 0x08082b19u, 0x08080808u, 0x08082b2bu, 0x08190819u, 0x08082b2bu,
  0x08191908u, 0x08082b2bu, 0x082b082bu, 0x08082b2bu, 0x082b2b08u, 0x08082b2bu, 0x082b2b2bu, 0x08082b2bu,
  0x19190808u, 0x08082b2bu, 0x2b192b19u, 0x08082b2bu, 0x08080819u, 0x08190808u, 0x08081908u, 0x08190808u,
  0x0808192bu, 0x08190808u, 0x08082b19u, 0x08190808u, 0x08190808u, 0x08190808u, 0x0819082bu, 0x08190808u,
  0x08191919u, 0x08190808u, 0x08192b08u, 0x08190808u, 0x082b0819u, 0x08190808u, 0x082b1908u, 0x08190808u,
  0x082b192bu, 0x08190808u, 0x19080808u, 0x08190808u, 0x1908082bu, 0x08190808u, 0x19081919u, 0x08190808u,
  0x19082b08u, 0x08190808u, 0x19190819u, 0x08190808u, 0x19191908u, 0x08190808u, 0x1919192bu, 0x08190808u,
  0x19192b19u, 0x08190808u, 0x192b0808u, 0x08190808u, 0x192b082bu, 0x08190808u, 0x192b1919u, 0x08190808u,
  0x192b2b08u, 0x08190808u, 0x2b080819u, 0x08190808u, 0x2b081908u, 0x08190808u, 0x2b08192bu, 0x08190808u,
  0x2b190808u, 0x08190808u, 0x2b191919u, 0x08190808u, 0x2b192b08u, 0x08190808u, 0x2b2b0819u, 0x08190808u,
  0x2b2b1908u, 0x08190808u, 0x08080808u, 0x08190819u, 0x0808082bu, 0x08190819u, 0x08081919u, 0x08190819u,
  0x08082b08u, 0x08190819u, 0x08082b2bu, 0x08190819u, 0x08190819u, 0x08190819u, 0x08191908u, 0x08190819u,
  0x0819192bu, 0x08190819u, 0x08192b19u, 0x08190819u, 0x082b0808u, 0x08190819u, 0x082b082bu, 0x08190819u,
  0x082b1919u, 0x08190819u, 0x082b2b08u, 0x08190819u, 0x19080819u, 0x08190819u, 0x19081908u, 0x08190819u,
  0x1908192bu, 0x08190819u, 0x19082b19u, 0x08190819u, 0x19190808u, 0x08190819u, 0x1919082bu, 0x08190819u,
  0x19191919u, 0x08190819u, 0x19192b08u, 0x08190819u, 0x192b0819u, 0x08190819u, 0x192b1908u, 0x08190819u,
  0x2b080808u, 0x08190819u, 0x2b08082bu, 0x08190819u, 0x2b081919u, 0x08190819u, 0x2b082b08u, 0x08190819u,
  0x2b190819u, 0x08190819u, 0x2b191908u, 0x08190819u, 0x08080819u, 0x0819082bu, 0x08081908u, 0x0819082bu,
  0x08082b19u, 0x0819082bu, 0x08190808u, 0x0819082bu, 0x08191919u, 0x0819082bu, 0x082b0819u, 0x0819082bu,
  0x082b1908u, 0x0819082bu, 0x19080808u, 0x0819082bu, 0x19081919u, 0x0819082bu, 0x19190819u, 0x0819082bu,
  0x19191908u, 0x0819082bu, 0x2b080819u, 0x0819082bu, 0x2b081908u, 0x0819082bu, 0x2b190808u, 0x0819082bu,
  0x08080808u, 0x08191908u, 0x0808082bu, 0x08191908u, 0x08081919u, 0x08191908u, 0x08082b08u, 0x08191908u,
  0x08190819u, 0x08191908u, 0x08191908u, 0x08191908u, 0x0819192bu, 0x08191908u, 0x08192b19u, 0x08191908u,
  0x082b0808u, 0x08191908u, 0x082b1919u, 0x08191908u, 0x082b2b08u, 0x08191908u, 0x19080819u, 0x08191908u,
  0x19081908u, 0x08191908u, 0x1908192bu, 0x08191908u, 0x19082b19u, 0x08191908u, 0x19190808u, 0x08191908u,
  0x1919082bu, 0x08191908u, 0x19191919u, 0x08191908u, 0x19192b08u, 0x08191908u, 0x192b0819u, 0x08191908u,
  0x192b1908u, 0x08191908u, 0x2b080808u, 0x08191908u, 0x2b08082bu, 0x08191908u, 0x2b081919u, 0x08191908u,
  0x2b082b08u, 0x08191908u, 0x2b190819u, 0x08191908u, 0x2b191908u, 0x08191908u, 0x2b2b0808u, 0x08191908u,
  0x08080819u, 0x08191919u, 0x08081908u, 0x08191919u, 0x0808192bu, 0x08191919u, 0x08082b19u, 0x08191919u,
  0x08190808u, 0x08191919u, 0x0819082bu, 0x08191919u, 0x08191919u, 0x08191919u, 0x08192b08u, 0x08191919u,
  0x082b0819u, 0x08191919u, 0x082b1908u, 0x08191919u, 0x19080808u, 0x08191919u, 0x1908082bu, 0x08191919u,
  0x19081919u, 0x08191919u, 0x19082b08u, 0x08191919u, 0x19190819u, 0x08191919u, 0x19191908u, 0x08191919u,
  0x192b0808u, 0x08191919u, 0x2b080819u, 0x08191919u, 0x2b081908u, 0x08191919u, 0x2b190808u, 0x08191919u,
  0x08080808u, 0x0819192bu, 0x08081919u, 0x0819192bu, 0x08082b08u, 0x0819192bu, 0x08190819u, 0x0819192bu,
  0x08191908u, 0x0819192bu, 0x082b0808u, 0x0819192bu, 0x19080819u, 0x0819192bu, 0x19081908u, 0x0819192bu,
  0x19190808u, 0x0819192bu, 0x2b080808u, 0x0819192bu, 0x2b2b2b2bu, 0x0819192bu, 0x08080819u, 0x08192b08u,
  0x08081908u, 0x08192b08u, 0x0808192bu, 0x08192b08u, 0x08082b19u, 0x08192b08u, 0x08190808u, 0x08192b08u,
  0x08191919u, 0x08192b08u, 0x08192b08u, 0x08192b08u, 0x082b0819u, 0x08192b08u, 0x19080808u, 0x08192b08u,
  0x1908082bu, 0x08192b08u, 0x19081919u, 0x08192b08u, 0x19082b08u, 0x08192b08u, 0x19190819u, 0x08192b08u,
  0x19191908u, 0x08192b08u, 0x192b0808u, 0x08192b08u, 0x2b080819u, 0x08192b08u, 0x2b081908u, 0x08192b08u,
  0x08080808u, 0x08192b19u, 0x0808082bu, 0x08192b19u, 0x08081919u, 0x08192b19u, 0x08082b08u, 0x08192b19u,
  0x08190819u, 0x08192b19u, 0x08191908u, 0x08192b19u, 0x082b0808u, 0x08192b19u, 0x19080819u, 0x08192b19u,
  0x19081908u, 0x08192b19u, 0x19190808u, 0x08192b19u, 0x192b2b19u, 0x08192b19u, 0x2b2b082bu, 0x08192b19u,
  0x08081908u, 0x08192b2bu, 0x08190808u, 0x08192b2bu, 0x19080808u, 0x08192b2bu, 0x1919192bu, 0x08192b2bu,
  0x08080808u, 0x082b0808u, 0x0808082bu, 0x082b0808u, 0x08081919u, 0x082b0808u, 0x08082b08u, 0x082b0808u,
  0x08190819u, 0x082b0808u, 0x08191908u, 0x082b0808u, 0x0819192bu, 0x082b0808u, 0x08192b19u, 0x082b0808u,
  0x082b0808u, 0x082b0808u, 0x082b1919u, 0x082b0808u, 0x082b2b2bu, 0x082b0808u, 0x19080819u, 0x082b0808u,
  0x19081908u, 0x082b0808u, 0x19190808u, 0x082b0808u, 0x1919082bu, 0x082b0808u, 0x19191919u, 0x082b0808u,
  0x192b1908u, 0x082b0808u, 0x2b080808u, 0x082b0808u, 0x2b082b2bu, 0x082b0808u, 0x2b191908u, 0x082b0808u,
  0x2b2b2b2bu, 0x082b0808u, 0x08080819u, 0x082b0819u, 0x08081908u, 0x082b0819u, 0x08190808u, 0x082b0819u,
  0x0819082bu, 0x082b0819u, 0x08191919u, 0x082b0819u, 0x082b0819u, 0x082b0819u, 0x19080808u, 0x082b0819u,
  0x1908082bu, 0x082b0819u, 0x19081919u, 0x082b0819u, 0x19190819u, 0x082b0819u, 0x19191908u, 0x082b0819u,
  0x192b0808u, 0x082b0819u, 0x2b080819u, 0x082b0819u, 0x2b081908u, 0x082b0819u, 0x2b190808u, 0x082b0819u,
  0x08080808u, 0x082b082bu, 0x08082b2bu, 0x082b082bu, 0x082b082bu, 0x082b082bu, 0x082b2b08u, 0x082b082bu,
  0x082b2b2bu, 0x082b082bu, 0x19081908u, 0x082b082bu, 0x19190808u, 0x082b082bu, 0x2b082b08u, 0x082b082bu,
  0x2b082b2bu, 0x082b082bu, 0x2b2b2b08u, 0x082b082bu, 0x08080819u, 0x082b1908u, 0x08081908u, 0x082b1908u,
  0x0808192bu, 0x082b1908u, 0x08082b19u, 0x082b1908u, 0x08190808u, 0x082b1908u, 0x08191919u, 0x082b1908u,
  0x08192b08u, 0x082b1908u, 0x082b0819u, 0x082b1908u, 0x082b1908u, 0x082b1908u, 0x19080808u, 0x082b1908u,
  0x1908082bu, 0x082b1908u, 0x19081919u, 0x082b1908u, 0x19082b08u, 0x082b1908u, 0x19190819u, 0x082b1908u,
  0x19191908u, 0x082b1908u, 0x192b0808u, 0x082b1908u, 0x2b080819u, 0x082b1908u, 0x2b081908u, 0x082b1908u,
  0x2b190808u, 0x082b1908u, 0x08080808u, 0x082b1919u, 0x08081919u, 0x082b1919u, 0x08082b08u, 0x082b1919u,
  0x08190819u, 0x082b1919u, 0x08191908u, 0x082b1919u, 0x082b0808u, 0x082b1919u, 0x19080819u, 0x082b1919u,
  0x19081908u, 0x082b1919u, 0x19190808u, 0x082b1919u, 0x192b192bu, 0x082b1919u, 0x2b080808u, 0x082b1919u,
  0x08080819u, 0x082b192bu, 0x08081908u, 0x082b192bu, 0x08190808u, 0x082b192bu, 0x19080808u, 0x082b192bu,
  0x19192b19u, 0x082b192bu, 0x08080808u, 0x082b2b08u, 0x08081919u, 0x082b2b08u, 0x08190819u, 0x082b2b08u,
  0x08191908u, 0x082b2b08u, 0x19080819u, 0x082b2b08u, 0x19081908u, 0x082b2b08u, 0x19190808u, 0x082b2b08u,
  0x2b082b2bu, 0x082b2b08u, 0x2b2b2b2bu, 0x082b2b08u, 0x08080819u, 0x082b2b19u, 0x08081908u, 0x082b2b19u,
  0x08190808u, 0x082b2b19u, 0x2b191919u, 0x082b2b19u, 0x08082b2bu, 0x082b2b2bu, 0x082b082bu, 0x082b2b2bu,
  0x192b1908u, 0x082b2b2bu, 0x2b082b08u, 0x082b2b2bu, 0x2b082b2bu, 0x082b2b2bu, 0x08080819u, 0x19080808u,
  0x08081908u, 0x19080808u, 0x0808192bu, 0x19080808u, 0x08082b19u, 0x19080808u, 0x08190808u, 0x19080808u,
  0x0819082bu, 0x19080808u, 0x08191919u, 0x19080808u, 0x08192b08u, 0x19080808u, 0x08192b2bu, 0x19080808u,
  0x082b0819u, 0x19080808u, 0x082b1908u, 0x19080808u, 0x082b192bu, 0x19080808u, 0x19080808u, 0x19080808u,
  0x1908082bu, 0x19080808u, 0x19081919u, 0x19080808u, 0x19082b08u, 0x19080808u, 0x19082b2bu, 0x19080808u,
  0x19190819u, 0x19080808u, 0x19191908u, 0x19080808u, 0x1919192bu, 0x19080808u, 0x19192b19u, 0x19080808u,
  0x192b0808u, 0x19080808u, 0x192b082bu, 0x19080808u, 0x192b1919u, 0x19080808u, 0x2b080819u, 0x19080808u,
  0x2b081908u, 0x19080808u, 0x2b190808u, 0x19080808u, 0x2b191919u, 0x19080808u, 0x2b192b08u, 0x19080808u,
  0x2b2b0819u, 0x19080808u, 0x2b2b1908u, 0x19080808u, 0x08080808u, 0x19080819u, 0x0808082bu, 0x19080819u,
  0x08081919u, 0x19080819u, 0x08082b08u, 0x19080819u, 0x08190819u, 0x19080819u, 0x08191908u, 0x19080819u,
  0x0819192bu, 0x19080819u, 0x08192b19u, 0x19080819u, 0x082b0808u, 0x19080819u, 0x082b082bu, 0x19080819u,
  0x082b1919u, 0x19080819u, 0x19080819u, 0x19080819u, 0x19081908u, 0x19080819u, 0x1908192bu, 0x19080819u,
  0x19082b19u, 0x19080819u, 0x19190808u, 0x19080819u, 0x1919082bu, 0x19080819u, 0x19191919u, 0x19080819u,
  0x19192b08u, 0x19080819u, 0x192b0819u, 0x19080819u, 0x192b1908u, 0x19080819u, 0x2b080808u, 0x19080819u,
  0x2b08082bu, 0x19080819u, 0x2b081919u, 0x19080819u, 0x2b082b08u, 0x19080819u, 0x2b190819u, 0x19080819u,
  0x2b191908u, 0x19080819u, 0x2b2b0808u, 0x19080819u, 0x08080819u, 0x1908082bu, 0x08081908u, 0x1908082bu,
  0x08190808u, 0x1908082bu, 0x0819082bu, 0x1908082bu, 0x08191919u, 0x1908082bu, 0x08192b08u, 0x1908082bu,
  0x082b1908u, 0x1908082bu, 0x19080808u, 0x1908082bu, 0x19081919u, 0x1908082bu, 0x19082b08u, 0x1908082bu,
  0x19190819u, 0x1908082bu, 0x19191908u, 0x1908082bu, 0x192b0808u, 0x1908082bu, 0x2b080819u, 0x1908082bu,
  0x2b081908u, 0x1908082bu, 0x08080808u, 0x19081908u, 0x0808082bu, 0x19081908u, 0x08081919u, 0x19081908u,
  0x08082b08u, 0x19081908u, 0x08082b2bu, 0x19081908u, 0x08190819u, 0x19081908u, 0x08191908u, 0x19081908u,
  0x0819192bu, 0x19081908u, 0x08192b19u, 0x19081908u, 0x082b0808u, 0x19081908u, 0x082b082bu, 0x19081908u,
  0x082b1919u, 0x19081908u, 0x082b2b08u, 0x19081908u, 0x19080819u, 0x19081908u, 0x19081908u, 0x19081908u,
  0x1908192bu, 0x19081908u, 0x19082b19u, 0x19081908u, 0x19190808u, 0x19081908u, 0x1919082bu, 0x19081908u,
  0x19191919u, 0x19081908u, 0x19192b08u, 0x19081908u, 0x192b0819u, 0x19081908u, 0x192b1908u, 0x19081908u,
  0x2b080808u, 0x19081908u, 0x2b08082bu, 0x19081908u, 0x2b081919u, 0x19081908u, 0x2b082b08u, 0x19081908u,
  0x2b190819u, 0x19081908u, 0x2b191908u, 0x19081908u, 0x2b2b0808u, 0x19081908u, 0x08080819u, 0x19081919u,
  0x08081908u, 0x19081919u, 0x0808192bu, 0x19081919u, 0x08082b19u, 0x19081919u, 0x08190808u, 0x19081919u,
  0x0819082bu, 0x19081919u, 0x08191919u, 0x19081919u, 0x08192b08u, 0x19081919u, 0x082b0819u, 0x19081919u,
  0x082b1908u, 0x19081919u, 0x19080808u, 0x19081919u, 0x1908082bu, 0x19081919u, 0x19081919u, 0x19081919u,
  0x19082b08u, 0x19081919u, 0x19190819u, 0x19081919u, 0x19191908u, 0x19081919u, 0x192b0808u, 0x19081919u,
  0x192b2b2bu, 0x19081919u, 0x2b080819u, 0x19081919u, 0x2b081908u, 0x19081919u, 0x2b190808u, 0x19081919u,
  0x08080808u, 0x1908192bu, 0x0808082bu, 0x1908192bu, 0x08081919u, 0x1908192bu, 0x08082b08u, 0x1908192bu,
  0x08190819u, 0x1908192bu, 0x08191908u, 0x1908192bu, 0x082b0808u, 0x1908192bu, 0x19080819u, 0x1908192bu,
  0x19081908u, 0x1908192bu, 0x19190808u, 0x1908192bu, 0x2b080808u, 0x1908192bu, 0x2b2b1919u, 0x1908192bu,
  0x08080819u, 0x19082b08u, 0x08081908u, 0x19082b08u, 0x08082b19u, 0x19082b08u, 0x08190808u, 0x19082b08u,
  0x0819082bu, 0x19082b08u, 0x08191919u, 0x19082b08u, 0x08192b08u, 0x19082b08u, 0x082b0819u, 0x19082b08u,
  0x082b1908u, 0x19082b08u, 0x19080808u, 0x19082b08u, 0x1908082bu, 0x19082b08u, 0x19081919u, 0x19082b08u,
  0x19082b08u, 0x19082b08u, 0x19190819u, 0x19082b08u, 0x19191908u, 0x19082b08u, 0x192b0808u, 0x19082b08u,
  0x2b081908u, 0x19082b08u, 0x2b190808u, 0x19082b08u, 0x08080808u, 0x19082b19u, 0x0808082bu, 0x19082b19u,
  0x08081919u, 0x19082b19u, 0x08082b08u, 0x19082b19u, 0x08190819u, 0x19082b19u, 0x08191908u, 0x19082b19u,
  0x082b0808u, 0x19082b19u, 0x19080819u, 0x19082b19u, 0x19081908u, 0x19082b19u, 0x19190808u, 0x19082b19u,
  0x2b080808u, 0x19082b19u, 0x2b19192bu, 0x19082b19u, 0x08080819u, 0x19082b2bu, 0x08081908u, 0x19082b2bu,
  0x08190808u, 0x19082b2bu, 0x19080808u, 0x19082b2bu, 0x08080808u, 0x19190808u, 0x0808082bu, 0x19190808u,
  0x08081919u, 0x19190808u, 0x08082b08u, 0x19190808u, 0x08190819u, 0x19190808u, 0x08191908u, 0x19190808u,
  0x0819192bu, 0x19190808u, 0x08192b19u, 0x19190808u, 0x082b0808u, 0x19190808u, 0x082b082bu, 0x19190808u,
  0x082b1919u, 0x19190808u, 0x082b2b08u, 0x19190808u, 0x19080819u, 0x19190808u, 0x19081908u, 0x19190808u,
  0x1908192bu, 0x19190808u, 0x19082b19u, 0x19190808u, 0x19190808u, 0x19190808u, 0x1919082bu, 0x19190808u,
  0x19191919u, 0x19190808u, 0x19192b08u, 0x19190808u, 0x192b0819u, 0x19190808u, 0x192b1908u, 0x19190808u,
  0x2b080808u, 0x19190808u, 0x2b08082bu, 0x19190808u, 0x2b081919u, 0x19190808u, 0x2b082b08u, 0x19190808u,
  0x2b190819u, 0x19190808u, 0x2b191908u, 0x19190808u, 0x08080819u, 0x19190819u, 0x08081908u, 0x19190819u,
  0x0808192bu, 0x19190819u, 0x08082b19u, 0x19190819u, 0x08190808u, 0x19190819u, 0x0819082bu, 0x19190819u,
  0x08191919u, 0x19190819u, 0x08192b08u, 0x19190819u, 0x082b0819u, 0x19190819u, 0x082b1908u, 0x19190819u,
  0x19080808u, 0x19190819u, 0x1908082bu, 0x19190819u, 0x19081919u, 0x19190819u, 0x19082b08u, 0x19190819u,
  0x19190819u, 0x19190819u, 0x19191908u, 0x19190819u, 0x192b0808u, 0x19190819u, 0x2b080819u, 0x19190819u,
  0x2b081908u, 0x19190819u, 0x2b190808u, 0x19190819u, 0x08080808u, 0x1919082bu, 0x08081919u, 0x1919082bu,
  0x08082b08u, 0x1919082bu, 0x08190819u, 0x1919082bu, 0x08191908u, 0x1919082bu, 0x082b0808u, 0x1919082bu,
  0x19080819u, 0x1919082bu, 0x19081908u, 0x1919082bu, 0x19190808u, 0x1919082bu, 0x192b2b19u, 0x1919082bu,
  0x2b080808u, 0x1919082bu, 0x08080819u, 0x19191908u, 0x08081908u, 0x19191908u, 0x0808192bu, 0x19191908u,
  0x08082b19u, 0x19191908u, 0x08190808u, 0x19191908u, 0x0819082bu, 0x19191908u, 0x08191919u, 0x19191908u,
  0x08192b08u, 0x19191908u, 0x082b0819u, 0x19191908u, 0x082b1908u, 0x19191908u, 0x19080808u, 0x19191908u,
  0x1908082bu, 0x19191908u, 0x19081919u, 0x19191908u, 0x19082b08u, 0x19191908u, 0x19190819u, 0x19191908u,
  0x19191908u, 0x19191908u, 0x192b0808u, 0x19191908u, 0x2b080819u, 0x19191908u, 0x2b081908u, 0x19191908u,
  0x2b190808u, 0x19191908u, 0x08080808u, 0x19191919u, 0x0808082bu, 0x19191919u, 0x08081919u, 0x19191919u,
  0x08082b08u, 0x19191919u, 0x08190819u, 0x19191919u, 0x08191908u, 0x19191919u, 0x082b0808u, 0x19191919u,
  0x19080819u, 0x19191919u, 0x19081908u, 0x19191919u, 0x19190808u, 0x19191919u, 0x2b080808u, 0x19191919u,
  0x08080819u, 0x1919192bu, 0x08081908u, 0x1919192bu, 0x08190808u, 0x1919192bu, 0x082b192bu, 0x1919192bu,
  0x19080808u, 0x1919192bu, 0x08080808u, 0x19192b08u, 0x0808082bu, 0x19192b08u, 0x08081919u, 0x19192b08u,
  0x08082b08u, 0x19192b08u, 0x08190819u, 0x19192b08u, 0x08191908u, 0x19192b08u, 0x082b0808u, 0x19192b08u,
  0x19080819u, 0x19192b08u, 0x19081908u, 0x19192b08u, 0x19190808u, 0x19192b08u, 0x19192b2bu, 0x19192b08u,
  0x2b080808u, 0x19192b08u, 0x08080819u, 0x19192b19u, 0x08081908u, 0x19192b19u, 0x08190808u, 0x19192b19u,
  0x19080808u, 0x19192b19u, 0x08080808u, 0x19192b2bu, 0x08192b19u, 0x19192b2bu, 0x2b081919u, 0x19192b2bu,
  0x2b2b2b08u, 0x19192b2bu, 0x08080819u, 0x192b0808u, 0x08081908u, 0x192b0808u, 0x0808192bu, 0x192b0808u,
  0x08190808u, 0x192b0808u, 0x0819082bu, 0x192b0808u, 0x08191919u, 0x192b0808u, 0x08192b08u, 0x192b0808u,
  0x082b0819u, 0x192b0808u, 0x082b1908u, 0x192b0808u, 0x19080808u, 0x192b0808u, 0x19081919u, 0x192b0808u,
  0x19082b08u, 0x192b0808u, 0x19190819u, 0x192b0808u, 0x19191908u, 0x192b0808u, 0x192b0808u, 0x192b0808u,
  0x2b081908u, 0x192b0808u, 0x2b190808u, 0x192b0808u, 0x08080808u, 0x192b0819u, 0x0808082bu, 0x192b0819u,
  0x08081919u, 0x192b0819u, 0x08082b08u, 0x192b0819u, 0x08190819u, 0x192b0819u, 0x08191908u, 0x192b0819u,
  0x082b0808u, 0x192b0819u, 0x19080819u, 0x192b0819u, 0x19081908u, 0x192b0819u, 0x19190808u, 0x192b0819u,
  0x2b080808u, 0x192b0819u, 0x2b192b19u, 0x192b0819u, 0x08081908u, 0x192b082bu, 0x08190808u, 0x192b082bu,
  0x19080808u, 0x192b082bu, 0x1919192bu, 0x192b082bu, 0x2b2b0819u, 0x192b082bu, 0x08080808u, 0x192b1908u,
  0x08081919u, 0x192b1908u, 0x08082b08u, 0x192b1908u, 0x08190819u, 0x192b1908u, 0x08191908u, 0x192b1908u,
  0x082b0808u, 0x192b1908u, 0x19080819u, 0x192b1908u, 0x19081908u, 0x192b1908u, 0x19190808u, 0x192b1908u,
  0x2b080808u, 0x192b1908u, 0x08080819u, 0x192b1919u, 0x08081908u, 0x192b1919u, 0x08190808u, 0x192b1919u,
  0x19080808u, 0x192b1919u, 0x19082b2bu, 0x192b1919u, 0x192b2b08u, 0x192b1919u, 0x2b19082bu, 0x192b1919u,
  0x08080808u, 0x192b192bu, 0x2b191908u, 0x192b192bu, 0x08080819u, 0x192b2b08u, 0x08081908u, 0x192b2b08u,
  0x08190808u, 0x192b2b08u, 0x192b1919u, 0x192b2b08u, 0x2b192b08u, 0x192b2b08u, 0x08080808u, 0x192b2b19u,
  0x082b2b2bu, 0x192b2b19u, 0x1908082bu, 0x192b2b2bu, 0x2b2b0819u, 0x192b2b2bu, 0x08080808u, 0x2b080808u,
  0x0808082bu, 0x2b080808u, 0x08081919u, 0x2b080808u, 0x08082b08u, 0x2b080808u, 0x08190819u, 0x2b080808u,
  0x08191908u, 0x2b080808u, 0x08192b19u, 0x2b080808u, 0x082b0808u, 0x2b080808u, 0x082b1919u, 0x2b080808u,
  0x19080819u, 0x2b080808u, 0x19081908u, 0x2b080808u, 0x19190808u, 0x2b080808u, 0x1919082bu, 0x2b080808u,
  0x19191919u, 0x2b080808u, 0x19192b08u, 0x2b080808u, 0x192b0819u, 0x2b080808u, 0x2b080808u, 0x2b080808u,
  0x2b081919u, 0x2b080808u, 0x2b190819u, 0x2b080808u, 0x2b191908u, 0x2b080808u, 0x08080819u, 0x2b080819u,
  0x08081908u, 0x2b080819u, 0x08082b19u, 0x2b080819u, 0x08190808u, 0x2b080819u, 0x0819082bu, 0x2b080819u,
  0x08191919u, 0x2b080819u, 0x08192b08u, 0x2b080819u, 0x082b0819u, 0x2b080819u, 0x082b1908u, 0x2b080819u,
  0x19080808u, 0x2b080819u, 0x1908082bu, 0x2b080819u, 0x19081919u, 0x2b080819u, 0x19082b08u, 0x2b080819u,
  0x19190819u, 0x2b080819u, 0x19191908u, 0x2b080819u, 0x2b080819u, 0x2b080819u, 0x2b081908u, 0x2b080819u,
  0x2b190808u, 0x2b080819u, 0x2b2b2b19u, 0x2b080819u, 0x08080808u, 0x2b08082bu, 0x08081919u, 0x2b08082bu,
  0x08082b2bu, 0x2b08082bu, 0x08190819u, 0x2b08082bu, 0x08191908u, 0x2b08082bu, 0x19080819u, 0x2b08082bu,
  0x19081908u, 0x2b08082bu, 0x19190808u, 0x2b08082bu, 0x08080819u, 0x2b081908u, 0x08081908u, 0x2b081908u,
  0x0808192bu, 0x2b081908u, 0x08082b19u, 0x2b081908u, 0x08190808u, 0x2b081908u, 0x0819082bu, 0x2b081908u,
  0x08191919u, 0x2b081908u, 0x08192b08u, 0x2b081908u, 0x082b0819u, 0x2b081908u, 0x19080808u, 0x2b081908u,
  0x1908082bu, 0x2b081908u, 0x19081919u, 0x2b081908u, 0x19082b08u, 0x2b081908u, 0x19190819u, 0x2b081908u,
  0x19191908u, 0x2b081908u, 0x192b0808u, 0x2b081908u, 0x2b080819u, 0x2b081908u, 0x2b081908u, 0x2b081908u,
  0x2b190808u, 0x2b081908u, 0x08080808u, 0x2b081919u, 0x0808082bu, 0x2b081919u, 0x08081919u, 0x2b081919u,
  0x08082b08u, 0x2b081919u, 0x08190819u, 0x2b081919u, 0x08191908u, 0x2b081919u, 0x082b0808u, 0x2b081919u,
  0x19080819u, 0x2b081919u, 0x19081908u, 0x2b081919u, 0x19190808u, 0x2b081919u, 0x2b080808u, 0x2b081919u,
  0x2b082b2bu, 0x2b081919u, 0x08080819u, 0x2b08192bu, 0x08081908u, 0x2b08192bu, 0x08190808u, 0x2b08192bu,
  0x082b2b19u, 0x2b08192bu, 0x19080808u, 0x2b08192bu, 0x08080808u, 0x2b082b08u, 0x08081919u, 0x2b082b08u,
  0x08190819u, 0x2b082b08u, 0x08191908u, 0x2b082b08u, 0x19080819u, 0x2b082b08u, 0x19081908u, 0x2b082b08u,
  0x19190808u, 0x2b082b08u, 0x2b2b082bu, 0x2b082b08u, 0x08080819u, 0x2b082b19u, 0x08081908u, 0x2b082b19u,
  0x19080808u, 0x2b082b19u, 0x192b1919u, 0x2b082b19u, 0x082b082bu, 0x2b082b2bu, 0x19192b08u, 0x2b082b2bu,
  0x19192b2bu, 0x2b082b2bu, 0x2b08082bu, 0x2b082b2bu, 0x2b2b082bu, 0x2b082b2bu, 0x08080819u, 0x2b190808u,
  0x08081908u, 0x2b190808u, 0x08082b19u, 0x2b190808u, 0x08190808u, 0x2b190808u, 0x0819082bu, 0x2b190808u,
  0x08191919u, 0x2b190808u, 0x08192b08u, 0x2b190808u, 0x082b1908u, 0x2b190808u, 0x19080808u, 0x2b190808u,
  0x1908082bu, 0x2b190808u, 0x19081919u, 0x2b190808u, 0x19082b08u, 0x2b190808u, 0x19190819u, 0x2b190808u,
  0x19191908u, 0x2b190808u, 0x192b0808u, 0x2b190808u, 0x2b080819u, 0x2b190808u, 0x2b081908u, 0x2b190808u,
  0x2b190808u, 0x2b190808u, 0x08080808u, 0x2b190819u, 0x08081919u, 0x2b190819u, 0x08190819u, 0x2b190819u,
  0x08191908u, 0x2b190819u, 0x19080819u, 0x2b190819u, 0x19081908u, 0x2b190819u, 0x19190808u, 0x2b190819u,
  0x19192b2bu, 0x2b190819u, 0x08080819u, 0x2b19082bu, 0x08081908u, 0x2b19082bu, 0x08190808u, 0x2b19082bu,
  0x19080808u, 0x2b19082bu, 0x2b2b192bu, 0x2b19082bu, 0x08080808u, 0x2b191908u, 0x0808082bu, 0x2b191908u,
  0x08081919u, 0x2b191908u, 0x08082b08u, 0x2b191908u, 0x08190819u, 0x2b191908u, 0x08191908u, 0x2b191908u,
  0x082b0808u, 0x2b191908u, 0x19080819u, 0x2b191908u, 0x19081908u, 0x2b191908u, 0x19190808u, 0x2b191908u,
  0x2b080808u, 0x2b191908u, 0x2b19192bu, 0x2b191908u, 0x08080819u, 0x2b191919u, 0x08081908u, 0x2b191919u,
  0x08190808u, 0x2b191919u, 0x19080808u, 0x2b191919u, 0x2b192b08u, 0x2b191919u, 0x2b2b0819u, 0x2b191919u,
  0x08080808u, 0x2b19192bu, 0x1908192bu, 0x2b19192bu, 0x192b1908u, 0x2b19192bu, 0x08080819u, 0x2b192b08u,
  0x08081908u, 0x2b192b08u, 0x08190808u, 0x2b192b08u, 0x082b192bu, 0x2b192b08u, 0x19080808u, 0x2b192b08u,
  0x2b2b2b19u, 0x2b192b08u, 0x08080808u, 0x2b192b19u, 0x19082b19u, 0x2b192b19u, 0x1919082bu, 0x2b192b19u,
  0x2b190808u, 0x2b192b2bu, 0x08080808u, 0x2b2b0808u, 0x08081919u, 0x2b2b0808u, 0x08082b2bu, 0x2b2b0808u,
  0x08191908u, 0x2b2b0808u, 0x082b082bu, 0x2b2b0808u, 0x082b2b2bu, 0x2b2b0808u, 0x19080819u, 0x2b2b0808u,
  0x19081908u, 0x2b2b0808u, 0x19190808u, 0x2b2b0808u, 0x2b2b082bu, 0x2b2b0808u, 0x2b2b2b2bu, 0x2b2b0808u,
  0x19080808u, 0x2b2b0819u, 0x192b1919u, 0x2b2b0819u, 0x0808082bu, 0x2b2b082bu, 0x08082b2bu, 0x2b2b082bu,
  0x082b082bu, 0x2b2b082bu, 0x082b2b08u, 0x2b2b082bu, 0x082b2b2bu, 0x2b2b082bu, 0x2b08082bu, 0x2b2b082bu,
  0x2b082b08u, 0x2b2b082bu, 0x2b082b2bu, 0x2b2b082bu, 0x2b2b2b08u, 0x2b2b082bu, 0x08080819u, 0x2b2b1908u,
  0x08081908u, 0x2b2b1908u, 0x08190808u, 0x2b2b1908u, 0x19080808u, 0x2b2b1908u, 0x2b082b19u, 0x2b2b1908u,
  0x2b2b1908u, 0x2b2b1908u, 0x08080808u, 0x2b2b1919u, 0x08192b19u, 0x2b2b1919u, 0x19190819u, 0x2b2b192bu,
  0x08082b2bu, 0x2b2b2b08u, 0x082b2b08u, 0x2b2b2b08u, 0x2b2b082bu, 0x2b2b2b08u, 0x19191908u, 0x2b2b2b19u,
  0x2b08192bu, 0x2b2b2b19u, 0x08082b08u, 0x2b2b2b2bu, 0x08082b2bu, 0x2b2b2b2bu, 0x082b0808u, 0x2b2b2b2bu,
  0x082b082bu, 0x2b2b2b2bu, 0x082b2b08u, 0x2b2b2b2bu, 0x2b082b08u, 0x2b2b2b2bu, 0x2b2b2b2bu, 0x2b2b2b2bu,
);
// </iq3-iq2s-tables-wgsl>

/** Magnitude byte j (0..3) of iq3xxs_grid entry `idx` (0..255). */
fn iq3xxs_grid_byte(idx: u32, j: u32) -> f32 {
  return f32((IQ3XXS_GRID[idx] >> (j * 8u)) & 0xFFu);
}
/** Magnitude byte j (0..3) of iq3s_grid entry `idx` (0..511). */
fn iq3s_grid_byte(idx: u32, j: u32) -> f32 {
  return f32((IQ3S_GRID[idx] >> (j * 8u)) & 0xFFu);
}
/** Magnitude byte j (0..7) of iq2s_grid entry `idx` (0..1023). */
fn iq2s_grid_byte(idx: u32, j: u32) -> f32 {
  let word = IQ2S_GRID[idx * 2u + (j >> 2u)];
  return f32((word >> ((j & 3u) * 8u)) & 0xFFu);
}

// ── IQ3_XXS: 256-elem superblocks, 25 u32 each; unit = 32-elem ib32 ─────
// Layout (repacked 98→100 B): d(f16) @0 | qs[64] grid idx @2 | scales_and_
// signs[32] u32 @66. db = d·(0.5+ls)·0.5; two 4-byte grid lookups per l;
// signs via ksigns_iq2xs (IQ2XXS_SIGNS).
@compute @workgroup_size(WG_SIZE, 1, 1)
fn matmul_gguf_iq3_xxs(@builtin(local_invocation_id) lid: vec3u,
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
    let ib = u % 8u;
    let wordBase = (n * nSB + sb) * 25u;
    let d = unpack2x16float(W[wordBase]).x;
    let qsByte = wordBase * 4u + 2u + ib * 8u;        // 8 grid indices
    let ssByte = wordBase * 4u + 66u + ib * 4u;       // aux32 for this ib32
    let aux1 = wbyte(ssByte) | (wbyte(ssByte + 1u) << 8u)
             | (wbyte(ssByte + 2u) << 16u) | (wbyte(ssByte + 3u) << 24u);
    let db = d * (0.5 + f32(aux1 >> 28u)) * 0.5;
    let aBase = m * K + sb * 256u + ib * 32u;
    var dot: f32 = 0.0;
    for (var l = 0u; l < 4u; l = l + 1u) {
      let signs = iq2xxs_sign_byte((aux1 >> (7u * l)) & 127u);
      let gi1 = wbyte(qsByte + 2u * l);
      let gi2 = wbyte(qsByte + 2u * l + 1u);
      for (var j = 0u; j < 4u; j = j + 1u) {
        let s1 = select(1.0, -1.0, ((signs >> j) & 1u) == 1u);
        let s2 = select(1.0, -1.0, ((signs >> (j + 4u)) & 1u) == 1u);
        dot = dot + A[aBase + l * 8u + j] * iq3xxs_grid_byte(gi1, j) * s1;
        dot = dot + A[aBase + l * 8u + j + 4u] * iq3xxs_grid_byte(gi2, j) * s2;
      }
    }
    let y = db * dot - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  reduce_and_store(tid, n, m, sum);
}

// ── IQ3_S: 256-elem superblocks, 28 u32 each; unit = 64-elem ib32 pair ──
// Layout (repacked 110→112 B): d(f16) @0 | qs[64] @2 | qh[8] @66 |
// signs[32] @74 | scales[4] @106. Each pair shares a scale byte (low/high
// nibble → db1/db2, NO 0.5 offset); each index gets 1 high bit from qh.
@compute @workgroup_size(WG_SIZE, 1, 1)
fn matmul_gguf_iq3_s(@builtin(local_invocation_id) lid: vec3u,
                     @builtin(workgroup_id) wid: vec3u) {
  let n = wid.x + wid.z * 65535u;
  let m = wid.y;
  let tid = lid.x;
  let K = params.K;
  let nSB = K / 256u;
  let nUnits = nSB * 4u;                               // 4 pairs per superblock

  var sum: f32 = 0.0;
  var comp: f32 = 0.0;
  for (var u = tid; u < nUnits; u = u + WG_SIZE) {
    let sb = u / 4u;
    let p = u % 4u;
    let wordBase = (n * nSB + sb) * 28u;
    let d = unpack2x16float(W[wordBase]).x;
    let qsByte = wordBase * 4u + 2u;
    let qhByte = wordBase * 4u + 66u;
    let sgByte = wordBase * 4u + 74u;
    let scByte = wordBase * 4u + 106u;
    let sc = wbyte(scByte + p);
    let db1 = d * (1.0 + 2.0 * f32(sc & 0xFu));
    let db2 = d * (1.0 + 2.0 * f32(sc >> 4u));
    let qh0 = wbyte(qhByte + 2u * p);
    let qh1 = wbyte(qhByte + 2u * p + 1u);
    let qsA = qsByte + p * 16u;
    let qsB = qsByte + p * 16u + 8u;
    let sgA = sgByte + p * 8u;
    let sgB = sgByte + p * 8u + 4u;
    let aBase = m * K + sb * 256u + p * 64u;
    var dot1: f32 = 0.0;                               // first half (qh0, db1)
    var dot2: f32 = 0.0;                               // second half (qh1, db2)
    for (var l = 0u; l < 4u; l = l + 1u) {
      let gi1a = wbyte(qsA + 2u * l) | ((qh0 << (8u - 2u * l)) & 256u);
      let gi2a = wbyte(qsA + 2u * l + 1u) | ((qh0 << (7u - 2u * l)) & 256u);
      let sga = wbyte(sgA + l);
      let gi1b = wbyte(qsB + 2u * l) | ((qh1 << (8u - 2u * l)) & 256u);
      let gi2b = wbyte(qsB + 2u * l + 1u) | ((qh1 << (7u - 2u * l)) & 256u);
      let sgb = wbyte(sgB + l);
      for (var j = 0u; j < 4u; j = j + 1u) {
        let sa1 = select(1.0, -1.0, ((sga >> j) & 1u) == 1u);
        let sa2 = select(1.0, -1.0, ((sga >> (j + 4u)) & 1u) == 1u);
        let sb1 = select(1.0, -1.0, ((sgb >> j) & 1u) == 1u);
        let sb2 = select(1.0, -1.0, ((sgb >> (j + 4u)) & 1u) == 1u);
        dot1 = dot1 + A[aBase + l * 8u + j] * iq3s_grid_byte(gi1a, j) * sa1;
        dot1 = dot1 + A[aBase + l * 8u + j + 4u] * iq3s_grid_byte(gi2a, j) * sa2;
        dot2 = dot2 + A[aBase + 32u + l * 8u + j] * iq3s_grid_byte(gi1b, j) * sb1;
        dot2 = dot2 + A[aBase + 32u + l * 8u + j + 4u] * iq3s_grid_byte(gi2b, j) * sb2;
      }
    }
    let y = (db1 * dot1 + db2 * dot2) - comp;
    let t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  reduce_and_store(tid, n, m, sum);
}

// ── IQ2_S: 256-elem superblocks, 21 u32 each; unit = 32-elem ib32 ───────
// Layout (repacked 82→84 B): d(f16) @0 | qs[32] idx + signs[32] @2 (qs
// region 64 B) | qh[8] @66 | scales[8] @74. db = d·(0.5+nibble)·0.25;
// each index gets 2 high bits from qh; signs are inline bytes.
@compute @workgroup_size(WG_SIZE, 1, 1)
fn matmul_gguf_iq2_s(@builtin(local_invocation_id) lid: vec3u,
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
    let ib = u % 8u;
    let wordBase = (n * nSB + sb) * 21u;
    let d = unpack2x16float(W[wordBase]).x;
    let qsByte = wordBase * 4u + 2u + ib * 4u;         // 4 grid indices
    let sgByte = wordBase * 4u + 34u + ib * 4u;        // 4 inline sign bytes
    let qh = wbyte(wordBase * 4u + 66u + ib);
    let sc = wbyte(wordBase * 4u + 74u + ib);
    let db0 = d * (0.5 + f32(sc & 0xFu)) * 0.25;
    let db1 = d * (0.5 + f32(sc >> 4u)) * 0.25;
    let aBase = m * K + sb * 256u + ib * 32u;
    var contrib: f32 = 0.0;
    for (var l = 0u; l < 4u; l = l + 1u) {
      let dl = select(db1, db0, l < 2u);
      let gi = wbyte(qsByte + l) | ((qh << (8u - 2u * l)) & 0x300u);
      let sg = wbyte(sgByte + l);
      var dot: f32 = 0.0;
      for (var j = 0u; j < 8u; j = j + 1u) {
        let s = select(1.0, -1.0, ((sg >> j) & 1u) == 1u);
        dot = dot + A[aBase + l * 8u + j] * iq2s_grid_byte(gi, j) * s;
      }
      contrib = contrib + dl * dot;
    }
    let y = contrib - comp;
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
