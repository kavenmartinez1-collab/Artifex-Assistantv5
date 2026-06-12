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
