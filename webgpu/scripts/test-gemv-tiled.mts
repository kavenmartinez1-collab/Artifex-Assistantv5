/**
 * Tiled GEMV kernel validation (lever 2) — JS port of the word-level bit
 * extraction in matmul_gguf_q4_k_tiled / matmul_gguf_q6_k_tiled, checked
 * against the CPU reference dequant (gguf-dequant.ts, itself validated
 * bit-exact vs the official Python gguf package). Kernel-audit rule.
 *
 * Two gates per format:
 *  1. Per-element dequant values reconstructed through the TILED word math
 *     (masked-word nibble/2-bit extraction, word-aligned ql/qh reads, staged
 *     a_tile indexing) must equal the reference Float32Array EXACTLY.
 *  2. Full tiled GEMV (chunk loop + staging map + per-row reduce, f64 accum)
 *     must match a f64 dot of the reference dequant row within 1e-9 relative
 *     (only summation order differs).
 *
 * Run: npx tsx scripts/test-gemv-tiled.mts
 */

import {
  f16ToF32, dequantGGML, repackGGUFForGPU,
} from '../src/model/gguf-dequant';
import { GGML_TYPES } from '../src/model/gguf';

// ── deterministic RNG ──────────────────────────────────────────────────
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Random finite normal-range f16 bit pattern (exponent 8..22, |x|≲256). */
function randF16Bits(rng: () => number): number {
  const sign = rng() < 0.5 ? 0 : 0x8000;
  const exp = 8 + Math.floor(rng() * 15);   // biased exponent, well clear of inf/nan
  const mant = Math.floor(rng() * 1024);
  return sign | (exp << 10) | mant;
}

// ── WGSL helper ports (must mirror matmul_gguf.wgsl word math exactly) ──
function wbyte(W: Uint32Array, off: number): number {
  return (W[off >>> 2] >>> ((off & 3) * 8)) & 0xFF;
}
function unpack2x16(W: Uint32Array, word: number): [number, number] {
  return [f16ToF32(W[word] & 0xFFFF), f16ToF32(W[word] >>> 16)];
}
function scaleMinK4(W: Uint32Array, j: number, sbyte: number): [number, number] {
  if (j < 4) {
    return [wbyte(W, sbyte + j) & 63, wbyte(W, sbyte + j + 4) & 63];
  }
  const sc = (wbyte(W, sbyte + j + 4) & 0x0F) | ((wbyte(W, sbyte + j - 4) >>> 6) << 4);
  const mn = (wbyte(W, sbyte + j + 4) >>> 4) | ((wbyte(W, sbyte + j) >>> 6) << 4);
  return [sc, mn];
}

// Per-unit decode through the tiled kernels' word-level path. Returns the 32
// dequantized element values for unit u of row n, computed with the same
// f64 expression shapes as the CPU reference so gate 1 can compare exactly.
function decodeUnitQ4K(W: Uint32Array, n: number, nSB: number, u: number): Float64Array {
  const sb = (u / 8) | 0, sub = u % 8;
  const sbBase = (n * nSB + sb) * 36;
  const [d, dmin] = unpack2x16(W, sbBase);
  const [sc, mn] = scaleMinK4(W, sub, sbBase * 4 + 4);
  const d1 = d * sc, min1 = dmin * mn;
  const qsWord = sbBase + 4 + (sub >>> 1) * 8;
  const hi = (sub & 1) === 1;
  const out = new Float64Array(32);
  for (let w = 0; w < 8; w++) {
    const word = W[qsWord + w];
    const nib = hi ? (word >>> 4) & 0x0F0F0F0F : word & 0x0F0F0F0F;
    out[w * 4 + 0] = d1 * (nib & 0xFF) - min1;
    out[w * 4 + 1] = d1 * ((nib >>> 8) & 0xFF) - min1;
    out[w * 4 + 2] = d1 * ((nib >>> 16) & 0xFF) - min1;
    out[w * 4 + 3] = d1 * (nib >>> 24) - min1;
  }
  return out;
}

function decodeUnitQ5K(W: Uint32Array, n: number, nSB: number, u: number): Float64Array {
  const sb = (u / 8) | 0, sub = u % 8;
  const sbBase = (n * nSB + sb) * 44;
  const [d, dmin] = unpack2x16(W, sbBase);
  const [sc, mn] = scaleMinK4(W, sub, sbBase * 4 + 4);
  const d1 = d * sc, min1 = dmin * mn;
  const qsWord = sbBase + 12 + (sub >>> 1) * 8;
  const qhWord = sbBase + 4;
  const hi = (sub & 1) === 1;
  const out = new Float64Array(32);
  for (let w = 0; w < 8; w++) {
    const word = W[qsWord + w];
    const hw = W[qhWord + w];
    const nib = hi ? (word >>> 4) & 0x0F0F0F0F : word & 0x0F0F0F0F;
    const q = (nib | (((hw >>> sub) & 0x01010101) << 4)) >>> 0;
    out[w * 4 + 0] = d1 * (q & 0xFF) - min1;
    out[w * 4 + 1] = d1 * ((q >>> 8) & 0xFF) - min1;
    out[w * 4 + 2] = d1 * ((q >>> 16) & 0xFF) - min1;
    out[w * 4 + 3] = d1 * ((q >>> 24) & 0xFF) - min1;
  }
  return out;
}

function decodeUnitQ6K(W: Uint32Array, n: number, nSB: number, u: number): Float64Array {
  const sb = (u / 8) | 0, sub = u % 8;
  const half = sub >>> 2, s = sub & 3;
  const sbBase = (n * nSB + sb) * 53;
  const sbByte = sbBase * 4;
  const d = unpack2x16(W, sbBase + 52)[0];
  const qlWord = sbBase + half * 16 + ((s & 1) === 1 ? 8 : 0);
  const qhWord = sbBase + 32 + half * 8;
  const scByte = sbByte + 192 + half * 8 + 2 * s;
  const hShift = 2 * s;
  const lowNib = s < 2;
  const sc0 = (wbyte(W, scByte) << 24) >> 24;        // i8
  const sc1 = (wbyte(W, scByte + 1) << 24) >> 24;    // i8
  const out = new Float64Array(32);
  for (let w = 0; w < 8; w++) {
    const lw = W[qlWord + w];
    const hw = W[qhWord + w];
    const l4 = lowNib ? lw & 0x0F0F0F0F : (lw >>> 4) & 0x0F0F0F0F;
    const q = (l4 | (((hw >>> hShift) & 0x03030303) << 4)) >>> 0;
    const sc = w < 4 ? sc0 : sc1;
    out[w * 4 + 0] = d * sc * ((q & 0xFF) - 32);
    out[w * 4 + 1] = d * sc * (((q >>> 8) & 0xFF) - 32);
    out[w * 4 + 2] = d * sc * (((q >>> 16) & 0xFF) - 32);
    out[w * 4 + 3] = d * sc * (((q >>> 24) & 0xFF) - 32);
  }
  return out;
}

// Q2_K/Q3_K sub-blocks are 16 elems; a 32-elem tile unit covers the pair
// sub0=2p (even), sub1=2p+1 — same group/shift, contiguous qs words.

function decodeUnitQ2K(W: Uint32Array, n: number, nSB: number, u: number): Float64Array {
  const sb = (u / 8) | 0, pair = u % 8, sub0 = pair * 2;
  const wordBase = (n * nSB + sb) * 21;
  const byteBase = wordBase * 4;
  const [d, dmin] = unpack2x16(W, wordBase + 20);      // d @80, dmin @82
  const sc0 = wbyte(W, byteBase + sub0);
  const sc1 = wbyte(W, byteBase + sub0 + 1);
  const dl0 = d * (sc0 & 0x0F), ml0 = dmin * (sc0 >>> 4);
  const dl1 = d * (sc1 & 0x0F), ml1 = dmin * (sc1 >>> 4);
  const group = sub0 >= 8 ? 1 : 0;
  const shift = ((sub0 & 7) >>> 1) * 2;
  const qWord = wordBase + 4 + group * 8;              // qs @16 + group*32 B
  const out = new Float64Array(32);
  for (let w = 0; w < 8; w++) {
    const q = (W[qWord + w] >>> shift) & 0x03030303;
    const dl = w < 4 ? dl0 : dl1, ml = w < 4 ? ml0 : ml1;
    out[w * 4 + 0] = dl * (q & 0xFF) - ml;
    out[w * 4 + 1] = dl * ((q >>> 8) & 0xFF) - ml;
    out[w * 4 + 2] = dl * ((q >>> 16) & 0xFF) - ml;
    out[w * 4 + 3] = dl * (q >>> 24) - ml;
  }
  return out;
}

function decodeUnitQ3K(W: Uint32Array, n: number, nSB: number, u: number): Float64Array {
  const KM1 = 0x03030303, KM2 = 0x0f0f0f0f;
  const sb = (u / 8) | 0, pair = u % 8, sub0 = pair * 2;
  const wordBase = (n * nSB + sb) * 28;
  const dAll = unpack2x16(W, wordBase + 27)[0];        // d @108
  const s0 = W[wordBase + 24], s1 = W[wordBase + 25], tmp = W[wordBase + 26];
  const widx = sub0 >>> 2;
  let auxWord: number;
  if (widx === 0) auxWord = ((s0 & KM2) | ((((tmp >>> 0) & KM1) << 4) >>> 0)) >>> 0;
  else if (widx === 1) auxWord = ((s1 & KM2) | ((((tmp >>> 2) & KM1) << 4) >>> 0)) >>> 0;
  else if (widx === 2) auxWord = (((s0 >>> 4) & KM2) | ((((tmp >>> 4) & KM1) << 4) >>> 0)) >>> 0;
  else auxWord = (((s1 >>> 4) & KM2) | ((((tmp >>> 6) & KM1) << 4) >>> 0)) >>> 0;
  const scB0 = (auxWord >>> ((sub0 & 3) * 8)) & 0xFF;
  const scB1 = (auxWord >>> (((sub0 & 3) + 1) * 8)) & 0xFF;
  const dl0 = dAll * (scB0 - 32);
  const dl1 = dAll * (scB1 - 32);
  const group = sub0 >= 8 ? 1 : 0;
  const j = (sub0 & 7) >>> 1;
  const shift = j * 2;
  const hbitpos = group * 4 + j;
  const qWord = wordBase + 8 + group * 8;              // qs @32 + group*32 B
  const hWord = wordBase;                              // hmask @0
  const out = new Float64Array(32);
  for (let w = 0; w < 8; w++) {
    const q3 = (W[qWord + w] >>> shift) & 0x03030303;
    const hb = (W[hWord + w] >>> hbitpos) & 0x01010101;
    const q = (q3 + ((hb << 2) >>> 0)) >>> 0;          // 0..7/byte, no carry
    const dl = w < 4 ? dl0 : dl1;
    out[w * 4 + 0] = dl * ((q & 0xFF) - 4);
    out[w * 4 + 1] = dl * (((q >>> 8) & 0xFF) - 4);
    out[w * 4 + 2] = dl * (((q >>> 16) & 0xFF) - 4);
    out[w * 4 + 3] = dl * (((q >>> 24) & 0xFF) - 4);
  }
  return out;
}

// ── Q2_K unit-contiguous repack ports (lever C, repack_q2k.wgsl +
// matmul_gguf_q2_k_tiled_r) ─────────────────────────────────────────────
// Repack permutation: unit p = u%8 of a superblock decodes group g = p>>2
// at shift plane t = p&3 (see decodeUnitQ2K: group = sub0>=8, shift =
// ((sub0&7)>>1)*2 with sub0 = 2p). The repack gives each unit 2 contiguous
// words at offset 4 + 2p: each source word's plane t (4×2-bit spread across
// 4 bytes) is compacted to 1 byte via b|(b>>6)|(b>>12)... so output byte j
// = elements of source word j, e0|e1<<2|e2<<4|e3<<6. Scales (words 0-3)
// and d/dmin (word 20) unchanged; stride stays 21 (zero VRAM growth).
// Mirrors repack_q2k.wgsl exactly (thread-per-superblock).
function repackQ2K(W: Uint32Array): Uint32Array {
  if (W.length % 21 !== 0) throw new Error('Q2_K buffer not a multiple of 21 words');
  const out = new Uint32Array(W.length);
  const nBlocks = W.length / 21;
  for (let b = 0; b < nBlocks; b++) {
    const base = b * 21;
    for (let i = 0; i < 4; i++) out[base + i] = W[base + i];
    out[base + 20] = W[base + 20];
    for (let p = 0; p < 8; p++) {
      const g = p >>> 2, t = p & 3;
      const srcBase = base + 4 + g * 8;
      for (let m = 0; m < 2; m++) {
        let word = 0;
        for (let j = 0; j < 4; j++) {
          const bb = (W[srcBase + m * 4 + j] >>> (2 * t)) & 0x03030303;
          const y = (bb | (bb >>> 6)) >>> 0;
          const z = (y | (y >>> 12)) >>> 0;
          word |= (z & 0xFF) << (j * 8);
        }
        out[base + 4 + p * 2 + m] = word >>> 0;
      }
    }
  }
  return out;
}

// Decode port of matmul_gguf_q2_k_tiled_r: reads the repacked layout. Unit
// element l = w*4+k sits at bit (w&3)*8 + 2k of word (w<4 ? 0 : 1) of the
// unit's pair. Scale math, dl/ml w<4 split, and expression shapes are
// byte-identical to decodeUnitQ2K → must match it EXACTLY on repacked data.
function decodeUnitQ2KR(W: Uint32Array, n: number, nSB: number, u: number): Float64Array {
  const sb = (u / 8) | 0, pair = u % 8, sub0 = pair * 2;
  const wordBase = (n * nSB + sb) * 21;
  const byteBase = wordBase * 4;
  const [d, dmin] = unpack2x16(W, wordBase + 20);
  const sc0 = wbyte(W, byteBase + sub0);
  const sc1 = wbyte(W, byteBase + sub0 + 1);
  const dl0 = d * (sc0 & 0x0F), ml0 = dmin * (sc0 >>> 4);
  const dl1 = d * (sc1 & 0x0F), ml1 = dmin * (sc1 >>> 4);
  const qBase = wordBase + 4 + pair * 2;
  const w0 = W[qBase], w1 = W[qBase + 1];
  const out = new Float64Array(32);
  for (let w = 0; w < 8; w++) {
    const src = w < 4 ? w0 : w1;
    const off = (w & 3) * 8;
    const dl = w < 4 ? dl0 : dl1, ml = w < 4 ? ml0 : ml1;
    out[w * 4 + 0] = dl * ((src >>> off) & 3) - ml;
    out[w * 4 + 1] = dl * ((src >>> (off + 2)) & 3) - ml;
    out[w * 4 + 2] = dl * ((src >>> (off + 4)) & 3) - ml;
    out[w * 4 + 3] = dl * ((src >>> (off + 6)) & 3) - ml;
  }
  return out;
}

// ── vec4 W-load decode ports (lever 4 phase 2, *_tiled_v4) ──────────────
// Same byte math as the tiled decoders, but every W access goes through a
// vec4 view: v4() throws on any non-multiple-of-4 word index (alignment
// gate), and Q4_K/Q5_K scales come from scaleMinK4W — a rewrite of
// scaleMinK4 taking the three scale words as values (the WGSL v4 entries
// cannot call wbyte(), which references the scalar W binding). Gate 4
// asserts decodeV4 == decode exactly for every (n, u).

function v4(W: Uint32Array, w: number): [number, number, number, number] {
  if (w % 4 !== 0) throw new Error(`unaligned vec4 word index ${w}`);
  return [W[w], W[w + 1], W[w + 2], W[w + 3]];
}

function scaleMinK4W(j: number, w1: number, w2: number, w3: number): [number, number] {
  if (j < 4) {
    return [(w1 >>> (j * 8)) & 63, (w2 >>> (j * 8)) & 63];
  }
  const b1 = (w1 >>> ((j - 4) * 8)) & 0xFF;
  const b2 = (w2 >>> ((j - 4) * 8)) & 0xFF;
  const b3 = (w3 >>> ((j - 4) * 8)) & 0xFF;
  return [(b3 & 0x0F) | ((b1 >>> 6) << 4), (b3 >>> 4) | ((b2 >>> 6) << 4)];
}

function decodeUnitQ4Kv4(W: Uint32Array, n: number, nSB: number, u: number): Float64Array {
  const sb = (u / 8) | 0, sub = u % 8;
  const sbBase = (n * nSB + sb) * 36;
  const [h0, h1, h2, h3] = v4(W, sbBase);          // d/dmin | scales[12]
  const d = f16ToF32(h0 & 0xFFFF), dmin = f16ToF32(h0 >>> 16);
  const [sc, mn] = scaleMinK4W(sub, h1, h2, h3);
  const d1 = d * sc, min1 = dmin * mn;
  const qsWord = sbBase + 4 + (sub >>> 1) * 8;
  const qs = [...v4(W, qsWord), ...v4(W, qsWord + 4)];
  const hi = (sub & 1) === 1;
  const out = new Float64Array(32);
  for (let w = 0; w < 8; w++) {
    const word = qs[w];
    const nib = hi ? (word >>> 4) & 0x0F0F0F0F : word & 0x0F0F0F0F;
    out[w * 4 + 0] = d1 * (nib & 0xFF) - min1;
    out[w * 4 + 1] = d1 * ((nib >>> 8) & 0xFF) - min1;
    out[w * 4 + 2] = d1 * ((nib >>> 16) & 0xFF) - min1;
    out[w * 4 + 3] = d1 * (nib >>> 24) - min1;
  }
  return out;
}

function decodeUnitQ5Kv4(W: Uint32Array, n: number, nSB: number, u: number): Float64Array {
  const sb = (u / 8) | 0, sub = u % 8;
  const sbBase = (n * nSB + sb) * 44;
  const [h0, h1, h2, h3] = v4(W, sbBase);          // d/dmin | scales[12]
  const d = f16ToF32(h0 & 0xFFFF), dmin = f16ToF32(h0 >>> 16);
  const [sc, mn] = scaleMinK4W(sub, h1, h2, h3);
  const d1 = d * sc, min1 = dmin * mn;
  const qsWord = sbBase + 12 + (sub >>> 1) * 8;
  const qs = [...v4(W, qsWord), ...v4(W, qsWord + 4)];
  const qh = [...v4(W, sbBase + 4), ...v4(W, sbBase + 8)];
  const hi = (sub & 1) === 1;
  const out = new Float64Array(32);
  for (let w = 0; w < 8; w++) {
    const nib = hi ? (qs[w] >>> 4) & 0x0F0F0F0F : qs[w] & 0x0F0F0F0F;
    const q = (nib | (((qh[w] >>> sub) & 0x01010101) << 4)) >>> 0;
    out[w * 4 + 0] = d1 * (q & 0xFF) - min1;
    out[w * 4 + 1] = d1 * ((q >>> 8) & 0xFF) - min1;
    out[w * 4 + 2] = d1 * ((q >>> 16) & 0xFF) - min1;
    out[w * 4 + 3] = d1 * ((q >>> 24) & 0xFF) - min1;
  }
  return out;
}

function decodeUnitQ3Kv4(W: Uint32Array, n: number, nSB: number, u: number): Float64Array {
  const KM1 = 0x03030303, KM2 = 0x0f0f0f0f;
  const sb = (u / 8) | 0, pair = u % 8, sub0 = pair * 2;
  const wordBase = (n * nSB + sb) * 28;
  const [s0, s1, tmp, dw] = v4(W, wordBase + 24);  // scales[12] | d
  const dAll = f16ToF32(dw & 0xFFFF);
  const widx = sub0 >>> 2;
  let auxWord: number;
  if (widx === 0) auxWord = ((s0 & KM2) | ((((tmp >>> 0) & KM1) << 4) >>> 0)) >>> 0;
  else if (widx === 1) auxWord = ((s1 & KM2) | ((((tmp >>> 2) & KM1) << 4) >>> 0)) >>> 0;
  else if (widx === 2) auxWord = (((s0 >>> 4) & KM2) | ((((tmp >>> 4) & KM1) << 4) >>> 0)) >>> 0;
  else auxWord = (((s1 >>> 4) & KM2) | ((((tmp >>> 6) & KM1) << 4) >>> 0)) >>> 0;
  const scB0 = (auxWord >>> ((sub0 & 3) * 8)) & 0xFF;
  const scB1 = (auxWord >>> (((sub0 & 3) + 1) * 8)) & 0xFF;
  const dl0 = dAll * (scB0 - 32);
  const dl1 = dAll * (scB1 - 32);
  const group = sub0 >= 8 ? 1 : 0;
  const j = (sub0 & 7) >>> 1;
  const shift = j * 2;
  const hbitpos = group * 4 + j;
  const qWord = wordBase + 8 + group * 8;          // qs @32 + group*32 B
  const qs = [...v4(W, qWord), ...v4(W, qWord + 4)];
  const hm = [...v4(W, wordBase), ...v4(W, wordBase + 4)];
  const out = new Float64Array(32);
  for (let w = 0; w < 8; w++) {
    const q3 = (qs[w] >>> shift) & 0x03030303;
    const hb = (hm[w] >>> hbitpos) & 0x01010101;
    const q = (q3 + ((hb << 2) >>> 0)) >>> 0;
    const dl = w < 4 ? dl0 : dl1;
    out[w * 4 + 0] = dl * ((q & 0xFF) - 4);
    out[w * 4 + 1] = dl * (((q >>> 8) & 0xFF) - 4);
    out[w * 4 + 2] = dl * (((q >>> 16) & 0xFF) - 4);
    out[w * 4 + 3] = dl * (((q >>> 24) & 0xFF) - 4);
  }
  return out;
}

// Full tiled GEMV simulation for one output row n: chunk loop, a_tile staging
// map, lane = unit ownership — exactly the kernel's index structure, f64 accum.
function tiledGemvRow(
  W: Uint32Array, A: Float32Array, n: number, K: number,
  decode: (W: Uint32Array, n: number, nSB: number, u: number) => Float64Array,
  TPR: number,
): number {
  const nSB = K / 256;
  const nUnits = nSB * 8;
  const nChunks = Math.ceil(nUnits / TPR);
  const laneAcc = new Float64Array(TPR);
  const aTile = new Float64Array(TPR * 32);     // vec4 staging flattened
  for (let c = 0; c < nChunks; c++) {
    // tile_stage: vec4 i covers elems elemBase + i*4 .. +3, zero past K
    const elemBase = c * TPR * 32;
    for (let i = 0; i < TPR * 8; i++) {
      const e = elemBase + i * 4;
      for (let j = 0; j < 4; j++) aTile[i * 4 + j] = e < K ? A[e + j] : 0;
    }
    for (let lane = 0; lane < TPR; lane++) {
      const u = c * TPR + lane;
      if (u >= nUnits) continue;
      const vals = decode(W, n, nSB, u);
      let dq = 0;
      // fround: compare against the f32 reference values so only summation
      // order differs between the two dots (gate 1 already proved bit-equality)
      for (let l = 0; l < 32; l++) dq += aTile[lane * 32 + l] * Math.fround(vals[l]);
      laneAcc[lane] += dq;
    }
  }
  // per-row tree reduce (order matters only at f32; here just sum)
  let sum = 0;
  for (let lane = 0; lane < TPR; lane++) sum += laneAcc[lane];
  return sum;
}

// No-stage GEMV (lever 4, *_tiled_ns): identical chunk loop and lane = unit
// ownership, but each lane reads A directly (A4 global vec4 loads) instead of
// the staged a_tile. aTile[lane*32+l] == A[u*32+l] for every valid unit
// (units never cross K), so the accumulation expressions are identical →
// must equal tiledGemvRow EXACTLY (gate 3).
function nsGemvRow(
  W: Uint32Array, A: Float32Array, n: number, K: number,
  decode: (W: Uint32Array, n: number, nSB: number, u: number) => Float64Array,
  TPR: number,
): number {
  const nSB = K / 256;
  const nUnits = nSB * 8;
  const nChunks = Math.ceil(nUnits / TPR);
  const laneAcc = new Float64Array(TPR);
  for (let c = 0; c < nChunks; c++) {
    for (let lane = 0; lane < TPR; lane++) {
      const u = c * TPR + lane;
      if (u >= nUnits) continue;
      const vals = decode(W, n, nSB, u);
      let dq = 0;
      for (let l = 0; l < 32; l++) dq += A[u * 32 + l] * Math.fround(vals[l]);
      laneAcc[lane] += dq;
    }
  }
  let sum = 0;
  for (let lane = 0; lane < TPR; lane++) sum += laneAcc[lane];
  return sum;
}

// ── test driver ────────────────────────────────────────────────────────
let failures = 0;

function check(label: string, ok: boolean, detail = '') {
  if (!ok) {
    failures++;
    console.log(`  FAIL ${label} ${detail}`);
  }
}

function runFormat(
  name: string, ggmlType: number, rawBytesPerSB: number,
  fixupSB: (raw: Uint8Array, base: number, rng: () => number) => void,
  decode: (W: Uint32Array, n: number, nSB: number, u: number) => Float64Array,
  decodeV4?: (W: Uint32Array, n: number, nSB: number, u: number) => Float64Array,
) {
  for (const K of [256, 1280, 4096]) {
    for (const TPR of [16, 32]) {           // TWG/TN: 128/8 and 256/8
      const rng = mulberry32(0xC0FFEE ^ K ^ (TPR << 8) ^ ggmlType);
      const N = 11;                          // not a multiple of TN=8
      const nSB = K / 256;
      const raw = new Uint8Array(N * nSB * rawBytesPerSB);
      for (let i = 0; i < raw.length; i++) raw[i] = Math.floor(rng() * 256);
      for (let b = 0; b < N * nSB; b++) fixupSB(raw, b * rawBytesPerSB, rng);

      const ref = dequantGGML(ggmlType, raw, N * K);     // row-major [N, K]
      const W = repackGGUFForGPU(ggmlType, raw, N * K);
      const A = new Float32Array(K);
      for (let i = 0; i < K; i++) A[i] = rng() * 2 - 1;

      // Gate 1: per-element bit-exact extraction
      let elemOK = true;
      for (let n = 0; n < N && elemOK; n++) {
        for (let u = 0; u < nSB * 8 && elemOK; u++) {
          const vals = decode(W, n, nSB, u);
          for (let l = 0; l < 32; l++) {
            const got = Math.fround(vals[l]);
            const want = ref[n * K + u * 32 + l];
            if (got !== want && !(Number.isNaN(got) && Number.isNaN(want))) {
              check(`${name} K=${K} elem`, false,
                `n=${n} u=${u} l=${l}: got ${got} want ${want}`);
              elemOK = false;
              break;
            }
          }
        }
      }
      if (elemOK) console.log(`  ok   ${name} K=${K} TPR=${TPR}: element extraction bit-exact (${N * K} elems)`);

      // Gate 2: tiled GEMV vs f64 reference dot
      let maxRel = 0;
      for (let n = 0; n < N; n++) {
        let refDot = 0;
        for (let i = 0; i < K; i++) refDot += A[i] * ref[n * K + i];
        const got = tiledGemvRow(W, A, n, K, decode, TPR);
        const rel = Math.abs(got - refDot) / Math.max(1e-30, Math.abs(refDot));
        maxRel = Math.max(maxRel, rel);
        check(`${name} K=${K} TPR=${TPR} gemv`, rel < 1e-9, `n=${n} rel=${rel}`);
      }
      console.log(`  ok   ${name} K=${K} TPR=${TPR}: gemv maxRel=${maxRel.toExponential(2)}`);

      // Gate 3 (lever 4): no-stage GEMV must equal the staged tiled GEMV
      // EXACTLY — same expressions, only the A read path differs.
      let nsOK = true;
      for (let n = 0; n < N; n++) {
        const staged = tiledGemvRow(W, A, n, K, decode, TPR);
        const ns = nsGemvRow(W, A, n, K, decode, TPR);
        if (ns !== staged) {
          check(`${name} K=${K} TPR=${TPR} ns`, false, `n=${n}: ${ns} != ${staged}`);
          nsOK = false;
        }
      }
      if (nsOK) console.log(`  ok   ${name} K=${K} TPR=${TPR}: no-stage == staged (exact)`);

      // Gate 4 (lever 4 phase 2): vec4 W-load decode must equal the scalar
      // decode EXACTLY for every (n, u) — v4() also asserts every vec4 word
      // index is a multiple of 4 (alignment gate).
      if (decodeV4) {
        let v4OK = true;
        for (let n = 0; n < N && v4OK; n++) {
          for (let u = 0; u < nSB * 8 && v4OK; u++) {
            const a = decode(W, n, nSB, u);
            const b = decodeV4(W, n, nSB, u);
            for (let l = 0; l < 32; l++) {
              if (a[l] !== b[l] && !(Number.isNaN(a[l]) && Number.isNaN(b[l]))) {
                check(`${name} K=${K} v4`, false, `n=${n} u=${u} l=${l}: ${b[l]} != ${a[l]}`);
                v4OK = false;
                break;
              }
            }
          }
        }
        if (v4OK) console.log(`  ok   ${name} K=${K} TPR=${TPR}: vec4 W decode == scalar (exact, aligned)`);
      }
    }
  }
}

console.log('Q4_K tiled:');
runFormat('Q4_K', GGML_TYPES.Q4_K, 144, (raw, base, rng) => {
  // d @0, dmin @2 — controlled finite f16
  const dv = new DataView(raw.buffer, raw.byteOffset);
  dv.setUint16(base, randF16Bits(rng), true);
  dv.setUint16(base + 2, randF16Bits(rng), true);
}, decodeUnitQ4K, decodeUnitQ4Kv4);

console.log('Q5_K tiled:');
runFormat('Q5_K', GGML_TYPES.Q5_K, 176, (raw, base, rng) => {
  // d @0, dmin @2 — controlled finite f16
  const dv = new DataView(raw.buffer, raw.byteOffset);
  dv.setUint16(base, randF16Bits(rng), true);
  dv.setUint16(base + 2, randF16Bits(rng), true);
}, decodeUnitQ5K, decodeUnitQ5Kv4);

console.log('Q6_K tiled:');
runFormat('Q6_K', GGML_TYPES.Q6_K, 210, (raw, base, rng) => {
  // d @208 — controlled finite f16 (ql/qh/scales stay random)
  const dv = new DataView(raw.buffer, raw.byteOffset);
  dv.setUint16(base + 208, randF16Bits(rng), true);
}, decodeUnitQ6K);

console.log('Q2_K tiled:');
runFormat('Q2_K', GGML_TYPES.Q2_K, 84, (raw, base, rng) => {
  // d @80, dmin @82 — controlled finite f16
  const dv = new DataView(raw.buffer, raw.byteOffset);
  dv.setUint16(base + 80, randF16Bits(rng), true);
  dv.setUint16(base + 82, randF16Bits(rng), true);
}, decodeUnitQ2K);

console.log('Q3_K tiled:');
runFormat('Q3_K', GGML_TYPES.Q3_K, 110, (raw, base, rng) => {
  // d @108 — controlled finite f16 (hmask/qs/scales stay random)
  const dv = new DataView(raw.buffer, raw.byteOffset);
  dv.setUint16(base + 108, randF16Bits(rng), true);
}, decodeUnitQ3K, decodeUnitQ3Kv4);

// Gate 5 (lever C): Q2_K repack + _r decode. (a) per-unit decode of the
// repacked buffer must equal the original tiled decode EXACTLY for every
// (n, u, l); (b) full tiled GEMV on repacked data must equal the original
// tiled GEMV EXACTLY (identical f32 values + identical accumulation order).
console.log('Q2_K repack (_r):');
for (const K of [256, 1280, 4096]) {
  for (const TPR of [16, 32]) {
    const rng = mulberry32(0xC0FFEE ^ K ^ (TPR << 8) ^ GGML_TYPES.Q2_K ^ 0x52);
    const N = 11;
    const nSB = K / 256;
    const raw = new Uint8Array(N * nSB * 84);
    for (let i = 0; i < raw.length; i++) raw[i] = Math.floor(rng() * 256);
    const dv = new DataView(raw.buffer, raw.byteOffset);
    for (let b = 0; b < N * nSB; b++) {
      dv.setUint16(b * 84 + 80, randF16Bits(rng), true);
      dv.setUint16(b * 84 + 82, randF16Bits(rng), true);
    }
    const W = repackGGUFForGPU(GGML_TYPES.Q2_K, raw, N * K);
    const Wr = repackQ2K(W);
    const A = new Float32Array(K);
    for (let i = 0; i < K; i++) A[i] = rng() * 2 - 1;

    let rOK = true;
    for (let n = 0; n < N && rOK; n++) {
      for (let u = 0; u < nSB * 8 && rOK; u++) {
        const a = decodeUnitQ2K(W, n, nSB, u);
        const b = decodeUnitQ2KR(Wr, n, nSB, u);
        for (let l = 0; l < 32; l++) {
          if (a[l] !== b[l] && !(Number.isNaN(a[l]) && Number.isNaN(b[l]))) {
            check(`Q2_K_r K=${K} elem`, false, `n=${n} u=${u} l=${l}: ${b[l]} != ${a[l]}`);
            rOK = false;
            break;
          }
        }
      }
    }
    if (rOK) console.log(`  ok   Q2_K_r K=${K} TPR=${TPR}: repacked decode == tiled decode (exact)`);

    let gOK = true;
    for (let n = 0; n < N; n++) {
      const orig = tiledGemvRow(W, A, n, K, decodeUnitQ2K, TPR);
      const rep = tiledGemvRow(Wr, A, n, K, decodeUnitQ2KR, TPR);
      if (rep !== orig) {
        check(`Q2_K_r K=${K} TPR=${TPR} gemv`, false, `n=${n}: ${rep} != ${orig}`);
        gOK = false;
      }
    }
    if (gOK) console.log(`  ok   Q2_K_r K=${K} TPR=${TPR}: repacked gemv == tiled gemv (exact)`);
  }
}

// Gate 6 (ILP probe): _r2 = _r with 2 units/lane and a single scale-WORD
// load (scw) instead of two wbyte() calls. (a) the scw extraction
// `(scw >> ((sub&3)*8)) & 0xFF` must equal wbyte(byteBase+sub0) for every
// unit (sub0 even → sc1 shares the word); (b) the 2-units-per-lane chunk
// order (ua = c*2*TPR + lane, ub = ua + TPR) visits per lane the SAME unit
// sequence as _r (lane, lane+TPR, lane+2*TPR, ...) so the GEMV must equal
// tiledGemvRow on repacked data EXACTLY.
function decodeUnitQ2KR2(W: Uint32Array, n: number, nSB: number, u: number): Float64Array {
  const sb = (u / 8) | 0, pair = u % 8, sub = pair * 2;
  const base = (n * nSB + sb) * 21;
  const [d, dmin] = unpack2x16(W, base + 20);
  const scw = W[base + (sub >>> 2)];
  const sc0 = (scw >>> ((sub & 3) * 8)) & 0xFF;
  const sc1 = (scw >>> (((sub & 3) + 1) * 8)) & 0xFF;
  const dl0 = d * (sc0 & 0x0F), ml0 = dmin * (sc0 >>> 4);
  const dl1 = d * (sc1 & 0x0F), ml1 = dmin * (sc1 >>> 4);
  const qBase = base + 4 + pair * 2;
  const w0 = W[qBase], w1 = W[qBase + 1];
  const out = new Float64Array(32);
  for (let w = 0; w < 8; w++) {
    const src = w < 4 ? w0 : w1;
    const off = (w & 3) * 8;
    const dl = w < 4 ? dl0 : dl1, ml = w < 4 ? ml0 : ml1;
    out[w * 4 + 0] = dl * ((src >>> off) & 3) - ml;
    out[w * 4 + 1] = dl * ((src >>> (off + 2)) & 3) - ml;
    out[w * 4 + 2] = dl * ((src >>> (off + 4)) & 3) - ml;
    out[w * 4 + 3] = dl * ((src >>> (off + 6)) & 3) - ml;
  }
  return out;
}

/** Port of matmul_gguf_q2_k_tiled_r2's chunk loop: UPC = 2*TPR units per
 *  chunk, each lane handles ua then ub = ua + TPR within the chunk. */
function r2GemvRow(W: Uint32Array, A: Float32Array, n: number, K: number, TPR: number): number {
  const nSB = K / 256;
  const nUnits = nSB * 8;
  const UPC = TPR * 2;
  const nChunks = Math.ceil(nUnits / UPC);
  const laneAcc = new Float64Array(TPR);
  for (let c = 0; c < nChunks; c++) {
    for (let lane = 0; lane < TPR; lane++) {
      for (const u of [c * UPC + lane, c * UPC + lane + TPR]) {
        if (u >= nUnits) continue;
        const vals = decodeUnitQ2KR2(W, n, nSB, u);
        let dq = 0;
        for (let l = 0; l < 32; l++) dq += A[u * 32 + l] * Math.fround(vals[l]);
        laneAcc[lane] += dq;
      }
    }
  }
  let sum = 0;
  for (let lane = 0; lane < TPR; lane++) sum += laneAcc[lane];
  return sum;
}

console.log('Q2_K repack 2-units/lane (_r2):');
for (const K of [256, 1280, 4096, 5120]) {
  for (const TPR of [16, 32]) {
    const rng = mulberry32(0xD00D ^ K ^ (TPR << 8));
    const N = 11;
    const nSB = K / 256;
    const raw = new Uint8Array(N * nSB * 84);
    for (let i = 0; i < raw.length; i++) raw[i] = Math.floor(rng() * 256);
    const dv = new DataView(raw.buffer, raw.byteOffset);
    for (let b = 0; b < N * nSB; b++) {
      dv.setUint16(b * 84 + 80, randF16Bits(rng), true);
      dv.setUint16(b * 84 + 82, randF16Bits(rng), true);
    }
    const W = repackGGUFForGPU(GGML_TYPES.Q2_K, raw, N * K);
    const Wr = repackQ2K(W);
    const A = new Float32Array(K);
    for (let i = 0; i < K; i++) A[i] = rng() * 2 - 1;

    let dOK = true;
    for (let n = 0; n < N && dOK; n++) {
      for (let u = 0; u < nSB * 8 && dOK; u++) {
        const a = decodeUnitQ2KR(Wr, n, nSB, u);
        const b = decodeUnitQ2KR2(Wr, n, nSB, u);
        for (let l = 0; l < 32; l++) {
          if (a[l] !== b[l] && !(Number.isNaN(a[l]) && Number.isNaN(b[l]))) {
            check(`Q2_K_r2 K=${K} elem`, false, `n=${n} u=${u} l=${l}: ${b[l]} != ${a[l]}`);
            dOK = false;
            break;
          }
        }
      }
    }
    if (dOK) console.log(`  ok   Q2_K_r2 K=${K} TPR=${TPR}: scw decode == _r decode (exact)`);

    let gOK = true;
    for (let n = 0; n < N; n++) {
      const orig = tiledGemvRow(Wr, A, n, K, decodeUnitQ2KR, TPR);
      const r2 = r2GemvRow(Wr, A, n, K, TPR);
      if (r2 !== orig) {
        check(`Q2_K_r2 K=${K} TPR=${TPR} gemv`, false, `n=${n}: ${r2} != ${orig}`);
        gOK = false;
      }
    }
    if (gOK) console.log(`  ok   Q2_K_r2 K=${K} TPR=${TPR}: 2-unit/lane gemv == _r gemv (exact)`);
  }
}

// Gate 7 (Phase C4): Q2_K M-reuse GEMM (matmul_gguf_q2_k_tiled_r_gemm). The
// kernel decodes each unit's raw 2-bit quants ONCE and reuses them across M
// activation rows, accumulating per-row with the SAME deferred-scale split as
// matmul_gguf_q2_k_tiled_r (dq0/as0/dq1/as1 then dl0*dq0 - ml0*as0 + ...).
// Ports both the _r single-row kernel and the GEMM kernel (deferred scale,
// matching the WGSL exactly — NOT the folded-scale decodeUnitQ2KR used by
// gates 1/5). Asserts: (a) _r single-row matches the f64 reference dot within
// 1e-9 (anchors the port to ground truth); (b) GEMM column row mi equals the
// _r single-row on that row EXACTLY for every mi (M-batching is per-row
// independent — the only thing C4 changes is decode REUSE, not arithmetic).

/** Per-unit decode of the repacked layout returning the deferred scales and
 *  the 8 raw 2-bit quad groups — mirrors the _r/GEMM kernel decode (no fold). */
function unitRawQ2KR(W: Uint32Array, n: number, nSB: number, u: number) {
  const sb = (u / 8) | 0, pair = u % 8, sub0 = pair * 2;
  const wordBase = (n * nSB + sb) * 21;
  const byteBase = wordBase * 4;
  const [d, dmin] = unpack2x16(W, wordBase + 20);
  const sc0 = wbyte(W, byteBase + sub0);
  const sc1 = wbyte(W, byteBase + sub0 + 1);
  const dl0 = d * (sc0 & 0x0F), ml0 = dmin * (sc0 >>> 4);
  const dl1 = d * (sc1 & 0x0F), ml1 = dmin * (sc1 >>> 4);
  const qBase = wordBase + 4 + pair * 2;
  const w0 = W[qBase], w1 = W[qBase + 1];
  const q: number[][] = [];
  for (let w = 0; w < 8; w++) {
    const src = w < 4 ? w0 : w1;
    const off = (w & 3) * 8;
    q.push([(src >>> off) & 3, (src >>> (off + 2)) & 3, (src >>> (off + 4)) & 3, (src >>> (off + 6)) & 3]);
  }
  return { dl0, ml0, dl1, ml1, q };
}

/** Port of matmul_gguf_q2_k_tiled_r (deferred-scale dq0/as0/dq1/as1 split). */
function rKernelRow(W: Uint32Array, A: Float32Array, n: number, K: number, TPR: number): number {
  const nSB = K / 256, nUnits = nSB * 8, nChunks = Math.ceil(nUnits / TPR);
  const laneAcc = new Float64Array(TPR);
  for (let c = 0; c < nChunks; c++) {
    for (let lane = 0; lane < TPR; lane++) {
      const u = c * TPR + lane;
      if (u >= nUnits) continue;
      const { dl0, ml0, dl1, ml1, q } = unitRawQ2KR(W, n, nSB, u);
      let dq0 = 0, as0 = 0, dq1 = 0, as1 = 0;
      for (let w = 0; w < 8; w++) {
        const b = u * 32 + w * 4;
        const a0 = A[b], a1 = A[b + 1], a2 = A[b + 2], a3 = A[b + 3];
        const dot = a0 * q[w][0] + a1 * q[w][1] + a2 * q[w][2] + a3 * q[w][3];
        const asum = a0 + a1 + a2 + a3;
        if (w < 4) { dq0 += dot; as0 += asum; } else { dq1 += dot; as1 += asum; }
      }
      laneAcc[lane] += (dl0 * dq0 - ml0 * as0) + (dl1 * dq1 - ml1 * as1);
    }
  }
  let sum = 0;
  for (let lane = 0; lane < TPR; lane++) sum += laneAcc[lane];
  return sum;
}

/** Port of matmul_gguf_q2_k_tiled_r_gemm: decode each unit once, accumulate
 *  per-m into acc[mi]. Arows is M activation rows of length K (flat). Returns
 *  the M column outputs for output row n. */
function gemmCol(W: Uint32Array, Arows: Float32Array[], n: number, K: number, TPR: number): Float64Array {
  const M = Arows.length;
  const nSB = K / 256, nUnits = nSB * 8, nChunks = Math.ceil(nUnits / TPR);
  const laneAcc = new Float64Array(TPR * M);
  for (let c = 0; c < nChunks; c++) {
    for (let lane = 0; lane < TPR; lane++) {
      const u = c * TPR + lane;
      if (u >= nUnits) continue;
      const { dl0, ml0, dl1, ml1, q } = unitRawQ2KR(W, n, nSB, u);
      for (let mi = 0; mi < M; mi++) {
        const A = Arows[mi];
        let dq0 = 0, as0 = 0, dq1 = 0, as1 = 0;
        for (let w = 0; w < 8; w++) {
          const b = u * 32 + w * 4;
          const a0 = A[b], a1 = A[b + 1], a2 = A[b + 2], a3 = A[b + 3];
          const dot = a0 * q[w][0] + a1 * q[w][1] + a2 * q[w][2] + a3 * q[w][3];
          const asum = a0 + a1 + a2 + a3;
          if (w < 4) { dq0 += dot; as0 += asum; } else { dq1 += dot; as1 += asum; }
        }
        laneAcc[mi * TPR + lane] += (dl0 * dq0 - ml0 * as0) + (dl1 * dq1 - ml1 * as1);
      }
    }
  }
  const out = new Float64Array(M);
  for (let mi = 0; mi < M; mi++) {
    let sum = 0;
    for (let lane = 0; lane < TPR; lane++) sum += laneAcc[mi * TPR + lane];
    out[mi] = sum;
  }
  return out;
}

console.log('Q2_K M-reuse GEMM (_r_gemm, Phase C4):');
for (const K of [256, 1280, 4096, 5120]) {
  for (const TPR of [16, 32]) {
    for (const M of [1, 2, 5, 8]) {
      const rng = mulberry32(0xC4 ^ K ^ (TPR << 8) ^ (M << 16));
      const N = 11;
      const nSB = K / 256;
      const raw = new Uint8Array(N * nSB * 84);
      for (let i = 0; i < raw.length; i++) raw[i] = Math.floor(rng() * 256);
      const dv = new DataView(raw.buffer, raw.byteOffset);
      for (let b = 0; b < N * nSB; b++) {
        dv.setUint16(b * 84 + 80, randF16Bits(rng), true);
        dv.setUint16(b * 84 + 82, randF16Bits(rng), true);
      }
      const W = repackGGUFForGPU(GGML_TYPES.Q2_K, raw, N * K);
      const Wr = repackQ2K(W);
      const ref = dequantGGML(GGML_TYPES.Q2_K, raw, N * K);
      const Arows: Float32Array[] = [];
      for (let mi = 0; mi < M; mi++) {
        const A = new Float32Array(K);
        for (let i = 0; i < K; i++) A[i] = rng() * 2 - 1;
        Arows.push(A);
      }

      // (a) _r single-row port anchored to the f64 reference dot.
      let anchorOK = true, maxRel = 0;
      for (let mi = 0; mi < M && anchorOK; mi++) {
        for (let n = 0; n < N; n++) {
          let refDot = 0, refMag = 0;
          for (let i = 0; i < K; i++) {
            const t = Arows[mi][i] * ref[n * K + i];
            refDot += t;
            refMag += Math.abs(t);
          }
          const got = rKernelRow(Wr, Arows[mi], n, K, TPR);
          // Scale the error by the accumulation magnitude (Σ|terms|), not by
          // |refDot| — rows where refDot ≈ 0 (catastrophic cancellation) make
          // a |refDot| denominator meaningless. The _r/GEMM path uses the
          // deferred-scale split dl0*Σdot - ml0*Σasum, algebraically equal to
          // but numerically distinct from the folded per-element reference
          // (drift ~1e-6 of the accumulation scale even in f64). Anchors the
          // port to ground truth; gate (b) is the exact GEMM==_r parity check.
          const rel = Math.abs(got - refDot) / Math.max(1e-30, refMag);
          maxRel = Math.max(maxRel, rel);
          if (rel >= 1e-5) {
            check(`_r_gemm K=${K} M=${M} anchor`, false, `n=${n} mi=${mi} rel=${rel}`);
            anchorOK = false;
            break;
          }
        }
      }

      // (b) GEMM column row mi == _r single-row on row mi EXACTLY.
      let gemmOK = true;
      for (let n = 0; n < N && gemmOK; n++) {
        const cols = gemmCol(Wr, Arows, n, K, TPR);
        for (let mi = 0; mi < M; mi++) {
          const single = rKernelRow(Wr, Arows[mi], n, K, TPR);
          if (cols[mi] !== single) {
            check(`_r_gemm K=${K} M=${M} gemm`, false, `n=${n} mi=${mi}: ${cols[mi]} != ${single}`);
            gemmOK = false;
            break;
          }
        }
      }
      if (anchorOK && gemmOK) {
        console.log(`  ok   _r_gemm K=${K} TPR=${TPR} M=${M}: gemm[mi] == _r row mi (exact), anchor maxRel=${maxRel.toExponential(2)}`);
      }
    }
  }
}

if (failures > 0) {
  console.log(`\n${failures} FAILURE(S)`);
  process.exit(1);
}
console.log('\nAll tiled GEMV checks passed.');
