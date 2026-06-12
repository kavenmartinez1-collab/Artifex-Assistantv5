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
    }
  }
}

console.log('Q4_K tiled:');
runFormat('Q4_K', GGML_TYPES.Q4_K, 144, (raw, base, rng) => {
  // d @0, dmin @2 — controlled finite f16
  const dv = new DataView(raw.buffer, raw.byteOffset);
  dv.setUint16(base, randF16Bits(rng), true);
  dv.setUint16(base + 2, randF16Bits(rng), true);
}, decodeUnitQ4K);

console.log('Q5_K tiled:');
runFormat('Q5_K', GGML_TYPES.Q5_K, 176, (raw, base, rng) => {
  // d @0, dmin @2 — controlled finite f16
  const dv = new DataView(raw.buffer, raw.byteOffset);
  dv.setUint16(base, randF16Bits(rng), true);
  dv.setUint16(base + 2, randF16Bits(rng), true);
}, decodeUnitQ5K);

console.log('Q6_K tiled:');
runFormat('Q6_K', GGML_TYPES.Q6_K, 210, (raw, base, rng) => {
  // d @208 — controlled finite f16 (ql/qh/scales stay random)
  const dv = new DataView(raw.buffer, raw.byteOffset);
  dv.setUint16(base + 208, randF16Bits(rng), true);
}, decodeUnitQ6K);

if (failures > 0) {
  console.log(`\n${failures} FAILURE(S)`);
  process.exit(1);
}
console.log('\nAll tiled GEMV checks passed.');
