/**
 * CPU reference dequantization for GGUF tensor types.
 *
 * This is the ground truth that every WGSL k-quant kernel is validated
 * against (kernel-audit rule). Implementations mirror llama.cpp
 * ggml-quants.c dequantize_row_* exactly — bit layouts below.
 *
 * Block layouts (little-endian):
 *   Q4_0 (32 elems, 18 B):  f16 d | u8 qs[16] (4-bit)       x = d·(q-8)
 *   Q5_0 (32 elems, 22 B):  f16 d | u8 qh[4] | u8 qs[16]     x = d·(q5-16)
 *   Q8_0 (32 elems, 34 B):  f16 d | i8 qs[32]              x = d·qs
 *   Q4_K (256, 144 B):      f16 d, dmin | u8 scales[12] | u8 qs[128] (4-bit pairs)
 *   Q5_K (256, 176 B):      f16 d, dmin | u8 scales[12] | u8 qh[32] | u8 qs[128]
 *   Q6_K (256, 210 B):      u8 ql[128] | u8 qh[64] | i8 scales[16] | f16 d
 *   IQ4_NL (32, 18 B):      f16 d | u8 qs[16] (4-bit)      x = d·kvalues[q]
 *   IQ4_XS (256, 136 B):    f16 d | u16 scales_h | u8 scales_l[4] | u8 qs[128]
 *
 * Q4_K/Q5_K 6-bit scale/min packing (get_scale_min_k4, j in 0..7):
 *   j < 4:  sc = q[j] & 63;              m = q[j+4] & 63
 *   j >= 4: sc = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
 *           m  = (q[j+4] >>  4) | ((q[j]   >> 6) << 4)
 */

// GGML type ids (spec constants, duplicated from gguf.ts so this file has
// zero imports and runs standalone under node --strip-types for validation)
const T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q5_0 = 6, T_Q8_0 = 8, T_Q2_K = 10, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_IQ2_XXS = 16, T_IQ4_NL = 20, T_IQ4_XS = 23, T_BF16 = 30;

/** IQ4 non-linear codebook (kvalues_iq4nl, ggml-quants.c) — shared by IQ4_NL and IQ4_XS. */
const KVALUES_IQ4NL = new Int8Array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]);

/** IQ2_XXS grid (iq2xxs_grid, ggml-common.h): 256 entries × 8 magnitude bytes.
 *  Generated bit-exact via scripts/gen-iq2xxs-tables.mjs. */
const IQ2XXS_GRID = new Uint8Array([8,8,8,8,8,8,8,8,43,8,8,8,8,8,8,8,25,25,8,8,8,8,8,8,8,43,8,8,8,8,8,8,43,43,8,8,8,8,8,8,25,8,25,8,8,8,8,8,8,25,25,8,8,8,8,8,8,8,43,8,8,8,8,8,43,8,43,8,8,8,8,8,8,43,43,8,8,8,8,8,43,43,43,8,8,8,8,8,25,8,8,25,8,8,8,8,8,25,8,25,8,8,8,8,8,8,25,25,8,8,8,8,8,43,25,25,8,8,8,8,25,8,43,25,8,8,8,8,8,25,43,25,8,8,8,8,8,8,8,43,8,8,8,8,43,8,8,43,8,8,8,8,43,43,8,43,8,8,8,8,43,8,43,43,8,8,8,8,25,8,8,8,25,8,8,8,8,25,8,8,25,8,8,8,8,8,25,8,25,8,8,8,25,25,25,8,25,8,8,8,8,8,8,25,25,8,8,8,8,25,8,43,25,8,8,8,8,43,25,43,25,8,8,8,8,8,8,8,43,8,8,8,43,8,8,8,43,8,8,8,43,8,43,8,43,8,8,8,43,8,8,43,43,8,8,8,25,8,8,8,8,25,8,8,8,25,8,8,8,25,8,8,8,8,25,8,8,25,8,8,25,8,43,8,8,25,8,8,8,25,43,8,8,25,8,8,8,8,8,25,8,25,8,8,43,8,8,25,8,25,8,8,8,43,8,25,8,25,8,8,8,8,43,25,8,25,8,8,25,8,8,43,8,25,8,8,8,25,8,43,8,25,8,8,8,8,25,43,8,25,8,8,8,25,43,43,8,25,8,8,8,8,8,8,25,25,8,8,43,8,8,8,25,25,8,8,8,43,8,8,25,25,8,8,8,8,43,8,25,25,8,8,43,25,8,25,25,25,8,8,25,43,43,25,25,25,8,8,8,8,8,43,25,25,8,8,25,8,25,43,25,25,8,8,25,43,8,8,43,25,8,8,8,8,25,8,43,25,8,8,8,8,8,25,43,25,8,8,8,25,8,43,43,25,8,8,8,25,43,43,43,25,8,8,8,8,8,8,8,43,8,8,25,25,8,8,8,43,8,8,8,43,8,8,8,43,8,8,8,25,25,8,8,43,8,8,8,43,43,8,8,43,8,8,25,8,8,25,8,43,8,8,8,25,8,25,8,43,8,8,8,8,25,25,8,43,8,8,43,8,25,25,8,43,8,8,8,43,8,43,8,43,8,8,8,25,8,8,25,43,8,8,8,8,8,25,25,43,8,8,43,8,8,8,43,43,8,8,8,25,25,8,43,43,8,8,25,8,8,8,8,8,25,8,8,25,8,8,8,8,25,8,8,8,25,8,8,8,25,8,25,8,43,8,8,8,25,8,8,8,8,25,8,8,25,8,8,8,43,25,8,8,25,8,8,25,8,43,8,8,25,8,8,8,25,43,8,8,25,8,25,25,25,43,8,8,25,8,8,8,8,8,25,8,25,8,8,43,8,8,25,8,25,8,8,8,43,8,25,8,25,8,8,8,25,25,25,8,25,8,43,43,25,25,25,8,25,8,8,8,8,43,25,8,25,8,8,25,43,8,43,8,25,8,25,25,8,25,43,8,25,8,8,8,8,8,8,25,25,8,8,43,8,8,8,25,25,8,8,8,43,8,8,25,25,8,25,25,43,8,8,25,25,8,25,43,8,25,8,25,25,8,8,8,8,43,8,25,25,8,8,43,25,8,25,25,25,8,43,8,43,25,25,25,25,8,8,8,8,8,43,25,25,8,43,25,25,8,43,25,25,8,25,8,8,8,8,43,25,8,8,25,8,8,8,43,25,8,8,8,25,8,8,43,25,8,8,8,8,25,8,43,25,8,25,8,8,43,8,43,25,8,8,8,8,8,25,43,25,8,25,25,8,8,25,43,25,8,8,8,43,43,25,43,25,8,25,8,25,25,43,43,25,8,8,8,8,8,8,8,43,8,43,8,8,8,8,8,43,8,43,43,8,8,8,8,43,8,8,25,8,25,8,8,43,8,25,8,43,25,8,8,43,8,8,8,8,43,8,8,43,8,43,8,8,43,8,8,43,8,25,43,43,8,25,8,43,8,8,43,8,25,25,8,43,8,8,8,8,8,43,8,43,8,43,8,8,8,43,8,43,8,25,8,8,8,8,25,43,8,8,25,8,8,8,25,43,8,8,8,25,8,8,25,43,8,8,8,8,25,8,25,43,8,43,25,25,25,8,25,43,8,8,8,8,8,25,25,43,8,25,8,8,25,25,25,43,8,8,25,43,25,25,25,43,8,8,8,25,43,43,25,43,8,8,43,8,8,8,43,43,8,8,8,43,8,8,43,43,8,8,25,25,43,8,43,43,8,8,25,8,25,43,43,43,8,25,8,8,8,8,8,8,25,8,25,8,8,8,8,8,25,8,8,25,8,8,8,8,25,8,43,25,8,8,8,8,25,25,8,43,8,8,8,8,25,8,25,43,8,8,8,8,25,8,8,8,25,8,8,8,25,8,43,8,25,8,8,8,25,43,25,25,25,8,8,8,25,8,8,43,25,8,8,8,25,25,8,8,43,8,8,8,25,8,25,8,43,8,8,8,25,8,8,25,43,8,8,8,25,8,8,8,8,25,8,8,25,8,8,43,8,25,8,8,25,25,8,43,25,25,8,8,25,8,8,8,43,25,8,8,25,25,25,8,43,25,8,8,25,25,8,8,8,43,8,8,25,8,8,25,8,43,8,8,25,8,43,8,25,43,8,8,25,43,25,25,25,43,8,8,25,8,43,43,25,43,8,8,25,8,8,8,8,8,25,8,25,8,43,8,8,8,25,8,25,8,8,43,8,8,25,8,25,8,8,8,43,8,25,8,25,25,43,25,43,8,25,8,25,43,8,25,8,25,25,8,25,8,25,43,8,25,25,8,25,8,8,8,8,43,25,8,25,25,8,8,8,8,43,8,25,8,25,8,8,8,43,8,25,8,8,25,8,8,43,8,25,8,8,8,25,8,43,8,25,25,25,8,25,8,43,8,25,8,8,8,8,25,43,8,25,8,43,25,25,25,43,8,25,25,8,43,25,25,43,8,25,43,8,8,43,25,43,8,25,25,25,8,25,43,43,8,25,8,8,25,43,43,43,8,25,8,8,8,8,8,8,25,25,8,43,8,8,8,8,25,25,25,8,25,8,8,8,25,25,25,43,25,8,8,8,25,25,8,8,43,8,8,8,25,25,8,8,8,43,8,8,25,25,8,43,8,43,8,8,25,25,8,25,8,8,25,8,25,25,43,8,8,25,25,8,25,25,8,25,43,43,25,8,25,25,25,8,25,43,43,8,25,25,8,8,25,43,8,25,25,25,43,8,25,43,8,25,25,25,43,43,8,8,25,25,25,25,25,8,8,8,43,25,25,25,8,25,25,25,43,25,25,25,8,8,8,8,8,43,25,25,25,8,25,8,8,43,25,25,25,43,25,8,8,43,25,25,8,25,43,25,8,43,25,25,8,8,8,25,25,43,25,25,8,43,8,8,43,43,25,25,8,25,8,8,8,8,43,25,8,8,25,8,8,8,43,25,8,8,8,25,8,8,43,25,8,43,43,25,8,8,43,25,8,8,8,8,25,8,43,25,25,25,25,25,25,8,43,25,8,43,25,8,43,8,43,25,8,8,43,25,43,8,43,25,8,8,8,8,8,25,43,25,25,25,8,8,8,25,43,25,8,8,25,8,25,25,43,25,43,8,25,8,25,25,43,25,8,25,8,43,25,25,43,25,43,8,8,25,8,43,43,25,8,8,8,8,8,8,8,43,43,8,8,8,8,8,8,43,43,43,8,8,8,8,8,43,25,8,8,25,8,8,8,43,43,8,8,43,8,8,8,43,8,25,8,8,25,8,8,43,8,43,25,8,25,8,8,43,8,8,8,25,25,8,8,43,25,8,25,8,43,8,8,43,25,8,8,8,8,25,8,43,8,25,8,8,8,25,8,43,8,8,25,8,8,25,8,43,25,25,25,8,8,25,8,43,8,8,8,25,8,25,8,43,8,8,43,25,8,25,8,43,8,8,8,8,25,25,8,43,43,25,8,25,25,25,8,43,8,25,25,43,25,25,8,43,25,43,8,8,43,25,8,43,8,8,8,25,43,25,8,43,8,8,43,25,43,25,8,43,43,8,8,8,8,43,8,43,8,25,8,8,25,43,8,43,25,8,25,8,43,43,8,43,8,25,8,8,8,8,25,43,8,8,25,8,8,8,25,43,8,25,43,8,8,8,25,43,8,8,8,25,8,8,25,43,25,8,43,43,8,8,25,43,43,25,25,8,25,8,25,43,8,8,8,43,25,8,25,43,25,25,8,25,43,8,25,43,8,8,8,8,8,25,25,43,43,8,43,8,8,25,25,43,8,25,8,25,8,25,25,43,25,8,25,25,25,25,25,43,25,8,8,43,8,43,25,43,8,8,43,8,25,43,25,43,43,8,8,8,8,8,43,43,8,8,25,25,8,8,43,43,25,25,8,43,8,8,43,43,25,43,8,8,25,8,43,43,8,8,8,8,43,8,43,43,8,43,25,8,8,25,43,43,8,8,25,25,8,43,43,43,8,25,8,8,25,43,43,43]);
/** ksigns_iq2xs (ggml-common.h): 128 sign-bit bytes, indexed by the 7-bit sign field. */
const KSIGNS_IQ2XS = new Uint8Array([0,129,130,3,132,5,6,135,136,9,10,139,12,141,142,15,144,17,18,147,20,149,150,23,24,153,154,27,156,29,30,159,160,33,34,163,36,165,166,39,40,169,170,43,172,45,46,175,48,177,178,51,180,53,54,183,184,57,58,187,60,189,190,63,192,65,66,195,68,197,198,71,72,201,202,75,204,77,78,207,80,209,210,83,212,85,86,215,216,89,90,219,92,221,222,95,96,225,226,99,228,101,102,231,232,105,106,235,108,237,238,111,240,113,114,243,116,245,246,119,120,249,250,123,252,125,126,255]);

// ── f16 / bf16 → f32 ───────────────────────────────────────────────────

export function f16ToF32(h: number): number {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * frac * 2 ** -24;            // subnormal
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

const bf16Buf = new ArrayBuffer(4);
const bf16U32 = new Uint32Array(bf16Buf);
const bf16F32 = new Float32Array(bf16Buf);
export function bf16ToF32(h: number): number {
  bf16U32[0] = h << 16;
  return bf16F32[0];
}

// ── Per-type dequant (entire tensor buffer → Float32Array) ─────────────

export function dequantF32(data: ArrayBuffer | Uint8Array, n: number): Float32Array {
  const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
  // May be unaligned (GGUF offsets are 32-aligned so usually fine) — copy-safe path
  const out = new Float32Array(n);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  for (let i = 0; i < n; i++) out[i] = dv.getFloat32(i * 4, true);
  return out;
}

export function dequantF16(data: ArrayBuffer | Uint8Array, n: number): Float32Array {
  const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = f16ToF32(dv.getUint16(i * 2, true));
  return out;
}

export function dequantBF16(data: ArrayBuffer | Uint8Array, n: number): Float32Array {
  const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = bf16ToF32(dv.getUint16(i * 2, true));
  return out;
}

/** Q4_0: 18-byte blocks of 32 elements. x[j] = d·((qs[j]&0xF)-8),
 *  x[j+16] = d·((qs[j]>>4)-8) for j in 0..15. */
export function dequantQ4_0(data: ArrayBuffer | Uint8Array, n: number): Float32Array {
  const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (n % 32 !== 0) throw new Error(`[Q4_0] n=${n} not divisible by 32`);
  const nb = n / 32;
  const out = new Float32Array(n);
  for (let b = 0; b < nb; b++) {
    const base = b * 18;
    const d = f16ToF32(dv.getUint16(base, true));
    for (let j = 0; j < 16; j++) {
      const q = u8[base + 2 + j];
      out[b * 32 + j] = d * ((q & 0xF) - 8);
      out[b * 32 + j + 16] = d * ((q >> 4) - 8);
    }
  }
  return out;
}

/** Q5_0: 22-byte blocks of 32 elements. 5th bit from qh (u32 little-endian).
 *  x[j] = d·(((qs[j]&0xF)|((qh>>j&1)<<4))-16), and similarly for j+16. */
export function dequantQ5_0(data: ArrayBuffer | Uint8Array, n: number): Float32Array {
  const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (n % 32 !== 0) throw new Error(`[Q5_0] n=${n} not divisible by 32`);
  const nb = n / 32;
  const out = new Float32Array(n);
  for (let b = 0; b < nb; b++) {
    const base = b * 22;
    const d = f16ToF32(dv.getUint16(base, true));
    const qh = dv.getUint32(base + 2, true);
    for (let j = 0; j < 16; j++) {
      const q = u8[base + 6 + j];
      const lo = (q & 0xF) | (((qh >>> j) & 1) << 4);
      const hi = (q >> 4) | (((qh >>> (j + 16)) & 1) << 4);
      out[b * 32 + j] = d * (lo - 16);
      out[b * 32 + j + 16] = d * (hi - 16);
    }
  }
  return out;
}

/** Q8_0: 34-byte blocks of 32 elements. */
export function dequantQ8_0(data: ArrayBuffer | Uint8Array, n: number): Float32Array {
  const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (n % 32 !== 0) throw new Error(`[Q8_0] n=${n} not divisible by 32`);
  const nb = n / 32;
  const out = new Float32Array(n);
  for (let b = 0; b < nb; b++) {
    const base = b * 34;
    const d = f16ToF32(dv.getUint16(base, true));
    for (let i = 0; i < 32; i++) {
      out[b * 32 + i] = d * dv.getInt8(base + 2 + i);
    }
  }
  return out;
}

/** Extract (scale, min) pair j (0..7) from the 12-byte packed scales of Q4_K/Q5_K. */
function getScaleMinK4(j: number, q: Uint8Array, base: number): [number, number] {
  if (j < 4) {
    return [q[base + j] & 63, q[base + j + 4] & 63];
  }
  const sc = (q[base + j + 4] & 0x0f) | ((q[base + j - 4] >> 6) << 4);
  const m = (q[base + j + 4] >> 4) | ((q[base + j] >> 6) << 4);
  return [sc, m];
}

/** Q2_K: 84-byte superblocks of 256 elements.
 *  Layout: scales[16] @0 | qs[64] @16 | d(f16) @80 | dmin(f16) @82.
 *  Mirrors dequantize_row_q2_K (ggml-quants.c). */
export function dequantQ2_K(data: ArrayBuffer | Uint8Array, n: number): Float32Array {
  const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (n % 256 !== 0) throw new Error(`[Q2_K] n=${n} not divisible by 256`);
  const nb = n / 256;
  const out = new Float32Array(n);
  for (let b = 0; b < nb; b++) {
    const base = b * 84;
    const d = f16ToF32(dv.getUint16(base + 80, true));
    const min = f16ToF32(dv.getUint16(base + 82, true));
    let oi = b * 256, is = 0;
    for (let nn = 0; nn < 256; nn += 128) {
      const qBase = base + 16 + (nn / 128) * 32;
      let shift = 0;
      for (let j = 0; j < 4; j++) {
        let sc = u8[base + is]; is++;
        let dl = d * (sc & 0xF), ml = min * (sc >> 4);
        for (let l = 0; l < 16; l++) out[oi++] = dl * ((u8[qBase + l] >> shift) & 3) - ml;
        sc = u8[base + is]; is++;
        dl = d * (sc & 0xF); ml = min * (sc >> 4);
        for (let l = 0; l < 16; l++) out[oi++] = dl * ((u8[qBase + 16 + l] >> shift) & 3) - ml;
        shift += 2;
      }
    }
  }
  return out;
}

/** Q3_K: 110-byte superblocks of 256 elements.
 *  Layout: hmask[32] @0 | qs[64] @32 | scales[12] @96 | d(f16) @108.
 *  Mirrors dequantize_row_q3_K (ggml-quants.c) including the 6-bit scale
 *  unpack and the per-element high-bit (hmask). */
export function dequantQ3_K(data: ArrayBuffer | Uint8Array, n: number): Float32Array {
  const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (n % 256 !== 0) throw new Error(`[Q3_K] n=${n} not divisible by 256`);
  const nb = n / 256;
  const out = new Float32Array(n);
  const KM1 = 0x03030303, KM2 = 0x0f0f0f0f;
  for (let b = 0; b < nb; b++) {
    const base = b * 110;
    const dAll = f16ToF32(dv.getUint16(base + 108, true));
    // Unpack 12 scale bytes → 16 signed 6-bit scales (aux shuffle from ggml).
    const s0 = dv.getUint32(base + 96, true) >>> 0;
    const s1 = dv.getUint32(base + 100, true) >>> 0;
    const tmp = dv.getUint32(base + 104, true) >>> 0;
    const aux2 = (((s0 >>> 4) & KM2) | ((((tmp >>> 4) & KM1) << 4) >>> 0)) >>> 0;
    const aux3 = (((s1 >>> 4) & KM2) | ((((tmp >>> 6) & KM1) << 4) >>> 0)) >>> 0;
    const aux0 = ((s0 & KM2) | ((((tmp >>> 0) & KM1) << 4) >>> 0)) >>> 0;
    const aux1 = ((s1 & KM2) | ((((tmp >>> 2) & KM1) << 4) >>> 0)) >>> 0;
    const auxBytes = new Uint8Array(new Uint32Array([aux0, aux1, aux2, aux3]).buffer);
    let oi = b * 256, is = 0, m = 1;
    for (let nn = 0; nn < 256; nn += 128) {
      const qBase = base + 32 + (nn / 128) * 32;
      let shift = 0;
      for (let j = 0; j < 4; j++) {
        let dl = dAll * (auxBytes[is] - 32); is++;
        for (let l = 0; l < 16; l++) {
          const q3 = (u8[qBase + l] >> shift) & 3;
          out[oi++] = dl * (q3 - ((u8[base + l] & m) ? 0 : 4));
        }
        dl = dAll * (auxBytes[is] - 32); is++;
        for (let l = 0; l < 16; l++) {
          const q3 = (u8[qBase + 16 + l] >> shift) & 3;
          out[oi++] = dl * (q3 - ((u8[base + 16 + l] & m) ? 0 : 4));
        }
        shift += 2; m <<= 1;
      }
    }
  }
  return out;
}

/** Q4_K: 144-byte superblocks of 256 elements. */
export function dequantQ4_K(data: ArrayBuffer | Uint8Array, n: number): Float32Array {
  const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (n % 256 !== 0) throw new Error(`[Q4_K] n=${n} not divisible by 256`);
  const nb = n / 256;
  const out = new Float32Array(n);
  for (let b = 0; b < nb; b++) {
    const base = b * 144;
    const d = f16ToF32(dv.getUint16(base, true));
    const dmin = f16ToF32(dv.getUint16(base + 2, true));
    const scalesBase = base + 4;
    const qsBase = base + 16;
    let y = b * 256;
    let qs = qsBase;
    let is = 0;
    for (let j = 0; j < 256; j += 64) {
      const [sc1, m1] = getScaleMinK4(is + 0, u8, scalesBase);
      const [sc2, m2] = getScaleMinK4(is + 1, u8, scalesBase);
      const d1 = d * sc1, min1 = dmin * m1;
      const d2 = d * sc2, min2 = dmin * m2;
      for (let l = 0; l < 32; l++) out[y + l] = d1 * (u8[qs + l] & 0x0f) - min1;
      for (let l = 0; l < 32; l++) out[y + 32 + l] = d2 * (u8[qs + l] >> 4) - min2;
      y += 64; qs += 32; is += 2;
    }
  }
  return out;
}

/** Q5_K: 176-byte superblocks of 256 elements. */
export function dequantQ5_K(data: ArrayBuffer | Uint8Array, n: number): Float32Array {
  const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (n % 256 !== 0) throw new Error(`[Q5_K] n=${n} not divisible by 256`);
  const nb = n / 256;
  const out = new Float32Array(n);
  for (let b = 0; b < nb; b++) {
    const base = b * 176;
    const d = f16ToF32(dv.getUint16(base, true));
    const dmin = f16ToF32(dv.getUint16(base + 2, true));
    const scalesBase = base + 4;
    const qhBase = base + 16;
    const qlBase = base + 48;
    let y = b * 256;
    let ql = qlBase;
    let is = 0;
    let u1 = 1, u2 = 2;
    for (let j = 0; j < 256; j += 64) {
      const [sc1, m1] = getScaleMinK4(is + 0, u8, scalesBase);
      const [sc2, m2] = getScaleMinK4(is + 1, u8, scalesBase);
      const d1 = d * sc1, min1 = dmin * m1;
      const d2 = d * sc2, min2 = dmin * m2;
      for (let l = 0; l < 32; l++) {
        out[y + l] = d1 * ((u8[ql + l] & 0x0f) + ((u8[qhBase + l] & u1) ? 16 : 0)) - min1;
      }
      for (let l = 0; l < 32; l++) {
        out[y + 32 + l] = d2 * ((u8[ql + l] >> 4) + ((u8[qhBase + l] & u2) ? 16 : 0)) - min2;
      }
      y += 64; ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
    }
  }
  return out;
}

/** IQ4_NL: 18-byte blocks of 32 elements — Q4_0 layout, non-linear codebook.
 *  x[j] = d·kvalues[qs[j]&0xF], x[j+16] = d·kvalues[qs[j]>>4] for j in 0..15.
 *  Mirrors dequantize_row_iq4_nl (ggml-quants.c). */
export function dequantIQ4_NL(data: ArrayBuffer | Uint8Array, n: number): Float32Array {
  const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (n % 32 !== 0) throw new Error(`[IQ4_NL] n=${n} not divisible by 32`);
  const nb = n / 32;
  const out = new Float32Array(n);
  for (let b = 0; b < nb; b++) {
    const base = b * 18;
    const d = f16ToF32(dv.getUint16(base, true));
    for (let j = 0; j < 16; j++) {
      const q = u8[base + 2 + j];
      out[b * 32 + j] = d * KVALUES_IQ4NL[q & 0xF];
      out[b * 32 + j + 16] = d * KVALUES_IQ4NL[q >> 4];
    }
  }
  return out;
}

/** IQ4_XS: 136-byte superblocks of 256 elements.
 *  Layout: d(f16) @0 | scales_h(u16) @2 | scales_l[4] @4 | qs[128] @8.
 *  Per-32-elem 6-bit sub-scale: low 4 bits from a scales_l nibble, high 2
 *  bits from scales_h; x = d·(ls-32)·kvalues[q]. Mirrors
 *  dequantize_row_iq4_xs (ggml-quants.c). */
export function dequantIQ4_XS(data: ArrayBuffer | Uint8Array, n: number): Float32Array {
  const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (n % 256 !== 0) throw new Error(`[IQ4_XS] n=${n} not divisible by 256`);
  const nb = n / 256;
  const out = new Float32Array(n);
  for (let b = 0; b < nb; b++) {
    const base = b * 136;
    const d = f16ToF32(dv.getUint16(base, true));
    const sh = dv.getUint16(base + 2, true);
    let y = b * 256, qs = base + 8;
    for (let ib = 0; ib < 8; ib++) {
      const ls = ((u8[base + 4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
      const dl = d * (ls - 32);
      for (let j = 0; j < 16; j++) {
        out[y + j] = dl * KVALUES_IQ4NL[u8[qs + j] & 0xF];
        out[y + j + 16] = dl * KVALUES_IQ4NL[u8[qs + j] >> 4];
      }
      y += 32; qs += 16;
    }
  }
  return out;
}

/** Q6_K: 210-byte superblocks of 256 elements. */
export function dequantQ6_K(data: ArrayBuffer | Uint8Array, n: number): Float32Array {
  const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (n % 256 !== 0) throw new Error(`[Q6_K] n=${n} not divisible by 256`);
  const nb = n / 256;
  const out = new Float32Array(n);
  for (let b = 0; b < nb; b++) {
    const base = b * 210;
    const qlBase = base;          // ql[128]
    const qhBase = base + 128;    // qh[64]
    const scBase = base + 192;    // scales[16] (i8)
    const d = f16ToF32(dv.getUint16(base + 208, true));
    let y = b * 256;
    let ql = qlBase, qh = qhBase, sc = scBase;
    for (let half = 0; half < 2; half++) {   // 2 × 128 elements
      for (let l = 0; l < 32; l++) {
        const is = (l / 16) | 0;
        const q1 = ((u8[ql + l] & 0x0f) | (((u8[qh + l] >> 0) & 3) << 4)) - 32;
        const q2 = ((u8[ql + l + 32] & 0x0f) | (((u8[qh + l] >> 2) & 3) << 4)) - 32;
        const q3 = ((u8[ql + l] >> 4) | (((u8[qh + l] >> 4) & 3) << 4)) - 32;
        const q4 = ((u8[ql + l + 32] >> 4) | (((u8[qh + l] >> 6) & 3) << 4)) - 32;
        out[y + l] = d * dv.getInt8(sc + is) * q1;
        out[y + l + 32] = d * dv.getInt8(sc + is + 2) * q2;
        out[y + l + 64] = d * dv.getInt8(sc + is + 4) * q3;
        out[y + l + 96] = d * dv.getInt8(sc + is + 6) * q4;
      }
      y += 128; ql += 64; qh += 32; sc += 8;
    }
  }
  return out;
}

/** IQ2_XXS: 66-byte superblocks of 256 elements.
 *  Layout: d(f16) @0 | qs[32] u16 @2 (8 ib32 groups, 8 bytes each).
 *  Per ib32: aux0 (4 grid indices) | aux1 (4×7-bit sign idx + 4-bit scale ls
 *  in top nibble). db = d·(0.5+ls)·0.25; value = db·grid[j]·sign(j).
 *  Mirrors dequantize_row_iq2_xxs (ggml-quants.c). */
export function dequantIQ2_XXS(data: ArrayBuffer | Uint8Array, n: number): Float32Array {
  const u8 = asU8(data);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (n % 256 !== 0) throw new Error(`[IQ2_XXS] n=${n} not divisible by 256`);
  const nb = n / 256;
  const out = new Float32Array(n);
  for (let b = 0; b < nb; b++) {
    const base = b * 66;
    const d = f16ToF32(dv.getUint16(base, true));
    for (let ib = 0; ib < 8; ib++) {
      const g = base + 2 + ib * 8;            // 8 bytes: aux0[4] | aux1(u32)
      const aux1 = dv.getUint32(g + 4, true);
      const ls = aux1 >>> 28;
      const db = d * (0.5 + ls) * 0.25;
      let y = b * 256 + ib * 32;
      for (let l = 0; l < 4; l++) {
        const gridIdx = u8[g + l];
        const signs = KSIGNS_IQ2XS[(aux1 >>> (7 * l)) & 127];
        for (let j = 0; j < 8; j++) {
          const sign = (signs >> j) & 1 ? -1 : 1;
          out[y + l * 8 + j] = db * IQ2XXS_GRID[gridIdx * 8 + j] * sign;
        }
      }
    }
  }
  return out;
}

// ── GPU repack (4-byte-aligned block strides for matmul_gguf.wgsl) ─────

/** Per-type GPU layout: block element count and u32 stride after repack. */
export const GGUF_GPU_LAYOUT: Record<number, { blockElems: number; strideU32: number }> = {
  [T_Q4_0]: { blockElems: 32, strideU32: 5 },    // 18 B → 20 B (d|pad|qs[16])
  [T_Q5_0]: { blockElems: 32, strideU32: 6 },    // 22 B → 24 B (d|pad|qh|qs[16])
  [T_Q2_K]: { blockElems: 256, strideU32: 21 },  // raw 84 B (4-aligned)
  [T_Q3_K]: { blockElems: 256, strideU32: 28 },  // 110 B → 112 B (+2 pad)
  [T_Q8_0]: { blockElems: 32, strideU32: 9 },    // 34 B → 36 B
  [T_Q4_K]: { blockElems: 256, strideU32: 36 },  // raw 144 B
  [T_Q5_K]: { blockElems: 256, strideU32: 44 },  // raw 176 B
  [T_Q6_K]: { blockElems: 256, strideU32: 53 },  // 210 B → 212 B
  [T_IQ2_XXS]: { blockElems: 256, strideU32: 17 }, // 66 B → 68 B (+2 pad)
  [T_IQ4_NL]: { blockElems: 32, strideU32: 5 },  // 18 B → 20 B (Q4_0-shaped)
  [T_IQ4_XS]: { blockElems: 256, strideU32: 34 }, // raw 136 B (4-aligned)
};

function asU8(data: ArrayBuffer | Uint8Array): Uint8Array {
  return data instanceof Uint8Array ? data : new Uint8Array(data);
}

function assertBytes(label: string, got: number, want: number): void {
  if (got !== want) throw new Error(`[${label}] byteLength ${got} != expected ${want}`);
}

/** Q4_0 18 B/block → 20 B/block: f16 d @ 0-1, pad @ 2-3, qs[16] @ 4-19. */
export function repackQ4_0(data: ArrayBuffer | Uint8Array, n: number): Uint32Array {
  const u8 = asU8(data);
  if (n % 32 !== 0) throw new Error(`[repackQ4_0] n=${n} not divisible by 32`);
  const nb = n / 32;
  assertBytes('repackQ4_0', u8.byteLength, nb * 18);
  const out = new Uint32Array(nb * 5);
  const o8 = new Uint8Array(out.buffer);
  for (let b = 0; b < nb; b++) {
    const src = b * 18, dst = b * 20;
    o8[dst] = u8[src];
    o8[dst + 1] = u8[src + 1];
    o8.set(u8.subarray(src + 2, src + 18), dst + 4);
  }
  return out;
}

/** Q5_0 22 B/block → 24 B/block: f16 d @ 0-1, pad @ 2-3, qh @ 4-7, qs[16] @ 8-23. */
export function repackQ5_0(data: ArrayBuffer | Uint8Array, n: number): Uint32Array {
  const u8 = asU8(data);
  if (n % 32 !== 0) throw new Error(`[repackQ5_0] n=${n} not divisible by 32`);
  const nb = n / 32;
  assertBytes('repackQ5_0', u8.byteLength, nb * 22);
  const out = new Uint32Array(nb * 6);
  const o8 = new Uint8Array(out.buffer);
  for (let b = 0; b < nb; b++) {
    const src = b * 22, dst = b * 24;
    o8[dst] = u8[src];
    o8[dst + 1] = u8[src + 1];
    o8.set(u8.subarray(src + 2, src + 6), dst + 4);    // qh
    o8.set(u8.subarray(src + 6, src + 22), dst + 8);   // qs[16]
  }
  return out;
}

/** Q8_0 34 B/block → 36 B/block: f16 d @ bytes 0-1, pad @ 2-3, qs[32] @ 4-35. */
export function repackQ8_0(data: ArrayBuffer | Uint8Array, n: number): Uint32Array {
  const u8 = asU8(data);
  if (n % 32 !== 0) throw new Error(`[repackQ8_0] n=${n} not divisible by 32`);
  const nb = n / 32;
  assertBytes('repackQ8_0', u8.byteLength, nb * 34);
  const out = new Uint32Array(nb * 9);
  const o8 = new Uint8Array(out.buffer);
  for (let b = 0; b < nb; b++) {
    const src = b * 34, dst = b * 36;
    o8[dst] = u8[src];
    o8[dst + 1] = u8[src + 1];
    o8.set(u8.subarray(src + 2, src + 34), dst + 4);
  }
  return out;
}

/** Q3_K 110 B/block → 112 B/block: raw copy + 2 pad bytes (f16 d ends @ 109). */
export function repackQ3_K(data: ArrayBuffer | Uint8Array, n: number): Uint32Array {
  const u8 = asU8(data);
  if (n % 256 !== 0) throw new Error(`[repackQ3_K] n=${n} not divisible by 256`);
  const nb = n / 256;
  assertBytes('repackQ3_K', u8.byteLength, nb * 110);
  const out = new Uint32Array(nb * 28);
  const o8 = new Uint8Array(out.buffer);
  for (let b = 0; b < nb; b++) o8.set(u8.subarray(b * 110, (b + 1) * 110), b * 112);
  return out;
}

/** Q6_K 210 B/block → 212 B/block: raw copy + 2 pad bytes (f16 d ends @ 209). */
export function repackQ6_K(data: ArrayBuffer | Uint8Array, n: number): Uint32Array {
  const u8 = asU8(data);
  if (n % 256 !== 0) throw new Error(`[repackQ6_K] n=${n} not divisible by 256`);
  const nb = n / 256;
  assertBytes('repackQ6_K', u8.byteLength, nb * 210);
  const out = new Uint32Array(nb * 53);
  const o8 = new Uint8Array(out.buffer);
  for (let b = 0; b < nb; b++) {
    o8.set(u8.subarray(b * 210, (b + 1) * 210), b * 212);
  }
  return out;
}

/** IQ2_XXS 66 B/block → 68 B/block: raw copy + 2 pad bytes (qs ends @ 65). */
export function repackIQ2_XXS(data: ArrayBuffer | Uint8Array, n: number): Uint32Array {
  const u8 = asU8(data);
  if (n % 256 !== 0) throw new Error(`[repackIQ2_XXS] n=${n} not divisible by 256`);
  const nb = n / 256;
  assertBytes('repackIQ2_XXS', u8.byteLength, nb * 66);
  const out = new Uint32Array(nb * 17);
  const o8 = new Uint8Array(out.buffer);
  for (let b = 0; b < nb; b++) o8.set(u8.subarray(b * 66, (b + 1) * 66), b * 68);
  return out;
}

/**
 * Repack/align a GGUF tensor for GPU upload (matmul_gguf.wgsl layouts).
 * Q4_K/Q5_K block sizes are already 4-byte multiples — aligned copy only.
 */
export function repackGGUFForGPU(ggmlType: number, data: ArrayBuffer | Uint8Array, n: number): Uint32Array {
  switch (ggmlType) {
    case T_Q4_0: return repackQ4_0(data, n);
    case T_IQ4_NL: return repackQ4_0(data, n);  // identical 18 B layout (f16 d | qs[16])
    case T_Q5_0: return repackQ5_0(data, n);
    case T_Q8_0: return repackQ8_0(data, n);
    case T_Q3_K: return repackQ3_K(data, n);
    case T_Q6_K: return repackQ6_K(data, n);
    case T_IQ2_XXS: return repackIQ2_XXS(data, n);
    case T_Q2_K:   // 84 B (4-aligned)
    case T_Q4_K:   // 144 B
    case T_Q5_K:   // 176 B
    case T_IQ4_XS: { // 136 B — all 4-byte multiples: aligned copy only
      const u8 = asU8(data);
      if (n % 256 !== 0) throw new Error(`[repackGGUF] n=${n} not divisible by 256`);
      const typeSize = ggmlType === T_Q2_K ? 84 : ggmlType === T_Q4_K ? 144 : ggmlType === T_Q5_K ? 176 : 136;
      assertBytes('repackGGUF', u8.byteLength, (n / 256) * typeSize);
      const out = new Uint32Array(u8.byteLength / 4);
      new Uint8Array(out.buffer).set(u8);
      return out;
    }
    default:
      throw new Error(`[GGUF-repack] No GPU layout for ggml type ${ggmlType}`);
  }
}

// ── Dispatch by ggml type ──────────────────────────────────────────────

/** Dequantize a full tensor buffer to f32 by ggml type id. */
export function dequantGGML(ggmlType: number, data: ArrayBuffer | Uint8Array, n: number): Float32Array {
  switch (ggmlType) {
    case T_F32:  return dequantF32(data, n);
    case T_F16:  return dequantF16(data, n);
    case T_BF16: return dequantBF16(data, n);
    case T_Q4_0: return dequantQ4_0(data, n);
    case T_Q5_0: return dequantQ5_0(data, n);
    case T_Q8_0: return dequantQ8_0(data, n);
    case T_Q2_K: return dequantQ2_K(data, n);
    case T_Q3_K: return dequantQ3_K(data, n);
    case T_Q4_K: return dequantQ4_K(data, n);
    case T_Q5_K: return dequantQ5_K(data, n);
    case T_Q6_K: return dequantQ6_K(data, n);
    case T_IQ2_XXS: return dequantIQ2_XXS(data, n);
    case T_IQ4_NL: return dequantIQ4_NL(data, n);
    case T_IQ4_XS: return dequantIQ4_XS(data, n);
    default:
      throw new Error(`[GGUF-dequant] No CPU reference for ggml type ${ggmlType}`);
  }
}
