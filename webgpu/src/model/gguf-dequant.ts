/**
 * CPU reference dequantization for GGUF tensor types.
 *
 * This is the ground truth that every WGSL k-quant kernel is validated
 * against (kernel-audit rule). Implementations mirror llama.cpp
 * ggml-quants.c dequantize_row_* exactly — bit layouts below.
 *
 * Block layouts (little-endian):
 *   Q8_0 (32 elems, 34 B):  f16 d | i8 qs[32]              x = d·qs
 *   Q4_K (256, 144 B):      f16 d, dmin | u8 scales[12] | u8 qs[128] (4-bit pairs)
 *   Q5_K (256, 176 B):      f16 d, dmin | u8 scales[12] | u8 qh[32] | u8 qs[128]
 *   Q6_K (256, 210 B):      u8 ql[128] | u8 qh[64] | i8 scales[16] | f16 d
 *
 * Q4_K/Q5_K 6-bit scale/min packing (get_scale_min_k4, j in 0..7):
 *   j < 4:  sc = q[j] & 63;              m = q[j+4] & 63
 *   j >= 4: sc = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
 *           m  = (q[j+4] >>  4) | ((q[j]   >> 6) << 4)
 */

// GGML type ids (spec constants, duplicated from gguf.ts so this file has
// zero imports and runs standalone under node --strip-types for validation)
const T_F32 = 0, T_F16 = 1, T_Q8_0 = 8, T_Q4_K = 12, T_Q5_K = 13, T_Q6_K = 14, T_BF16 = 30;

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

// ── GPU repack (4-byte-aligned block strides for matmul_gguf.wgsl) ─────

/** Per-type GPU layout: block element count and u32 stride after repack. */
export const GGUF_GPU_LAYOUT: Record<number, { blockElems: number; strideU32: number }> = {
  [T_Q8_0]: { blockElems: 32, strideU32: 9 },    // 34 B → 36 B
  [T_Q4_K]: { blockElems: 256, strideU32: 36 },  // raw 144 B
  [T_Q5_K]: { blockElems: 256, strideU32: 44 },  // raw 176 B
  [T_Q6_K]: { blockElems: 256, strideU32: 53 },  // 210 B → 212 B
};

function asU8(data: ArrayBuffer | Uint8Array): Uint8Array {
  return data instanceof Uint8Array ? data : new Uint8Array(data);
}

function assertBytes(label: string, got: number, want: number): void {
  if (got !== want) throw new Error(`[${label}] byteLength ${got} != expected ${want}`);
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

/**
 * Repack/align a GGUF tensor for GPU upload (matmul_gguf.wgsl layouts).
 * Q4_K/Q5_K block sizes are already 4-byte multiples — aligned copy only.
 */
export function repackGGUFForGPU(ggmlType: number, data: ArrayBuffer | Uint8Array, n: number): Uint32Array {
  switch (ggmlType) {
    case T_Q8_0: return repackQ8_0(data, n);
    case T_Q6_K: return repackQ6_K(data, n);
    case T_Q4_K:
    case T_Q5_K: {
      const u8 = asU8(data);
      if (n % 256 !== 0) throw new Error(`[repackGGUF] n=${n} not divisible by 256`);
      const typeSize = ggmlType === T_Q4_K ? 144 : 176;
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
    case T_Q8_0: return dequantQ8_0(data, n);
    case T_Q4_K: return dequantQ4_K(data, n);
    case T_Q5_K: return dequantQ5_K(data, n);
    case T_Q6_K: return dequantQ6_K(data, n);
    default:
      throw new Error(`[GGUF-dequant] No CPU reference for ggml type ${ggmlType}`);
  }
}
