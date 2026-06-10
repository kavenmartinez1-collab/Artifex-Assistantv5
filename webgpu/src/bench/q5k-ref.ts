/**
 * JS reference implementation of Q5_K dequant-dot (llama.cpp-compatible).
 *
 * Used to validate the wasm32 SIMD kernel bit-for-bit on the integer path
 * (kernel-audit rule: validate every kernel against a reference before
 * trusting its numbers). Also the seed of Phase B's gguf-dequant.ts.
 */

export const QK_K = 256;
export const Q5K_BLOCK_BYTES = 176; // d(2) + dmin(2) + scales(12) + qh(32) + qs(128)
export const Q8_ACT_BLOCK_BYTES = 276; // d(4) + q(256) + bsums(16)

export function f16ToF32(h: number): number {
  const sign = (h & 0x8000) << 16;
  let exp = (h >> 10) & 0x1f;
  let man = h & 0x3ff;
  let bits: number;
  if (exp === 0) {
    if (man === 0) {
      bits = sign;
    } else {
      let e = 127 - 15 + 1;
      while (!(man & 0x400)) {
        man = (man << 1) & 0x7ff;
        e--;
      }
      man &= 0x3ff;
      bits = sign | (e << 23) | (man << 13);
    }
  } else if (exp === 31) {
    bits = sign | 0x7f800000 | (man << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  const buf = new DataView(new ArrayBuffer(4));
  buf.setUint32(0, bits >>> 0);
  return buf.getFloat32(0);
}

/** ggml get_scale_min_k4. */
export function getScaleMin(j: number, scales: Uint8Array): [number, number] {
  if (j < 4) {
    return [scales[j] & 63, scales[j + 4] & 63];
  }
  return [
    (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4),
    (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
  ];
}

export interface Q8Act {
  d: Float32Array; // per-block scale
  q: Int8Array; // quants
  bsums: Int16Array; // per-32 subgroup sums
}

/** Quantize f32 activations to Q8 blocks — mirrors the C q8_quantize exactly. */
export function q8Quantize(x: Float32Array): Q8Act {
  const nb = x.length / QK_K;
  const d = new Float32Array(nb);
  const q = new Int8Array(x.length);
  const bsums = new Int16Array(nb * 8);
  for (let i = 0; i < nb; i++) {
    let amax = 0;
    for (let j = 0; j < QK_K; j++) {
      const v = Math.abs(x[i * QK_K + j]);
      if (v > amax) amax = v;
    }
    const scale = Math.fround(amax / 127);
    const id = scale !== 0 ? Math.fround(1 / scale) : 0;
    d[i] = scale;
    for (let g = 0; g < 8; g++) {
      let sum = 0;
      for (let j = 0; j < 32; j++) {
        const v = Math.fround(x[i * QK_K + g * 32 + j] * id);
        let qq = Math.trunc(v + (v >= 0 ? 0.5 : -0.5));
        if (qq > 127) qq = 127;
        if (qq < -128) qq = -128;
        q[i * QK_K + g * 32 + j] = qq;
        sum += qq;
      }
      bsums[i * 8 + g] = sum;
    }
  }
  return { d, q, bsums };
}

/**
 * Dot of one Q5_K row with Q8 activations — integer path identical to the
 * wasm kernel (sumi/smin in exact integer arithmetic, f32 final combine).
 */
export function q5kDotRowRef(row: Uint8Array, act: Q8Act, nb: number): number {
  let sumf = 0;
  for (let i = 0; i < nb; i++) {
    const o = i * Q5K_BLOCK_BYTES;
    const dv = new DataView(row.buffer, row.byteOffset + o, Q5K_BLOCK_BYTES);
    const d = Math.fround(f16ToF32(dv.getUint16(0, true)) * act.d[i]);
    const dmin = Math.fround(f16ToF32(dv.getUint16(2, true)) * act.d[i]);
    const scales = row.subarray(o + 4, o + 16);
    const qh = row.subarray(o + 16, o + 48);
    const qs = row.subarray(o + 48, o + 176);

    let smin = 0;
    const sc = new Uint8Array(8);
    for (let j = 0; j < 8; j++) {
      const [s, m] = getScaleMin(j, scales);
      sc[j] = s;
      smin += m * act.bsums[i * 8 + j];
    }

    let sumi = 0;
    for (let j = 0; j < 4; j++) {
      // 64 elements: 32 low nibbles (qh bit 2j), 32 high nibbles (qh bit 2j+1)
      let isumLo = 0;
      let isumHi = 0;
      for (let l = 0; l < 32; l++) {
        const ql = qs[j * 32 + l];
        const wLo = (ql & 0x0f) | (((qh[l] >> (2 * j)) & 1) << 4);
        const wHi = (ql >> 4) | (((qh[l] >> (2 * j + 1)) & 1) << 4);
        isumLo += wLo * act.q[i * QK_K + j * 64 + l];
        isumHi += wHi * act.q[i * QK_K + j * 64 + 32 + l];
      }
      sumi += sc[2 * j] * isumLo + sc[2 * j + 1] * isumHi;
    }

    sumf = Math.fround(sumf + Math.fround(d * sumi) - Math.fround(dmin * smin));
  }
  return sumf;
}

/** Deterministic xorshift32 byte fill (bench data generation). */
export function xorshiftFill(buf: Uint8Array, seed: number): void {
  let s = seed >>> 0 || 0x9e3779b9;
  const words = new Uint32Array(buf.buffer, buf.byteOffset, buf.byteLength >> 2);
  for (let i = 0; i < words.length; i++) {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5;
    s >>>= 0;
    words[i] = s;
  }
}

/**
 * Stamp sane f16 d/dmin into every Q5_K block of a weight buffer so random
 * bytes can't produce inf/nan super-scales (d≈0.1, dmin≈0.05).
 */
export function stampBlockScales(weights: Uint8Array): void {
  for (let o = 0; o + Q5K_BLOCK_BYTES <= weights.byteLength; o += Q5K_BLOCK_BYTES) {
    weights[o] = 0x66;
    weights[o + 1] = 0x2e; // d ≈ 0.0999 (f16 0x2E66, little-endian)
    weights[o + 2] = 0x66;
    weights[o + 3] = 0x2a; // dmin ≈ 0.05 (f16 0x2A66)
  }
}
