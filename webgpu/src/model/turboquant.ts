/**
 * TurboQuant — KV Cache Compression for WebGPU Inference
 *
 * Implements Google's TurboQuant (ICLR 2026, arXiv:2504.19874) for
 * compressing attention Key/Value cache vectors at 3-4 bits per channel
 * with near-zero quality loss.
 *
 * Two-stage algorithm:
 *   Stage 1 (PolarQuant): Random rotation → optimal scalar quantization
 *   Stage 2 (QJL):        1-bit residual correction for unbiased inner products
 *
 * All heavy computation runs on the GPU via WGSL kernels.
 * This module handles the one-time setup: rotation matrix, codebook,
 * JL sign matrix, and GPU buffer uploads.
 */

// ── Lloyd-Max Optimal Codebook ─────────────────────────────────────────────
//
// After random rotation, each coordinate follows ~N(0, 1/d) in high dimensions.
// These are the optimal scalar quantizer centroids and decision thresholds
// for a standard normal distribution, scaled by 1/sqrt(d) at encode time.
//
// Values from Lloyd-Max algorithm applied to N(0,1):
//   b=1: 2 centroids  (symmetric ±)
//   b=2: 4 centroids  (symmetric ±±)
//   b=3: 8 centroids
//   b=4: 16 centroids

/** Centroids for optimal scalar quantization of N(0,1). Symmetric — only positive half stored. */
const LLOYD_MAX_CENTROIDS: Record<number, number[]> = {
  // b=1: threshold at 0, centroids at ±0.7979 (= sqrt(2/pi))
  1: [0.7979],
  // b=2: thresholds at 0, ±0.9816; centroids at ±0.4528, ±1.5104
  2: [0.4528, 1.5104],
  // b=3: 4 positive centroids
  3: [0.2451, 0.7560, 1.3440, 2.1520],
  // b=4: 8 positive centroids
  4: [0.1284, 0.3881, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326],
};

/** Decision thresholds between centroids (positive half, excluding 0). */
const LLOYD_MAX_THRESHOLDS: Record<number, number[]> = {
  // b=1: single threshold at 0 (handled implicitly by sign)
  1: [],
  // b=2: threshold between the two positive centroids
  2: [0.9816],
  // b=3: thresholds between the 4 positive centroids
  3: [0.5006, 1.0500, 1.7480],
  // b=4: thresholds between the 8 positive centroids
  4: [0.2582, 0.5224, 0.7996, 1.0993, 1.4371, 1.8435, 2.4008],
};

export interface TurboQuantCodebook {
  /** Bits per coordinate (1-4) */
  bits: number;
  /** Positive centroids (length = 2^(bits-1)) */
  centroids: Float32Array;
  /** Decision thresholds between positive centroids (length = 2^(bits-1) - 1) */
  thresholds: Float32Array;
  /** Total number of quantization levels = 2^bits */
  levels: number;
}

/**
 * Build the codebook for a given bit width.
 * Centroids are for a unit normal — the WGSL kernel scales by 1/sqrt(d).
 */
export function buildCodebook(bits: number): TurboQuantCodebook {
  if (bits < 1 || bits > 4) throw new Error(`TurboQuant supports 1-4 bits, got ${bits}`);

  const c = LLOYD_MAX_CENTROIDS[bits];
  const t = LLOYD_MAX_THRESHOLDS[bits];

  return {
    bits,
    centroids: new Float32Array(c),
    thresholds: new Float32Array(t),
    levels: 1 << bits,
  };
}


// ── Random Orthogonal Matrix (Rotation) ────────────────────────────────────
//
// TurboQuant's key insight: multiplying by a random orthogonal matrix makes
// coordinates near-independent with a known distribution, enabling optimal
// scalar quantization WITHOUT per-block scale/zero-point storage.
//
// We generate this via QR decomposition of a Gaussian random matrix.
// For typical head dimensions (d=64 or d=128), this is a 64×64 or 128×128
// matrix — 16KB or 64KB, generated once and reused for all vectors.

/**
 * Seeded PRNG (xoshiro128**) for reproducible matrix generation.
 * Deterministic across runs so encode/decode use the same matrices.
 */
function createRNG(seed: number): () => number {
  // Initialize state from seed using splitmix32
  let s0 = seed >>> 0;
  const splitmix = (): number => {
    s0 = (s0 + 0x9e3779b9) >>> 0;
    let z = s0;
    z = ((z ^ (z >>> 16)) * 0x85ebca6b) >>> 0;
    z = ((z ^ (z >>> 13)) * 0xc2b2ae35) >>> 0;
    return (z ^ (z >>> 16)) >>> 0;
  };

  let a = splitmix(), b = splitmix(), c = splitmix(), d = splitmix();

  // xoshiro128** returns float in [0, 1)
  return (): number => {
    const result = (((a * 5) << 7 | (a * 5) >>> 25) * 9) >>> 0;
    const t = b << 9;
    c ^= a; d ^= b; b ^= c; a ^= d;
    c ^= t;
    d = (d << 11 | d >>> 21);
    return result / 0x100000000;
  };
}

/**
 * Generate a random value from N(0,1) using Box-Muller transform.
 */
function gaussianRandom(rng: () => number): number {
  const u1 = rng();
  const u2 = rng();
  // Avoid log(0)
  const r = Math.sqrt(-2 * Math.log(Math.max(u1, 1e-10)));
  return r * Math.cos(2 * Math.PI * u2);
}

/**
 * QR decomposition via modified Gram-Schmidt.
 * Input: d×d matrix A (column-major Float32Array).
 * Output: Q (orthogonal, column-major Float32Array).
 */
function qrOrthogonal(A: Float32Array, d: number): Float32Array {
  const Q = new Float32Array(A);

  for (let j = 0; j < d; j++) {
    // Orthogonalize column j against all previous columns
    for (let k = 0; k < j; k++) {
      // dot product of column k and column j
      let dot = 0;
      for (let i = 0; i < d; i++) {
        dot += Q[k * d + i] * Q[j * d + i];
      }
      // Subtract projection
      for (let i = 0; i < d; i++) {
        Q[j * d + i] -= dot * Q[k * d + i];
      }
    }
    // Normalize column j
    let norm = 0;
    for (let i = 0; i < d; i++) {
      norm += Q[j * d + i] * Q[j * d + i];
    }
    norm = Math.sqrt(norm);
    if (norm > 1e-10) {
      for (let i = 0; i < d; i++) {
        Q[j * d + i] /= norm;
      }
    }
  }

  return Q;
}

/**
 * Generate a d×d random orthogonal matrix (the rotation matrix Pi).
 * Uses QR decomposition of a Gaussian random matrix.
 *
 * @param d   - Head dimension (e.g., 64 or 128)
 * @param seed - Deterministic seed (must match between encode and decode)
 * @returns Column-major d×d Float32Array
 */
export function generateRotationMatrix(d: number, seed = 42): Float32Array {
  const rng = createRNG(seed);
  const A = new Float32Array(d * d);

  // Fill with iid N(0,1) entries
  for (let i = 0; i < d * d; i++) {
    A[i] = gaussianRandom(rng);
  }

  return qrOrthogonal(A, d);
}


// ── QJL Sign Matrix ────────────────────────────────────────────────────────
//
// The Johnson-Lindenstrauss matrix S is a d×d matrix of iid N(0,1) entries.
// In the QJL step, we compute sign(S · residual) to get 1-bit corrections.
// Unlike the rotation matrix, S does NOT need to be orthogonal.

/**
 * Generate the d×d JL sign matrix for QJL residual correction.
 * Entries are N(0, 1/d) as required by the TurboQuant paper — this
 * ensures the correction scale factor is simply sqrt(pi/2).
 *
 * @param d   - Head dimension
 * @param seed - Deterministic seed (must match between encode and decode)
 * @returns Column-major d×d Float32Array
 */
export function generateJLMatrix(d: number, seed = 137): Float32Array {
  const rng = createRNG(seed);
  const S = new Float32Array(d * d);
  const scale = 1.0 / Math.sqrt(d);  // N(0, 1/d) = N(0,1) * (1/sqrt(d))

  for (let i = 0; i < d * d; i++) {
    S[i] = gaussianRandom(rng) * scale;
  }

  return S;
}


// ── GPU Buffer Setup ───────────────────────────────────────────────────────

export interface TurboQuantBuffers {
  /** d×d rotation matrix (column-major f32) */
  rotationMatrix: GPUBuffer;
  /** d×d JL matrix (column-major f32) */
  jlMatrix: GPUBuffer;
  /** Positive centroids (f32, length = 2^(bits-1)) */
  centroids: GPUBuffer;
  /** Decision thresholds (f32, length = 2^(bits-1) - 1) */
  thresholds: GPUBuffer;
  /** Params uniform buffer */
  params: GPUBuffer;
  /** Head dimension */
  headDim: number;
  /** Bits per coordinate */
  bits: number;
}

export interface TurboQuantConfig {
  /** Attention head dimension (typically 64 or 128) */
  headDim: number;
  /** Bits per coordinate for PolarQuant stage (1-4, recommend 3) */
  bits: number;
  /** Seed for rotation matrix */
  rotationSeed?: number;
  /** Seed for JL matrix */
  jlSeed?: number;
}

/**
 * Initialize TurboQuant: generate matrices, build codebook, upload to GPU.
 * Call once during model setup. All returned buffers are reused for every
 * encode/decode operation.
 */
export function initTurboQuant(
  device: GPUDevice,
  config: TurboQuantConfig,
): TurboQuantBuffers {
  const { headDim: d, bits, rotationSeed = 42, jlSeed = 137 } = config;

  // Generate CPU-side data
  const rotMatrix = generateRotationMatrix(d, rotationSeed);
  const jlMatrix = generateJLMatrix(d, jlSeed);
  const codebook = buildCodebook(bits);

  // Upload rotation matrix (d×d × 4 bytes)
  const rotBuf = device.createBuffer({
    size: d * d * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: 'turboquant-rotation',
    mappedAtCreation: true,
  });
  new Float32Array(rotBuf.getMappedRange()).set(rotMatrix);
  rotBuf.unmap();

  // Upload JL matrix
  const jlBuf = device.createBuffer({
    size: d * d * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: 'turboquant-jl',
    mappedAtCreation: true,
  });
  new Float32Array(jlBuf.getMappedRange()).set(jlMatrix);
  jlBuf.unmap();

  // Upload centroids — pad to minimum 16 bytes for WebGPU alignment
  const centroidData = codebook.centroids;
  const centBuf = device.createBuffer({
    size: Math.max(centroidData.byteLength, 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: 'turboquant-centroids',
    mappedAtCreation: true,
  });
  new Float32Array(centBuf.getMappedRange(0, centroidData.byteLength)).set(centroidData);
  centBuf.unmap();

  // Upload thresholds — pad to minimum 16 bytes
  // For b=1, thresholds is empty — still need a valid buffer
  const threshData = codebook.thresholds;
  const threshSize = Math.max(threshData.byteLength, 16);
  const threshBuf = device.createBuffer({
    size: threshSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: 'turboquant-thresholds',
    mappedAtCreation: true,
  });
  if (threshData.byteLength > 0) {
    new Float32Array(threshBuf.getMappedRange(0, threshData.byteLength)).set(threshData);
  }
  threshBuf.unmap();

  // Params uniform: [headDim, bits, numCentroids, numThresholds, out_vec_offset]
  // 5 fields to match the encode shader's Params struct (out_vec_offset defaults to 0)
  const paramsData = new Uint32Array([
    d,
    bits,
    centroidData.length,
    threshData.length,
    0, // out_vec_offset — default 0, overridden per-dispatch in forward pass
  ]);
  const paramsBuf = device.createBuffer({
    size: Math.max(paramsData.byteLength, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'turboquant-params',
    mappedAtCreation: true,
  });
  new Uint32Array(paramsBuf.getMappedRange(0, paramsData.byteLength)).set(paramsData);
  paramsBuf.unmap();

  return {
    rotationMatrix: rotBuf,
    jlMatrix: jlBuf,
    centroids: centBuf,
    thresholds: threshBuf,
    params: paramsBuf,
    headDim: d,
    bits,
  };
}


// ── CPU Reference Implementation (for testing) ────────────────────────────

/**
 * CPU-side TurboQuant encode for testing against GPU results.
 * Returns packed quantized data matching the GPU kernel output format.
 */
export function cpuEncode(
  vector: Float32Array,
  rotMatrix: Float32Array,
  jlMatrix: Float32Array,
  codebook: TurboQuantCodebook,
  d: number,
): { quantized: Uint32Array; signBits: Uint32Array; norm: number } {
  const bits = codebook.bits;
  const scale = 1.0 / Math.sqrt(d);

  // Stage 0: Normalize to unit vector
  let normSq = 0;
  for (let i = 0; i < d; i++) normSq += vector[i] * vector[i];
  const norm = Math.sqrt(normSq);
  const invNorm = norm > 1e-8 ? 1.0 / norm : 0;

  // Stage 1a: Rotate normalized vector — y = Pi · (x / ||x||)
  const rotated = new Float32Array(d);
  for (let i = 0; i < d; i++) {
    let sum = 0;
    for (let j = 0; j < d; j++) {
      sum += rotMatrix[i * d + j] * vector[j] * invNorm;
    }
    rotated[i] = sum;
  }

  // Stage 1b: Scalar quantize each coordinate
  // Normalize to unit-normal scale for codebook lookup
  const indices = new Uint32Array(d);
  const dequantized = new Float32Array(d);
  const centroids = codebook.centroids;
  const thresholds = codebook.thresholds;

  for (let i = 0; i < d; i++) {
    const val = rotated[i] * Math.sqrt(d); // scale to N(0,1)
    const absVal = Math.abs(val);
    const sign = val >= 0 ? 1.0 : -1.0;

    // Find which centroid bin this falls into (positive half)
    let bin = 0;
    for (let t = 0; t < thresholds.length; t++) {
      if (absVal > thresholds[t]) bin = t + 1;
    }

    // Index encoding: positive bins = bin, negative bins = numPositive + bin
    const numPositive = centroids.length;
    indices[i] = val >= 0 ? bin : numPositive + bin;

    // Dequantize for residual computation
    dequantized[i] = sign * centroids[bin] * scale;
  }

  // Pack indices into u32 values (floor(32 / bits) indices per u32)
  const indicesPerU32 = Math.floor(32 / bits);
  const packedLen = Math.ceil(d / indicesPerU32);
  const quantized = new Uint32Array(packedLen);

  for (let i = 0; i < d; i++) {
    const wordIdx = Math.floor(i / indicesPerU32);
    const bitOffset = (i % indicesPerU32) * bits;
    quantized[wordIdx] |= (indices[i] & ((1 << bits) - 1)) << bitOffset;
  }

  // Stage 2: QJL residual correction
  // residual = rotated - dequantized (in rotated space)
  const residual = new Float32Array(d);
  for (let i = 0; i < d; i++) {
    residual[i] = rotated[i] - dequantized[i];
  }

  // sign_bits = sign(S · residual)
  const signWordsLen = Math.ceil(d / 32);
  const signBits = new Uint32Array(signWordsLen);

  for (let i = 0; i < d; i++) {
    let dot = 0;
    for (let j = 0; j < d; j++) {
      dot += jlMatrix[i * d + j] * residual[j];
    }
    if (dot >= 0) {
      signBits[Math.floor(i / 32)] |= 1 << (i % 32);
    }
  }

  return { quantized, signBits, norm };
}

/**
 * CPU-side TurboQuant decode for testing against GPU results.
 */
export function cpuDecode(
  quantized: Uint32Array,
  signBits: Uint32Array,
  norm: number,
  rotMatrix: Float32Array,
  jlMatrix: Float32Array,
  codebook: TurboQuantCodebook,
  d: number,
): Float32Array {
  const bits = codebook.bits;
  const scale = 1.0 / Math.sqrt(d);
  const centroids = codebook.centroids;
  const numPositive = centroids.length;
  const indicesPerU32 = Math.floor(32 / bits);
  const mask = (1 << bits) - 1;

  // Unpack indices and dequantize (PolarQuant reconstruction in rotated space)
  const dequantized = new Float32Array(d);
  for (let i = 0; i < d; i++) {
    const wordIdx = Math.floor(i / indicesPerU32);
    const bitOffset = (i % indicesPerU32) * bits;
    const idx = (quantized[wordIdx] >> bitOffset) & mask;

    const isNegative = idx >= numPositive;
    const bin = isNegative ? idx - numPositive : idx;
    const sign = isNegative ? -1.0 : 1.0;

    dequantized[i] = sign * centroids[bin] * scale;
  }

  // QJL correction is disabled for vector reconstruction — it's an inner product
  // estimator, not a reconstruction estimator. High per-coordinate variance makes
  // MSE worse. QJL should be applied during attention (Q·K^T) where averaging
  // over d dimensions concentrates the estimate. See Phase 4 attention kernel.
  const jlScale = 0;

  const correction = new Float32Array(d);
  for (let i = 0; i < d; i++) {
    let sum = 0;
    for (let j = 0; j < d; j++) {
      // sign_vector[j] = +1 if bit set, -1 if not
      const bit = (signBits[Math.floor(j / 32)] >> (j % 32)) & 1;
      const signVal = bit ? 1.0 : -1.0;
      sum += jlMatrix[j * d + i] * signVal; // S^T: transpose access
    }
    correction[i] = sum * jlScale;
  }

  // Reconstructed in rotated space = dequantized + correction
  const reconstructedRotated = new Float32Array(d);
  for (let i = 0; i < d; i++) {
    reconstructedRotated[i] = dequantized[i] + correction[i];
  }

  // Inverse rotation: x_hat = Pi^T · y, then rescale by norm
  const output = new Float32Array(d);
  for (let i = 0; i < d; i++) {
    let sum = 0;
    for (let j = 0; j < d; j++) {
      sum += rotMatrix[j * d + i] * reconstructedRotated[j];
    }
    output[i] = sum * norm;
  }

  return output;
}

/**
 * Compute MSE between two vectors.
 */
export function computeMSE(a: Float32Array, b: Float32Array): number {
  let mse = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    mse += diff * diff;
  }
  return mse / a.length;
}

/**
 * Compute relative MSE: ||x - x_hat||^2 / ||x||^2
 */
export function computeRelativeMSE(original: Float32Array, reconstructed: Float32Array): number {
  let errorSq = 0;
  let normSq = 0;
  for (let i = 0; i < original.length; i++) {
    const diff = original[i] - reconstructed[i];
    errorSq += diff * diff;
    normSq += original[i] * original[i];
  }
  return normSq > 0 ? errorSq / normSq : 0;
}
