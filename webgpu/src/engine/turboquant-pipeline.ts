/**
 * TurboQuant Pipeline — GPU Encode/Decode for KV Cache Compression
 *
 * Wires TurboQuant WGSL kernels into the Artifex WebGPU engine.
 * Provides high-level encode() and decode() functions that:
 *   1. Accept f32 KV vectors
 *   2. Dispatch the GPU kernel
 *   3. Return compressed (or reconstructed) data
 *
 * Usage:
 *   const tq = await createTurboQuantPipeline(device, { headDim: 128, bits: 3 });
 *   const compressed = await tq.encode(kvVectors);
 *   const reconstructed = await tq.decode(compressed);
 */

import {
  createComputePipeline,
  createBindGroup,
  dispatch,
  workgroupCount,
} from './compute';
import {
  createStorageBuffer,
  readBuffer,
} from './buffers';
import {
  initTurboQuant,
  type TurboQuantConfig,
  type TurboQuantBuffers,
} from '../model/turboquant';

// Import shaders as raw strings (Vite handles ?raw imports)
import encodeShaderSource from '../shaders/turboquant_encode.wgsl?raw';
import decodeShaderSource from '../shaders/turboquant_decode.wgsl?raw';

/** Compressed KV cache data returned by encode(). */
export interface CompressedKV {
  /** Packed quantized indices (u32 array) */
  quantizedBuffer: GPUBuffer;
  /** QJL sign bits (u32 array) */
  signBitsBuffer: GPUBuffer;
  /** Norms per vector (f32 array) */
  normsBuffer: GPUBuffer;
  /** Number of vectors encoded */
  numVectors: number;
  /** Words per vector for quantized data */
  packedWordsPerVector: number;
  /** Words per vector for sign bits */
  signWordsPerVector: number;
  /** Total bytes used (quantized + sign bits + norms) */
  compressedBytes: number;
  /** Original bytes (uncompressed f32) */
  originalBytes: number;
  /** Compression ratio */
  ratio: number;
}

export interface TurboQuantPipeline {
  /** Encode f32 KV vectors to compressed format on the GPU. */
  encode(inputBuffer: GPUBuffer, numVectors: number): CompressedKV;

  /** Decode compressed KV vectors back to f32 on the GPU. */
  decode(compressed: CompressedKV): GPUBuffer;

  /** Encode and wait for GPU completion (for testing). */
  encodeAndWait(inputBuffer: GPUBuffer, numVectors: number): Promise<CompressedKV>;

  /** Decode and read back results to CPU (for testing). */
  decodeAndRead(compressed: CompressedKV): Promise<Float32Array>;

  /** Get the compression ratio for this configuration. */
  compressionRatio(): number;

  /** Get config info. */
  readonly config: TurboQuantConfig;
  readonly buffers: TurboQuantBuffers;
}

/**
 * Create a TurboQuant pipeline for encoding/decoding KV cache vectors.
 *
 * @param device - WebGPU device
 * @param config - Head dimension and bit width
 */
export function createTurboQuantPipeline(
  device: GPUDevice,
  config: TurboQuantConfig,
): TurboQuantPipeline {
  const { headDim: d, bits } = config;

  // Initialize matrices, codebook, and GPU buffers
  const tqBuffers = initTurboQuant(device, config);

  // Compile shader pipelines
  const encodePipeline = createComputePipeline(
    device, encodeShaderSource, 'encode', 'turboquant-encode',
  );
  const decodePipeline = createComputePipeline(
    device, decodeShaderSource, 'decode', 'turboquant-decode',
  );

  // Precompute sizes
  const indicesPerU32 = Math.floor(32 / bits);
  const packedWordsPerVector = Math.ceil(d / indicesPerU32);
  const signWordsPerVector = Math.ceil(d / 32);

  // Create reusable bind group for matrices/codebook (group 1) and params (group 2)
  // These don't change between encode/decode calls

  const matricesBindGroupEncode = createBindGroup(
    device, encodePipeline, 1,
    [
      { binding: 0, resource: { buffer: tqBuffers.rotationMatrix } },
      { binding: 1, resource: { buffer: tqBuffers.jlMatrix } },
      { binding: 2, resource: { buffer: tqBuffers.centroids } },
      { binding: 3, resource: { buffer: tqBuffers.thresholds } },
    ],
    'tq-matrices-encode',
  );

  const paramsBindGroupEncode = createBindGroup(
    device, encodePipeline, 2,
    [{ binding: 0, resource: { buffer: tqBuffers.params } }],
    'tq-params-encode',
  );

  // Decode shader doesn't use thresholds — only 3 bindings in group 1
  const matricesBindGroupDecode = createBindGroup(
    device, decodePipeline, 1,
    [
      { binding: 0, resource: { buffer: tqBuffers.rotationMatrix } },
      { binding: 1, resource: { buffer: tqBuffers.jlMatrix } },
      { binding: 2, resource: { buffer: tqBuffers.centroids } },
    ],
    'tq-matrices-decode',
  );

  const paramsBindGroupDecode = createBindGroup(
    device, decodePipeline, 2,
    [{ binding: 0, resource: { buffer: tqBuffers.params } }],
    'tq-params-decode',
  );

  // ── Encode ─────────────────────────────────────────────────────────────

  function encode(inputBuffer: GPUBuffer, numVectors: number): CompressedKV {
    const quantizedSize = numVectors * packedWordsPerVector * 4;
    const signBitsSize = numVectors * signWordsPerVector * 4;
    const normsSize = numVectors * 4; // 1 f32 per vector

    const quantizedBuffer = createStorageBuffer(
      device, null, quantizedSize, 'tq-quantized', true,
    );
    const signBitsBuffer = createStorageBuffer(
      device, null, signBitsSize, 'tq-signbits', true,
    );
    const normsBuffer = createStorageBuffer(
      device, null, normsSize, 'tq-norms', true,
    );

    // Bind group 0: input/output buffers (changes per call)
    const ioBindGroup = createBindGroup(
      device, encodePipeline, 0,
      [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: quantizedBuffer } },
        { binding: 2, resource: { buffer: signBitsBuffer } },
        { binding: 3, resource: { buffer: normsBuffer } },
      ],
      'tq-encode-io',
    );

    // Dispatch: one workgroup per vector
    dispatch(
      device, encodePipeline,
      [ioBindGroup, matricesBindGroupEncode, paramsBindGroupEncode],
      [numVectors],
      'turboquant-encode',
    );

    const originalBytes = numVectors * d * 4;
    const compressedBytes = quantizedSize + signBitsSize + normsSize;

    return {
      quantizedBuffer,
      signBitsBuffer,
      normsBuffer,
      numVectors,
      packedWordsPerVector,
      signWordsPerVector,
      compressedBytes,
      originalBytes,
      ratio: originalBytes / compressedBytes,
    };
  }

  // ── Decode ─────────────────────────────────────────────────────────────

  function decode(compressed: CompressedKV): GPUBuffer {
    const outputSize = compressed.numVectors * d * 4;
    const outputBuffer = createStorageBuffer(
      device, null, outputSize, 'tq-decoded', true,
    );

    const ioBindGroup = createBindGroup(
      device, decodePipeline, 0,
      [
        { binding: 0, resource: { buffer: compressed.quantizedBuffer } },
        { binding: 1, resource: { buffer: compressed.signBitsBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: compressed.normsBuffer } },
      ],
      'tq-decode-io',
    );

    dispatch(
      device, decodePipeline,
      [ioBindGroup, matricesBindGroupDecode, paramsBindGroupDecode],
      [compressed.numVectors],
      'turboquant-decode',
    );

    return outputBuffer;
  }

  // ── Testing helpers ────────────────────────────────────────────────────

  async function encodeAndWait(inputBuffer: GPUBuffer, numVectors: number): Promise<CompressedKV> {
    const result = encode(inputBuffer, numVectors);
    await device.queue.onSubmittedWorkDone();
    return result;
  }

  async function decodeAndRead(compressed: CompressedKV): Promise<Float32Array> {
    const outputBuffer = decode(compressed);
    await device.queue.onSubmittedWorkDone();
    const raw = await readBuffer(device, outputBuffer, compressed.numVectors * d * 4);
    outputBuffer.destroy();
    return new Float32Array(raw);
  }

  function compressionRatio(): number {
    // PolarQuant uses (bits-1) bits per coordinate for indices
    // QJL uses 1 bit per coordinate for sign correction
    // Total: bits per coordinate
    // Original: 32 bits (f32) per coordinate
    return 32 / bits;
  }

  return {
    encode,
    decode,
    encodeAndWait,
    decodeAndRead,
    compressionRatio,
    config,
    buffers: tqBuffers,
  };
}
