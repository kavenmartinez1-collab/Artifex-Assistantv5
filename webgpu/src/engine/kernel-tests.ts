/**
 * GPU Kernel Tests — Validates WGSL shaders against CPU reference values.
 * Reports results via metrics webhook.
 */

import { createStorageBuffer, createUniformBuffer, readBuffer } from './buffers';
import { createComputePipeline, createBindGroup, dispatchAndWait, workgroupCount } from './compute';

import elementwiseWGSL from '../shaders/elementwise.wgsl?raw';
import matmulWGSL from '../shaders/matmul.wgsl?raw';
import softmaxWGSL from '../shaders/softmax.wgsl?raw';
import rmsnormWGSL from '../shaders/rmsnorm.wgsl?raw';
import ropeWGSL from '../shaders/rope.wgsl?raw';
import { createTurboQuantPipeline } from './turboquant-pipeline';
import {
  generateRotationMatrix, generateJLMatrix, generateSPiMatrix, buildCodebook,
  cpuEncode, cpuDecode, computeMSE, computeRelativeMSE, cpuAsymmetricScore,
} from '../model/turboquant';

export interface TestResult {
  name: string;
  passed: boolean;
  elapsed_ms?: number;
  error?: string;
  maxDiff?: number;
}

const TOLERANCE = 1e-4;

function arrClose(a: Float32Array, b: Float32Array, tol = TOLERANCE): { close: boolean; maxDiff: number } {
  let maxDiff = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = Math.abs(a[i] - b[i]);
    if (diff > maxDiff) maxDiff = diff;
  }
  return { close: maxDiff < tol, maxDiff };
}

// ─── CPU Reference Implementations ──────────────────────────────────────────

function cpuSiLU(x: Float32Array): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = x[i] / (1 + Math.exp(-x[i]));
  }
  return out;
}

function cpuAdd(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

function cpuMul(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] * b[i];
  return out;
}

function cpuMatmul(a: Float32Array, b: Float32Array, M: number, N: number, K: number): Float32Array {
  const out = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += a[i * K + k] * b[k * N + j];
      }
      out[i * N + j] = sum;
    }
  }
  return out;
}

/** C[M,N] = A[M,K] @ B^T where B is stored as [N,K] (row-major, HF weight layout) */
function cpuMatmulBT(a: Float32Array, b: Float32Array, M: number, N: number, K: number): Float32Array {
  const out = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += a[i * K + k] * b[j * K + k];
      }
      out[i * N + j] = sum;
    }
  }
  return out;
}

/** Encode f32 values as BF16 packed into u32 pairs (little-endian: even=low16, odd=high16) */
function f32ToBF16Packed(values: Float32Array): Uint32Array {
  if (values.length % 2 !== 0) throw new Error('BF16 packing requires even element count');
  const packed = new Uint32Array(values.length / 2);
  const f32View = new DataView(new ArrayBuffer(4));
  for (let i = 0; i < values.length; i += 2) {
    // BF16 = upper 16 bits of f32
    f32View.setFloat32(0, values[i], true);
    const bf16_even = (f32View.getUint32(0, true) >>> 16) & 0xFFFF;
    f32View.setFloat32(0, values[i + 1], true);
    const bf16_odd = (f32View.getUint32(0, true) >>> 16) & 0xFFFF;
    packed[i / 2] = bf16_even | (bf16_odd << 16);
  }
  return packed;
}

function cpuSoftmax(x: Float32Array, cols: number): Float32Array {
  const rows = x.length / cols;
  const out = new Float32Array(x.length);
  for (let r = 0; r < rows; r++) {
    let max = -Infinity;
    for (let c = 0; c < cols; c++) max = Math.max(max, x[r * cols + c]);
    let sum = 0;
    for (let c = 0; c < cols; c++) {
      out[r * cols + c] = Math.exp(x[r * cols + c] - max);
      sum += out[r * cols + c];
    }
    for (let c = 0; c < cols; c++) out[r * cols + c] /= sum;
  }
  return out;
}

function cpuRMSNorm(x: Float32Array, weight: Float32Array, hidden: number, eps: number): Float32Array {
  const rows = x.length / hidden;
  const out = new Float32Array(x.length);
  for (let r = 0; r < rows; r++) {
    let sumSq = 0;
    for (let c = 0; c < hidden; c++) sumSq += x[r * hidden + c] ** 2;
    const rmsInv = 1 / Math.sqrt(sumSq / hidden + eps);
    for (let c = 0; c < hidden; c++) {
      out[r * hidden + c] = x[r * hidden + c] * rmsInv * weight[c];
    }
  }
  return out;
}

// ─── Test Runner ─────────────────────────────────────────────────────────────

async function testSiLU(device: GPUDevice): Promise<TestResult> {
  const N = 1024;
  const input = new Float32Array(N);
  for (let i = 0; i < N; i++) input[i] = (Math.random() - 0.5) * 4;

  const expected = cpuSiLU(input);

  const inputBuf = createStorageBuffer(device, input, N * 4, 'silu-input');
  const outputBuf = createStorageBuffer(device, null, N * 4, 'silu-output', true);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([N]), 'silu-params');

  const pipeline = createComputePipeline(device, elementwiseWGSL, 'silu', 'test-silu');
  const bg = createBindGroup(device, pipeline, 0, [
    { binding: 0, resource: { buffer: inputBuf } },
    { binding: 1, resource: { buffer: outputBuf } },
    { binding: 2, resource: { buffer: paramsBuf } },
  ]);

  const elapsed = await dispatchAndWait(device, pipeline, [bg], [workgroupCount(N, 256)], 'silu');

  const result = new Float32Array(await readBuffer(device, outputBuf));
  const { close, maxDiff } = arrClose(result, expected);

  inputBuf.destroy(); outputBuf.destroy(); paramsBuf.destroy();
  return { name: 'SiLU', passed: close, elapsed_ms: elapsed, maxDiff };
}

async function testAdd(device: GPUDevice): Promise<TestResult> {
  const N = 1024;
  const a = new Float32Array(N);
  const b = new Float32Array(N);
  for (let i = 0; i < N; i++) { a[i] = Math.random() * 2 - 1; b[i] = Math.random() * 2 - 1; }

  const expected = cpuAdd(a, b);

  const aBuf = createStorageBuffer(device, a, N * 4, 'add-a');
  const outBuf = createStorageBuffer(device, null, N * 4, 'add-out', true);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([N]), 'add-params');
  const bBuf = createStorageBuffer(device, b, N * 4, 'add-b');

  const pipeline = createComputePipeline(device, elementwiseWGSL, 'add', 'test-add');
  const bg = createBindGroup(device, pipeline, 0, [
    { binding: 0, resource: { buffer: aBuf } },
    { binding: 1, resource: { buffer: outBuf } },
    { binding: 2, resource: { buffer: paramsBuf } },
    { binding: 3, resource: { buffer: bBuf } },
  ]);

  const elapsed = await dispatchAndWait(device, pipeline, [bg], [workgroupCount(N, 256)], 'add');
  const result = new Float32Array(await readBuffer(device, outBuf));
  const { close, maxDiff } = arrClose(result, expected);

  aBuf.destroy(); outBuf.destroy(); paramsBuf.destroy(); bBuf.destroy();
  return { name: 'Add', passed: close, elapsed_ms: elapsed, maxDiff };
}

async function testMul(device: GPUDevice): Promise<TestResult> {
  const N = 1024;
  const a = new Float32Array(N);
  const b = new Float32Array(N);
  for (let i = 0; i < N; i++) { a[i] = Math.random() * 2 - 1; b[i] = Math.random() * 2 - 1; }

  const expected = cpuMul(a, b);

  const aBuf = createStorageBuffer(device, a, N * 4, 'mul-a');
  const outBuf = createStorageBuffer(device, null, N * 4, 'mul-out', true);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([N]), 'mul-params');
  const bBuf = createStorageBuffer(device, b, N * 4, 'mul-b');

  const pipeline = createComputePipeline(device, elementwiseWGSL, 'mul', 'test-mul');
  const bg = createBindGroup(device, pipeline, 0, [
    { binding: 0, resource: { buffer: aBuf } },
    { binding: 1, resource: { buffer: outBuf } },
    { binding: 2, resource: { buffer: paramsBuf } },
    { binding: 3, resource: { buffer: bBuf } },
  ]);

  const elapsed = await dispatchAndWait(device, pipeline, [bg], [workgroupCount(N, 256)], 'mul');
  const result = new Float32Array(await readBuffer(device, outBuf));
  const { close, maxDiff } = arrClose(result, expected);

  aBuf.destroy(); outBuf.destroy(); paramsBuf.destroy(); bBuf.destroy();
  return { name: 'Multiply', passed: close, elapsed_ms: elapsed, maxDiff };
}

async function testMatmulNaive(device: GPUDevice): Promise<TestResult> {
  const M = 32, K = 64, N = 32;
  const a = new Float32Array(M * K);
  const b = new Float32Array(K * N);
  for (let i = 0; i < a.length; i++) a[i] = Math.random() * 0.5 - 0.25;
  for (let i = 0; i < b.length; i++) b[i] = Math.random() * 0.5 - 0.25;

  const expected = cpuMatmul(a, b, M, N, K);

  const aBuf = createStorageBuffer(device, a, a.byteLength, 'mm-a');
  const bBuf = createStorageBuffer(device, b, b.byteLength, 'mm-b');
  const cBuf = createStorageBuffer(device, null, M * N * 4, 'mm-c', true);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([M, N, K]), 'mm-params');

  const pipeline = createComputePipeline(device, matmulWGSL, 'matmul_naive', 'test-mm-naive');
  const bg = createBindGroup(device, pipeline, 0, [
    { binding: 0, resource: { buffer: aBuf } },
    { binding: 1, resource: { buffer: bBuf } },
    { binding: 2, resource: { buffer: cBuf } },
    { binding: 3, resource: { buffer: paramsBuf } },
  ]);

  const elapsed = await dispatchAndWait(device, pipeline, [bg], [workgroupCount(M * N, 256)], 'mm-naive');
  const result = new Float32Array(await readBuffer(device, cBuf));
  const { close, maxDiff } = arrClose(result, expected, 1e-3); // matmul accumulates error

  aBuf.destroy(); bBuf.destroy(); cBuf.destroy(); paramsBuf.destroy();
  return { name: 'Matmul (naive)', passed: close, elapsed_ms: elapsed, maxDiff };
}

async function testMatmulTiled(device: GPUDevice): Promise<TestResult> {
  const M = 64, K = 64, N = 64;
  const a = new Float32Array(M * K);
  const b = new Float32Array(K * N);
  for (let i = 0; i < a.length; i++) a[i] = Math.random() * 0.5 - 0.25;
  for (let i = 0; i < b.length; i++) b[i] = Math.random() * 0.5 - 0.25;

  const expected = cpuMatmul(a, b, M, N, K);

  const aBuf = createStorageBuffer(device, a, a.byteLength, 'tmm-a');
  const bBuf = createStorageBuffer(device, b, b.byteLength, 'tmm-b');
  const cBuf = createStorageBuffer(device, null, M * N * 4, 'tmm-c', true);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([M, N, K]), 'tmm-params');

  const pipeline = createComputePipeline(device, matmulWGSL, 'matmul', 'test-mm-tiled');
  const bg = createBindGroup(device, pipeline, 0, [
    { binding: 0, resource: { buffer: aBuf } },
    { binding: 1, resource: { buffer: bBuf } },
    { binding: 2, resource: { buffer: cBuf } },
    { binding: 3, resource: { buffer: paramsBuf } },
  ]);

  const TILE = 16;
  const elapsed = await dispatchAndWait(device, pipeline, [bg],
    [Math.ceil(M / TILE), Math.ceil(N / TILE)], 'mm-tiled');
  const result = new Float32Array(await readBuffer(device, cBuf));
  const { close, maxDiff } = arrClose(result, expected, 1e-3);

  aBuf.destroy(); bBuf.destroy(); cBuf.destroy(); paramsBuf.destroy();
  return { name: 'Matmul (tiled)', passed: close, elapsed_ms: elapsed, maxDiff };
}

async function testSoftmax(device: GPUDevice): Promise<TestResult> {
  const rows = 4, cols = 128;
  const input = new Float32Array(rows * cols);
  for (let i = 0; i < input.length; i++) input[i] = Math.random() * 4 - 2;

  const expected = cpuSoftmax(input, cols);

  const inBuf = createStorageBuffer(device, input, input.byteLength, 'sm-in');
  const outBuf = createStorageBuffer(device, null, input.byteLength, 'sm-out', true);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([cols, rows]), 'sm-params');

  const pipeline = createComputePipeline(device, softmaxWGSL, 'softmax', 'test-softmax');
  const bg = createBindGroup(device, pipeline, 0, [
    { binding: 0, resource: { buffer: inBuf } },
    { binding: 1, resource: { buffer: outBuf } },
    { binding: 2, resource: { buffer: paramsBuf } },
  ]);

  const elapsed = await dispatchAndWait(device, pipeline, [bg], [rows], 'softmax');
  const result = new Float32Array(await readBuffer(device, outBuf));
  const { close, maxDiff } = arrClose(result, expected, 1e-3);

  inBuf.destroy(); outBuf.destroy(); paramsBuf.destroy();
  return { name: 'Softmax', passed: close, elapsed_ms: elapsed, maxDiff };
}

async function testRMSNorm(device: GPUDevice): Promise<TestResult> {
  const hidden = 256;
  const rows = 2;
  const eps = 1e-6;

  const input = new Float32Array(rows * hidden);
  const weight = new Float32Array(hidden);
  for (let i = 0; i < input.length; i++) input[i] = Math.random() * 2 - 1;
  for (let i = 0; i < hidden; i++) weight[i] = 0.5 + Math.random();

  const expected = cpuRMSNorm(input, weight, hidden, eps);

  const inBuf = createStorageBuffer(device, input, input.byteLength, 'rms-in');
  const outBuf = createStorageBuffer(device, null, input.byteLength, 'rms-out', true);
  const wBuf = createStorageBuffer(device, weight, weight.byteLength, 'rms-w');

  // Params: hidden_size (u32) + eps (f32)
  const paramsData = new ArrayBuffer(8);
  new Uint32Array(paramsData, 0, 1)[0] = hidden;
  new Float32Array(paramsData, 4, 1)[0] = eps;
  const paramsBuf = createUniformBuffer(device, new Uint8Array(paramsData), 'rms-params');

  const pipeline = createComputePipeline(device, rmsnormWGSL, 'rmsnorm', 'test-rmsnorm');
  const bg = createBindGroup(device, pipeline, 0, [
    { binding: 0, resource: { buffer: inBuf } },
    { binding: 1, resource: { buffer: outBuf } },
    { binding: 2, resource: { buffer: wBuf } },
    { binding: 3, resource: { buffer: paramsBuf } },
  ]);

  const elapsed = await dispatchAndWait(device, pipeline, [bg], [rows], 'rmsnorm');
  const result = new Float32Array(await readBuffer(device, outBuf));
  const { close, maxDiff } = arrClose(result, expected, 1e-3);

  inBuf.destroy(); outBuf.destroy(); wBuf.destroy(); paramsBuf.destroy();
  return { name: 'RMSNorm', passed: close, elapsed_ms: elapsed, maxDiff };
}

// ─── B-Transposed Matmul Tests ───────────────────────────────────────────────

async function testMatmulBT(device: GPUDevice): Promise<TestResult> {
  const M = 1, K = 64, N = 32;
  const a = new Float32Array(M * K);
  const b = new Float32Array(N * K); // [N, K] layout (B-transposed)
  for (let i = 0; i < a.length; i++) a[i] = Math.random() * 0.5 - 0.25;
  for (let i = 0; i < b.length; i++) b[i] = Math.random() * 0.5 - 0.25;

  const expected = cpuMatmulBT(a, b, M, N, K);

  const aBuf = createStorageBuffer(device, a, a.byteLength, 'bt-a');
  const bBuf = createStorageBuffer(device, b, b.byteLength, 'bt-b');
  const cBuf = createStorageBuffer(device, null, M * N * 4, 'bt-c', true);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([M, N, K, 0]), 'bt-params');

  const pipeline = createComputePipeline(device, matmulWGSL, 'matmul_bt', 'test-mm-bt');
  const bg = createBindGroup(device, pipeline, 0, [
    { binding: 0, resource: { buffer: aBuf } },
    { binding: 1, resource: { buffer: bBuf } },
    { binding: 2, resource: { buffer: cBuf } },
    { binding: 3, resource: { buffer: paramsBuf } },
  ]);

  const TILE = 16;
  const elapsed = await dispatchAndWait(device, pipeline, [bg],
    [Math.ceil(M / TILE), Math.ceil(N / TILE)], 'mm-bt');
  const result = new Float32Array(await readBuffer(device, cBuf));
  const { close, maxDiff } = arrClose(result, expected, 1e-3);

  aBuf.destroy(); bBuf.destroy(); cBuf.destroy(); paramsBuf.destroy();
  return { name: 'Matmul BT (f32)', passed: close, elapsed_ms: elapsed, maxDiff };
}

async function testMatmulBTBF16Small(device: GPUDevice): Promise<TestResult> {
  // Small test: M=1, N=4, K=8 (single tile, verifies BF16 decode + accumulation)
  const M = 1, K = 8, N = 4;
  const a = new Float32Array(M * K);
  const bF32 = new Float32Array(N * K); // [N, K] layout
  for (let i = 0; i < a.length; i++) a[i] = (i + 1) * 0.1; // [0.1, 0.2, ..., 0.8]
  for (let i = 0; i < bF32.length; i++) bF32[i] = ((i % K) + 1) * 0.25; // repeating pattern

  const expected = cpuMatmulBT(a, bF32, M, N, K);
  const bPacked = f32ToBF16Packed(bF32); // [N * K / 2] u32 values

  // Pre-fill output with sentinel (99.0) to distinguish "writes zeros" from "doesn't write"
  const sentinel = new Float32Array(M * N).fill(99.0);
  const aBuf = createStorageBuffer(device, a, a.byteLength, 'bf16s-a');
  const bBuf = createStorageBuffer(device, bPacked, bPacked.byteLength, 'bf16s-b');
  const cBuf = createStorageBuffer(device, sentinel, M * N * 4, 'bf16s-c', true);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([M, N, K, 0]), 'bf16s-params');

  const pipeline = createComputePipeline(device, matmulWGSL, 'matmul_bt_bf16', 'test-mm-bt-bf16-s');
  const bg = createBindGroup(device, pipeline, 0, [
    { binding: 0, resource: { buffer: aBuf } },
    { binding: 2, resource: { buffer: cBuf } },
    { binding: 3, resource: { buffer: paramsBuf } },
    { binding: 5, resource: { buffer: bBuf } },
  ]);

  const TILE = 16;
  const elapsed = await dispatchAndWait(device, pipeline, [bg],
    [Math.ceil(M / TILE), Math.ceil(N / TILE)], 'mm-bt-bf16-s');
  const result = new Float32Array(await readBuffer(device, cBuf));
  // BF16 truncates mantissa so allow slightly larger tolerance
  const { close, maxDiff } = arrClose(result, expected, 5e-3);

  if (!close) {
    const allSentinel = Array.from(result).every(v => v === 99.0);
    const allZero = Array.from(result).every(v => v === 0.0);
    console.log(`[BF16 small] expected: [${Array.from(expected).map(v => v.toFixed(4)).join(', ')}]`);
    console.log(`[BF16 small] got:      [${Array.from(result).map(v => v.toFixed(4)).join(', ')}]`);
    console.log(`[BF16 small] sentinel=${allSentinel} (kernel never wrote), allZero=${allZero} (kernel wrote zeros)`);
    console.log(`[BF16 small] A: [${Array.from(a).map(v => v.toFixed(4)).join(', ')}]`);
    console.log(`[BF16 small] B packed u32: [${Array.from(bPacked).map(v => '0x' + v.toString(16).padStart(8, '0')).join(', ')}]`);
  }

  aBuf.destroy(); bBuf.destroy(); cBuf.destroy(); paramsBuf.destroy();
  return { name: 'Matmul BT BF16 (small)', passed: close, elapsed_ms: elapsed, maxDiff };
}

async function testMatmulBTBF16Large(device: GPUDevice): Promise<TestResult> {
  // Larger test: M=1, N=64, K=128 — multiple tiles, realistic projection sizes
  const M = 1, K = 128, N = 64;
  const a = new Float32Array(M * K);
  const bF32 = new Float32Array(N * K);
  for (let i = 0; i < a.length; i++) a[i] = Math.random() * 0.5 - 0.25;
  for (let i = 0; i < bF32.length; i++) bF32[i] = Math.random() * 0.5 - 0.25;

  const expected = cpuMatmulBT(a, bF32, M, N, K);
  const bPacked = f32ToBF16Packed(bF32);

  const aBuf = createStorageBuffer(device, a, a.byteLength, 'bf16l-a');
  const bBuf = createStorageBuffer(device, bPacked, bPacked.byteLength, 'bf16l-b');
  const cBuf = createStorageBuffer(device, null, M * N * 4, 'bf16l-c', true);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([M, N, K, 0]), 'bf16l-params');

  const pipeline = createComputePipeline(device, matmulWGSL, 'matmul_bt_bf16', 'test-mm-bt-bf16-l');
  const bg = createBindGroup(device, pipeline, 0, [
    { binding: 0, resource: { buffer: aBuf } },
    { binding: 2, resource: { buffer: cBuf } },
    { binding: 3, resource: { buffer: paramsBuf } },
    { binding: 5, resource: { buffer: bBuf } },
  ]);

  const TILE = 16;
  const elapsed = await dispatchAndWait(device, pipeline, [bg],
    [Math.ceil(M / TILE), Math.ceil(N / TILE)], 'mm-bt-bf16-l');
  const result = new Float32Array(await readBuffer(device, cBuf));
  const { close, maxDiff } = arrClose(result, expected, 5e-3);

  if (!close) {
    console.log(`[BF16 large] expected first 8: [${Array.from(expected.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);
    console.log(`[BF16 large] got first 8:      [${Array.from(result.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);
  }

  aBuf.destroy(); bBuf.destroy(); cBuf.destroy(); paramsBuf.destroy();
  return { name: 'Matmul BT BF16 (large)', passed: close, elapsed_ms: elapsed, maxDiff };
}

async function testMatmulBTBF16RealDims(device: GPUDevice): Promise<TestResult> {
  // Real model dimensions: M=1, K=2048 (hidden_size), N=5632 (intermediate_size)
  const M = 1, K = 2048, N = 256; // Use N=256 (not 5632) to avoid huge buffers in test
  const a = new Float32Array(M * K);
  const bF32 = new Float32Array(N * K);
  for (let i = 0; i < a.length; i++) a[i] = (Math.random() - 0.5) * 0.1;
  for (let i = 0; i < bF32.length; i++) bF32[i] = (Math.random() - 0.5) * 0.1;

  const expected = cpuMatmulBT(a, bF32, M, N, K);
  const bPacked = f32ToBF16Packed(bF32);

  const sentinel = new Float32Array(M * N).fill(99.0);
  const aBuf = createStorageBuffer(device, a, a.byteLength, 'bf16r-a');
  const bBuf = createStorageBuffer(device, bPacked, bPacked.byteLength, 'bf16r-b');
  const cBuf = createStorageBuffer(device, sentinel, M * N * 4, 'bf16r-c', true);
  const paramsBuf = createUniformBuffer(device, new Uint32Array([M, N, K, 0]), 'bf16r-params');

  const pipeline = createComputePipeline(device, matmulWGSL, 'matmul_bt_bf16', 'test-mm-bt-bf16-r');
  const bg = createBindGroup(device, pipeline, 0, [
    { binding: 0, resource: { buffer: aBuf } },
    { binding: 2, resource: { buffer: cBuf } },
    { binding: 3, resource: { buffer: paramsBuf } },
    { binding: 5, resource: { buffer: bBuf } },
  ]);

  const TILE = 16;
  const elapsed = await dispatchAndWait(device, pipeline, [bg],
    [Math.ceil(M / TILE), Math.ceil(N / TILE)], 'mm-bt-bf16-r');
  const result = new Float32Array(await readBuffer(device, cBuf));
  const { close, maxDiff } = arrClose(result, expected, 0.02); // BF16 + K=2048 accumulation

  if (!close) {
    const allSentinel = Array.from(result.slice(0, 8)).every(v => v === 99.0);
    const allZero = Array.from(result.slice(0, 8)).every(v => v === 0.0);
    console.log(`[BF16 real] expected[0:8]: [${Array.from(expected.slice(0, 8)).map(v => v.toFixed(6)).join(', ')}]`);
    console.log(`[BF16 real] got[0:8]:      [${Array.from(result.slice(0, 8)).map(v => v.toFixed(6)).join(', ')}]`);
    console.log(`[BF16 real] sentinel=${allSentinel}, allZero=${allZero}, M=${M} N=${N} K=${K}`);
  }

  aBuf.destroy(); bBuf.destroy(); cBuf.destroy(); paramsBuf.destroy();
  return { name: 'Matmul BT BF16 (K=2048)', passed: close, elapsed_ms: elapsed, maxDiff };
}

// ─── TurboQuant Round-trip Test ──────────────────────────────────────────────

async function testTurboQuant3bit(device: GPUDevice): Promise<TestResult> {
  const d = 64;       // head dimension
  const bits = 3;
  const numVectors = 8;

  // Generate random input vectors (simulating KV cache rows)
  const input = new Float32Array(numVectors * d);
  for (let i = 0; i < input.length; i++) {
    input[i] = (Math.random() - 0.5) * 2;
  }

  // CPU reference: encode → decode → measure distortion
  const rotMatrix = generateRotationMatrix(d, 42);
  const jlMatrix = generateJLMatrix(d, 137);
  const codebook = buildCodebook(bits);

  let cpuTotalRelMSE = 0;
  for (let v = 0; v < numVectors; v++) {
    const vec = input.slice(v * d, (v + 1) * d);
    const encoded = cpuEncode(vec, rotMatrix, jlMatrix, codebook, d);
    const decoded = cpuDecode(encoded.quantized, encoded.signBits, encoded.norm, rotMatrix, jlMatrix, codebook, d);
    cpuTotalRelMSE += computeRelativeMSE(vec, decoded);
  }
  const cpuAvgRelMSE = cpuTotalRelMSE / numVectors;

  // GPU: encode → decode → read back
  const tq = createTurboQuantPipeline(device, { headDim: d, bits });

  const inputBuf = createStorageBuffer(device, input, input.byteLength, 'tq-test-input');
  const compressed = await tq.encodeAndWait(inputBuf, numVectors);
  const gpuDecoded = await tq.decodeAndRead(compressed);

  // Measure GPU distortion
  let gpuTotalRelMSE = 0;
  for (let v = 0; v < numVectors; v++) {
    const orig = input.slice(v * d, (v + 1) * d);
    const recon = gpuDecoded.slice(v * d, (v + 1) * d);
    gpuTotalRelMSE += computeRelativeMSE(orig, recon);
  }
  const gpuAvgRelMSE = gpuTotalRelMSE / numVectors;

  // Cleanup
  inputBuf.destroy();
  compressed.quantizedBuffer.destroy();
  compressed.signBitsBuffer.destroy();
  compressed.normsBuffer.destroy();
  compressed.residualNormsBuffer.destroy();

  // Pass criteria:
  // 1. GPU distortion should be reasonable (< 0.15 relative MSE for 3-bit)
  // 2. GPU and CPU should produce similar distortion (within 50% of each other)
  const distortionOk = gpuAvgRelMSE < 0.15;
  const cpuGpuMatch = Math.abs(gpuAvgRelMSE - cpuAvgRelMSE) / Math.max(cpuAvgRelMSE, 1e-6) < 0.5;
  const passed = distortionOk && cpuGpuMatch;

  return {
    name: `TurboQuant (3-bit, d=${d})`,
    passed,
    maxDiff: gpuAvgRelMSE,
    error: passed ? undefined :
      `GPU relMSE=${gpuAvgRelMSE.toFixed(4)}, CPU relMSE=${cpuAvgRelMSE.toFixed(4)}, ` +
      `ratio=${compressed.ratio.toFixed(1)}x`,
  };
}

async function testTurboQuant4bit(device: GPUDevice): Promise<TestResult> {
  const d = 128;      // larger head dimension
  const bits = 4;
  const numVectors = 4;

  const input = new Float32Array(numVectors * d);
  for (let i = 0; i < input.length; i++) {
    input[i] = (Math.random() - 0.5) * 2;
  }

  const tq = createTurboQuantPipeline(device, { headDim: d, bits });

  const inputBuf = createStorageBuffer(device, input, input.byteLength, 'tq-test-input-4bit');
  const compressed = await tq.encodeAndWait(inputBuf, numVectors);
  const gpuDecoded = await tq.decodeAndRead(compressed);

  let gpuTotalRelMSE = 0;
  for (let v = 0; v < numVectors; v++) {
    const orig = input.slice(v * d, (v + 1) * d);
    const recon = gpuDecoded.slice(v * d, (v + 1) * d);
    gpuTotalRelMSE += computeRelativeMSE(orig, recon);
  }
  const gpuAvgRelMSE = gpuTotalRelMSE / numVectors;

  inputBuf.destroy();
  compressed.quantizedBuffer.destroy();
  compressed.signBitsBuffer.destroy();
  compressed.normsBuffer.destroy();
  compressed.residualNormsBuffer.destroy();

  // 4-bit should have lower distortion than 3-bit (< 0.05 relative MSE)
  const passed = gpuAvgRelMSE < 0.05;

  return {
    name: `TurboQuant (4-bit, d=${d})`,
    passed,
    maxDiff: gpuAvgRelMSE,
    error: passed ? undefined :
      `GPU relMSE=${gpuAvgRelMSE.toFixed(4)}, ratio=${compressed.ratio.toFixed(1)}x`,
  };
}

// ─── Lloyd-Max Codebook MSE Validation ────────────────────────────────────────
// Validates our codebook against tonbistudio's numbers from real Qwen2.5-3B KV tensors:
//   3-bit MSE: 0.034 (paper bound: 0.043)
//   4-bit MSE: 0.009 (paper bound: 0.011)
// We test with N(0,1) vectors (ideal case) so our numbers should be at or below theirs.

async function testLloydMaxMSE(device: GPUDevice): Promise<TestResult> {
  const numVectors = 1000;
  const d = 128;
  const errors: string[] = [];

  for (const bits of [3, 4] as const) {
    const codebook = buildCodebook(bits);
    const scale = 1.0 / Math.sqrt(d);

    // Generate N(0,1) vectors and measure quantization MSE
    // After rotation, each coord is ~N(0, 1/d). The codebook targets N(0,1),
    // so we test on N(0,1) directly (pre-scaling) for codebook validation.
    let totalMSE = 0;
    for (let v = 0; v < numVectors; v++) {
      let vecMSE = 0;
      for (let i = 0; i < d; i++) {
        // Box-Muller for N(0,1)
        const u1 = Math.random();
        const u2 = Math.random();
        const val = Math.sqrt(-2 * Math.log(Math.max(u1, 1e-10))) * Math.cos(2 * Math.PI * u2);

        // Quantize: find closest centroid
        const absVal = Math.abs(val);
        let bin = 0;
        for (let t = 0; t < codebook.thresholds.length; t++) {
          if (absVal > codebook.thresholds[t]) bin = t + 1;
        }
        const recon = (val >= 0 ? 1 : -1) * codebook.centroids[bin];
        const diff = val - recon;
        vecMSE += diff * diff;
      }
      totalMSE += vecMSE / d;
    }
    const avgMSE = totalMSE / numVectors;

    // tonbistudio reference: 3-bit=0.034, 4-bit=0.009. Paper bounds: 0.043, 0.011.
    // Our ideal-case MSE should be at or below the paper bounds.
    const paperBound = bits === 3 ? 0.043 : 0.011;
    if (avgMSE > paperBound * 1.1) { // 10% tolerance
      errors.push(`${bits}-bit MSE=${avgMSE.toFixed(4)} exceeds paper bound ${paperBound}`);
    }
  }

  return {
    name: 'Lloyd-Max Codebook MSE',
    passed: errors.length === 0,
    error: errors.length > 0 ? errors.join('; ') : undefined,
  };
}

// ─── Asymmetric Inner Product Test ────────────────────────────────────────────
// Validates that cpuAsymmetricScore gives a more accurate <q,k> estimate
// than the naive <q, k̂_PQ> (decode-then-dot).

async function testAsymmetricScore(device: GPUDevice): Promise<TestResult> {
  const d = 128;
  const bits = 3;
  const numPairs = 50;

  const rotMatrix = generateRotationMatrix(d, 42);
  const jlMatrix = generateJLMatrix(d, 137);
  const spiMatrix = generateSPiMatrix(jlMatrix, rotMatrix, d);
  const codebook = buildCodebook(bits);

  let naiveTotalError = 0;
  let asymTotalError = 0;
  let trueTotalMag = 0;

  for (let p = 0; p < numPairs; p++) {
    // Random query and key vectors
    const query = new Float32Array(d);
    const key = new Float32Array(d);
    for (let i = 0; i < d; i++) {
      const u1 = Math.random(), u2 = Math.random();
      query[i] = Math.sqrt(-2 * Math.log(Math.max(u1, 1e-10))) * Math.cos(2 * Math.PI * u2);
      const u3 = Math.random(), u4 = Math.random();
      key[i] = Math.sqrt(-2 * Math.log(Math.max(u3, 1e-10))) * Math.cos(2 * Math.PI * u4);
    }

    // True inner product
    let trueDot = 0;
    for (let i = 0; i < d; i++) trueDot += query[i] * key[i];

    // Encode key
    const encoded = cpuEncode(key, rotMatrix, jlMatrix, codebook, d);

    // Decode (PolarQuant only, no QJL — same as our GPU decode)
    const decoded = cpuDecode(encoded.quantized, encoded.signBits, encoded.norm,
      rotMatrix, jlMatrix, codebook, d);

    // Naive: <q, k̂_PQ>
    let naiveDot = 0;
    for (let i = 0; i < d; i++) naiveDot += query[i] * decoded[i];

    // Asymmetric: <q, k̂_PQ> + QJL correction
    const asymDot = cpuAsymmetricScore(
      query, decoded, encoded.signBits,
      encoded.norm, encoded.residualNorm, spiMatrix, d,
    );

    naiveTotalError += Math.abs(naiveDot - trueDot);
    asymTotalError += Math.abs(asymDot - trueDot);
    trueTotalMag += Math.abs(trueDot);
  }

  const naiveRelError = naiveTotalError / trueTotalMag;
  const asymRelError = asymTotalError / trueTotalMag;
  const improvement = (naiveRelError - asymRelError) / naiveRelError;

  // Asymmetric should have lower error than naive in most cases
  const passed = asymRelError < naiveRelError;

  return {
    name: `Asymmetric Score (3-bit, d=${d})`,
    passed,
    maxDiff: asymRelError,
    error: passed ? undefined :
      `Asymmetric relErr=${asymRelError.toFixed(4)} >= naive relErr=${naiveRelError.toFixed(4)} ` +
      `(improvement=${(improvement * 100).toFixed(1)}%)`,
  };
}

// ─── Run All Tests ───────────────────────────────────────────────────────────

export async function runKernelTests(device: GPUDevice): Promise<TestResult[]> {
  const tests = [
    { name: 'SiLU', fn: testSiLU },
    { name: 'Add', fn: testAdd },
    { name: 'Multiply', fn: testMul },
    { name: 'Matmul (naive)', fn: testMatmulNaive },
    { name: 'Matmul (tiled)', fn: testMatmulTiled },
    { name: 'Matmul BT (f32)', fn: testMatmulBT },
    { name: 'Matmul BT BF16 (small)', fn: testMatmulBTBF16Small },
    { name: 'Matmul BT BF16 (large)', fn: testMatmulBTBF16Large },
    { name: 'Matmul BT BF16 (K=2048)', fn: testMatmulBTBF16RealDims },
    { name: 'Softmax', fn: testSoftmax },
    { name: 'RMSNorm', fn: testRMSNorm },
    { name: 'TurboQuant (3-bit)', fn: testTurboQuant3bit },
    { name: 'TurboQuant (4-bit)', fn: testTurboQuant4bit },
    { name: 'Lloyd-Max Codebook MSE', fn: testLloydMaxMSE },
    { name: 'Asymmetric Score', fn: testAsymmetricScore },
  ];

  const results: TestResult[] = [];

  for (const test of tests) {
    try {
      const result = await test.fn(device);
      results.push(result);
    } catch (err) {
      results.push({
        name: test.name,
        passed: false,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }

  return results;
}
