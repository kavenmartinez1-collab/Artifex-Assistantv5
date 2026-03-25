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
  generateRotationMatrix, generateJLMatrix, buildCodebook,
  cpuEncode, cpuDecode, computeRelativeMSE,
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

// ─── Run All Tests ───────────────────────────────────────────────────────────

export async function runKernelTests(device: GPUDevice): Promise<TestResult[]> {
  const tests = [
    { name: 'SiLU', fn: testSiLU },
    { name: 'Add', fn: testAdd },
    { name: 'Multiply', fn: testMul },
    { name: 'Matmul (naive)', fn: testMatmulNaive },
    { name: 'Matmul (tiled)', fn: testMatmulTiled },
    { name: 'Softmax', fn: testSoftmax },
    { name: 'RMSNorm', fn: testRMSNorm },
    { name: 'TurboQuant (3-bit)', fn: testTurboQuant3bit },
    { name: 'TurboQuant (4-bit)', fn: testTurboQuant4bit },
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
