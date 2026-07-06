/**
 * Real-GPU correctness for the IQ3_XXS / IQ3_S / IQ2_S legacy GEMV kernels
 * (matmul_gguf_iq3_xxs / _iq3_s / _iq2_s). The CPU-port audit (test-iq-parity)
 * proves the dequant math; this proves the actual WGSL shader COMPILES and the
 * kernel produces the right GEMV — catching shader-validation errors, grid
 * array placement, and runtime OOB that tsc + CPU parity cannot (per the
 * wgsl-needs-gpu-bench rule).
 *
 * Self-contained: serves a blank loopback page (secure context for WebGPU) via
 * Playwright route fulfillment, so no vite dev server is required. Synthesizes
 * weight blocks in node, repacks them with the engine's repackGGUFForGPU, dots
 * against a known activation on the GPU, and compares to dequantGGML on CPU.
 *
 * Run: npx tsx scripts/test-iq-gpu.mts
 */
import { chromium } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { dequantGGML, repackGGUFForGPU } from '../src/model/gguf-dequant.ts';
import { GGML_TYPES } from '../src/model/gguf.ts';

const here = dirname(fileURLToPath(import.meta.url));
const shaderSrc = readFileSync(resolve(here, '../src/shaders/matmul_gguf.wgsl'), 'utf8');

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Case {
  name: string; entry: string; ggmlType: number; rawBytesPerSB: number;
}
const CASES: Case[] = [
  { name: 'IQ3_XXS', entry: 'matmul_gguf_iq3_xxs', ggmlType: GGML_TYPES.IQ3_XXS, rawBytesPerSB: 98 },
  { name: 'IQ3_S', entry: 'matmul_gguf_iq3_s', ggmlType: GGML_TYPES.IQ3_S, rawBytesPerSB: 110 },
  { name: 'IQ2_S', entry: 'matmul_gguf_iq2_s', ggmlType: GGML_TYPES.IQ2_S, rawBytesPerSB: 82 },
];

const N = 8;        // output columns (weight rows)
const SB = 2;       // superblocks per row → K = 512
const K = SB * 256;

// Build synthetic problems on CPU: repacked W bytes, activation A, reference C.
const problems = CASES.map((c) => {
  const rng = mulberry32(0xA11CE ^ c.ggmlType);
  const raw = new Uint8Array(N * SB * c.rawBytesPerSB);
  for (let i = 0; i < raw.length; i++) raw[i] = Math.floor(rng() * 256);
  // Force a sane positive f16 scale (d @ byte 0 of every superblock) so the
  // result isn't dominated by NaN/Inf from a random scale word.
  const dv = new DataView(raw.buffer);
  const dBits = 0x2C00; // f16 0.0625
  for (let b = 0; b < N * SB; b++) dv.setUint16(b * c.rawBytesPerSB, dBits, true);

  const ref = dequantGGML(c.ggmlType, raw, N * K);           // [N, K] row-major
  const W = repackGGUFForGPU(c.ggmlType, raw, N * K) as Uint32Array;
  const A = new Float32Array(K);
  for (let i = 0; i < K; i++) A[i] = rng() * 2 - 1;
  const expected = new Float32Array(N);
  for (let n = 0; n < N; n++) {
    let s = 0;
    for (let i = 0; i < K; i++) s += A[i] * ref[n * K + i];
    expected[n] = s;
  }
  return { name: c.name, entry: c.entry, W: Array.from(W), A: Array.from(A), expected: Array.from(expected) };
});

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
let failures = 0;
try {
  const page = await browser.newPage();
  page.on('console', (m) => { const t = m.text(); if (/error|warn|fail/i.test(t)) console.log(`[page] ${t}`); });
  page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
  // Loopback is a secure context → navigator.gpu is available. Fulfill any
  // request with a blank page so no real server is needed.
  await page.route('**/*', (route) =>
    route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><meta charset=utf8><title>iq-gpu</title>' }));
  await page.goto('http://127.0.0.1/');

  const results = await page.evaluate(async ({ shaderSrc, problems, N, K }) => {
    const g = (navigator as any).gpu;
    if (!g) return { error: 'no navigator.gpu' };
    const adapter = await g.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { error: 'no adapter' };
    const device = await adapter.requestDevice();
    const out: Array<{ name: string; got: number[]; compileMsgs: string[] }> = [];

    for (const p of problems) {
      device.pushErrorScope('validation');
      const module = device.createShaderModule({ code: shaderSrc });
      const info = await module.getCompilationInfo();
      const compileMsgs = info.messages
        .filter((mm: any) => mm.type === 'error')
        .map((mm: any) => `${mm.lineNum}:${mm.linePos} ${mm.message}`);

      const Wu = new Uint32Array(p.W);
      const Af = new Float32Array(p.A);
      const aBuf = device.createBuffer({ size: Af.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(aBuf, 0, Af);
      const wBuf = device.createBuffer({ size: Wu.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(wBuf, 0, Wu);
      const cBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const pBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(pBuf, 0, new Uint32Array([1, N, K, 0]));

      const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: p.entry },
      });
      const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: aBuf } },
          { binding: 1, resource: { buffer: wBuf } },
          { binding: 2, resource: { buffer: cBuf } },
          { binding: 3, resource: { buffer: pBuf } },
        ],
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(N, 1, 1);
      pass.end();
      const readBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      enc.copyBufferToBuffer(cBuf, 0, readBuf, 0, N * 4);
      device.queue.submit([enc.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ);
      const got = Array.from(new Float32Array(readBuf.getMappedRange().slice(0)));
      readBuf.unmap();
      const err = await device.popErrorScope();
      if (err) compileMsgs.push(`validation: ${err.message}`);
      out.push({ name: p.name, got, compileMsgs });
    }
    return { out };
  }, { shaderSrc, problems, N, K });

  if ((results as any).error) {
    console.log(`GPU unavailable: ${(results as any).error}`);
    process.exit(2);
  }
  for (const r of (results as any).out) {
    const prob = problems.find((p) => p.name === r.name)!;
    if (r.compileMsgs.length) {
      failures++;
      console.log(`  FAIL ${r.name}: shader errors:\n    ${r.compileMsgs.join('\n    ')}`);
      continue;
    }
    let maxRel = 0, maxAbs = 0;
    for (let n = 0; n < N; n++) {
      const want = prob.expected[n], got = r.got[n];
      const abs = Math.abs(got - want);
      maxAbs = Math.max(maxAbs, abs);
      maxRel = Math.max(maxRel, abs / Math.max(1e-4, Math.abs(want)));
    }
    const ok = maxRel < 2e-3;   // f32 GPU vs f64-ish CPU dot, Kahan reduce
    if (!ok) failures++;
    console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${r.name}: maxRel=${maxRel.toExponential(2)} maxAbs=${maxAbs.toExponential(2)} over ${N} cols`);
    if (!ok) console.log(`    got=[${r.got.map((x: number) => x.toFixed(4)).join(', ')}]\n    want=[${prob.expected.map((x) => x.toFixed(4)).join(', ')}]`);
  }
} finally {
  await browser.close();
}
if (failures > 0) { console.log(`\n${failures} FAILURE(S)`); process.exit(1); }
console.log('\nALL GPU KERNEL CHECKS PASSED');
