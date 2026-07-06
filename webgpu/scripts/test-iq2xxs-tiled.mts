/**
 * Real-GPU correctness for matmul_gguf_iq2_xxs_tiled (lever C5 rewrite:
 * aligned u32 loads + unpack4xU8 grid decode + XOR sign flip) against
 * (a) the CPU dequant reference and (b) the legacy matmul_gguf_iq2_xxs
 * kernel on the same GPU. Catches shader-validation errors (incl. missing
 * unpack4xU8 support), grid-array placement, and runtime OOB that tsc +
 * CPU parity cannot (per the wgsl-needs-gpu-bench rule).
 *
 * Shapes exercise the edge guards: N=12 is not divisible by TN=8 (valid
 * guard), K=768 gives nUnits=24 vs TPR=16 (partial last chunk).
 *
 * Self-contained: Playwright loopback page, no vite server.
 * Run: npx tsx scripts/test-iq2xxs-tiled.mts
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

const TN = 8, TWG = 128;
const SHAPES = [
  { name: 'N16 K512 M1', N: 16, K: 512, M: 1 },
  { name: 'N12 K768 M1 (edge: N%TN!=0, partial chunk)', N: 12, K: 768, M: 1 },
  // M>1 exercises the M-reuse GEMM entry (spec-decode verify path).
  { name: 'N16 K512 M2', N: 16, K: 512, M: 2 },
  { name: 'N12 K768 M5 (edge: N%TN!=0, partial chunk)', N: 12, K: 768, M: 5 },
  { name: 'N16 K768 M8 (MAXM cap)', N: 16, K: 768, M: 8 },
];

const RAW_SB = 66; // bytes per raw IQ2_XXS superblock

const problems = SHAPES.map((s, si) => {
  const rng = mulberry32(0x122C5 ^ si);
  const nSB = s.K / 256;
  const raw = new Uint8Array(s.N * nSB * RAW_SB);
  for (let i = 0; i < raw.length; i++) raw[i] = Math.floor(rng() * 256);
  // Sane positive f16 scale d @ byte 0 of every superblock.
  const dv = new DataView(raw.buffer);
  for (let b = 0; b < s.N * nSB; b++) dv.setUint16(b * RAW_SB, 0x2C00 /* 0.0625 */, true);

  const ref = dequantGGML(GGML_TYPES.IQ2_XXS, raw, s.N * s.K);
  const W = repackGGUFForGPU(GGML_TYPES.IQ2_XXS, raw, s.N * s.K) as Uint32Array;
  const A = new Float32Array(s.M * s.K);
  for (let i = 0; i < A.length; i++) A[i] = rng() * 2 - 1;
  const expected = new Float32Array(s.M * s.N);
  for (let m = 0; m < s.M; m++) {
    for (let n = 0; n < s.N; n++) {
      let acc = 0;
      for (let i = 0; i < s.K; i++) acc += A[m * s.K + i] * ref[n * s.K + i];
      expected[m * s.N + n] = acc;
    }
  }
  return { name: s.name, N: s.N, K: s.K, M: s.M, W: Array.from(W), A: Array.from(A), expected: Array.from(expected) };
});

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
let failures = 0;
try {
  const page = await browser.newPage();
  page.on('console', (m) => { const t = m.text(); if (/error|warn|fail/i.test(t)) console.log(`[page] ${t}`); });
  page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
  await page.route('**/*', (route) =>
    route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><meta charset=utf8><title>iq2xxs-tiled</title>' }));
  await page.goto('http://127.0.0.1/');

  const results = await page.evaluate(async ({ shaderSrc, problems, TN, TWG }) => {
    const g = (navigator as any).gpu;
    if (!g) return { error: 'no navigator.gpu' };
    const adapter = await g.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { error: 'no adapter' };
    // Match the engine (gpu-device.ts): the gemm entry needs a_tile_m (16 KB)
    // + row_acc, above the 16 KB default workgroup-storage limit.
    const device = await adapter.requestDevice({
      requiredLimits: { maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize },
    });

    const module = device.createShaderModule({ code: shaderSrc });
    const info = await module.getCompilationInfo();
    const compileMsgs = info.messages
      .filter((mm: any) => mm.type === 'error')
      .map((mm: any) => `${mm.lineNum}:${mm.linePos} ${mm.message}`);
    if (compileMsgs.length) return { error: `shader errors:\n${compileMsgs.join('\n')}` };

    const out: Array<{ name: string; legacy: number[]; tiled: number[]; gemm: number[]; errs: string[] }> = [];
    for (const p of problems) {
      const errs: string[] = [];
      const variants = [
        // legacy + tiled batch M via grid.y; gemm handles all M internally (grid.y=1).
        { key: 'legacy', entry: 'matmul_gguf_iq2_xxs', constants: undefined as Record<string, number> | undefined, gridX: p.N, gridY: p.M },
        { key: 'tiled', entry: 'matmul_gguf_iq2_xxs_tiled', constants: { TN, TWG }, gridX: Math.ceil(p.N / TN), gridY: p.M },
        { key: 'gemm', entry: 'matmul_gguf_iq2_xxs_gemm', constants: { TN, TWG }, gridX: Math.ceil(p.N / TN), gridY: 1 },
      ];
      const got: Record<string, number[]> = {};
      for (const v of variants) {
        device.pushErrorScope('validation');
        const Wu = new Uint32Array(p.W);
        const Af = new Float32Array(p.A);
        const aBuf = device.createBuffer({ size: Af.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(aBuf, 0, Af);
        const wBuf = device.createBuffer({ size: Wu.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(wBuf, 0, Wu);
        const cBuf = device.createBuffer({ size: p.M * p.N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const pBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(pBuf, 0, new Uint32Array([p.M, p.N, p.K, 0]));
        const pipeline = device.createComputePipeline({
          layout: 'auto',
          compute: { module, entryPoint: v.entry, constants: v.constants },
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
        pass.dispatchWorkgroups(v.gridX, v.gridY, 1);
        pass.end();
        const readBuf = device.createBuffer({ size: p.M * p.N * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        enc.copyBufferToBuffer(cBuf, 0, readBuf, 0, p.M * p.N * 4);
        device.queue.submit([enc.finish()]);
        await readBuf.mapAsync(GPUMapMode.READ);
        got[v.key] = Array.from(new Float32Array(readBuf.getMappedRange().slice(0)));
        readBuf.unmap();
        const err = await device.popErrorScope();
        if (err) errs.push(`${v.key}: ${err.message}`);
      }
      out.push({ name: p.name, legacy: got.legacy, tiled: got.tiled, gemm: got.gemm, errs });
    }
    return { out };
  }, { shaderSrc, problems, TN, TWG });

  if ((results as any).error) {
    console.log(`FAIL: ${(results as any).error}`);
    process.exit(1);
  }
  for (const r of (results as any).out) {
    const prob = problems.find((p) => p.name === r.name)!;
    if (r.errs.length) {
      failures++;
      console.log(`  FAIL ${r.name}:\n    ${r.errs.join('\n    ')}`);
      continue;
    }
    const check = (label: string, got: number[], want: number[], tol: number) => {
      let maxRel = 0, maxAbs = 0;
      for (let n = 0; n < prob.M * prob.N; n++) {
        const abs = Math.abs(got[n] - want[n]);
        maxAbs = Math.max(maxAbs, abs);
        maxRel = Math.max(maxRel, abs / Math.max(1e-4, Math.abs(want[n])));
      }
      const ok = maxRel < tol;
      if (!ok) failures++;
      console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${r.name} ${label}: maxRel=${maxRel.toExponential(2)} maxAbs=${maxAbs.toExponential(2)}`);
      if (!ok) {
        console.log(`    got=[${got.map((x) => x.toFixed(4)).join(', ')}]`);
        console.log(`    want=[${want.map((x) => x.toFixed(4)).join(', ')}]`);
      }
    };
    check('legacy vs CPU', r.legacy, prob.expected as unknown as number[], 2e-3);
    check('tiled  vs CPU', r.tiled, prob.expected as unknown as number[], 2e-3);
    check('gemm   vs CPU', r.gemm, prob.expected as unknown as number[], 2e-3);
    check('tiled  vs legacy(GPU)', r.tiled, r.legacy, 1e-3);
    check('gemm   vs legacy(GPU)', r.gemm, r.legacy, 1e-3);
  }
} finally {
  await browser.close();
}
if (failures > 0) { console.log(`\n${failures} FAILURE(S)`); process.exit(1); }
console.log('\nALL IQ2_XXS TILED CHECKS PASSED');
