/**
 * Decode-shape hang repro for the legacy IQ3_XXS / IQ3_S / IQ2_S GEMV kernels.
 *
 * The UD-IQ2_XXS 27B loads + completes 3 prefill forwards (M=16/8) then trips
 * DXGI_ERROR_DEVICE_HUNG on the FIRST decode forward (M=1). A control run of
 * the pure-IQ2_XXS MTP model through the identical headless harness decodes
 * fine — so the hang is specific to the UD file's extra IQ3_XXS/IQ3_S/IQ2_S
 * tensors. Those tensors use the legacy one-row-per-workgroup GEMV for BOTH
 * prefill and decode; decode just shrinks grid.y from M→1.
 *
 * This isolates kernel-vs-dispatch: synthesize each tensor at its REAL decode
 * dimensions, dispatch the legacy kernel at M=16 (prefill shape) then M=1
 * (decode shape), and watch device.lost. If M=1 hangs but M=16 doesn't, the
 * fault is in the kernel at the decode shape; if neither hangs, the WGSL is
 * innocent and the bug is in forward-pass's decode dispatch/binding.
 *
 * No 9 GB model load. Run: npx tsx scripts/test-iq-decode-hang.mts
 */
import { chromium } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { repackGGUFForGPU } from '../src/model/gguf-dequant.ts';
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

// Real UD-IQ2_XXS decode shapes (GGUF shape = [K, N]).
interface Case { name: string; entry: string; ggmlType: number; bytesPerSB: number; N: number; K: number; }
const CASES: Case[] = [
  { name: 'IQ3_XXS attn_v',    entry: 'matmul_gguf_iq3_xxs', ggmlType: GGML_TYPES.IQ3_XXS, bytesPerSB: 98,  N: 1024, K: 5120 },
  { name: 'IQ3_XXS attn_gate', entry: 'matmul_gguf_iq3_xxs', ggmlType: GGML_TYPES.IQ3_XXS, bytesPerSB: 98,  N: 6144, K: 5120 },
  { name: 'IQ3_S ffn_down',    entry: 'matmul_gguf_iq3_s',   ggmlType: GGML_TYPES.IQ3_S,   bytesPerSB: 110, N: 5120, K: 17408 },
  { name: 'IQ2_S ffn_down',    entry: 'matmul_gguf_iq2_s',   ggmlType: GGML_TYPES.IQ2_S,   bytesPerSB: 82,  N: 5120, K: 17408 },
];

// Build repacked weight bytes for one case (CPU). Force a sane f16 scale at the
// front of every superblock so values aren't NaN/Inf garbage. Keep the bytes
// as a Buffer (served via route, fetched in the browser) — serializing the
// ~20M u32 across cases through page.evaluate blows the node heap.
const problems = CASES.map((c) => {
  const SB = c.K / 256;
  const rng = mulberry32(0xD3C0DE ^ (c.ggmlType * 131 + c.N));
  const raw = new Uint8Array(c.N * SB * c.bytesPerSB);
  for (let i = 0; i < raw.length; i++) raw[i] = Math.floor(rng() * 256);
  const dv = new DataView(raw.buffer);
  for (let b = 0; b < c.N * SB; b++) dv.setUint16(b * c.bytesPerSB, 0x2C00, true); // f16 0.0625
  const W = repackGGUFForGPU(c.ggmlType, raw, c.N * c.K) as Uint32Array;
  return { name: c.name, entry: c.entry, N: c.N, K: c.K, buf: Buffer.from(W.buffer, W.byteOffset, W.byteLength) };
});
const meta = problems.map((p, i) => ({ name: p.name, entry: p.entry, N: p.N, K: p.K, idx: i }));

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
try {
  const page = await browser.newPage();
  page.on('console', (m) => { const t = m.text(); if (/error|warn|fail|lost|hung/i.test(t)) console.log(`[page] ${t}`); });
  page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
  await page.route('**/*', (route) => {
    const m = /\/w\/(\d+)\.bin$/.exec(route.request().url());
    if (m) return route.fulfill({ status: 200, contentType: 'application/octet-stream', body: problems[Number(m[1])].buf });
    return route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><meta charset=utf8><title>iq-hang</title>' });
  });
  // tsx/esbuild keepNames wraps inner functions in __name(); inject it as a
  // browser global (string content is not transpiled) so the serialized
  // evaluate body resolves it.
  await page.addInitScript({ content: 'globalThis.__name = globalThis.__name || function(f){return f;};' });
  await page.goto('http://127.0.0.1/');

  const results = await page.evaluate(async ({ shaderSrc, problems }) => {
    const g = (navigator as any).gpu;
    if (!g) return { error: 'no navigator.gpu' };
    // Match the real harness: high-performance picks the discrete AMD card.
    const adapter = await g.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { error: 'no adapter' };
    const info = adapter.info ?? (await adapter.requestAdapterInfo?.());
    const device = await adapter.requestDevice();
    let lost: any = null;
    device.lost.then((i: any) => { lost = i; });
    device.addEventListener?.('uncapturederror', (e: any) => console.log('uncaptured: ' + e.error?.message));

    const module = device.createShaderModule({ code: shaderSrc });
    const cinfo = await module.getCompilationInfo();
    const cmsgs = cinfo.messages.filter((mm: any) => mm.type === 'error')
      .map((mm: any) => `${mm.lineNum}:${mm.linePos} ${mm.message}`);

    const log: Array<{ name: string; M: number; ms: number; ok: boolean; note: string }> = [];

    const onDone = () =>
      Promise.race([
        device.queue.onSubmittedWorkDone(),
        new Promise((res) => setTimeout(() => res('timeout'), 8000)),
      ]);

    for (const p of problems) {
      const wbytes = await (await fetch(`/w/${p.idx}.bin`)).arrayBuffer();
      const Wu = new Uint32Array(wbytes);
      const wBuf = device.createBuffer({ size: Wu.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(wBuf, 0, Wu);
      const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: p.entry } });

      // M=16 (prefill shape) THEN M=1 (decode shape) — same kernel, fewer rows.
      for (const M of [16, 1]) {
        if (lost) { log.push({ name: p.name, M, ms: 0, ok: false, note: 'device already lost' }); continue; }
        const Af = new Float32Array(M * p.K);
        for (let i = 0; i < Af.length; i++) Af[i] = Math.fround((i % 97) / 97 - 0.5);
        const aBuf = device.createBuffer({ size: Af.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(aBuf, 0, Af);
        const cBuf = device.createBuffer({ size: M * p.N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const pBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(pBuf, 0, new Uint32Array([M, p.N, p.K, 0]));
        const bind = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: aBuf } },
            { binding: 1, resource: { buffer: wBuf } },
            { binding: 2, resource: { buffer: cBuf } },
            { binding: 3, resource: { buffer: pBuf } },
          ],
        });
        device.pushErrorScope('validation');
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        // EXACT forward-pass grid for the legacy (else) branch.
        pass.dispatchWorkgroups(Math.min(p.N, 65535), M, Math.ceil(p.N / 65535));
        pass.end();
        const t0 = performance.now();
        device.queue.submit([enc.finish()]);
        const r = await onDone();
        const verr = await device.popErrorScope();
        const ms = performance.now() - t0;
        const note = lost ? `DEVICE LOST: ${lost.reason} ${lost.message}`
          : r === 'timeout' ? 'onSubmittedWorkDone TIMEOUT (8s) — hang'
          : verr ? `validation: ${verr.message}` : 'ok';
        log.push({ name: p.name, M, ms: Math.round(ms), ok: !lost && r !== 'timeout' && !verr, note });
        aBuf.destroy(); cBuf.destroy(); pBuf.destroy();
        if (lost) break;
      }
      wBuf.destroy();
      if (lost) break;
    }
    return { adapter: `${info?.vendor ?? '?'} ${info?.architecture ?? ''} ${info?.description ?? ''}`.trim(), cmsgs, log, lost: lost ? `${lost.reason}: ${lost.message}` : null };
  }, { shaderSrc, problems: meta });

  if ((results as any).error) { console.log(`GPU unavailable: ${(results as any).error}`); process.exit(2); }
  const r = results as any;
  console.log(`adapter: ${r.adapter}`);
  if (r.cmsgs.length) { console.log(`SHADER COMPILE ERRORS:\n  ${r.cmsgs.join('\n  ')}`); process.exit(1); }
  for (const e of r.log) {
    console.log(`  ${e.ok ? 'ok  ' : 'FAIL'} ${e.name.padEnd(18)} M=${String(e.M).padStart(2)}  ${String(e.ms).padStart(5)}ms  ${e.note}`);
  }
  if (r.lost) { console.log(`\nDEVICE LOST: ${r.lost}`); process.exit(1); }
  if (r.log.some((e: any) => !e.ok)) { console.log('\nFAILURE(S) — see notes above'); process.exit(1); }
  console.log('\nALL DECODE-SHAPE DISPATCHES COMPLETED (no hang)');
} finally {
  await browser.close();
}
