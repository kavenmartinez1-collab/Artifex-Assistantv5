/**
 * Phase 0b gate: matmul_bt_bf16 throughput at real FLUX.2-klein DiT shapes
 * on the RX 6700 XT (HEADED Chrome by default, per headed-bench-gate rule).
 *
 * Gates (from the approved plan):
 *   - worst single dispatch > ~800 ms  => N-slicing is MANDATORY (TDR ~2 s)
 *   - effective TFLOPS < ~0.5          => schedule GEMM kernel upgrade before Phase 3
 *
 * M values: 1536 = 512 txt + 1024 img (512px), 4608 = 512 txt + 4096 img (1024px).
 * All data is generated inside the page (no giant arrays over the CDP wire).
 * Correctness is spot-checked (64 random outputs vs CPU f64 dot on the exact
 * same bf16-dequantized weights).
 *
 * Run: npx tsx scripts/bench-flux2-gemm.mts        (headed)
 *      HEADLESS=1 npx tsx scripts/bench-flux2-gemm.mts  (logic smoke only)
 */
import { chromium } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const shaderSrc = readFileSync(resolve(here, '../src/shaders/matmul.wgsl'), 'utf8');

// Real DiT shapes (K = in features, N = out features; weights [N, K] bf16).
// Ordered by FLOPs ascending so we collect data before any TDR on the worst one.
const SHAPES = [
  // context_embedder runs once at M=512 (txt tokens)
  { name: 'context_embedder 512x7680->3072', M: 512, K: 7680, N: 3072 },
  // 512px (M=1536 joint)
  { name: 'attn/out 1536x3072->3072', M: 1536, K: 3072, N: 3072 },
  { name: 'ffn_out 1536x9216->3072', M: 1536, K: 9216, N: 3072 },
  { name: 'single_out 1536x12288->3072', M: 1536, K: 12288, N: 3072 },
  { name: 'ffn_in 1536x3072->18432', M: 1536, K: 3072, N: 18432 },
  { name: 'single_qkv_mlp 1536x3072->27648', M: 1536, K: 3072, N: 27648 },
  // 1024px (M=4608 joint)
  { name: 'attn/out 4608x3072->3072', M: 4608, K: 3072, N: 3072 },
  { name: 'ffn_out 4608x9216->3072', M: 4608, K: 9216, N: 3072 },
  { name: 'single_out 4608x12288->3072', M: 4608, K: 12288, N: 3072 },
  { name: 'ffn_in 4608x3072->18432', M: 4608, K: 3072, N: 18432 },
  // N-slice proxy: quarter of the worst shape (what Phase 3 would dispatch per slice)
  { name: 'single_qkv_mlp SLICE 4608x3072->6912', M: 4608, K: 3072, N: 6912 },
  { name: 'single_qkv_mlp 4608x3072->27648 (WORST)', M: 4608, K: 3072, N: 27648 },
];

const HEADLESS = process.env.HEADLESS === '1';
const browser = await chromium.launch({
  channel: 'chrome',
  headless: HEADLESS,
  args: ['--enable-unsafe-webgpu', '--window-size=360,240', '--window-position=20,20'],
});

let failed = false;
try {
  const page = await browser.newPage();
  page.on('console', (m) => console.log(`[page] ${m.text()}`));
  page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
  await page.route('**/*', (route) =>
    route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><meta charset=utf8><title>flux2-gemm</title>' }));
  await page.goto('http://127.0.0.1/');

  const res = await page.evaluate(async ({ shaderSrc, shapes }) => {
    (globalThis as any).__name = (f: any) => f; // tsx/esbuild keepNames shim
    const g = (navigator as any).gpu;
    if (!g) return { error: 'no navigator.gpu' };
    const adapter = await g.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { error: 'no adapter' };
    const info = adapter.info ?? {};
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    let lost: string | null = null;
    device.lost.then((l: any) => { lost = `${l.reason}: ${l.message}`; });

    const module = device.createShaderModule({ code: shaderSrc });
    const ci = await module.getCompilationInfo();
    const errs = ci.messages.filter((m: any) => m.type === 'error').map((m: any) => `${m.lineNum}: ${m.message}`);
    if (errs.length) return { error: `shader errors:\n${errs.join('\n')}` };
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'matmul_bt_bf16' },
    });

    // seeded RNG (mulberry32)
    let seed = 0xF10C5 >>> 0;
    const rng = () => {
      seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const f32buf = new Float32Array(1);
    const u32view = new Uint32Array(f32buf.buffer);
    const toBf16 = (x: number) => { f32buf[0] = x; return u32view[0] >>> 16; };
    const fromBf16 = (b: number) => { u32view[0] = b << 16; return f32buf[0]; };

    const out: any[] = [];
    for (const s of shapes) {
      if (lost) break;
      const { M, N, K } = s;
      const A = new Float32Array(M * K);
      for (let i = 0; i < A.length; i++) A[i] = (rng() * 2 - 1) * 0.5;
      const W = new Uint32Array(N * K / 2); // [N, K/2] u32, two bf16 each
      for (let i = 0; i < W.length; i++) {
        W[i] = toBf16((rng() * 2 - 1) * 0.1) | (toBf16((rng() * 2 - 1) * 0.1) << 16);
      }

      const mk = (size: number, usage: number) => device.createBuffer({ size, usage });
      const aBuf = mk(A.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(aBuf, 0, A);
      const wBuf = mk(W.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(wBuf, 0, W);
      const cBuf = mk(M * N * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const pBuf = mk(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(pBuf, 0, new Uint32Array([M, N, K, 0]));
      // dummy f32 B (binding 1 unused by bt_bf16 but layout 'auto' drops unused;
      // bind only what the entry actually uses)
      const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: aBuf } },
          { binding: 2, resource: { buffer: cBuf } },
          { binding: 3, resource: { buffer: pBuf } },
          { binding: 5, resource: { buffer: wBuf } },
        ],
      });

      const gx = Math.ceil(M / 16), gy = Math.ceil(N / 16);
      const runOnce = async () => {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(gx, gy, 1);
        pass.end();
        const t0 = performance.now();
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        return performance.now() - t0;
      };

      device.pushErrorScope('validation');
      await runOnce(); // warmup
      const times: number[] = [];
      for (let i = 0; i < 3 && !lost; i++) times.push(await runOnce());
      const vErr = await device.popErrorScope();
      if (vErr) { out.push({ name: s.name, error: vErr.message }); continue; }
      if (lost) { out.push({ name: s.name, error: `device lost: ${lost}` }); break; }

      // spot-check 64 random outputs vs CPU (f64 acc, identical bf16 weights)
      const readBuf = mk(M * N * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      {
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(cBuf, 0, readBuf, 0, M * N * 4);
        device.queue.submit([enc.finish()]);
        await readBuf.mapAsync(GPUMapMode.READ);
      }
      const C = new Float32Array(readBuf.getMappedRange());
      let maxRel = 0;
      for (let t = 0; t < 64; t++) {
        const m = Math.floor(rng() * M), n = Math.floor(rng() * N);
        let acc = 0;
        for (let k = 0; k < K; k += 2) {
          const packed = W[n * (K / 2) + k / 2];
          acc += A[m * K + k] * fromBf16(packed & 0xffff);
          acc += A[m * K + k + 1] * fromBf16(packed >>> 16);
        }
        const got = C[m * N + n];
        const rel = Math.abs(got - acc) / Math.max(1e-4, Math.abs(acc));
        if (rel > maxRel) maxRel = rel;
      }
      readBuf.unmap();
      readBuf.destroy();

      const best = Math.min(...times);
      const avg = times.reduce((a, b) => a + b, 0) / times.length;
      const gflop = (2 * M * N * K) / 1e9;
            // GFLOP / ms == TFLOP / s exactly (1e9 / 1e-3 = 1e12)
      out.push({ name: s.name, M, N, K, gflop, bestMs: best, avgMs: avg, tflops: gflop / avg, maxRel });
      aBuf.destroy(); wBuf.destroy(); cBuf.destroy(); pBuf.destroy();
    }
    return { adapter: `${info.vendor ?? '?'} ${info.architecture ?? '?'} ${info.description ?? ''}`, out, lost };
  }, { shaderSrc, shapes: SHAPES });

  if ((res as any).error) { console.log(`FAIL: ${(res as any).error}`); process.exit(1); }
  console.log(`adapter: ${(res as any).adapter}`);
  let worstMs = 0, worstName = '', minTflops = Infinity;
  for (const r of (res as any).out) {
    if (r.error) { failed = true; console.log(`  FAIL ${r.name}: ${r.error}`); continue; }
    const ok = r.maxRel < 1e-4;
    if (!ok) failed = true;
    console.log(
      `  ${ok ? 'ok  ' : 'FAIL'} ${r.name}: avg=${r.avgMs.toFixed(1)}ms best=${r.bestMs.toFixed(1)}ms ` +
      `${r.gflop.toFixed(0)} GFLOP -> ${r.tflops.toFixed(3)} TFLOPS eff (maxRel=${r.maxRel.toExponential(1)})`,
    );
    if (r.avgMs > worstMs) { worstMs = r.avgMs; worstName = r.name; }
    if (r.tflops < minTflops) minTflops = r.tflops;
  }
  if ((res as any).lost) { failed = true; console.log(`DEVICE LOST during bench: ${(res as any).lost}`); }

  console.log('\n=== Phase 0b gate ===');
  console.log(`worst dispatch: ${worstMs.toFixed(1)} ms (${worstName})`);
  console.log(`min effective TFLOPS: ${minTflops === Infinity ? 'n/a' : minTflops.toFixed(3)}`);
  console.log(`N-slicing mandatory (>800ms): ${worstMs > 800 ? 'YES' : 'no'}`);
  console.log(`GEMM upgrade needed (<0.5 TFLOPS): ${minTflops < 0.5 ? 'YES' : 'no'}`);
  if (HEADLESS) console.log('NOTE: HEADLESS run — numbers are NOT the gate; rerun headed on the 6700 XT.');
} finally {
  await browser.close();
}
process.exit(failed ? 1 : 0);
