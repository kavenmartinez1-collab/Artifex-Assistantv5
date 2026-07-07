/**
 * Phase 0c spike: streaming non-causal attention (attention_stream.wgsl)
 * at the FLUX.2 shapes that exceed attention.wgsl's 2048-position cap:
 *   - DiT joint attention: 768 / 2560 / 4608 pos, 24 heads x 128 dim
 *   - VAE mid-block attention: 4096 / 16384 pos, 1 head x 512 dim
 *
 * TDR discipline (two headed runs device-hung before this version):
 *   1. every shape is dispatched in query slices via q_offset;
 *   2. the slice budget is ADAPTIVE: it starts at a worst-case-measured
 *      2 GFLOP (~100 ms at the naive kernel's 0.02 TFLOPS) and is
 *      recalibrated to ~350 ms from each shape's measured throughput;
 *   3. each shape runs in its own page so a device loss can't cascade.
 *
 * K is fed TRANSPOSED [H, D, S] per the kernel contract (coalesced scores).
 *
 * Correctness: 16 random (query, head) rows per shape vs CPU f64 full-softmax
 * reference. Timing: per-slice wall time -> per-shape totals.
 *
 * Run: npx tsx scripts/test-flux2-attn-spike.mts
 *      HEADLESS=1 npx tsx scripts/test-flux2-attn-spike.mts  (logic smoke only)
 *      ENTRY=attention_stream_qt QTILE=8   (q-tiled DiT variant; D=128
 *      shapes only — the VAE D=512 shapes are skipped, they stay on the
 *      original entry)
 */
import { chromium } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const shaderSrc = readFileSync(resolve(here, '../src/shaders/attention_stream.wgsl'), 'utf8');

const ENTRY = process.env.ENTRY ?? 'attention_stream';
const QTILE = parseInt(process.env.QTILE ?? '1'); // queries per workgroup

const SHAPES = [
  { name: 'joint 256px 768x24hx128d', S: 768, H: 24, D: 128 },
  { name: 'joint 512px-edit 2560x24hx128d', S: 2560, H: 24, D: 128 },
  { name: 'joint 1024px 4608x24hx128d', S: 4608, H: 24, D: 128 },
  { name: 'vae-mid 512px 4096x1hx512d', S: 4096, H: 1, D: 512 },
  { name: 'vae-mid 1024px 16384x1hx512d', S: 16384, H: 1, D: 512 },
].filter((s) => QTILE === 1 || s.D <= 128)
 .map((s) => ({ ...s, gflop: (4 * s.S * s.S * s.D * s.H) / 1e9 }));

const HEADLESS = process.env.HEADLESS === '1';
const browser = await chromium.launch({
  channel: 'chrome',
  headless: HEADLESS,
  args: ['--enable-unsafe-webgpu', '--window-size=360,240', '--window-position=20,20'],
});

let failed = false;
let worstSliceMs = 0;
let worstName = '';
let budgetGflop = 2; // worst-case seed: ~100 ms at 0.02 TFLOPS
try {
  for (const shape of SHAPES) {
    const page = await browser.newPage();
    page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
    await page.route('**/*', (route) =>
      route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><meta charset=utf8><title>flux2-attn</title>' }));
    await page.goto('http://127.0.0.1/');

    // slice so each submit stays within the current GFLOP budget
    const nSlices = Math.max(1, Math.ceil(shape.gflop / budgetGflop));
    const sliceQ = Math.ceil(shape.S / nSlices);

    let r: any;
    try {
      r = await page.evaluate(async ({ shaderSrc, s, sliceQ, entry, qtile }) => {
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
          compute: { module, entryPoint: entry },
        });

        let seed = (0xA77E17 ^ s.S ^ (s.D << 8)) >>> 0;
        const rng = () => {
          seed = (seed + 0x6D2B79F5) | 0;
          let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
          t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };

        const { S, H, D } = s;
        const n = S * H * D;
        const Q = new Float32Array(n), K = new Float32Array(n), V = new Float32Array(n);
        for (let i = 0; i < n; i++) {
          Q[i] = (rng() * 2 - 1) * 0.7;
          K[i] = (rng() * 2 - 1) * 0.7;
          V[i] = (rng() * 2 - 1) * 1.0;
        }
        // K row-major [S, H*D] -> KT [H, D, S]
        const KT = new Float32Array(n);
        for (let j = 0; j < S; j++) {
          for (let h = 0; h < H; h++) {
            for (let d = 0; d < D; d++) {
              KT[(h * D + d) * S + j] = K[(j * H + h) * D + d];
            }
          }
        }

        const mk = (size: number, usage: number) => device.createBuffer({ size, usage });
        const qB = mk(Q.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        const kB = mk(KT.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        const vB = mk(V.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        device.queue.writeBuffer(qB, 0, Q);
        device.queue.writeBuffer(kB, 0, KT);
        device.queue.writeBuffer(vB, 0, V);
        const oB = mk(n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

        const nSlices = Math.ceil(S / sliceQ);
        const binds: any[] = [];
        for (let sl = 0; sl < nSlices; sl++) {
          const pB = mk(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
          device.queue.writeBuffer(pB, 0, new Uint32Array([H, D, S, S, sl * sliceQ]));
          binds.push(device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: qB } },
              { binding: 1, resource: { buffer: kB } },
              { binding: 2, resource: { buffer: vB } },
              { binding: 3, resource: { buffer: oB } },
              { binding: 4, resource: { buffer: pB } },
            ],
          }));
        }

        const runAllSlices = async () => {
          const sliceMs: number[] = [];
          for (let sl = 0; sl < nSlices; sl++) {
            const qCount = Math.min(sliceQ, S - sl * sliceQ);
            const enc = device.createCommandEncoder();
            const pass = enc.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, binds[sl]);
            pass.dispatchWorkgroups(Math.ceil(qCount / qtile), H, 1);
            pass.end();
            const t0 = performance.now();
            device.queue.submit([enc.finish()]);
            await device.queue.onSubmittedWorkDone();
            sliceMs.push(performance.now() - t0);
            if (lost) break;
          }
          return sliceMs;
        };

        device.pushErrorScope('validation');
        await runAllSlices(); // warmup
        const runs: number[][] = [];
        for (let i = 0; i < 3 && !lost; i++) runs.push(await runAllSlices());
        const vErr = await device.popErrorScope();
        if (vErr) return { error: vErr.message };
        if (lost) return { error: `device lost: ${lost}` };

        // readback + CPU row spot-check (f64 softmax, original K layout)
        const rB = mk(n * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        {
          const enc = device.createCommandEncoder();
          enc.copyBufferToBuffer(oB, 0, rB, 0, n * 4);
          device.queue.submit([enc.finish()]);
          await rB.mapAsync(GPUMapMode.READ);
        }
        const O = new Float32Array(rB.getMappedRange());
        let maxRel = 0;
        const scale = 1 / Math.sqrt(D);
        for (let trial = 0; trial < 16; trial++) {
          const qi = Math.floor(rng() * S), h = Math.floor(rng() * H);
          const qb = (qi * H + h) * D;
          const sc = new Float64Array(S);
          let m = -Infinity;
          for (let j = 0; j < S; j++) {
            let dot = 0;
            const kb = (j * H + h) * D;
            for (let d = 0; d < D; d++) dot += Q[qb + d] * K[kb + d];
            sc[j] = dot * scale;
            if (sc[j] > m) m = sc[j];
          }
          let l = 0;
          for (let j = 0; j < S; j++) { sc[j] = Math.exp(sc[j] - m); l += sc[j]; }
          for (let d = 0; d < D; d++) {
            let acc = 0;
            for (let j = 0; j < S; j++) acc += sc[j] * V[(j * H + h) * D + d];
            const want = acc / l;
            const got = O[qb + d];
            const rel = Math.abs(got - want) / Math.max(1e-3, Math.abs(want));
            if (rel > maxRel) maxRel = rel;
          }
        }
        rB.unmap();

        const totals = runs.map((rr) => rr.reduce((a, b) => a + b, 0));
        const best = totals.indexOf(Math.min(...totals));
        return {
          adapter: `${info.vendor ?? '?'} ${info.architecture ?? '?'}`,
          nSlices,
          sliceMs: runs[best].map((x) => Math.round(x * 10) / 10),
          totalMs: totals[best],
          maxRel,
        };
      }, { shaderSrc, s: shape, sliceQ, entry: ENTRY, qtile: QTILE });
    } catch (e: any) {
      r = { error: `evaluate crashed (likely TDR): ${e.message?.split('\n')[0]}` };
    }
    await page.close().catch(() => {});

    if (r.error) {
      failed = true;
      console.log(`  FAIL ${shape.name}: ${r.error}`);
      continue;
    }
    const ok = r.maxRel < 1e-4;
    if (!ok) failed = true;
    const maxSlice = Math.max(...r.sliceMs);
    if (maxSlice > worstSliceMs) { worstSliceMs = maxSlice; worstName = shape.name; }
    const tflops = shape.gflop / r.totalMs;
    console.log(
      `  ${ok ? 'ok  ' : 'FAIL'} ${shape.name}: total=${r.totalMs.toFixed(1)}ms in ${r.nSlices} slice(s) ` +
      `[${r.sliceMs.join(', ')}] ${shape.gflop.toFixed(1)} GFLOP -> ${tflops.toFixed(3)} TFLOPS ` +
      `(maxRel=${r.maxRel.toExponential(1)})`,
    );
    // recalibrate: target ~350 ms per slice at measured throughput
    budgetGflop = Math.max(2, tflops * 350 * 0.8);
  }

  console.log('\n=== Phase 0c gate ===');
  console.log(`worst single slice: ${worstSliceMs.toFixed(1)} ms (${worstName})`);
  console.log(`slice budget OK (<800ms): ${worstSliceMs < 800 ? 'yes' : 'NO — shrink budget seed'}`);
  if (HEADLESS) console.log('NOTE: HEADLESS run — timing is NOT the gate; rerun headed on the 6700 XT.');
} finally {
  await browser.close();
}
process.exit(failed ? 1 : 0);
