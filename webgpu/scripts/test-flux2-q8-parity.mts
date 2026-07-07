/**
 * Phase 7 gate (P7.2): Q8_0 all-resident DiT path.
 *
 * Gate A (bit-exact): GPU dequant_q8_bf16 output vs a CPU reference dequant
 * of the same .artq tensor bytes. The kernel computes f32(d_f16)*q (exact in
 * f32) then RNE-rounds to bf16 — both sides deterministic, so ZERO u16
 * mismatches is the gate.
 *
 * Gate B (quant quality): full 4-step 256px t2i trajectory through
 * loadFlux2DitQ8 + Flux2Pipeline vs the SAME bf16 fixture used by
 * test-flux2-dit-parity.mts. Error is now Q8 weight error, not kernel error.
 * Measured on the 6700 XT (2026-07-07): noise_pred relL2 1.08e-2, 1-cos
 * 5.6e-5; latents 7.0e-4 / 6.6e-3 / 2.2e-2 / 9.9e-2 — the per-step model
 * output stays direction-true while the Euler trajectory amplifies the
 * perturbation ~3x/step (sample drifts to a nearby but different point, as
 * any weight perturbation does). Gates: noise_pred relL2 <= 5e-2 AND 1-cos
 * <= 2e-3 (the actual quant quality), latents <= 2e-1 (guards against
 * blow-up only; the shipping quality bar is the P7.4 perceptual e2e).
 *
 * Run: npx tsx scripts/test-flux2-q8-parity.mts   (vite dev server up;
 *      HEADLESS=1 to opt out of headed)
 */
import { chromium } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const fixDir = resolve(here, 'flux2_fixture');
const manifest = JSON.parse(readFileSync(resolve(fixDir, 'manifest.json'), 'utf8'));
const BASE = 'http://127.0.0.1:5173';

function fixFile(name: string): string {
  const t = manifest.tensors[name];
  if (!t) throw new Error(`fixture tensor missing: ${name}`);
  return t.file;
}

const files = {
  pe: fixFile('te.p0.prompt_embeds'),
  noise: fixFile('dit.256.noise'),
  np0: fixFile('dit.256.step0.noise_pred'),
  lat: [0, 1, 2, 3].map((i) => fixFile(`dit.256.step${i}.latents`)),
};

let failed = 0;
const report = (name: string, err: number, tol: number) => {
  const ok = err <= tol && Number.isFinite(err);
  if (!ok) failed++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'} ${name.padEnd(30)} ${err.toExponential(3)} (tol ${tol.toExponential(0)})`);
};

const HEADLESS = process.env.HEADLESS === '1';
const browser = await chromium.launch({
  channel: 'chrome',
  headless: HEADLESS,
  args: ['--enable-unsafe-webgpu', '--window-size=380,240', '--window-position=20,20'],
});

try {
  const page = await browser.newPage();
  page.on('console', (msg) => console.log(`  [page] ${msg.text()}`));
  page.on('pageerror', (e) => console.log(`  [pageerror] ${e.message}`));
  await page.route('**/*', (route) => {
    const url = route.request().url();
    if (url.includes('/__fixture/')) {
      const file = url.split('/__fixture/')[1];
      try {
        route.fulfill({ body: readFileSync(resolve(fixDir, file)), contentType: 'application/octet-stream' });
      } catch (e) {
        route.fulfill({ status: 404, body: String(e) });
      }
      return;
    }
    if (route.request().resourceType() === 'document') {
      route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><meta charset=utf8><title>flux2-q8</title>' });
      return;
    }
    route.fallback(); // vite modules + /api/hf-cache (.artq)
  });
  await page.goto(`${BASE}/__q8-parity`);

  const res: any = await page.evaluate(async ({ files }) => {
    (globalThis as any).__name = (f: any) => f; // tsx/esbuild keepNames shim
    const log = (s: string) => console.log(s);
    try {
      const hub: any = await import('/src/model/hf-hub.ts');
      const loaderMod: any = await import('/src/diffusion/flux2-loader.ts');
      const pipeMod: any = await import('/src/diffusion/flux2-pipeline.ts');
      const dequantWGSL: string =
        (await import('/src/shaders/dequant_q8.wgsl?raw')).default;
      hub.useLocalCache();
      const url = hub.resolveFileUrl('local/flux.2-klein-4b', 'transformer/diffusion_pytorch_model.q8_0.artq');

      const g = (navigator as any).gpu;
      if (!g) return { error: 'no navigator.gpu' };
      const adapter = await g.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) return { error: 'no adapter' };
      const device = await adapter.requestDevice({
        requiredLimits: {
          maxBufferSize: adapter.limits.maxBufferSize,
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        },
      });
      let lost: string | null = null;
      device.lost.then((l: any) => { lost = `${l.reason}: ${l.message}`; });
      device.onuncapturederror = (e: any) => log(`UNCAPTURED GPU ERROR: ${e.error?.message}`);
      const info = adapter.info ?? {};
      log(`adapter: ${info.vendor ?? '?'} ${info.architecture ?? '?'}`);
      const results: { name: string; err: number; tol: number }[] = [];

      // ---------- Gate A: bit-exact dequant on sample tensors ----------
      {
        const head = new DataView(await (await fetch(url, { headers: { Range: 'bytes=0-15' } })).arrayBuffer());
        const jsonLen = Number(head.getBigUint64(8, true));
        const jsonBuf = await (await fetch(url, { headers: { Range: `bytes=16-${15 + jsonLen}` } })).arrayBuffer();
        const tensors = JSON.parse(new TextDecoder().decode(jsonBuf)).tensors as any[];
        const dataStart = 16 + jsonLen;
        const byName = new Map(tensors.map((t) => [t.name, t]));

        const f16f32 = (h: number) => {
          const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
          if (e === 0) return s * m * 2 ** -24;
          if (e === 31) return m ? NaN : s * Infinity;
          return s * (1 + m / 1024) * 2 ** (e - 15);
        };
        const rne = (f: number) => {
          const b = new DataView(new ArrayBuffer(4));
          b.setFloat32(0, f, true);
          const u = b.getUint32(0, true);
          return ((u + 0x7fff + ((u >>> 16) & 1)) >>> 16) & 0xffff;
        };

        const pipe = device.createComputePipeline({
          layout: 'auto',
          compute: { module: device.createShaderModule({ code: dequantWGSL }), entryPoint: 'dequant_q8_bf16' },
        });

        for (const name of ['x_embedder.weight', 'proj_out.weight',
          'time_guidance_embed.timestep_embedder.linear_1.weight']) {
          const t = byName.get(name);
          if (!t) return { error: `artq tensor missing: ${name}` };
          const end = t.scaleOffset + t.scaleBytes;
          const raw = await (await fetch(url, {
            headers: { Range: `bytes=${dataStart + t.offset}-${dataStart + end - 1}` },
          })).arrayBuffer();
          if (raw.byteLength !== end - t.offset) return { error: `${name}: short read ${raw.byteLength}` };
          const [n, k] = t.shape;
          const blocks = (n * k) / 32;
          const src = device.createBuffer({ size: Math.ceil(raw.byteLength / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
          device.queue.writeBuffer(src, 0, raw, 0, raw.byteLength - (raw.byteLength % 4));
          const dst = device.createBuffer({ size: n * k * 2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
          const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
          device.queue.writeBuffer(uni, 0, new Uint32Array([blocks, (t.scaleOffset - t.offset) / 4, 0, 0]));
          const enc = device.createCommandEncoder();
          const pass = enc.beginComputePass();
          pass.setPipeline(pipe);
          pass.setBindGroup(0, device.createBindGroup({
            layout: pipe.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: src } },
              { binding: 1, resource: { buffer: dst } },
              { binding: 2, resource: { buffer: uni } }],
          }));
          pass.dispatchWorkgroups(Math.ceil(blocks / 256), 1, 1);
          pass.end();
          const staging = device.createBuffer({ size: n * k * 2, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
          enc.copyBufferToBuffer(dst, 0, staging, 0, n * k * 2);
          device.queue.submit([enc.finish()]);
          await staging.mapAsync(GPUMapMode.READ);
          const got = new Uint16Array(staging.getMappedRange()).slice();
          staging.unmap();

          // CPU reference
          const q = new Int8Array(raw, 0, n * k);
          const scales = new Uint16Array(raw, t.scaleOffset - t.offset, blocks);
          let mismatches = 0;
          for (let b = 0; b < blocks; b++) {
            const d = f16f32(scales[b]);
            for (let j = 0; j < 32; j++) {
              const want = rne(Math.fround(d * q[b * 32 + j]));
              if (got[b * 32 + j] !== want && mismatches++ < 3) {
                log(`  ${name} blk ${b} j ${j}: got ${got[b * 32 + j]} want ${want} (d=${d} q=${q[b * 32 + j]})`);
              }
            }
          }
          results.push({ name: `dequant bit-exact ${name.split('.')[0]}`, err: mismatches, tol: 0 });
          src.destroy(); dst.destroy(); uni.destroy(); staging.destroy();
        }
      }

      // ---------- Gate B: full Q8 DiT 4-step trajectory (256px t2i) ----------
      const t0 = Date.now();
      let lastLog = 0;
      const weights = await loaderMod.loadFlux2DitQ8(device, url, (done: number, total: number) => {
        if (done - lastLog > 512 * 1024 * 1024 || done === total) {
          lastLog = done;
          log(`Q8 weights: ${(done / 2 ** 30).toFixed(2)} / ${(total / 2 ** 30).toFixed(2)} GiB`);
        }
      });
      log(`Q8 weights loaded in ${((Date.now() - t0) / 1000).toFixed(0)}s (max mat ${(weights.maxMatElems * 2 / 2 ** 20).toFixed(0)} MB scratch/slot)`);

      const fetchF32 = async (f: string) =>
        new Float32Array(await (await fetch(`/__fixture/${f}`)).arrayBuffer());
      const relL2 = (got: Float32Array, want: Float32Array) => {
        let num = 0, den = 0;
        for (let i = 0; i < want.length; i++) {
          const d = got[i] - want[i];
          num += d * d; den += want[i] * want[i];
        }
        return Math.sqrt(num / Math.max(den, 1e-30));
      };
      const cosSim = (a: Float32Array, b: Float32Array) => {
        let ab = 0, aa = 0, bb = 0;
        for (let i = 0; i < a.length; i++) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
        return ab / Math.sqrt(aa * bb);
      };

      const pipe2 = new pipeMod.Flux2Pipeline(device, weights);
      const pe = await fetchF32(files.pe);
      const noise = await fetchF32(files.noise);
      const latRecs: { i: number; lat: Float32Array; np: Float32Array }[] = [];
      const tRun = Date.now();
      await pipe2.generate({
        promptEmbeds: pe, noise, gridH: 16, gridW: 16,
        onLatents: (i: number, lat: Float32Array, np: Float32Array) =>
          latRecs.push({ i, lat, np: np.slice() }),
        onProgress: (i: number, n: number) => log(`Q8 256px step ${i + 1}/${n}...`),
      });
      log(`Q8 256px x4 steps in ${((Date.now() - tRun) / 1000).toFixed(1)}s`);

      {
        const want = await fetchF32(files.np0);
        results.push({ name: 'noise_pred q8 step0', err: relL2(latRecs[0].np, want), tol: 5e-2 });
        results.push({ name: 'noise_pred q8 step0 (1-cos)', err: 1 - cosSim(latRecs[0].np, want), tol: 2e-3 });
      }
      for (const r of latRecs) {
        const want = await fetchF32(files.lat[r.i]);
        results.push({ name: `latents q8 step${r.i}`, err: relL2(r.lat, want), tol: 2e-1 });
      }

      pipe2.destroy();
      weights.destroy();
      if (lost) return { error: `device lost: ${lost}`, results };
      return { results };
    } catch (e: any) {
      return { error: `${e?.message ?? e}\n${e?.stack ?? ''}` };
    }
  }, { files });

  if (res.error) {
    console.log(`FAIL: ${res.error}`);
    failed++;
  }
  if (res.results) {
    console.log('\nGPU gates:');
    for (const r of res.results) report(r.name, r.err, r.tol);
  }
} finally {
  await browser.close();
}

console.log(`\n=== P7.2 Q8 parity: ${failed === 0 ? 'PASS' : `FAIL (${failed})`} ===`);
if (HEADLESS) console.log('NOTE: HEADLESS run — rerun headed on the 6700 XT before commit.');
process.exit(failed === 0 ? 0 : 1);
