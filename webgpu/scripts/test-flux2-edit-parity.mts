/**
 * Phase 6 gate (P6.3): FLUX.2 EDIT-path DiT parity vs the Python fixture.
 *
 * Fixture (gen_flux2_fixture.py --stage edit): 256px edit run, prompt p1,
 * ref = vae.enc256.image encoded+packed to edit.image_latents (256 tokens,
 * T=10 ids), gen noise seed 46 (edit.noise). Joint seq = 512 txt + 256 gen
 * + 256 ref = 1024.
 *
 * Node-side: scheduler mu vs edit.mu (exact — mu from GEN tokens only,
 * computed before the ref concat).
 * GPU (vite dev server on 127.0.0.1:5173; 7.75 GB DiT via /api/hf-cache):
 *   - noise_pred step 0 (gen slice): relL2 <= 5e-3 AND 1-cos <= 1e-4
 *   - 4-step latent trajectory vs edit.step{i}.latents (<= 5e-3 each)
 *
 * Run: npx tsx scripts/test-flux2-edit-parity.mts    (HEADLESS=1 to opt out)
 */
import { chromium } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { flux2Schedule } from '../src/diffusion/scheduler';

const here = dirname(fileURLToPath(import.meta.url));
const fixDir = resolve(here, 'flux2_fixture');
const manifest = JSON.parse(readFileSync(resolve(fixDir, 'manifest.json'), 'utf8'));
const BASE = 'http://127.0.0.1:5173';

function fixFile(name: string): string {
  const t = manifest.tensors[name];
  if (!t) throw new Error(`fixture tensor missing: ${name}`);
  return t.file;
}

let failed = 0;
const report = (name: string, err: number, tol: number) => {
  const ok = err <= tol && Number.isFinite(err);
  if (!ok) failed++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'} ${name.padEnd(28)} relL2 ${err.toExponential(3)} (tol ${tol.toExponential(0)})`);
};

// ---------- 1) scheduler mu (node, exact; gen tokens only) ----------
{
  const wantMu = manifest.meta['edit.mu'] as number;
  const got = flux2Schedule(256, 4);
  report('edit.mu (exact)', Math.abs(got.mu - wantMu), 1e-12);
}

const files = {
  pe: fixFile('te.p1.prompt_embeds'),
  noise: fixFile('edit.noise'),
  refLat: fixFile('edit.image_latents'),
  np0: fixFile('edit.step0.noise_pred'),
  lat: [0, 1, 2, 3].map((i) => fixFile(`edit.step${i}.latents`)),
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
      route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><meta charset=utf8><title>flux2-edit</title>' });
      return;
    }
    route.fallback(); // vite modules + /api/hf-cache weights
  });
  await page.goto(`${BASE}/__edit-parity`);

  const res: any = await page.evaluate(async ({ files }) => {
    (globalThis as any).__name = (f: any) => f; // tsx/esbuild keepNames shim
    const log = (s: string) => console.log(s);
    try {
      const hub: any = await import('/src/model/hf-hub.ts');
      const loaderMod: any = await import('/src/diffusion/flux2-loader.ts');
      const pipeMod: any = await import('/src/diffusion/flux2-pipeline.ts');
      hub.useLocalCache();
      const url = hub.resolveFileUrl('local/flux.2-klein-4b', 'transformer/diffusion_pytorch_model.safetensors');

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
      log(`adapter: ${info.vendor ?? '?'} ${info.architecture ?? '?'} ${info.description ?? ''}`);

      const t0 = Date.now();
      let lastLog = 0;
      const weights = await loaderMod.loadFlux2Dit(device, url, (done: number, total: number) => {
        if (done - lastLog > 512 * 1024 * 1024 || done === total) {
          lastLog = done;
          log(`weights: ${(done / 2 ** 30).toFixed(2)} / ${(total / 2 ** 30).toFixed(2)} GiB`);
        }
      });
      log(`weights loaded in ${((Date.now() - t0) / 1000).toFixed(0)}s`);

      const pipe = new pipeMod.Flux2Pipeline(device, weights);
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

      const pe = await fetchF32(files.pe);
      const noise = await fetchF32(files.noise);
      const refLat = await fetchF32(files.refLat);
      const results: { name: string; err: number }[] = [];

      // ---- 256px edit: 512 txt + 256 gen + 256 ref (T=10) ----
      const latRecs: { i: number; lat: Float32Array; np: Float32Array }[] = [];
      const tRun = Date.now();
      await pipe.generate({
        promptEmbeds: pe, noise, gridH: 16, gridW: 16,
        refLatents: refLat, refs: [{ h: 16, w: 16, t: 10 }],
        onLatents: (i: number, lat: Float32Array, np: Float32Array) =>
          latRecs.push({ i, lat, np: np.slice() }),
        onProgress: (i: number, n: number) => log(`edit 256px step ${i + 1}/${n}...`),
      });
      log(`edit 256px x4 steps in ${((Date.now() - tRun) / 1000).toFixed(1)}s`);

      {
        const want = await fetchF32(files.np0);
        const np0 = latRecs[0].np;
        results.push({ name: 'noise_pred edit step0', err: relL2(np0, want) });
        results.push({ name: 'noise_pred edit step0 (1-cos)', err: 1 - cosSim(np0, want) });
      }
      for (const r of latRecs) {
        const want = await fetchF32(files.lat[r.i]);
        results.push({ name: `latents edit step${r.i}`, err: relL2(r.lat, want) });
      }

      pipe.destroy();
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
    console.log('\nGPU parity:');
    for (const r of res.results) {
      const tol = r.name.includes('(1-cos)') ? 1e-4 : 5e-3;
      report(r.name, r.err, tol);
    }
  }
} finally {
  await browser.close();
}

console.log(`\n=== P6.3 edit parity: ${failed === 0 ? 'PASS' : `FAIL (${failed})`} ===`);
if (HEADLESS) console.log('NOTE: HEADLESS run — rerun headed on the 6700 XT before commit.');
process.exit(failed === 0 ? 0 : 1);
