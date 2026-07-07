/**
 * Phase 6 gate (P6.2): FLUX.2 VAE ENCODER parity vs the Python fixture.
 *
 * All gates run in-page (vae.ts imports WGSL through vite; the encoder half
 * loads through /api/hf-cache — vite dev server must be running):
 *   - encode 256: fixture image (3,256,256) in [-1,1] -> GPU Encoder ->
 *     posterior mode (32,32,32) vs vae.enc256.latents_mode (relL2 <= 1e-3)
 *   - packRefLatents: CPU patchify + bn normalize of the FIXTURE mode vs
 *     edit.image_latents (256,128) — exact f32 op chain, <= 1e-6
 *   - chain: packRefLatents(GPU mode) vs edit.image_latents (<= 1e-3) —
 *     this is the tensor the DiT actually consumes in the edit path
 *
 * Run: npx tsx scripts/test-flux2-vae-encode.mts    (HEADLESS=1 to opt out)
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
  image: fixFile('vae.enc256.image'),
  mode: fixFile('vae.enc256.latents_mode'),
  packed: fixFile('edit.image_latents'),
};

const TOLS: Record<string, number> = {
  'encode 256 mode relL2': 1e-3,
  'packRefLatents (exact f32)': 1e-6,
  'encode->pack chain relL2': 1e-3,
};

let failed = 0;
const report = (name: string, err: number) => {
  const tol = TOLS[name] ?? 1e-3;
  const ok = err <= tol && Number.isFinite(err);
  if (!ok) failed++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'} ${name.padEnd(32)} err ${err.toExponential(3)} (tol ${tol.toExponential(1)})`);
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
      route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><meta charset=utf8><title>flux2-vae-enc</title>' });
      return;
    }
    route.fallback(); // vite modules + /api/hf-cache weights
  });
  await page.goto(`${BASE}/__vae-encode-parity`);

  const res: any = await page.evaluate(async ({ files }) => {
    (globalThis as any).__name = (f: any) => f; // tsx/esbuild keepNames shim
    const log = (s: string) => console.log(s);
    try {
      const hub: any = await import('/src/model/hf-hub.ts');
      const vaeMod: any = await import('/src/diffusion/vae.ts');
      hub.useLocalCache();
      const url = hub.resolveFileUrl('local/flux.2-klein-4b', 'vae/diffusion_pytorch_model.safetensors');

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
      const weights = await vaeMod.loadFlux2Vae(device, url, undefined, 'encoder');
      log(`VAE encoder weights loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s (${weights.bufs.size} tensors)`);

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

      const results: { name: string; err: number }[] = [];
      const image = await fetchF32(files.image);
      const wantMode = await fetchF32(files.mode);
      const wantPacked = await fetchF32(files.packed);

      // ---- 1) GPU encode ----
      const dec = new vaeMod.Flux2VaeDecoder(device, weights);
      const t = Date.now();
      const mode = await dec.encode(image, 256, (s: string) => log(`encode ${s}...`));
      log(`256px encode in ${((Date.now() - t) / 1000).toFixed(1)}s`);
      results.push({ name: 'encode 256 mode relL2', err: relL2(mode, wantMode) });

      // ---- 2) CPU pack of the fixture mode (exact) ----
      const packedRef = vaeMod.packRefLatents(wantMode, 32, weights.bnMean, weights.bnVar);
      results.push({ name: 'packRefLatents (exact f32)', err: relL2(packedRef, wantPacked) });

      // ---- 3) full chain: GPU mode -> pack (the DiT-facing tensor) ----
      const packedGpu = vaeMod.packRefLatents(mode, 32, weights.bnMean, weights.bnVar);
      results.push({ name: 'encode->pack chain relL2', err: relL2(packedGpu, wantPacked) });

      dec.destroy();
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
    for (const r of res.results) report(r.name, r.err);
  }
} finally {
  await browser.close();
}

console.log(`\n=== P6.2 VAE encode parity: ${failed === 0 ? 'PASS' : `FAIL (${failed})`} ===`);
if (HEADLESS) console.log('NOTE: HEADLESS run — rerun headed on the 6700 XT before commit.');
process.exit(failed === 0 ? 0 : 1);
