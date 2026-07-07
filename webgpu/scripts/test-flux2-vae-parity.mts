/**
 * Phase 4 gate: FLUX.2 VAE decoder parity vs the Python fixture.
 *
 * All gates run in-page (vae.ts imports WGSL through vite; the 168 MB bf16
 * VAE loads through /api/hf-cache — vite dev server must be running):
 *   - unpack: CPU bn de-normalize + 2x2 unpatchify of dit.256.step3.latents
 *     vs vae.e2e256.latents_in (f32-exact op chain, <= 1e-6)
 *   - mid_attn: fixture Attention-module input (512,32,32) through
 *     GN + q/k/v + streaming SDPA + out + residual vs module output (1e-4)
 *   - decode 256/512: seeded random latents -> pixels; gate max abs error
 *     <= 2/255 in display units (pixels live in [-1,1] => 4/255 = 1.57e-2)
 *     plus rel-L2 <= 5e-3 as a distribution check
 *   - e2e 256: unpacked real denoised latents -> pixels vs vae.e2e256.pixels
 *
 * Run: npx tsx scripts/test-flux2-vae-parity.mts    (HEADLESS=1 to opt out)
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
  packed256: fixFile('dit.256.step3.latents'),
  e2eLatents: fixFile('vae.e2e256.latents_in'),
  e2ePixels: fixFile('vae.e2e256.pixels'),
  midIn: fixFile('vae.256.mid_attn.in'),
  midOut: fixFile('vae.256.mid_attn.out'),
  rand256Lat: fixFile('vae.256.rand_latents'),
  rand256Pix: fixFile('vae.256.rand_pixels'),
  rand512Lat: fixFile('vae.512.rand_latents'),
  rand512Pix: fixFile('vae.512.rand_pixels'),
};

// name -> [tolerance, metric]  (maxabs gates are in [-1,1] pixel units)
const TOLS: Record<string, number> = {
  'unpack e2e256 (exact f32)': 1e-6,
  'mid_attn 256': 1e-4,
  'pixels 256 rand relL2': 5e-3,
  'pixels 256 rand maxabs(<=2/255)': 4 / 255,
  'pixels 512 rand relL2': 5e-3,
  'pixels 512 rand maxabs(<=2/255)': 4 / 255,
  'pixels e2e256 relL2': 5e-3,
  'pixels e2e256 maxabs(<=2/255)': 4 / 255,
};

let failed = 0;
const report = (name: string, err: number) => {
  const tol = TOLS[name] ?? 5e-3;
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
      route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><meta charset=utf8><title>flux2-vae</title>' });
      return;
    }
    route.fallback(); // vite modules + /api/hf-cache weights
  });
  await page.goto(`${BASE}/__vae-parity`);

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
      const weights = await vaeMod.loadFlux2Vae(device, url);
      log(`VAE weights loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s (${weights.bufs.size} tensors)`);

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
      const maxAbs = (got: Float32Array, want: Float32Array) => {
        let m = 0;
        for (let i = 0; i < want.length; i++) m = Math.max(m, Math.abs(got[i] - want[i]));
        return m;
      };

      const results: { name: string; err: number }[] = [];

      // ---- 1) CPU unpack (bn de-normalize + unpatchify) ----
      const packed = await fetchF32(files.packed256);
      const unpacked = vaeMod.unpackLatents(packed, 16, 16, weights.bnMean, weights.bnVar);
      const wantUnpacked = await fetchF32(files.e2eLatents);
      results.push({ name: 'unpack e2e256 (exact f32)', err: relL2(unpacked, wantUnpacked) });

      const dec = new vaeMod.Flux2VaeDecoder(device, weights);

      // ---- 2) mid-attention module ----
      {
        const xin = await fetchF32(files.midIn);
        const got = await dec.midAttn(xin, 32);
        const want = await fetchF32(files.midOut);
        results.push({ name: 'mid_attn 256', err: relL2(got, want) });
      }

      // ---- 3) decoder on seeded random latents ----
      for (const [px, latF, pixF] of [
        [256, files.rand256Lat, files.rand256Pix],
        [512, files.rand512Lat, files.rand512Pix],
      ] as [number, string, string][]) {
        const lat = await fetchF32(latF);
        const t = Date.now();
        const pix = await dec.decode(lat, px / 8, (s: string) => log(`${px}px ${s}...`));
        log(`${px}px decode in ${((Date.now() - t) / 1000).toFixed(1)}s`);
        const want = await fetchF32(pixF);
        results.push({ name: `pixels ${px} rand relL2`, err: relL2(pix, want) });
        results.push({ name: `pixels ${px} rand maxabs(<=2/255)`, err: maxAbs(pix, want) });
      }

      // ---- 4) e2e: real denoised 256px latents ----
      {
        const t = Date.now();
        const pix = await dec.decode(unpacked, 32);
        log(`e2e 256px decode in ${((Date.now() - t) / 1000).toFixed(1)}s`);
        const want = await fetchF32(files.e2ePixels);
        results.push({ name: 'pixels e2e256 relL2', err: relL2(pix, want) });
        results.push({ name: 'pixels e2e256 maxabs(<=2/255)', err: maxAbs(pix, want) });
      }

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

console.log(`\n=== Phase 4 VAE parity: ${failed === 0 ? 'PASS' : `FAIL (${failed})`} ===`);
if (HEADLESS) console.log('NOTE: HEADLESS run — rerun headed on the 6700 XT before commit.');
process.exit(failed === 0 ? 0 : 1);
