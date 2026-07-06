/**
 * Phase 3 gate: FLUX.2 DiT parity vs the Python fixture.
 *
 * Node-side (no GPU):
 *   - scheduler: mu + sigmas + timesteps vs sched.{256,512,1024} — exact
 *     (<=1e-12; the TS code replicates numpy/torch f32 rounding)
 *
 * GPU (vite dev server must already be running on 127.0.0.1:5173 — the page
 * imports /src/diffusion/*.ts through vite and Range-loads the 7.75 GB DiT
 * through /api/hf-cache; weights load takes a few minutes):
 *   - 256px step 0 with fixture noise + fixture (f32-exact) prompt_embeds,
 *     captures compared in-page against dit.cap.*:
 *       temb/modulations (1e-3: f32 transcendental divergence in the
 *       sinusoid; layout bugs are O(1)), embedders (1e-5), double block 0
 *       (5e-4), double 4 / single 0 (1e-3), single 19 (2e-3), norm_out (5e-3)
 *   - noise_pred step 0: rel-L2 <= 5e-3 AND cosine >= 0.9999
 *   - 4-step latent trajectories at 256px and 512px vs dit.{px}.step{i}.latents
 *     (<= 5e-3 each; latents are noise-dominated so drift shows late)
 *
 * Run: npx tsx scripts/test-flux2-dit-parity.mts    (HEADLESS=1 to opt out)
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
function loadF32(name: string): Float32Array {
  const raw = readFileSync(resolve(fixDir, fixFile(name)));
  return new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
}

let failed = 0;
const report = (name: string, err: number, tol: number) => {
  const ok = err <= tol && Number.isFinite(err);
  if (!ok) failed++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'} ${name.padEnd(28)} relL2 ${err.toExponential(3)} (tol ${tol.toExponential(0)})`);
};

// ---------- 1) scheduler gate (node, exact) ----------
console.log('scheduler: sigmas/timesteps/mu vs fixture');
for (const px of [256, 512, 1024]) {
  const seqLen = manifest.meta[`sched.${px}.image_seq_len`] as number;
  const wantMu = manifest.meta[`sched.${px}.mu`] as number;
  const wantSig = loadF32(`sched.${px}.sigmas`);
  const wantTs = loadF32(`sched.${px}.timesteps`);
  const got = flux2Schedule(seqLen, 4);
  let maxErr = Math.abs(got.mu - wantMu);
  for (let i = 0; i < wantSig.length; i++) maxErr = Math.max(maxErr, Math.abs(got.sigmas[i] - wantSig[i]));
  for (let i = 0; i < wantTs.length; i++) maxErr = Math.max(maxErr, Math.abs(got.timesteps[i] - wantTs[i]) / 1000);
  report(`sched.${px} (exact)`, maxErr, 1e-12);
}

// ---------- 2) GPU: DiT parity ----------
// capture name -> [fixture tensor, tolerance]
const CAPS: Record<string, [string, number]> = {
  temb: ['dit.cap.temb.out', 1e-3],
  mod_double_img: ['dit.cap.mod_double_img.out', 1e-3],
  mod_double_txt: ['dit.cap.mod_double_txt.out', 1e-3],
  mod_single: ['dit.cap.mod_single.out', 1e-3],
  context_embedder: ['dit.cap.context_embedder.out', 1e-5],
  x_embedder: ['dit.cap.x_embedder.out', 1e-5],
  'double0.txt': ['dit.cap.double0.out0', 5e-4],
  'double0.img': ['dit.cap.double0.out1', 5e-4],
  'double4.txt': ['dit.cap.double4.out0', 1e-3],
  'double4.img': ['dit.cap.double4.out1', 1e-3],
  single0: ['dit.cap.single0.out', 1e-3],
  single19: ['dit.cap.single19.out', 2e-3],
  norm_out: ['dit.cap.norm_out.out', 5e-3],
};
const capFiles: Record<string, string> = {};
for (const [cap, [fix]] of Object.entries(CAPS)) capFiles[cap] = fixFile(fix);
const runFiles = {
  pe: fixFile('te.p0.prompt_embeds'),
  noise256: fixFile('dit.256.noise'),
  noise512: fixFile('dit.512.noise'),
  np256s0: fixFile('dit.256.step0.noise_pred'),
  lat256: [0, 1, 2, 3].map((i) => fixFile(`dit.256.step${i}.latents`)),
  lat512: [0, 1, 2, 3].map((i) => fixFile(`dit.512.step${i}.latents`)),
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
      route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><meta charset=utf8><title>flux2-dit</title>' });
      return;
    }
    route.fallback(); // vite modules + /api/hf-cache weights
  });
  await page.goto(`${BASE}/__dit-parity`);

  const res: any = await page.evaluate(async ({ capFiles, runFiles }) => {
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
      log(`weights loaded in ${((Date.now() - t0) / 1000).toFixed(0)}s (${weights.mats.size} mats, ${weights.vecs.size} vecs)`);

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

      const pe = await fetchF32(runFiles.pe);
      const results: { name: string; err: number }[] = [];

      // ---- 256px: step-0 captures + noise_pred + 4-step trajectory ----
      const noise256 = await fetchF32(runFiles.noise256);
      const latRecs: { i: number; lat: Float32Array; np: Float32Array }[] = [];
      const tRun = Date.now();
      const gen256 = await pipe.generate({
        promptEmbeds: pe, noise: noise256, gridH: 16, gridW: 16,
        captureStep0: new Set(Object.keys(capFiles)),
        onLatents: (i: number, lat: Float32Array, np: Float32Array) =>
          latRecs.push({ i, lat, np }),
        onProgress: (i: number, n: number) => log(`256px step ${i + 1}/${n}...`),
      });
      log(`256px x4 steps in ${((Date.now() - tRun) / 1000).toFixed(1)}s`);

      for (const [cap, file] of Object.entries(capFiles)) {
        const got = gen256.caps.get(cap);
        if (!got) { results.push({ name: `cap ${cap} MISSING`, err: Infinity }); continue; }
        const want = await fetchF32(file as string);
        results.push({ name: `cap ${cap}`, err: relL2(got, want) });
      }
      {
        const want = await fetchF32(runFiles.np256s0);
        const np0 = latRecs[0].np;
        results.push({ name: 'noise_pred 256 step0', err: relL2(np0, want) });
        results.push({ name: 'noise_pred 256 step0 (1-cos)', err: 1 - cosSim(np0, want) });
      }
      for (const r of latRecs) {
        const want = await fetchF32(runFiles.lat256[r.i]);
        results.push({ name: `latents 256 step${r.i}`, err: relL2(r.lat, want) });
      }

      // ---- 512px trajectory ----
      const noise512 = await fetchF32(runFiles.noise512);
      const latRecs512: { i: number; lat: Float32Array }[] = [];
      const tRun2 = Date.now();
      await pipe.generate({
        promptEmbeds: pe, noise: noise512, gridH: 32, gridW: 32,
        onLatents: (i: number, lat: Float32Array) => latRecs512.push({ i, lat }),
        onProgress: (i: number, n: number) => log(`512px step ${i + 1}/${n}...`),
      });
      log(`512px x4 steps in ${((Date.now() - tRun2) / 1000).toFixed(1)}s`);
      for (const r of latRecs512) {
        const want = await fetchF32(runFiles.lat512[r.i]);
        results.push({ name: `latents 512 step${r.i}`, err: relL2(r.lat, want) });
      }

      pipe.destroy();
      weights.destroy();
      if (lost) return { error: `device lost: ${lost}`, results };
      return { results };
    } catch (e: any) {
      return { error: `${e?.message ?? e}\n${e?.stack ?? ''}` };
    }
  }, { capFiles, runFiles });

  if (res.error) {
    console.log(`FAIL: ${res.error}`);
    failed++;
  }
  if (res.results) {
    console.log('\nGPU parity:');
    for (const r of res.results) {
      let tol = 5e-3;
      if (r.name.startsWith('cap ')) tol = CAPS[r.name.slice(4)]?.[1] ?? 5e-3;
      else if (r.name.includes('(1-cos)')) tol = 1e-4;
      report(r.name, r.err, tol);
    }
  }
} finally {
  await browser.close();
}

console.log(`\n=== Phase 3 DiT parity: ${failed === 0 ? 'PASS' : `FAIL (${failed})`} ===`);
if (HEADLESS) console.log('NOTE: HEADLESS run — rerun headed on the 6700 XT before commit.');
process.exit(failed === 0 ? 0 : 1);
