/**
 * Piper HiFiGAN decoder GPU parity gate — Phase P6.
 *
 * Feeds the onnxruntime fixture flow output z [192, F] into the WebGPU dec
 * (piper-dec-gpu.ts / piper_dec.wgsl) and compares the waveform against the
 * fixture (relL2 <= 1e-3, the same tolerance as the CPU decForward gate).
 *
 * Runs in-page: the dec shader imports through vite, so the vite dev server
 * must already be running on 127.0.0.1:5173 (start it yourself — this script
 * does not spawn servers). Weights come from the converter safetensors, routed
 * in as /__weights; fixtures as /__fixture/.
 *
 * Run: npx tsx scripts/test-piper-dec-gpu.mts      (HEADLESS=1 to opt out)
 */
import { chromium } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const fixDir = resolve(here, 'piper_fixture');
const modelPath = resolve(here, '../models/piper-en-us-joe-medium/model.safetensors');
const manifest = JSON.parse(readFileSync(resolve(fixDir, 'manifest.json'), 'utf8'));
const BASE = 'http://127.0.0.1:5173';

const TOL = 1e-3;
let failed = 0;
const report = (name: string, err: number) => {
  const ok = err <= TOL && Number.isFinite(err);
  if (!ok) failed++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'} ${name.padEnd(16)} relL2 ${err.toExponential(3)} (tol ${TOL.toExponential(0)})`);
};

const sentences = ['s0', 's1'].map((s) => ({
  s,
  ids: manifest.meta[`${s}.ids`] as number[],
  zFile: manifest.tensors[`${s}.z`].file,
  zShape: manifest.tensors[`${s}.z`].shape as number[],
  wavFile: manifest.tensors[`${s}.waveform`].file,
}));

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
    if (url.includes('/__weights')) {
      route.fulfill({ body: readFileSync(modelPath), contentType: 'application/octet-stream' });
      return;
    }
    if (route.request().resourceType() === 'document') {
      route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><meta charset=utf8><title>piper-dec</title>' });
      return;
    }
    route.fallback(); // vite modules
  });
  await page.goto(`${BASE}/__piper-dec`);

  const res: any = await page.evaluate(async ({ sentences }) => {
    (globalThis as any).__name = (f: any) => f; // tsx/esbuild keepNames shim
    const log = (s: string) => console.log(s);
    try {
      const st: any = await import('/src/model/safetensors.ts');
      const decMod: any = await import('/src/audio/piper-dec-gpu.ts');
      const piper: any = await import('/src/audio/piper.ts');

      const g = (navigator as any).gpu;
      if (!g) return { error: 'no navigator.gpu' };
      const adapter = await g.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) return { error: 'no adapter' };
      const device = await adapter.requestDevice();
      let lost: string | null = null;
      device.lost.then((l: any) => { lost = `${l.reason}: ${l.message}`; });
      device.onuncapturederror = (e: any) => log(`UNCAPTURED GPU ERROR: ${e.error?.message}`);
      const info = adapter.info ?? {};
      log(`adapter: ${info.vendor ?? '?'} ${info.architecture ?? '?'} ${info.description ?? ''}`);

      // ---- load the piper safetensors (all f32; synthesize needs the lot) ----
      const wbuf = await (await fetch('/__weights')).arrayBuffer();
      const header = st.parseHeader(wbuf);
      const weights = new Map<string, { shape: number[]; data: Float32Array }>();
      for (const [name, tinfo] of header.tensors) {
        const raw = st.extractTensorData(wbuf, tinfo, header.headerByteLength);
        weights.set(name, { shape: tinfo.shape, data: st.tensorToFloat32(raw, tinfo.dtype) });
      }
      log(`loaded ${weights.size} tensors`);

      const fetchF32 = async (f: string) =>
        new Float32Array(await (await fetch(`/__fixture/${f}`)).arrayBuffer());
      const relL2 = (got: Float32Array, want: Float32Array) => {
        if (got.length !== want.length) return NaN;
        let num = 0, den = 0;
        for (let i = 0; i < want.length; i++) {
          const dd = got[i] - want[i];
          num += dd * dd; den += want[i] * want[i];
        }
        return Math.sqrt(num) / (Math.sqrt(den) || 1);
      };

      const dec = new decMod.PiperDecGpu(device, weights);
      const results: { name: string; err: number }[] = [];

      // (a) dec-only: fixture z [192,F] -> waveform on the GPU
      for (const { s, zFile, zShape, wavFile } of sentences) {
        const z = await fetchF32(zFile);
        const F = zShape[1];
        const t = Date.now();
        const wav = await dec.forward(z, F);
        const want = await fetchF32(wavFile);
        log(`${s} dec: F=${F} samples=${wav.length} (fixture ${want.length}) in ${Date.now() - t}ms`);
        results.push({ name: `${s}.waveform`, err: relL2(wav, want) });
      }

      // (b) full public API on GPU: synthesize(ids) with the GPU decoder plugged
      // in, zero-noise ⇒ must match the fixture waveform end-to-end.
      for (const { s, ids, wavFile } of sentences) {
        const t = Date.now();
        const res = await piper.synthesize(ids, weights, {
          noiseScale: 0, noiseW: 0, lengthScale: 1,
          decode: (z: Float32Array, F: number) => dec.forward(z, F),
        });
        const want = await fetchF32(wavFile);
        log(`${s} synth: F=${res.F} samples=${res.audio.length} in ${Date.now() - t}ms`);
        results.push({ name: `${s}.synth`, err: relL2(res.audio, want) });
      }
      dec.destroy();
      if (lost) return { error: `device lost: ${lost}`, results };
      return { results };
    } catch (e: any) {
      return { error: `${e?.message ?? e}\n${e?.stack ?? ''}` };
    }
  }, { sentences });

  if (res.error) {
    console.log(`FAIL: ${res.error}`);
    failed++;
  }
  if (res.results) {
    console.log('\nGPU dec parity:');
    for (const r of res.results) report(r.name, r.err);
  }
} finally {
  await browser.close();
}

console.log(`\n=== Piper dec GPU parity: ${failed === 0 ? 'PASS' : `FAIL (${failed})`} ===`);
if (HEADLESS) console.log('NOTE: HEADLESS run — rerun headed on the 6700 XT before commit.');
process.exit(failed === 0 ? 0 : 1);
