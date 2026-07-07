/**
 * Isolation harness for the e2e conditioning bug: the Python 256px reference
 * (vae.e2e256.pixels) is a proper prompt-following image, but the runtime
 * e2e produced an unconditioned logo. Every stage is individually
 * parity-proven, so the bug must be in one of the two runtime-only inputs.
 *
 * One DiT load, three 256px runs (prompt = fixture p0, "cat on windowsill"):
 *   A) fixture prompt_embeds + fixture noise  -> sanity (must match step3
 *      latents AND decode to the cat; if not, harness/device drift)
 *   B) fixture prompt_embeds + rng.randn(seed 7) -> tests OUR NOISE
 *   C) runtime Q8-TE embeds  + fixture noise  -> tests Q8 EMBEDS through DiT
 *
 * All three are VAE-decoded; PNGs land in scripts/out-iso-{a,b,c}.png.
 * Whichever of B/C loses the cat is the culprit.
 *
 * Run: npx tsx scripts/test-flux2-runtime-isolation.mts   (vite dev server up)
 */
import { chromium } from '@playwright/test';
import { readFileSync, writeFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const fixDir = resolve(here, 'flux2_fixture');
const manifest = JSON.parse(readFileSync(resolve(fixDir, 'manifest.json'), 'utf8'));
const BASE = 'http://127.0.0.1:5173';

const fixFile = (n: string) => {
  const t = manifest.tensors[n];
  if (!t) throw new Error(`fixture tensor missing: ${n}`);
  return t.file as string;
};
const files = {
  pe: fixFile('te.p0.prompt_embeds'),
  noise: fixFile('dit.256.noise'),
  lat3: fixFile('dit.256.step3.latents'),
};
const prompt = manifest.meta['te.p0.prompt'] as string;

// Minimal PNG writer (RGB8).
function writePng(path: string, rgb: Uint8Array, w: number, h: number) {
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (w * 3 + 1)] = 0;
    rgb.subarray(y * w * 3, (y + 1) * w * 3).forEach((v, i) => { raw[y * (w * 3 + 1) + 1 + i] = v; });
  }
  const crcT: number[] = [];
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; crcT.push(c >>> 0); }
  const crc = (b: Buffer) => { let c = 0xffffffff; for (const x of b) c = crcT[(c ^ x) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
  const chunk = (type: string, data: Buffer) => {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type), data]);
    const c = Buffer.alloc(4); c.writeUInt32BE(crc(td));
    return Buffer.concat([len, td, c]);
  };
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2;
  writeFileSync(path, Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0)),
  ]));
}

const browser = await chromium.launch({
  channel: 'chrome', headless: false, args: ['--enable-unsafe-webgpu'],
});
let ok = false;
try {
  const page = await browser.newPage();
  page.on('console', (m) => console.log(`  [page] ${m.text()}`));
  page.on('pageerror', (e) => console.log(`  [pageerror] ${e.message}`));
  await page.route('**/*', (route) => {
    const url = route.request().url();
    if (url.includes('/__fixture/')) {
      const f = url.split('/__fixture/')[1];
      try {
        route.fulfill({ body: readFileSync(resolve(fixDir, f)), contentType: 'application/octet-stream' });
      } catch (e) { route.fulfill({ status: 404, body: String(e) }); }
      return;
    }
    if (route.request().resourceType() === 'document') {
      route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><meta charset=utf8>' });
      return;
    }
    route.fallback();
  });
  await page.goto(`${BASE}/__runtime-iso`);

  const res: any = await page.evaluate(async ({ files, prompt }) => {
    (globalThis as any).__name = (f: any) => f;
    const log = (s: string) => console.log(s);
    try {
      const hub: any = await import('/src/model/hf-hub.ts');
      hub.useLocalCache();
      const teMod: any = await import('/src/diffusion/flux2-te.ts');
      const embMod: any = await import('/src/diffusion/text-embedder.ts');
      const loaderMod: any = await import('/src/diffusion/flux2-loader.ts');
      const pipeMod: any = await import('/src/diffusion/flux2-pipeline.ts');
      const vaeMod: any = await import('/src/diffusion/vae.ts');
      const rngMod: any = await import('/src/diffusion/rng.ts');

      const g = (navigator as any).gpu;
      const adapter = await g.requestAdapter({ powerPreference: 'high-performance' });
      const device = await adapter.requestDevice({
        requiredLimits: {
          maxBufferSize: adapter.limits.maxBufferSize,
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        },
      });
      let lost: string | null = null;
      device.lost.then((l: any) => { lost = `${l.reason}: ${l.message}`; });
      device.onuncapturederror = (e: any) => log(`UNCAPTURED: ${e.error?.message}`);
      log(`adapter: ${adapter.info?.vendor} ${adapter.info?.architecture}`);

      const fetchF32 = async (f: string) =>
        new Float32Array(await (await fetch(`/__fixture/${f}`)).arrayBuffer());
      const relL2 = (a: Float32Array, b: Float32Array) => {
        let n = 0, d = 0;
        for (let i = 0; i < b.length; i++) { const e = a[i] - b[i]; n += e * e; d += b[i] * b[i]; }
        return Math.sqrt(n / Math.max(d, 1e-30));
      };

      // ---- runtime Q8 TE embeds (load, embed, free BEFORE the DiT) ----
      let t0 = Date.now();
      const te = await teMod.loadFlux2TextEncoder(device, 'local/flux2-te-qwen3-4b-q8_0', 'flux2-te-qwen3-4b-q8_0.gguf');
      const emb = await embMod.embedFlux2Prompt(te.engine, te.tokenizer, prompt);
      te.destroy();
      log(`TE embeds ready in ${((Date.now() - t0) / 1000).toFixed(0)}s (validLen ${emb.validLen})`);

      const peFix = await fetchF32(files.pe);
      const noiseFix = await fetchF32(files.noise);
      const lat3Fix = await fetchF32(files.lat3);
      log(`Q8 vs fixture embeds relL2: ${relL2(emb.promptEmbeds, peFix).toExponential(3)}`);

      // ---- DiT: three 256px runs ----
      t0 = Date.now();
      const url = hub.resolveFileUrl('local/flux.2-klein-4b', 'transformer/diffusion_pytorch_model.safetensors');
      const weights = await loaderMod.loadFlux2Dit(device, url);
      log(`DiT loaded in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
      const pipe = new pipeMod.Flux2Pipeline(device, weights);

      const runs: { name: string; pe: Float32Array; noise: Float32Array }[] = [
        { name: 'A fixture-pe + fixture-noise', pe: peFix, noise: noiseFix },
        { name: 'B fixture-pe + rng-noise(7)', pe: peFix, noise: rngMod.randn(256 * 128, 7) },
        { name: 'C q8-pe + fixture-noise', pe: emb.promptEmbeds, noise: noiseFix },
      ];
      const lats: Float32Array[] = [];
      const errs: Record<string, number> = {};
      for (const r of runs) {
        t0 = Date.now();
        const g256 = await pipe.generate({
          promptEmbeds: r.pe, noise: r.noise, gridH: 16, gridW: 16, numSteps: 4,
          onProgress: (i: number, n: number) => log(`${r.name}: step ${i + 1}/${n}`),
        });
        lats.push(g256.latents);
        errs[r.name] = relL2(g256.latents, lat3Fix);
        log(`${r.name}: ${(Date.now() - t0) / 1000 | 0}s, latents relL2 vs fixture step3 = ${errs[r.name].toExponential(3)}`);
      }
      pipe.destroy();
      weights.destroy();

      // ---- VAE decode all three ----
      const vaeUrl = hub.resolveFileUrl('local/flux.2-klein-4b', 'vae/diffusion_pytorch_model.safetensors');
      const vw = await vaeMod.loadFlux2Vae(device, vaeUrl);
      const dec = new vaeMod.Flux2VaeDecoder(device, vw);
      const imgs: string[] = [];
      for (let i = 0; i < lats.length; i++) {
        const unpacked = vaeMod.unpackLatents(lats[i], 16, 16, vw.bnMean, vw.bnVar);
        const px = await dec.decode(unpacked, 32);
        const hw = 256 * 256;
        const rgb = new Uint8Array(hw * 3);
        for (let p = 0; p < hw; p++) {
          rgb[p * 3] = Math.max(0, Math.min(255, (px[p] + 1) * 127.5));
          rgb[p * 3 + 1] = Math.max(0, Math.min(255, (px[hw + p] + 1) * 127.5));
          rgb[p * 3 + 2] = Math.max(0, Math.min(255, (px[2 * hw + p] + 1) * 127.5));
        }
        let bin = '';
        for (let p = 0; p < rgb.length; p += 8192) bin += String.fromCharCode(...rgb.subarray(p, p + 8192));
        imgs.push(btoa(bin));
        log(`decoded run ${i}`);
      }
      dec.destroy();
      vw.destroy();

      if (lost) return { error: `device lost: ${lost}` };
      return { errs, imgs, q8rel: relL2(emb.promptEmbeds, peFix) };
    } catch (e: any) {
      return { error: `${e?.message ?? e}\n${e?.stack ?? ''}` };
    }
  }, { files, prompt });

  if (res.error) {
    console.log(`FAIL: ${res.error}`);
  } else {
    const names = ['a', 'b', 'c'];
    res.imgs.forEach((b64: string, i: number) => {
      const out = resolve(here, `out-iso-${names[i]}.png`);
      writePng(out, new Uint8Array(Buffer.from(b64, 'base64')), 256, 256);
      console.log(`wrote ${out}`);
    });
    console.log(`\nlatents relL2 vs fixture step3:`);
    for (const [k, v] of Object.entries(res.errs)) console.log(`  ${k}: ${(v as number).toExponential(3)}`);
    ok = true;
  }
} finally {
  await browser.close();
}
console.log(`\n=== runtime isolation: ${ok ? 'DONE' : 'FAIL'} ===`);
process.exit(ok ? 0 : 1);
