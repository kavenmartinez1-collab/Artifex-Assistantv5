/**
 * P7.4 gate: all-resident lifecycle e2e (headed, 6700 XT).
 *
 * One device, four consecutive generations through the SHIPPED runtime
 * (src/diffusion/flux2-image.ts):
 *   1. 512px t2i  (first gen: pays the one-time resident load)
 *   2. 512px t2i  same prompt, new seed  (reroll: cached embeds, no loads)
 *   3. 512px t2i  new prompt             (TE forward only, no loads)
 *   4. 512px edit of gen 1's output      (transient encoder + resident rest)
 * then releaseFlux2Resident().
 *
 * PASS = all four complete, the 'load' progress stage appears ONLY during
 * gen 1, device is never lost, no uncaptured errors. VRAM stability across
 * generations is proven by the device surviving all four on the 12 GB card.
 * PNGs land in scripts/ for the perceptual check (Q8 DiT + Q4 TE quality).
 *
 * Run: npx tsx scripts/test-flux2-resident-e2e.mts   (vite dev server up)
 */
import { chromium } from '@playwright/test';
import { writeFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const BASE = 'http://127.0.0.1:5173';
const PX = 512;
const PROMPT_A = 'a watercolor painting of a red fox standing in fresh snow, soft morning light';
const PROMPT_B = 'a cozy wooden cabin by a mountain lake at sunset, photorealistic';
const PROMPT_EDIT = 'make it snowy and add falling snowflakes';

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
    if (route.request().resourceType() === 'document') {
      route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><meta charset=utf8>' });
      return;
    }
    route.fallback();
  });
  await page.goto(`${BASE}/__resident-e2e`);

  const res: any = await page.evaluate(async ({ PX, PROMPT_A, PROMPT_B, PROMPT_EDIT }) => {
    (globalThis as any).__name = (f: any) => f;
    const log = (s: string) => console.log(s);
    try {
      const hub: any = await import('/src/model/hf-hub.ts');
      hub.useLocalCache();
      const imgMod: any = await import('/src/diffusion/flux2-image.ts');

      const g = (navigator as any).gpu;
      const adapter = await g.requestAdapter({ powerPreference: 'high-performance' });
      // Match the shipped device (gpu-device.ts) — without the storage-buffer
      // limit the TE engine's tq-encode pipeline (9 bindings) fails to build.
      const device = await adapter.requestDevice({
        requiredLimits: {
          maxBufferSize: adapter.limits.maxBufferSize,
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
          maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
        },
      });
      let lost: string | null = null;
      let uncaptured = 0;
      device.lost.then((l: any) => { lost = `${l.reason}: ${l.message}`; });
      device.onuncapturederror = (e: any) => { uncaptured++; log(`UNCAPTURED: ${e.error?.message}`); };
      log(`adapter: ${adapter.info?.vendor} ${adapter.info?.architecture}`);

      const runs: any[] = [];
      const gen = async (name: string, prompt: string, seed: number, refImage?: any) => {
        const stages = new Set<string>();
        const t0 = Date.now();
        const r = await imgMod.generateFlux2Image(device, {
          prompt, px: PX, seed, refImage,
          onProgress: (stage: string, detail: string) => {
            stages.add(stage);
            log(`[${name}][${stage}] ${detail}`);
          },
        });
        runs.push({
          name, totalS: (Date.now() - t0) / 1000, loaded: stages.has('load'),
          teS: r.timings.teMs / 1000, ditS: r.timings.ditMs / 1000,
          vaeS: r.timings.vaeMs / 1000, encS: r.timings.encMs / 1000,
        });
        return r;
      };

      const r1 = await gen('gen1', PROMPT_A, 7);
      const r2 = await gen('gen2-reroll', PROMPT_A, 8);
      const r3 = await gen('gen3-newprompt', PROMPT_B, 9);

      // Edit: gen1's RGBA -> PNG blob -> the shipped preprocess path.
      const cv = new OffscreenCanvas(PX, PX);
      cv.getContext('2d')!.putImageData(
        new ImageData(new Uint8ClampedArray(r1.rgba), PX, PX), 0, 0);
      const blob = await cv.convertToBlob({ type: 'image/png' });
      const refData = await imgMod.preprocessRefImage(blob, PX);
      const r4 = await gen('gen4-edit', PROMPT_EDIT, 11, { data: refData, px: PX });

      imgMod.releaseFlux2Resident();

      const pack = (r: any) => {
        const hw = PX * PX;
        const rgb = new Uint8Array(hw * 3);
        for (let p = 0; p < hw; p++) {
          rgb[p * 3] = r.rgba[p * 4];
          rgb[p * 3 + 1] = r.rgba[p * 4 + 1];
          rgb[p * 3 + 2] = r.rgba[p * 4 + 2];
        }
        let bin = '';
        for (let p = 0; p < rgb.length; p += 8192) bin += String.fromCharCode(...rgb.subarray(p, p + 8192));
        return btoa(bin);
      };
      if (lost) return { error: `device lost: ${lost}` };
      return { runs, uncaptured, imgs: [pack(r1), pack(r2), pack(r3), pack(r4)] };
    } catch (e: any) {
      return { error: `${e?.message ?? e}\n${e?.stack ?? ''}` };
    }
  }, { PX, PROMPT_A, PROMPT_B, PROMPT_EDIT });

  if (res.error) {
    console.log(`FAIL: ${res.error}`);
  } else {
    const names = ['gen1', 'gen2-reroll', 'gen3-newprompt', 'gen4-edit'];
    res.imgs.forEach((b64: string, i: number) => {
      const out = resolve(here, `out-resident-${names[i]}.png`);
      writePng(out, new Uint8Array(Buffer.from(b64, 'base64')), PX, PX);
      console.log(`wrote ${out}`);
    });
    console.log('\nrun               total     TE    DiT    VAE    enc  load-stage');
    for (const r of res.runs) {
      console.log(
        `${r.name.padEnd(16)} ${r.totalS.toFixed(0).padStart(5)}s `
        + `${r.teS.toFixed(1).padStart(6)} ${r.ditS.toFixed(1).padStart(6)} `
        + `${r.vaeS.toFixed(1).padStart(6)} ${r.encS.toFixed(1).padStart(6)}  ${r.loaded}`);
    }
    const loadOnlyFirst = res.runs[0].loaded && res.runs.slice(1).every((r: any) => !r.loaded);
    console.log(`\nload stage only in gen1: ${loadOnlyFirst}`);
    console.log(`uncaptured errors: ${res.uncaptured}`);
    ok = loadOnlyFirst && res.uncaptured === 0;
  }
} finally {
  await browser.close();
}
console.log(`\n=== P7.4 resident e2e: ${ok ? 'PASS' : 'FAIL'} ===`);
process.exit(ok ? 0 : 1);
