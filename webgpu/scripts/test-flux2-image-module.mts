/**
 * Runs the SHIPPED runtime module (src/diffusion/flux2-image.ts
 * generateFlux2Image) with the exact e2e inputs (fox prompt, 256px, seed 7)
 * on a harness device. Isolation runs A/B/C proved noise + Q8 embeds + DiT +
 * VAE all condition correctly with the cat prompt, so this discriminates:
 *   - fox at 256px is a logo here too  -> prompt/resolution issue (check Python)
 *   - fox appears                      -> bug is in the app layer (device/UI)
 *
 * Run: npx tsx scripts/test-flux2-image-module.mts   (vite dev server up)
 *   GEN_PROMPT/SEED/PX env to override. (GEN_PROMPT, not PROMPT: cmd.exe
 *   injects PROMPT=$P$G into npx child processes on Windows.)
 */
import { chromium } from '@playwright/test';
import { writeFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const BASE = 'http://127.0.0.1:5173';
const PROMPT = process.env.GEN_PROMPT ?? 'a watercolor painting of a red fox standing in fresh snow, soft morning light';
const SEED = parseInt(process.env.SEED ?? '7');
const PX = parseInt(process.env.PX ?? '256');

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
  await page.goto(`${BASE}/__image-module`);

  const res: any = await page.evaluate(async ({ PROMPT, SEED, PX }) => {
    (globalThis as any).__name = (f: any) => f;
    const log = (s: string) => console.log(s);
    try {
      const hub: any = await import('/src/model/hf-hub.ts');
      hub.useLocalCache();
      const imgMod: any = await import('/src/diffusion/flux2-image.ts');

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

      const res = await imgMod.generateFlux2Image(device, {
        prompt: PROMPT, px: PX, seed: SEED,
        onProgress: (stage: string, detail: string) => log(`[${stage}] ${detail}`),
      });

      const hw = PX * PX;
      const rgb = new Uint8Array(hw * 3);
      for (let p = 0; p < hw; p++) {
        rgb[p * 3] = res.rgba[p * 4];
        rgb[p * 3 + 1] = res.rgba[p * 4 + 1];
        rgb[p * 3 + 2] = res.rgba[p * 4 + 2];
      }
      let bin = '';
      for (let p = 0; p < rgb.length; p += 8192) bin += String.fromCharCode(...rgb.subarray(p, p + 8192));
      if (lost) return { error: `device lost: ${lost}` };
      return { img: btoa(bin), timings: res.timings };
    } catch (e: any) {
      return { error: `${e?.message ?? e}\n${e?.stack ?? ''}` };
    }
  }, { PROMPT, SEED, PX });

  if (res.error) {
    console.log(`FAIL: ${res.error}`);
  } else {
    const out = resolve(here, 'out-image-module.png');
    writePng(out, new Uint8Array(Buffer.from(res.img, 'base64')), PX, PX);
    console.log(`wrote ${out}`);
    console.log(`timings: TE ${(res.timings.teMs / 1000).toFixed(0)}s DiT ${(res.timings.ditMs / 1000).toFixed(0)}s VAE ${(res.timings.vaeMs / 1000).toFixed(0)}s`);
    ok = true;
  }
} finally {
  await browser.close();
}
console.log(`\n=== image module run: ${ok ? 'DONE' : 'FAIL'} ===`);
process.exit(ok ? 0 : 1);
