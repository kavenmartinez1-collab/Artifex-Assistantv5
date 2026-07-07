/**
 * Phase 6 headed e2e (P6.4): the SHIPPED edit path (flux2-image.ts
 * generateFlux2Image with refImage) end-to-end on the 6700 XT.
 *
 * Rebuilds the fixture's synthetic 256px ref (r=x grad, g=y grad,
 * b=sin(3x)cos(2y)) as a canvas Blob, runs preprocessRefImage +
 * VAE-encode + TE + DiT(+ref tokens) + VAE-decode with the p1 lighthouse
 * prompt. Runtime RNG differs from the torch fixture noise, so the gate is
 * perceptual: eyeball out-edit-e2e.png against flux2_fixture/edit_256px.png
 * (same prompt/ref through Python) — composition should be ref-conditioned.
 *
 * Run: npx tsx scripts/test-flux2-edit-e2e.mts   (vite dev server up)
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
const PROMPT = process.env.GEN_PROMPT
  ?? 'A watercolor painting of a lighthouse on a rocky coast under a stormy sky';
const SEED = parseInt(process.env.SEED ?? '46');
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
  await page.goto(`${BASE}/__edit-e2e`);

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
          // TE stage: forward-pass tq-encode uses 9 storage buffers (app
          // devices request adapter max; default 8 trips validation noise)
          maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
        },
      });
      let lost: string | null = null;
      device.lost.then((l: any) => { lost = `${l.reason}: ${l.message}`; });
      device.onuncapturederror = (e: any) => log(`UNCAPTURED: ${e.error?.message}`);
      log(`adapter: ${adapter.info?.vendor} ${adapter.info?.architecture}`);

      // Synthetic ref as a real Blob so preprocessRefImage runs too
      // (fixture formula at 256, upscaled by the preprocessor if PX > 256).
      const rc = new OffscreenCanvas(256, 256);
      const rctx = rc.getContext('2d')!;
      const id = rctx.createImageData(256, 256);
      for (let y = 0; y < 256; y++) {
        for (let x = 0; x < 256; x++) {
          const u = (x / 255) * 2 - 1, v = (y / 255) * 2 - 1;
          const i = (y * 256 + x) * 4;
          id.data[i] = (u + 1) * 127.5;
          id.data[i + 1] = (v + 1) * 127.5;
          id.data[i + 2] = (Math.sin(3 * u) * Math.cos(2 * v) + 1) * 127.5;
          id.data[i + 3] = 255;
        }
      }
      rctx.putImageData(id, 0, 0);
      const blob = await rc.convertToBlob({ type: 'image/png' });
      const refData = await imgMod.preprocessRefImage(blob, PX);
      log(`ref preprocessed: ${refData.length} floats (${PX}px)`);

      const res = await imgMod.generateFlux2Image(device, {
        prompt: PROMPT, px: PX, seed: SEED,
        refImage: { data: refData, px: PX },
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
    const out = resolve(here, `out-edit-e2e-${PX}.png`);
    writePng(out, new Uint8Array(Buffer.from(res.img, 'base64')), PX, PX);
    console.log(`wrote ${out}`);
    const t = res.timings;
    console.log(`timings: enc ${(t.encMs / 1000).toFixed(0)}s TE ${(t.teMs / 1000).toFixed(0)}s`
      + ` DiT ${(t.ditMs / 1000).toFixed(0)}s VAE ${(t.vaeMs / 1000).toFixed(0)}s`);
    ok = true;
  }
} finally {
  await browser.close();
}
console.log(`\n=== edit e2e (${PX}px): ${ok ? 'DONE' : 'FAIL'} ===`);
process.exit(ok ? 0 : 1);
