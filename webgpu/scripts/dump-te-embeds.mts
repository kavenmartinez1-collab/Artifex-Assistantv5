/**
 * Debug utility: embed PROMPT through the runtime TE (src/diffusion/flux2-te)
 * and dump the [512,7680] f32 prompt_embeds + input_ids to scripts/, for
 * offline diffing against a Python f32 TE run.
 *
 * Run: GEN_PROMPT="..." OUT=fox npx tsx scripts/dump-te-embeds.mts  (vite up)
 * (GEN_PROMPT, not PROMPT: cmd.exe injects PROMPT=$P$G on Windows.)
 */
import { chromium } from '@playwright/test';
import { writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const BASE = 'http://127.0.0.1:5173';
const PROMPT = process.env.GEN_PROMPT ?? 'a watercolor painting of a red fox standing in fresh snow, soft morning light';
const OUT = process.env.OUT ?? 'fox';

const browser = await chromium.launch({ channel: 'chrome', headless: false, args: ['--enable-unsafe-webgpu'] });
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
  await page.goto(`${BASE}/__te-dump`);

  const res: any = await page.evaluate(async (prompt) => {
    (globalThis as any).__name = (f: any) => f;
    try {
      const hub: any = await import('/src/model/hf-hub.ts');
      hub.useLocalCache();
      const te: any = await import('/src/diffusion/flux2-te.ts');
      const emb: any = await import('/src/diffusion/text-embedder.ts');
      const g = (navigator as any).gpu;
      const adapter = await g.requestAdapter();
      const device = await adapter.requestDevice({
        requiredLimits: {
          maxBufferSize: adapter.limits.maxBufferSize,
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        },
      });
      const enc = await te.loadFlux2TextEncoder(device, 'local/flux2-te-qwen3-4b-q4_k_m', 'flux2-te-qwen3-4b-q4_k_m.gguf');
      const out = await emb.embedFlux2Prompt(enc.engine, enc.tokenizer, prompt);
      enc.destroy();
      const u8 = new Uint8Array(out.promptEmbeds.buffer);
      let bin = '';
      for (let p = 0; p < u8.length; p += 8192) bin += String.fromCharCode(...u8.subarray(p, p + 8192));
      return { b64: btoa(bin), ids: Array.from(out.inputIds as Int32Array), validLen: out.validLen };
    } catch (e: any) {
      return { error: `${e?.message ?? e}\n${e?.stack ?? ''}` };
    }
  }, PROMPT);

  if (res.error) {
    console.log(`FAIL: ${res.error}`);
  } else {
    writeFileSync(resolve(here, `${OUT}_embeds_webgpu.bin`), Buffer.from(res.b64, 'base64'));
    writeFileSync(resolve(here, `${OUT}_ids_webgpu.json`), JSON.stringify({ validLen: res.validLen, ids: res.ids }));
    console.log(`wrote ${OUT}_embeds_webgpu.bin (validLen ${res.validLen})`);
    ok = true;
  }
} finally {
  await browser.close();
}
process.exit(ok ? 0 : 1);
