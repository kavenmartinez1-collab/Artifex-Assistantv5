/**
 * Debug gate: the programmatic TE loader (src/diffusion/flux2-te.ts, used by
 * the image-gen runtime) must reproduce the Phase 2 parity numbers that the
 * full-app path achieved (valid rows relL2 <= 1e-2 vs the Python fixture).
 *
 * Run: npx tsx scripts/test-flux2-te-module.mts   (vite dev server required)
 */
import { chromium } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const fixDir = resolve(here, 'flux2_fixture');
const manifest = JSON.parse(readFileSync(resolve(fixDir, 'manifest.json'), 'utf8'));
const BASE = 'http://127.0.0.1:5173';

const prompt = manifest.meta['te.p0.prompt'] as string;
const validLen = manifest.meta['te.p0.valid_len'] as number;
const peFile = manifest.tensors['te.p0.prompt_embeds'].file as string;

// P7.3: override to gate the Q4_K_M TE (larger quant error than Q8's 1e-2):
//   TE_REPO=local/flux2-te-qwen3-4b-q4_k_m TE_GGUF=flux2-te-qwen3-4b-q4_k_m.gguf \
//   TE_TOL=5e-2 TE_TOL_ALL=1e-1 npx tsx scripts/test-flux2-te-module.mts
const TE_REPO = process.env.TE_REPO ?? 'local/flux2-te-qwen3-4b-q8_0';
const TE_GGUF = process.env.TE_GGUF ?? 'flux2-te-qwen3-4b-q8_0.gguf';
const TE_TOL = Number(process.env.TE_TOL ?? '1e-2');
const TE_TOL_ALL = Number(process.env.TE_TOL_ALL ?? '2.5e-2');

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
  await page.goto(`${BASE}/__te-module`);

  const res: any = await page.evaluate(async ({ prompt, peFile, validLen, teRepo, teGguf }) => {
    (globalThis as any).__name = (f: any) => f;
    try {
      const hub: any = await import('/src/model/hf-hub.ts');
      hub.useLocalCache();
      const te: any = await import('/src/diffusion/flux2-te.ts');
      const emb: any = await import('/src/diffusion/text-embedder.ts');

      const g = (navigator as any).gpu;
      const adapter = await g.requestAdapter({ powerPreference: 'high-performance' });
      const device = await adapter.requestDevice({
        requiredLimits: {
          maxBufferSize: adapter.limits.maxBufferSize,
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        },
      });
      console.log(`adapter: ${adapter.info?.vendor} ${adapter.info?.architecture}`);

      const t0 = Date.now();
      const enc = await te.loadFlux2TextEncoder(device, teRepo, teGguf);
      console.log(`TE loaded in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
      const out = await emb.embedFlux2Prompt(enc.engine, enc.tokenizer, prompt);
      enc.destroy();

      const want = new Float32Array(await (await fetch(`/__fixture/${peFile}`)).arrayBuffer());
      const got = out.promptEmbeds;
      const rel = (rows: number) => {
        let d = 0, n = 0;
        for (let i = 0; i < rows * 7680; i++) {
          const e = got[i] - want[i];
          d += e * e; n += want[i] * want[i];
        }
        return Math.sqrt(d / Math.max(n, 1e-30));
      };
      return { validLen: out.validLen, relValid: rel(validLen), relAll: rel(512) };
    } catch (e: any) {
      return { error: `${e?.message ?? e}\n${e?.stack ?? ''}` };
    }
  }, { prompt, peFile, validLen, teRepo: TE_REPO, teGguf: TE_GGUF });

  if (res.error) {
    console.log(`FAIL: ${res.error}`);
  } else {
    console.log(`validLen ${res.validLen} (want ${validLen})`);
    console.log(`relL2 valid rows ${res.relValid.toExponential(3)} (tol ${TE_TOL})`);
    console.log(`relL2 all rows   ${res.relAll.toExponential(3)} (tol ${TE_TOL_ALL})`);
    ok = res.validLen === validLen && res.relValid <= TE_TOL && res.relAll <= TE_TOL_ALL;
  }
} finally {
  await browser.close();
}
console.log(`\n=== TE module parity: ${ok ? 'PASS' : 'FAIL'} ===`);
process.exit(ok ? 0 : 1);
