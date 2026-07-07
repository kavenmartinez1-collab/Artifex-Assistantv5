/**
 * Phase 4 end-to-end UI gate: FLUX.2 klein text-to-image through the real app.
 *
 * Drives the shipped UI exactly as a user would (vite dev server must be
 * running): select the AMD adapter, load local/flux.2-klein-4b (routes into
 * image-gen mode via the model_index.json probe), send a prompt with a pinned
 * seed, and wait for the generated <img class="md-image"> to appear. The PNG
 * is written to scripts/out-e2e-flux2.png for visual inspection.
 *
 * Run: npx tsx scripts/test-flux2-e2e-ui.mts          (headed, 6700 XT)
 *   PX=512 SEED=7 GEN_PROMPT="..." to override the defaults.
 *   (GEN_PROMPT, not PROMPT: on Windows, cmd.exe — which npx.cmd runs
 *   through — injects PROMPT=$P$G into the child env, silently replacing
 *   any default. The model dutifully painted a P&G logo.)
 */
import { chromium } from '@playwright/test';
import { writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const BASE = 'http://127.0.0.1:5173';
const ADAPTER_RE = /radeon|6700|amd|rdna/i;
const PX = process.env.PX ?? '256';
const SEED = process.env.SEED ?? '7';
const PROMPT = process.env.GEN_PROMPT ?? 'a watercolor painting of a red fox standing in fresh snow, soft morning light';
const GEN_TIMEOUT = 15 * 60_000;

try {
  await fetch(BASE, { signal: AbortSignal.timeout(3000) });
} catch {
  console.error(`FATAL: vite dev server not reachable at ${BASE} — start it first.`);
  process.exit(1);
}

const browser = await chromium.launch({
  channel: 'chrome',
  headless: false, // e2e VRAM gate — headed is the point
  args: ['--enable-unsafe-webgpu'],
});
let ok = false;
try {
  const page = await browser.newPage();
  page.on('console', (msg) => {
    const t = msg.text();
    if (/error|fail|\[Flux2|\[GGUF|\[HF Hub\]|UNCAPTURED/i.test(t)) console.log(`  [page] ${t}`);
  });
  page.on('pageerror', (e) => console.log(`  [pageerror] ${e.message}`));
  await page.addInitScript((gb) => localStorage.setItem('vramBudgetGB', gb), process.env.VRAMGB ?? '11.8');
  await page.goto(`${BASE}/`);

  // AMD adapter
  await page.waitForFunction(() => {
    const sel = document.getElementById('gpu-select') as HTMLSelectElement | null;
    return sel !== null && sel.options.length > 0;
  }, undefined, { timeout: 60_000 });
  const labels = await page.evaluate(() => Array.from(
    (document.getElementById('gpu-select') as HTMLSelectElement).options).map(o => o.textContent ?? ''));
  const idx = labels.findIndex(l => ADAPTER_RE.test(l));
  if (idx < 0) throw new Error(`AMD adapter not found: [${labels.join(' | ')}]`);
  const cur = await page.evaluate(() => (document.getElementById('gpu-select') as HTMLSelectElement).selectedIndex);
  if (cur !== idx) await page.selectOption('#gpu-select', String(idx));
  await page.waitForFunction((re) => new RegExp(re, 'i').test(
    document.getElementById('f-gpu')?.textContent ?? ''), ADAPTER_RE.source, { timeout: 60_000 });
  console.log(`adapter: ${await page.locator('#f-gpu').textContent()}`);

  // Load the image-gen "model" (routes via model_index.json probe)
  await page.fill('#model-repo', 'local/flux.2-klein-4b');
  await page.click('#load-btn');
  await page.waitForFunction(() =>
    (document.getElementById('status')?.textContent ?? '').includes('image gen'),
    undefined, { timeout: 60_000 });
  console.log('image-gen mode ready');

  // Send the prompt
  const promptText = `/${PX} /seed ${SEED} ${PROMPT}`;
  console.log(`prompt: ${promptText}`);
  await page.fill('#prompt', promptText);
  const t0 = Date.now();
  await page.click('#send-btn');

  // Live progress: poll the assistant bubble text until the image lands.
  const poll = setInterval(async () => {
    try {
      const s = await page.evaluate(() =>
        document.getElementById('status')?.textContent ?? '');
      process.stdout.write(`\r  ${s.padEnd(90).slice(0, 90)}`);
    } catch { /* page busy */ }
  }, 2000);
  try {
    await page.waitForSelector('img.md-image', { timeout: GEN_TIMEOUT });
  } finally {
    clearInterval(poll);
    process.stdout.write('\n');
  }
  const genS = ((Date.now() - t0) / 1000).toFixed(0);

  const info = await page.evaluate(() => {
    const img = document.querySelector('img.md-image') as HTMLImageElement;
    return { src: img.src, w: img.naturalWidth, h: img.naturalHeight };
  });
  if (info.w !== parseInt(PX) || info.h !== parseInt(PX)) {
    throw new Error(`image is ${info.w}x${info.h}, expected ${PX}x${PX}`);
  }
  const b64 = info.src.split('base64,')[1];
  const out = resolve(here, 'out-e2e-flux2.png');
  writeFileSync(out, Buffer.from(b64, 'base64'));
  console.log(`PASS: ${info.w}x${info.h} image generated in ${genS}s -> ${out}`);

  // Reroll check: same prompt, new seed — must reuse the cached embedding.
  ok = true;
} finally {
  await browser.close();
}
console.log(`\n=== Phase 4 e2e UI: ${ok ? 'PASS' : 'FAIL'} ===`);
process.exit(ok ? 0 : 1);
