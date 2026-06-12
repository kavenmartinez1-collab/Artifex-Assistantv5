/**
 * First load of Qwen3.5-27B Q2_K on the RX 6700 XT (headless Chrome via
 * Playwright, vite dev server must already be running on 127.0.0.1:5173).
 * Greedy (Deterministic preset) per the sampler-before-kernel rule, so any
 * incoherence points at kernels/quant, not sampling. Q2_K/Q3_K currently run
 * on the LEGACY GEMV kernels (tiled covers only Q4_K/Q5_K/Q6_K) — expect
 * slower than the 9B until lever 3.5.
 *
 * Run: npx tsx scripts/firstload-27b.mts
 */
import { chromium } from '@playwright/test';

const BASE = 'http://127.0.0.1:5173';
const REPO = 'local/qwen3.5-27b-gguf';
const ADAPTER_RE = /radeon|6700|amd|rdna/i;
const PROMPT =
  'Explain, in two detailed paragraphs, how a refrigerator keeps food cold.';
const LOAD_TIMEOUT = 900_000; // 10.1 GB off disk + upload — be generous
const GEN_TIMEOUT = 900_000;

(async () => {
  const browser = await chromium.launch({
    channel: 'chrome',
    headless: true,
    // Vulkan-forcing flags break adapter discovery on this box — bare flag only.
    args: ['--enable-unsafe-webgpu'],
  });
  try {
    const page = await browser.newPage();
    // First load: log everything useful, not just Generate/Init.
    page.on('console', (msg) => {
      const t = msg.text();
      if (/\[|error|warn|fail/i.test(t)) console.log(`[page] ${t}`);
    });
    page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
    // /api/gpu-info is nvidia-smi-only, so the AMD card gets the loader's
    // 6.5 GB default — too small for the 27B (~10.4 GB). Manual override:
    // 11 GB on the 12 GB card leaves ~1 GB for the compositor.
    await page.addInitScript(() => localStorage.setItem('vramBudgetGB', '11'));
    await page.goto(`${BASE}/`);

    // Select the AMD adapter (headless label is "rdna-2").
    await page.waitForFunction(() => {
      const sel = document.getElementById('gpu-select') as HTMLSelectElement | null;
      return sel !== null && sel.options.length > 0;
    }, undefined, { timeout: 60_000 });
    const labels = await page.evaluate(() => {
      const sel = document.getElementById('gpu-select') as HTMLSelectElement;
      return Array.from(sel.options).map(o => o.textContent ?? '');
    });
    const idx = labels.findIndex(l => ADAPTER_RE.test(l));
    if (idx < 0) throw new Error(`AMD adapter not found: [${labels.join(' | ')}]`);
    const current = await page.evaluate(
      () => (document.getElementById('gpu-select') as HTMLSelectElement).selectedIndex);
    if (current !== idx) await page.selectOption('#gpu-select', String(idx));
    await page.waitForFunction((re) => {
      const t = document.getElementById('f-gpu')?.textContent ?? '';
      return new RegExp(re, 'i').test(t);
    }, ADAPTER_RE.source, { timeout: 60_000 });
    console.log(`adapter: ${await page.locator('#f-gpu').textContent()}`);

    await page.selectOption('#sampler-preset', 'deterministic').catch(async () => {
      await page.selectOption('#sampler-preset', { label: 'Deterministic' });
    });

    console.log(`loading ${REPO} ...`);
    const t0 = Date.now();
    await page.fill('#model-repo', REPO);
    await page.click('#load-btn');
    await page.waitForFunction(() => {
      const b = document.getElementById('send-btn') as HTMLButtonElement | null;
      return b !== null && !b.disabled;
    }, undefined, { timeout: LOAD_TIMEOUT });
    console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(0)}s`);

    const metasBefore = await page.locator('.meta').count();
    await page.fill('#prompt', PROMPT);
    await page.click('#send-btn');
    await page.waitForFunction((n) =>
      document.querySelectorAll('.meta').length > n,
    metasBefore, { timeout: GEN_TIMEOUT });

    // .meta is appended inside the response div — parent holds the text.
    const { meta, text } = await page.evaluate(() => {
      const metas = document.querySelectorAll('.meta');
      const m = metas[metas.length - 1] as HTMLElement;
      const parent = m.parentElement as HTMLElement;
      const t = (parent.textContent ?? '').replace(m.textContent ?? '', '');
      return { meta: m.textContent ?? '', text: t };
    });
    console.log(`\n──── meta ────\n${meta}`);
    console.log(`\n──── response (greedy) ────\n${text}`);
  } finally {
    await browser.close();
  }
})();
