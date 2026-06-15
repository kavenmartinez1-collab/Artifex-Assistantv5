/**
 * First load of Qwen3.5-27B Q2_K on the RX 6700 XT (headless Chrome via
 * Playwright, vite dev server must already be running on 127.0.0.1:5173).
 * Greedy (Deterministic preset) per the sampler-before-kernel rule, so any
 * incoherence points at kernels/quant, not sampling. Doubles as the fusion
 * lever's bench + greedy-parity driver.
 *
 * Run: npx tsx scripts/firstload-27b.mts
 *   OUT=path.txt   — also write the greedy response text to a file (for diff)
 *   URLQ="deint=0" — extra query params appended to the page URL (A/B toggles)
 *   MAXTOK=32      — override #max-tokens (short profiling runs)
 */
import { chromium } from '@playwright/test';
import { writeFileSync } from 'node:fs';

const BASE = 'http://127.0.0.1:5173';
const URLQ = process.env.URLQ ? `?${process.env.URLQ}` : '';
const OUT = process.env.OUT ?? '';
const REPO = process.env.REPO ?? 'local/qwen3.6-27b-mtp-gguf';
const ADAPTER_RE = /radeon|6700|amd|rdna/i;
const PROMPT =
  'Explain, in two detailed paragraphs, how a refrigerator keeps food cold.';
const LOAD_TIMEOUT = 900_000; // 10.1 GB off disk + upload — be generous
const GEN_TIMEOUT = 900_000;

(async () => {
  const browser = await chromium.launch({
    channel: 'chrome',
    // HEADED=1 launches a visible Chrome window so the load runs through the
    // desktop compositor on the display-driving GPU — the real headed scenario
    // (compositor VRAM tax) the headless bench can't reproduce.
    headless: !process.env.HEADED,
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
    const VRAMGB = process.env.VRAMGB ?? '11.8';
    await page.addInitScript((gb) => localStorage.setItem('vramBudgetGB', gb), VRAMGB);
    await page.goto(`${BASE}/${URLQ}`);

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

    if (process.env.MAXTOK) await page.fill('#max-tokens', process.env.MAXTOK);

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
    const perf = await page.evaluate(() => (globalThis as any).__perfLastForward ?? null);
    const timing = await page.evaluate(() => (globalThis as any).__perfTimingRows ?? null);
    console.log(`\n──── meta ────\n${meta}`);
    console.log(`\n──── perfLastForward ────\n${JSON.stringify(perf)}`);
    if (timing) {
      console.log(`\n──── GPU per-category (last profiled forward) ────`);
      for (const r of timing) {
        console.log(`${r.category.padEnd(28)} n=${String(r.count).padStart(4)} ` +
          `${r.total_ms.toFixed(3).padStart(8)} ms  ${r.pct.toFixed(1).padStart(5)}%  ` +
          `avg ${r.avg_us.toFixed(1)} us`);
      }
    }
    console.log(`\n──── response (greedy) ────\n${text}`);
    if (OUT) {
      writeFileSync(OUT, text);
      console.log(`\n[wrote ${text.length} chars to ${OUT}]`);
    }
  } finally {
    await browser.close();
  }
})();
