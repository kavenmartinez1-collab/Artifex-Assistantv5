/**
 * Lever 3: GEMV tile sweep — drives headless Chromium via Playwright against
 * the already-running vite dev server (127.0.0.1:5173, reused — never started
 * here). For each ?gemvTN/?gemvTWG config it:
 *   1. opens the app with the tile params (pipeline-override constants are
 *      read from the URL at engine creation, so each config needs a fresh load)
 *   2. selects the AMD Radeon adapter in #gpu-select (app's own discovery)
 *   3. loads the 9B abliterated GGUF from the local cache
 *   4. sends a fixed prompt with the Deterministic preset (greedy — identical
 *      tokens across configs, so tok/s is directly comparable)
 *   5. scrapes "N tokens | X tok/s | ..." from the response .meta element
 *
 * Run: npx tsx scripts/sweep-gemv-tiles.mts
 */
import { chromium, type Page } from '@playwright/test';

const BASE = 'http://127.0.0.1:5173';
const REPO = 'local/qwen3.5-9b-abliterated-gguf';
// Headless Chrome strips the marketing name — adapter shows as "rdna-2".
const ADAPTER_RE = /radeon|6700|amd|rdna/i;
const PROMPT =
  'Explain, in two detailed paragraphs, how a refrigerator keeps food cold.';

// TWG/TN must leave threads-per-row a power of two. Default (8,256) first as
// the headless baseline against the user's headed 17 tok/s.
const CONFIGS: Array<[tn: number, twg: number]> = [
  [8, 256], [4, 128], [4, 256], [8, 128], [16, 128], [16, 256],
];

const LOAD_TIMEOUT = 300_000;
const GEN_TIMEOUT = 300_000;

interface SweepResult {
  tn: number;
  twg: number;
  tokS: number;
  meta: string;
  gpu: string;
}

async function selectAmdAdapter(page: Page): Promise<string> {
  // Wait for adapter discovery to populate the dropdown.
  await page.waitForFunction(() => {
    const sel = document.getElementById('gpu-select') as HTMLSelectElement | null;
    return sel !== null && sel.options.length > 0;
  }, undefined, { timeout: 60_000 });

  const labels = await page.evaluate(() => {
    const sel = document.getElementById('gpu-select') as HTMLSelectElement;
    return Array.from(sel.options).map(o => o.textContent ?? '');
  });
  const idx = labels.findIndex(l => ADAPTER_RE.test(l));
  if (idx < 0) throw new Error(`AMD adapter not in #gpu-select: [${labels.join(' | ')}]`);

  const current = await page.evaluate(
    () => (document.getElementById('gpu-select') as HTMLSelectElement).selectedIndex);
  if (current !== idx) {
    await page.selectOption('#gpu-select', String(idx));
  }
  // initGPU() updates #f-gpu when the switch completes.
  await page.waitForFunction((re) => {
    const t = document.getElementById('f-gpu')?.textContent ?? '';
    return new RegExp(re, 'i').test(t);
  }, ADAPTER_RE.source, { timeout: 60_000 });
  return (await page.locator('#f-gpu').textContent()) ?? '?';
}

async function runConfig(tn: number, twg: number): Promise<SweepResult> {
  const browser = await chromium.launch({
    channel: 'chrome', // user's installed Chrome — no Playwright browser download
    headless: true,
    // NOTE: do NOT force Vulkan (--enable-features=Vulkan / --use-angle=vulkan)
    // — on this Windows box those flags make Chrome report zero WebGPU
    // adapters. Bare unsafe-webgpu exposes the AMD RDNA2 adapter fine.
    args: ['--enable-unsafe-webgpu'],
  });
  try {
    const page = await browser.newPage();
    page.on('console', (msg) => {
      const t = msg.text();
      if (/\[Generate\]|\[Init\]|error/i.test(t)) console.log(`    [page] ${t}`);
    });
    await page.goto(`${BASE}/?gemvTN=${tn}&gemvTWG=${twg}`);

    const gpu = await selectAmdAdapter(page);
    console.log(`    adapter: ${gpu}`);

    // Greedy sampling so every config decodes the identical token sequence.
    await page.selectOption('#sampler-preset', 'deterministic').catch(async () => {
      await page.selectOption('#sampler-preset', { label: 'Deterministic' });
    });

    await page.fill('#model-repo', REPO);
    await page.click('#load-btn');
    // send-btn flips enabled when the session exists (main.ts).
    await page.waitForFunction(() => {
      const b = document.getElementById('send-btn') as HTMLButtonElement | null;
      return b !== null && !b.disabled;
    }, undefined, { timeout: LOAD_TIMEOUT });

    const metasBefore = await page.locator('.meta').count();
    await page.fill('#prompt', PROMPT);
    await page.click('#send-btn');
    await page.waitForFunction((n) =>
      document.querySelectorAll('.meta').length > n,
    metasBefore, { timeout: GEN_TIMEOUT });

    const meta = (await page.locator('.meta').last().textContent()) ?? '';
    const m = meta.match(/([\d.]+)\s*tok\/s/);
    if (!m) throw new Error(`no tok/s in meta: "${meta}"`);
    return { tn, twg, tokS: parseFloat(m[1]), meta: meta.trim(), gpu };
  } finally {
    await browser.close();
  }
}

(async () => {
  const results: SweepResult[] = [];
  for (const [tn, twg] of CONFIGS) {
    console.log(`\n=== TN=${tn} TWG=${twg} ===`);
    try {
      const r = await runConfig(tn, twg);
      results.push(r);
      console.log(`    RESULT TN=${tn} TWG=${twg}: ${r.tokS} tok/s  (${r.meta})`);
    } catch (e) {
      console.log(`    FAILED TN=${tn} TWG=${twg}: ${(e as Error).message}`);
    }
  }

  console.log('\n──── sweep summary (RX 6700 XT, 9B i1-Q4_K_M, greedy) ────');
  console.log('TN\tTWG\ttok/s');
  for (const r of [...results].sort((a, b) => b.tokS - a.tokS)) {
    console.log(`${r.tn}\t${r.twg}\t${r.tokS}`);
  }
})();
