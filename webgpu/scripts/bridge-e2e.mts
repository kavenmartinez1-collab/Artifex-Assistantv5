/**
 * Bridge E2E page host — loads a model in a fresh headed Chrome on the AMD
 * adapter and keeps the page alive while the Python-side verifier
 * (agent_bench/live_bridge_check.py) drives generations through the bridge.
 *
 * Adapted from firstload-27b.mts (adapter select / VRAM budget / load flow).
 * Exits when the verifier log contains "E2E: PASS|FAIL" or on timeout.
 *
 * Run (vite must already be on 127.0.0.1:5173, bridge on 8790):
 *   HEADED=1 npx tsx scripts/bridge-e2e.mts
 *   REPO=local/qwen3.6-27b-ud-iq2xxs-gguf   (default)
 */
import { chromium } from '@playwright/test';
import { readFileSync } from 'node:fs';

const BASE = 'http://127.0.0.1:5173';
const REPO = process.env.REPO ?? 'local/qwen3.6-27b-ud-iq2xxs-gguf';
const ADAPTER_RE = /radeon|6700|amd|rdna/i;
const VERIFIER_LOG = process.env.VERIFIER_LOG
  ?? 'C:/Artifex-Assistant-V5/agent_bench/results/logs/live_e2e.log';
const LOAD_TIMEOUT = 900_000;
const KEEPALIVE_MS = 15 * 60_000;

(async () => {
  const browser = await chromium.launch({
    channel: 'chrome',
    headless: !process.env.HEADED,
    args: ['--enable-unsafe-webgpu'],
  });
  try {
    const page = await browser.newPage();
    page.on('console', (msg) => {
      const t = msg.text();
      if (/bridge|error|fail/i.test(t)) console.log(`[page] ${t}`);
    });
    page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
    const VRAMGB = process.env.VRAMGB ?? '11.8';
    await page.addInitScript((gb) => localStorage.setItem('vramBudgetGB', gb), VRAMGB);
    await page.goto(`${BASE}/`);

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

    // Agent preset — the same sampler contract the Python side uses.
    await page.selectOption('#sampler-preset', 'agent').catch(() => {});

    console.log(`loading ${REPO} ...`);
    const t0 = Date.now();
    await page.fill('#model-repo', REPO);
    await page.click('#load-btn');
    await page.waitForFunction(() => {
      const b = document.getElementById('send-btn') as HTMLButtonElement | null;
      return b !== null && !b.disabled;
    }, undefined, { timeout: LOAD_TIMEOUT });
    console.log(`model ready in ${((Date.now() - t0) / 1000).toFixed(0)}s — ` +
                `holding page open for the bridge verifier...`);

    const deadline = Date.now() + KEEPALIVE_MS;
    let verdict = 'TIMEOUT';
    while (Date.now() < deadline) {
      let log = '';
      try { log = readFileSync(VERIFIER_LOG, 'utf-8'); } catch {}
      const m = /E2E: (PASS|FAIL)/.exec(log);
      if (m) {
        verdict = m[1];
        console.log('verifier finished: E2E:', verdict);
        break;
      }
      await new Promise(r => setTimeout(r, 5000));
    }
    if (verdict !== 'PASS') {
      // Post-mortem before the browser goes away: is the page's JS loop
      // still alive (frozen renderer = TDR/device-loss suspect)?
      try {
        const probe = await Promise.race([
          page.evaluate(() => ({
            status: document.getElementById('status')?.textContent ?? '',
            alive: true,
          })),
          new Promise((r) => setTimeout(() => r({ alive: false }), 8000)),
        ]);
        console.log('post-mortem:', JSON.stringify(probe));
        await page.screenshot({
          path: 'C:/Artifex-Assistant-V5/agent_bench/results/logs/bridge_e2e_postmortem.png',
          timeout: 8000,
        }).catch(() => console.log('post-mortem: screenshot failed (renderer frozen?)'));
      } catch (e) {
        console.log('post-mortem: page unresponsive —', String(e).slice(0, 120));
      }
    }
  } finally {
    await browser.close();
  }
})();
