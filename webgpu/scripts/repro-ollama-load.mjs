/**
 * Repro: load ollama/gemma4:latest in a real browser and capture every
 * network request + console line until the failure (or 45s of loading).
 * Usage: node scripts/repro-ollama-load.mjs
 */
import { chromium } from '@playwright/test';

const browser = await chromium.launch({
  headless: true,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan'],
});
const page = await browser.newPage();

const lines = [];
page.on('console', (msg) => lines.push(`[console.${msg.type()}] ${msg.text()}`));
page.on('response', (resp) => {
  const status = resp.status();
  if (status >= 400 || resp.url().includes('huggingface.co')) {
    lines.push(`[net ${status}] ${resp.url().slice(0, 160)}`);
  }
});
page.on('requestfailed', (req) => lines.push(`[net FAILED] ${req.url().slice(0, 160)} — ${req.failure()?.errorText}`));

await page.goto('http://localhost:5173/');
await page.waitForTimeout(3000);

await page.fill('#model-repo', 'ollama/gemma4:latest');
await page.click('#load-btn');

// Wait until a failure message appears or 45s pass
const deadline = Date.now() + 45_000;
let failureText = null;
while (Date.now() < deadline) {
  failureText = await page.evaluate(() => {
    const msgs = [...document.querySelectorAll('.message.system')].map(m => m.textContent ?? '');
    return msgs.find(t => t.includes('Failed to load')) ?? null;
  });
  if (failureText) break;
  await page.waitForTimeout(1500);
}

console.log('=== UI failure message ===');
console.log(failureText ?? '(none within 45s — load may be progressing)');
console.log('=== last 45 captured lines ===');
for (const l of lines.slice(-45)) console.log(l);

await browser.close();
