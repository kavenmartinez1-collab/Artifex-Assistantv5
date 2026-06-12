/**
 * Diagnostic: find Chrome flag combo that exposes WebGPU adapters in
 * headless mode on this machine. Tries several flag sets, prints adapter
 * info for both powerPreferences. Scratch tool — delete after lever 3.
 */
import { chromium } from '@playwright/test';

const COMBOS: Array<[name: string, args: string[]]> = [
  ['vulkan (current sweep flags)', [
    '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan',
  ]],
  ['bare', ['--enable-unsafe-webgpu']],
  ['d3d11', ['--enable-unsafe-webgpu', '--use-angle=d3d11']],
  ['gpu-forced', [
    '--enable-unsafe-webgpu', '--enable-gpu', '--disable-gpu-sandbox',
    '--ignore-gpu-blocklist',
  ]],
];

async function probe(name: string, args: string[], headless: boolean) {
  const browser = await chromium.launch({ channel: 'chrome', headless, args });
  try {
    const page = await browser.newPage();
    // navigator.gpu only exists in secure contexts — about:blank is not one
    await page.goto('http://127.0.0.1:5173/?gemvProbe=1');
    const result = await page.evaluate(async () => {
      if (!('gpu' in navigator)) return 'no navigator.gpu';
      const out: string[] = [];
      for (const pp of ['low-power', 'high-performance'] as const) {
        const a = await navigator.gpu.requestAdapter({ powerPreference: pp });
        out.push(`${pp}: ${a ? `${a.info?.vendor ?? '?'} ${a.info?.description || a.info?.architecture || ''}`.trim() : 'null'}`);
      }
      return out.join(' | ');
    });
    console.log(`[${headless ? 'headless' : 'headed'}] ${name}: ${result}`);
  } catch (e) {
    console.log(`[${headless ? 'headless' : 'headed'}] ${name}: ERROR ${(e as Error).message.split('\n')[0]}`);
  } finally {
    await browser.close();
  }
}

(async () => {
  for (const [name, args] of COMBOS) await probe(name, args, true);
  // headed fallback with current flags, to confirm the card is reachable at all
  await probe('vulkan (current sweep flags)', COMBOS[0][1], false);
})();
