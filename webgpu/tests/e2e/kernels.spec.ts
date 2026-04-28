import { test, expect, type Page } from '@playwright/test';

const KERNEL_TIMEOUT = 90_000;

async function waitForGPU(page: Page): Promise<void> {
  await page.waitForFunction(
    () => document.querySelector('#gpu-status')?.textContent?.includes('ready')
      ?? document.querySelector('#test-kernels') !== null,
    { timeout: 30_000 },
  );
}

async function getTestResults(page: Page): Promise<{
  passed: number;
  failed: number;
  results: Array<{ name: string; passed: boolean; maxDiff?: number; error?: string }>;
}> {
  return page.evaluate(() => {
    return (window as any).__kernelTestResults ?? { passed: 0, failed: 0, results: [] };
  });
}

test.describe('WebGPU Kernel Tests', () => {
  test('all GPU kernels pass validation against CPU reference', async ({ page }) => {
    await page.goto('/');
    await waitForGPU(page);

    await page.evaluate(() => {
      (window as any).__kernelTestResults = null;
    });

    const testBtn = page.locator('#test-kernels');
    await expect(testBtn).toBeVisible({ timeout: 10_000 });
    await testBtn.click();

    await page.waitForFunction(
      () => (window as any).__kernelTestResults !== null,
      { timeout: KERNEL_TIMEOUT },
    );

    const results = await getTestResults(page);
    const failedTests = results.results.filter(r => !r.passed);

    if (failedTests.length > 0) {
      const summary = failedTests
        .map(t => `  ${t.name}: ${t.error ?? `maxDiff=${t.maxDiff}`}`)
        .join('\n');
      test.fail(true, `${failedTests.length} kernel(s) failed:\n${summary}`);
    }

    expect(results.passed).toBeGreaterThan(0);
    expect(results.failed).toBe(0);
  });

  test('debug API round-trip works', async ({ page }) => {
    const payload = { test: true, ts: Date.now() };
    const postResp = await page.request.post('/api/debug', { data: payload });
    expect(postResp.ok()).toBe(true);

    const getResp = await page.request.get('/api/debug');
    expect(getResp.ok()).toBe(true);
    const body = await getResp.json();
    expect(body.test).toBe(true);
  });
});
