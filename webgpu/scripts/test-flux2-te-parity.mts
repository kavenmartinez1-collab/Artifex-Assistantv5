/**
 * FLUX.2 Phase 2 gate: text-encoder parity vs the Python fixture.
 *
 * Loads the klein TE (default: the shipped Q4_K_M — the Q8_0 it originally
 * gated was retired in Phase 7; Q8-era numbers kept below for reference)
 * through the real app
 * (vite dev server must already be running on 127.0.0.1:5173), drives the
 * __flux2Embed hook for both fixture prompts, and gates:
 *   - input_ids  EXACT match vs te.pX.input_ids (tokenizer + template + pad)
 *   - prompt_embeds VALID rows rel-L2 <= 1e-2 — any mask/RoPE/tap bug shows
 *     here at O(1); measured Q8 noise is ~4.6e-3
 *   - hidden taps 9/18/27 + full-512-row prompt_embeds rel-L2 <= 2.5e-2 —
 *     pad rows accumulate Q8 noise faster (measured ~2x per 9 layers:
 *     4.8e-3 @h9 -> 1.0e-2 @h18 -> 2.1e-2 @h27, consistent across prompts;
 *     a pad-semantics bug would instead be O(1) already at h9 since pads
 *     are 488/512 rows)
 *
 * Fixture tensors are served into the page via route interception so the
 * 15.7 MB embeddings never cross the evaluate bridge — only metrics return.
 *
 * Run: npx tsx scripts/test-flux2-te-parity.mts   (HEADED=1 for the real gate)
 */
import { chromium } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const fixDir = resolve(here, 'flux2_fixture');
const manifest = JSON.parse(readFileSync(resolve(fixDir, 'manifest.json'), 'utf8'));

const BASE = 'http://127.0.0.1:5173';
const REPO = process.env.REPO ?? 'local/flux2-te-qwen3-4b-q4_k_m';
const ADAPTER_RE = /radeon|6700|amd|rdna/i;
const LOAD_TIMEOUT = 600_000;
// Q4_K_M tolerances (P7.3 measured valid 2.6e-2 / all 7.6e-2);
// the Q8-era gates were 1e-2 / 2.5e-2.
const REL_TOL_FULL = 1e-1;    // full 512 rows incl. pad-row quant accumulation
const REL_TOL_VALID = 5e-2;   // valid rows only — the bug-sensitive gate

function fixInfo(name: string): { file: string; shape: number[]; dtype: string } {
  const t = manifest.tensors[name];
  if (!t) throw new Error(`fixture tensor missing: ${name}`);
  return t;
}
function loadI32(name: string): Int32Array {
  const t = fixInfo(name);
  const raw = readFileSync(resolve(fixDir, t.file));
  const n = t.shape.reduce((a: number, b: number) => a * b, 1);
  const arr = new Int32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  if (arr.length !== n) throw new Error(`${name}: ${arr.length} elems, want ${n}`);
  return arr;
}

(async () => {
  // Server-up probe first — never start a server from here.
  try {
    await fetch(BASE, { signal: AbortSignal.timeout(3000) });
  } catch {
    console.error(`FATAL: vite dev server not reachable at ${BASE} — start it first (npm run dev).`);
    process.exit(1);
  }

  const browser = await chromium.launch({
    channel: 'chrome',
    headless: !process.env.HEADED,
    args: ['--enable-unsafe-webgpu'],
  });
  let failed = 0;
  try {
    const page = await browser.newPage();
    page.on('console', (msg) => {
      const t = msg.text();
      if (/\[|error|warn|fail/i.test(t)) console.log(`[page] ${t}`);
    });
    page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));

    // Serve fixture bins into the page (vite doesn't know about scripts/).
    await page.route(`${BASE}/__fixture/*`, (route) => {
      const file = route.request().url().split('/__fixture/')[1];
      try {
        route.fulfill({ body: readFileSync(resolve(fixDir, file)), contentType: 'application/octet-stream' });
      } catch (e) {
        route.fulfill({ status: 404, body: String(e) });
      }
    });

    const VRAMGB = process.env.VRAMGB ?? '11.8';
    await page.addInitScript((gb) => localStorage.setItem('vramBudgetGB', gb), VRAMGB);
    await page.goto(`${BASE}/`);

    // Select the AMD adapter.
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

    console.log(`loading ${REPO} ...`);
    const t0 = Date.now();
    await page.fill('#model-repo', REPO);
    await page.click('#load-btn');
    await page.waitForFunction(() => {
      const b = document.getElementById('send-btn') as HTMLButtonElement | null;
      return b !== null && !b.disabled;
    }, undefined, { timeout: LOAD_TIMEOUT });
    console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(0)}s`);

    for (const px of ['p0', 'p1']) {
      const prompt = manifest.meta[`te.${px}.prompt`] as string;
      const wantValid = manifest.meta[`te.${px}.valid_len`] as number;
      const wantIds = loadI32(`te.${px}.input_ids`);
      const files = {
        h9: fixInfo(`te.${px}.hidden9`).file,
        h18: fixInfo(`te.${px}.hidden18`).file,
        h27: fixInfo(`te.${px}.hidden27`).file,
        pe: fixInfo(`te.${px}.prompt_embeds`).file,
      };

      console.log(`\n── ${px}: "${prompt}"`);
      const t1 = Date.now();
      const res = await page.evaluate(async ({ prompt, files, validLen }) => {
        (globalThis as any).__name = (f: any) => f; // tsx/esbuild keepNames shim
        const out = await (globalThis as any).__flux2Embed(prompt);
        const got: Float32Array = out.promptEmbeds; // [512, 7680] = per token [hs9|hs18|hs27]
        const S = 512, H = 2560;

        const fetchF32 = async (f: string) => new Float32Array(
          await (await fetch(`/__fixture/${f}`)).arrayBuffer());
        const want = {
          h9: await fetchF32(files.h9), h18: await fetchF32(files.h18),
          h27: await fetchF32(files.h27), pe: await fetchF32(files.pe),
        };

        // rel-L2 of a column band [c0, c0+w) of got vs a [512, w] reference,
        // over rows [0, rows).
        const relBand = (ref: Float32Array, c0: number, w: number, rows: number) => {
          let d = 0, n = 0;
          for (let s = 0; s < rows; s++) {
            for (let c = 0; c < w; c++) {
              const g = got[s * 3 * H + c0 + c], r = ref[s * w + c];
              d += (g - r) * (g - r); n += r * r;
            }
          }
          return Math.sqrt(d / Math.max(n, 1e-30));
        };
        // prompt_embeds is the same layout — full-width band.
        const relPE = (rows: number) => {
          let d = 0, n = 0;
          for (let i = 0; i < rows * 3 * H; i++) {
            const g = got[i], r = want.pe[i];
            d += (g - r) * (g - r); n += r * r;
          }
          return Math.sqrt(d / Math.max(n, 1e-30));
        };

        return {
          validLen: out.validLen,
          inputIds: Array.from(out.inputIds as Int32Array),
          rel9: relBand(want.h9, 0, H, S),
          rel18: relBand(want.h18, H, H, S),
          rel27: relBand(want.h27, 2 * H, H, S),
          relAll: relPE(S),
          relValid: relPE(validLen),
        };
      }, { prompt, files, validLen: wantValid });
      console.log(`   embed in ${((Date.now() - t1) / 1000).toFixed(1)}s`);

      // Gate 1: input_ids exact.
      let idErr = -1;
      for (let i = 0; i < 512; i++) {
        if (res.inputIds[i] !== wantIds[i]) { idErr = i; break; }
      }
      const idsOk = idErr < 0 && res.validLen === wantValid;
      if (!idsOk) {
        failed++;
        console.log(`   FAIL input_ids: validLen ${res.validLen} vs ${wantValid}` +
          (idErr >= 0 ? `; first diff @${idErr}: got ${res.inputIds[idErr]} want ${wantIds[idErr]}` : ''));
      } else {
        console.log(`   PASS input_ids exact (validLen ${res.validLen})`);
      }

      // Gate 2: hidden taps + prompt_embeds rel-L2.
      for (const [k, v] of Object.entries({
        hidden9: res.rel9, hidden18: res.rel18, hidden27: res.rel27,
        prompt_embeds: res.relAll,
      })) {
        const ok = v <= REL_TOL_FULL;
        if (!ok) failed++;
        console.log(`   ${ok ? 'PASS' : 'FAIL'} ${k.padEnd(14)} relL2 ${v.toExponential(3)} (tol ${REL_TOL_FULL})`);
      }
      // Gate 3: valid rows — where an implementation bug would surface.
      {
        const ok = res.relValid <= REL_TOL_VALID;
        if (!ok) failed++;
        console.log(`   ${ok ? 'PASS' : 'FAIL'} valid-rows     relL2 ${res.relValid.toExponential(3)} (tol ${REL_TOL_VALID})`);
      }
    }
  } finally {
    await browser.close();
  }
  console.log(failed === 0 ? '\nALL TE PARITY GATES PASS' : `\n${failed} GATE(S) FAILED`);
  process.exit(failed === 0 ? 0 : 1);
})();
