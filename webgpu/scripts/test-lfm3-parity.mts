/**
 * LFM3 gate: Artifex Web's WebGPU forward pass vs llama.cpp.
 *
 * Reference: test-fixtures/lfm3-1.7b-llamacpp-reference.json — llama.cpp
 * b11945 (CPU) prompt tokens, 24 greedy tokens and the full-vocab top-5
 * probabilities at every step, for three prompts: a short one, one of 709
 * tokens (crosses the 256-token sliding window and the 512-token prefill
 * chunk), and a chat-templated one.
 *
 * Loads the model through the real app (the vite dev server must already be
 * running on 127.0.0.1:5173) and gates:
 *   1. tokenizer: prompt ids EXACT (raw text + chat template)
 *   2. teacher-forced decode (the engine is fed the reference's tokens so all
 *      24 steps stay comparable): the engine's argmax equals the reference
 *      token, except where the reference's own top-2 gap is a near-tie
 *   3. probabilities of the reference top-5 ids vs llama.cpp CPU AND CUDA
 *      (`steps_cuda`): max / mean abs diff. The two llama.cpp builds differ
 *      from each other by up to ~0.07 (activation rounding: Q8_K on CPU,
 *      Q8_1 on CUDA, f16 K/V); the engine stays f32 and is gated at that
 *      level — an operator bug (chunk order, gate, RoPE, window) is O(0.1-1).
 *   4. chunking: the long prompt prefilled in 64-token chunks must match the
 *      512-token run (conv state + KV carried across chunk boundaries).
 *
 * Run: npx tsx scripts/test-lfm3-parity.mts
 *   REPO=local/lfm3-1.7b (default)  ADAPTER=<regex> picks a GPU  HEADED=1
 */
import { chromium } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(readFileSync(
  resolve(here, '..', 'test-fixtures', 'lfm3-1.7b-llamacpp-reference.json'), 'utf8'));

const BASE = 'http://127.0.0.1:5173';
const REPO = process.env.REPO ?? 'local/lfm3-1.7b';
const ADAPTER = process.env.ADAPTER ? new RegExp(process.env.ADAPTER, 'i') : null;
const LOAD_TIMEOUT = Number(process.env.LOAD_TIMEOUT ?? 600_000);
const NEAR_TIE = 0.05;      // reference top-1 minus top-2 below this = either pick is fine
// llama.cpp disagrees with ITSELF by up to 0.07 (CUDA vs CPU, mean ~0.004)
// on these prompts — its CPU path rounds activations to Q8_K, CUDA to Q8_1.
// The engine is gated at that self-disagreement level against each build
// (max) plus a tight mean; an operator bug moves probabilities by O(0.1-1).
const MAX_ABS_TOL = 0.08;
const MEAN_ABS_TOL = 0.01;
const CHUNK_TOL = 2e-3;     // 64- vs 512-token prefill, same engine

type Step = { token: number; top: Array<{ id: number; prob: number }> };
type Probe = Array<{ token: number; top: Array<{ id: number; prob: number }>; probs: Record<string, number> }>;

(async () => {
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
  const fail = (msg: string) => { failed++; console.log(`   FAIL ${msg}`); };
  try {
    const page = await browser.newPage();
    page.on('console', (msg) => {
      const t = msg.text();
      if (process.env.VERBOSE || /error|warn|fail|\[GGUF\]|\[Tokenizer\]/i.test(t)) console.log(`[page] ${t}`);
    });
    page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
    await page.goto(`${BASE}/`);

    await page.waitForFunction(() => {
      const sel = document.getElementById('gpu-select') as HTMLSelectElement | null;
      return sel !== null && sel.options.length > 0;
    }, undefined, { timeout: 60_000 });
    if (ADAPTER) {
      const labels = await page.evaluate(() =>
        Array.from((document.getElementById('gpu-select') as HTMLSelectElement).options)
          .map(o => o.textContent ?? ''));
      const idx = labels.findIndex(l => ADAPTER.test(l));
      if (idx < 0) throw new Error(`adapter ${ADAPTER} not found: [${labels.join(' | ')}]`);
      await page.selectOption('#gpu-select', String(idx));
      await page.waitForFunction((re) => new RegExp(re, 'i').test(
        document.getElementById('f-gpu')?.textContent ?? ''), ADAPTER.source, { timeout: 60_000 });
    }
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

    // Report probabilities for the union of both references' top-5 ids.
    const probe = (ids: number[], steps: Step[], alt: Step[], chunk: number) => page.evaluate(
      ({ ids, forced, want, chunk }) => (globalThis as any).__logitProbe(ids, forced.length, chunk, forced, want),
      {
        ids, chunk, forced: steps.map(s => s.token),
        want: steps.map((s, i) => [...new Set([...s.top, ...alt[i].top].map(t => t.id))]),
      },
    ) as Promise<Probe>;

    for (const [name, c] of Object.entries<any>(ref.cases)) {
      console.log(`\n── ${name} (${c.n_prompt_tokens} prompt tokens)`);

      // Gate 1: tokenizer
      const got: number[] = name === 'chat_template'
        ? await page.evaluate((m) => (globalThis as any).__chatTokens(m, true), ref.chat_messages)
        : await page.evaluate((t) => (globalThis as any).__tokenize(t), c.prompt);
      if (JSON.stringify(got) === JSON.stringify(c.prompt_tokens)) {
        console.log(`   PASS prompt ids exact`);
      } else {
        const d = got.findIndex((v, i) => v !== c.prompt_tokens[i]);
        fail(`prompt ids differ at ${d}: got ${got.slice(d, d + 5)} want ${c.prompt_tokens.slice(d, d + 5)}`);
      }

      // Gates 2 + 3: teacher-forced decode
      const t1 = Date.now();
      const res = await probe(c.prompt_tokens, c.steps, c.steps_cuda, 512);
      console.log(`   probe ${((Date.now() - t1) / 1000).toFixed(1)}s`);
      let agree = 0, ties = 0;
      const bad: string[] = [];
      c.steps.forEach((st: Step, s: number) => {
        const gap = st.top[0].prob - st.top[1].prob;
        if (res[s].token === st.token) agree++;
        else if (gap < NEAR_TIE) ties++;
        else bad.push(`step ${s}: engine ${res[s].token} vs ref ${st.token} (ref gap ${gap.toFixed(3)})`);
      });
      console.log(`   argmax ${agree}/${c.steps.length} agree, ${ties} near-tie flips`);
      if (bad.length) fail(`argmax disagrees outside near-ties: ${bad.join('; ')}`);
      // |p_a - p_b| over the reference top-5 ids of every step.
      const diff = (a: (s: number, id: number) => number, b: Step[]) => {
        let max = 0, sum = 0, n = 0, at = '';
        b.forEach((st, s) => {
          for (const t of st.top) {
            const d = Math.abs(a(s, t.id) - t.prob);
            if (Number.isNaN(d)) continue;   // id outside the other run's top-5
            if (d > max) { max = d; at = `step ${s} id ${t.id}`; }
            sum += d; n++;
          }
        });
        return { max, mean: sum / n, at };
      };
      const engineP = (s: number, id: number) => res[s].probs[id];
      const cpu = diff(engineP, c.steps);
      const cuda = diff(engineP, c.steps_cuda);
      const cudaTop = new Map(c.steps_cuda.map((st: Step, s: number) => [s, st]));
      const spread = diff((s, id) => (cudaTop.get(s) as Step).top.find(t => t.id === id)?.prob ?? NaN, c.steps);
      const fmt = (d: { max: number; mean: number; at: string }) =>
        `max ${d.max.toFixed(4)} mean ${d.mean.toFixed(4)} (worst ${d.at})`;
      console.log(`   engine vs llama.cpp CPU : ${fmt(cpu)}`);
      console.log(`   engine vs llama.cpp CUDA: ${fmt(cuda)}`);
      console.log(`   llama.cpp CUDA vs CPU   : ${fmt(spread)}`);
      if (cuda.max > MAX_ABS_TOL) fail(`engine vs CUDA max ${cuda.max.toFixed(4)} > ${MAX_ABS_TOL}`);
      if (cuda.mean > MEAN_ABS_TOL) fail(`engine vs CUDA mean ${cuda.mean.toFixed(4)} > ${MEAN_ABS_TOL}`);
      if (cpu.max > MAX_ABS_TOL) fail(`engine vs CPU max ${cpu.max.toFixed(4)} > ${MAX_ABS_TOL}`);
      if (cpu.mean > MEAN_ABS_TOL) fail(`engine vs CPU mean ${cpu.mean.toFixed(4)} > ${MEAN_ABS_TOL}`);

      // Gate 4: chunked prefill consistency (the long prompt only)
      if (c.n_prompt_tokens > 64) {
        const res64 = await probe(c.prompt_tokens, c.steps, c.steps_cuda, 64);
        let worst = 0;
        res64.forEach((r, s) => {
          for (const id of Object.keys(r.probs)) worst = Math.max(worst, Math.abs(r.probs[id] - res[s].probs[id]));
        });
        console.log(`   64-token chunks vs 512: max prob diff ${worst.toExponential(2)}`);
        if (worst > CHUNK_TOL) fail(`chunked prefill drifts ${worst.toExponential(2)} > ${CHUNK_TOL}`);
      }
    }
  } finally {
    await browser.close();
  }
  console.log(failed ? `\nLFM3 PARITY: ${failed} FAILURE(S)` : '\nLFM3 PARITY: PASS');
  process.exit(failed ? 1 : 0);
})();
