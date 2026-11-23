/**
 * Parity gate: Artifex Web's WebGPU forward pass vs llama.cpp, for the model
 * Artifex Web ships (Qwen3.5-2B since 2026-11-23).
 *
 * Reference: test-fixtures/<FIXTURE> written by scripts/gen-llamacpp-reference.py
 * - llama.cpp's prompt tokens, 24 greedy tokens and the full-vocab top-5
 * probabilities at every step, from a CPU and a GPU build, for three prompts:
 * a short one, a long one (crosses the 512-token prefill chunk) and a
 * chat-templated one.
 *
 * Loads the model through the real app (the vite dev server must already be
 * running on 127.0.0.1:5173) and gates:
 *   1. tokenizer: prompt ids EXACT (raw text + chat template)
 *   2. teacher-forced decode (the engine is fed the reference's tokens so all
 *      24 steps stay comparable): the engine's argmax equals the reference
 *      token, except where the reference's own top-2 gap is a near-tie
 *   3. probabilities of the reference top-5 ids vs llama.cpp CPU AND GPU
 *      (`steps_cuda`): mean abs diff per build, and the max distance from the
 *      range the two builds span (their own disagreement is the tolerance);
 *      an operator bug is O(0.1-1).
 *   4. chunking: the long prompt prefilled in 64-token chunks must match the
 *      512-token run (recurrent state + KV carried across chunk boundaries).
 *
 * Run: npx tsx scripts/test-model-parity.mts
 *   REPO=local/qwen3.5-2b (default)  FIXTURE=qwen3.5-2b-llamacpp-reference.json (default)
 *   ADAPTER=<regex> picks a GPU  HEADED=1
 */
import { chromium } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(readFileSync(
  resolve(here, '..', 'test-fixtures', process.env.FIXTURE ?? 'qwen3.5-2b-llamacpp-reference.json'), 'utf8'));

const BASE = 'http://127.0.0.1:5173';
const REPO = process.env.REPO ?? 'local/qwen3.5-2b';
const ADAPTER = process.env.ADAPTER ? new RegExp(process.env.ADAPTER, 'i') : null;
const LOAD_TIMEOUT = Number(process.env.LOAD_TIMEOUT ?? 600_000);
const NEAR_TIE = 0.05;      // reference top-1 minus top-2 below this = either pick is fine
// llama.cpp disagrees with ITSELF by up to 0.07 (CUDA vs CPU, mean ~0.004)
// on these prompts — its CPU path rounds activations to Q8_K, CUDA to Q8_1.
// The engine is gated at that self-disagreement level against each build
// (max) plus a tight mean; an operator bug moves probabilities by O(0.1-1).
// A fixture may set its own max_abs_tol, with its reason in max_abs_tol_note.
const MAX_ABS_TOL: number = ref.max_abs_tol ?? 0.08;
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

    // The app's own prefill chunk, and the largest the engine allows.
    const limits = await page.evaluate(() => (globalThis as any).__prefillLimits()) as { chunk: number; max: number };
    const altChunk = limits.max !== limits.chunk ? limits.max : 64;
    console.log(`prefill chunk ${limits.chunk} (engine max ${limits.max})`);

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
      const res = await probe(c.prompt_tokens, c.steps, c.steps_cuda, limits.chunk);
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
      // Max gate: distance from the range llama.cpp's two builds span at each
      // (step, id). Where the builds themselves disagree by 0.08 (a model-
      // dependent soft spot), an engine on either side of that range is as
      // right as llama.cpp is; per-build max alone would fail it.
      let outside = 0, outAt = '';
      c.steps.forEach((st: Step, s: number) => {
        for (const t of st.top) {
          const g = (cudaTop.get(s) as Step).top.find(u => u.id === t.id)?.prob;
          const e = engineP(s, t.id);
          if (g === undefined || Number.isNaN(e)) continue;
          const lo = Math.min(t.prob, g), hi = Math.max(t.prob, g);
          const d = e < lo ? lo - e : e > hi ? e - hi : 0;
          if (d > outside) { outside = d; outAt = `step ${s} id ${t.id}`; }
        }
      });
      console.log(`   engine outside llama.cpp's CPU..GPU range: max ${outside.toFixed(4)}${outAt ? ` (worst ${outAt})` : ''}`);
      if (outside > MAX_ABS_TOL) fail(`engine is ${outside.toFixed(4)} outside llama.cpp's own range > ${MAX_ABS_TOL}`);
      if (cuda.mean > MEAN_ABS_TOL) fail(`engine vs CUDA mean ${cuda.mean.toFixed(4)} > ${MEAN_ABS_TOL}`);
      if (cpu.mean > MEAN_ABS_TOL) fail(`engine vs CPU mean ${cpu.mean.toFixed(4)} > ${MEAN_ABS_TOL}`);

      // Gate 4: chunked prefill consistency (the long prompt only)
      if (c.n_prompt_tokens > Math.min(limits.chunk, altChunk)) {
        const res64 = await probe(c.prompt_tokens, c.steps, c.steps_cuda, altChunk);
        let worst = 0;
        res64.forEach((r, s) => {
          for (const id of Object.keys(r.probs)) worst = Math.max(worst, Math.abs(r.probs[id] - res[s].probs[id]));
        });
        console.log(`   ${altChunk}-token chunks vs ${limits.chunk}: max prob diff ${worst.toExponential(2)}`);
        if (worst > CHUNK_TOL) fail(`chunked prefill drifts ${worst.toExponential(2)} > ${CHUNK_TOL}`);
      }
    }
  } finally {
    await browser.close();
  }
  console.log(failed ? `\nMODEL PARITY (${REPO}): ${failed} FAILURE(S)` : `\nMODEL PARITY (${REPO}): PASS`);
  process.exit(failed ? 1 : 0);
})();
