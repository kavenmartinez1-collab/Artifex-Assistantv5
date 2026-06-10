/**
 * Greedy parity diff: browser engine vs llama-server, teacher-forced.
 *
 * Usage:
 *   1. In the browser console (before sending a message): __DEBUG_PARITY__ = true
 *   2. Send the prompt with the Deterministic preset (temp 0).
 *   3. In the console: copy(JSON.stringify(__PARITY_LOG__))
 *   4. Paste into a file, e.g. webgpu/.parity-browser.json
 *   5. Start llama-server with the SAME GGUF (e.g. --cpu-moe, port 8090), then:
 *        node scripts/parity-diff.mjs .parity-browser.json [http://127.0.0.1:8090]
 *
 * Teacher-forcing: at each step i, llama-server is given the browser's exact
 * context (prompt + browser-chosen tokens 0..i-1) and asked for 1 greedy token
 * with top-5 probs. This keeps every step comparable even after a divergence
 * (a free-running compare forks at the first mismatch and tells you nothing
 * afterwards). cache_prompt=true makes each incremental call cheap.
 *
 * Output per mismatch: step, browser top-5 raw logits, llama top-5 probs,
 * browser top1-top2 gap (near-tie indicator). Summary: mismatch rate, and how
 * often the browser pick was inside llama's top-5 (quant noise) vs outside
 * (real corruption).
 */

import { readFileSync } from 'node:fs';

const file = process.argv[2];
const base = process.argv[3] ?? 'http://127.0.0.1:8090';
if (!file) {
  console.error('usage: node scripts/parity-diff.mjs <browser-parity.json> [server-url]');
  process.exit(1);
}

const browser = JSON.parse(readFileSync(file, 'utf8'));
const promptIds = browser.promptIds;
const steps = browser.steps;
if (!Array.isArray(promptIds) || !Array.isArray(steps) || steps.length === 0) {
  console.error('bad parity log: need { promptIds: number[], steps: [...] }');
  process.exit(1);
}
console.log(`browser log: ${promptIds.length} prompt tokens, ${steps.length} decode steps`);

async function llamaStep(ctx) {
  const resp = await fetch(`${base}/completion`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: ctx,              // raw token IDs — bypasses server-side templating
      n_predict: 1,
      temperature: 0,
      top_k: 1,
      n_probs: 5,
      cache_prompt: true,       // incremental: each call prefills only new tokens
      samplers: ['top_k'],
    }),
  });
  if (!resp.ok) throw new Error(`llama-server ${resp.status}: ${await resp.text()}`);
  const data = await resp.json();
  const cp = data.completion_probabilities;
  if (!Array.isArray(cp) || cp.length === 0) {
    throw new Error(`no completion_probabilities; keys: ${Object.keys(data)}`);
  }
  const entry = cp[0];
  const list = entry.probs ?? entry.top_logprobs ?? entry.top_probs ?? [];
  const top = list.map((p) => ({
    id: p.id ?? p.tok_id ?? -1,
    v: p.prob ?? p.logprob ?? NaN,
    s: p.tok_str ?? p.token ?? '',
  }));
  return { chosen: entry.id ?? entry.tok_id ?? (top[0]?.id ?? -1), top };
}

const ctx = [...promptIds];
let mismatches = 0;
let inTop5 = 0;       // mismatched but browser pick was in llama's top-5
let nearTies = 0;     // mismatched with browser top1-top2 gap < 0.05
let firstDiv = -1;

for (let i = 0; i < steps.length; i++) {
  const b = steps[i];
  const r = await llamaStep(ctx);

  if (b.chosen !== r.chosen) {
    mismatches++;
    if (firstDiv === -1) firstDiv = i;
    const gap = b.top5.length >= 2 ? b.top5[0][1] - b.top5[1][1] : NaN;
    if (gap < 0.05) nearTies++;
    if (r.top.some((t) => t.id === b.chosen)) inTop5++;
    console.log(`\nMISMATCH step ${i}: browser ${b.chosen} vs llama ${r.chosen} (browser gap ${gap.toFixed(4)})`);
    console.log('  browser top5 (raw logits): ' + b.top5.map(([id, v]) => `${id}:${v.toFixed(3)}`).join('  '));
    console.log('  llama   top5 (probs):      ' + r.top.map((t) => `${t.id}:${(+t.v).toFixed(3)}${t.s ? JSON.stringify(t.s) : ''}`).join('  '));
  } else if (i % 16 === 0) {
    process.stdout.write(`step ${i} ok\r`);
  }

  // Teacher-force with the BROWSER's choice so contexts stay identical.
  ctx.push(b.chosen);
}

console.log(`\n--- summary ---`);
console.log(`steps compared:   ${steps.length}`);
console.log(`mismatches:       ${mismatches} (${(100 * mismatches / steps.length).toFixed(1)}%)`);
if (mismatches > 0) {
  console.log(`first divergence: step ${firstDiv}`);
  console.log(`browser pick in llama top5: ${inTop5}/${mismatches} (high → quant noise on near-ties)`);
  console.log(`browser near-tie (gap<0.05): ${nearTies}/${mismatches}`);
}
process.exit(mismatches > 0 ? 2 : 0);
