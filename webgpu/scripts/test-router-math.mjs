/**
 * Phase C2 node test — router math equivalence. Run: npx tsx scripts/test-router-math.mjs
 *
 * Our fast path (moe-cpu.ts topKSoftmax) does top-k on raw logits, then
 * softmax over just those k. Ground truth (llama.cpp build_moe_ffn,
 * llama-graph.cpp:1436-1584) does full softmax over all E experts, takes
 * the top-k probabilities, then renormalizes. These are mathematically
 * identical (the full-softmax denominator cancels in the renormalize) —
 * this test proves it numerically to ~1e-7 over random logits.
 */

import { topKSoftmax } from '../src/engine/moe-cpu.ts';

/** Literal build_moe_ffn port: softmax(E) → top-k by prob → renormalize. */
function refRouter(logits, k) {
  const E = logits.length;
  let max = -Infinity;
  for (let e = 0; e < E; e++) if (logits[e] > max) max = logits[e];
  const probs = new Float64Array(E);
  let sum = 0;
  for (let e = 0; e < E; e++) { probs[e] = Math.exp(logits[e] - max); sum += probs[e]; }
  for (let e = 0; e < E; e++) probs[e] /= sum;

  const ids = [];
  const chosen = new Uint8Array(E);
  for (let j = 0; j < k; j++) {
    let best = -1;
    let bestV = -Infinity;
    for (let e = 0; e < E; e++) {
      if (!chosen[e] && probs[e] > bestV) { bestV = probs[e]; best = e; }
    }
    chosen[best] = 1;
    ids.push(best);
  }
  let wSum = 0;
  for (const e of ids) wSum += probs[e];
  const weights = ids.map((e) => probs[e] / wSum);
  return { ids, weights };
}

function seededRandFloats(n, seed, scale) {
  const out = new Float32Array(n);
  let s = seed >>> 0 || 0x9e3779b9;
  for (let i = 0; i < n; i++) {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    out[i] = ((s / 0xffffffff) * 2 - 1) * scale;
  }
  return out;
}

const E = 256;
const K = 8;
const TRIALS = 200;
let maxErr = 0;
let failures = 0;

for (let t = 0; t < TRIALS; t++) {
  // Mix of scales: tame logits, spiky logits, near-ties.
  const scale = [1, 4, 12, 0.05][t % 4];
  const logits = seededRandFloats(E, 0xfeed + t * 2654435761, scale);
  const got = topKSoftmax(logits, K);
  const ref = refRouter(logits, K);

  // Same expert set (order may differ only among exact ties — compare as sets).
  const gotSet = new Set(got.ids);
  const sameIds = ref.ids.every((e) => gotSet.has(e)) && gotSet.size === K;

  // Weight per expert id must match to ~1e-7.
  let err = 0;
  if (sameIds) {
    for (let j = 0; j < K; j++) {
      const gi = got.ids.indexOf(ref.ids[j]);
      err = Math.max(err, Math.abs(got.weights[gi] - ref.weights[j]));
    }
  }
  if (err > maxErr) maxErr = err;

  if (!sameIds || err > 1e-6) {
    failures++;
    console.error(`FAIL trial ${t} (scale ${scale}): sameIds=${sameIds}, maxWeightErr=${err.toExponential(2)}`);
    console.error(`  got ids ${[...got.ids]}, ref ids ${ref.ids}`);
  }
}

// Sanity: weights sum to 1 and are descending in logit order.
{
  const logits = seededRandFloats(E, 0xabcdef, 3);
  const { ids, weights } = topKSoftmax(logits, K);
  const sum = weights.reduce((a, b) => a + b, 0);
  if (Math.abs(sum - 1) > 1e-6) { failures++; console.error(`FAIL: weights sum ${sum}`); }
  for (let j = 1; j < K; j++) {
    if (logits[ids[j]] > logits[ids[j - 1]]) { failures++; console.error('FAIL: ids not logit-descending'); }
  }
}

if (failures > 0) {
  console.error(`\n${failures} trial(s) FAILED`);
  process.exit(1);
}
console.log(`all ${TRIALS} router trials PASS — top-${K}-then-softmax ≡ softmax-then-top-${K}-renorm (max weight err ${maxErr.toExponential(2)})`);
