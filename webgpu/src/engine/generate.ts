/**
 * Generation Loop — Autoregressive Text Generation
 *
 * Ties together the full inference pipeline:
 *   1. Tokenize the prompt
 *   2. Prefill (forward pass on all prompt tokens)
 *   3. Sample next token from logits
 *   4. Decode loop (forward pass on one token → sample → repeat)
 *   5. Stop on EOS or max length
 *   6. Detokenize output → text
 *
 * Sampling supports temperature, top-k, and top-p (nucleus sampling).
 */

import { readBuffer } from './buffers';
import { MAX_ATTN_SEQ_LEN, type ForwardPassEngine, type KVCache } from './forward-pass';
import type { Tokenizer } from '../model/tokenizer';

// ── Sampling Configuration ───────────────────────────────────────────────

export interface SamplingConfig {
  /** Temperature for logit scaling. 0 = greedy, 1 = proportional. Default: 0.7 */
  temperature?: number;
  /** Top-k: only consider the k most likely tokens. 0 = disabled. Default: 50 */
  topK?: number;
  /** Top-p (nucleus): keep tokens until cumulative probability exceeds p. Default: 0.9 */
  topP?: number;
  /** Maximum tokens to generate (not counting prompt). Default: 512 */
  maxNewTokens?: number;
  /** Repetition penalty. 1.0 = disabled. Default: 1.0 */
  repetitionPenalty?: number;
  /**
   * Min-p: keep tokens whose prob >= topProb * minP. Dynamic nucleus —
   * adapts to model confidence. 2026 canonical setting is 0.02-0.1.
   * 0 = disabled. Default: 0.05.
   */
  minP?: number;
  /**
   * DRY (Don't Repeat Yourself) penalty multiplier. Penalizes tokens that
   * would create an n-gram repeat with the recent context. 0 = disabled.
   * Standard range 0.5-1.5. Default: 0.8.
   */
  dryMultiplier?: number;
  /** DRY exponential base — penalty = multiplier * base^(match_len - allowedLen). Default: 1.75 */
  dryBase?: number;
  /** DRY minimum n-gram length that triggers a penalty. Default: 2 */
  dryAllowedLength?: number;
  /** DRY lookback window in tokens. Default: 512 */
  dryRangeLastN?: number;
  /** Use TurboQuant compressed KV cache (saves ~80% KV memory). Default: false */
  useCompressedKV?: boolean;
}

// ── Generation Output ────────────────────────────────────────────────────

export interface GenerationResult {
  /** Generated text (not including the prompt) */
  text: string;
  /** Generated token IDs */
  tokenIds: number[];
  /** Total tokens generated */
  numTokens: number;
  /** Prompt token count */
  promptTokens: number;
  /** Total time in milliseconds */
  totalMs: number;
  /** Tokens per second (generation only, not counting prefill) */
  tokensPerSecond: number;
  /** Stop reason: 'eos', 'max_length', or 'aborted' */
  stopReason: 'eos' | 'max_length' | 'aborted';
}

/** Callback invoked for each generated token (for streaming). */
export type OnTokenCallback = (token: string, tokenId: number, index: number) => void;

/** Abort handle returned by generate(). Call abort() to stop early. */
export interface GenerationHandle {
  /** Promise that resolves with the final result. */
  result: Promise<GenerationResult>;
  /** Call to stop generation early. */
  abort: () => void;
}

// ── Persistent KV session (multi-turn chat) ──────────────────────────────

/**
 * Holds a KV cache that survives across generate() calls so consecutive chat
 * turns don't re-prefill the whole conversation.
 *
 * Reuse rule: the new prompt's token ids must EXTEND cachedTokenIds exactly
 * (common prefix === full cached sequence). Then only the delta is prefilled.
 * Anything else — divergence, TurboQuant toggle, capacity overflow — resets
 * the cache and re-prefills from scratch. No rewind: hybrid models carry SSM
 * state (a running recurrence with no per-position history), so partial
 * rollback is unsound there; extension-or-reset is the rule for all models.
 *
 * cachedTokenIds tracks exactly what's IN the cache: prompt tokens plus every
 * generated token that was actually forwarded (the final sampled token of a
 * max_length/abort exit never enters the cache).
 */
export interface KVSessionState {
  kvCache: KVCache | null;
  cachedTokenIds: number[];
  /** Cache allocation in tokens. Grows when a turn needs more. */
  capacity: number;
  /** TurboQuant flag the current cache was created with. */
  compressed: boolean;
}

export function createKVSession(capacity: number): KVSessionState {
  return { kvCache: null, cachedTokenIds: [], capacity, compressed: false };
}

export interface GenerateOptions {
  /** Persistent KV session for multi-turn reuse. When set, generate() keeps
   *  the cache alive across calls instead of destroying it. */
  kvSession?: KVSessionState;
}

// ── Multimodal prompts ───────────────────────────────────────────────────

/** One image's patch embeddings, occupying a span of placeholder tokens
 *  (e.g. Qwen-VL's <|image_pad|> run) inside the prompt. */
export interface PromptImage {
  /** Index of the first placeholder position in tokenIds. */
  start: number;
  /** Number of placeholder positions (= embedding rows). */
  count: number;
  /** count × hiddenSize f32 rows from the vision encoder. */
  embeddings: Float32Array;
  /** Qwen3-VL DeepStack features: one count × hiddenSize array per early
   *  text layer, added to that layer's output at this image's positions. */
  deepstack?: Float32Array[];
  /** This image's span attends bidirectionally (Gemma text-side semantics).
   *  Requires the span to fit one prefill chunk. */
  bidirectional?: boolean;
}

/** Pre-tokenized prompt with image spans. Text positions embed normally;
 *  image spans inject precomputed rows via ForwardOptions.embeddings. */
export interface MultimodalPrompt {
  tokenIds: number[];
  images: PromptImage[];
}

function commonPrefixLen(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}

// ── Sampling Functions ───────────────────────────────────────────────────

/**
 * Sample a token ID from logits using temperature, top-k, and top-p.
 *
 * This runs on the CPU after reading logits back from the GPU.
 * For a 150K vocabulary, this takes < 1ms — not a bottleneck.
 */
/** Detect if the last N tokens form a repeating n-gram pattern */
function detectRepetition(ids: number[], ngramSize = 3, minRepeats = 3): boolean {
  if (ids.length < ngramSize * minRepeats) return false;
  // Check if the last ngramSize tokens have appeared minRepeats times consecutively
  const tail = ids.slice(-ngramSize);
  let repeats = 0;
  for (let i = ids.length - ngramSize; i >= 0; i -= ngramSize) {
    const chunk = ids.slice(i, i + ngramSize);
    if (chunk.length === ngramSize && chunk.every((v, j) => v === tail[j])) {
      repeats++;
      if (repeats >= minRepeats) return true;
    } else {
      break;
    }
  }
  return false;
}

/**
 * DRY (Don't Repeat Yourself) penalty — logit-space.
 *
 * For each position i in the recent window, compute the length L of the longest
 * suffix ending at i that matches the most-recent suffix ending at the current
 * token. If L >= allowedLength, the token at position i+1 is what we'd regurgitate
 * if we continued the match — penalize it by `multiplier * base^(L - allowedLength)`.
 *
 * Matches exllamav2 / llama.cpp DRY semantics. O(N^2) in the window size, which at
 * N=512 is ~262K comparisons worst case (sub-ms on modern CPU). The early-break in
 * the match-length loop keeps the typical case far below that.
 *
 * Notes on placement in the sampler pipeline:
 *   - Applied in RAW logit space, before temperature and before top-K filtering.
 *     Applying after temperature would make the multiplier scale with temperature,
 *     which is not what the algorithm expects.
 *   - Composes with frequency-repetition penalty — they attack different failure
 *     modes (frequency penalizes over-used tokens; DRY penalizes exact repeats).
 */
function applyDRY(
  logits: Float32Array,
  tokens: number[],
  multiplier: number,
  base: number,
  allowedLength: number,
  rangeLastN: number,
): void {
  if (multiplier <= 0 || tokens.length < 2) return;
  const start = Math.max(0, tokens.length - rangeLastN);
  const rel = tokens.length - start === tokens.length ? tokens : tokens.slice(start);
  const N = rel.length;
  if (N < 2) return;

  // penalty[tokenId] = max match length that would regurgitate that token
  const penalty = new Map<number, number>();

  // For each past position i, measure how far back rel[i - L] matches rel[N - 1 - L].
  // If the match is long enough, rel[i + 1] is the token we'd "complete the loop"
  // with, so penalize it.
  for (let i = 0; i < N - 1; i++) {
    let L = 0;
    while (i - L >= 0 && N - 1 - L >= 0 && rel[i - L] === rel[N - 1 - L]) {
      L++;
    }
    if (L >= allowedLength) {
      const candidate = rel[i + 1];
      const prev = penalty.get(candidate);
      if (prev === undefined || L > prev) penalty.set(candidate, L);
    }
  }

  const vocabSize = logits.length;
  for (const [tokenId, L] of penalty) {
    if (tokenId < vocabSize) {
      logits[tokenId] -= multiplier * Math.pow(base, L - allowedLength);
    }
  }
}

function sampleFromLogits(
  logits: Float32Array,
  config: Required<SamplingConfig>,
  generatedIds: number[],
): number {
  const vocabSize = logits.length;

  // ── Logit-distribution probe ──────────────────────────────────────────
  // Gated on __DEBUG_LOGIT_TOPK__ (number): interval in decode steps.
  // Logs raw top-5 token ids + values, plus distribution stats, so we can
  // tell whether the model's logits are concentrated (collapse) or broad.
  {
    const probeInterval = (globalThis as any).__DEBUG_LOGIT_TOPK__ as number | undefined;
    if (typeof probeInterval === 'number' && probeInterval > 0
        && generatedIds.length % probeInterval === 0) {
      let max = -Infinity, min = Infinity, sum = 0;
      for (let i = 0; i < vocabSize; i++) {
        const v = logits[i];
        if (v > max) max = v;
        if (v < min) min = v;
        sum += v;
      }
      const mean = sum / vocabSize;
      // Top-5 via simple selection — vocabSize is ~150k, one-shot cost is fine
      const top: { id: number; val: number }[] = [];
      const K = 5;
      for (let i = 0; i < vocabSize; i++) {
        const v = logits[i];
        if (top.length < K) {
          top.push({ id: i, val: v });
          if (top.length === K) top.sort((a, b) => b.val - a.val);
        } else if (v > top[K - 1].val) {
          top[K - 1] = { id: i, val: v };
          top.sort((a, b) => b.val - a.val);
        }
      }
      const topStr = top.map(t => `${t.id}:${t.val.toFixed(2)}`).join(',');
      const gap = top.length >= 2 ? (top[0].val - top[1].val).toFixed(2) : 'n/a';
      console.log(
        `[LOGIT-PROBE step=${generatedIds.length}] `
        + `max=${max.toFixed(2)} min=${min.toFixed(2)} mean=${mean.toFixed(3)} `
        + `top1-gap=${gap} top5=[${topStr}]`
      );
    }
  }

  // Apply repetition penalty — frequency-scaled (stronger for tokens seen more often)
  if (config.repetitionPenalty !== 1.0) {
    const freq = new Map<number, number>();
    for (const id of generatedIds) freq.set(id, (freq.get(id) ?? 0) + 1);
    for (const [id, count] of freq) {
      if (id < vocabSize) {
        // Scale penalty with frequency: penalty^(1 + log2(count))
        const scaledPenalty = Math.pow(config.repetitionPenalty, 1 + Math.log2(count));
        if (logits[id] > 0) {
          logits[id] /= scaledPenalty;
        } else {
          logits[id] *= scaledPenalty;
        }
      }
    }
  }

  // DRY penalty (operates on raw logits, before temperature & top-K).
  // See applyDRY() for algorithm notes.
  if (config.dryMultiplier > 0) {
    applyDRY(
      logits,
      generatedIds,
      config.dryMultiplier,
      config.dryBase,
      config.dryAllowedLength,
      config.dryRangeLastN,
    );
  }

  // Greedy: just return argmax
  if (config.temperature === 0) {
    let maxIdx = 0;
    let maxVal = logits[0];
    for (let i = 1; i < vocabSize; i++) {
      if (logits[i] > maxVal) {
        maxVal = logits[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  // ── PERF (Step 2): top-K via heap, not full sort ─────────────────────
  // Old code allocated ~vocabSize two-element arrays and sorted the entire
  // vocabulary just to keep the top-K. With vocab=248,320 this measured at
  // ~80ms/token (22% of total decode time). The replacement uses a min-heap
  // of size K over typed arrays — O(N log K) with zero per-element allocation.
  // Temperature is also deferred until AFTER top-K so we don't pay the
  // multiply for the ~99.98% of vocab entries that get discarded.
  //
  // Old (full-sort) implementation kept here for easy revert:
  // const invTemp = 1.0 / config.temperature;
  // for (let i = 0; i < vocabSize; i++) logits[i] *= invTemp;
  // const indexed: Array<[number, number]> = [];
  // for (let i = 0; i < vocabSize; i++) indexed.push([i, logits[i]]);
  // indexed.sort((a, b) => b[1] - a[1]);
  // let candidates = indexed;
  // if (config.topK > 0 && config.topK < vocabSize) candidates = candidates.slice(0, config.topK);

  // Resolve effective K: clamp to vocab and treat <=0 as "no limit" (full vocab)
  const k = (config.topK > 0 && config.topK < vocabSize) ? config.topK : vocabSize;

  // Heap-based top-K. Min-heap so root holds the smallest survivor; any new
  // candidate larger than root replaces it and we sift down.
  const topIdx = new Int32Array(k);
  const topVal = new Float32Array(k);
  // Seed heap with first K logits in heap order (we just fill then heapify).
  for (let i = 0; i < k; i++) { topIdx[i] = i; topVal[i] = logits[i]; }
  // Heapify (Floyd's algorithm) — siftDown from last parent to root.
  for (let parent = (k >>> 1) - 1; parent >= 0; parent--) {
    let p = parent;
    while (true) {
      const l = (p << 1) + 1; if (l >= k) break;
      const r = l + 1;
      let smallest = p;
      if (topVal[l] < topVal[smallest]) smallest = l;
      if (r < k && topVal[r] < topVal[smallest]) smallest = r;
      if (smallest === p) break;
      const tv = topVal[p]; topVal[p] = topVal[smallest]; topVal[smallest] = tv;
      const ti = topIdx[p]; topIdx[p] = topIdx[smallest]; topIdx[smallest] = ti;
      p = smallest;
    }
  }
  // Stream remaining logits; replace heap root whenever we find a bigger one.
  for (let i = k; i < vocabSize; i++) {
    const v = logits[i];
    if (v > topVal[0]) {
      topVal[0] = v;
      topIdx[0] = i;
      // siftDown root
      let p = 0;
      while (true) {
        const l = (p << 1) + 1; if (l >= k) break;
        const r = l + 1;
        let smallest = p;
        if (topVal[l] < topVal[smallest]) smallest = l;
        if (r < k && topVal[r] < topVal[smallest]) smallest = r;
        if (smallest === p) break;
        const tv = topVal[p]; topVal[p] = topVal[smallest]; topVal[smallest] = tv;
        const ti = topIdx[p]; topIdx[p] = topIdx[smallest]; topIdx[smallest] = ti;
        p = smallest;
      }
    }
  }
  // Sort the K survivors descending so top-P (cumulative) works in order.
  // K is small (default 50), so a tiny indirect sort is fine.
  const order = new Int32Array(k);
  for (let i = 0; i < k; i++) order[i] = i;
  // Simple insertion sort — K=50 → ~1250 comparisons, negligible.
  for (let i = 1; i < k; i++) {
    const oi = order[i]; const v = topVal[oi];
    let j = i - 1;
    while (j >= 0 && topVal[order[j]] < v) { order[j + 1] = order[j]; j--; }
    order[j + 1] = oi;
  }
  // Materialize candidates in the same [idx, logit] shape the rest of the
  // function expects, applying temperature now (only on K elements, not 248K).
  const invTemp = 1.0 / config.temperature;
  let candidates: Array<[number, number]> = new Array(k);
  for (let i = 0; i < k; i++) {
    const o = order[i];
    candidates[i] = [topIdx[o], topVal[o] * invTemp];
  }

  // Softmax on candidates
  let maxLogit = candidates[0][1];
  let sumExp = 0;
  const probs: number[] = [];
  for (const [, logit] of candidates) {
    const e = Math.exp(logit - maxLogit);
    probs.push(e);
    sumExp += e;
  }
  for (let i = 0; i < probs.length; i++) {
    probs[i] /= sumExp;
  }

  // Top-p (nucleus) filtering
  if (config.topP < 1.0) {
    let cumulative = 0;
    let cutoff = probs.length;
    for (let i = 0; i < probs.length; i++) {
      cumulative += probs[i];
      if (cumulative >= config.topP) {
        cutoff = i + 1;
        break;
      }
    }
    candidates = candidates.slice(0, cutoff);
    probs.length = cutoff;

    // Renormalize
    const newSum = probs.reduce((a, b) => a + b, 0);
    for (let i = 0; i < probs.length; i++) {
      probs[i] /= newSum;
    }
  }

  // Min-p (dynamic nucleus). Candidates are already sorted desc by logit,
  // so probs[0] is the top — threshold at topProb * minP and cut anything
  // below. This is stricter than top-p when the model is confident and
  // looser when it's uncertain — which is the whole point.
  if (config.minP > 0 && probs.length > 1) {
    const threshold = probs[0] * config.minP;
    let cutoff = probs.length;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] < threshold) { cutoff = i; break; }
    }
    if (cutoff < probs.length && cutoff > 0) {
      candidates = candidates.slice(0, cutoff);
      probs.length = cutoff;
      let newSum = 0;
      for (let i = 0; i < probs.length; i++) newSum += probs[i];
      for (let i = 0; i < probs.length; i++) probs[i] /= newSum;
    }
  }

  // Weighted random sample
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (r < cumulative) {
      return candidates[i][0];
    }
  }

  // Fallback (rounding errors)
  return candidates[candidates.length - 1][0];
}

// ── Parity capture ───────────────────────────────────────────────────────
// Gated on __DEBUG_PARITY__ (truthy): records prompt token IDs and, per decode
// step, the RAW top-5 logits (before any penalty/sampler mutation) plus the
// chosen token into globalThis.__PARITY_LOG__. Dump from the console with
// copy(JSON.stringify(__PARITY_LOG__)) and diff against llama-server via
// scripts/parity-diff.mjs.
function parityTop5(logits: Float32Array): Array<[number, number]> {
  const K = 5;
  const top: Array<[number, number]> = [];
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i];
    if (top.length < K) {
      top.push([i, v]);
      if (top.length === K) top.sort((a, b) => b[1] - a[1]);
    } else if (v > top[K - 1][1]) {
      top[K - 1] = [i, v];
      top.sort((a, b) => b[1] - a[1]);
    }
  }
  return top;
}

// ── Generation Loop ──────────────────────────────────────────────────────

/**
 * Generate text from a prompt (string or pre-tokenized IDs).
 *
 * @param device    - WebGPU device (for reading logits back)
 * @param engine    - Forward pass engine
 * @param tokenizer - Tokenizer for decode
 * @param prompt    - Input text string OR pre-tokenized token IDs (from applyChatTemplate)
 * @param sampling  - Sampling configuration
 * @param onToken   - Optional callback for streaming output
 * @returns GenerationHandle with result promise and abort function
 */
export function generate(
  device: GPUDevice,
  engine: ForwardPassEngine,
  tokenizer: Tokenizer,
  prompt: string | number[] | MultimodalPrompt,
  sampling: SamplingConfig = {},
  onToken?: OnTokenCallback,
  opts?: GenerateOptions,
): GenerationHandle {
  let aborted = false;

  const config: Required<SamplingConfig> = {
    temperature: sampling.temperature ?? 0.7,
    topK: sampling.topK ?? 50,
    topP: sampling.topP ?? 0.9,
    maxNewTokens: sampling.maxNewTokens ?? 512,
    repetitionPenalty: sampling.repetitionPenalty ?? 1.0,
    // Sampler defaults are NEUTRAL (off) — the caller (UI / preset layer) is
    // responsible for opting into any aggressive sampler. Aggressive defaults
    // here were previously invisible from the chat UI and caused word-chain
    // collapse past ~500 tokens on Qwen3.5 (2026-04-19 investigation: greedy
    // decode produced coherent output while the same engine with minP=0.05 +
    // DRY=0.8 collapsed). Keep these at zero; let presets flip them on.
    minP: sampling.minP ?? 0,
    dryMultiplier: sampling.dryMultiplier ?? 0,
    dryBase: sampling.dryBase ?? 1.75,
    dryAllowedLength: sampling.dryAllowedLength ?? 2,
    dryRangeLastN: sampling.dryRangeLastN ?? 512,
    useCompressedKV: sampling.useCompressedKV ?? false,
  };

  const result = (async (): Promise<GenerationResult> => {
    const startTime = performance.now();

    // ── Step 1: Tokenize ─────────────────────────────────────────────
    const mm: MultimodalPrompt | null =
      typeof prompt === 'object' && !Array.isArray(prompt) ? prompt : null;
    const promptIds = typeof prompt === 'string'
      ? tokenizer.encode(prompt)
      : mm ? mm.tokenIds : prompt as number[];
    const hasImages = mm !== null && mm.images.length > 0;
    const promptTokens = promptIds.length;
    const parityLog: { promptIds: number[]; steps: Array<{ top5: Array<[number, number]>; chosen: number }> } | null =
      (globalThis as any).__DEBUG_PARITY__ ? { promptIds: [...promptIds], steps: [] } : null;
    if (parityLog) {
      (globalThis as any).__PARITY_LOG__ = parityLog;
      console.log('[Parity] capture armed — dump with copy(JSON.stringify(__PARITY_LOG__))');
    }
    console.log(`[Generate] Prompt: ${promptTokens} tokens, first 5: [${promptIds.slice(0, 5).join(', ')}]`);
    // Debug: decode the full prompt to see what the model actually sees
    console.log(`[Generate] Decoded prompt: "${tokenizer.decode(promptIds)}"`);

    if (promptTokens === 0) {
      return {
        text: '', tokenIds: [], numTokens: 0, promptTokens: 0,
        totalMs: 0, tokensPerSecond: 0, stopReason: 'max_length',
      };
    }

    // ── Step 2: Create or reuse KV cache ─────────────────────────────
    const sess = opts?.kvSession ?? null;
    let kvCache: KVCache;
    let prefillIds = promptIds;
    if (sess) {
      // The attention kernels can read at most MAX_ATTN_SEQ_LEN positions —
      // clamp the cache size AND the generation budget to it. The sliding
      // window (chat/context.ts) keys its token budget off sess.capacity, so
      // prompts stay under this by construction; the clamp is the backstop.
      if (promptTokens >= MAX_ATTN_SEQ_LEN) {
        throw new Error(
          `Prompt is ${promptTokens} tokens but the attention kernel supports at most `
          + `${MAX_ATTN_SEQ_LEN - 1} prompt tokens. Shorten the input or enable history compression.`);
      }
      const effectiveMaxNew = Math.min(config.maxNewTokens, MAX_ATTN_SEQ_LEN - promptTokens);
      if (effectiveMaxNew < config.maxNewTokens) {
        console.warn(
          `[Generate] maxNewTokens clamped ${config.maxNewTokens} -> ${effectiveMaxNew} `
          + `(attention kernel limit ${MAX_ATTN_SEQ_LEN})`);
        config.maxNewTokens = effectiveMaxNew;
      }
      sess.capacity = Math.min(Math.max(sess.capacity, promptTokens + config.maxNewTokens), MAX_ATTN_SEQ_LEN);
    }
    const needed = promptTokens + config.maxNewTokens;
    if (sess) {
      const prefixLen = commonPrefixLen(sess.cachedTokenIds, promptIds);
      const pureExtension =
        sess.kvCache !== null
        && !hasImages                  // image pads are textually identical across
                                       // different images — token prefix match is unsound
        && sess.compressed === config.useCompressedKV
        && sess.kvCache.position === sess.cachedTokenIds.length  // sanity: tracker matches cache
        && prefixLen === sess.cachedTokenIds.length
        && prefixLen > 0
        && promptTokens > prefixLen                              // non-empty delta to prefill
        && needed <= sess.kvCache.maxSeqLen;
      if (pureExtension) {
        kvCache = sess.kvCache!;
        prefillIds = promptIds.slice(prefixLen);
        console.log(`[Generate] KV reuse: ${prefixLen} cached tokens, prefilling ${prefillIds.length} new`);
      } else {
        const hadTokens = sess.cachedTokenIds.length;
        if (sess.kvCache) engine.destroyKVCache(sess.kvCache);
        kvCache = engine.createKVCache(sess.capacity, config.useCompressedKV);
        sess.kvCache = kvCache;
        sess.compressed = config.useCompressedKV;
        sess.cachedTokenIds = [];
        if (hadTokens > 0) {
          console.log(`[Generate] KV reset: prefix match ${prefixLen}/${hadTokens} — full prefill`);
        }
      }
    } else {
      kvCache = engine.createKVCache(needed, config.useCompressedKV);
    }
    if (config.useCompressedKV) {
      console.log(`[Generate] Using TurboQuant compressed KV cache`);
    }

    // ── Step 3: Prefill ─────────────────────────────────────────────
    // Hybrid models (Gated DeltaNet) have a sequential SSM recurrence inside each
    // SSM layer, but the surrounding stages (embed, full-attention layers, FFN,
    // final norm, lm_head) all process seqLen > 1 in a single dispatch. Option A
    // wraps the existing single-token SSM block in a per-token JS loop INSIDE
    // engine.forward, so multi-token chunks no longer require N separate JS→GPU
    // round-trips for the non-SSM stages.
    const isHybrid = engine.config.isHybrid === true;
    // const PREFILL_CHUNK = isHybrid ? 1 : 512;  // Pre-Option-A: 1 token/chunk
    const PREFILL_CHUNK_DEFAULT = isHybrid ? 16 : 512;
    // Diagnostic override: caller may set __DEBUG_PREFILL_CHUNK__ (positive int)
    // to force a specific chunk size, e.g. 1 to take the single-row path on every
    // prompt token (slow, but isolates batched-GEMM precision from chunked-prefill
    // semantics). When unset, use the production default.
    const chunkOverride = (globalThis as any).__DEBUG_PREFILL_CHUNK__;
    const PREFILL_CHUNK = (typeof chunkOverride === 'number' && chunkOverride > 0)
      ? Math.floor(chunkOverride)
      : PREFILL_CHUNK_DEFAULT;
    if (PREFILL_CHUNK !== PREFILL_CHUNK_DEFAULT) {
      console.log(`[Generate] PREFILL_CHUNK override: ${PREFILL_CHUNK} (default ${PREFILL_CHUNK_DEFAULT})`);
    }
    // Debug: if first-forward-pass debug is armed, also fire debug for the
    // last prefill position (true generation-context snapshot).
    if ((globalThis as any).__DEBUG_FORWARD_PASS__ === true) {
      (globalThis as any).__DEBUG_LAST_PREFILL_POS__ = promptTokens - 1;
    }
    let prefillOutput;
    // Divergence probe: if the caller pre-armed __DEBUG_DUMP_STATS__ (any
    // truthy value), use that as the trigger for multi-step dumps. The caller
    // may also set __DEBUG_DUMP_DECODE_STEPS__ = [1, 10, 50, ...] to collect
    // dumps at those decode positions. The flag is consumed by forward() each
    // time it fires; we re-arm with a unique tag per dump point.
    const probeEnabled = !!(globalThis as any).__DEBUG_DUMP_STATS__;
    const decodeDumpSteps: number[] = Array.isArray((globalThis as any).__DEBUG_DUMP_DECODE_STEPS__)
      ? (globalThis as any).__DEBUG_DUMP_DECODE_STEPS__ as number[]
      : [];
    // Start clean — we'll set the flag with a tag string at the right moments.
    (globalThis as any).__DEBUG_DUMP_STATS__ = false;
    if (probeEnabled) {
      (globalThis as any).__DEBUG_DUMP_RESULT__ = {};  // reset accumulator
      console.log(`[Probe] enabled; decode dump steps: [${decodeDumpSteps.join(', ')}]`);
    }
    // Build prefill segments: text runs embed normally; image spans inject
    // precomputed rows. Chunks never straddle a segment boundary. (With
    // images, KV reuse is off, so prefillIds === promptIds.)
    const Hdim = engine.config.hiddenSize;
    const segments: Array<{
      ids: number[]; embeds?: Float32Array; deepstack?: Float32Array[]; bidirectional?: boolean;
    }> = [];
    if (hasImages) {
      const imgs = [...mm!.images].sort((a, b) => a.start - b.start);
      let cursor = 0;
      for (const img of imgs) {
        if (img.embeddings.length !== img.count * Hdim) {
          throw new Error(
            `[Generate] image at ${img.start}: ${img.embeddings.length} floats != count*H = ${img.count * Hdim}`);
        }
        if (img.start < cursor) throw new Error('[Generate] overlapping image spans');
        if (img.start > cursor) segments.push({ ids: promptIds.slice(cursor, img.start) });
        segments.push({
          ids: promptIds.slice(img.start, img.start + img.count),
          embeds: img.embeddings,
          deepstack: img.deepstack,
          bidirectional: img.bidirectional,
        });
        cursor = img.start + img.count;
      }
      if (cursor < promptIds.length) segments.push({ ids: promptIds.slice(cursor) });
    } else {
      segments.push({ ids: prefillIds });
    }

    const prefillCount = segments.reduce((s, seg) => s + seg.ids.length, 0);
    for (let s = 0; s < segments.length; s++) {
      const seg = segments[s];
      for (let i = 0; i < seg.ids.length; i += PREFILL_CHUNK) {
        const chunkEnd = Math.min(i + PREFILL_CHUNK, seg.ids.length);
        const chunk = new Uint32Array(seg.ids.slice(i, chunkEnd));
        const isLastChunk = s === segments.length - 1 && chunkEnd >= seg.ids.length;
        if (probeEnabled && isLastChunk) {
          (globalThis as any).__DEBUG_DUMP_STATS__ = 'prefill-end';
        }
        if (seg.bidirectional && seg.ids.length > PREFILL_CHUNK) {
          throw new Error(
            `[Generate] bidirectional image span (${seg.ids.length} tokens) exceeds the `
            + `prefill chunk (${PREFILL_CHUNK}) — bidirectionality cannot span chunks`);
        }
        const fwdOpts = seg.embeds
          ? {
              embeddings: seg.embeds.subarray(i * Hdim, chunkEnd * Hdim),
              deepstackFeatures: seg.deepstack?.map(f => f.subarray(i * Hdim, chunkEnd * Hdim)),
              bidirectional: seg.bidirectional,
            }
          : undefined;
        prefillOutput = await engine.forward(chunk, kvCache, fwdOpts);
      }
    }
    await device.queue.onSubmittedWorkDone();
    // The cache now holds every prompt token (cached prefix + the delta we just prefilled).
    if (sess) sess.cachedTokenIds = [...promptIds];

    const prefillEnd = performance.now();
    const prefillMs = prefillEnd - startTime;
    const chunks = Math.ceil(prefillCount / PREFILL_CHUNK);
    console.log(`[Generate] Prefill: ${prefillCount} tokens in ${chunks} chunk(s), ${prefillMs.toFixed(0)}ms (${(prefillCount / prefillMs * 1000).toFixed(0)} tok/s)`);

    // ── Step 4: Sample first token from prefill logits ───────────────
    const generatedIds: number[] = [];
    let stopReason: 'eos' | 'max_length' | 'aborted' = 'max_length';
    const decodeStart = performance.now();

    // Streaming decode: decoding one token at a time splits multi-byte
    // characters (emoji stream as U+FFFD). Decode the full sequence each
    // step, hold back a trailing replacement char (possibly an incomplete
    // codepoint), and emit only the stable delta.
    let emittedText = '';
    const emitDelta = (tokenId: number, index: number) => {
      if (!onToken) return;
      let full = tokenizer.decode(generatedIds);
      if (full.endsWith('�')) full = full.slice(0, -1);
      if (full.length > emittedText.length && full.startsWith(emittedText)) {
        const delta = full.slice(emittedText.length);
        emittedText = full;
        onToken(delta, tokenId, index);
      }
    };

    // Read logits from the last prefill step
    const firstLogitsRaw = await readBuffer(
      device, prefillOutput!.logitsBuffer, engine.config.vocabSize * 4,
    );
    const firstLogits = new Float32Array(firstLogitsRaw);
    // Parity: capture raw top-5 BEFORE sampleFromLogits (it mutates logits in place)
    if (parityLog) parityLog.steps.push({ top5: parityTop5(firstLogits), chosen: -1 });
    let lastTokenId = sampleFromLogits(firstLogits, config, generatedIds);
    if (parityLog) parityLog.steps[parityLog.steps.length - 1].chosen = lastTokenId;

    if (tokenizer.isEos(lastTokenId)) {
      stopReason = 'eos';
    } else {
      generatedIds.push(lastTokenId);
      emitDelta(lastTokenId, 0);

      // ── Step 5: Decode loop ──────────────────────────────────────
      // Step 1 (perf measurement): per-step CPU/GPU/readback/sample breakdown
      // for the first few decode tokens. Lets us confirm whether the per-token
      // bottleneck is JS dispatch building, GPU compute, or logits readback —
      // critical before deciding which optimization (bind-group cache, encoder
      // persistence, kernel fusion) is actually worth doing.
      const __perfStepBudget = 5; // log first N decode steps
      let __perfFwdCpuSum = 0, __perfGpuWaitSum = 0, __perfReadbackSum = 0, __perfSampleSum = 0;
      let __perfDispatchSum = 0, __perfStepCount = 0;
      let __perfMoESyncSum = 0, __perfMoEExpertSum = 0;
      // Reset MoE backend perf so the SUMMARY covers decode only (not prefill).
      ((globalThis as any).__MOE_PERF__ as { reset(): void } | undefined)?.reset?.();
      for (let step = 1; step < config.maxNewTokens; step++) {
        if (aborted) {
          stopReason = 'aborted';
          break;
        }

        // Divergence probe: re-arm dump flag with a step-specific tag when
        // step number matches the caller-configured decode dump schedule.
        if (probeEnabled && decodeDumpSteps.includes(step)) {
          (globalThis as any).__DEBUG_DUMP_STATS__ = `decode-${step}`;
        }

        // Forward pass with the last GENERATED token (not prompt token)
        const input = new Uint32Array([lastTokenId]);
        const __tFwdStart = performance.now();
        const output = await engine.forward(input, kvCache);
        if (sess) sess.cachedTokenIds.push(lastTokenId);  // token is now IN the cache
        const __tFwdEnd = performance.now();

        // Read logits back from GPU
        await device.queue.onSubmittedWorkDone();
        const __tGpuDone = performance.now();
        const logitsRaw = await readBuffer(
          device,
          output.logitsBuffer,
          engine.config.vocabSize * 4,
        );
        const __tReadback = performance.now();
        const logits = new Float32Array(logitsRaw);

      // Debug: log top-5 logits on first decode step
      if (step === 0) {
        const indexed = Array.from(logits).map((v, i) => [i, v] as [number, number]);
        indexed.sort((a, b) => b[1] - a[1]);
        const top5 = indexed.slice(0, 5);
        console.log(`[DEBUG logits] top 5:`, top5.map(([id, v]) => `${id}(${v.toFixed(2)})`).join(', '));
        let lo = Infinity, hi = -Infinity, sum = 0;
        for (let i = 0; i < logits.length; i++) { lo = Math.min(lo, logits[i]); hi = Math.max(hi, logits[i]); sum += logits[i]; }
        console.log(`[DEBUG logits] min=${lo.toFixed(2)}, max=${hi.toFixed(2)}, mean=${(sum / logits.length).toFixed(4)}`);
        // Decode top token
        const topText = tokenizer.decode([top5[0][0]]);
        console.log(`[DEBUG logits] top token: "${topText}" (id=${top5[0][0]})`);
      }

      // Sample next token
      if (parityLog) parityLog.steps.push({ top5: parityTop5(logits), chosen: -1 });
      const nextId = sampleFromLogits(logits, config, generatedIds);
      if (parityLog) parityLog.steps[parityLog.steps.length - 1].chosen = nextId;
      const __tSample = performance.now();

      // ── Per-step perf accounting (Step 1 measurement layer) ──────────
      // fwdCpu  = wall time inside engine.forward (encoder build + submit)
      // gpuWait = time blocked on device.queue.onSubmittedWorkDone (GPU compute)
      // readback= time mapping the logits buffer
      // sample  = sampleFromLogits cost
      // dispatches = how many compute dispatches the forward pass enqueued
      const __fwdCpuMs = __tFwdEnd - __tFwdStart;
      const __gpuWaitMs = __tGpuDone - __tFwdEnd;
      const __readbackMs = __tReadback - __tGpuDone;
      const __sampleMs = __tSample - __tReadback;
      const __lastFwd = (globalThis as any).__perfLastForward as
        | { dispatches: number; copies: number; cpuMs: number; bgCreates?: number;
            moeSyncMs?: number; moeExpertMs?: number } | undefined;
      const __dispatchCount = __lastFwd ? __lastFwd.dispatches : -1;
      const __bgCreates = __lastFwd?.bgCreates ?? -1;
      const __moeSyncMs = __lastFwd?.moeSyncMs ?? 0;
      const __moeExpertMs = __lastFwd?.moeExpertMs ?? 0;
      __perfFwdCpuSum += __fwdCpuMs;
      __perfGpuWaitSum += __gpuWaitMs;
      __perfReadbackSum += __readbackMs;
      __perfSampleSum += __sampleMs;
      __perfDispatchSum += __dispatchCount;
      __perfMoESyncSum += __moeSyncMs;
      __perfMoEExpertSum += __moeExpertMs;
      __perfStepCount++;
      if (step <= __perfStepBudget) {
        const __total = __fwdCpuMs + __gpuWaitMs + __readbackMs + __sampleMs;
        console.log(
          `[perf decode #${step}] fwd_cpu=${__fwdCpuMs.toFixed(1)}ms `
          + `gpu_wait=${__gpuWaitMs.toFixed(1)}ms `
          + `readback=${__readbackMs.toFixed(1)}ms `
          + `sample=${__sampleMs.toFixed(1)}ms `
          + `total=${__total.toFixed(1)}ms `
          + `dispatches=${__dispatchCount} bg_creates=${__bgCreates}`
          + (__moeSyncMs + __moeExpertMs > 0
            ? ` | moe_sync=${__moeSyncMs.toFixed(1)}ms moe_experts=${__moeExpertMs.toFixed(1)}ms (inside fwd_cpu)`
            : '')
        );
      } else if (step === __perfStepBudget + 1) {
        // Print bottleneck diagnosis once, then go silent
        const __sum = __fwdCpuMs + __gpuWaitMs + __readbackMs + __sampleMs;
        const pct = (n: number) => ((n / __sum) * 100).toFixed(0) + '%';
        let diag = 'mixed';
        if (__fwdCpuMs > __gpuWaitMs * 1.5) diag = 'CPU/dispatch-bound — bind-group cache & encoder fusion will help most';
        else if (__gpuWaitMs > __fwdCpuMs * 1.5) diag = 'GPU-bound — kernel fusion (norm+proj, FFN) is the win';
        else if (__readbackMs > __fwdCpuMs && __readbackMs > __gpuWaitMs) diag = 'readback-bound — async overlap will help most';
        console.log(
          `[perf decode] DIAGNOSIS: ${diag} `
          + `(fwd_cpu=${pct(__fwdCpuMs)}, gpu_wait=${pct(__gpuWaitMs)}, `
          + `readback=${pct(__readbackMs)}, sample=${pct(__sampleMs)})`
        );
      }

      // Check for EOS
      if (tokenizer.isEos(nextId)) {
        stopReason = 'eos';
        break;
      }

      // Store and emit
      generatedIds.push(nextId);
      lastTokenId = nextId;

      emitDelta(nextId, step);

      // Stop if stuck in a repeating loop (saves wasted tokens)
      if (detectRepetition(generatedIds, 3, 4) || detectRepetition(generatedIds, 2, 6)) {
        stopReason = 'eos';
        console.log(`[Generate] Stopping: repetition detected at step ${step}`);
        break;
      }
      }

      // ── Per-decode summary (Step 1 measurement layer) ────────────────
      // Averaged breakdown across ALL decode steps; this is the load-bearing
      // number for picking the next optimization. 'fwd_cpu' = JS dispatch
      // building inside engine.forward; 'gpu_wait' = GPU compute time;
      // 'readback' = mapAsync of the logits buffer; 'sample' = JS sampling.
      if (__perfStepCount > 0) {
        const n = __perfStepCount;
        const avgFwd = __perfFwdCpuSum / n;
        const avgGpu = __perfGpuWaitSum / n;
        const avgRb = __perfReadbackSum / n;
        const avgSp = __perfSampleSum / n;
        const avgDisp = __perfDispatchSum / n;
        const avgTotal = avgFwd + avgGpu + avgRb + avgSp;
        const pct = (n: number) => ((n / avgTotal) * 100).toFixed(0) + '%';
        let diag = 'mixed (no clear single bottleneck)';
        if (avgFwd > avgGpu * 1.5) diag = 'CPU/dispatch-bound — bind-group cache & encoder fusion are highest leverage';
        else if (avgGpu > avgFwd * 1.5) diag = 'GPU-bound — kernel fusion (norm+proj, FFN swiglu) is the win';
        else if (avgRb > avgFwd && avgRb > avgGpu) diag = 'readback-bound — async overlap will help most';
        const avgMoESync = __perfMoESyncSum / n;
        const avgMoEExp = __perfMoEExpertSum / n;
        console.log(
          `[perf decode SUMMARY over ${n} steps] `
          + `avg_total=${avgTotal.toFixed(1)}ms (=${(1000 / avgTotal).toFixed(2)} tok/s) | `
          + `fwd_cpu=${avgFwd.toFixed(1)}ms (${pct(avgFwd)}) `
          + `gpu_wait=${avgGpu.toFixed(1)}ms (${pct(avgGpu)}) `
          + `readback=${avgRb.toFixed(1)}ms (${pct(avgRb)}) `
          + `sample=${avgSp.toFixed(1)}ms (${pct(avgSp)}) | `
          + `avg_dispatches/token=${avgDisp.toFixed(0)}`
          + (avgMoESync + avgMoEExp > 0
            ? ` | moe_sync=${avgMoESync.toFixed(1)}ms moe_experts=${avgMoEExp.toFixed(1)}ms (inside fwd_cpu)`
            : '')
        );
        const __moePerf = (globalThis as any).__MOE_PERF__ as
          | { calls: number; dispatchMs: number; waitMs: number; workerBusyMs: number;
              workerBusyAvgMs: number; maxExpertsPerWorker: number; activeWorkers: number; sumMs: number }
          | undefined;
        if (__moePerf && __moePerf.calls > 0) {
          const per = (v: number) => (v / __moePerf.calls).toFixed(2);
          console.log(
            `[perf decode SUMMARY] MoE per layer-call (n=${__moePerf.calls}): `
            + `dispatch=${per(__moePerf.dispatchMs)}ms `
            + `wait=${per(__moePerf.waitMs)}ms `
            + `(worker_busy_max=${per(__moePerf.workerBusyMs)}ms `
            + `busy_avg=${per(__moePerf.workerBusyAvgMs)}ms → `
            + `sched_overhead=${per(__moePerf.waitMs - __moePerf.workerBusyMs)}ms) `
            + `sum=${per(__moePerf.sumMs)}ms | `
            + `experts_on_busiest=${per(__moePerf.maxExpertsPerWorker)} `
            + `active_workers=${per(__moePerf.activeWorkers)}/${(globalThis as any).__MOE_BACKEND__?.numWorkers ?? '?'} `
            + `ms/expert_busiest=${(__moePerf.workerBusyMs / __moePerf.maxExpertsPerWorker).toFixed(2)}`
          );
        }
        console.log(`[perf decode SUMMARY] DIAGNOSIS: ${diag}`);
      }
    } // end if !eos from first token

    const endTime = performance.now();
    const totalMs = endTime - startTime;
    const decodeMs = endTime - decodeStart;
    const tokensPerSecond = generatedIds.length > 0
      ? (generatedIds.length / decodeMs) * 1000
      : 0;

    // ── Step 5: Detokenize ───────────────────────────────────────────
    const text = generatedIds.length > 0 ? tokenizer.decode(generatedIds) : '';
    // Flush any held-back tail (e.g. a final emoji that completed at EOS).
    if (onToken && text.length > emittedText.length && text.startsWith(emittedText)) {
      onToken(text.slice(emittedText.length), -1, generatedIds.length);
    }

    // Clean up KV cache buffers — unless a session owns the cache (multi-turn
    // reuse). Session caches are destroyed by the owner (reset/unload).
    if (!sess) engine.destroyKVCache(kvCache);
    // Image turns poison prefix reuse: a future prompt with the SAME pad
    // tokens but a DIFFERENT image would false-match the token prefix.
    if (sess && hasImages) sess.cachedTokenIds = [];

    return {
      text,
      tokenIds: generatedIds,
      numTokens: generatedIds.length,
      promptTokens,
      totalMs,
      tokensPerSecond,
      stopReason,
    };
  })();

  return {
    result,
    abort: () => { aborted = true; },
  };
}
