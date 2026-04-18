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
import type { ForwardPassEngine, KVCache } from './forward-pass';
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

function sampleFromLogits(
  logits: Float32Array,
  config: Required<SamplingConfig>,
  generatedIds: number[],
): number {
  const vocabSize = logits.length;

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
  prompt: string | number[],
  sampling: SamplingConfig = {},
  onToken?: OnTokenCallback,
): GenerationHandle {
  let aborted = false;

  const config: Required<SamplingConfig> = {
    temperature: sampling.temperature ?? 0.7,
    topK: sampling.topK ?? 50,
    topP: sampling.topP ?? 0.9,
    maxNewTokens: sampling.maxNewTokens ?? 512,
    repetitionPenalty: sampling.repetitionPenalty ?? 1.0,
    useCompressedKV: sampling.useCompressedKV ?? false,
  };

  const result = (async (): Promise<GenerationResult> => {
    const startTime = performance.now();

    // ── Step 1: Tokenize ─────────────────────────────────────────────
    const promptIds = typeof prompt === 'string' ? tokenizer.encode(prompt) : prompt;
    const promptTokens = promptIds.length;
    console.log(`[Generate] Prompt: ${promptTokens} tokens, first 5: [${promptIds.slice(0, 5).join(', ')}]`);
    // Debug: decode the full prompt to see what the model actually sees
    console.log(`[Generate] Decoded prompt: "${tokenizer.decode(promptIds)}"`);

    if (promptTokens === 0) {
      return {
        text: '', tokenIds: [], numTokens: 0, promptTokens: 0,
        totalMs: 0, tokensPerSecond: 0, stopReason: 'max_length',
      };
    }

    // ── Step 2: Create KV cache ──────────────────────────────────────
    const maxSeq = promptTokens + config.maxNewTokens;
    const kvCache = engine.createKVCache(maxSeq, config.useCompressedKV);
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
    const PREFILL_CHUNK = isHybrid ? 16 : 512;
    // Debug: if first-forward-pass debug is armed, also fire debug for the
    // last prefill position (true generation-context snapshot).
    if ((globalThis as any).__DEBUG_FORWARD_PASS__ === true) {
      (globalThis as any).__DEBUG_LAST_PREFILL_POS__ = promptTokens - 1;
    }
    let prefillOutput;
    for (let i = 0; i < promptTokens; i += PREFILL_CHUNK) {
      const chunkEnd = Math.min(i + PREFILL_CHUNK, promptTokens);
      const chunk = new Uint32Array(promptIds.slice(i, chunkEnd));
      prefillOutput = await engine.forward(chunk, kvCache);
    }
    await device.queue.onSubmittedWorkDone();

    const prefillEnd = performance.now();
    const prefillMs = prefillEnd - startTime;
    const chunks = Math.ceil(promptTokens / PREFILL_CHUNK);
    console.log(`[Generate] Prefill: ${promptTokens} tokens in ${chunks} chunk(s), ${prefillMs.toFixed(0)}ms (${(promptTokens / prefillMs * 1000).toFixed(0)} tok/s)`);

    // ── Step 4: Sample first token from prefill logits ───────────────
    const generatedIds: number[] = [];
    let stopReason: 'eos' | 'max_length' | 'aborted' = 'max_length';
    const decodeStart = performance.now();

    // Read logits from the last prefill step
    const firstLogitsRaw = await readBuffer(
      device, prefillOutput!.logitsBuffer, engine.config.vocabSize * 4,
    );
    const firstLogits = new Float32Array(firstLogitsRaw);
    let lastTokenId = sampleFromLogits(firstLogits, config, generatedIds);

    if (tokenizer.isEos(lastTokenId)) {
      stopReason = 'eos';
    } else {
      generatedIds.push(lastTokenId);
      if (onToken) {
        onToken(tokenizer.decode([lastTokenId]), lastTokenId, 0);
      }

      // ── Step 5: Decode loop ──────────────────────────────────────
      // Step 1 (perf measurement): per-step CPU/GPU/readback/sample breakdown
      // for the first few decode tokens. Lets us confirm whether the per-token
      // bottleneck is JS dispatch building, GPU compute, or logits readback —
      // critical before deciding which optimization (bind-group cache, encoder
      // persistence, kernel fusion) is actually worth doing.
      const __perfStepBudget = 5; // log first N decode steps
      let __perfFwdCpuSum = 0, __perfGpuWaitSum = 0, __perfReadbackSum = 0, __perfSampleSum = 0;
      let __perfDispatchSum = 0, __perfStepCount = 0;
      for (let step = 1; step < config.maxNewTokens; step++) {
        if (aborted) {
          stopReason = 'aborted';
          break;
        }

        // Forward pass with the last GENERATED token (not prompt token)
        const input = new Uint32Array([lastTokenId]);
        const __tFwdStart = performance.now();
        const output = await engine.forward(input, kvCache);
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
      const nextId = sampleFromLogits(logits, config, generatedIds);
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
        | { dispatches: number; copies: number; cpuMs: number } | undefined;
      const __dispatchCount = __lastFwd ? __lastFwd.dispatches : -1;
      __perfFwdCpuSum += __fwdCpuMs;
      __perfGpuWaitSum += __gpuWaitMs;
      __perfReadbackSum += __readbackMs;
      __perfSampleSum += __sampleMs;
      __perfDispatchSum += __dispatchCount;
      __perfStepCount++;
      if (step <= __perfStepBudget) {
        const __total = __fwdCpuMs + __gpuWaitMs + __readbackMs + __sampleMs;
        console.log(
          `[perf decode #${step}] fwd_cpu=${__fwdCpuMs.toFixed(1)}ms `
          + `gpu_wait=${__gpuWaitMs.toFixed(1)}ms `
          + `readback=${__readbackMs.toFixed(1)}ms `
          + `sample=${__sampleMs.toFixed(1)}ms `
          + `total=${__total.toFixed(1)}ms `
          + `dispatches=${__dispatchCount}`
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

      if (onToken) {
        const tokenText = tokenizer.decode([nextId]);
        onToken(tokenText, nextId, step);
      }

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
        console.log(
          `[perf decode SUMMARY over ${n} steps] `
          + `avg_total=${avgTotal.toFixed(1)}ms (=${(1000 / avgTotal).toFixed(2)} tok/s) | `
          + `fwd_cpu=${avgFwd.toFixed(1)}ms (${pct(avgFwd)}) `
          + `gpu_wait=${avgGpu.toFixed(1)}ms (${pct(avgGpu)}) `
          + `readback=${avgRb.toFixed(1)}ms (${pct(avgRb)}) `
          + `sample=${avgSp.toFixed(1)}ms (${pct(avgSp)}) | `
          + `avg_dispatches/token=${avgDisp.toFixed(0)}`
        );
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

    // Clean up KV cache buffers
    engine.destroyKVCache(kvCache);

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
