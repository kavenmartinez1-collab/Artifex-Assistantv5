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

  // Apply temperature
  const invTemp = 1.0 / config.temperature;
  for (let i = 0; i < vocabSize; i++) {
    logits[i] *= invTemp;
  }

  // Build (index, logit) pairs for sorting
  const indexed: Array<[number, number]> = [];
  for (let i = 0; i < vocabSize; i++) {
    indexed.push([i, logits[i]]);
  }

  // Sort descending by logit
  indexed.sort((a, b) => b[1] - a[1]);

  // Top-k filtering
  let candidates = indexed;
  if (config.topK > 0 && config.topK < vocabSize) {
    candidates = candidates.slice(0, config.topK);
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
    // Hybrid models (with Gated DeltaNet linear attention) need token-by-token
    // prefill because the SSM recurrence must process each token sequentially.
    // Standard transformer models can batch prefill up to 512 tokens at once.
    const isHybrid = engine.config.isHybrid === true;
    const PREFILL_CHUNK = isHybrid ? 1 : 512;
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
      for (let step = 1; step < config.maxNewTokens; step++) {
        if (aborted) {
          stopReason = 'aborted';
          break;
        }

        // Forward pass with the last GENERATED token (not prompt token)
        const input = new Uint32Array([lastTokenId]);
        const output = await engine.forward(input, kvCache);

        // Read logits back from GPU
        await device.queue.onSubmittedWorkDone();
        const logitsRaw = await readBuffer(
          device,
          output.logitsBuffer,
          engine.config.vocabSize * 4,
        );
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
