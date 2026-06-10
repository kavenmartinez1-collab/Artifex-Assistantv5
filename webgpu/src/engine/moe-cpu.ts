/**
 * MoE CPU backend — Phase C control plane for Qwen3.6-35B-A3B experts.
 *
 * ROW-SPLIT layout: every worker owns a 1/NW row-strip of ALL experts
 * (worker w: gate/up rows [w·ffnDim/NW, …), down rows [w·H/NW, …), for all
 * 256 experts × 40 layers, ~22.25/NW GB each), fetched via the dev-server's
 * strided-gather endpoint. Every routed expert is computed by ALL workers in
 * parallel — perfect load balance, no straggler (the old ownership layout
 * was pinned at busy_max ≈ 2.1 ms/layer by E[max experts/worker] × the
 * slowest core's serial GEMV speed).
 *
 * Decode generation (SAB + Atomics):
 *   main: xq8 + request list → GEN++ → workers phase A: per expert k,
 *   gate/up strip GEMVs + silu⊙ → act2 slice into SAB → DONE_A barrier
 *   (worker-to-worker) → phase B: per expert k, quantize full act2[k] →
 *   down strip GEMV → weighted accumulate into the worker's DISJOINT
 *   out[w·H/NW …) slice → DONE. Main reads out[0..H) directly — no summing.
 *
 * The shared expert and router run on GPU — this backend only computes
 * Σ wᵢ·downᵢ(silu(gateᵢ·x) ⊙ upᵢ(x)) over the selected routed experts.
 *
 * Token batching (C4): a generation carries up to MOE_MAX_TOKENS tokens
 * (one for decode; a whole prefill chunk otherwise). One SAB round-trip per
 * layer per CHUNK instead of per token, and workers iterate (token, expert)
 * pairs expert-major so a strip fetched into cache is reused by every token
 * that routed to it.
 *
 * SAB layout (bytes; see sabLayout; M = MOE_MAX_TOKENS, K = MOE_MAX_K):
 *   ctl Int32[4]      — [0] GEN generation (-1 = shutdown), [1] DONE counter,
 *                       [2] LAYER index, [3] DONE_A phase-A barrier counter
 *   nTokens Int32[1]      — tokens this generation (≤ M; 0 for touch)
 *   reqCount Int32[M]     — routed experts per token (≤ K)
 *   reqExpert Int32[M×K]  — GLOBAL expert indices, token-major
 *   reqWeight Float32[M×K] — routing weights (softmax over top-8)
 *   xq8 M slots           — hidden vectors as Q8 act blocks (shared input)
 *   act2 Float32[M×K×ffnDim] — phase-A exchange: silu(gate)⊙up, slot
 *                            (t·K+k); worker w writes rows [w·ffnDim/NW, …)
 *   out Float32[M×H]      — routed output; worker w writes [t·H + w·H/NW, …)
 *   perf Float32[NW]      — per-worker busy ms for the last generation
 *
 * Protocol invariant: all non-atomic SAB writes happen BEFORE the Atomics
 * op that publishes them (GEN store on main; DONE_A/DONE add on workers),
 * which establishes the happens-before edge for the reader.
 */

import { ggmlTypeTraits, GGML_TYPES, type GGUFTensorInfo } from '../model/gguf';
import { q8Quantize } from '../bench/q5k-ref';

// ── Shared constants / SAB layout (single source of truth for the worker) ──

/**
 * Adaptive worker count. Row-split sharding requires the count to divide
 * both ffnDim (gate/up rows) and hiddenSize (down rows); powers of two do.
 * Work per worker is uniform (1/NW of every routed expert), so the only
 * scaling limit is hardware threads. Default: largest power of two ≤
 * hardwareConcurrency, clamped to [4, 16]. Override via
 * MoEBackendOpts.numWorkers (?moeWorkers= in main.ts) for A/B testing.
 */
export function pickMoEWorkerCount(ffnDim: number, hiddenSize: number): number {
  const hc = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 8;
  let n = 4;
  while (n * 2 <= hc && n * 2 <= 16) n *= 2;
  while (n > 1 && (ffnDim % n !== 0 || hiddenSize % n !== 0)) n /= 2;
  return n;
}
/** Max experts routed per token (top-8) — also worst case on one worker. */
export const MOE_MAX_K = 8;
/**
 * Max tokens per generation. Matches the hybrid PREFILL_CHUNK in generate.ts
 * (SSM recurrence caps prefill chunks at 16); decode always sends 1.
 */
export const MOE_MAX_TOKENS = 16;
/** block_q8_act: f32 d + 256 int8 q + 8 int16 bsums. */
export const Q8_BLOCK_BYTES = 276;

export const CTL_GEN = 0;
export const CTL_DONE = 1;
export const CTL_LAYER = 2;
/** Phase-A barrier counter (workers wait on each other, not main). */
export const CTL_DONE_A = 3;
/** CTL_LAYER sentinel: stream the whole shard (paging/bandwidth probe). */
export const TOUCH_LAYER = -2;

export interface SABLayout {
  /** Int32[1] — tokens this generation (0 for touch generations). */
  nTokensOff: number;
  /** Int32[M] — routed experts per token. */
  reqCountOff: number;
  /** Int32[M×K] — global expert ids, token-major (slot t·K+k). */
  reqExpertOff: number;
  /** Float32[M×K] — routing weights, token-major. */
  reqWeightOff: number;
  /** M consecutive Q8-act slots of xq8Bytes each. */
  xq8Off: number;
  /** Bytes per Q8-act slot (one hidden vector). */
  xq8Bytes: number;
  /** Float32[M×K×ffnDim] — phase-A act2 exchange (silu(gate)⊙up, slot t·K+k). */
  act2Off: number;
  /** Float32[M×H] — routed output, disjoint per-worker row slices per token. */
  outOff: number;
  /** Float32[NW] — per-worker busy ms for the last generation (perf probe). */
  perfOff: number;
  totalBytes: number;
}

export function sabLayout(numWorkers: number, hiddenSize: number, ffnDim: number): SABLayout {
  const K = MOE_MAX_K;
  const M = MOE_MAX_TOKENS;
  const nTokensOff = 16; // after ctl Int32[4]
  const reqCountOff = nTokensOff + 4;
  const reqExpertOff = reqCountOff + M * 4;
  const reqWeightOff = reqExpertOff + M * K * 4;
  const xq8Off = reqWeightOff + M * K * 4;
  const xq8Bytes = (hiddenSize / 256) * Q8_BLOCK_BYTES;
  let act2Off = xq8Off + M * xq8Bytes;
  act2Off = act2Off + ((16 - (act2Off % 16)) % 16);
  const outOff = act2Off + M * K * ffnDim * 4;
  const perfOff = outOff + M * hiddenSize * 4;
  const totalBytes = perfOff + numWorkers * 4;
  return { nTokensOff, reqCountOff, reqExpertOff, reqWeightOff, xq8Off, xq8Bytes, act2Off, outOff, perfOff, totalBytes };
}

export interface ExpertSlabDesc {
  /** Absolute file offset of the FULL (256-expert) tensor. */
  offset: number;
  ggmlType: number;
  /** Per-expert GEMV shape: y[rows] = W[rows×cols] @ x. */
  rows: number;
  cols: number;
  bytesPerExpert: number;
}

export interface MoELayerSlabs {
  gate: ExpertSlabDesc;
  up: ExpertSlabDesc;
  down: ExpertSlabDesc;
}

export interface MoEWorkerInitMsg {
  cmd: 'init';
  workerId: number;
  url: string;
  sab: SharedArrayBuffer;
  numExperts: number;
  hiddenSize: number;
  ffnDim: number;
  numWorkers: number;
  layers: MoELayerSlabs[];
}

/**
 * Router: top-k on raw logits, then softmax over just those k.
 * Mathematically identical to llama.cpp build_moe_ffn's
 * softmax(all)→top-k→renormalize (the full-softmax denominator cancels).
 * Validated against the literal port in scripts/test-router-math.mjs.
 */
export function topKSoftmax(logits: Float32Array, k: number): { ids: Int32Array; weights: Float32Array } {
  const E = logits.length;
  const ids = new Int32Array(k);
  const chosen = new Uint8Array(E);
  for (let j = 0; j < k; j++) {
    let best = -1;
    let bestV = -Infinity;
    for (let e = 0; e < E; e++) {
      if (!chosen[e] && logits[e] > bestV) { bestV = logits[e]; best = e; }
    }
    chosen[best] = 1;
    ids[j] = best;
  }
  const weights = new Float32Array(k);
  let max = -Infinity;
  for (let j = 0; j < k; j++) if (logits[ids[j]] > max) max = logits[ids[j]];
  let sum = 0;
  for (let j = 0; j < k; j++) {
    weights[j] = Math.exp(logits[ids[j]] - max);
    sum += weights[j];
  }
  for (let j = 0; j < k; j++) weights[j] /= sum;
  return { ids, weights };
}

// ── Backend ────────────────────────────────────────────────────────────

export interface MoERoute {
  ids: Int32Array | number[];
  weights: Float32Array | number[];
}

export interface MoEBackend {
  /**
   * Weighted routed-expert FFN for one token:
   * returns Σ weights[k] · expert_{expertIds[k]}(hidden), length hiddenSize.
   */
  computeExperts(
    layer: number,
    hidden: Float32Array,
    expertIds: Int32Array | number[],
    weights: Float32Array | number[],
  ): Promise<Float32Array>;
  /**
   * Token-batched variant (prefill chunks): hidden is token-major
   * [routes.length × hiddenSize]; returns the routed outputs in the same
   * layout. One worker generation for the whole chunk — workers iterate
   * (token, expert) pairs expert-major for strip cache reuse.
   * routes.length ≤ MOE_MAX_TOKENS.
   */
  computeExpertsBatch(
    layer: number,
    hidden: Float32Array,
    routes: MoERoute[],
  ): Promise<Float32Array>;
  /**
   * Warm-up / paging probe: workers [waveStart, waveStart+waveCount) stream
   * their FULL shard sequentially and report ms + GB/s (defaults: all).
   * Resident memory ≈ bench speed (~2 GB/s/worker); paged-out memory is
   * 10-100× slower. Run from console: await __MOE_BACKEND__.touchTest()
   */
  touchTest?(waveStart?: number, waveCount?: number): Promise<{ workerMs: number[]; gbps: number[] }>;
  destroy(): void;
  readonly numWorkers: number;
}

export interface MoEBackendOpts {
  /** Resolved GGUF file URL (LoadedGGUFModel.url) — workers Range-fetch it. */
  url: string;
  /** LoadedGGUFModel.expertTensors. */
  expertTensors: Map<string, GGUFTensorInfo>;
  numLayers: number;
  numExperts: number;
  hiddenSize: number;
  /** Expert FFN dim (gate/up rows). */
  ffnDim: number;
  numWorkers?: number;
  onProgress?: (message: string, frac: number) => void;
}

function slabDesc(t: GGUFTensorInfo, numExperts: number): ExpertSlabDesc {
  const [cols, rows, ne2] = t.ne;
  if (ne2 !== numExperts) {
    throw new Error(`[MoE] ${t.name}: ne[2]=${ne2}, expected ${numExperts} experts`);
  }
  const { blockSize, typeSize } = ggmlTypeTraits(t.ggmlType);
  if (cols % blockSize !== 0) throw new Error(`[MoE] ${t.name}: cols ${cols} not block-aligned`);
  const bytesPerExpert = rows * (cols / blockSize) * typeSize;
  if (bytesPerExpert * numExperts !== t.byteLength) {
    throw new Error(
      `[MoE] ${t.name}: computed ${bytesPerExpert}×${numExperts} ≠ byteLength ${t.byteLength}`,
    );
  }
  return { offset: t.offset, ggmlType: t.ggmlType, rows, cols, bytesPerExpert };
}

/** Serialize JS Q8Act into the wasm block_q8_act layout at `dst`. */
export function serializeQ8(act: ReturnType<typeof q8Quantize>, dst: Uint8Array): void {
  const nb = act.d.length;
  if (dst.byteLength !== nb * Q8_BLOCK_BYTES) {
    throw new Error(`[MoE] serializeQ8: dst ${dst.byteLength} B, want ${nb * Q8_BLOCK_BYTES}`);
  }
  const dv = new DataView(dst.buffer, dst.byteOffset, dst.byteLength);
  for (let i = 0; i < nb; i++) {
    const o = i * Q8_BLOCK_BYTES;
    dv.setFloat32(o, act.d[i], true);
    for (let j = 0; j < 256; j++) dst[o + 4 + j] = act.q[i * 256 + j] & 0xff;
    for (let g = 0; g < 8; g++) dv.setInt16(o + 260 + g * 2, act.bsums[i * 8 + g], true);
  }
}

/** Decode-perf accumulators, inspectable as __MOE_PERF__ in the console. */
export interface MoEPerf {
  calls: number;
  /** Q8 quantize + serialize + request routing + signal (main thread). */
  dispatchMs: number;
  /** Signal → all workers DONE (includes wake latency + compute). */
  waitMs: number;
  /** Max per-worker busy time per call, summed — the pure compute floor. */
  workerBusyMs: number;
  /** Mean busy time across ACTIVE workers per call, summed. */
  workerBusyAvgMs: number;
  /** Experts on the busiest worker per call, summed (routing imbalance). */
  maxExpertsPerWorker: number;
  /** Workers with ≥1 expert per call, summed. */
  activeWorkers: number;
  /** Partial-sum reduction (main thread). */
  sumMs: number;
  reset(): void;
}

function getMoEPerf(): MoEPerf {
  const g = globalThis as unknown as { __MOE_PERF__?: MoEPerf };
  if (!g.__MOE_PERF__) {
    g.__MOE_PERF__ = {
      calls: 0,
      dispatchMs: 0,
      waitMs: 0,
      workerBusyMs: 0,
      workerBusyAvgMs: 0,
      maxExpertsPerWorker: 0,
      activeWorkers: 0,
      sumMs: 0,
      reset() {
        this.calls = 0;
        this.dispatchMs = 0;
        this.waitMs = 0;
        this.workerBusyMs = 0;
        this.workerBusyAvgMs = 0;
        this.maxExpertsPerWorker = 0;
        this.activeWorkers = 0;
        this.sumMs = 0;
      },
    };
  }
  return g.__MOE_PERF__;
}

export async function createMoECPUBackend(opts: MoEBackendOpts): Promise<MoEBackend> {
  const { numLayers, numExperts, hiddenSize, ffnDim } = opts;
  const numWorkers = opts.numWorkers ?? pickMoEWorkerCount(ffnDim, hiddenSize);
  if (ffnDim % numWorkers !== 0 || hiddenSize % numWorkers !== 0) {
    throw new Error(`[MoE] row-split needs ${numWorkers} workers to divide ffnDim ${ffnDim} and hidden ${hiddenSize}`);
  }
  if (!crossOriginIsolated) {
    throw new Error('[MoE] not crossOriginIsolated — SharedArrayBuffer unavailable (COOP/COEP headers missing)');
  }

  // Per-layer slab descriptors from the deferred expert tensors.
  const layers: MoELayerSlabs[] = [];
  for (let l = 0; l < numLayers; l++) {
    const get = (kind: string): GGUFTensorInfo => {
      const t = opts.expertTensors.get(`blk.${l}.ffn_${kind}_exps.weight`);
      if (!t) throw new Error(`[MoE] missing expert tensor blk.${l}.ffn_${kind}_exps.weight`);
      return t;
    };
    const gate = slabDesc(get('gate'), numExperts);
    const up = slabDesc(get('up'), numExperts);
    const down = slabDesc(get('down'), numExperts);
    for (const s of [gate, up, down]) {
      if (s.ggmlType !== GGML_TYPES.Q5_K && s.ggmlType !== GGML_TYPES.Q6_K) {
        throw new Error(`[MoE] blk.${l} expert tensor has unsupported ggml type ${s.ggmlType} (only Q5_K/Q6_K wasm kernels exist)`);
      }
    }
    if (gate.rows !== ffnDim || gate.cols !== hiddenSize) {
      throw new Error(`[MoE] blk.${l} gate shape [${gate.rows},${gate.cols}] ≠ [${ffnDim},${hiddenSize}]`);
    }
    if (down.rows !== hiddenSize || down.cols !== ffnDim) {
      throw new Error(`[MoE] blk.${l} down shape [${down.rows},${down.cols}] ≠ [${hiddenSize},${ffnDim}]`);
    }
    layers.push({ gate, up, down });
  }

  const layout = sabLayout(numWorkers, hiddenSize, ffnDim);
  const sab = new SharedArrayBuffer(layout.totalBytes);
  const ctl = new Int32Array(sab, 0, 4);
  const nTokens = new Int32Array(sab, layout.nTokensOff, 1);
  const reqCount = new Int32Array(sab, layout.reqCountOff, MOE_MAX_TOKENS);
  const reqExpert = new Int32Array(sab, layout.reqExpertOff, MOE_MAX_TOKENS * MOE_MAX_K);
  const reqWeight = new Float32Array(sab, layout.reqWeightOff, MOE_MAX_TOKENS * MOE_MAX_K);
  const xq8 = new Uint8Array(sab, layout.xq8Off, MOE_MAX_TOKENS * layout.xq8Bytes);
  const out = new Float32Array(sab, layout.outOff, MOE_MAX_TOKENS * hiddenSize);
  const perfBusy = new Float32Array(sab, layout.perfOff, numWorkers);

  // ── Spawn + load the fleet ──
  const workers: Worker[] = [];
  const loaded = new Array<number>(numWorkers).fill(0);
  // Row-split: every worker holds 1/NW of every expert tensor.
  const totalBytes = layers.reduce(
    (s, L) => s + (L.gate.bytesPerExpert + L.up.bytesPerExpert + L.down.bytesPerExpert) * numExperts,
    0,
  );
  const shardBytes = totalBytes / numWorkers;

  try {
    await Promise.all(
      Array.from({ length: numWorkers }, (_, w) => {
        const worker = new Worker(new URL('./moe-worker.ts', import.meta.url), { type: 'module' });
        workers.push(worker);
        return new Promise<void>((resolve, reject) => {
          worker.onmessage = (ev) => {
            const m = ev.data;
            if (m.cmd === 'progress') {
              loaded[m.workerId] = m.loadedBytes;
              const sum = loaded.reduce((a, b) => a + b, 0);
              opts.onProgress?.(
                `Loading experts to RAM: ${(sum / 1e9).toFixed(1)} / ${(totalBytes / 1e9).toFixed(1)} GB`,
                sum / totalBytes,
              );
            } else if (m.cmd === 'ready') {
              resolve();
            } else if (m.cmd === 'error') {
              reject(new Error(`[MoE worker ${w}] ${m.message}`));
            }
          };
          worker.onerror = (e) => reject(new Error(`[MoE worker ${w}] ${e.message}`));
          const init: MoEWorkerInitMsg = {
            cmd: 'init',
            workerId: w,
            url: opts.url,
            sab,
            numExperts,
            hiddenSize,
            ffnDim,
            numWorkers,
            layers,
          };
          worker.postMessage(init);
        });
      }),
    );
  } catch (err) {
    workers.forEach((w) => w.terminate());
    throw err;
  }

  console.log(
    `[MoE] fleet ready: ${numWorkers} workers, row-split 1/${numWorkers} of all `
    + `${numExperts} experts × ${numLayers} layers (${(totalBytes / 1e9).toFixed(2)} GB in RAM)`,
  );

  // Sub-ms event-loop yield (same MessageChannel trick as the mapAsync pump
  // in gpu-bench/forward-pass). Atomics.waitAsync resolution rides the
  // macrotask queue (~1 ms each) and main re-armed it after EVERY worker's
  // DONE increment — up to 8 wakeups/layer ≈ 8 ms/layer of pure scheduling.
  const yieldChan = new MessageChannel();
  yieldChan.port1.start();
  const microYield = () =>
    new Promise<void>((resolve) => {
      yieldChan.port1.addEventListener('message', () => resolve(), { once: true });
      yieldChan.port2.postMessage(0);
    });

  const perf = getMoEPerf();
  let generation = 0;
  let inFlight = false;
  let destroyed = false;

  async function computeExpertsBatch(
    layer: number,
    hidden: Float32Array,
    routes: MoERoute[],
  ): Promise<Float32Array> {
    const T = routes.length;
    if (destroyed) throw new Error('[MoE] backend destroyed');
    if (inFlight) throw new Error('[MoE] computeExperts is not reentrant — decode is sequential');
    if (T < 1 || T > MOE_MAX_TOKENS) throw new Error(`[MoE] ${T} tokens > MOE_MAX_TOKENS ${MOE_MAX_TOKENS}`);
    if (hidden.length !== T * hiddenSize) {
      throw new Error(`[MoE] hidden length ${hidden.length} ≠ ${T}×${hiddenSize}`);
    }
    inFlight = true;
    try {
      const t0 = performance.now();
      let totalPairs = 0;
      for (let t = 0; t < T; t++) {
        // Quantize each token's hidden vector once, share with all workers.
        serializeQ8(
          q8Quantize(hidden.subarray(t * hiddenSize, (t + 1) * hiddenSize)),
          xq8.subarray(t * layout.xq8Bytes, (t + 1) * layout.xq8Bytes),
        );
        const { ids, weights } = routes[t];
        const n = ids.length;
        if (n !== weights.length) throw new Error('[MoE] expertIds/weights length mismatch');
        if (n > MOE_MAX_K) throw new Error(`[MoE] ${n} experts > MOE_MAX_K ${MOE_MAX_K}`);
        for (let k = 0; k < n; k++) {
          const e = ids[k] as number;
          if (e < 0 || e >= numExperts) throw new Error(`[MoE] expert id ${e} out of range`);
          reqExpert[t * MOE_MAX_K + k] = e;
          reqWeight[t * MOE_MAX_K + k] = weights[k] as number;
        }
        reqCount[t] = n;
        totalPairs += n;
      }
      nTokens[0] = T;

      Atomics.store(ctl, CTL_DONE, 0);
      Atomics.store(ctl, CTL_DONE_A, 0);
      Atomics.store(ctl, CTL_LAYER, layer);
      generation++;
      Atomics.store(ctl, CTL_GEN, generation);
      Atomics.notify(ctl, CTL_GEN);
      const t1 = performance.now();

      // Wait for all workers via sub-ms MessageChannel poll (NOT
      // Atomics.waitAsync — see yieldChan comment above).
      while (Atomics.load(ctl, CTL_DONE) !== numWorkers) {
        await microYield();
      }
      const t2 = performance.now();

      // Workers wrote disjoint row slices — the SAB out region IS the result.
      const result = out.slice(0, T * hiddenSize);

      let maxBusy = 0;
      let busySum = 0;
      for (let w = 0; w < numWorkers; w++) {
        if (perfBusy[w] > maxBusy) maxBusy = perfBusy[w];
        busySum += perfBusy[w];
      }
      perf.calls++;
      perf.dispatchMs += t1 - t0;
      perf.waitMs += t2 - t1;
      perf.workerBusyMs += maxBusy;
      perf.workerBusyAvgMs += busySum / numWorkers;
      perf.maxExpertsPerWorker += totalPairs; // row-split: every worker computes all pairs
      perf.activeWorkers += numWorkers;
      perf.sumMs += performance.now() - t2;
      return result;
    } finally {
      inFlight = false;
    }
  }

  function computeExperts(
    layer: number,
    hidden: Float32Array,
    expertIds: Int32Array | number[],
    weights: Float32Array | number[],
  ): Promise<Float32Array> {
    return computeExpertsBatch(layer, hidden, [{ ids: expertIds, weights }]);
  }

  async function touchTest(
    waveStart = 0,
    waveCount = numWorkers,
  ): Promise<{ workerMs: number[]; gbps: number[] }> {
    if (destroyed) throw new Error('[MoE] backend destroyed');
    if (inFlight) throw new Error('[MoE] busy — run touchTest while idle');
    inFlight = true;
    try {
      nTokens[0] = 0;
      // Wave bounds (reqExpert is unused by touch generations): workers
      // outside [start, end) check in without streaming. Waves keep the
      // pagefile from being thrashed by NW concurrent streams at near-OOM.
      reqExpert[0] = waveStart;
      reqExpert[1] = waveStart + waveCount;
      Atomics.store(ctl, CTL_DONE, 0);
      Atomics.store(ctl, CTL_DONE_A, 0);
      Atomics.store(ctl, CTL_LAYER, TOUCH_LAYER);
      generation++;
      Atomics.store(ctl, CTL_GEN, generation);
      Atomics.notify(ctl, CTL_GEN);
      while (Atomics.load(ctl, CTL_DONE) !== numWorkers) {
        await microYield();
      }
      const workerMs = Array.from(perfBusy).slice(waveStart, waveStart + waveCount);
      const gbps = workerMs.map((ms) => shardBytes / 1e9 / (ms / 1000));
      console.log(
        `[MoE touchTest] workers ${waveStart}-${waveStart + waveCount - 1} | `
        + `shard=${(shardBytes / 1e9).toFixed(2)} GB/worker | `
        + `ms: ${workerMs.map((m) => m.toFixed(0)).join(' ')} | `
        + `GB/s: ${gbps.map((g) => g.toFixed(2)).join(' ')}`,
      );
      return { workerMs, gbps };
    } finally {
      inFlight = false;
    }
  }

  function destroy(): void {
    if (destroyed) return;
    destroyed = true;
    Atomics.store(ctl, CTL_GEN, -1);
    Atomics.notify(ctl, CTL_GEN);
    workers.forEach((w) => w.terminate());
  }

  const backend: MoEBackend = { computeExperts, computeExpertsBatch, touchTest, destroy, numWorkers };
  // Console-accessible debug handle (touchTest paging probe).
  (globalThis as unknown as { __MOE_BACKEND__?: MoEBackend }).__MOE_BACKEND__ = backend;
  return backend;
}
