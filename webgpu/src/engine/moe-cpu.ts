/**
 * MoE CPU backend — Phase C control plane for Qwen3.6-35B-A3B experts.
 *
 * Spawns a fleet of WASM workers; each holds a contiguous shard of experts
 * (default 8 workers × 32 experts × 40 layers, ~2.78 GB each) fetched
 * directly from the GGUF file via HTTP Range. Decode-time dispatch rides a
 * SharedArrayBuffer + Atomics: main quantizes the hidden vector to Q8 once,
 * writes per-worker expert requests, bumps a generation counter; workers
 * GEMV their selected experts (gate→up→silu⊙→Q8→down) and accumulate the
 * weighted routed output into per-worker outboxes which main sums.
 *
 * The shared expert and router run on GPU — this backend only computes
 * Σ wᵢ·downᵢ(silu(gateᵢ·x) ⊙ upᵢ(x)) over the selected routed experts.
 *
 * SAB layout (bytes; see sabLayout):
 *   ctl Int32[4]      — [0] GEN generation (-1 = shutdown), [1] DONE counter,
 *                       [2] LAYER index for this generation
 *   reqCount Int32[NW]        — selected experts owned by each worker
 *   reqExpert Int32[NW×K]     — LOCAL expert indices (worker-relative)
 *   reqWeight Float32[NW×K]   — routing weights (softmax over top-8)
 *   xq8 bytes                 — hidden vector as Q8 act blocks (shared input)
 *   out Float32[NW×H]         — per-worker weighted routed-output partials
 *
 * Protocol invariant: all non-atomic SAB writes happen BEFORE the
 * Atomics.store to GEN (main) / DONE (worker), which establishes the
 * happens-before edge for the reader.
 */

import { ggmlTypeTraits, GGML_TYPES, type GGUFTensorInfo } from '../model/gguf';
import { q8Quantize } from '../bench/q5k-ref';

// ── Shared constants / SAB layout (single source of truth for the worker) ──

export const MOE_NUM_WORKERS = 8;
/** Max experts routed per token (top-8) — also worst case on one worker. */
export const MOE_MAX_K = 8;
/** block_q8_act: f32 d + 256 int8 q + 8 int16 bsums. */
export const Q8_BLOCK_BYTES = 276;

export const CTL_GEN = 0;
export const CTL_DONE = 1;
export const CTL_LAYER = 2;

export interface SABLayout {
  reqCountOff: number;
  reqExpertOff: number;
  reqWeightOff: number;
  xq8Off: number;
  xq8Bytes: number;
  outOff: number;
  totalBytes: number;
}

export function sabLayout(numWorkers: number, hiddenSize: number): SABLayout {
  const K = MOE_MAX_K;
  const reqCountOff = 16; // after ctl Int32[4]
  const reqExpertOff = reqCountOff + numWorkers * 4;
  const reqWeightOff = reqExpertOff + numWorkers * K * 4;
  const xq8Off = reqWeightOff + numWorkers * K * 4;
  const xq8Bytes = (hiddenSize / 256) * Q8_BLOCK_BYTES;
  let outOff = xq8Off + xq8Bytes;
  outOff = outOff + ((16 - (outOff % 16)) % 16);
  const totalBytes = outOff + numWorkers * hiddenSize * 4;
  return { reqCountOff, reqExpertOff, reqWeightOff, xq8Off, xq8Bytes, outOff, totalBytes };
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
  expertStart: number;
  expertCount: number;
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
  destroy(): void;
  readonly numWorkers: number;
  readonly expertsPerWorker: number;
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

type WaitAsyncFn = (
  ta: Int32Array,
  i: number,
  v: number,
) => { async: boolean; value: Promise<string> | string };

export async function createMoECPUBackend(opts: MoEBackendOpts): Promise<MoEBackend> {
  const numWorkers = opts.numWorkers ?? MOE_NUM_WORKERS;
  const { numLayers, numExperts, hiddenSize, ffnDim } = opts;
  if (numExperts % numWorkers !== 0) {
    throw new Error(`[MoE] ${numExperts} experts not divisible by ${numWorkers} workers`);
  }
  if (!crossOriginIsolated) {
    throw new Error('[MoE] not crossOriginIsolated — SharedArrayBuffer unavailable (COOP/COEP headers missing)');
  }
  const expertsPerWorker = numExperts / numWorkers;

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

  const layout = sabLayout(numWorkers, hiddenSize);
  const sab = new SharedArrayBuffer(layout.totalBytes);
  const ctl = new Int32Array(sab, 0, 4);
  const reqCount = new Int32Array(sab, layout.reqCountOff, numWorkers);
  const reqExpert = new Int32Array(sab, layout.reqExpertOff, numWorkers * MOE_MAX_K);
  const reqWeight = new Float32Array(sab, layout.reqWeightOff, numWorkers * MOE_MAX_K);
  const xq8 = new Uint8Array(sab, layout.xq8Off, layout.xq8Bytes);
  const out = new Float32Array(sab, layout.outOff, numWorkers * hiddenSize);

  // ── Spawn + load the fleet ──
  const workers: Worker[] = [];
  const loaded = new Array<number>(numWorkers).fill(0);
  const shardBytes = layers.reduce(
    (s, L) => s + (L.gate.bytesPerExpert + L.up.bytesPerExpert + L.down.bytesPerExpert) * expertsPerWorker,
    0,
  );
  const totalBytes = shardBytes * numWorkers;

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
            expertStart: w * expertsPerWorker,
            expertCount: expertsPerWorker,
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
    `[MoE] fleet ready: ${numWorkers} workers × ${expertsPerWorker} experts × ${numLayers} layers `
    + `(${(totalBytes / 1e9).toFixed(2)} GB in RAM)`,
  );

  const waitAsync = (Atomics as unknown as { waitAsync?: WaitAsyncFn }).waitAsync;
  let generation = 0;
  let inFlight = false;
  let destroyed = false;

  async function computeExperts(
    layer: number,
    hidden: Float32Array,
    expertIds: Int32Array | number[],
    weights: Float32Array | number[],
  ): Promise<Float32Array> {
    if (destroyed) throw new Error('[MoE] backend destroyed');
    if (inFlight) throw new Error('[MoE] computeExperts is not reentrant — decode is sequential');
    if (hidden.length !== hiddenSize) throw new Error(`[MoE] hidden length ${hidden.length} ≠ ${hiddenSize}`);
    if (expertIds.length !== weights.length) throw new Error('[MoE] expertIds/weights length mismatch');
    inFlight = true;
    try {
      // Quantize the hidden vector once, share with all workers.
      serializeQ8(q8Quantize(hidden), xq8);

      reqCount.fill(0);
      for (let k = 0; k < expertIds.length; k++) {
        const e = expertIds[k] as number;
        if (e < 0 || e >= numExperts) throw new Error(`[MoE] expert id ${e} out of range`);
        const w = Math.floor(e / expertsPerWorker);
        const n = reqCount[w];
        if (n >= MOE_MAX_K) throw new Error(`[MoE] >${MOE_MAX_K} experts routed to worker ${w}`);
        reqExpert[w * MOE_MAX_K + n] = e - w * expertsPerWorker; // local index
        reqWeight[w * MOE_MAX_K + n] = weights[k] as number;
        reqCount[w] = n + 1;
      }

      Atomics.store(ctl, CTL_DONE, 0);
      Atomics.store(ctl, CTL_LAYER, layer);
      generation++;
      Atomics.store(ctl, CTL_GEN, generation);
      Atomics.notify(ctl, CTL_GEN);

      // Wait for all workers (idle ones ack immediately).
      for (;;) {
        const done = Atomics.load(ctl, CTL_DONE);
        if (done === numWorkers) break;
        if (waitAsync) {
          const r = waitAsync(ctl, CTL_DONE, done);
          if (r.async) await r.value;
        } else {
          await new Promise((res) => setTimeout(res, 0)); // poll fallback
        }
      }

      const result = new Float32Array(hiddenSize);
      for (let w = 0; w < numWorkers; w++) {
        if (reqCount[w] === 0) continue;
        const part = out.subarray(w * hiddenSize, (w + 1) * hiddenSize);
        for (let i = 0; i < hiddenSize; i++) result[i] += part[i];
      }
      return result;
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

  return { computeExperts, destroy, numWorkers, expertsPerWorker };
}
