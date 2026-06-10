/**
 * MoE expert worker — owns a contiguous shard of routed experts for ALL
 * layers (default 32 of 256 experts × 40 layers ≈ 2.78 GB Q5_K/Q6_K) inside
 * one wasm32 linear memory, fetched at init via HTTP Range from the GGUF.
 *
 * Decode loop (SAB + Atomics, see moe-cpu.ts for the layout):
 *   wait GEN ≠ lastGen → read LAYER + my request slice → per local expert:
 *   gate GEMV → up GEMV → JS silu(gate)⊙up → wasm q8_quantize → down GEMV
 *   (Q5_K or Q6_K per-tensor) → weighted accumulate into my SAB outbox →
 *   Atomics.add(DONE) + notify. GEN = -1 means shutdown.
 *
 * CAUTION: pointers exceed 2 GB — all address math uses arithmetic (%, /),
 * NEVER JS bitwise ops (signed 32-bit). Passing >2^31 addresses to wasm
 * exports is fine (ToInt32 wraps; the u32 bit pattern is preserved).
 */

import { GGML_TYPES } from '../model/gguf';
import {
  CTL_GEN,
  CTL_DONE,
  CTL_LAYER,
  MOE_MAX_K,
  Q8_BLOCK_BYTES,
  sabLayout,
  type MoEWorkerInitMsg,
  type ExpertSlabDesc,
} from './moe-cpu';

interface WasmExports {
  memory: WebAssembly.Memory;
  __heap_base: WebAssembly.Global;
  q8_quantize(xPtr: number, yPtr: number, n: number): void;
  q5k_gemv(wPtr: number, xPtr: number, yPtr: number, rows: number, cols: number): void;
  q6k_gemv(wPtr: number, xPtr: number, yPtr: number, rows: number, cols: number): void;
}

const PAGE = 65536;

/** 16-align without bitwise ops (pointers exceed 2^31). */
function align16(p: number): number {
  return p + ((16 - (p % 16)) % 16);
}

function post(msg: unknown): void {
  (self as unknown as Worker).postMessage(msg);
}

async function fetchInto(url: string, start: number, len: number, dst: Uint8Array): Promise<void> {
  const resp = await fetch(url, { headers: { Range: `bytes=${start}-${start + len - 1}` } });
  if (!resp.ok) throw new Error(`Range fetch failed: HTTP ${resp.status} at ${start}+${len}`);
  const buf = await resp.arrayBuffer();
  // feedback_arraybuffer_slice: a 200 (full-file) response or short read must
  // never be silently accepted — assert the exact byte count.
  if (buf.byteLength !== len) {
    throw new Error(`Range fetch size mismatch at ${start}: got ${buf.byteLength}, want ${len}`);
  }
  dst.set(new Uint8Array(buf));
}

async function run(init: MoEWorkerInitMsg): Promise<void> {
  const { workerId, url, sab, expertStart, expertCount, hiddenSize, ffnDim, numWorkers, layers } = init;

  const resp = await fetch('/wasm/q5k_gemv.wasm');
  if (!resp.ok) throw new Error(`failed to fetch wasm: ${resp.status}`);
  const { instance } = await WebAssembly.instantiate(await resp.arrayBuffer(), {});
  const wasm = instance.exports as unknown as WasmExports;

  // ── Plan the linear-memory layout: all slabs, then scratch ──
  // slabPtrs[layer][0|1|2] = gate|up|down base for THIS worker's experts.
  let p = align16(wasm.__heap_base.value as number);
  const slabPtrs: number[][] = [];
  for (const L of layers) {
    const ptrs: number[] = [];
    for (const slab of [L.gate, L.up, L.down]) {
      ptrs.push(p);
      p = align16(p + slab.bytesPerExpert * expertCount);
    }
    slabPtrs.push(ptrs);
  }
  const xq8Ptr = p;                                          // hidden as Q8 blocks
  p = align16(p + (hiddenSize / 256) * Q8_BLOCK_BYTES);
  const gateOutPtr = p; p = align16(p + ffnDim * 4);
  const upOutPtr = p; p = align16(p + ffnDim * 4);
  const act2Ptr = p; p = align16(p + ffnDim * 4);            // silu(gate)⊙up f32
  const act2Q8Ptr = p; p = align16(p + (ffnDim / 256) * Q8_BLOCK_BYTES);
  const downOutPtr = p; p = align16(p + hiddenSize * 4);
  const end = p;

  const mem = wasm.memory;
  const needPages = Math.ceil(end / PAGE) - mem.buffer.byteLength / PAGE;
  if (needPages > 0) mem.grow(needPages);
  // mem.buffer is replaced by grow — take views AFTER growth, never before.
  const heap = new Uint8Array(mem.buffer);

  // ── Fetch this worker's expert shard (contiguous range per tensor) ──
  const t0 = performance.now();
  let loadedBytes = 0;
  let totalBytes = 0;
  for (const L of layers) for (const s of [L.gate, L.up, L.down]) totalBytes += s.bytesPerExpert * expertCount;
  for (let l = 0; l < layers.length; l++) {
    const L = layers[l];
    const kinds: ExpertSlabDesc[] = [L.gate, L.up, L.down];
    for (let k = 0; k < 3; k++) {
      const slab = kinds[k];
      const len = slab.bytesPerExpert * expertCount;
      const fileStart = slab.offset + slab.bytesPerExpert * expertStart;
      await fetchInto(url, fileStart, len, heap.subarray(slabPtrs[l][k], slabPtrs[l][k] + len));
      loadedBytes += len;
    }
    post({ cmd: 'progress', workerId, loadedBytes, totalBytes });
  }
  post({ cmd: 'ready', workerId, totalBytes, loadMs: performance.now() - t0 });

  // ── SAB views ──
  const layout = sabLayout(numWorkers, hiddenSize);
  if (layout.totalBytes !== sab.byteLength) {
    throw new Error(`SAB size ${sab.byteLength} ≠ layout ${layout.totalBytes} — layout mismatch with main thread`);
  }
  const ctl = new Int32Array(sab, 0, 4);
  const reqCount = new Int32Array(sab, layout.reqCountOff, numWorkers);
  const reqExpert = new Int32Array(sab, layout.reqExpertOff, numWorkers * MOE_MAX_K);
  const reqWeight = new Float32Array(sab, layout.reqWeightOff, numWorkers * MOE_MAX_K);
  const xq8Shared = new Uint8Array(sab, layout.xq8Off, layout.xq8Bytes);
  const myOut = new Float32Array(sab, layout.outOff + workerId * hiddenSize * 4, hiddenSize);

  const gateOut = new Float32Array(mem.buffer, gateOutPtr, ffnDim);
  const upOut = new Float32Array(mem.buffer, upOutPtr, ffnDim);
  const act2 = new Float32Array(mem.buffer, act2Ptr, ffnDim);
  const downOut = new Float32Array(mem.buffer, downOutPtr, hiddenSize);
  const xq8Heap = heap.subarray(xq8Ptr, xq8Ptr + layout.xq8Bytes);

  // ── Decode loop ──
  let lastGen = 0;
  for (;;) {
    Atomics.wait(ctl, CTL_GEN, lastGen);
    const gen = Atomics.load(ctl, CTL_GEN);
    if (gen === -1) return;
    if (gen === lastGen) continue;
    lastGen = gen;

    const n = Atomics.load(reqCount, workerId);
    if (n > 0) {
      const layer = Atomics.load(ctl, CTL_LAYER);
      const L = layers[layer];
      xq8Heap.set(xq8Shared);
      myOut.fill(0);

      const gemv = (slab: ExpertSlabDesc, wPtr: number, xPtr: number, yPtr: number) =>
        slab.ggmlType === GGML_TYPES.Q6_K
          ? wasm.q6k_gemv(wPtr, xPtr, yPtr, slab.rows, slab.cols)
          : wasm.q5k_gemv(wPtr, xPtr, yPtr, slab.rows, slab.cols);

      for (let k = 0; k < n; k++) {
        const local = reqExpert[workerId * MOE_MAX_K + k];
        const weight = reqWeight[workerId * MOE_MAX_K + k];

        gemv(L.gate, slabPtrs[layer][0] + L.gate.bytesPerExpert * local, xq8Ptr, gateOutPtr);
        gemv(L.up, slabPtrs[layer][1] + L.up.bytesPerExpert * local, xq8Ptr, upOutPtr);

        for (let i = 0; i < ffnDim; i++) {
          const g = gateOut[i];
          act2[i] = (g / (1 + Math.exp(-g))) * upOut[i]; // silu(gate) ⊙ up
        }
        wasm.q8_quantize(act2Ptr, act2Q8Ptr, ffnDim);

        gemv(L.down, slabPtrs[layer][2] + L.down.bytesPerExpert * local, act2Q8Ptr, downOutPtr);
        for (let i = 0; i < hiddenSize; i++) myOut[i] += weight * downOut[i];
      }
    }

    Atomics.add(ctl, CTL_DONE, 1);
    Atomics.notify(ctl, CTL_DONE);
  }
}

self.onmessage = (e: MessageEvent) => {
  const msg = e.data;
  if (msg.cmd === 'init') {
    run(msg as MoEWorkerInitMsg).catch((err) => {
      post({ cmd: 'error', workerId: msg.workerId, message: err instanceof Error ? err.message : String(err) });
    });
  }
};
