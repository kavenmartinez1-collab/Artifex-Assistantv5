/**
 * MoE expert worker — ROW-SPLIT: owns a 1/NW row-strip of ALL experts for
 * all layers (gate/up rows [w·ffnDim/NW, …), down rows [w·H/NW, …); default
 * 16 workers ≈ 1.39 GB each) inside one wasm32 linear memory, fetched at
 * init via the dev-server strided-gather endpoint (one request per
 * layer-tensor returns this worker's strip of every expert, concatenated).
 *
 * Decode generation (SAB + Atomics, see moe-cpu.ts for the layout):
 *   wait GEN ≠ lastGen → read LAYER + the shared request list → phase A:
 *   per routed expert, gate/up strip GEMVs + JS silu(gate)⊙up → act2 row
 *   slice into the SAB exchange area → DONE_A barrier (workers wait on each
 *   other) → phase B: per routed expert, copy full act2 from SAB → wasm
 *   q8_quantize → down strip GEMV → weighted accumulate into this worker's
 *   DISJOINT out[w·H/NW …) slice → Atomics.add(DONE), last worker notifies.
 *   GEN = -1 means shutdown.
 *
 * Every worker does identical work (1/NW of each routed expert) — no
 * straggler from routing imbalance, unlike the old expert-ownership layout.
 *
 * CAUTION: pointers can exceed 2 GB — all address math uses arithmetic
 * (%, /), NEVER JS bitwise ops (signed 32-bit). Passing >2^31 addresses to
 * wasm exports is fine (ToInt32 wraps; the u32 bit pattern is preserved).
 */

import { GGML_TYPES } from '../model/gguf';
import {
  CTL_GEN,
  CTL_DONE,
  CTL_DONE_A,
  CTL_LAYER,
  TOUCH_LAYER,
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

/** 16-align without bitwise ops (pointers can exceed 2^31). */
function align16(p: number): number {
  return p + ((16 - (p % 16)) % 16);
}

function post(msg: unknown): void {
  (self as unknown as Worker).postMessage(msg);
}

/** Strided-gather fetch: this worker's strip of every expert in one tensor. */
async function gatherInto(
  url: string,
  slab: ExpertSlabDesc,
  workerId: number,
  numWorkers: number,
  numExperts: number,
  dst: Uint8Array,
): Promise<void> {
  const stripBytes = slab.bytesPerExpert / numWorkers;
  const sep = url.includes('?') ? '&' : '?';
  const gurl = `${url}${sep}gatherStart=${slab.offset + workerId * stripBytes}`
    + `&gatherStride=${slab.bytesPerExpert}&gatherChunk=${stripBytes}&gatherCount=${numExperts}`;
  const resp = await fetch(gurl);
  if (!resp.ok) throw new Error(`gather fetch failed: HTTP ${resp.status} at ${slab.offset}`);
  const buf = await resp.arrayBuffer();
  // feedback_arraybuffer_slice: a short read must never be silently accepted —
  // assert the exact byte count.
  const want = stripBytes * numExperts;
  if (buf.byteLength !== want) {
    throw new Error(`gather size mismatch at ${slab.offset}: got ${buf.byteLength}, want ${want}`);
  }
  dst.set(new Uint8Array(buf));
}

async function run(init: MoEWorkerInitMsg): Promise<void> {
  const { workerId, url, sab, numExperts, hiddenSize, ffnDim, numWorkers, layers } = init;
  const rowsA = ffnDim / numWorkers;     // gate/up strip rows
  const rowsB = hiddenSize / numWorkers; // down strip rows

  const resp = await fetch('/wasm/q5k_gemv.wasm');
  if (!resp.ok) throw new Error(`failed to fetch wasm: ${resp.status}`);
  const { instance } = await WebAssembly.instantiate(await resp.arrayBuffer(), {});
  const wasm = instance.exports as unknown as WasmExports;

  // ── Plan the linear-memory layout: all strip regions, then scratch ──
  // stripPtrs[layer][0|1|2] = gate|up|down strip-region base (expert-major:
  // expert e's strip at base + e * bytesPerExpert/NW).
  let p = align16(wasm.__heap_base.value as number);
  const stripPtrs: number[][] = [];
  for (const L of layers) {
    const ptrs: number[] = [];
    for (const slab of [L.gate, L.up, L.down]) {
      ptrs.push(p);
      p = align16(p + (slab.bytesPerExpert / numWorkers) * numExperts);
    }
    stripPtrs.push(ptrs);
  }
  const xq8Ptr = p;                                          // hidden as Q8 blocks
  p = align16(p + (hiddenSize / 256) * Q8_BLOCK_BYTES);
  const gateOutPtr = p; p = align16(p + rowsA * 4);
  const upOutPtr = p; p = align16(p + rowsA * 4);
  const act2Ptr = p; p = align16(p + ffnDim * 4);            // full act2 (from SAB) f32
  const act2Q8Ptr = p; p = align16(p + (ffnDim / 256) * Q8_BLOCK_BYTES);
  const downOutPtr = p; p = align16(p + rowsB * 4);
  const end = p;

  const mem = wasm.memory;
  const needPages = Math.ceil(end / PAGE) - mem.buffer.byteLength / PAGE;
  if (needPages > 0) mem.grow(needPages);
  // mem.buffer is replaced by grow — take views AFTER growth, never before.
  const heap = new Uint8Array(mem.buffer);

  // ── Fetch this worker's row-strips (one gather request per tensor) ──
  const t0 = performance.now();
  let loadedBytes = 0;
  let totalBytes = 0;
  for (const L of layers) for (const s of [L.gate, L.up, L.down]) {
    totalBytes += (s.bytesPerExpert / numWorkers) * numExperts;
  }
  for (let l = 0; l < layers.length; l++) {
    const L = layers[l];
    const kinds: ExpertSlabDesc[] = [L.gate, L.up, L.down];
    for (let k = 0; k < 3; k++) {
      const slab = kinds[k];
      const len = (slab.bytesPerExpert / numWorkers) * numExperts;
      await gatherInto(url, slab, workerId, numWorkers, numExperts,
        heap.subarray(stripPtrs[l][k], stripPtrs[l][k] + len));
      loadedBytes += len;
    }
    post({ cmd: 'progress', workerId, loadedBytes, totalBytes });
  }
  post({ cmd: 'ready', workerId, totalBytes, loadMs: performance.now() - t0 });

  // ── SAB views ──
  const layout = sabLayout(numWorkers, hiddenSize, ffnDim);
  if (layout.totalBytes !== sab.byteLength) {
    throw new Error(`SAB size ${sab.byteLength} ≠ layout ${layout.totalBytes} — layout mismatch with main thread`);
  }
  const ctl = new Int32Array(sab, 0, 4);
  const reqCount = new Int32Array(sab, layout.reqCountOff, 1);
  const reqExpert = new Int32Array(sab, layout.reqExpertOff, MOE_MAX_K);
  const reqWeight = new Float32Array(sab, layout.reqWeightOff, MOE_MAX_K);
  const xq8Shared = new Uint8Array(sab, layout.xq8Off, layout.xq8Bytes);
  const act2Shared = new Float32Array(sab, layout.act2Off, MOE_MAX_K * ffnDim);
  const myOut = new Float32Array(sab, layout.outOff + workerId * rowsB * 4, rowsB);
  const perfBusy = new Float32Array(sab, layout.perfOff, numWorkers);

  const gateOut = new Float32Array(mem.buffer, gateOutPtr, rowsA);
  const upOut = new Float32Array(mem.buffer, upOutPtr, rowsA);
  const act2 = new Float32Array(mem.buffer, act2Ptr, ffnDim);
  const downOut = new Float32Array(mem.buffer, downOutPtr, rowsB);
  const xq8Heap = heap.subarray(xq8Ptr, xq8Ptr + layout.xq8Bytes);

  const gemvStrip = (slab: ExpertSlabDesc, regionBase: number, e: number, xPtr: number, yPtr: number, rows: number) => {
    const wPtr = regionBase + (slab.bytesPerExpert / numWorkers) * e;
    if (slab.ggmlType === GGML_TYPES.Q6_K) wasm.q6k_gemv(wPtr, xPtr, yPtr, rows, slab.cols);
    else wasm.q5k_gemv(wPtr, xPtr, yPtr, rows, slab.cols);
  };

  // ── Decode loop ──
  let lastGen = 0;
  for (;;) {
    Atomics.wait(ctl, CTL_GEN, lastGen);
    const gen = Atomics.load(ctl, CTL_GEN);
    if (gen === -1) return;
    if (gen === lastGen) continue;
    lastGen = gen;
    const tBusy0 = performance.now();
    const layer = Atomics.load(ctl, CTL_LAYER);

    if (layer === TOUCH_LAYER) {
      // Wave bounds ride in reqExpert[0..1] (unused by touch generations):
      // workers outside [start, end) check in immediately. Warming in waves
      // serializes disk traffic — all-NW passes thrash the pagefile when
      // commit is near RAM capacity (observed 0.01 GB/s at 32 workers).
      const waveStart = reqExpert[0], waveEnd = reqExpert[1];
      if (workerId < waveStart || workerId >= waveEnd) {
        perfBusy[workerId] = 0;
        if (Atomics.add(ctl, CTL_DONE, 1) + 1 === numWorkers) {
          Atomics.notify(ctl, CTL_DONE);
        }
        continue;
      }
      // Paging/bandwidth probe: stream the WHOLE strip set sequentially via
      // f64 reads (region ptrs are 16-aligned; strip bytes are 8-divisible).
      // No DONE_A barrier here — touch generations are single-phase.
      let acc = 0;
      for (let l = 0; l < layers.length; l++) {
        const Lx = layers[l];
        const kinds: ExpertSlabDesc[] = [Lx.gate, Lx.up, Lx.down];
        for (let k = 0; k < 3; k++) {
          const len = (kinds[k].bytesPerExpert / numWorkers) * numExperts;
          const f64 = new Float64Array(mem.buffer, stripPtrs[l][k], len / 8);
          for (let i = 0; i < f64.length; i++) acc += f64[i];
        }
      }
      // Publish acc so the sum can't be dead-code-eliminated. Main never
      // reads touch output and the compute path re-zeroes myOut.
      myOut[rowsB - 1] = acc;
      perfBusy[workerId] = performance.now() - tBusy0;
      if (Atomics.add(ctl, CTL_DONE, 1) + 1 === numWorkers) {
        Atomics.notify(ctl, CTL_DONE);
      }
      continue;
    }

    const n = Atomics.load(reqCount, 0);
    const L = layers[layer];

    // ── Phase A: gate/up strips + silu⊙ → act2 slices into SAB ──
    if (n > 0) {
      xq8Heap.set(xq8Shared);
      for (let k = 0; k < n; k++) {
        const e = reqExpert[k];
        gemvStrip(L.gate, stripPtrs[layer][0], e, xq8Ptr, gateOutPtr, rowsA);
        gemvStrip(L.up, stripPtrs[layer][1], e, xq8Ptr, upOutPtr, rowsA);
        const base = k * ffnDim + workerId * rowsA;
        for (let i = 0; i < rowsA; i++) {
          const g = gateOut[i];
          act2Shared[base + i] = (g / (1 + Math.exp(-g))) * upOut[i]; // silu(gate) ⊙ up
        }
      }
    }

    // ── DONE_A barrier: act2 writes above happen-before the add; reading
    // DONE_A == NW synchronizes-with every other worker's add. ──
    if (Atomics.add(ctl, CTL_DONE_A, 1) + 1 !== numWorkers) {
      let v;
      while ((v = Atomics.load(ctl, CTL_DONE_A)) !== numWorkers) {
        Atomics.wait(ctl, CTL_DONE_A, v);
      }
    } else {
      Atomics.notify(ctl, CTL_DONE_A);
    }

    // ── Phase B: full act2 from SAB → Q8 → down strip → weighted out slice ──
    if (n > 0) {
      myOut.fill(0);
      for (let k = 0; k < n; k++) {
        const e = reqExpert[k];
        const weight = reqWeight[k];
        act2.set(act2Shared.subarray(k * ffnDim, (k + 1) * ffnDim));
        wasm.q8_quantize(act2Ptr, act2Q8Ptr, ffnDim);
        gemvStrip(L.down, stripPtrs[layer][2], e, act2Q8Ptr, downOutPtr, rowsB);
        for (let i = 0; i < rowsB; i++) myOut[i] += weight * downOut[i];
      }
    }

    // Busy-time probe (read by main after DONE==numWorkers; the Atomics.add
    // below publishes it). Idle generations record ~0.
    perfBusy[workerId] = performance.now() - tBusy0;
    // Only the LAST worker notifies — main polls DONE via MessageChannel
    // yields, and per-increment notifies just add scheduling churn.
    if (Atomics.add(ctl, CTL_DONE, 1) + 1 === numWorkers) {
      Atomics.notify(ctl, CTL_DONE);
    }
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
