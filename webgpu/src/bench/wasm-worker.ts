/**
 * Bench worker: hosts one wasm32 SIMD Q5_K GEMV kernel instance with its own
 * linear memory and a private slice of synthetic expert weights.
 *
 * Protocol (postMessage):
 *   {cmd:'init', workerId, weightsMB, rows, cols}
 *     → {cmd:'init-done', ok, maxRelErr, nMatrices} (after self-validation vs JS ref)
 *   {cmd:'run', ms}
 *     → {cmd:'run-done', bytes, elapsedMs, gemvs}
 *   {cmd:'wake', sab}  — enter Atomics round-trip loop (main measures latency)
 */

import {
  QK_K,
  Q5K_BLOCK_BYTES,
  Q8_ACT_BLOCK_BYTES,
  q5kDotRowRef,
  q8Quantize,
  xorshiftFill,
  stampBlockScales,
} from './q5k-ref';

interface WasmExports {
  memory: WebAssembly.Memory;
  __heap_base: WebAssembly.Global;
  q8_quantize(xPtr: number, yPtr: number, n: number): void;
  q5k_gemv(wPtr: number, xPtr: number, yPtr: number, rows: number, cols: number): void;
}

let exports_: WasmExports | null = null;
let weightsPtr = 0;
let q8Ptr = 0;
let xPtr = 0;
let yPtr = 0;
let rows_ = 0;
let cols_ = 0;
let matrixBytes = 0;
let nMatrices = 0;

const PAGE = 65536;

function align(p: number, a: number): number {
  return (p + a - 1) & ~(a - 1);
}

async function init(workerId: number, weightsMB: number, rows: number, cols: number) {
  const resp = await fetch('/wasm/q5k_gemv.wasm');
  if (!resp.ok) throw new Error(`failed to fetch wasm: ${resp.status}`);
  const { instance } = await WebAssembly.instantiate(await resp.arrayBuffer(), {});
  exports_ = instance.exports as unknown as WasmExports;

  rows_ = rows;
  cols_ = cols;
  const nbRow = cols / QK_K;
  const rowBytes = nbRow * Q5K_BLOCK_BYTES;
  matrixBytes = rows * rowBytes;
  const weightsBytes = weightsMB * 1024 * 1024;
  nMatrices = Math.floor(weightsBytes / matrixBytes);
  if (nMatrices < 1) throw new Error('weightsMB too small for one matrix');

  // Memory layout from __heap_base: weights | q8 acts | x f32 | y f32
  const heapBase = exports_.__heap_base.value as number;
  weightsPtr = align(heapBase, 16);
  q8Ptr = align(weightsPtr + nMatrices * matrixBytes, 16);
  xPtr = align(q8Ptr + nbRow * Q8_ACT_BLOCK_BYTES, 16);
  yPtr = align(xPtr + cols * 4, 16);
  const end = yPtr + rows * 4;

  const mem = exports_.memory;
  const needPages = Math.ceil(end / PAGE) - mem.buffer.byteLength / PAGE;
  if (needPages > 0) mem.grow(needPages);

  // Fill weights deterministically; stamp sane f16 block scales.
  const weights = new Uint8Array(mem.buffer, weightsPtr, nMatrices * matrixBytes);
  xorshiftFill(weights, 0xc0ffee01 + workerId * 7919);
  stampBlockScales(weights);

  // Activations in [-1, 1).
  const x = new Float32Array(mem.buffer, xPtr, cols);
  const xb = new Uint8Array(mem.buffer, xPtr, cols * 4);
  xorshiftFill(xb, 0xabad1dea + workerId);
  for (let i = 0; i < cols; i++) {
    const dv = new DataView(mem.buffer, xPtr + i * 4, 4);
    x[i] = (dv.getUint32(0, true) / 0xffffffff) * 2 - 1;
  }

  exports_.q8_quantize(xPtr, q8Ptr, cols);

  // ── Self-validation: wasm GEMV vs JS reference on first 4 rows ──
  exports_.q5k_gemv(weightsPtr, q8Ptr, yPtr, rows, cols);
  const y = new Float32Array(mem.buffer, yPtr, rows);
  const act = q8Quantize(x.slice());
  let maxRelErr = 0;
  for (let r = 0; r < Math.min(4, rows); r++) {
    const rowBuf = new Uint8Array(mem.buffer, weightsPtr + r * rowBytes, rowBytes);
    const ref = q5kDotRowRef(rowBuf, act, nbRow);
    const got = y[r];
    const rel = Math.abs(got - ref) / Math.max(1e-6, Math.abs(ref));
    if (rel > maxRelErr) maxRelErr = rel;
  }
  return { ok: maxRelErr < 1e-4, maxRelErr, nMatrices };
}

function run(ms: number) {
  if (!exports_) throw new Error('not initialized');
  let gemvs = 0;
  let i = 0;
  const t0 = performance.now();
  let elapsed = 0;
  // Walk matrices in a non-sequential order so the working set defeats caches.
  do {
    const m = Number((BigInt(i) * 2654435761n) % BigInt(nMatrices));
    exports_.q5k_gemv(weightsPtr + m * matrixBytes, q8Ptr, yPtr, rows_, cols_);
    gemvs++;
    i++;
    elapsed = performance.now() - t0;
  } while (elapsed < ms);
  return { bytes: gemvs * matrixBytes, elapsedMs: elapsed, gemvs };
}

function wakeLoop(sab: SharedArrayBuffer) {
  // Round-trip: main sets [0]=generation and notifies; we echo to [1] and notify.
  const ctl = new Int32Array(sab);
  for (;;) {
    Atomics.wait(ctl, 0, Atomics.load(ctl, 1));
    const gen = Atomics.load(ctl, 0);
    if (gen < 0) return; // shutdown
    Atomics.store(ctl, 1, gen);
    Atomics.notify(ctl, 1);
  }
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  try {
    if (msg.cmd === 'init') {
      const r = await init(msg.workerId, msg.weightsMB, msg.rows, msg.cols);
      self.postMessage({ cmd: 'init-done', ...r });
    } else if (msg.cmd === 'run') {
      self.postMessage({ cmd: 'run-done', ...run(msg.ms) });
    } else if (msg.cmd === 'wake') {
      wakeLoop(msg.sab);
      self.postMessage({ cmd: 'wake-done' });
    }
  } catch (err) {
    self.postMessage({ cmd: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
