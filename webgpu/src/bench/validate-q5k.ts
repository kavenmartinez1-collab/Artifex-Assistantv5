/**
 * Node-side validation of the wasm32 SIMD Q5_K GEMV kernel vs the JS
 * reference (kernel-audit rule). Run: npx tsx src/bench/validate-q5k.ts
 */

import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import {
  QK_K,
  Q5K_BLOCK_BYTES,
  Q8_ACT_BLOCK_BYTES,
  q5kDotRowRef,
  q8Quantize,
  xorshiftFill,
  stampBlockScales,
} from './q5k-ref';

const here = dirname(fileURLToPath(import.meta.url));
const wasmPath = join(here, '../../public/wasm/q5k_gemv.wasm');

const ROWS = 64;
const COLS = 2048;

async function main() {
  const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath), {});
  const ex = instance.exports as any;
  const mem: WebAssembly.Memory = ex.memory;

  const nbRow = COLS / QK_K;
  const rowBytes = nbRow * Q5K_BLOCK_BYTES;
  const heapBase = ex.__heap_base.value as number;
  const align = (p: number) => (p + 15) & ~15;
  const wPtr = align(heapBase);
  const q8Ptr = align(wPtr + ROWS * rowBytes);
  const xPtr = align(q8Ptr + nbRow * Q8_ACT_BLOCK_BYTES);
  const yPtr = align(xPtr + COLS * 4);
  const end = yPtr + ROWS * 4;
  const need = Math.ceil(end / 65536) - mem.buffer.byteLength / 65536;
  if (need > 0) mem.grow(need);

  const weights = new Uint8Array(mem.buffer, wPtr, ROWS * rowBytes);
  xorshiftFill(weights, 0xdeadbeef);
  stampBlockScales(weights);

  const x = new Float32Array(mem.buffer, xPtr, COLS);
  for (let i = 0; i < COLS; i++) {
    // deterministic pseudo-random in [-1, 1)
    const s = Math.sin(i * 12.9898) * 43758.5453;
    x[i] = (s - Math.floor(s)) * 2 - 1;
  }

  ex.q8_quantize(xPtr, q8Ptr, COLS);
  ex.q5k_gemv(wPtr, q8Ptr, yPtr, ROWS, COLS);
  const y = new Float32Array(mem.buffer, yPtr, ROWS);

  // Cross-check the wasm q8 quantization against the JS reference first.
  const refAct = q8Quantize(x.slice());
  const wasmQ8 = new Int8Array(mem.buffer, q8Ptr + 4, QK_K); // block 0: d(4) then q[256]
  let q8Mismatches = 0;
  for (let j = 0; j < QK_K; j++) {
    if (wasmQ8[j] !== refAct.q[j]) q8Mismatches++;
  }

  let maxRelErr = 0;
  let worstRow = -1;
  for (let r = 0; r < ROWS; r++) {
    const rowBuf = new Uint8Array(mem.buffer, wPtr + r * rowBytes, rowBytes);
    const ref = q5kDotRowRef(rowBuf, refAct, nbRow);
    const rel = Math.abs(y[r] - ref) / Math.max(1e-6, Math.abs(ref));
    if (rel > maxRelErr) {
      maxRelErr = rel;
      worstRow = r;
    }
  }

  console.log(`q8 quantization mismatches (block 0): ${q8Mismatches}/256`);
  console.log(`GEMV maxRelErr over ${ROWS} rows: ${maxRelErr.toExponential(3)} (worst row ${worstRow})`);
  if (q8Mismatches > 0 || maxRelErr >= 1e-4) {
    console.error('VALIDATION FAILED');
    process.exit(1);
  }
  console.log('VALIDATION PASSED');
}

main();
