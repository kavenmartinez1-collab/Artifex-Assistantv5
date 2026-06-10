/**
 * Phase C1 node test — expert slab indexing + wasm GEMV correctness,
 * no browser needed. Run: npx tsx scripts/test-expert-ffn.mjs
 *
 * 1. Parses the real 35B-A3B GGUF from disk and reads individual expert
 *    slabs of ffn_{gate,up,down}_exps at computed per-expert offsets.
 * 2. Runs the wasm q5k_gemv/q6k_gemv kernels on them and compares against
 *    (a) the integer-exact JS reference (q5kDotRowRef / inline Q6_K ref)
 *    (b) a full f64 dot over dequantized weights (dequantQ5_K/dequantQ6_K).
 * 3. Re-runs the same GEMV with the weights placed at a ~3 GB pointer after
 *    growing the memory — validates the 4 GB max-memory build and that
 *    >2^31 addresses survive the JS→wasm i32 boundary bit-exactly.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseGGUF, ggmlTypeTraits, GGML_TYPES } from '../src/model/gguf.ts';
import { dequantQ5_K, dequantQ6_K } from '../src/model/gguf-dequant.ts';
import { q8Quantize, q5kDotRowRef, f16ToF32, QK_K } from '../src/bench/q5k-ref.ts';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const GGUF_PATH = process.env.GGUF_PATH
  || path.resolve(__dirname, '../../models/qwen3.6-35b-a3b-gguf/Qwen3.6-35B-A3B-UD-Q5_K_S.gguf');
const WASM_PATH = path.resolve(__dirname, '../public/wasm/q5k_gemv.wasm');

const PAGE = 65536;
const Q8_BLOCK_BYTES = 276;
const HIGH_PTR_TARGET = 3_000_000_000; // > 2^31, < 4 GB

// ── helpers ────────────────────────────────────────────────────────────

function align16(p) {
  return p + ((16 - (p % 16)) % 16);
}

/** Integer-exact Q6_K row dot vs Q8 acts — mirrors the wasm kernel. */
function q6kDotRowRef(row, act, nb) {
  let sumf = 0;
  for (let i = 0; i < nb; i++) {
    const o = i * 210;
    const dv = new DataView(row.buffer, row.byteOffset + o, 210);
    const d = Math.fround(f16ToF32(dv.getUint16(208, true)) * act.d[i]);
    let isum = 0;
    for (let n = 0; n < 2; n++) {
      const qlO = o + n * 64;
      const qhO = o + 128 + n * 32;
      const scO = o + 192 + n * 8;
      const q8Base = i * QK_K + n * 128;
      for (let l = 0; l < 32; l++) {
        const ql = row[qlO + l];
        const ql32 = row[qlO + l + 32];
        const qh = row[qhO + l];
        const q1 = ((ql & 0xf) | (((qh >> 0) & 3) << 4)) - 32;
        const q2 = ((ql32 & 0xf) | (((qh >> 2) & 3) << 4)) - 32;
        const q3 = ((ql >> 4) | (((qh >> 4) & 3) << 4)) - 32;
        const q4 = ((ql32 >> 4) | (((qh >> 6) & 3) << 4)) - 32;
        const si = (j) => (row[scO + j] << 24) >> 24; // int8
        isum += si(Math.floor(l / 16) + 0) * q1 * act.q[q8Base + l]
              + si(Math.floor(l / 16) + 2) * q2 * act.q[q8Base + l + 32]
              + si(Math.floor(l / 16) + 4) * q3 * act.q[q8Base + l + 64]
              + si(Math.floor(l / 16) + 6) * q4 * act.q[q8Base + l + 96];
      }
    }
    sumf = Math.fround(sumf + Math.fround(d * isum));
  }
  return sumf;
}

function seededRandFloats(n, seed) {
  const out = new Float32Array(n);
  let s = seed >>> 0 || 0x9e3779b9;
  for (let i = 0; i < n; i++) {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    out[i] = (s / 0xffffffff) * 2 - 1;
  }
  return out;
}

// ── load GGUF header + wasm ────────────────────────────────────────────

const fd = fs.openSync(GGUF_PATH, 'r');
const readRange = async (start, end) => {
  const buf = Buffer.alloc(end - start);
  const got = fs.readSync(fd, buf, 0, end - start, start);
  if (got !== end - start) throw new Error(`short read at ${start}: ${got}/${end - start}`);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + got);
};

const file = await parseGGUF(readRange);
console.log(`parsed ${path.basename(GGUF_PATH)}: ${file.tensorCount} tensors`);

const { instance } = await WebAssembly.instantiate(fs.readFileSync(WASM_PATH), {});
const wasm = instance.exports;
const mem = wasm.memory;

// Memory plan: low scratch + low weights, then a >2^31 high weights region.
let p = align16(wasm.__heap_base.value);
const xPtr = p; p = align16(p + 2048 * 4);
const xq8Ptr = p; p = align16(p + (2048 / QK_K) * Q8_BLOCK_BYTES);
const yPtr = p; p = align16(p + 2048 * 4);
const yPtr2 = p; p = align16(p + 2048 * 4);
const wLowPtr = p; p = align16(p + 1_000_000);
const wHighPtr = align16(HIGH_PTR_TARGET);
const end = wHighPtr + 1_000_000;

const needPages = Math.ceil(end / PAGE) - mem.buffer.byteLength / PAGE;
const t0 = performance.now();
if (needPages > 0) mem.grow(needPages);
console.log(`grew wasm memory to ${(mem.buffer.byteLength / 2 ** 30).toFixed(2)} GiB in ${(performance.now() - t0).toFixed(0)} ms (high ptr ${wHighPtr} > 2^31: ${wHighPtr > 2 ** 31})`);

// ── test cases: real expert slabs ──────────────────────────────────────

const cases = [
  { tensor: 'blk.0.ffn_gate_exps.weight', experts: [0, 17, 255] },
  { tensor: 'blk.0.ffn_up_exps.weight', experts: [31] },
  { tensor: 'blk.0.ffn_down_exps.weight', experts: [0, 200] },
  { tensor: 'blk.34.ffn_down_exps.weight', experts: [0, 99, 255] }, // Q6_K bump layer
  { tensor: 'blk.39.ffn_gate_exps.weight', experts: [128] },
];

let failures = 0;

for (const c of cases) {
  const t = file.tensors.get(c.tensor);
  if (!t) throw new Error(`tensor ${c.tensor} not found`);
  const [cols, rows, nExperts] = t.ne;
  const { blockSize, typeSize } = ggmlTypeTraits(t.ggmlType);
  const bytesPerExpert = rows * (cols / blockSize) * typeSize;
  if (bytesPerExpert * nExperts !== t.byteLength) {
    throw new Error(`${c.tensor}: slab math ${bytesPerExpert}×${nExperts} ≠ ${t.byteLength}`);
  }
  const isQ6 = t.ggmlType === GGML_TYPES.Q6_K;
  const gemv = isQ6 ? wasm.q6k_gemv : wasm.q5k_gemv;
  const dequant = isQ6 ? dequantQ6_K : dequantQ5_K;
  const nb = cols / QK_K;
  const rowBytes = (cols / blockSize) * typeSize;

  for (const e of c.experts) {
    // Read the expert slab straight from disk at the computed offset.
    const slab = Buffer.alloc(bytesPerExpert);
    const got = fs.readSync(fd, slab, 0, bytesPerExpert, t.offset + e * bytesPerExpert);
    if (got !== bytesPerExpert) throw new Error('short slab read');
    const slabU8 = new Uint8Array(slab.buffer, slab.byteOffset, bytesPerExpert);

    // Random activations → Q8 (wasm), mirrored in JS for the references.
    const x = seededRandFloats(cols, 0xc0ffee + e * 7919);
    new Float32Array(mem.buffer, xPtr, cols).set(x);
    wasm.q8_quantize(xPtr, xq8Ptr, cols);
    const act = q8Quantize(x);

    // wasm GEMV, weights at low pointer.
    new Uint8Array(mem.buffer, wLowPtr, bytesPerExpert).set(slabU8);
    gemv(wLowPtr, xq8Ptr, yPtr, rows, cols);
    const yLow = new Float32Array(mem.buffer, yPtr, rows).slice();

    // wasm GEMV, same weights at >2^31 pointer — must be bit-identical.
    new Uint8Array(mem.buffer, wHighPtr, bytesPerExpert).set(slabU8);
    gemv(wHighPtr, xq8Ptr, yPtr2, rows, cols);
    const yHigh = new Float32Array(mem.buffer, yPtr2, rows);
    let highMismatch = 0;
    for (let r = 0; r < rows; r++) if (yLow[r] !== yHigh[r]) highMismatch++;

    // (a) integer-exact JS reference on sampled rows.
    let intErr = 0;
    for (const r of [0, 1, 2, 3, rows - 1]) {
      const rowU8 = slabU8.subarray(r * rowBytes, (r + 1) * rowBytes);
      const ref = isQ6 ? q6kDotRowRef(rowU8, act, nb) : q5kDotRowRef(rowU8, act, nb);
      const rel = Math.abs(yLow[r] - ref) / Math.max(1e-6, Math.abs(ref));
      if (rel > intErr) intErr = rel;
    }

    // (b) f64 dot over fully dequantized weights and dequantized Q8 acts.
    const wF32 = dequant(slabU8, rows * cols);
    const xDeq = new Float64Array(cols);
    for (let j = 0; j < cols; j++) xDeq[j] = act.q[j] * act.d[Math.floor(j / QK_K)];
    let f64Err = 0;
    let maxAbs = 0;
    for (let r = 0; r < rows; r++) maxAbs = Math.max(maxAbs, Math.abs(yLow[r]));
    for (let r = 0; r < rows; r++) {
      let dot = 0;
      for (let j = 0; j < cols; j++) dot += wF32[r * cols + j] * xDeq[j];
      const rel = Math.abs(yLow[r] - dot) / Math.max(1e-3 * maxAbs, Math.abs(dot));
      if (rel > f64Err) f64Err = rel;
    }

    const ok = highMismatch === 0 && intErr < 1e-5 && f64Err < 5e-3;
    if (!ok) failures++;
    console.log(
      `${ok ? 'PASS' : 'FAIL'} ${c.tensor} expert ${String(e).padStart(3)} `
      + `[${rows}×${cols} ${isQ6 ? 'Q6_K' : 'Q5_K'}] intRef ${intErr.toExponential(1)}, `
      + `f64Ref ${f64Err.toExponential(1)}, highPtr mismatches ${highMismatch}`,
    );
  }
}

fs.closeSync(fd);
if (failures > 0) {
  console.error(`\n${failures} case(s) FAILED`);
  process.exit(1);
}
console.log('\nall expert-FFN kernel tests PASS');
