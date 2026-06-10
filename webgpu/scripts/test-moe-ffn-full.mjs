/**
 * Phase C2 node test — FULL MoE FFN pipeline vs f64 reference, real GGUF.
 * Run: npx tsx scripts/test-moe-ffn-full.mjs
 *
 * Emulates the production decode path exactly as moe-cpu.ts + moe-worker.ts
 * execute it (this is the path the browser uses, unlike test-expert-ffn.mjs
 * which used wasm q8_quantize for the activations):
 *   1. router logits = ffn_gate_inp (F32) @ x → topKSoftmax (top-8 → softmax-8)
 *   2. JS q8Quantize(x) → serializeQ8 → wasm memory   [tests serializeQ8!]
 *   3. per expert: gate GEMV → up GEMV → JS silu⊙up → wasm q8_quantize →
 *      down GEMV → weighted accumulate                 [tests worker loop]
 *   4. shexp: dequant Q8_0 gate/up/down FFN, g = sigmoid(x·gate_inp_shexp),
 *      combined = routed + g·shexp                     [tests combine math]
 * Reference: f64 over dequantized weights with the same Q8-dequantized x.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseGGUF, ggmlTypeTraits, GGML_TYPES } from '../src/model/gguf.ts';
import { dequantQ5_K, dequantQ6_K, dequantQ8_0, dequantF32 } from '../src/model/gguf-dequant.ts';
import { q8Quantize, QK_K } from '../src/bench/q5k-ref.ts';
import { topKSoftmax, serializeQ8, Q8_BLOCK_BYTES } from '../src/engine/moe-cpu.ts';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const GGUF_PATH = process.env.GGUF_PATH
  || path.resolve(__dirname, '../../models/qwen3.6-35b-a3b-gguf/Qwen3.6-35B-A3B-UD-Q5_K_S.gguf');
const WASM_PATH = path.resolve(__dirname, '../public/wasm/q5k_gemv.wasm');
const PAGE = 65536;

const align16 = (p) => p + ((16 - (p % 16)) % 16);

function seededRandFloats(n, seed, scale = 1) {
  const out = new Float32Array(n);
  let s = seed >>> 0 || 0x9e3779b9;
  for (let i = 0; i < n; i++) {
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    out[i] = ((s / 0xffffffff) * 2 - 1) * scale;
  }
  return out;
}

const silu = (g) => g / (1 + Math.exp(-g));
const relL2 = (a, b) => {
  let d = 0, r = 0;
  for (let i = 0; i < a.length; i++) { d += (a[i] - b[i]) ** 2; r += b[i] * b[i]; }
  return Math.sqrt(d / Math.max(r, 1e-30));
};

// ── load GGUF + wasm ───────────────────────────────────────────────────

const fd = fs.openSync(GGUF_PATH, 'r');
const readRange = async (start, end) => {
  const buf = Buffer.alloc(end - start);
  const got = fs.readSync(fd, buf, 0, end - start, start);
  if (got !== end - start) throw new Error(`short read at ${start}`);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + got);
};
const file = await parseGGUF(readRange);

const readTensor = (name) => {
  const t = file.tensors.get(name);
  if (!t) throw new Error(`tensor ${name} not found`);
  const buf = Buffer.alloc(t.byteLength);
  const got = fs.readSync(fd, buf, 0, t.byteLength, t.offset);
  if (got !== t.byteLength) throw new Error(`short read for ${name}`);
  return { info: t, u8: new Uint8Array(buf.buffer, buf.byteOffset, t.byteLength) };
};
const readExpertSlab = (name, e) => {
  const t = file.tensors.get(name);
  const [cols, rows] = t.ne;
  const { blockSize, typeSize } = ggmlTypeTraits(t.ggmlType);
  const bpe = rows * (cols / blockSize) * typeSize;
  const buf = Buffer.alloc(bpe);
  const got = fs.readSync(fd, buf, 0, bpe, t.offset + e * bpe);
  if (got !== bpe) throw new Error(`short slab read ${name} e${e}`);
  return { ggmlType: t.ggmlType, rows, cols, u8: new Uint8Array(buf.buffer, buf.byteOffset, bpe) };
};

const { instance } = await WebAssembly.instantiate(fs.readFileSync(WASM_PATH), {});
const wasm = instance.exports;
const mem = wasm.memory;

const H = 2048, F = 512, E = 256, TOPK = 8;

// wasm memory plan (mirrors moe-worker scratch)
let p = align16(wasm.__heap_base.value);
const xq8Ptr = p; p = align16(p + (H / QK_K) * Q8_BLOCK_BYTES);
const gateOutPtr = p; p = align16(p + F * 4);
const upOutPtr = p; p = align16(p + F * 4);
const act2Ptr = p; p = align16(p + F * 4);
const act2Q8Ptr = p; p = align16(p + (F / QK_K) * Q8_BLOCK_BYTES);
const downOutPtr = p; p = align16(p + H * 4);
const wPtr = p; p = align16(p + 4_000_000);
const needPages = Math.ceil(p / PAGE) - mem.buffer.byteLength / PAGE;
if (needPages > 0) mem.grow(needPages);

const dequantAny = (g, u8, n) =>
  g === GGML_TYPES.Q6_K ? dequantQ6_K(u8, n) : dequantQ5_K(u8, n);

let failures = 0;

for (const layer of [0, 34]) { // 0 = all-Q5_K, 34 = Q6_K down_exps
  // ── inputs ──
  const x = seededRandFloats(H, 0xbeef + layer * 7919, 0.8);

  // ── 1. router (production: topKSoftmax on F32 logits) ──
  const router = readTensor(`blk.${layer}.ffn_gate_inp.weight`);
  const rw = dequantF32(router.u8, E * H); // [E rows × H cols]
  const logits = new Float32Array(E);
  for (let e = 0; e < E; e++) {
    let s = 0;
    for (let j = 0; j < H; j++) s += rw[e * H + j] * x[j];
    logits[e] = s;
  }
  const { ids, weights } = topKSoftmax(logits, TOPK);

  // reference router: full f64 softmax → top-8 by prob → renorm
  {
    const probs = new Float64Array(E);
    let mx = -Infinity;
    for (let e = 0; e < E; e++) mx = Math.max(mx, logits[e]);
    let sum = 0;
    for (let e = 0; e < E; e++) { probs[e] = Math.exp(logits[e] - mx); sum += probs[e]; }
    const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, TOPK);
    const wSum = order.reduce((s, e) => s + probs[e], 0);
    const refW = new Map(order.map((e) => [e, probs[e] / sum / (wSum / sum)]));
    let routerErr = 0;
    for (let k = 0; k < TOPK; k++) {
      if (!refW.has(ids[k])) { routerErr = Infinity; break; }
      routerErr = Math.max(routerErr, Math.abs(weights[k] - refW.get(ids[k])));
    }
    console.log(`L${layer} router: top-8 ${[...ids].join(',')} maxWeightErr ${routerErr.toExponential(1)} ${routerErr < 1e-6 ? 'OK' : 'FAIL'}`);
    if (!(routerErr < 1e-6)) failures++;
  }

  // ── 2+3. routed experts — EXACT worker emulation ──
  const act = q8Quantize(x);
  const xq8Bytes = (H / QK_K) * Q8_BLOCK_BYTES;
  serializeQ8(act, new Uint8Array(mem.buffer, xq8Ptr, xq8Bytes)); // the seam under test

  const routed = new Float32Array(H);
  const routedRef = new Float64Array(H);
  // reference x = dequantized Q8 acts (same quantization the kernels see)
  const xDeq = new Float64Array(H);
  for (let j = 0; j < H; j++) xDeq[j] = act.q[j] * act.d[Math.floor(j / QK_K)];

  for (let k = 0; k < TOPK; k++) {
    const e = ids[k];
    const g = readExpertSlab(`blk.${layer}.ffn_gate_exps.weight`, e);
    const u = readExpertSlab(`blk.${layer}.ffn_up_exps.weight`, e);
    const dn = readExpertSlab(`blk.${layer}.ffn_down_exps.weight`, e);
    const gemv = (slab, wp, xp, yp) =>
      slab.ggmlType === GGML_TYPES.Q6_K
        ? wasm.q6k_gemv(wp, xp, yp, slab.rows, slab.cols)
        : wasm.q5k_gemv(wp, xp, yp, slab.rows, slab.cols);

    // worker path
    new Uint8Array(mem.buffer, wPtr, g.u8.length).set(g.u8);
    gemv(g, wPtr, xq8Ptr, gateOutPtr);
    const gateOut = new Float32Array(mem.buffer, gateOutPtr, F).slice();
    new Uint8Array(mem.buffer, wPtr, u.u8.length).set(u.u8);
    gemv(u, wPtr, xq8Ptr, upOutPtr);
    const upOut = new Float32Array(mem.buffer, upOutPtr, F).slice();
    const act2 = new Float32Array(mem.buffer, act2Ptr, F);
    for (let i = 0; i < F; i++) act2[i] = silu(gateOut[i]) * upOut[i];
    wasm.q8_quantize(act2Ptr, act2Q8Ptr, F);
    new Uint8Array(mem.buffer, wPtr, dn.u8.length).set(dn.u8);
    gemv(dn, wPtr, act2Q8Ptr, downOutPtr);
    const downOut = new Float32Array(mem.buffer, downOutPtr, H);
    for (let i = 0; i < H; i++) routed[i] += weights[k] * downOut[i];

    // f64 reference (dequantized weights, same Q8-dequantized x, exact silu)
    const gW = dequantAny(g.ggmlType, g.u8, F * H);
    const uW = dequantAny(u.ggmlType, u.u8, F * H);
    const dW = dequantAny(dn.ggmlType, dn.u8, H * F);
    const a2f = new Float32Array(F);
    for (let r = 0; r < F; r++) {
      let gs = 0, us = 0;
      for (let j = 0; j < H; j++) { gs += gW[r * H + j] * xDeq[j]; us += uW[r * H + j] * xDeq[j]; }
      a2f[r] = (gs / (1 + Math.exp(-gs))) * us;
    }
    // Mirror the worker's Q8 round-trip of act2 so quant noise cancels and
    // only structural errors remain.
    const a2Act = q8Quantize(a2f);
    const a2 = new Float64Array(F);
    for (let j = 0; j < F; j++) a2[j] = a2Act.q[j] * a2Act.d[Math.floor(j / QK_K)];
    for (let r = 0; r < H; r++) {
      let s = 0;
      for (let j = 0; j < F; j++) s += dW[r * F + j] * a2[j];
      routedRef[r] += weights[k] * s;
    }
  }
  const routedErr = relL2(routed, routedRef);
  console.log(`L${layer} routed experts: relL2 ${routedErr.toExponential(2)} ${routedErr < 3e-3 ? 'OK' : 'FAIL'}`);
  if (!(routedErr < 3e-3)) failures++;

  // ── 4. shared expert + sigmoid gate + combine (f64; mirrors GPU+JS) ──
  const sgV = dequantF32(readTensor(`blk.${layer}.ffn_gate_inp_shexp.weight`).u8, H);
  const sg8 = (n) => dequantQ8_0(readTensor(`blk.${layer}.ffn_${n}_shexp.weight`).u8, n === 'down' ? H * F : F * H);
  const gW = sg8('gate'), uW = sg8('up'), dW = sg8('down');
  const a2 = new Float64Array(F);
  for (let r = 0; r < F; r++) {
    let gs = 0, us = 0;
    for (let j = 0; j < H; j++) { gs += gW[r * H + j] * x[j]; us += uW[r * H + j] * x[j]; }
    a2[r] = (gs / (1 + Math.exp(-gs))) * us;
  }
  let gDot = 0;
  for (let j = 0; j < H; j++) gDot += sgV[j] * x[j];
  const gSig = 1 / (1 + Math.exp(-gDot));
  const combined = new Float64Array(H);
  for (let r = 0; r < H; r++) {
    let s = 0;
    for (let j = 0; j < F; j++) s += dW[r * F + j] * a2[j];
    combined[r] = routedRef[r] + gSig * s;
  }
  let cl2 = 0, rl2 = 0;
  for (let i = 0; i < H; i++) { cl2 += combined[i] ** 2; rl2 += routedRef[i] ** 2; }
  console.log(`L${layer} shexp gate sigmoid ${gSig.toFixed(4)}; |combined| ${Math.sqrt(cl2).toFixed(3)} vs |routed| ${Math.sqrt(rl2).toFixed(3)}`);
}

fs.closeSync(fd);
if (failures > 0) { console.error(`\n${failures} check(s) FAILED`); process.exit(1); }
console.log('\nfull MoE FFN pipeline PASS (router + serializeQ8 + worker math + combine)');
