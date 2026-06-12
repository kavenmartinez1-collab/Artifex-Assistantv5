/**
 * Fusion-lever kernel validation (kernel-audit rule) — JS ports of the new
 * fused WGSL entry points checked against JS ports of the legacy command
 * sequences they replace. Every fusion item was designed to be bit-exact, so
 * all gates here are EXACT equality (f32 semantics emulated with fround at
 * every operation, matching WGSL's round-after-every-op behavior; exp() is
 * the same call in both ports so its precision cancels).
 *
 *  F1 deinterleave_qgate  vs per-head CPU copy loop          (data movement)
 *  F4 conv1d_silu_update  vs conv1d → update_state → silu    (same expr order)
 *  F7 gate_sigmoid        vs sigmoid_op → copy → mul         (same expr order)
 *
 * Run: npx tsx scripts/test-fusion-kernels.mts
 */

// ── deterministic RNG ──────────────────────────────────────────────────
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const fr = Math.fround;

let failures = 0;
function check(label: string, ok: boolean, detail = '') {
  if (!ok) { failures++; console.log(`  FAIL ${label} ${detail}`); }
}

// ── F1: deinterleave_qgate ─────────────────────────────────────────────
// Kernel port (deinterleave.wgsl): per output element idx,
//   block = idx / d_head; e = idx % d_head; src = block*2*d_head + e
function f1Test() {
  for (const [seq, nHeads, dHead] of [[1, 28, 128], [3, 28, 128], [2, 4, 64]]) {
    const rng = mulberry32(0xF1 ^ (seq << 4) ^ (nHeads << 9) ^ dHead);
    const n = seq * nHeads * dHead;
    const input = new Float32Array(n * 2);
    for (let i = 0; i < input.length; i++) input[i] = fr(rng() * 4 - 2);

    // Fused kernel port
    const q = new Float32Array(n), g = new Float32Array(n);
    for (let idx = 0; idx < n; idx++) {
      const block = (idx / dHead) | 0;
      const e = idx % dHead;
      const src = block * dHead * 2 + e;
      q[idx] = input[src];
      g[idx] = input[src + dHead];
    }

    // Legacy reference: per-head block copy loop (forward-pass CPU path)
    // row layout per token: [Q_h0 (dHead) | gate_h0 (dHead) | Q_h1 | ...]
    const qRef = new Float32Array(n), gRef = new Float32Array(n);
    for (let s = 0; s < seq; s++) {
      for (let h = 0; h < nHeads; h++) {
        const srcBase = (s * nHeads + h) * dHead * 2;
        const dstBase = (s * nHeads + h) * dHead;
        for (let e = 0; e < dHead; e++) {
          qRef[dstBase + e] = input[srcBase + e];
          gRef[dstBase + e] = input[srcBase + dHead + e];
        }
      }
    }

    let ok = true;
    for (let i = 0; i < n && ok; i++) {
      if (q[i] !== qRef[i]) { check('F1 q', false, `seq=${seq} i=${i}: ${q[i]} != ${qRef[i]}`); ok = false; }
      if (g[i] !== gRef[i]) { check('F1 gate', false, `seq=${seq} i=${i}: ${g[i]} != ${gRef[i]}`); ok = false; }
    }
    if (ok) console.log(`  ok   F1 deinterleave seq=${seq} heads=${nHeads} dHead=${dHead}: bit-exact (${n} elems x2)`);
  }
}

// ── F4: conv1d_silu_update ─────────────────────────────────────────────
// Both ports share the conv-sum expression: sum = fround(sum + fround(w*x)),
// state shift is pure data movement, SiLU = fround(x / fround(1 + fround(exp(-x)))).
// The legacy path stores the conv sum to a buffer and reloads it for SiLU —
// an f32 store/load roundtrip, which is exact, so fusing SiLU onto the
// register value is identical.
function convSum(state: Float32Array, x: Float32Array, w: Float32Array,
                 d: number, dim: number, ks: number): number {
  const sl = ks - 1;
  let sum = 0;
  for (let k = 0; k < sl; k++) sum = fr(sum + fr(state[k * dim + d] * w[d * ks + k]));
  return fr(sum + fr(x[d] * w[d * ks + sl]));
}
function silu(x: number): number {
  return fr(x / fr(1 + fr(Math.exp(-x))));
}
function f4Test() {
  for (const [dim, ks] of [[8192, 4], [1280, 4], [256, 3]]) {
    const rng = mulberry32(0xF4 ^ (dim << 3) ^ ks);
    const sl = ks - 1;
    const state0 = new Float32Array(sl * dim);
    const x = new Float32Array(dim);
    const w = new Float32Array(dim * ks);
    for (let i = 0; i < state0.length; i++) state0[i] = fr(rng() * 2 - 1);
    for (let i = 0; i < dim; i++) x[i] = fr(rng() * 2 - 1);
    for (let i = 0; i < w.length; i++) w[i] = fr(rng() * 2 - 1);

    // Fused port (conv1d_silu_update): per channel — sum, shift+append, silu
    const stateF = state0.slice();
    const outF = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
      const sum = convSum(stateF, x, w, d, dim, ks); // reads pre-shift state
      for (let k = 0; k < sl - 1; k++) stateF[k * dim + d] = stateF[(k + 1) * dim + d];
      stateF[(sl - 1) * dim + d] = x[d];
      outF[d] = silu(sum);
    }

    // Legacy port: conv1d (all channels) → update_state (all) → silu (all)
    const stateL = state0.slice();
    const conv = new Float32Array(dim);
    for (let d = 0; d < dim; d++) conv[d] = convSum(stateL, x, w, d, dim, ks);
    for (let d = 0; d < dim; d++) {
      for (let k = 0; k < sl - 1; k++) stateL[k * dim + d] = stateL[(k + 1) * dim + d];
      stateL[(sl - 1) * dim + d] = x[d];
    }
    const outL = new Float32Array(dim);
    for (let d = 0; d < dim; d++) outL[d] = silu(conv[d]); // store/load roundtrip exact

    let ok = true;
    for (let d = 0; d < dim && ok; d++) {
      if (outF[d] !== outL[d]) { check('F4 out', false, `dim=${dim} d=${d}: ${outF[d]} != ${outL[d]}`); ok = false; }
    }
    for (let i = 0; i < stateF.length && ok; i++) {
      if (stateF[i] !== stateL[i]) { check('F4 state', false, `dim=${dim} i=${i}: ${stateF[i]} != ${stateL[i]}`); ok = false; }
    }
    if (ok) console.log(`  ok   F4 conv1d_silu_update dim=${dim} ks=${ks}: output+state bit-exact`);
  }
}

// ── F7: gate_sigmoid ───────────────────────────────────────────────────
// Fused: s = fround(1 / fround(1 + fround(exp(-b)))); out = fround(a * s)
// Legacy: sigmoid_op writes s (same expression) to a buffer, copy a, then
// mul: out = fround(a * s). Same two roundings — must be exact.
function f7Test() {
  for (const n of [3584, 1024]) {
    const rng = mulberry32(0xF7 ^ n);
    const a = new Float32Array(n), b = new Float32Array(n);
    for (let i = 0; i < n; i++) { a[i] = fr(rng() * 8 - 4); b[i] = fr(rng() * 16 - 8); }

    const sig = (x: number) => fr(1 / fr(1 + fr(Math.exp(-x))));

    const outF = new Float32Array(n);
    for (let i = 0; i < n; i++) outF[i] = fr(a[i] * sig(b[i]));

    const s = new Float32Array(n);
    for (let i = 0; i < n; i++) s[i] = sig(b[i]);          // sigmoid_op → buffer
    const aCopy = a.slice();                               // batchCopy
    const outL = new Float32Array(n);
    for (let i = 0; i < n; i++) outL[i] = fr(aCopy[i] * s[i]); // mul

    let ok = true;
    for (let i = 0; i < n && ok; i++) {
      if (outF[i] !== outL[i]) { check('F7', false, `n=${n} i=${i}: ${outF[i]} != ${outL[i]}`); ok = false; }
    }
    if (ok) console.log(`  ok   F7 gate_sigmoid n=${n}: bit-exact`);
  }
}

// ── F5: beta/decay prologue in ssm_step ────────────────────────────────
// Fused prologue (first nvh threads): beta = sigmoid(B), dt = softplus(A +
// dt_bias), decay = exp(-exp(A_log) * dt) — expressions verbatim from
// elementwise.wgsl (sigmoid_op / softplus incl. the x>20 branch /
// decay_compute). Legacy path runs them as three dispatches with f32
// store/load roundtrips between; roundtrips are exact, so register-chained
// values must be bit-identical.
function f5Test() {
  const nvh = 32;
  const rng = mulberry32(0xF5);
  const A = new Float32Array(nvh), B = new Float32Array(nvh);
  const dtBias = new Float32Array(nvh), ALog = new Float32Array(nvh);
  for (let i = 0; i < nvh; i++) {
    A[i] = fr(rng() * 30 - 5);        // exercise softplus x>20 branch too
    B[i] = fr(rng() * 8 - 4);
    dtBias[i] = fr(rng() * 2 - 1);
    ALog[i] = fr(rng() * 4 - 2);
  }
  const sigmoid = (x: number) => fr(1 / fr(1 + fr(Math.exp(-x))));
  const softplus = (x: number) => x > 20 ? x : fr(Math.log(fr(1 + fr(Math.exp(x)))));
  const decayOf = (aLog: number, dt: number) => fr(Math.exp(fr(fr(-Math.exp(aLog)) * dt)));

  // Fused port: register-chained per thread
  const betaF = new Float32Array(nvh), decayF = new Float32Array(nvh);
  for (let i = 0; i < nvh; i++) {
    betaF[i] = sigmoid(B[i]);
    const dt = softplus(fr(A[i] + dtBias[i]));
    decayF[i] = decayOf(ALog[i], dt);
  }

  // Legacy port: three dispatches, f32 buffers between (Float32Array stores)
  const dtBuf = new Float32Array(nvh);
  for (let i = 0; i < nvh; i++) dtBuf[i] = softplus(fr(A[i] + dtBias[i]));   // softplus
  const decayL = new Float32Array(nvh);
  for (let i = 0; i < nvh; i++) decayL[i] = decayOf(ALog[i], dtBuf[i]);      // decay_compute
  const betaL = new Float32Array(nvh);
  for (let i = 0; i < nvh; i++) betaL[i] = sigmoid(B[i]);                    // sigmoid_op

  let ok = true;
  for (let i = 0; i < nvh && ok; i++) {
    if (betaF[i] !== betaL[i]) { check('F5 beta', false, `i=${i}: ${betaF[i]} != ${betaL[i]}`); ok = false; }
    if (decayF[i] !== decayL[i]) { check('F5 decay', false, `i=${i}: ${decayF[i]} != ${decayL[i]}`); ok = false; }
  }
  if (ok) console.log(`  ok   F5 beta/decay prologue nvh=${nvh}: bit-exact`);
}

// ── Occupancy split: ssm_step_split vs ssm_step_fused ──────────────────
// The fused kernel runs 5 array-wide passes separated by barriers, but every
// (kh, v) column of h is fully independent: each step only touches
// h[h_base + k*gvd + v] for its own v, and the per-column k-loop order is
// identical. The split kernel assigns one thread per column with grid
// (nkh, ceil(gvd/WG)) and fuses steps 1+2 and 4+5 in registers:
//   pass A: hk = d*h[k]; h[k] = hk; kv_mem += hk*K[k]
//   pass B: hk = h[k] + K[k]*delta; h[k] = hk; dot += hk*Q[k]
// The fused kernel reads h back from memory between steps — an f32
// store/load roundtrip, which is exact — so register chaining is
// bit-identical. beta/decay move from the LDS prologue to per-thread
// registers (same expressions, same inputs → same f32 results).
function ssmSplitTest() {
  const sigmoid = (x: number) => fr(1 / fr(1 + fr(Math.exp(-x))));
  const softplus = (x: number) => x > 20 ? x : fr(Math.log(fr(1 + fr(Math.exp(x)))));

  for (const [nkh, nvh, kd, gvd] of [[16, 32, 128, 256], [4, 8, 32, 64]]) {
    for (const tiled of [true, false]) {
      for (const WG of [64, 48]) { // 48 doesn't divide gvd → exercises the v guard
        const rng = mulberry32(0x55 ^ (nkh << 2) ^ kd ^ (tiled ? 1 : 0) ^ (WG << 8));
        const vhPerKh = nvh / nkh;
        const vhd = gvd / vhPerKh;
        const Q = new Float32Array(nkh * kd), K = new Float32Array(nkh * kd);
        const V = new Float32Array(nvh * vhd);
        const B = new Float32Array(nvh), A = new Float32Array(nvh);
        const dtBias = new Float32Array(nvh), ALog = new Float32Array(nvh);
        const h0 = new Float32Array(nkh * kd * gvd);
        for (let i = 0; i < Q.length; i++) { Q[i] = fr(rng() * 2 - 1); K[i] = fr(rng() * 2 - 1); }
        for (let i = 0; i < V.length; i++) V[i] = fr(rng() * 2 - 1);
        for (let i = 0; i < nvh; i++) {
          B[i] = fr(rng() * 8 - 4);
          A[i] = fr(rng() * 30 - 5); // exercise softplus x>20 branch
          dtBias[i] = fr(rng() * 2 - 1);
          ALog[i] = fr(rng() * 4 - 2);
        }
        for (let i = 0; i < h0.length; i++) h0[i] = fr(rng() * 2 - 1);
        const scale = fr(1 / Math.sqrt(kd));
        const vhOf = (kh: number, v: number) =>
          tiled ? Math.floor(v / vhd) * nkh + kh : kh * vhPerKh + Math.floor(v / vhd);
        const vIdxOf = (kh: number, v: number, vh: number) =>
          tiled ? vh * vhd + (v % vhd) : kh * gvd + v;

        // ── Fused port (ssm_step_fused): prologue → 5 barrier-separated passes
        const wgBeta = new Float32Array(nvh), wgDecay = new Float32Array(nvh);
        for (let i = 0; i < nvh; i++) {
          wgBeta[i] = sigmoid(B[i]);
          const dt = softplus(fr(A[i] + dtBias[i]));
          wgDecay[i] = fr(Math.exp(fr(fr(-Math.exp(ALog[i])) * dt)));
        }
        const hF = h0.slice();
        const outF = new Float32Array(nkh * gvd);
        for (let kh = 0; kh < nkh; kh++) {
          const hBase = kh * kd * gvd;
          for (let v = 0; v < gvd; v++) { // step 1
            const d = wgDecay[vhOf(kh, v)];
            for (let k = 0; k < kd; k++) hF[hBase + k * gvd + v] = fr(d * hF[hBase + k * gvd + v]);
          }
          for (let v = 0; v < gvd; v++) { // steps 2-4
            let kvMem = 0;
            for (let k = 0; k < kd; k++) kvMem = fr(kvMem + fr(hF[hBase + k * gvd + v] * K[kh * kd + k]));
            const vh = vhOf(kh, v);
            const delta = fr(fr(V[vIdxOf(kh, v, vh)] - kvMem) * wgBeta[vh]);
            for (let k = 0; k < kd; k++) {
              const hi = hBase + k * gvd + v;
              hF[hi] = fr(hF[hi] + fr(K[kh * kd + k] * delta));
            }
          }
          for (let v = 0; v < gvd; v++) { // step 5
            let dot = 0;
            for (let k = 0; k < kd; k++) dot = fr(dot + fr(hF[hBase + k * gvd + v] * Q[kh * kd + k]));
            outF[vIdxOf(kh, v, vhOf(kh, v))] = fr(dot * scale);
          }
        }

        // ── Split port (ssm_step_split): grid (nkh, ceil(gvd/WG)), thread/column
        const hS = h0.slice();
        const outS = new Float32Array(nkh * gvd);
        for (let kh = 0; kh < nkh; kh++) {
          for (let wy = 0; wy < Math.ceil(gvd / WG); wy++) {
            for (let tid = 0; tid < WG; tid++) {
              const v = wy * WG + tid;
              if (v >= gvd) continue;
              const hBase = kh * kd * gvd;
              const vh = vhOf(kh, v);
              const betaVal = sigmoid(B[vh]);
              const dt = softplus(fr(A[vh] + dtBias[vh]));
              const d = fr(Math.exp(fr(fr(-Math.exp(ALog[vh])) * dt)));
              let kvMem = 0; // pass A: steps 1+2 fused
              for (let k = 0; k < kd; k++) {
                const hi = hBase + k * gvd + v;
                const hk = fr(d * hS[hi]);
                hS[hi] = hk;
                kvMem = fr(kvMem + fr(hk * K[kh * kd + k]));
              }
              const delta = fr(fr(V[vIdxOf(kh, v, vh)] - kvMem) * betaVal);
              let dot = 0; // pass B: steps 4+5 fused
              for (let k = 0; k < kd; k++) {
                const hi = hBase + k * gvd + v;
                const hk = fr(hS[hi] + fr(K[kh * kd + k] * delta));
                hS[hi] = hk;
                dot = fr(dot + fr(hk * Q[kh * kd + k]));
              }
              outS[vIdxOf(kh, v, vh)] = fr(dot * scale);
            }
          }
        }

        let ok = true;
        for (let i = 0; i < outF.length && ok; i++) {
          if (outS[i] !== outF[i]) { check('split out', false, `nkh=${nkh} tiled=${tiled} WG=${WG} i=${i}: ${outS[i]} != ${outF[i]}`); ok = false; }
        }
        for (let i = 0; i < hF.length && ok; i++) {
          if (hS[i] !== hF[i]) { check('split h', false, `nkh=${nkh} tiled=${tiled} WG=${WG} i=${i}: ${hS[i]} != ${hF[i]}`); ok = false; }
        }
        if (ok) console.log(`  ok   ssm_step_split nkh=${nkh} kd=${kd} gvd=${gvd} tiled=${tiled} WG=${WG}: h+output bit-exact`);
      }
    }
  }
}

console.log('F1 deinterleave_qgate:');
f1Test();
console.log('F4 conv1d_silu_update:');
f4Test();
console.log('F7 gate_sigmoid:');
f7Test();
console.log('F5 beta/decay prologue:');
f5Test();
console.log('ssm_step_split (occupancy lever):');
ssmSplitTest();

if (failures > 0) {
  console.log(`\n${failures} FAILURE(S)`);
  process.exit(1);
}
console.log('\nAll fusion kernel checks passed.');
