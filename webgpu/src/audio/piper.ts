/**
 * Piper VITS runtime (en_US-joe-medium) — Phase P6.
 *
 * CPU f32 typed-array implementation of the small VITS modules (enc_p / dp /
 * flow); the HiFiGAN decoder (dec, ~95% of FLOPs) runs on the GPU via
 * conv1d_gen.wgsl (added in a later step). Weights come from the converter
 * output `webgpu/models/piper-en-us-joe-medium/model.safetensors` (all f32,
 * graph-derived names — see scripts/convert-piper-onnx.py).
 *
 * Tensor layout convention: VITS is channels-first. All activations here are
 * flat Float32Array in [C, T] row-major (channel-major): x[c * T + t]. Conv
 * weights are PyTorch [C_out, C_in, K] flat.
 *
 * Parity ground truth: scripts/piper_fixture/ (scales=[0,1,0] zero-noise),
 * gate scripts/test-piper-parity.mts.
 */

// ─── Weights ─────────────────────────────────────────────────────────────────

export interface PiperTensor {
  shape: number[];
  data: Float32Array;
}
export type PiperWeights = Map<string, PiperTensor>;

function get(w: PiperWeights, name: string): PiperTensor {
  const t = w.get(name);
  if (!t) throw new Error(`piper: missing weight ${name}`);
  return t;
}

// ─── Elementwise / small ops ─────────────────────────────────────────────────

/** LayerNorm over the channel dim, per time step (VITS modules.LayerNorm). */
function layerNormCh(
  x: Float32Array, C: number, T: number,
  gamma: Float32Array, beta: Float32Array, eps = 1e-5,
): Float32Array {
  const out = new Float32Array(C * T);
  for (let t = 0; t < T; t++) {
    let mean = 0;
    for (let c = 0; c < C; c++) mean += x[c * T + t];
    mean /= C;
    let varr = 0;
    for (let c = 0; c < C; c++) {
      const d = x[c * T + t] - mean;
      varr += d * d;
    }
    varr /= C;
    const inv = 1 / Math.sqrt(varr + eps);
    for (let c = 0; c < C; c++) {
      out[c * T + t] = (x[c * T + t] - mean) * inv * gamma[c] + beta[c];
    }
  }
  return out;
}

/**
 * Dense conv1d over [C_in, T] → [C_out, T_out]. weight [C_out, C_in, K] flat,
 * bias [C_out] (or null). Symmetric same-pad via padL/padR (zeros), plus
 * dilation and stride. Matches PyTorch Conv1d semantics.
 */
function conv1d(
  x: Float32Array, Cin: number, T: number,
  weight: Float32Array, bias: Float32Array | null,
  Cout: number, K: number,
  padL: number, padR: number, dilation = 1, stride = 1,
): { data: Float32Array; Tout: number } {
  const Tpad = T + padL + padR;
  const Tout = Math.floor((Tpad - (dilation * (K - 1) + 1)) / stride) + 1;
  const out = new Float32Array(Cout * Tout);
  for (let co = 0; co < Cout; co++) {
    const b = bias ? bias[co] : 0;
    const wBase = co * Cin * K;
    for (let to = 0; to < Tout; to++) {
      let acc = b;
      const start = to * stride - padL;
      for (let ci = 0; ci < Cin; ci++) {
        const xBase = ci * T;
        const wc = wBase + ci * K;
        for (let k = 0; k < K; k++) {
          const ti = start + k * dilation;
          if (ti >= 0 && ti < T) acc += weight[wc + k] * x[xBase + ti];
        }
      }
      out[co * Tout + to] = acc;
    }
  }
  return { data: out, Tout };
}

/** Pointwise conv (K=1) = per-timestep linear. weight [Cout, Cin, 1]. */
function conv1x1(
  x: Float32Array, Cin: number, T: number,
  weight: Float32Array, bias: Float32Array | null, Cout: number,
): Float32Array {
  return conv1d(x, Cin, T, weight, bias, Cout, 1, 0, 0).data;
}

// ─── enc_p relative-position attention (VITS modules.MultiHeadAttention) ──────

/**
 * _get_relative_embeddings(rel [2W+1, D], length) → [2L-1, D].
 * W = window_size = 4, so rel has 9 positions. For L >= W+1 (our case) the
 * result is the symmetric zero-pad of rel to length 2L-1.
 */
function getRelEmb(rel: Float32Array, D: number, W: number, L: number): Float32Array {
  const M = 2 * L - 1;
  const out = new Float32Array(M * D); // zero-filled
  const padLength = Math.max(L - (W + 1), 0);
  const sliceStart = Math.max((W + 1) - L, 0);
  // padded position p in [0, 9 + 2*padLength); rel index = p - padLength.
  // used = padded[sliceStart : sliceStart + M].
  for (let m = 0; m < M; m++) {
    const p = sliceStart + m;
    const r = p - padLength;
    if (r >= 0 && r < 2 * W + 1) {
      for (let d = 0; d < D; d++) out[m * D + d] = rel[r * D + d];
    }
  }
  return out;
}

/** relative_position_to_absolute_position: [L, 2L-1] → [L, L]. */
function relToAbs(x: Float32Array, L: number): Float32Array {
  const M = 2 * L - 1;
  // pad last dim +1 → rows of width 2L, flatten, then read with the skew.
  const rowW = 2 * L;
  const flat = new Float32Array(L * rowW + (L - 1)); // + right pad (L-1)
  for (let i = 0; i < L; i++) {
    for (let m = 0; m < M; m++) flat[i * rowW + m] = x[i * M + m];
    // slot (rowW-1) left as 0 (the F.pad [0,1])
  }
  // reshape [L+1, 2L-1], take [:L, L-1:]
  const out = new Float32Array(L * L);
  const rw2 = 2 * L - 1;
  for (let i = 0; i < L; i++) {
    for (let j = 0; j < L; j++) {
      out[i * L + j] = flat[i * rw2 + (L - 1) + j];
    }
  }
  return out;
}

/** absolute_position_to_relative_position: [L, L] → [L, 2L-1]. */
function absToRel(x: Float32Array, L: number): Float32Array {
  const M = 2 * L - 1;
  // pad last dim right by (L-1) → width 2L-1, flatten, left-pad by L → width 2L.
  const flat = new Float32Array(L * (2 * L - 1));
  for (let i = 0; i < L; i++) {
    for (let c = 0; c < L; c++) flat[i * (2 * L - 1) + c] = x[i * L + c];
    // cols L..2L-2 left as 0 (right pad L-1)
  }
  const G = new Float32Array(L * 2 * L);
  G.set(flat, L); // left pad L zeros
  // reshape [L, 2L], drop first col → [L, 2L-1]
  const out = new Float32Array(L * M);
  for (let i = 0; i < L; i++) {
    for (let m = 0; m < M; m++) out[i * M + m] = G[i * 2 * L + m + 1];
  }
  return out;
}

function relPosAttention(
  x: Float32Array, T: number, w: PiperWeights, prefix: string,
): Float32Array {
  const C = 192, H = 2, D = 96, W = 4;
  const scale = 1 / Math.sqrt(D);
  const q = conv1x1(x, C, T, get(w, `${prefix}.conv_q.weight`).data, get(w, `${prefix}.conv_q.bias`).data, C);
  const k = conv1x1(x, C, T, get(w, `${prefix}.conv_k.weight`).data, get(w, `${prefix}.conv_k.bias`).data, C);
  const v = conv1x1(x, C, T, get(w, `${prefix}.conv_v.weight`).data, get(w, `${prefix}.conv_v.bias`).data, C);
  const relK = get(w, `${prefix}.emb_rel_k`).data; // [9, 96]
  const relV = get(w, `${prefix}.emb_rel_v`).data;
  const relEmbK = getRelEmb(relK, D, W, T); // [2T-1, 96]
  const relEmbV = getRelEmb(relV, D, W, T);
  const M = 2 * T - 1;

  const out = new Float32Array(C * T);
  const scores = new Float32Array(T * T);
  const relLogits = new Float32Array(T * M);
  const probs = new Float32Array(T * T);

  for (let h = 0; h < H; h++) {
    const base = h * D; // channel offset for this head
    // content scores + relative-key logits
    for (let i = 0; i < T; i++) {
      // rel logits[i][m] = scale * <q_i, relEmbK[m]>
      for (let m = 0; m < M; m++) {
        let acc = 0;
        for (let d = 0; d < D; d++) acc += q[(base + d) * T + i] * relEmbK[m * D + d];
        relLogits[i * M + m] = acc * scale;
      }
      for (let j = 0; j < T; j++) {
        let acc = 0;
        for (let d = 0; d < D; d++) acc += q[(base + d) * T + i] * k[(base + d) * T + j];
        scores[i * T + j] = acc * scale;
      }
    }
    // scores += rel_to_abs(rel_logits)
    const scoresLocal = relToAbs(relLogits, T);
    for (let n = 0; n < T * T; n++) scores[n] += scoresLocal[n];
    // softmax over j (no mask: single unpadded sentence)
    for (let i = 0; i < T; i++) {
      let mx = -Infinity;
      for (let j = 0; j < T; j++) mx = Math.max(mx, scores[i * T + j]);
      let sum = 0;
      for (let j = 0; j < T; j++) {
        const e = Math.exp(scores[i * T + j] - mx);
        probs[i * T + j] = e;
        sum += e;
      }
      const inv = 1 / sum;
      for (let j = 0; j < T; j++) probs[i * T + j] *= inv;
    }
    // output[i][d] = sum_j P[i][j] v_j[d]
    for (let i = 0; i < T; i++) {
      for (let d = 0; d < D; d++) {
        let acc = 0;
        for (let j = 0; j < T; j++) acc += probs[i * T + j] * v[(base + d) * T + j];
        out[(base + d) * T + i] = acc;
      }
    }
    // + relative values: rel_weights = abs_to_rel(P) [T, 2T-1]; out += rw @ relEmbV
    const relWeights = absToRel(probs, T);
    for (let i = 0; i < T; i++) {
      for (let d = 0; d < D; d++) {
        let acc = 0;
        for (let m = 0; m < M; m++) acc += relWeights[i * M + m] * relEmbV[m * D + d];
        out[(base + d) * T + i] += acc;
      }
    }
  }
  return conv1x1(out, C, T, get(w, `${prefix}.conv_o.weight`).data, get(w, `${prefix}.conv_o.bias`).data, C);
}

// ─── enc_p FFN (VITS attentions.FFN, relu, same-pad k3) ──────────────────────

function ffn(x: Float32Array, T: number, w: PiperWeights, prefix: string): Float32Array {
  const C = 192, F = 768, K = 3;
  const w1 = get(w, `${prefix}.conv_1.weight`).data;
  const b1 = get(w, `${prefix}.conv_1.bias`).data;
  const h = conv1d(x, C, T, w1, b1, F, K, 1, 1).data; // same pad
  for (let n = 0; n < h.length; n++) if (h[n] < 0) h[n] = 0; // relu
  const w2 = get(w, `${prefix}.conv_2.weight`).data;
  const b2 = get(w, `${prefix}.conv_2.bias`).data;
  return conv1d(h, F, T, w2, b2, C, K, 1, 1).data;
}

// ─── enc_p forward (VITS TextEncoder) ────────────────────────────────────────

export interface EncPOut {
  m: Float32Array;    // [192, T]
  logs: Float32Array; // [192, T]
  x: Float32Array;    // [192, T] encoder output (for downstream flow conditioning)
  T: number;
}

/**
 * enc_p text encoder. `ids` are phoneme ids (BOS/EOS/PAD already interleaved,
 * i.e. the piper phonemes_to_ids output). Returns m_p / logs_p [192, T].
 */
export function encP(ids: number[], w: PiperWeights): EncPOut {
  const C = 192, LAYERS = 6;
  const T = ids.length;
  const emb = get(w, 'enc_p.emb.weight').data; // [256, 192]
  const embScale = Math.sqrt(C);
  // x[c, t] = emb[id_t, c] * sqrt(C)   (channels-first)
  let x: Float32Array = new Float32Array(C * T);
  for (let t = 0; t < T; t++) {
    const id = ids[t];
    for (let c = 0; c < C; c++) x[c * T + t] = emb[id * C + c] * embScale;
  }

  for (let l = 0; l < LAYERS; l++) {
    const y = relPosAttention(x, T, w, `enc_p.encoder.attn_layers.${l}`);
    for (let n = 0; n < x.length; n++) x[n] += y[n]; // residual
    x = layerNormCh(
      x, C, T,
      get(w, `enc_p.encoder.norm_layers_1.${l}.gamma`).data,
      get(w, `enc_p.encoder.norm_layers_1.${l}.beta`).data,
    );
    const y2 = ffn(x, T, w, `enc_p.encoder.ffn_layers.${l}`);
    for (let n = 0; n < x.length; n++) x[n] += y2[n];
    x = layerNormCh(
      x, C, T,
      get(w, `enc_p.encoder.norm_layers_2.${l}.gamma`).data,
      get(w, `enc_p.encoder.norm_layers_2.${l}.beta`).data,
    );
  }

  // proj: Conv1d(192, 384, 1) → split into m_p ‖ logs_p
  const stats = conv1x1(x, C, T, get(w, 'enc_p.proj.weight').data, get(w, 'enc_p.proj.bias').data, 2 * C);
  const m = new Float32Array(C * T);
  const logs = new Float32Array(C * T);
  m.set(stats.subarray(0, C * T));
  logs.set(stats.subarray(C * T, 2 * C * T));
  return { m, logs, x, T };
}

// ─── Alignment expansion (text → frames via durations) ───────────────────────

/**
 * Expand a [C, T] tensor to [C, F] by repeating each column `dur[i]` times
 * (monotonic alignment; VITS generate_path). F = sum(dur).
 */
export function expandByDuration(
  src: Float32Array, C: number, T: number, dur: number[],
): { data: Float32Array; F: number } {
  let F = 0;
  for (let i = 0; i < T; i++) F += dur[i];
  const out = new Float32Array(C * F);
  for (let c = 0; c < C; c++) {
    let f = 0;
    const sBase = c * T, oBase = c * F;
    for (let i = 0; i < T; i++) {
      const val = src[sBase + i];
      for (let r = 0; r < dur[i]; r++) out[oBase + f++] = val;
    }
  }
  return { data: out, F };
}

// ─── flow: WaveNet + residual affine coupling (VITS ResidualCouplingBlock) ────

/** WaveNet (VITS modules.WN): 4 layers, hidden 192, kernel 5, dilation 1, no g. */
function wn(x: Float32Array, T: number, w: PiperWeights, prefix: string): Float32Array {
  const H = 192, LAYERS = 4, K = 5, PAD = 2;
  const state = x.slice(); // mutated residual stream
  const output = new Float32Array(H * T);
  for (let i = 0; i < LAYERS; i++) {
    const inW = get(w, `${prefix}.in_layers.${i}.weight`).data;
    const inB = get(w, `${prefix}.in_layers.${i}.bias`).data;
    const xin = conv1d(state, H, T, inW, inB, 2 * H, K, PAD, PAD).data; // [384, T]
    // fused tanh(top) * sigmoid(bottom) → [192, T]
    const acts = new Float32Array(H * T);
    for (let c = 0; c < H; c++) {
      for (let t = 0; t < T; t++) {
        const a = xin[c * T + t];
        const b = xin[(c + H) * T + t];
        acts[c * T + t] = Math.tanh(a) * (1 / (1 + Math.exp(-b)));
      }
    }
    const last = i === LAYERS - 1;
    const rsCout = last ? H : 2 * H;
    const rsW = get(w, `${prefix}.res_skip_layers.${i}.weight`).data;
    const rsB = get(w, `${prefix}.res_skip_layers.${i}.bias`).data;
    const rs = conv1x1(acts, H, T, rsW, rsB, rsCout);
    if (!last) {
      for (let n = 0; n < H * T; n++) state[n] += rs[n];         // res into stream
      for (let c = 0; c < H; c++)
        for (let t = 0; t < T; t++) output[c * T + t] += rs[(c + H) * T + t]; // skip
    } else {
      for (let n = 0; n < H * T; n++) output[n] += rs[n];
    }
  }
  return output;
}

/** One residual coupling layer, reverse (mean_only): x1 -= post(WN(pre(x0))). */
function couplingReverse(x: Float32Array, T: number, w: PiperWeights, prefix: string): Float32Array {
  const C = 192, HALF = 96;
  const x0 = x.subarray(0, HALF * T);          // channels 0..95 (contiguous)
  const x1 = x.subarray(HALF * T, C * T);      // channels 96..191
  const h = conv1x1(x0, HALF, T, get(w, `${prefix}.pre.weight`).data, get(w, `${prefix}.pre.bias`).data, C);
  const hw = wn(h, T, w, `${prefix}.enc`);
  const m = conv1x1(hw, C, T, get(w, `${prefix}.post.weight`).data, get(w, `${prefix}.post.bias`).data, HALF);
  const out = new Float32Array(C * T);
  out.set(x0, 0);
  for (let n = 0; n < HALF * T; n++) out[HALF * T + n] = x1[n] - m[n]; // logs=0 ⇒ subtract mean
  return out;
}

/** Reverse channel order (VITS Flip). */
function flipChannels(x: Float32Array, C: number, T: number): Float32Array {
  const out = new Float32Array(C * T);
  for (let c = 0; c < C; c++) {
    const src = (C - 1 - c) * T;
    out.set(x.subarray(src, src + T), c * T);
  }
  return out;
}

/**
 * Flow decoder in reverse (VITS ResidualCouplingBlock, reverse=True). flows =
 * [C0, Flip, C2, Flip, C4, Flip, C6, Flip] applied in reverse:
 * Flip, C6, Flip, C4, Flip, C2, Flip, C0.
 */
export function flowReverse(zp: Float32Array, F: number, w: PiperWeights): Float32Array {
  const C = 192;
  let x: Float32Array = zp.slice();
  for (const idx of [6, 4, 2, 0]) {
    x = flipChannels(x, C, F);
    x = couplingReverse(x, F, w, `flow.flows.${idx}`);
  }
  return x;
}

// ─── dp: stochastic duration predictor (VITS StochasticDurationPredictor) ─────
//
// Run in REVERSE at inference. With scales=[*,*,0] the duration noise noise_w=0,
// so the flow input z = zeros(2, T) and the whole predictor is deterministic.
// The reversed flow list is [Flip, CF7, Flip, CF5, Flip, CF3, Flip, EA0] — VITS
// drops the second-to-last (CF1) via `flows[:-2] + flows[-1:]`, which is why the
// converter never emitted dp.flows.1 weights.

/** erf via Abramowitz & Stegun 7.1.26 (max abs err ~1.5e-7). */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
  return sign * y;
}

/** Exact-erf GELU (F.gelu default): 0.5·x·(1 + erf(x/√2)). */
function geluInPlace(x: Float32Array): void {
  const INV_SQRT2 = 0.7071067811865476;
  for (let n = 0; n < x.length; n++) x[n] = 0.5 * x[n] * (1 + erf(x[n] * INV_SQRT2));
}

/** Numerically-stable softplus. */
function softplus(x: number): number {
  return x > 20 ? x : Math.log1p(Math.exp(x));
}

/** Depthwise conv1d (groups=C): weight [C, 1, K], symmetric pad, dilation. */
function depthwiseConv1d(
  x: Float32Array, C: number, T: number,
  weight: Float32Array, bias: Float32Array, K: number, pad: number, dilation: number,
): Float32Array {
  const out = new Float32Array(C * T);
  for (let c = 0; c < C; c++) {
    const wBase = c * K, b = bias[c], xBase = c * T;
    for (let t = 0; t < T; t++) {
      let acc = b;
      const start = t - pad;
      for (let k = 0; k < K; k++) {
        const ti = start + k * dilation;
        if (ti >= 0 && ti < T) acc += weight[wBase + k] * x[xBase + ti];
      }
      out[c * T + t] = acc;
    }
  }
  return out;
}

/**
 * DDSConv (dilated depthwise-separable conv, VITS modules.DDSConv): 3 layers,
 * channels 192, kernel 3, dilation 3^i, exact-erf gelu, LayerNorm ×2, residual.
 * Optional conditioning g added once at the start.
 */
function ddsconv(
  x: Float32Array, T: number, w: PiperWeights, prefix: string, g: Float32Array | null,
): Float32Array {
  const C = 192, K = 3, LAYERS = 3;
  const out = x.slice();
  if (g) for (let n = 0; n < C * T; n++) out[n] += g[n];
  for (let i = 0; i < LAYERS; i++) {
    const dil = 3 ** i;
    let y = depthwiseConv1d(
      out, C, T,
      get(w, `${prefix}.convs_sep.${i}.weight`).data,
      get(w, `${prefix}.convs_sep.${i}.bias`).data, K, dil, dil,
    );
    y = layerNormCh(y, C, T, get(w, `${prefix}.norms_1.${i}.gamma`).data, get(w, `${prefix}.norms_1.${i}.beta`).data);
    geluInPlace(y);
    y = conv1x1(y, C, T, get(w, `${prefix}.convs_1x1.${i}.weight`).data, get(w, `${prefix}.convs_1x1.${i}.bias`).data, C);
    y = layerNormCh(y, C, T, get(w, `${prefix}.norms_2.${i}.gamma`).data, get(w, `${prefix}.norms_2.${i}.beta`).data);
    geluInPlace(y);
    for (let n = 0; n < C * T; n++) out[n] += y[n];
  }
  return out;
}

/**
 * Inverse of one piecewise rational-quadratic spline element (linear tails,
 * 10 bins, tail_bound 5) — VITS/nflows rational_quadratic_spline(inverse=True).
 * uw/uh: 10 unnormalized widths/heights (already /√192); ud: 9 derivatives.
 */
function rqsInverse(input: number, uw: Float32Array, uh: Float32Array, ud: Float32Array): number {
  const TAIL = 5, NB = 10, minBW = 1e-3, minBH = 1e-3, minD = 1e-3;
  if (input < -TAIL || input > TAIL) return input; // linear tail = identity

  // widths → cumwidths [11] scaled to [-5, 5]
  const wsm = softmaxVec(uw, NB);
  const CW = new Float32Array(NB + 1);
  let cw = 0;
  for (let i = 0; i < NB; i++) { cw += minBW + (1 - minBW * NB) * wsm[i]; CW[i + 1] = 2 * TAIL * cw - TAIL; }
  CW[0] = -TAIL; CW[NB] = TAIL;

  // heights → cumheights [11]
  const hsm = softmaxVec(uh, NB);
  const CH = new Float32Array(NB + 1);
  let ch = 0;
  for (let i = 0; i < NB; i++) { ch += minBH + (1 - minBH * NB) * hsm[i]; CH[i + 1] = 2 * TAIL * ch - TAIL; }
  CH[0] = -TAIL; CH[NB] = TAIL;

  // derivatives: pad ud (9) → 11 with the tail constant, then minD + softplus
  const CONST = Math.log(Math.exp(1 - minD) - 1);
  const der = new Float32Array(NB + 1);
  der[0] = minD + softplus(CONST);
  der[NB] = minD + softplus(CONST);
  for (let i = 0; i < NB - 1; i++) der[i + 1] = minD + softplus(ud[i]);

  // searchsorted(cumheights, input) − 1  (eps on the last edge)
  let idx = -1;
  for (let j = 0; j <= NB; j++) {
    const loc = j === NB ? CH[j] + 1e-6 : CH[j];
    if (input >= loc) idx++;
  }
  if (idx < 0) idx = 0;
  if (idx > NB - 1) idx = NB - 1;

  const icw = CW[idx], ibw = CW[idx + 1] - CW[idx];
  const ich = CH[idx], ih = CH[idx + 1] - CH[idx];
  const idelta = ih / ibw;
  const ider = der[idx], iderp1 = der[idx + 1];

  const dy = input - ich;
  const a = dy * (ider + iderp1 - 2 * idelta) + ih * (idelta - ider);
  const b = ih * ider - dy * (ider + iderp1 - 2 * idelta);
  const c = -idelta * dy;
  const disc = b * b - 4 * a * c;
  const root = (2 * c) / (-b - Math.sqrt(disc));
  return root * ibw + icw;
}

function softmaxVec(x: Float32Array, n: number): Float32Array {
  let mx = -Infinity;
  for (let i = 0; i < n; i++) mx = Math.max(mx, x[i]);
  const out = new Float32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) { const e = Math.exp(x[i] - mx); out[i] = e; sum += e; }
  for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}

/** ConvFlow reverse (VITS modules.ConvFlow, reverse=True), 2-channel z [2, T]. */
function convFlowReverse(z: Float32Array, T: number, w: PiperWeights, prefix: string, g: Float32Array): Float32Array {
  const FC = 192, NB = 10;
  const sq = Math.sqrt(FC);
  const x0 = z.subarray(0, T);      // channel 0
  const x1 = z.subarray(T, 2 * T);  // channel 1 (transformed)
  let h = conv1x1(x0, 1, T, get(w, `${prefix}.pre.weight`).data, get(w, `${prefix}.pre.bias`).data, FC);
  h = ddsconv(h, T, w, `${prefix}.convs`, g);
  h = conv1x1(h, FC, T, get(w, `${prefix}.proj.weight`).data, get(w, `${prefix}.proj.bias`).data, 3 * NB - 1); // [29, T]

  const out = new Float32Array(2 * T);
  out.set(x0, 0);
  const uw = new Float32Array(NB), uh = new Float32Array(NB), ud = new Float32Array(NB - 1);
  for (let t = 0; t < T; t++) {
    for (let k = 0; k < NB; k++) { uw[k] = h[k * T + t] / sq; uh[k] = h[(NB + k) * T + t] / sq; }
    for (let k = 0; k < NB - 1; k++) ud[k] = h[(2 * NB + k) * T + t];
    out[T + t] = rqsInverse(x1[t], uw, uh, ud);
  }
  return out;
}

/** ElementwiseAffine reverse: z_c = (z_c − m_c)·mul_const_c (mul_const = exp(−logs)). */
function eaReverse(z: Float32Array, T: number, w: PiperWeights): Float32Array {
  const m = get(w, 'dp.flows.0.m').data;               // [2, 1]
  const mul = get(w, 'dp.flows.0.mul_const').data;     // [2, 1]
  const out = new Float32Array(2 * T);
  for (let c = 0; c < 2; c++)
    for (let t = 0; t < T; t++) out[c * T + t] = (z[c * T + t] - m[c]) * mul[c];
  return out;
}

/** Swap the 2 channels of z [2, T] (VITS Flip on a 2-channel tensor). */
function flip2(z: Float32Array, T: number): Float32Array {
  const out = new Float32Array(2 * T);
  out.set(z.subarray(T, 2 * T), 0);
  out.set(z.subarray(0, T), T);
  return out;
}

/**
 * Stochastic duration predictor in reverse. `encX` is the enc_p encoder output
 * [192, T]. The flow input is z = randn(2, T)·noiseW (VITS noise_scale_w);
 * noiseW = 0 ⇒ z = 0 and the predictor is deterministic (fixture path).
 * Returns integer durations d_t = ceil(exp(logw_t) · lengthScale).
 */
export function dpReverse(
  encX: Float32Array, T: number, w: PiperWeights,
  lengthScale = 1, noiseW = 0, randn?: () => number,
): number[] {
  const C = 192;
  // conditioning: pre → DDSConv → proj
  let cond: Float32Array = conv1x1(encX, C, T, get(w, 'dp.pre.weight').data, get(w, 'dp.pre.bias').data, C);
  cond = ddsconv(cond, T, w, 'dp.convs', null);
  cond = conv1x1(cond, C, T, get(w, 'dp.proj.weight').data, get(w, 'dp.proj.bias').data, C);

  let z: Float32Array = new Float32Array(2 * T); // noise_w = 0 ⇒ zeros
  if (noiseW > 0 && randn) for (let n = 0; n < 2 * T; n++) z[n] = randn() * noiseW;
  z = flip2(z, T);
  z = convFlowReverse(z, T, w, 'dp.flows.7', cond);
  z = flip2(z, T);
  z = convFlowReverse(z, T, w, 'dp.flows.5', cond);
  z = flip2(z, T);
  z = convFlowReverse(z, T, w, 'dp.flows.3', cond);
  z = flip2(z, T);
  z = eaReverse(z, T, w);

  const dur = new Array<number>(T);
  for (let t = 0; t < T; t++) dur[t] = Math.ceil(Math.exp(z[t]) * lengthScale);
  return dur;
}

// ─── dec: HiFiGAN generator (CPU reference for the WGSL kernel) ───────────────
//
// Standard HiFiGAN: conv_pre → 3 × (leaky_relu, ConvTranspose1d upsample, MRF
// of 3 ResBlock2 / 3) → leaky_relu → conv_post → tanh. Upsample 8·8·4 = 256 =
// hop_length, so waveform length = F·256. This is the ~95%-of-FLOPs module; the
// GPU port (conv1d_gen.wgsl) is validated against this reference.

const LRELU = 0.1; // HiFiGAN LRELU_SLOPE (loop + resblocks)

function leakyReluInPlace(x: Float32Array, slope = LRELU): void {
  for (let n = 0; n < x.length; n++) if (x[n] < 0) x[n] *= slope;
}

function getPadding(k: number, d: number): number {
  return Math.floor((k * d - d) / 2);
}

/**
 * ConvTranspose1d (scatter form). input [Cin, Tin], weight [Cin, Cout, K],
 * bias [Cout] | null, stride S, symmetric padding P, output_padding 0.
 * Tout = (Tin−1)·S − 2·P + K.
 */
function convTranspose1d(
  x: Float32Array, Cin: number, Tin: number,
  weight: Float32Array, bias: Float32Array | null,
  Cout: number, K: number, S: number, P: number,
): { data: Float32Array; Tout: number } {
  const Tout = (Tin - 1) * S - 2 * P + K;
  const out = new Float32Array(Cout * Tout);
  if (bias) for (let co = 0; co < Cout; co++) out.fill(bias[co], co * Tout, (co + 1) * Tout);
  for (let ci = 0; ci < Cin; ci++) {
    const xBase = ci * Tin;
    for (let ti = 0; ti < Tin; ti++) {
      const xv = x[xBase + ti];
      if (xv === 0) continue;
      const base = ti * S - P;
      for (let co = 0; co < Cout; co++) {
        const wc = (ci * Cout + co) * K;
        const oBase = co * Tout;
        for (let k = 0; k < K; k++) {
          const to = base + k;
          if (to >= 0 && to < Tout) out[oBase + to] += weight[wc + k] * xv;
        }
      }
    }
  }
  return { data: out, Tout };
}

/** ResBlock2: 2 convs, x = x + conv_i(leaky_relu(x)); dilations [d0, d1]. */
function resBlock(
  x: Float32Array, C: number, T: number, w: PiperWeights, prefix: string, K: number, dils: number[],
): Float32Array {
  let out = x;
  for (let i = 0; i < 2; i++) {
    const xt = out.slice();
    leakyReluInPlace(xt);
    const d = dils[i], p = getPadding(K, d);
    const c = conv1d(
      xt, C, T,
      get(w, `${prefix}.convs.${i}.weight`).data,
      get(w, `${prefix}.convs.${i}.bias`).data, C, K, p, p, d,
    ).data;
    const nxt = new Float32Array(C * T);
    for (let n = 0; n < C * T; n++) nxt[n] = out[n] + c[n];
    out = nxt;
  }
  return out;
}

/**
 * HiFiGAN generator forward. `z` is the flow output [192, F]. Returns the mono
 * waveform Float32Array of length F·256 in [-1, 1].
 */
export function decForward(z: Float32Array, F: number, w: PiperWeights): Float32Array {
  const RES_K = [3, 5, 7];
  const RES_D = [[1, 2], [2, 6], [3, 12]];
  const UPS_K = [16, 16, 8], UPS_S = [8, 8, 4];
  const UPS_CIN = [256, 128, 64], UPS_COUT = [128, 64, 32];

  // conv_pre: 192 → 256, k7 pad3
  let x: Float32Array = conv1d(z, 192, F, get(w, 'dec.conv_pre.weight').data, get(w, 'dec.conv_pre.bias').data, 256, 7, 3, 3).data;
  let T = F;

  for (let i = 0; i < 3; i++) {
    leakyReluInPlace(x);
    const Cin = UPS_CIN[i], Cout = UPS_COUT[i], K = UPS_K[i], S = UPS_S[i];
    const up = convTranspose1d(x, Cin, T, get(w, `dec.ups.${i}.weight`).data, get(w, `dec.ups.${i}.bias`).data, Cout, K, S, Math.floor((K - S) / 2));
    x = up.data; T = up.Tout;
    // MRF: mean of 3 resblocks
    let xs: Float32Array | null = null;
    for (let j = 0; j < 3; j++) {
      const rb = resBlock(x, Cout, T, w, `dec.resblocks.${i * 3 + j}`, RES_K[j], RES_D[j]);
      if (xs === null) xs = rb;
      else for (let n = 0; n < Cout * T; n++) xs[n] += rb[n];
    }
    x = xs!;
    for (let n = 0; n < Cout * T; n++) x[n] /= 3;
  }

  leakyReluInPlace(x, 0.01); // final leaky uses F.leaky_relu default slope (0.01), not 0.1
  // conv_post: 32 → 1, k7 pad3, NO bias
  const y = conv1d(x, 32, T, get(w, 'dec.conv_post.weight').data, null, 1, 7, 3, 3).data;
  for (let n = 0; n < y.length; n++) y[n] = Math.tanh(y[n]);
  return y;
}

// ─── Public synthesis API ────────────────────────────────────────────────────

/** en_US-joe-medium output sample rate (config.audio.sample_rate). */
export const SAMPLE_RATE = 22050;

/** Deterministic PRNG (mulberry32) → uniform [0, 1). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard-normal sampler (Marsaglia polar) driven by a uniform source. */
function gaussianSampler(uniform: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0, v = 0, s = 0;
    do { u = uniform() * 2 - 1; v = uniform() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
    const mul = Math.sqrt(-2 * Math.log(s) / s);
    spare = v * mul;
    return u * mul;
  };
}

export interface SynthOptions {
  /** z_p Gaussian noise scale (VITS noise_scale). Default 0.667. 0 ⇒ z_p = m_p. */
  noiseScale?: number;
  /** Duration multiplier (VITS length_scale). Default 1. >1 slower, <1 faster. */
  lengthScale?: number;
  /** Duration-predictor noise scale (VITS noise_scale_w). Default 0.8. */
  noiseW?: number;
  /** PRNG seed for reproducibility. Omit for a fresh random voice each call. */
  seed?: number;
  /** Waveform decoder. Default = CPU decForward; pass the GPU PiperDecGpu here. */
  decode?: (z: Float32Array, F: number) => Float32Array | Promise<Float32Array>;
}

export interface SynthResult {
  audio: Float32Array;   // mono f32 in [-1, 1]
  sampleRate: number;    // SAMPLE_RATE
  durations: number[];   // per-phoneme frame counts
  F: number;             // total frames (audio.length / hop_length)
}

/**
 * Full VITS synthesis: phoneme ids → waveform. Ties enc_p → dp → flow → dec.
 * With noiseScale = 0 and noiseW = 0 this reduces exactly to the zero-noise
 * fixture path (the parity-gated pipeline). The decoder is pluggable so the
 * caller can use the CPU reference (default) or the GPU PiperDecGpu.
 */
export async function synthesize(
  ids: number[], w: PiperWeights, opts: SynthOptions = {},
): Promise<SynthResult> {
  const noiseScale = opts.noiseScale ?? 0.667;
  const lengthScale = opts.lengthScale ?? 1;
  const noiseW = opts.noiseW ?? 0.8;
  const decode = opts.decode ?? ((z, F) => decForward(z, F, w));
  const seed = opts.seed ?? (Math.random() * 0x100000000) >>> 0;
  const randn = gaussianSampler(mulberry32(seed));

  const enc = encP(ids, w);
  const durations = dpReverse(enc.x, enc.T, w, lengthScale, noiseW, randn);
  const { data: mP, F } = expandByDuration(enc.m, 192, enc.T, durations);

  let zp: Float32Array = mP;
  if (noiseScale > 0) {
    const { data: logsP } = expandByDuration(enc.logs, 192, enc.T, durations);
    zp = new Float32Array(192 * F);
    for (let n = 0; n < zp.length; n++) zp[n] = mP[n] + randn() * Math.exp(logsP[n]) * noiseScale;
  }

  const z = flowReverse(zp, F, w);
  const audio = await decode(z, F);
  return { audio, sampleRate: SAMPLE_RATE, durations, F };
}
