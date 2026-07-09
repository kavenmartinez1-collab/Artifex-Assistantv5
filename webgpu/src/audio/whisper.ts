/**
 * Whisper-base.en STT runtime — Phases W1 (log-mel frontend) + W2 (encoder).
 *
 * CPU f32 typed-array port, gated against HF (scripts/gen_whisper_fixture.py):
 *   W1 WhisperFeatureExtractor → logMelSpectrogram
 *   W2 model.encoder           → encode  (conv stem /2 + 6 transformer layers)
 * Later phases add the decoder (W3) and greedy/BPE decode (W4); the heavy
 * transformer matmuls move to the GPU, mirroring the Piper HiFiGAN decoder.
 *
 * Weights come from scripts/convert-whisper-hf.py output
 * `webgpu/models/whisper-base-en/` (HF tensor names, all f32) plus the frozen
 * Slaney mel filterbank `mel_filters.bin` ([80, 201] row-major).
 *
 * Parity ground truth: scripts/whisper_fixture/, gate scripts/test-whisper-parity.mts.
 */

// ─── Weights ─────────────────────────────────────────────────────────────────

export interface WhisperTensor {
  shape: number[];
  data: Float32Array;
}
export type WhisperWeights = Map<string, WhisperTensor>;

function get(w: WhisperWeights, name: string): Float32Array {
  const t = w.get(name);
  if (!t) throw new Error(`whisper: missing weight ${name}`);
  return t.data;
}

// ─── Frontend constants (preprocessor_config.json / WhisperFeatureExtractor) ──

export const N_FFT = 400;
export const HOP_LENGTH = 160;
export const N_MELS = 80;
export const N_FREQ = N_FFT / 2 + 1; // 201
export const N_SAMPLES = 480_000; // 30 s @ 16 kHz
export const N_FRAMES = N_SAMPLES / HOP_LENGTH; // 3000

// Periodic Hann window (torch.hann_window default periodic=True):
// w[n] = 0.5 - 0.5*cos(2*pi*n/N), n = 0..N-1.
const HANN = (() => {
  const w = new Float32Array(N_FFT);
  for (let n = 0; n < N_FFT; n++) w[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT);
  return w;
})();

// Precomputed DFT cos/sin tables: [N_FREQ][N_FFT]. Only the 201 one-sided bins
// are needed. n_fft=400 is not a power of two, so a direct DFT is used for the
// reference — the runtime FFT moves to the GPU in a later phase.
const COS = new Float32Array(N_FREQ * N_FFT);
const SIN = new Float32Array(N_FREQ * N_FFT);
for (let k = 0; k < N_FREQ; k++) {
  for (let n = 0; n < N_FFT; n++) {
    const a = (-2 * Math.PI * k * n) / N_FFT;
    COS[k * N_FFT + n] = Math.cos(a);
    SIN[k * N_FFT + n] = Math.sin(a);
  }
}

/**
 * Pad-or-truncate `audio` to N_SAMPLES, then reflect-pad by N_FFT/2 on each side
 * (torch.stft center=True), matching WhisperFeatureExtractor exactly.
 */
function padCenter(audio: Float32Array): Float32Array {
  const base = new Float32Array(N_SAMPLES); // zeros; truncates if longer
  base.set(audio.subarray(0, Math.min(audio.length, N_SAMPLES)));
  const pad = N_FFT / 2; // 200
  const out = new Float32Array(N_SAMPLES + 2 * pad);
  out.set(base, pad);
  // reflect (no edge repeat): out[pad-1-i] = base[i+1]; mirror at the tail too
  for (let i = 0; i < pad; i++) {
    out[pad - 1 - i] = base[i + 1];
    out[pad + N_SAMPLES + i] = base[N_SAMPLES - 2 - i];
  }
  return out;
}

/**
 * Whisper log-mel spectrogram. `melFilters` is the Slaney filterbank
 * [N_MELS, N_FREQ] row-major (from mel_filters.bin). Returns [N_MELS, N_FRAMES]
 * row-major (mel-major), identical to HF input_features[0].
 */
export function logMelSpectrogram(
  audio: Float32Array,
  melFilters: Float32Array,
): Float32Array {
  const padded = padCenter(audio);

  // power spectrogram: [N_FREQ, N_FRAMES]
  const power = new Float32Array(N_FREQ * N_FRAMES);
  const frame = new Float32Array(N_FFT);
  for (let t = 0; t < N_FRAMES; t++) {
    const start = t * HOP_LENGTH;
    for (let n = 0; n < N_FFT; n++) frame[n] = padded[start + n] * HANN[n];
    for (let k = 0; k < N_FREQ; k++) {
      let re = 0;
      let im = 0;
      const base = k * N_FFT;
      for (let n = 0; n < N_FFT; n++) {
        re += frame[n] * COS[base + n];
        im += frame[n] * SIN[base + n];
      }
      power[k * N_FRAMES + t] = re * re + im * im;
    }
  }

  // mel projection: [N_MELS, N_FREQ] @ [N_FREQ, N_FRAMES] → [N_MELS, N_FRAMES]
  const mel = new Float32Array(N_MELS * N_FRAMES);
  for (let m = 0; m < N_MELS; m++) {
    const wBase = m * N_FREQ;
    for (let t = 0; t < N_FRAMES; t++) {
      let acc = 0;
      for (let k = 0; k < N_FREQ; k++) acc += melFilters[wBase + k] * power[k * N_FRAMES + t];
      mel[m * N_FRAMES + t] = acc;
    }
  }

  // log10 + clamp + normalize (WhisperFeatureExtractor):
  //   log = log10(max(mel, 1e-10)); log = max(log, log.max()-8); (log+4)/4
  let maxLog = -Infinity;
  for (let i = 0; i < mel.length; i++) {
    const v = Math.log10(Math.max(mel[i], 1e-10));
    mel[i] = v;
    if (v > maxLog) maxLog = v;
  }
  const floor = maxLog - 8.0;
  for (let i = 0; i < mel.length; i++) {
    const v = mel[i] < floor ? floor : mel[i];
    mel[i] = (v + 4.0) / 4.0;
  }
  return mel;
}

// ─── W2: audio encoder ───────────────────────────────────────────────────────

export const D_MODEL = 512;
export const ENC_LAYERS = 6;
export const ENC_HEADS = 8;
export const HEAD_DIM = D_MODEL / ENC_HEADS; // 64
export const ENC_FFN = 2048;
export const ENC_FRAMES = N_FRAMES / 2; // 1500 (conv2 stride 2)

/** Exact (erf-based) GELU, matching torch nn.GELU / ACT2FN["gelu"]. */
function erf(x: number): number {
  // Abramowitz & Stegun 7.1.26, |err| < 1.5e-7
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x < 0 ? -y : y;
}
function gelu(x: number): number {
  return 0.5 * x * (1 + erf(x / Math.SQRT2));
}

/** LayerNorm over the last (feature) dim of a [T, C] row-major activation. */
function layerNorm(
  x: Float32Array, T: number, C: number,
  gamma: Float32Array, beta: Float32Array, eps = 1e-5,
): Float32Array {
  const out = new Float32Array(T * C);
  for (let t = 0; t < T; t++) {
    const off = t * C;
    let mean = 0;
    for (let c = 0; c < C; c++) mean += x[off + c];
    mean /= C;
    let varr = 0;
    for (let c = 0; c < C; c++) {
      const d = x[off + c] - mean;
      varr += d * d;
    }
    varr /= C;
    const inv = 1 / Math.sqrt(varr + eps);
    for (let c = 0; c < C; c++) out[off + c] = (x[off + c] - mean) * inv * gamma[c] + beta[c];
  }
  return out;
}

/** y = x @ W^T + b. x [T, Cin], W [Cout, Cin] row-major, b [Cout] or null. */
function linear(
  x: Float32Array, T: number, Cin: number,
  W: Float32Array, b: Float32Array | null, Cout: number,
): Float32Array {
  const out = new Float32Array(T * Cout);
  for (let t = 0; t < T; t++) {
    const xoff = t * Cin;
    const ooff = t * Cout;
    for (let o = 0; o < Cout; o++) {
      let acc = b ? b[o] : 0;
      const woff = o * Cin;
      for (let i = 0; i < Cin; i++) acc += x[xoff + i] * W[woff + i];
      out[ooff + o] = acc;
    }
  }
  return out;
}

/**
 * conv1d over channel-first [Cin, T], weight [Cout, Cin, K] flat, symmetric
 * pad `pad` (zeros), stride. Returns { data [Cout, Tout], Tout }.
 */
function conv1dCF(
  x: Float32Array, Cin: number, T: number,
  weight: Float32Array, bias: Float32Array, Cout: number,
  K: number, pad: number, stride: number,
): { data: Float32Array; Tout: number } {
  const Tout = Math.floor((T + 2 * pad - K) / stride) + 1;
  const out = new Float32Array(Cout * Tout);
  for (let co = 0; co < Cout; co++) {
    const wc = co * Cin * K;
    for (let ot = 0; ot < Tout; ot++) {
      let acc = bias[co];
      const start = ot * stride - pad;
      for (let ci = 0; ci < Cin; ci++) {
        const xrow = ci * T;
        const wrow = wc + ci * K;
        for (let k = 0; k < K; k++) {
          const it = start + k;
          if (it >= 0 && it < T) acc += x[xrow + it] * weight[wrow + k];
        }
      }
      out[co * Tout + ot] = acc;
    }
  }
  return { data: out, Tout };
}

/**
 * Bidirectional multi-head self-attention (no mask) over [T, D]. Whisper's
 * q_proj scales by head_dim^-0.5; k_proj has NO bias.
 */
function selfAttention(
  x: Float32Array, T: number, p: string, w: WhisperWeights,
): Float32Array {
  const q = linear(x, T, D_MODEL, get(w, `${p}.q_proj.weight`), get(w, `${p}.q_proj.bias`), D_MODEL);
  const k = linear(x, T, D_MODEL, get(w, `${p}.k_proj.weight`), null, D_MODEL);
  const v = linear(x, T, D_MODEL, get(w, `${p}.v_proj.weight`), get(w, `${p}.v_proj.bias`), D_MODEL);
  const scale = 1 / Math.sqrt(HEAD_DIM);
  const ctx = new Float32Array(T * D_MODEL);
  const scores = new Float32Array(T);
  for (let h = 0; h < ENC_HEADS; h++) {
    const hoff = h * HEAD_DIM;
    for (let ti = 0; ti < T; ti++) {
      const qoff = ti * D_MODEL + hoff;
      let maxS = -Infinity;
      for (let tj = 0; tj < T; tj++) {
        const koff = tj * D_MODEL + hoff;
        let s = 0;
        for (let d = 0; d < HEAD_DIM; d++) s += q[qoff + d] * k[koff + d];
        s *= scale;
        scores[tj] = s;
        if (s > maxS) maxS = s;
      }
      let sum = 0;
      for (let tj = 0; tj < T; tj++) {
        const e = Math.exp(scores[tj] - maxS);
        scores[tj] = e;
        sum += e;
      }
      const inv = 1 / sum;
      const coff = ti * D_MODEL + hoff;
      for (let tj = 0; tj < T; tj++) {
        const a = scores[tj] * inv;
        const voff = tj * D_MODEL + hoff;
        for (let d = 0; d < HEAD_DIM; d++) ctx[coff + d] += a * v[voff + d];
      }
    }
  }
  return linear(ctx, T, D_MODEL, get(w, `${p}.out_proj.weight`), get(w, `${p}.out_proj.bias`), D_MODEL);
}

/**
 * Whisper audio encoder. `mel` is [N_MELS, N_FRAMES] row-major (logMelSpectrogram
 * output). Returns encoder hidden states [ENC_FRAMES, D_MODEL] row-major,
 * matching HF model.encoder(...).last_hidden_state[0].
 */
export function encode(mel: Float32Array, w: WhisperWeights): Float32Array {
  const E = 'model.encoder';
  // conv stem (channel-first): gelu(conv1) → gelu(conv2, stride 2)
  const c1 = conv1dCF(mel, N_MELS, N_FRAMES, get(w, `${E}.conv1.weight`), get(w, `${E}.conv1.bias`), D_MODEL, 3, 1, 1);
  for (let i = 0; i < c1.data.length; i++) c1.data[i] = gelu(c1.data[i]);
  const c2 = conv1dCF(c1.data, D_MODEL, c1.Tout, get(w, `${E}.conv2.weight`), get(w, `${E}.conv2.bias`), D_MODEL, 3, 1, 2);
  for (let i = 0; i < c2.data.length; i++) c2.data[i] = gelu(c2.data[i]);
  const T = c2.Tout; // 1500

  // transpose [D, T] → [T, D] and add sinusoidal position embeddings
  const pos = get(w, `${E}.embed_positions.weight`);
  let x = new Float32Array(T * D_MODEL);
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < D_MODEL; c++) x[t * D_MODEL + c] = c2.data[c * T + t] + pos[t * D_MODEL + c];
  }

  // 6 pre-LN transformer layers
  for (let l = 0; l < ENC_LAYERS; l++) {
    const L = `${E}.layers.${l}`;
    const n1 = layerNorm(x, T, D_MODEL, get(w, `${L}.self_attn_layer_norm.weight`), get(w, `${L}.self_attn_layer_norm.bias`));
    const attn = selfAttention(n1, T, `${L}.self_attn`, w);
    for (let i = 0; i < x.length; i++) x[i] += attn[i];
    const n2 = layerNorm(x, T, D_MODEL, get(w, `${L}.final_layer_norm.weight`), get(w, `${L}.final_layer_norm.bias`));
    const h1 = linear(n2, T, D_MODEL, get(w, `${L}.fc1.weight`), get(w, `${L}.fc1.bias`), ENC_FFN);
    for (let i = 0; i < h1.length; i++) h1[i] = gelu(h1[i]);
    const h2 = linear(h1, T, ENC_FFN, get(w, `${L}.fc2.weight`), get(w, `${L}.fc2.bias`), D_MODEL);
    for (let i = 0; i < x.length; i++) x[i] += h2[i];
  }

  return layerNorm(x, T, D_MODEL, get(w, `${E}.layer_norm.weight`), get(w, `${E}.layer_norm.bias`));
}

// ─── W3: text decoder ────────────────────────────────────────────────────────

export const DEC_LAYERS = 6;
export const DEC_HEADS = 8;
export const MAX_TARGET = 448;

/**
 * General multi-head attention: queries from `qIn` [Tq, D], keys/values from
 * `kvIn` [Tkv, D]. `causal` masks future keys (self-attn); false for cross-attn.
 * k_proj has no bias (Whisper). Precomputed K/V (`kProj`/`vProj`) may be supplied
 * to reuse a cross-attn projection across decoder positions.
 */
function mha(
  qIn: Float32Array, Tq: number,
  kvIn: Float32Array | null, Tkv: number, p: string, w: WhisperWeights,
  causal: boolean, kvCache?: { k: Float32Array; v: Float32Array },
): Float32Array {
  const q = linear(qIn, Tq, D_MODEL, get(w, `${p}.q_proj.weight`), get(w, `${p}.q_proj.bias`), D_MODEL);
  const k = kvCache ? kvCache.k : linear(kvIn!, Tkv, D_MODEL, get(w, `${p}.k_proj.weight`), null, D_MODEL);
  const v = kvCache ? kvCache.v : linear(kvIn!, Tkv, D_MODEL, get(w, `${p}.v_proj.weight`), get(w, `${p}.v_proj.bias`), D_MODEL);
  const scale = 1 / Math.sqrt(HEAD_DIM);
  const ctx = new Float32Array(Tq * D_MODEL);
  const scores = new Float32Array(Tkv);
  for (let h = 0; h < DEC_HEADS; h++) {
    const hoff = h * HEAD_DIM;
    for (let ti = 0; ti < Tq; ti++) {
      const qoff = ti * D_MODEL + hoff;
      const last = causal ? ti : Tkv - 1;
      let maxS = -Infinity;
      for (let tj = 0; tj <= last; tj++) {
        const koff = tj * D_MODEL + hoff;
        let s = 0;
        for (let d = 0; d < HEAD_DIM; d++) s += q[qoff + d] * k[koff + d];
        s *= scale;
        scores[tj] = s;
        if (s > maxS) maxS = s;
      }
      let sum = 0;
      for (let tj = 0; tj <= last; tj++) {
        const e = Math.exp(scores[tj] - maxS);
        scores[tj] = e;
        sum += e;
      }
      const inv = 1 / sum;
      const coff = ti * D_MODEL + hoff;
      for (let tj = 0; tj <= last; tj++) {
        const a = scores[tj] * inv;
        const voff = tj * D_MODEL + hoff;
        for (let d = 0; d < HEAD_DIM; d++) ctx[coff + d] += a * v[voff + d];
      }
    }
  }
  return linear(ctx, Tq, D_MODEL, get(w, `${p}.out_proj.weight`), get(w, `${p}.out_proj.bias`), D_MODEL);
}

/** Decoder trunk: token ids + encoder states → final hidden [T, D_MODEL]. */
function decoderHidden(ids: number[], enc: Float32Array, w: WhisperWeights): Float32Array {
  const D = 'model.decoder';
  const T = ids.length;
  const embed = get(w, `${D}.embed_tokens.weight`);
  const pos = get(w, `${D}.embed_positions.weight`);
  const x = new Float32Array(T * D_MODEL);
  for (let t = 0; t < T; t++) {
    const e = ids[t] * D_MODEL;
    const pOff = t * D_MODEL;
    for (let c = 0; c < D_MODEL; c++) x[t * D_MODEL + c] = embed[e + c] + pos[pOff + c];
  }
  for (let l = 0; l < DEC_LAYERS; l++) {
    const L = `${D}.layers.${l}`;
    const n1 = layerNorm(x, T, D_MODEL, get(w, `${L}.self_attn_layer_norm.weight`), get(w, `${L}.self_attn_layer_norm.bias`));
    const sa = mha(n1, T, n1, T, `${L}.self_attn`, w, true);
    for (let i = 0; i < x.length; i++) x[i] += sa[i];
    const n2 = layerNorm(x, T, D_MODEL, get(w, `${L}.encoder_attn_layer_norm.weight`), get(w, `${L}.encoder_attn_layer_norm.bias`));
    const ca = mha(n2, T, enc, ENC_FRAMES, `${L}.encoder_attn`, w, false);
    for (let i = 0; i < x.length; i++) x[i] += ca[i];
    const n3 = layerNorm(x, T, D_MODEL, get(w, `${L}.final_layer_norm.weight`), get(w, `${L}.final_layer_norm.bias`));
    const h1 = linear(n3, T, D_MODEL, get(w, `${L}.fc1.weight`), get(w, `${L}.fc1.bias`), ENC_FFN);
    for (let i = 0; i < h1.length; i++) h1[i] = gelu(h1[i]);
    const h2 = linear(h1, T, ENC_FFN, get(w, `${L}.fc2.weight`), get(w, `${L}.fc2.bias`), D_MODEL);
    for (let i = 0; i < x.length; i++) x[i] += h2[i];
  }
  return layerNorm(x, T, D_MODEL, get(w, `${D}.layer_norm.weight`), get(w, `${D}.layer_norm.bias`));
}

/** Project a single hidden row [D_MODEL] → logits [vocab] (tied embedding). */
function projectRow(hiddenRow: Float32Array, embed: Float32Array, vocab: number): Float32Array {
  const logits = new Float32Array(vocab);
  for (let vtok = 0; vtok < vocab; vtok++) {
    let acc = 0;
    const eoff = vtok * D_MODEL;
    for (let c = 0; c < D_MODEL; c++) acc += hiddenRow[c] * embed[eoff + c];
    logits[vtok] = acc;
  }
  return logits;
}

/**
 * Whisper text decoder. `ids` is the token prefix, `enc` the encoder hidden
 * states [ENC_FRAMES, D_MODEL]. Returns logits [ids.length, vocab] (tied
 * embedding projection), matching HF proj_out(model.decoder(...)).
 */
export function decode(ids: number[], enc: Float32Array, w: WhisperWeights): Float32Array {
  const embed = get(w, 'model.decoder.embed_tokens.weight');
  const vocab = w.get('model.decoder.embed_tokens.weight')!.shape[0];
  const hidden = decoderHidden(ids, enc, w);
  const T = ids.length;
  const logits = new Float32Array(T * vocab);
  for (let t = 0; t < T; t++) {
    const row = projectRow(hidden.subarray(t * D_MODEL, (t + 1) * D_MODEL), embed, vocab);
    logits.set(row, t * vocab);
  }
  return logits;
}

// ─── W4: greedy decode ───────────────────────────────────────────────────────

export interface GreedyConfig {
  /** Forced start tokens, e.g. [50257 (<|sot|>), 50362 (<|notimestamps|>)]. */
  forcedPrefix: number[];
  /** Token ids masked to -inf at EVERY step (generation_config.suppress_tokens). */
  suppress: number[];
  /** Token ids masked only at the FIRST generated step (begin_suppress_tokens). */
  beginSuppress: number[];
  eosTokenId: number;
  maxNewTokens: number;
}

/**
 * Greedy autoregressive decode, replicating HF model.generate() for base.en
 * (num_beams=1, do_sample=false): forced prefix, suppress_tokens every step,
 * begin_suppress_tokens at the first generated step, stop at eos. Returns the
 * full token sequence INCLUDING the forced prefix (matches generate() output).
 *
 * Recomputes the decoder over the growing prefix each step (no KV cache) — a
 * correct reference; the runtime caches K/V and moves matmuls to the GPU later.
 */
export function greedyDecode(enc: Float32Array, w: WhisperWeights, cfg: GreedyConfig): number[] {
  const embed = get(w, 'model.decoder.embed_tokens.weight');
  const vocab = w.get('model.decoder.embed_tokens.weight')!.shape[0];
  const ids = [...cfg.forcedPrefix];
  const beginIndex = ids.length;
  for (let step = 0; step < cfg.maxNewTokens; step++) {
    const hidden = decoderHidden(ids, enc, w);
    const last = hidden.subarray((ids.length - 1) * D_MODEL, ids.length * D_MODEL);
    const logits = projectRow(last, embed, vocab);
    for (const t of cfg.suppress) logits[t] = -Infinity;
    if (ids.length === beginIndex) for (const t of cfg.beginSuppress) logits[t] = -Infinity;
    let best = 0;
    let bestV = -Infinity;
    for (let v = 0; v < vocab; v++) if (logits[v] > bestV) { bestV = logits[v]; best = v; }
    if (best === cfg.eosTokenId) break;
    ids.push(best);
  }
  return ids;
}
