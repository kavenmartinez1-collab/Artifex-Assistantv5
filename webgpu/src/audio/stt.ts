/**
 * STT orchestrator (browser) — Phase W6.
 *
 * One call that ties the parity-gated pieces together:
 *   16 kHz mono f32 → logMelSpectrogram (W1) → encode (W2)
 *                   → greedyDecode (W3+W4) → tokenizer.decode → text.
 *
 * The model (~290 MB f32 weights + mel LUT + greedy config) and the GPT2-BPE
 * tokenizer are loaded once and cached across calls; re-parsing either per
 * utterance would be wasteful. Detokenization reuses the engine's existing
 * @huggingface/transformers tokenizer (loads the copied tokenizer.json).
 */
import { loadWhisper, type WhisperModel } from './whisper-loader';
import { logMelSpectrogram, encode, greedyDecode } from './whisper';
import { createTokenizer, type Tokenizer } from '../model/tokenizer';

let modelPromise: Promise<WhisperModel> | null = null;
let tokenizerPromise: Promise<Tokenizer> | null = null;

/** True once the STT model has been requested (loading or resident). */
export function isSttLoaded(): boolean {
  return modelPromise !== null;
}

async function ensureModel(
  onProgress?: (loaded: number, total: number) => void,
): Promise<WhisperModel> {
  if (!modelPromise) {
    modelPromise = loadWhisper(onProgress).catch((err) => {
      modelPromise = null; // allow a retry after a failed load
      throw err;
    });
  }
  return modelPromise;
}

async function ensureTokenizer(): Promise<Tokenizer> {
  if (!tokenizerPromise) {
    tokenizerPromise = createTokenizer({ modelId: 'local/whisper-base-en' }).catch((err) => {
      tokenizerPromise = null;
      throw err;
    });
  }
  return tokenizerPromise;
}

/**
 * Transcribe a 16 kHz mono f32 [-1,1] clip to text. `onProgress` reports the
 * one-time model download (bytes loaded / total).
 */
export async function transcribe(
  audio16k: Float32Array,
  onProgress?: (loaded: number, total: number) => void,
): Promise<string> {
  const [model, tok] = await Promise.all([ensureModel(onProgress), ensureTokenizer()]);
  const mel = logMelSpectrogram(audio16k, model.melFilters);
  const enc = encode(mel, model.weights);
  const ids = greedyDecode(enc, model.weights, model.greedy);
  const tail = ids.slice(model.greedy.forcedPrefix.length);
  return tok.decode(tail).trim();
}
