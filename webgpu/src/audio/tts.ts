/**
 * TTS orchestrator (browser) — Phase P7.
 *
 * One call that ties the parity-gated pieces together:
 *   text → phonemize (espeak-ng WASM, P5) → phonemesToIds (P5)
 *        → synthesize (enc_p/dp/flow on CPU + HiFiGAN on the GPU, P6)
 *        → mono f32 PCM.
 *
 * The voice (weights + phoneme id map) and the GPU decoder are loaded once and
 * cached across calls: the f32 weights are ~63 MB and the PiperDecGpu holds the
 * dec.* GPU buffers, so re-creating either per utterance would be wasteful.
 */
import { loadPiperVoice, type PiperVoice } from './piper-loader';
import { phonemize, phonemesToIds } from './g2p';
import { synthesize, type SynthOptions, type SynthResult } from './piper';
import { PiperDecGpu } from './piper-dec-gpu';

let voicePromise: Promise<PiperVoice> | null = null;
let decGpu: PiperDecGpu | null = null;
let decDevice: GPUDevice | null = null;

/** True once the voice has been requested (weights are loading or resident). */
export function isVoiceLoaded(): boolean {
  return voicePromise !== null;
}

async function ensureVoice(
  onProgress?: (loaded: number, total: number) => void,
): Promise<PiperVoice> {
  if (!voicePromise) {
    voicePromise = loadPiperVoice(onProgress).catch((err) => {
      voicePromise = null; // allow a retry after a failed load
      throw err;
    });
  }
  return voicePromise;
}

/**
 * Synthesize speech for `text`. When a GPU device is supplied the HiFiGAN
 * decoder runs on it (PiperDecGpu, ~95% of FLOPs); otherwise it falls back to
 * the CPU reference inside synthesize(). `onProgress` reports the one-time
 * voice download (bytes loaded / total).
 */
export async function speak(
  device: GPUDevice | null,
  text: string,
  opts: SynthOptions = {},
  onProgress?: (loaded: number, total: number) => void,
): Promise<SynthResult> {
  const voice = await ensureVoice(onProgress);
  const phonemes = await phonemize(text);
  const ids = phonemesToIds(phonemes, voice.idMap);

  let decode = opts.decode;
  if (!decode && device) {
    if (!decGpu || decDevice !== device) {
      decGpu?.destroy();
      decGpu = new PiperDecGpu(device, voice.weights);
      decDevice = device;
    }
    const dg = decGpu;
    decode = (z, F) => dg.forward(z, F);
  }
  return synthesize(ids, voice.weights, { ...opts, decode });
}

/** Drop the cached GPU decoder (call when the device is lost/replaced). */
export function releaseTtsDecoder(): void {
  decGpu?.destroy();
  decGpu = null;
  decDevice = null;
}
