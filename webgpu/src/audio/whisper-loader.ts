/**
 * Whisper voice-in loader (browser) — Phase W6.
 *
 * Fetches the whisper-base.en model the dev server surfaces as a local model
 * (webgpu/models/whisper-base-en/ → /api/hf-cache/local/…):
 *   - model.safetensors (~290 MB, all f32) → WhisperWeights map (same parse path
 *     as scripts/test-whisper-parity.mts, just fetched over HTTP)
 *   - mel_filters.bin                       → Slaney [80,201] frontend LUT
 *   - generation_config.json                → forced prefix + suppress ids (greedy)
 *
 * The tensor math + greedy decode are already parity-gated (W1-W4); this is only
 * the runtime fetch/parse glue for the browser.
 */
import { parseHeader, extractTensorData, tensorToFloat32 } from '../model/safetensors';
import { downloadFile } from '../model/hf-hub';
import type { WhisperWeights, GreedyConfig } from './whisper';

export interface WhisperModel {
  weights: WhisperWeights;
  melFilters: Float32Array;
  greedy: GreedyConfig;
}

// Fixed local asset — always served by the dev server, never the HF CDN, so we
// address it directly rather than routing through the global useLocalCache()
// toggle (which the LLM/image paths flip for their own remote/local modes).
const REPO = 'local/whisper-base-en';
const BASE = `/api/hf-cache/${REPO}`;

export async function loadWhisper(
  onProgress?: (loaded: number, total: number) => void,
): Promise<WhisperModel> {
  const gcResp = await fetch(`${BASE}/raw/main/generation_config.json`);
  if (!gcResp.ok) throw new Error(`[whisper] generation_config.json: ${gcResp.status}`);
  const gc = await gcResp.json();
  const greedy: GreedyConfig = {
    forcedPrefix: [gc.decoder_start_token_id, ...gc.forced_decoder_ids.map((p: number[]) => p[1])],
    suppress: gc.suppress_tokens ?? [],
    beginSuppress: gc.begin_suppress_tokens ?? [],
    eosTokenId: gc.eos_token_id,
    maxNewTokens: 128,
  };

  const melResp = await fetch(`${BASE}/resolve/main/mel_filters.bin`);
  if (!melResp.ok) throw new Error(`[whisper] mel_filters.bin: ${melResp.status}`);
  const melFilters = new Float32Array(await melResp.arrayBuffer());

  const ab = await downloadFile(`${BASE}/resolve/main/model.safetensors`, onProgress);
  const header = parseHeader(ab);
  const weights: WhisperWeights = new Map();
  for (const [name, info] of header.tensors) {
    const raw = extractTensorData(ab, info, header.headerByteLength);
    weights.set(name, { shape: info.shape, data: tensorToFloat32(raw, info.dtype) });
  }
  return { weights, melFilters, greedy };
}
