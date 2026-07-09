/**
 * Piper voice loader (browser) — Phase P7.
 *
 * Fetches the en_US-joe-medium voice the dev server surfaces as a local model
 * (webgpu/models/piper-en-us-joe-medium/ → /api/hf-cache/local/…):
 *   - model.safetensors (~63 MB, all f32) → PiperWeights map (same parse path
 *     as scripts/test-piper-parity.mts, just fetched over HTTP)
 *   - config.json                          → phoneme_id_map + sample_rate
 *
 * The tensor math and id table are already parity-gated (P5/P6); this is only
 * the runtime fetch/parse glue for the browser.
 */
import { parseHeader, extractTensorData, tensorToFloat32 } from '../model/safetensors';
import { downloadFile } from '../model/hf-hub';
import type { PiperWeights } from './piper';
import type { PhonemeIdMap } from './g2p';

export interface PiperVoice {
  weights: PiperWeights;
  idMap: PhonemeIdMap;
  sampleRate: number;
}

// Fixed local asset — always served by the dev server, never the HF CDN, so we
// address it directly rather than routing through the global useLocalCache()
// toggle (which the LLM/image paths flip for their own remote/local modes).
const REPO = 'local/piper-en-us-joe-medium';
const BASE = `/api/hf-cache/${REPO}`;

export async function loadPiperVoice(
  onProgress?: (loaded: number, total: number) => void,
): Promise<PiperVoice> {
  const cfgResp = await fetch(`${BASE}/raw/main/config.json`);
  if (!cfgResp.ok) throw new Error(`[piper] config.json: ${cfgResp.status}`);
  const cfg = await cfgResp.json();
  const idMap = cfg.phoneme_id_map as PhonemeIdMap;
  if (!idMap) throw new Error('[piper] config.json missing phoneme_id_map');
  const sampleRate: number = cfg.sample_rate ?? 22050;

  const ab = await downloadFile(`${BASE}/resolve/main/model.safetensors`, onProgress);
  const header = parseHeader(ab);
  const weights: PiperWeights = new Map();
  for (const [name, info] of header.tensors) {
    const raw = extractTensorData(ab, info, header.headerByteLength);
    weights.set(name, { shape: info.shape, data: tensorToFloat32(raw, info.dtype) });
  }
  return { weights, idMap, sampleRate };
}
