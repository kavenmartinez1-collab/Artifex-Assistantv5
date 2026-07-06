// FLUX.2-klein text encoder: Qwen3-4B hidden-state taps → prompt embeddings.
//
// klein does NOT use the LM head or the final norm — it taps the residual
// stream at HF hidden_states indices 9/18/27 (= after 0-indexed layers
// 8/17/26) and concatenates them per token into a (512, 7680) embedding
// that feeds the DiT's context_embedder.
//
// Template (diffusers Flux2KleinPipeline, enable_thinking=False — verified
// against the Python fixture's templated_text):
//   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
// Right-padded to 512 with <|endoftext|> (151643). The pad rows DO flow into
// the DiT; the attention_mask only lives inside the TE, which is what
// ForwardOptions.validLen implements (pad keys masked, pad query rows still
// produce output — matching HF hidden_states semantics exactly).

import type { ForwardPassEngine } from '../engine/forward-pass';

export const FLUX2_TE_MAX_SEQ = 512;
export const FLUX2_TE_PAD_ID = 151643; // <|endoftext|>
export const FLUX2_TE_TAPS = [9, 18, 27] as const;

export function flux2TemplatePrompt(prompt: string): string {
  return `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
}

export interface Flux2PromptEmbedding {
  /** [512, 7680] row-major: per token [hs9 | hs18 | hs27]. */
  promptEmbeds: Float32Array;
  /** The padded [512] token ids that produced it (parity checks). */
  inputIds: Int32Array;
  /** Number of real (non-pad) tokens. */
  validLen: number;
}

/** Encode one prompt through the Qwen3 TE. The engine must be a freshly
 *  loaded klein TE GGUF (vocab 151936); runs layers 0..26 only. */
export async function embedFlux2Prompt(
  engine: ForwardPassEngine,
  tokenizer: { encode(text: string): number[] },
  prompt: string,
): Promise<Flux2PromptEmbedding> {
  const ids = tokenizer.encode(flux2TemplatePrompt(prompt));
  if (ids.length > FLUX2_TE_MAX_SEQ) {
    throw new Error(`[Flux2 TE] prompt is ${ids.length} tokens; max ${FLUX2_TE_MAX_SEQ}`);
  }
  const validLen = ids.length;
  const tokenIds = new Uint32Array(FLUX2_TE_MAX_SEQ).fill(FLUX2_TE_PAD_ID);
  tokenIds.set(ids);

  const H = engine.config.hiddenSize; // 2560 for the klein TE
  const kv = engine.createKVCache(FLUX2_TE_MAX_SEQ);
  let taps: Map<number, Float32Array>;
  try {
    const out = await engine.forward(tokenIds, kv, {
      captureHiddenLayers: [...FLUX2_TE_TAPS],
      stopAfterLayer: 27,
      validLen,
    });
    if (!out.hiddenTaps) throw new Error('[Flux2 TE] forward returned no hiddenTaps');
    taps = out.hiddenTaps;
  } finally {
    engine.destroyKVCache(kv);
  }

  // Per-token concat [hs9 | hs18 | hs27] → (512, 3H).
  const promptEmbeds = new Float32Array(FLUX2_TE_MAX_SEQ * 3 * H);
  for (let t = 0; t < FLUX2_TE_TAPS.length; t++) {
    const hs = taps.get(FLUX2_TE_TAPS[t])!;
    for (let s = 0; s < FLUX2_TE_MAX_SEQ; s++) {
      promptEmbeds.set(hs.subarray(s * H, (s + 1) * H), s * 3 * H + t * H);
    }
  }

  return { promptEmbeds, inputIds: new Int32Array(tokenIds), validLen };
}
