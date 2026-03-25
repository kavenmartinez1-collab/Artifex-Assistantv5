/**
 * Tokenizer — BPE text ↔ token ID conversion for WebGPU inference.
 *
 * Uses @huggingface/transformers for the actual BPE implementation.
 * Supports any HuggingFace tokenizer (Qwen, Llama, Mistral, etc.).
 *
 * Usage:
 *   const tok = await createTokenizer('Qwen/Qwen3.5-0.6B');
 *   const ids = tok.encode('Hello, world!');
 *   const text = tok.decode(ids);
 */

import { AutoTokenizer, type PreTrainedTokenizer } from '@huggingface/transformers';

export interface Tokenizer {
  /** Convert text to token IDs. */
  encode(text: string): number[];

  /** Convert token IDs back to text. */
  decode(ids: number[]): string;

  /** Get the vocabulary size. */
  vocabSize: number;

  /** Get special token IDs. */
  bosTokenId: number | null;
  eosTokenId: number | null;
  padTokenId: number | null;

  /** The model ID this tokenizer was loaded from. */
  modelId: string;

  /** Check if a token ID is an end-of-generation token. */
  isEos(id: number): boolean;

  /** Get the underlying HuggingFace tokenizer (for advanced use). */
  readonly inner: PreTrainedTokenizer;
}

export interface TokenizerConfig {
  /** HuggingFace model ID (e.g., 'Qwen/Qwen3.5-0.6B') */
  modelId: string;
  /**
   * Progress callback for tokenizer download.
   * Called with (loaded_bytes, total_bytes) or null for total if unknown.
   */
  onProgress?: (loaded: number, total: number | null) => void;
}

/**
 * Load a tokenizer from HuggingFace Hub.
 * Downloads tokenizer.json, tokenizer_config.json, and special_tokens_map.json.
 * These are small files (~2-5 MB total) and are cached by the browser.
 */
export async function createTokenizer(config: TokenizerConfig): Promise<Tokenizer> {
  const { modelId } = config;

  const hfTokenizer = await AutoTokenizer.from_pretrained(modelId, {
    progress_callback: config.onProgress
      ? (progress: any) => {
          if (progress && typeof progress.loaded === 'number') {
            config.onProgress!(progress.loaded, progress.total ?? null);
          }
        }
      : undefined,
  });

  // Extract special token IDs
  const bosTokenId = hfTokenizer.bos_token_id ?? null;
  const eosTokenId = hfTokenizer.eos_token_id ?? null;
  const padTokenId = hfTokenizer.pad_token_id ?? eosTokenId;

  // Build EOS token set — check multiple sources for stop tokens
  const eosTokenIds = new Set<number>();
  if (eosTokenId !== null) eosTokenIds.add(eosTokenId);

  // Also add the EOS token by encoding known stop strings
  const stopStrings = ['<|endoftext|>', '<|im_end|>', '</s>', '<|eot_id|>'];
  for (const s of stopStrings) {
    try {
      const ids = hfTokenizer.encode(s);
      // If the string encodes to a single token, it's a special token
      if (ids.length === 1) {
        eosTokenIds.add(Number(ids[0]));
      }
    } catch { /* not in vocab */ }
  }

  // Check added_tokens in the tokenizer model
  const model = hfTokenizer.model;
  if (model && 'added_tokens' in model) {
    const addedTokens = (model as any).added_tokens;
    if (Array.isArray(addedTokens)) {
      for (const token of addedTokens) {
        if (token.special && stopStrings.includes(token.content)) {
          eosTokenIds.add(token.id);
        }
      }
    }
  }

  console.log(`[Tokenizer] EOS token IDs: [${[...eosTokenIds].join(', ')}]`);

  // Get vocab size from the tokenizer model
  const vocabSize = hfTokenizer.model?.vocab?.length
    ?? (hfTokenizer as any).vocab_size
    ?? 152064; // Qwen3.5 default

  return {
    encode(text: string): number[] {
      const result = hfTokenizer.encode(text);
      // HF transformers.js returns BigInt64Array or number[] depending on version
      return Array.from(result).map(Number);
    },

    decode(ids: number[]): string {
      return hfTokenizer.decode(ids, { skip_special_tokens: false });
    },

    vocabSize,
    bosTokenId,
    eosTokenId,
    padTokenId,
    modelId,

    isEos(id: number): boolean {
      return eosTokenIds.has(id);
    },

    get inner() {
      return hfTokenizer;
    },
  };
}

/**
 * Apply the model's chat template to messages.
 * Uses the HF tokenizer's built-in template when available (model-agnostic),
 * falls back to ChatML format for models without a template.
 */
export function applyChatTemplate(
  tokenizer: Tokenizer,
  messages: Array<{ role: string; content: string }>,
): string {
  // Try the HF tokenizer's built-in template first (correct for every model)
  try {
    const result = (tokenizer.inner as any).apply_chat_template(messages, {
      add_generation_prompt: true,
      tokenize: false,
    });
    if (typeof result === 'string' && result.length > 0) {
      return result;
    }
  } catch {
    // Fall through to manual template
  }

  // Fallback: ChatML format (Qwen, SmolLM2, etc.)
  const parts: string[] = [];
  for (const msg of messages) {
    parts.push(`<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`);
  }
  parts.push('<|im_start|>assistant\n');
  return parts.join('');
}
