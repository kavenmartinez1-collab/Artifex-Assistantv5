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

import { AutoTokenizer, PreTrainedTokenizer } from '@huggingface/transformers';

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
  let { modelId } = config;
  let localTokenizer: PreTrainedTokenizer | null = null;

  // Local models (local/xxx) can't use HF CDN for tokenizer — fall back to base model
  // The tokenizer is the same across quantization variants
  if (modelId.startsWith('local/')) {
    // Preferred: the model dir ships its own tokenizer files (e.g. Gemma 4
    // GGUF + tokenizer.json downloaded together) — construct directly from
    // the dev-server cache, no HF Hub round-trip (mirrors the D0 node test).
    try {
      const [tjResp, tcResp] = await Promise.all([
        fetch(`/api/hf-cache/${modelId}/raw/main/tokenizer.json`),
        fetch(`/api/hf-cache/${modelId}/raw/main/tokenizer_config.json`),
      ]);
      if (tjResp.ok && tcResp.ok) {
        localTokenizer = new PreTrainedTokenizer(await tjResp.json(), await tcResp.json());
        console.log(`[Tokenizer] Loaded from local model dir: ${modelId}`);
      }
    } catch { /* fall through to base-model heuristics */ }
  }
  if (!localTokenizer && modelId.startsWith('local/')) {
    // Try to find the base model from the tokenizer_config.json served by local cache
    try {
      const resp = await fetch(`/api/hf-cache/${modelId}/raw/main/tokenizer_config.json`);
      if (resp.ok) {
        const tc = await resp.json();
        // Use the tokenizer_class to infer the base model family
        if (tc.chat_template && (tc.eos_token === '<|im_end|>' || tc.model_type?.includes('qwen'))) {
          modelId = 'Qwen/Qwen3.5-9B'; // Qwen3.5 family shares tokenizers
          console.log(`[Tokenizer] Local model, using base tokenizer: ${modelId}`);
        }
      }
    } catch {}
    // If still local/, try a generic fallback based on known patterns
    if (modelId.startsWith('local/')) {
      const name = modelId.toLowerCase();
      // Order matters: 'qwen3.6' also contains 'qwen3'. Qwen3.6 uses a NEW
      // 248320-token vocab (EOS 248046) — the Qwen3/3.5 tokenizers are NOT
      // compatible (vocab ~151936) and produce garbage encode/decode.
      if (name.includes('qwen3.6')) modelId = 'Qwen/Qwen3.6-35B-A3B';
      else if (name.includes('qwen3.5')) modelId = 'Qwen/Qwen3.5-9B';
      else if (name.includes('qwen3')) modelId = 'Qwen/Qwen3-8B';
      else if (name.includes('qwen2.5')) modelId = 'Qwen/Qwen2.5-0.5B-Instruct';
      console.log(`[Tokenizer] Local model fallback, using tokenizer from: ${modelId}`);
    }
  }

  const hfTokenizer = localTokenizer ?? await AutoTokenizer.from_pretrained(modelId, {
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
  // '<turn|>' is Gemma 4's end-of-turn (id 106) — its eos_token <eos> (id 1)
  // is NOT what the model emits to close a chat turn.
  const stopStrings = ['<|endoftext|>', '<|im_end|>', '</s>', '<|eot_id|>', '<turn|>'];
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
 * Apply the model's chat template and return token IDs directly.
 * Uses the HF tokenizer's built-in template (model-agnostic).
 * Returns token IDs, NOT a string — this correctly handles special tokens
 * like <|im_start|> as single tokens instead of encoding them as text.
 */
/** True if `s` is a single special token in this tokenizer's vocab. */
function isSingleToken(tokenizer: Tokenizer, s: string): boolean {
  try {
    const ids = (tokenizer.inner as any).encode(s, { add_special_tokens: false });
    return ids.length === 1;
  } catch {
    return false;
  }
}

export function applyChatTemplate(
  tokenizer: Tokenizer,
  messages: Array<{ role: string; content: string }>,
  options?: { enableThinking?: boolean },
): number[] {
  const enableThinking = options?.enableThinking ?? true;

  // ── Gemma 4: <|turn>role\n ... <turn|>\n turns ─────────────────────────
  // Simplified from the GGUF's official template (no tools/multimodal):
  // thinking is enabled by a <|think|> token at the top of the FIRST system
  // turn (opened even without a system message); assistant renders as
  // 'model'; generation prompt is '<|turn>model\n'.
  if (isSingleToken(tokenizer, '<|turn>')) {
    const msgs = messages.map((m) => ({
      role: m.role === 'assistant' ? 'model' : m.role,
      content: m.content,
    }));
    if (enableThinking && msgs[0]?.role !== 'system') {
      msgs.unshift({ role: 'system', content: '' });
    }
    const think = enableThinking ? '<|think|>\\n' : '';
    const tpl =
      `{{ bos_token }}{% for message in messages %}`
      + `{% if loop.first and message['role'] == 'system' %}`
      + `{{ '<|turn>system\\n${think}' + message['content'] + '<turn|>\\n' }}`
      + `{% else %}`
      + `{{ '<|turn>' + message['role'] + '\\n' + message['content'] + '<turn|>\\n' }}`
      + `{% endif %}{% endfor %}`
      + `{% if add_generation_prompt %}{{ '<|turn>model\\n' }}{% endif %}`;
    const result = (tokenizer.inner as any).apply_chat_template(msgs, {
      add_generation_prompt: true,
      tokenize: true,
      return_tensor: false,
      chat_template: tpl,
    });
    console.log(`[Tokenizer] Gemma template applied (thinking=${enableThinking})`);
    return Array.from(result as ArrayLike<number | bigint>).map(Number);
  }

  // Use a clean ChatML template — Qwen3.5's built-in template adds empty <think></think>
  // blocks when thinking is disabled, which confuses smaller models. We control thinking
  // explicitly: when enabled, end with <think>\n so the model generates its own reasoning;
  // when disabled, end with just assistant\n for direct response.
  const genPrompt = enableThinking
    ? `<|im_start|>assistant\n<think>\n`
    : `<|im_start|>assistant\n`;

  // Try the HF tokenizer's apply_chat_template with our explicit template
  try {
    const chatml = `{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '${genPrompt}' }}{% endif %}`;
    const result = (tokenizer.inner as any).apply_chat_template(messages, {
      add_generation_prompt: true,
      tokenize: true,
      return_tensor: false,
      chat_template: chatml,
    });
    if (result && result.length > 0) {
      console.log(`[Tokenizer] ChatML template applied (thinking=${enableThinking})`);
      return Array.from(result).map(Number);
    }
  } catch (e) {
    console.warn('[Tokenizer] apply_chat_template failed, using fallback:', e);
  }

  // Fallback: encode with ChatML format (won't handle special tokens perfectly)
  const parts: string[] = [];
  for (const msg of messages) {
    parts.push(`<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`);
  }
  parts.push(enableThinking ? '<|im_start|>assistant\n<think>\n' : '<|im_start|>assistant\n');
  return tokenizer.encode(parts.join(''));
}
