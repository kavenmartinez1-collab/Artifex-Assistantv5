// D0 smoke test: Gemma 4 tokenizer.json loads with @huggingface/transformers
// and round-trips text + special tokens correctly.
//
// Run: node scripts/test-gemma4-tokenizer.mjs
import { PreTrainedTokenizer } from '@huggingface/transformers';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.resolve(__dirname, '..', '..', 'models', 'gemma-4-e4b-it-gguf');

const tokenizerJSON = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'tokenizer.json'), 'utf-8'));
const tokenizerConfig = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'tokenizer_config.json'), 'utf-8'));

const tok = new PreTrainedTokenizer(tokenizerJSON, tokenizerConfig);

let failures = 0;
function check(name, cond, detail = '') {
  if (cond) console.log(`PASS  ${name}`);
  else { failures++; console.error(`FAIL  ${name}  ${detail}`); }
}

// 1. Vocab size matches config (262144)
const vocabSize = tok.model?.vocab?.length ?? tok.vocab_size;
check('vocab size 262144', vocabSize === 262144, `got ${vocabSize}`);

// 2. Round-trip plain text
const text = 'The quick brown fox jumps over the lazy dog. 你好世界 123';
const ids = Array.from(tok.encode(text)).map(Number);
const back = tok.decode(ids, { skip_special_tokens: true });
check('round-trip text', back.trim() === text || back === text, `got "${back}"`);
console.log(`  encoded length: ${ids.length}, first ids: ${ids.slice(0, 8).join(',')}`);

// 3. Special tokens encode as single tokens (Gemma 4 turn markers: <|turn>=105, <turn|>=106)
const expected = { '<|turn>': 105, '<turn|>': 106, '<bos>': 2, '<eos>': 1, '<|think|>': 98, '<|channel>': 100, '<channel|>': 101 };
for (const [s, want] of Object.entries(expected)) {
  const sids = Array.from(tok.encode(s, { add_special_tokens: false })).map(Number);
  check(`"${s}" -> [${want}]`, sids.length === 1 && sids[0] === want, `got [${sids.join(',')}]`);
}

// 4. BOS/EOS ids
console.log(`  bos_token_id=${tok.bos_token_id ?? 'null'} eos_token_id=${tok.eos_token_id ?? 'null'} (gen stop = <turn|> 106)`);

// 5. Gemma 4 chat template (simplified: <|turn>role\n ... <turn|>\n, assistant -> model)
const messages = [{ role: 'user', content: 'Hello!' }];
try {
  const tpl = `{{ bos_token }}{% for message in messages %}{{ '<|turn>' + (message['role'] == 'assistant' and 'model' or message['role']) + '\n' + message['content'] + '<turn|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|turn>model\n' }}{% endif %}`;
  const out = tok.apply_chat_template(messages, {
    add_generation_prompt: true, tokenize: false, chat_template: tpl,
  });
  check('chat template renders', typeof out === 'string' && out.includes('<|turn>user') && out.endsWith('<|turn>model\n'), `got "${String(out).slice(0, 120)}"`);
  const tplIds = Array.from(tok.apply_chat_template(messages, { add_generation_prompt: true, tokenize: true, return_tensor: false, chat_template: tpl })).map(Number);
  console.log(`  templated ids (${tplIds.length}): ${tplIds.join(',')}`);
  // Expected shape: 2(bos),105,...role/newline...,Hello tokens,106,...,105,...model...
  check('template ids contain turn markers as 105/106', tplIds.includes(105) && tplIds.includes(106), `ids=${tplIds.join(',')}`);
} catch (e) {
  failures++;
  console.error('FAIL  chat template threw:', e.message);
}

process.exit(failures ? 1 : 0);
