// D5 smoke test: verifies the EXACT Gemma 4 chat-template logic from
// src/model/tokenizer.ts applyChatTemplate (Gemma branch) against the real
// tokenizer. The Jinja template string below must stay byte-identical to the
// TS source (note: `\\n` in TS source = literal backslash-n in the Jinja
// source, which the Jinja engine unescapes to a newline).
//
// Run: node scripts/test-gemma4-template.mjs
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

// ── Mirror of applyChatTemplate's Gemma branch ──────────────────────────────
function gemmaTemplate(messages, enableThinking) {
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
  const result = tok.apply_chat_template(msgs, {
    add_generation_prompt: true,
    tokenize: true,
    return_tensor: false,
    chat_template: tpl,
  });
  const text = tok.apply_chat_template(msgs, {
    add_generation_prompt: true,
    tokenize: false,
    chat_template: tpl,
  });
  return { ids: Array.from(result).map(Number), text };
}

const NL = Array.from(tok.encode('\n', { add_special_tokens: false })).map(Number);
console.log(`  ids: bos=2 <|turn>=105 <turn|>=106 <|think|>=98, '\\n' -> [${NL.join(',')}]`);

// 1. thinking=true, no system message → empty system turn with <|think|>
{
  const { ids, text } = gemmaTemplate([{ role: 'user', content: 'What is the capital of France?' }], true);
  console.log(`  [think] text: ${JSON.stringify(text)}`);
  console.log(`  [think] ids (${ids.length}): ${ids.join(',')}`);
  check('think: starts with bos(2)', ids[0] === 2, `got ${ids[0]}`);
  check('think: contains <|think|>(98)', ids.includes(98));
  check('think: rendered text shape', text.startsWith('<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n') && text.endsWith('<turn|>\n<|turn>model\n'), JSON.stringify(text));
  check('think: 105/106 markers present', ids.includes(105) && ids.includes(106));
  check('think: ends with <|turn>model\\n', ids[ids.length - 2] === 105 || ids.slice(-4).includes(105), `tail=${ids.slice(-6).join(',')}`);
}

// 2. thinking=false → no system turn injected, no <|think|>
{
  const { ids, text } = gemmaTemplate([{ role: 'user', content: 'Hello!' }], false);
  console.log(`  [no-think] text: ${JSON.stringify(text)}`);
  check('no-think: no <|think|>(98)', !ids.includes(98));
  check('no-think: rendered text shape', text === '<bos><|turn>user\nHello!<turn|>\n<|turn>model\n', JSON.stringify(text));
}

// 3. multi-turn with explicit system + assistant→model mapping
{
  const messages = [
    { role: 'system', content: 'Be brief.' },
    { role: 'user', content: 'Hi' },
    { role: 'assistant', content: 'Hello.' },
    { role: 'user', content: 'Bye' },
  ];
  const { text } = gemmaTemplate(messages, true);
  console.log(`  [multi] text: ${JSON.stringify(text)}`);
  check('multi: system gets <|think|> prefix', text.startsWith('<bos><|turn>system\n<|think|>\nBe brief.<turn|>\n'), JSON.stringify(text));
  check('multi: assistant rendered as model', text.includes('<|turn>model\nHello.<turn|>\n'));
  check('multi: gen prompt at end', text.endsWith('<|turn>user\nBye<turn|>\n<|turn>model\n'));
}

// 4. <turn|> encodes to single token 106 (EOS-set membership in tokenizer.ts)
{
  const sids = Array.from(tok.encode('<turn|>', { add_special_tokens: false })).map(Number);
  check('<turn|> -> [106]', sids.length === 1 && sids[0] === 106, `got [${sids.join(',')}]`);
}

process.exit(failures ? 1 : 0);
