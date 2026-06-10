// D0: capture llama-server greedy baseline for Gemma 4 E4B Q4_K_M.
// Server must be running: llama-server -m gemma-4-E4B-it-Q4_K_M.gguf -ngl 99 --port 8089
//
// Run: node scripts/capture-gemma4-baseline.mjs
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(__dirname, '..', 'test-fixtures', 'gemma4-e4b-llamacpp-baseline.json');
const SERVER = process.env.LLAMA_SERVER ?? 'http://127.0.0.1:8089';

const PROMPTS = [
  'Explain why the sky is blue in two sentences.',
  'Write a haiku about a mountain stream.',
  'What is 17 * 23? Show your reasoning step by step.',
  'List three uses for a paperclip besides holding paper.',
  'Translate "good morning, my friend" into French and German.',
];

// Same simplified template the WebGPU engine will use (no thinking):
// <bos><|turn>user\n{msg}<turn|>\n<|turn>model\n
// llama-server /completion adds BOS itself (add_special), so omit <bos> here.
const wrap = (msg) => `<|turn>user\n${msg}<turn|>\n<|turn>model\n`;

const results = [];
for (const p of PROMPTS) {
  const resp = await fetch(`${SERVER}/completion`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: wrap(p),
      n_predict: 64,
      temperature: 0,
      samplers: ['top_k'], top_k: 1, // belt-and-braces greedy
      return_tokens: true,
      cache_prompt: false,
    }),
  });
  if (!resp.ok) throw new Error(`completion failed: ${resp.status} ${await resp.text()}`);
  const j = await resp.json();
  results.push({
    prompt: p,
    wrapped: wrap(p),
    prompt_tokens: j.tokens_evaluated ?? null,
    content: j.content,
    tokens: j.tokens ?? null,
    stop_type: j.stop_type ?? j.stopped_eos ?? null,
  });
  console.log(`--- ${p}\n${j.content}\n`);
}

const props = await (await fetch(`${SERVER}/props`)).json();
fs.writeFileSync(OUT, JSON.stringify({
  model: 'ggml-org/gemma-4-E4B-it-GGUF Q4_K_M',
  server_settings: { n_predict: 64, temperature: 0, top_k: 1 },
  build: props.build_info ?? null,
  generated: new Date().toISOString(),
  results,
}, null, 2));
console.log(`wrote ${OUT}`);
