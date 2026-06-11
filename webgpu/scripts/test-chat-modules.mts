/**
 * Parity test: src/chat/{compression,context}.ts vs core/inference.py.
 *
 * Replays the inputs from test-fixtures/chat-modules-golden.json (generated
 * by gen_chat_fixtures.py from the REAL Python implementations) and diffs
 * every output. Run:
 *     npx tsx scripts/test-chat-modules.mts
 */
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

import { extractKeyPoint, compressHistory, type ChatMessage } from '../src/chat/compression';
import { buildActiveMessages, trimMessagesToContext } from '../src/chat/context';

const here = dirname(fileURLToPath(import.meta.url));
const golden = JSON.parse(
  readFileSync(join(here, '..', 'test-fixtures', 'chat-modules-golden.json'), 'utf-8'));

let pass = 0;
let fail = 0;

function check(label: string, actual: unknown, expected: unknown) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a === e) {
    pass++;
  } else {
    fail++;
    console.error(`FAIL ${label}`);
    console.error(`  expected: ${e.slice(0, 300)}`);
    console.error(`  actual:   ${a.slice(0, 300)}`);
  }
}

for (const [i, c] of golden.extract_key_point.entries()) {
  check(`extract_key_point[${i}]`, extractKeyPoint(c.input as ChatMessage), c.expected);
}

for (const [i, c] of golden.compress_history.entries()) {
  check(`compress_history[${i}] (window=${c.context_window})`,
    compressHistory(c.history, c.context_window), c.expected);
}

for (const [i, c] of golden.build_active_messages.entries()) {
  const { history, active } = buildActiveMessages(c.history, {
    contextWindow: c.context_window,
    maxHistoryTokens: c.max_history_tokens ?? undefined,
    engineCtx: c.engine_ctx ?? 0,
    // no countTokens — exercises the chars//4 fallback, matching the Python run
  });
  check(`build_active_messages[${i}].history`, history, c.expected_history);
  check(`build_active_messages[${i}].active`, active, c.expected_active);
}

for (const [i, c] of golden.trim_messages_to_context.entries()) {
  check(`trim_messages_to_context[${i}] (cap=${c.max_input_tokens})`,
    trimMessagesToContext(c.messages, c.max_input_tokens), c.expected);
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail > 0 ? 1 : 0);
