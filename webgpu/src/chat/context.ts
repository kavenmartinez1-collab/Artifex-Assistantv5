/**
 * Sliding-window context management — keeps multi-turn chats inside the
 * model's context (and the GPU's KV budget) without losing the thread.
 *
 * Port of core/inference.py `build_active_messages()`, `auto_compact_if_needed()`
 * and `trim_messages_to_context()` with the same shrink policy, so the webgpu
 * app windows conversations identically to the Artifex GUI/API.
 *
 * Differences from the Python side:
 *   - Token counting uses the REAL tokenizer (always available in-browser via
 *     the loaded model) — the chars/4 heuristic is only a fallback.
 *   - `engineCtx` comes from the KV session capacity rather than a GPU-tier
 *     profile; in a browser the binding constraint is the KV cache allocation.
 */

import { compressHistory, type ChatMessage } from './compression';

export type { ChatMessage };

export type CountTokens = (text: string) => number;

/** Matches core/config.py ContextProfile STANDARD (used when engineCtx is unknown). */
const PROFILE_MAX_TOTAL_INPUT = 10000;
const PROFILE_MAX_HISTORY = 6000;

/**
 * Count tokens across messages — exact port of core/inference.py _count_tokens:
 * with a tokenizer, contents are joined with '\n' and encoded; without one,
 * the fallback is sum(len(content)) // 4 (NO join separators).
 */
export function countMessageTokens(messages: ChatMessage[], countTokens?: CountTokens): number {
  if (countTokens) {
    const text = messages.map(m => m.content ?? '').join('\n');
    return countTokens(text);
  }
  return Math.floor(messages.reduce((s, m) => s + (m.content ?? '').length, 0) / 4);
}

export interface ContextOptions {
  /** Max number of recent messages to consider (Python context_window). Default 10. */
  contextWindow?: number;
  /** Override for the history token cap. */
  maxHistoryTokens?: number;
  /** Engine context size in tokens (KV session capacity). >0 scales budget to 70%. */
  engineCtx?: number;
  countTokens?: CountTokens;
}

export interface ActiveMessagesResult {
  /** Possibly-compressed full history (write back as the new source of truth). */
  history: ChatMessage[];
  /** The windowed messages to actually send to the model. */
  active: ChatMessage[];
}

/**
 * Build the active message list for the next generation call.
 * `history[0]` must be the system message (empty content is fine — the caller
 * drops it before templating).
 */
export function buildActiveMessages(
  history: ChatMessage[],
  opts: ContextOptions = {},
): ActiveMessagesResult {
  const countTokens = opts.countTokens;
  let contextWindow = opts.contextWindow ?? 10;
  const engineCtx = opts.engineCtx ?? 0;

  const systemTokens = countMessageTokens([history[0]], countTokens);

  let historyBudget: number;
  if (engineCtx > 0) {
    const totalCap = Math.floor(engineCtx * 0.70);
    historyBudget = opts.maxHistoryTokens ?? Math.max(totalCap - systemTokens, 200);
    contextWindow = Math.min(Math.max(Math.floor(historyBudget / 500), contextWindow), 200);
  } else {
    const totalCap = PROFILE_MAX_TOTAL_INPUT;
    historyBudget = opts.maxHistoryTokens ?? Math.max(totalCap - systemTokens, 200);
    if (opts.maxHistoryTokens === undefined) {
      historyBudget = Math.min(historyBudget, PROFILE_MAX_HISTORY);
    }
  }

  let active = [history[0], ...history.slice(1).slice(-contextWindow)];

  if (countMessageTokens(active.slice(1), countTokens) <= historyBudget) {
    return { history: [...history], active };
  }

  const compressed = compressHistory(history, contextWindow);
  for (const shrink of [Math.floor(contextWindow / 2), Math.floor(contextWindow / 4), 2]) {
    active = [compressed[0], ...compressed.slice(1).slice(-Math.max(shrink, 2))];
    if (countMessageTokens(active.slice(1), countTokens) <= historyBudget) {
      return { history: compressed, active };
    }
  }

  active = [compressed[0], ...compressed.slice(1).slice(-2)];
  return { history: compressed, active };
}

/**
 * Auto-compact conversation when token count approaches engine context.
 * Triggers at threshold (default 60%) of engineCtx.
 */
export function autoCompactIfNeeded(
  messages: ChatMessage[],
  engineCtx: number,
  contextWindow: number,
  threshold = 0.60,
  countTokens?: CountTokens,
): { messages: ChatMessage[]; compacted: boolean } {
  if (engineCtx <= 0 || messages.length <= 4) return { messages, compacted: false };

  const tokenCount = countMessageTokens(messages, countTokens);
  const limit = Math.floor(engineCtx * threshold);
  if (tokenCount <= limit) return { messages, compacted: false };

  const scaledWindow = Math.min(Math.max(Math.floor(limit / 500), contextWindow), 200);
  const compacted = compressHistory(messages, scaledWindow);
  console.log(
    `[Context] Auto-compacted: ${tokenCount} tok -> ${countMessageTokens(compacted, countTokens)} tok `
    + `(${messages.length} -> ${compacted.length} messages)`);
  return { messages: compacted, compacted: true };
}

/**
 * Hard-cap safety net: drop oldest middle messages so total tokens fit.
 * Keeps system prompt + most recent 2 messages; truncates the last message
 * content if it alone exceeds the budget.
 */
export function trimMessagesToContext(
  messages: ChatMessage[],
  maxInputTokens: number,
  countTokens?: CountTokens,
): ChatMessage[] {
  if (maxInputTokens <= 0 || messages.length <= 1) return messages;

  const est = (msgs: ChatMessage[]) => countMessageTokens(msgs, countTokens);
  if (est(messages) <= maxInputTokens) return messages;

  const system = messages.slice(0, 1);
  const middle = messages.length > 3 ? messages.slice(1, -2) : [];
  const tail = messages.length >= 2 ? messages.slice(-2) : [];

  while (middle.length > 0 && est([...system, ...middle, ...tail]) > maxInputTokens) {
    middle.shift();
  }

  const result = [...system, ...middle, ...tail];

  if (est(result) > maxInputTokens && result.length >= 2) {
    const last = { ...result[result.length - 1] };
    const content = last.content ?? '';
    const overshoot = est(result) - maxInputTokens;
    const charsToCut = overshoot * 4 + 200;
    if (charsToCut < content.length) {
      last.content = content.slice(0, content.length - charsToCut) + '\n[...trimmed...]';
      result[result.length - 1] = last;
    }
  }

  return result;
}
