/**
 * Conversation compression — key-point extraction for long chats.
 *
 * Direct port of core/inference.py `_extract_key_point()` / `compress_history()`
 * so webgpu sessions compress identically to the Artifex GUI/API. Verified
 * against Python-generated golden fixtures (scripts/test-chat-modules.mjs).
 *
 * Strategy: keep the system prompt and recent messages intact, PIN the first
 * user message (the original task request), and collapse everything between
 * into a single "[EARLIER CONVERSATION — key points]" user message.
 */

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

/** Pull the essential content from a message for compression. */
export function extractKeyPoint(msg: ChatMessage): string {
  const content = msg.content;

  if (content.startsWith('[TOOL OUTPUT') || content.startsWith('[TOOL RESULT')) {
    const lines = content.split('\n').map(l => l.trim()).filter(l => l.length > 0);

    if (lines.slice(0, 3).some(l => l.toLowerCase().includes('architecture'))) {
      return 'Ran: @architecture() — full project map generated';
    }
    for (const line of lines) {
      if (line.startsWith('Function:')) return `Read: ${line.slice(0, 120)}`;
    }
    for (const line of lines) {
      if (line.includes('SKELETON VIEW')) {
        const fname = lines.find(l => l.startsWith('File:')) ?? '';
        return fname ? `Read skeleton: ${fname.slice(0, 100)}` : 'Read: skeleton view';
      }
    }
    for (const line of lines) {
      if (line.startsWith('File:')) return `Read: ${line.slice(0, 120)}`;
    }
    for (const line of lines) {
      if (line.startsWith('Found ')) return line.slice(0, 120);
    }
    for (const line of lines) {
      if (line.startsWith('$') || line.startsWith('`')) return `Ran: ${line.slice(0, 80)}`;
    }
    return lines.length > 0 ? lines[0].slice(0, 80) : '(tool output)';
  }

  if (msg.role === 'user') {
    return content.split('\n')[0].trim().slice(0, 100);
  }

  // Assistant: first substantial sentence (matches Python re.split(r'[.!?\n]'))
  const sentences = content.split(/[.!?\n]/);
  for (let s of sentences) {
    s = s.trim().replace(/^[#*\- ]+/, '');
    if (s.length > 15) return s.slice(0, 100);
  }
  return content.slice(0, 100);
}

/**
 * Compress old messages into key-point summaries.
 *
 * Keeps the system prompt (history[0]) and recent messages intact.
 * PINS the first user message — the original task request.
 */
export function compressHistory(history: ChatMessage[], contextWindow: number): ChatMessage[] {
  const convo = history.slice(1);
  if (convo.length <= contextWindow) return [...history];

  const pinnedMsg = convo.length > 0 && convo[0].role === 'user' ? convo[0] : null;
  const compressible = pinnedMsg ? convo.slice(1) : convo;

  const keepCount = Math.max(contextWindow - 1, 1);
  if (compressible.length <= keepCount) return [...history];

  const oldMessages = compressible.slice(0, -keepCount);
  const recentMessages = compressible.slice(-keepCount);

  const points: string[] = [];
  for (const msg of oldMessages) {
    const role = msg.role === 'user' ? 'Q' : 'A';
    points.push(`${role}: ${extractKeyPoint(msg)}`);
  }

  const summaryText = '[EARLIER CONVERSATION — key points]\n' + points.join('\n');

  const result: ChatMessage[] = [history[0]];
  if (pinnedMsg) result.push(pinnedMsg);
  result.push({ role: 'user', content: summaryText });
  result.push(...recentMessages);
  return result;
}
