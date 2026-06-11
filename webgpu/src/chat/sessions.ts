/**
 * Session persistence — save and restore conversations across page loads.
 *
 * File format matches core/session.py exactly (name/timestamp/saved_at/
 * message_count/messages/session_map/metadata), so exported .json sessions
 * are interchangeable with the Artifex GUI's sessions/ directory and vice
 * versa. metadata.backend is 'webgpu' for sessions saved here.
 *
 * Storage: localStorage for autosaves (rotating _autosave_1..3, matching the
 * Python rotation) and named saves; export/import as .json files for moving
 * between machines or into the Artifex sessions/ folder.
 */

import type { ChatMessage } from './compression';

export interface SessionFile {
  name: string;
  timestamp: string;          // YYYYMMDD_HHMMSS (matches time.strftime format)
  saved_at: number;           // unix seconds (matches Python time.time())
  message_count: number;
  messages: ChatMessage[];
  session_map: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

const STORE_PREFIX = 'artifex-session:';
const AUTOSAVE_NAMES = ['_autosave_1', '_autosave_2', '_autosave_3'];

function formatTimestamp(d = new Date()): string {
  const p = (n: number, w = 2) => String(n).padStart(w, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

/** Sanitize like Python: keep alnum/-/_/space, max 60 chars, spaces→underscores. */
function sanitizeName(name: string): string {
  const safe = [...name].filter(c => /[a-zA-Z0-9\-_ ]/.test(c)).join('').slice(0, 60).trim();
  return safe.replace(/ /g, '_') || 'session';
}

/** Build a session.py-compatible state object. Strips system messages (runtime state). */
export function buildSessionState(
  name: string,
  messages: ChatMessage[],
  metadata: Record<string, unknown> = {},
): SessionFile {
  const savedMessages = messages.filter(m => m.role !== 'system');
  return {
    name,
    timestamp: formatTimestamp(),
    saved_at: Date.now() / 1000,
    message_count: savedMessages.length,
    messages: savedMessages,
    session_map: {},
    metadata,
  };
}

/** Save a named session to localStorage. Returns the storage key. */
export function saveSession(
  name: string,
  messages: ChatMessage[],
  metadata: Record<string, unknown> = {},
): string {
  const state = buildSessionState(name, messages, metadata);
  const key = `${STORE_PREFIX}${sanitizeName(name)}_${state.timestamp}`;
  localStorage.setItem(key, JSON.stringify(state));
  return key;
}

/** Rotating autosave — overwrites _autosave_1, shifting 1→2→3 (3 dropped). */
export function autoSave(messages: ChatMessage[], metadata: Record<string, unknown> = {}): void {
  if (messages.filter(m => m.role !== 'system').length === 0) return;
  try {
    for (let i = AUTOSAVE_NAMES.length - 1; i >= 1; i--) {
      const prev = localStorage.getItem(STORE_PREFIX + AUTOSAVE_NAMES[i - 1]);
      if (prev !== null) localStorage.setItem(STORE_PREFIX + AUTOSAVE_NAMES[i], prev);
    }
    const state = buildSessionState(AUTOSAVE_NAMES[0], messages, metadata);
    localStorage.setItem(STORE_PREFIX + AUTOSAVE_NAMES[0], JSON.stringify(state));
  } catch (err) {
    // localStorage quota — autosave is best-effort, never break the chat over it
    console.warn('[Sessions] autosave failed:', err);
  }
}

export interface SessionListing {
  key: string;
  name: string;
  timestamp: string;
  saved_at: number;
  message_count: number;
  isAutosave: boolean;
}

/** List all stored sessions, most recent first. */
export function listSessions(): SessionListing[] {
  const out: SessionListing[] = [];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (!key || !key.startsWith(STORE_PREFIX)) continue;
    try {
      const state = JSON.parse(localStorage.getItem(key)!) as SessionFile;
      out.push({
        key,
        name: state.name,
        timestamp: state.timestamp,
        saved_at: state.saved_at,
        message_count: state.message_count,
        isAutosave: state.name.startsWith('_autosave'),
      });
    } catch { /* skip corrupt entries */ }
  }
  return out.sort((a, b) => b.saved_at - a.saved_at);
}

/** Load a session by storage key. Returns null if missing/corrupt. */
export function loadSession(key: string): SessionFile | null {
  const raw = localStorage.getItem(key);
  if (raw === null) return null;
  try {
    const data = JSON.parse(raw);
    return {
      name: data.name ?? 'session',
      timestamp: data.timestamp ?? '',
      saved_at: data.saved_at ?? 0,
      message_count: data.message_count ?? (data.messages?.length ?? 0),
      messages: data.messages ?? [],
      session_map: data.session_map ?? {},
      metadata: data.metadata ?? {},
    };
  } catch (err) {
    console.error(`[Sessions] failed to load ${key}:`, err);
    return null;
  }
}

export function deleteSession(key: string): void {
  localStorage.removeItem(key);
}

/** Download a session as a .json file (drop it into Artifex's sessions/ to share). */
export function exportSessionFile(state: SessionFile): void {
  const blob = new Blob([JSON.stringify(state, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${sanitizeName(state.name)}_${state.timestamp}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

/** Parse an imported session .json (from this app or the Artifex sessions/ dir). */
export async function importSessionFile(file: File): Promise<SessionFile | null> {
  try {
    const data = JSON.parse(await file.text());
    if (!Array.isArray(data.messages)) return null;
    return {
      name: data.name ?? file.name.replace(/\.json$/, ''),
      timestamp: data.timestamp ?? '',
      saved_at: data.saved_at ?? 0,
      message_count: data.message_count ?? data.messages.length,
      messages: data.messages,
      session_map: data.session_map ?? {},
      metadata: data.metadata ?? {},
    };
  } catch (err) {
    console.error('[Sessions] import failed:', err);
    return null;
  }
}
