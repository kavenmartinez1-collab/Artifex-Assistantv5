/**
 * Artifex WebGPU Engine — Main Entry Point
 *
 * Initializes WebGPU, runs kernel tests, and sets up the chat UI.
 * Phase 0-1: device detection, compute foundation, UI shell.
 * Phase 2: SafeTensors weight loading from HuggingFace.
 */

import { initWebGPU, discoverAdapters, type GPUContext, type DiscoveredAdapter } from './engine/gpu-device';
import { reportMetric, reportError, timed } from './utils/metrics';
import { runKernelTests } from './engine/kernel-tests';
import { loadModel, unloadModel, previewModel, formatBytes, getCacheStats, clearCache, removeModelFromCache, type LoadedModel } from './model';
import { setAuthToken, useLocalCache, resetToRemote, resolveFileUrl, fetchRange } from './model/hf-hub';
import { visionDescriptorFromHFConfig, type VisionDescriptor } from './vision/vision-descriptor';
import { preprocessImage, type PreprocessedImage } from './vision/preprocess';
import type { VisionEncoder, VisionEncodeResult } from './vision/vision-encoder';
import { createInferenceSession, type InferenceSession } from './engine/inference';
import { parseModelConfig, estimateVRAM } from './model/model-config';
import { descriptorFromHFConfig } from './model/model-descriptor';
import type { TensorRole } from './model/tensor-locator';
import { MAX_ATTN_SEQ_LEN } from './engine/forward-pass';
import { buildActiveMessages, type ChatMessage } from './chat/context';
import {
  autoSave, saveSession, listSessions, loadSession,
  buildSessionState, exportSessionFile, importSessionFile, type SessionFile,
} from './chat/sessions';
import { escapeHtml, markdownToHtml } from './chat/markdown';
import { parseToolCall, executeToolCall, toolSystemPreamble } from './chat/tools';
import type { GenerationHandle } from './engine/generate';

// ─── State ───────────────────────────────────────────────────────────────────

let gpu: GPUContext | null = null;
let currentModel: LoadedModel | null = null;
let session: InferenceSession | null = null;

// FLUX.2 image-gen mode: no LLM session — Send routes to runImageGeneration.
let imageGenRepo: string | null = null;
let imageGenBusy = false;

// Conversation history (user/assistant only — system prompt is read live from
// the UI each turn). Source of truth for windowing, sessions, and autosave.
let chatHistory: ChatMessage[] = [];

// The in-flight generation, if any. Stop button / Esc call .abort() on it;
// the decode loop then exits with stopReason 'aborted' and the result promise
// resolves normally with the partial text.
let activeHandle: GenerationHandle | null = null;

// ─── Vision state ────────────────────────────────────────────────────────────
// Set when the loaded model is a (verified) multimodal checkpoint. The tower
// loads lazily on first image; pending images preprocess at attach time so
// their token cost shows in the chip before sending.
let activeVisionDesc: VisionDescriptor | null = null;
/** Where the tower weights live: HF safetensors (model.visual.*) or a GGUF
 *  mmproj file alongside the text GGUF. */
let activeVisionSource: { kind: 'hf' } | { kind: 'gguf'; file: string } = { kind: 'hf' };
let visionEncoder: VisionEncoder | null = null;
let visionEncoderLoading: Promise<VisionEncoder> | null = null;
interface PendingImage { name: string; pre: PreprocessedImage }
let pendingImages: PendingImage[] = [];

// ─── Text/code file attachments ─────────────────────────────────────────────
// Unlike images these need no special model — the file content is inlined
// into the user message as a fenced [FILE: name] block. The chat window shows
// a compact 📄 line; history carries the full text (so switching models,
// sessions, and export all keep the document).
interface PendingFile { name: string; text: string; truncated: boolean }
let pendingFiles: PendingFile[] = [];
/** ~6K tokens/file — enough for real source files without blowing an 8K KV budget. */
const MAX_FILE_CHARS = 24_000;
const TEXT_FILE_EXT = /\.(txt|md|markdown|py|js|mjs|ts|tsx|jsx|json|jsonl|csv|tsv|ya?ml|toml|html?|css|c|h|cpp|hpp|cc|rs|go|java|kt|swift|rb|php|sh|bash|ps1|bat|sql|xml|log|ini|cfg|conf|wgsl|glsl|cu)$/i;

function isTextAttachment(f: File): boolean {
  return f.type.startsWith('text/')
    || f.type === 'application/json'
    || f.type === 'application/xml'
    || TEXT_FILE_EXT.test(f.name);
}

async function addPendingFile(f: File): Promise<void> {
  try {
    let text = await f.text();
    const truncated = text.length > MAX_FILE_CHARS;
    if (truncated) text = text.slice(0, MAX_FILE_CHARS);
    pendingFiles.push({ name: f.name, text, truncated });
    renderChips();
  } catch (err) {
    addMessage('system', `Could not read "${f.name}": ${err}`);
  }
}

/** Merge pending files into the outgoing message. Returns the text the model
 *  sees and the compact text the chat window shows. */
function consumePendingFiles(typed: string): { sendText: string; displayText: string } {
  if (pendingFiles.length === 0) return { sendText: typed, displayText: typed };
  const files = pendingFiles;
  pendingFiles = [];
  renderChips();
  const blocks = files.map(f =>
    `[FILE: ${f.name}${f.truncated ? ` — truncated to first ${MAX_FILE_CHARS} chars` : ''}]\n\`\`\`\n${f.text}\n\`\`\``);
  return {
    sendText: blocks.join('\n\n') + (typed ? `\n\n${typed}` : ''),
    displayText: files.map(f => `📄 ${f.name}`).join('  ') + (typed ? `\n${typed}` : ''),
  };
}

// Thinking markers by model family. Qwen emits <think>...</think> (the opener
// usually lives in the PROMPT, so generated text may contain only the closer);
// Gemma 4 emits <|channel>thought ... <channel|> entirely in the generation.
const THINK_MARKERS = [
  { open: '<think>', close: '</think>' },
  { open: '<|channel>thought', close: '<channel|>' },
];

/** Split generated text into in-progress/completed thinking and the answer.
 *  assumeOpen: treat marker-less text as thinking (Qwen-style templates end
 *  the prompt with an opener, so generation starts mid-thought). */
function splitThinking(text: string, assumeOpen: boolean): { thinking: string; answer: string } {
  for (const m of THINK_MARKERS) {
    const closeIdx = text.indexOf(m.close);
    if (closeIdx !== -1) {
      const thinking = text.slice(0, closeIdx).replace(m.open, '').trim();
      const answer = text.slice(closeIdx + m.close.length).replace(/^\s+/, '');
      return { thinking, answer };
    }
  }
  for (const m of THINK_MARKERS) {
    if (text.includes(m.open)) {
      return { thinking: text.replace(m.open, '').trim(), answer: '' };
    }
  }
  return assumeOpen ? { thinking: text.trim(), answer: '' } : { thinking: '', answer: text };
}

/** Reasoning stays out of history (matches Artifex GUI semantics + Qwen
 *  multi-turn guidance). On a mid-thought cutoff, keep what we have. */
function stripThinking(text: string): string {
  const { thinking, answer } = splitThinking(text, false);
  return answer || thinking;
}

// ─── DOM Elements ────────────────────────────────────────────────────────────

const $ = (id: string) => document.getElementById(id)!;
const gpuBadge = $('gpu-badge');
const gpuSelect = $('gpu-select') as HTMLSelectElement;
const statusEl = $('status');
const messagesEl = $('messages');
const promptEl = $('prompt') as HTMLTextAreaElement;
const sendBtn = $('send-btn') as HTMLButtonElement;
const stopBtn = $('stop-btn') as HTMLButtonElement;
const loadBtn = $('load-btn') as HTMLButtonElement;
const clearBtn = $('clear-btn') as HTMLButtonElement;
const exportBtn = $('export-btn') as HTMLButtonElement;
const unloadBtn = $('unload-btn') as HTMLButtonElement;
const testBtn = $('test-kernels') as HTMLButtonElement;
const testResults = $('test-results');
const tempSlider = $('temperature') as HTMLInputElement;
const toppSlider = $('top-p') as HTMLInputElement;
const topkInput = $('top-k') as HTMLInputElement;
const minpSlider = $('min-p') as HTMLInputElement;
const reppenSlider = $('rep-pen') as HTMLInputElement;
const drySlider = $('dry-mult') as HTMLInputElement;
const presetSelect = $('sampler-preset') as HTMLSelectElement;
const tempVal = $('temp-val');
const toppVal = $('topp-val');
const minpVal = $('minp-val');
const reppenVal = $('reppen-val');
const dryVal = $('dry-val');

// ─── Sampler Presets ─────────────────────────────────────────────────────────
// Presets resolve to concrete slider values. Each preset defines the full
// sampler stack. `custom` is not a preset — it's the state when the user
// hand-edits any slider.
type PresetName = 'balanced' | 'deterministic' | 'creative' | 'reference';
interface PresetValues {
  temperature: number; topP: number; topK: number;
  minP: number; repPen: number; dryMult: number;
}
const PRESETS: Record<PresetName, PresetValues> = {
  // llama.cpp-style neutral defaults — temperature + top-p only
  balanced:      { temperature: 0.7, topP: 0.9,  topK: 40, minP: 0,    repPen: 1.0, dryMult: 0    },
  // Greedy argmax — no sampling randomness, for coherence diagnostics
  deterministic: { temperature: 0,   topP: 1.0,  topK: 0,  minP: 0,    repPen: 1.0, dryMult: 0    },
  // Adds min-p + DRY for long-form diversity (may induce word-chain collapse
  // on some models — opt-in only)
  creative:      { temperature: 0.9, topP: 0.95, topK: 50, minP: 0.05, repPen: 1.0, dryMult: 0.8  },
  // Matches HuggingFace transformers generate() do_sample=True defaults
  reference:     { temperature: 1.0, topP: 1.0,  topK: 50, minP: 0,    repPen: 1.0, dryMult: 0    },
};

let suppressPresetFlip = false;
function applyPreset(name: PresetName): void {
  const p = PRESETS[name];
  suppressPresetFlip = true;
  tempSlider.value = String(p.temperature);
  toppSlider.value = String(p.topP);
  topkInput.value = String(p.topK);
  minpSlider.value = String(p.minP);
  reppenSlider.value = String(p.repPen);
  drySlider.value = String(p.dryMult);
  // Fire input events so the _val spans update
  [tempSlider, toppSlider, minpSlider, reppenSlider, drySlider].forEach(el =>
    el.dispatchEvent(new Event('input'))
  );
  suppressPresetFlip = false;
}
const browseBtn = $('browse-btn') as HTMLButtonElement;
const modelBrowser = $('model-browser');
const modelList = $('model-list');
const clusterStatusEl = $('cluster-status');

// ─── Dev-server detection ────────────────────────────────────────────────────
// Hosted static builds (GitHub Pages etc.) have no /api routes — a fetch there
// gets the SPA fallback page or a 404, never JSON. Probe once and gate every
// dev-only affordance (test pollers, WS hub, local model browsing) on it, so
// the hosted app doesn't burn cycles polling endpoints that don't exist.
const devApiAvailable: Promise<boolean> = (async () => {
  try {
    const resp = await fetch('/api/gpu-info');
    return resp.ok && (resp.headers.get('content-type') ?? '').includes('json');
  } catch {
    return false;
  }
})();

// ─── Orchestration Hub Connection ────────────────────────────────────────────

function connectOrchestrator(): void {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${location.host}/ws`;

  const ws = new WebSocket(wsUrl);
  ws.onopen = () => {
    ws.send(JSON.stringify({
      type: 'orchestrator:connect',
      id: `orch-${Date.now()}`,
      timestamp: Date.now(),
      source: 'orchestrator',
      payload: {},
    }));
    console.log('[Orchestrator] Connected to hub');
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === 'cluster:status') {
        const { workers, tasks } = msg.payload;
        if (workers.length === 0) {
          clusterStatusEl.textContent = 'No workers connected';
        } else {
          const lines = workers.map((w: any) =>
            `${w.id}: ${w.gpu.device} [${w.status}]${w.model ? ` — ${w.model.repo}` : ''}`
          );
          clusterStatusEl.textContent = `${workers.length} worker(s)\n${lines.join('\n')}`;
        }
      } else if (msg.type === 'task:token') {
        // Future: display streaming agent output
      } else if (msg.type === 'task:complete') {
        console.log('[Orchestrator] Task complete:', msg.payload.taskId);
      }
    } catch {}
  };

  ws.onclose = () => {
    console.log('[Orchestrator] Hub disconnected, reconnecting in 5s...');
    clusterStatusEl.textContent = 'Hub disconnected — reconnecting...';
    setTimeout(connectOrchestrator, 5000);
  };

  ws.onerror = () => {
    // Silently retry — hub might not be running
  };
}

// Connect to hub (non-blocking — doesn't prevent app from working without
// hub; skipped entirely on hosted builds where no hub can exist)
devApiAvailable.then(ok => {
  if (ok) setTimeout(connectOrchestrator, 1000);
  else clusterStatusEl.textContent = 'Needs the local dev server (npm run dev)';
});

// ─── Browse Local Models ─────────────────────────────────────────────────────

let browseOpen = false;

browseBtn.addEventListener('click', async () => {
  if (browseOpen) {
    modelBrowser.style.display = 'none';
    browseOpen = false;
    return;
  }

  modelList.innerHTML = '<div style="padding:8px;color:var(--dim)">Scanning...</div>';
  modelBrowser.style.display = 'block';
  browseOpen = true;

  if (!(await devApiAvailable)) {
    modelList.innerHTML = '<div style="padding:8px;color:var(--dim)">'
      + 'Browsing machine-local models needs the local dev server (npm run dev). '
      + 'On this hosted build, type a HuggingFace repo above or use ✨ to pick one.</div>';
    return;
  }

  try {
    const resp = await fetch('/api/hf-cache/models');
    if (!resp.ok) throw new Error(`${resp.status}`);
    const models = await resp.json() as Array<{ repo: string; files: string[]; totalSize: number }>;

    if (models.length === 0) {
      modelList.innerHTML = '<div style="padding:8px;color:var(--dim)">No local models found</div>';
      return;
    }

    // Sort: machine-local models (local/, ollama/) first, then by size descending
    const isLocalRepo = (r: string) => r.startsWith('local/') || r.startsWith('ollama/');
    models.sort((a, b) => {
      const aLocal = isLocalRepo(a.repo) ? 0 : 1;
      const bLocal = isLocalRepo(b.repo) ? 0 : 1;
      if (aLocal !== bLocal) return aLocal - bLocal;
      return b.totalSize - a.totalSize;
    });

    modelList.innerHTML = '';
    for (const m of models) {
      const el = document.createElement('div');
      const sizeGB = (m.totalSize / 1024 / 1024 / 1024).toFixed(1);
      const safetensors = m.files.filter(f => f.endsWith('.safetensors')).length;
      const ggufs = m.files.filter(f => f.toLowerCase().endsWith('.gguf')).length;
      const isLocal = isLocalRepo(m.repo);
      const icon = m.repo.startsWith('ollama/') ? '🦙 ' : isLocal ? '📁 ' : '';
      const kind = safetensors === 0 && ggufs > 0
        ? `${ggufs} gguf`
        : `${safetensors} shard${safetensors !== 1 ? 's' : ''}`;
      el.style.cssText = 'padding:6px 8px;cursor:pointer;border-bottom:1px solid #222;display:flex;justify-content:space-between;align-items:center';
      el.innerHTML = `
        <span style="color:${isLocal ? 'var(--accent)' : 'var(--text)'}">${icon}${m.repo}</span>
        <span style="color:var(--dim);font-size:10px;white-space:nowrap;margin-left:8px">${sizeGB} GB · ${kind}</span>
      `;
      el.addEventListener('mouseenter', () => { el.style.background = '#1a1e33'; });
      el.addEventListener('mouseleave', () => { el.style.background = 'none'; });
      el.addEventListener('click', () => {
        ($('model-repo') as HTMLInputElement).value = m.repo;
        modelBrowser.style.display = 'none';
        browseOpen = false;
      });
      modelList.appendChild(el);
    }
  } catch (err) {
    modelList.innerHTML = `<div style="padding:8px;color:var(--error)">Failed to scan: ${err}</div>`;
  }
});

// ─── Model Recommender ("Help me pick a model") ─────────────────────────────
// Turns the VRAM budget into 2-4 concrete, one-click choices instead of a
// raw text box — models already on this machine first, then curated HF
// downloads that fit. Same principle as the Artifex GUI's pipeline model
// guidance: the UI should steer model choice, not just accept it.

const suggestBtn = $('suggest-btn') as HTMLButtonElement;
const modelSuggest = $('model-suggest');
const suggestList = $('suggest-list');
let suggestOpen = false;

/** Curated remote picks — safetensors repos on verified families only (the
 *  remote load path is safetensors; GGUF loads come from local dirs). needGB
 *  includes weights at the dtype the engine will actually keep (f32 below the
 *  6 GB threshold, BF16 native above) plus ~1.5 GB KV/overhead headroom. */
const CURATED_MODELS: Array<{ repo: string; needGB: number; blurb: string; gated?: boolean }> = [
  { repo: 'Qwen/Qwen3-0.6B',             needGB: 4,    blurb: 'small + snappy — quick chat on any GPU' },
  { repo: 'Qwen/Qwen3-1.7B',             needGB: 5,    blurb: 'good quality for light tasks' },
  { repo: 'Qwen/Qwen3.5-2B',             needGB: 5.5,  blurb: 'newer family, strong for its size', gated: true },
  { repo: 'Qwen/Qwen3-4B-Instruct-2507', needGB: 9.5,  blurb: 'solid daily driver' },
  { repo: 'Qwen/Qwen3.5-4B',             needGB: 9.5,  blurb: 'newer family daily driver', gated: true },
  { repo: 'Qwen/Qwen3-VL-4B-Instruct',   needGB: 11,   blurb: 'understands images (📎 attach)' },
  { repo: 'Qwen/Qwen3-8B',               needGB: 18,   blurb: 'best quality on a big card' },
  { repo: 'Qwen/Qwen3.5-9B',             needGB: 20,   blurb: 'best quality, newest family', gated: true },
];

function suggestRow(title: string, sub: string, onPick?: () => void): HTMLElement {
  const el = document.createElement('div');
  el.style.cssText = 'padding:6px 8px;border-bottom:1px solid #222'
    + (onPick ? ';cursor:pointer' : ';color:var(--dim)');
  const t = document.createElement('div');
  t.textContent = title;
  t.style.color = onPick ? 'var(--accent)' : 'var(--dim)';
  const s = document.createElement('div');
  s.textContent = sub;
  s.style.cssText = 'color:var(--dim);font-size:10px';
  el.append(t, s);
  if (onPick) {
    el.addEventListener('mouseenter', () => { el.style.background = '#1a1e33'; });
    el.addEventListener('mouseleave', () => { el.style.background = 'none'; });
    el.addEventListener('click', onPick);
  }
  return el;
}

async function renderSuggestions(): Promise<void> {
  suggestList.innerHTML = '<div style="padding:8px;color:var(--dim)">Checking your GPU...</div>';
  const budget = await resolveVRAMBudget();
  const budgetGB = budget !== undefined ? budget / 1e9 : undefined;

  // Models already on this machine (dev server scan). Heuristic family filter
  // keeps non-LLM repos (whisper etc.) out of the list.
  let locals: Array<{ repo: string; gb: number }> = [];
  try {
    const resp = await fetch('/api/hf-cache/models');
    if (resp.ok) {
      const models = await resp.json() as Array<{ repo: string; files: string[]; totalSize: number }>;
      locals = models
        .filter(m => /qwen|llama|gemma|mistral/i.test(m.repo) && !/mmproj|whisper/i.test(m.repo))
        .filter(m => m.files.some(f => f.endsWith('.safetensors') || f.toLowerCase().endsWith('.gguf')))
        .map(m => ({ repo: m.repo, gb: m.totalSize / 1e9 }));
    }
  } catch { /* no dev server — HF-only mode */ }

  suggestList.innerHTML = '';
  suggestList.appendChild(suggestRow(
    budgetGB !== undefined
      ? `Detected ~${budgetGB.toFixed(1)} GB usable VRAM`
      : 'VRAM unknown — showing options with their needs',
    'estimates include KV-cache headroom'));

  const pick = (repo: string) => {
    ($('model-repo') as HTMLInputElement).value = repo;
    modelSuggest.style.display = 'none';
    suggestOpen = false;
    loadBtn.click();
  };

  let shown = 0;
  // Local models that fit: file size ≈ resident weights for GGUF/BF16,
  // +10% and 1 GB KV headroom.
  for (const m of locals.sort((a, b) => b.gb - a.gb)) {
    const need = m.gb * 1.1 + 1;
    if (budgetGB !== undefined && need > budgetGB) continue;
    if (shown >= 2) break;
    suggestList.appendChild(suggestRow(
      `📁 ${m.repo}`,
      `already on this machine · ~${need.toFixed(1)} GB needed — click to load`,
      () => pick(m.repo)));
    shown++;
  }

  // FLUX.2 klein image gen: the pipeline dir has no top-level weight files,
  // so it never appears in the models listing — probe model_index.json.
  try {
    const p = await fetch('/api/hf-cache/local/flux.2-klein-4b/resolve/main/model_index.json', { method: 'HEAD' });
    if (p.ok) {
      suggestList.appendChild(suggestRow(
        '🎨 local/flux.2-klein-4b',
        'FLUX.2 klein image generation · stages stream in/out of VRAM — click to load',
        () => pick('local/flux.2-klein-4b')));
    }
  } catch { /* no dev server */ }

  // Curated downloads that fit — largest (best) first, up to 3
  const fitting = CURATED_MODELS
    .filter(c => budgetGB === undefined || c.needGB <= budgetGB)
    .sort((a, b) => b.needGB - a.needGB)
    .slice(0, 3);
  const fallback = fitting.length === 0 ? [CURATED_MODELS[0]] : fitting;
  for (const c of fallback) {
    suggestList.appendChild(suggestRow(
      c.repo,
      `${c.blurb} · ~${c.needGB} GB needed${c.gated ? ' · needs HF token' : ''} — click to download + load`,
      () => pick(c.repo)));
    shown++;
  }
  if (fitting.length === 0) {
    suggestList.appendChild(suggestRow(
      'Tight fit on this GPU',
      'the smallest curated model is shown — it may still spill; close other GPU apps first'));
  }
}

suggestBtn.addEventListener('click', () => {
  suggestOpen = !suggestOpen;
  modelSuggest.style.display = suggestOpen ? 'block' : 'none';
  if (suggestOpen) void renderSuggestions();
});

// ─── UI Helpers ──────────────────────────────────────────────────────────────

function setStatus(text: string) {
  statusEl.textContent = text;
}

/** Autoscroll only when the user is already near the bottom — scrolling up to
 *  reread must not fight the stream. */
function scrollMessages(force = false) {
  const nearBottom =
    messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 60;
  if (force || nearBottom) messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addMessage(role: 'user' | 'assistant' | 'system', content: string, meta?: string) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.textContent = content;
  if (meta) {
    const metaEl = document.createElement('div');
    metaEl.className = 'meta';
    metaEl.textContent = meta;
    div.appendChild(metaEl);
  }
  messagesEl.appendChild(div);
  scrollMessages(true);
  return div;
}

/** Render assistant output: dimmed thinking block (plain text) above the
 *  answer (markdown). Replaces the div's content — call attachMeta/attachActions
 *  AFTER the final render. */
function renderAssistant(div: HTMLElement, fullText: string, assumeOpen: boolean) {
  const { thinking, answer } = splitThinking(fullText, assumeOpen);
  const thinkHtml = thinking
    ? `<div class="md-think">${answer ? '' : '💭 '}${escapeHtml(thinking)}</div>`
    : '';
  div.innerHTML = thinkHtml + (answer ? `<div class="md">${markdownToHtml(answer)}</div>` : '');
}

function attachMeta(div: HTMLElement, meta: string) {
  const metaEl = document.createElement('div');
  metaEl.className = 'meta';
  metaEl.textContent = meta;
  div.appendChild(metaEl);
}

/** Link a chat DOM node to its history entry (by object identity — compression
 *  keeps kept-message references, so indexOf() finds live messages). */
function linkMessage(div: HTMLElement, msg: ChatMessage) {
  (div as any).__msg = msg;
}

function findMessageIndex(div: HTMLElement): number {
  const msg = (div as any).__msg as ChatMessage | undefined;
  return msg ? chatHistory.indexOf(msg) : -1;
}

/** Hover actions: ✎ edit on user messages; ⧉ copy + ⟳ retry on assistant. */
function attachActions(div: HTMLElement, role: 'user' | 'assistant') {
  div.querySelector('.msg-actions')?.remove();
  const bar = document.createElement('div');
  bar.className = 'msg-actions';
  const mkBtn = (label: string, title: string, onClick: () => void) => {
    const b = document.createElement('button');
    b.textContent = label;
    b.title = title;
    b.addEventListener('click', onClick);
    bar.appendChild(b);
  };
  if (role === 'user') {
    mkBtn('✎ edit', 'Edit this message and re-send (discards later messages)', () => editMessage(div));
  } else {
    mkBtn('⧉ copy', 'Copy response text', () => {
      const msg = (div as any).__msg as ChatMessage | undefined;
      const text = msg?.content ?? div.textContent ?? '';
      navigator.clipboard?.writeText(text).catch(() => {});
    });
    mkBtn('⟳ retry', 'Regenerate this response (discards it and later messages)', () => regenerateMessage(div));
  }
  div.appendChild(bar);
}

/** Add a message from history to the DOM with full rendering + actions. */
function addChatMessage(msg: ChatMessage): HTMLElement {
  if (msg.role === 'assistant') {
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.innerHTML = `<div class="md">${markdownToHtml(msg.content)}</div>`;
    messagesEl.appendChild(div);
    linkMessage(div, msg);
    attachActions(div, 'assistant');
    scrollMessages(true);
    return div;
  }
  const div = addMessage(msg.role as 'user' | 'system', msg.content);
  if (msg.role === 'user') {
    linkMessage(div, msg);
    attachActions(div, 'user');
  }
  return div;
}

/** Truncate history at (and including) the given message's index, and remove
 *  its DOM node plus everything after it. Returns the removed message, or
 *  null if the message was compressed out of the active history. */
function truncateAt(div: HTMLElement): ChatMessage | null {
  const idx = findMessageIndex(div);
  if (idx === -1) {
    addMessage('system', 'That message was compressed out of the active history and can no longer be edited.');
    return null;
  }
  const msg = chatHistory[idx];
  chatHistory = chatHistory.slice(0, idx);
  const doomed: Element[] = [];
  let n: Element | null = div;
  while (n) { doomed.push(n); n = n.nextElementSibling; }
  doomed.forEach(e => e.remove());
  session?.resetKV?.();  // history changed — next send re-prefills
  return msg;
}

function editMessage(div: HTMLElement) {
  if (activeHandle) return;  // not while generating
  const msg = truncateAt(div);
  if (!msg) return;
  promptEl.value = msg.content;
  promptEl.style.height = 'auto';
  promptEl.style.height = Math.min(promptEl.scrollHeight, 120) + 'px';
  promptEl.focus();
}

function regenerateMessage(div: HTMLElement) {
  if (activeHandle) return;
  if (!session) {
    addMessage('system', 'No model loaded. Load a model first.');
    return;
  }
  // Find the user message that produced this response, walking back in history
  const idx = findMessageIndex(div);
  if (idx === -1) {
    addMessage('system', 'That message was compressed out of the active history and can no longer be regenerated.');
    return;
  }
  let u = idx - 1;
  while (u >= 0 && chatHistory[u].role !== 'user') u--;
  if (u < 0) return;
  const userMsg = chatHistory[u];
  const text = userMsg.content;
  // Locate the user message's DOM node so it survives the truncation
  const userDiv = [...messagesEl.children].find(el => (el as any).__msg === userMsg) as HTMLElement | undefined;
  chatHistory = chatHistory.slice(0, u);
  if (userDiv) {
    const doomed: Element[] = [];
    let n: Element | null = userDiv.nextElementSibling;
    while (n) { doomed.push(n); n = n.nextElementSibling; }
    doomed.forEach(e => e.remove());
  } else {
    div.remove();
  }
  session.resetKV?.();
  void runGeneration(text, userDiv ?? addMessage('user', text));
}

/** Toggle Send/Stop while a generation is in flight. */
function setGenerating(on: boolean) {
  sendBtn.style.display = on ? 'none' : 'inline-block';
  stopBtn.style.display = on ? 'inline-block' : 'none';
  sendBtn.disabled = on;
}

stopBtn.addEventListener('click', () => activeHandle?.abort());
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && activeHandle) activeHandle.abort();
});

// Code-block copy buttons (delegated — buttons are re-created on each
// streaming re-render, the listener survives)
messagesEl.addEventListener('click', (e) => {
  const btn = (e.target as HTMLElement).closest?.('.md-copy') as HTMLButtonElement | null;
  if (!btn) return;
  const pre = btn.closest('.md-code')?.querySelector('pre');
  if (!pre) return;
  navigator.clipboard?.writeText(pre.textContent ?? '').then(() => {
    btn.textContent = 'copied';
    setTimeout(() => { btn.textContent = 'copy'; }, 1200);
  }).catch(() => {});
});

function updateFooter(data: Record<string, string>) {
  for (const [key, value] of Object.entries(data)) {
    const el = document.getElementById(`f-${key}`);
    if (el) el.textContent = value;
  }
}

// ─── Slider Bindings ─────────────────────────────────────────────────────────

function flipToCustom(): void {
  if (!suppressPresetFlip && presetSelect.value !== 'custom') {
    presetSelect.value = 'custom';
  }
}

tempSlider.addEventListener('input', () => {
  tempVal.textContent = tempSlider.value;
  flipToCustom();
});

toppSlider.addEventListener('input', () => {
  toppVal.textContent = toppSlider.value;
  flipToCustom();
});

minpSlider.addEventListener('input', () => {
  minpVal.textContent = parseFloat(minpSlider.value).toFixed(2);
  flipToCustom();
});

reppenSlider.addEventListener('input', () => {
  reppenVal.textContent = parseFloat(reppenSlider.value).toFixed(2);
  flipToCustom();
});

drySlider.addEventListener('input', () => {
  dryVal.textContent = parseFloat(drySlider.value).toFixed(2);
  flipToCustom();
});

topkInput.addEventListener('input', flipToCustom);

presetSelect.addEventListener('change', () => {
  const v = presetSelect.value;
  if (v !== 'custom') applyPreset(v as PresetName);
});

// Apply the default preset once at startup so the extra sliders reflect it
applyPreset('balanced');

// Auto-resize textarea
promptEl.addEventListener('input', () => {
  promptEl.style.height = 'auto';
  promptEl.style.height = Math.min(promptEl.scrollHeight, 120) + 'px';
});

// Enter to send (Shift+Enter for newline)
promptEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});

// ─── WebGPU Initialization ───────────────────────────────────────────────────

// Track discovered adapters for GPU switching
let discoveredAdapters: DiscoveredAdapter[] = [];

async function initGPU(adapter?: GPUAdapter) {
  gpu = await timed('perf', 'webgpu-init', () => initWebGPU(adapter));
  const info = gpu.adapterInfo;
  const label = info.device || info.description || info.architecture || 'Unknown GPU';
  const maxMB = Math.round(gpu.maxBufferSize / (1024 * 1024));

  gpuBadge.textContent = `${label} (${maxMB} MB max buffer)`;
  gpuBadge.classList.remove('error');
  updateFooter({ gpu: label, vram: `${maxMB} MB max` });

  addMessage('system',
    `GPU: ${label}\n` +
    `Vendor: ${info.vendor || 'unknown'} | Arch: ${info.architecture || 'unknown'}\n` +
    `Max buffer: ${maxMB} MB | Max storage bindings: ${gpu.limits.maxStorageBuffersPerShaderStage}\n` +
    `Max workgroup: ${gpu.limits.maxComputeWorkgroupSizeX}x${gpu.limits.maxComputeWorkgroupSizeY}x${gpu.limits.maxComputeWorkgroupSizeZ}`
  );
}

async function init() {
  setStatus('Detecting GPUs...');

  // Phase C MoE expert workers need SharedArrayBuffer, which requires
  // cross-origin isolation (COOP/COEP — served by vite.config.ts).
  if (!crossOriginIsolated) {
    console.warn('[Init] NOT crossOriginIsolated — SharedArrayBuffer unavailable; MoE (GGUF expert-offload) models will not load. Check COOP/COEP headers.');
  } else {
    console.log('[Init] crossOriginIsolated — SharedArrayBuffer available');
  }

  try {
    // Discover all available GPU adapters
    discoveredAdapters = await discoverAdapters();
    console.log(`[Init] Discovered ${discoveredAdapters.length} GPU adapter(s)`);

    // Always show GPU selector dropdown (even with 1 GPU, for visibility)
    gpuSelect.innerHTML = '';
    if (discoveredAdapters.length > 0) {
      for (let i = 0; i < discoveredAdapters.length; i++) {
        const da = discoveredAdapters[i];
        const opt = document.createElement('option');
        opt.value = String(i);
        opt.textContent = `${da.label} (${da.maxBufferMB} MB)`;
        gpuSelect.appendChild(opt);
      }
      gpuSelect.style.display = 'inline-block';
      gpuBadge.style.display = 'none';

      // Initialize with first (high-performance) adapter
      await initGPU(discoveredAdapters[0].adapter);

      if (discoveredAdapters.length > 1) {
        setStatus(`WebGPU ready — ${discoveredAdapters.length} GPUs detected. Select GPU and load a model.`);
      } else {
        setStatus('WebGPU ready — load a model or run kernel tests');
      }

      // Handle GPU switch
      gpuSelect.addEventListener('change', async () => {
        const idx = parseInt(gpuSelect.value);
        const da = discoveredAdapters[idx];
        if (currentModel) {
          addMessage('system', 'Unload the current model before switching GPUs.');
          return;
        }
        setStatus(`Switching to ${da.label}...`);
        try {
          await initGPU(da.adapter);
          setStatus(`Switched to ${da.label} — ready to load a model.`);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          addMessage('system', `Failed to switch GPU: ${msg}`);
        }
      });
    } else {
      // No adapters found via discovery — fall back to direct init
      gpuSelect.style.display = 'none';
      gpuBadge.style.display = 'inline-block';
      await initGPU();
      setStatus('WebGPU ready — load a model or run kernel tests');
    }

    startKernelTestPoller();

  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    gpuBadge.textContent = 'No WebGPU';
    gpuBadge.classList.add('error');
    gpuSelect.style.display = 'none';
    gpuBadge.style.display = 'inline-block';
    setStatus(`WebGPU Error: ${msg}`);
    addMessage('system', `WebGPU initialization failed: ${msg}`);
    reportError('init', err);
  }
}

// ─── Kernel Tests ────────────────────────────────────────────────────────────

/** Run kernel tests on the current device and POST results to /api/debug. */
async function runKernelTestsToDebugAPI(): Promise<void> {
  const results = await runKernelTests(gpu!.device);
  const passed = results.filter(r => r.passed).length;
  console.log(`[AutoTest] Kernel tests: ${passed}/${results.length} passed`);
  await fetch('/api/debug', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind: 'kernel-tests', passed, failed: results.length - passed, results }),
  });
}

/**
 * Pre-session /api/test poller: lets the dev loop trigger kernel tests via
 * `POST /api/test {"kernelTests":true}` without a model loaded. Stops as soon
 * as a session exists (the per-session poller owns the queue from then on,
 * and handles kernelTests itself). Prompt items popped here without a session
 * are re-queued untouched.
 */
let kernelPollerActive = false;
function startKernelTestPoller(): void {
  if (kernelPollerActive) return;
  kernelPollerActive = true;
  (async () => {
    if (!(await devApiAvailable)) { kernelPollerActive = false; return; }
    while (!session) {
      try {
        const resp = await fetch('/api/test');
        const test = await resp.json();
        if (test && test.kernelTests === true && gpu) {
          await runKernelTestsToDebugAPI();
        } else if (test && typeof test.loadRepo === 'string') {
          // Dev-loop model load: set the repo field and click Load.
          ($('model-repo') as HTMLInputElement).value = test.loadRepo;
          loadBtn.click();
        } else if (test && test.prompt) {
          // Not ours — a prompt test queued before any model loaded. Put it back.
          await fetch('/api/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(test),
          });
        }
      } catch { /* dev-server hiccup — keep polling */ }
      await new Promise(r => setTimeout(r, 1000));
    }
    kernelPollerActive = false;
  })();
}

testBtn.addEventListener('click', async () => {
  if (!gpu) {
    testResults.textContent = 'No GPU — initialize WebGPU first';
    return;
  }

  testBtn.disabled = true;
  testResults.textContent = 'Running kernel tests...\n';

  try {
    const results = await runKernelTests(gpu.device);

    let output = '';
    let passed = 0;
    let failed = 0;

    for (const r of results) {
      const icon = r.passed ? '\u2713' : '\u2717';
      const color = r.passed ? 'color:var(--success)' : 'color:var(--error)';
      output += `${icon} ${r.name}: ${r.passed ? 'PASS' : 'FAIL'}`;
      if (r.elapsed_ms !== undefined) {
        output += ` (${r.elapsed_ms.toFixed(2)}ms)`;
      }
      if (r.error) {
        output += ` — ${r.error}`;
      }
      output += '\n';
      r.passed ? passed++ : failed++;
    }

    output += `\n${passed} passed, ${failed} failed`;
    testResults.textContent = output;

    await reportMetric('kernel-test', { passed, failed, results });

    addMessage('system', `Kernel tests: ${passed} passed, ${failed} failed`);

  } catch (err) {
    testResults.textContent = `Error: ${err}`;
    reportError('kernel-test', err);
  } finally {
    testBtn.disabled = false;
  }
});

// ─── Vision: attach / paste / drop ───────────────────────────────────────────

const attachBtn = $('attach-btn') as HTMLButtonElement;
const imageInput = $('image-input') as HTMLInputElement;
const chipsEl = $('image-chips') as HTMLDivElement;
const SUPPORTED_IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/webp', 'image/gif', 'image/bmp'];
const SUPPORTED_IMAGE_LABEL = 'PNG, JPEG, WebP, GIF, BMP';

// The 📎 button is always live: text/code files attach on any model, images
// additionally need a vision model. visionAvailable gates only the image
// route; the title explains what's possible right now.
let visionAvailable = false;

function setAttachState(available: boolean, title: string) {
  visionAvailable = available;
  attachBtn.title = available ? title : `${title} Text/code files attach on any model.`;
  attachBtn.style.opacity = '1';
}

function setVisionDesc(desc: VisionDescriptor | null) {
  activeVisionDesc = desc;
  if (!desc) {
    setAttachState(false, 'This model does not support images — load a vision model (e.g. local/qwen3-vl-4b-instruct).');
  } else if (!desc.verified) {
    setAttachState(true, `Attach images (${SUPPORTED_IMAGE_LABEL}) — ${desc.family} vision is EXPERIMENTAL`);
    addMessage('system',
      `Vision detected (${desc.family}) — EXPERIMENTAL: this family hasn't passed a parity run yet. `
      + `Attach images with 📎 to try it; judge the output accordingly.`);
  } else {
    setAttachState(true, `Attach images (${SUPPORTED_IMAGE_LABEL}) — or paste / drag-drop`);
    addMessage('system',
      `Vision model detected — attach images with 📎, paste, or drag-drop. Supported: ${SUPPORTED_IMAGE_LABEL}.`);
  }
}

function resetVision() {
  visionEncoder?.destroy();
  visionEncoder = null;
  visionEncoderLoading = null;
  pendingImages = [];
  renderChips();
  setVisionDescQuiet(null);
}
// setVisionDesc(null) without the announcement (used on unload)
function setVisionDescQuiet(desc: VisionDescriptor | null) {
  activeVisionDesc = desc;
  setAttachState(false, 'This model does not support images — load a vision model (e.g. local/qwen3-vl-4b-instruct).');
}

function renderChips() {
  chipsEl.style.display = (pendingImages.length + pendingFiles.length) > 0 ? 'flex' : 'none';
  chipsEl.innerHTML = '';
  const addChip = (label: string, onRemove: () => void) => {
    const chip = document.createElement('span');
    chip.style.cssText =
      'display:inline-flex;align-items:center;gap:4px;background:#1a1e33;border:1px solid #333;'
      + 'border-radius:12px;padding:2px 8px;font-size:11px';
    chip.textContent = label;
    const x = document.createElement('button');
    x.textContent = '✕';
    x.style.cssText = 'background:none;border:none;color:var(--dim);cursor:pointer;font-size:11px';
    x.addEventListener('click', onRemove);
    chip.appendChild(x);
    chipsEl.appendChild(chip);
  };
  pendingImages.forEach((img, i) => {
    addChip(`🖼 ${img.name} (~${img.pre.numTokens} tok)`,
      () => { pendingImages.splice(i, 1); renderChips(); });
  });
  pendingFiles.forEach((f, i) => {
    addChip(`📄 ${f.name} (~${Math.ceil(f.text.length / 4)} tok${f.truncated ? ', truncated' : ''})`,
      () => { pendingFiles.splice(i, 1); renderChips(); });
  });
}

async function addPendingImage(blob: Blob, name: string) {
  if (!activeVisionDesc || !visionAvailable) {
    addMessage('system', attachBtn.title);
    return;
  }
  if (blob.type && blob.type.startsWith('image/') && !SUPPORTED_IMAGE_TYPES.includes(blob.type)) {
    addMessage('system', `Unsupported image type "${blob.type}". Supported: ${SUPPORTED_IMAGE_LABEL}.`);
    return;
  }
  try {
    const pre = await preprocessImage(blob, activeVisionDesc);
    pendingImages.push({ name, pre });
    renderChips();
  } catch (err) {
    addMessage('system', `Could not read image "${name}": ${err}`);
  }
}

async function ensureVisionEncoder(): Promise<VisionEncoder> {
  if (visionEncoder) return visionEncoder;
  if (!visionEncoderLoading) {
    visionEncoderLoading = (async () => {
      const { loadVisionWeights } = await import('./vision/vision-loader');
      const { createVisionEncoder } = await import('./vision/vision-encoder');
      // The text load path resets hf-hub to the CDN once the model is up;
      // machine-local repos must switch back to the dev-server cache for the
      // lazy tower fetch (the CDN answers 401/404 for local/ and ollama/).
      const repo = currentModel!.repo;
      const isLocalRepo = repo.startsWith('local/') || repo.startsWith('ollama/');
      if (isLocalRepo) useLocalCache();
      try {
        const desc = activeVisionDesc!;
        const weights = activeVisionSource.kind === 'gguf'
          ? desc.towerVariant === 'gemma4'
            ? await (await import('./vision/vision-loader-gguf')).loadGemmaVisionWeightsGGUF(
                gpu!.device, repo, activeVisionSource.file, desc, (m) => setStatus(m))
            : await (await import('./vision/vision-loader-gguf')).loadVisionWeightsGGUF(
                gpu!.device, repo, activeVisionSource.file, desc, (m) => setStatus(m))
          : await loadVisionWeights(
              gpu!.device, repo, desc, (m) => setStatus(m));
        visionEncoder = createVisionEncoder(gpu!.device, desc, weights);
        addMessage('system',
          `Vision tower loaded: ${(weights.totalGPUBytes / 1e9).toFixed(2)} GB GPU.`, 'vision ready');
        return visionEncoder;
      } finally {
        if (isLocalRepo) resetToRemote();
      }
    })().catch((err) => {
      visionEncoderLoading = null;  // allow retry after a failure
      throw err;
    });
  }
  return visionEncoderLoading;
}

attachBtn.addEventListener('click', () => {
  imageInput.click();
});
imageInput.addEventListener('change', async (e) => {
  const input = e.target as HTMLInputElement;
  for (const f of Array.from(input.files ?? [])) {
    if (f.type.startsWith('image/')) await addPendingImage(f, f.name);
    else if (isTextAttachment(f)) await addPendingFile(f);
    else addMessage('system', `"${f.name}" isn't a supported attachment — images (${SUPPORTED_IMAGE_LABEL}) or text/code files.`);
  }
  input.value = '';
});
promptEl.addEventListener('paste', async (e) => {
  const items = Array.from(e.clipboardData?.items ?? []).filter(it => it.type.startsWith('image/'));
  if (items.length === 0) return;
  e.preventDefault();
  for (const it of items) {
    const f = it.getAsFile();
    if (f) await addPendingImage(f, f.name || 'pasted-image');
  }
});
document.addEventListener('dragover', (e) => {
  if (e.dataTransfer?.types.includes('Files')) e.preventDefault();
});
document.addEventListener('drop', async (e) => {
  const files = Array.from(e.dataTransfer?.files ?? [])
    .filter(f => f.type.startsWith('image/') || isTextAttachment(f));
  if (files.length === 0) return;
  e.preventDefault();
  for (const f of files) {
    if (f.type.startsWith('image/')) await addPendingImage(f, f.name);
    else await addPendingFile(f);
  }
});

// ─── Vision parity harness (console: __VISION_PARITY__()) ──────────────────
// Compares the browser pipeline stage-by-stage against the Python golden
// fixture (webgpu/scripts/gen_vision_fixture.py): preprocessing patch matrix,
// then tower output + deepstack features. Requires a vision model loaded.
(globalThis as any).__VISION_PARITY__ = async () => {
  if (!activeVisionDesc) { console.error('Load a vision model first'); return; }
  // Each family has its own golden fixture — comparing across families is
  // meaningless (different patch counts, hidden sizes). Pick by family.
  const fixtureByFamily: Record<string, string> = {
    qwen3_vl: 'vision-qwen3vl-golden.json',
    gemma4: 'vision-gemma4-golden.json',
  };
  const fixtureName = fixtureByFamily[activeVisionDesc.family];
  if (!fixtureName) {
    console.error(`[parity] no golden fixture for family "${activeVisionDesc.family}"`);
    return;
  }
  const resp = await fetch(`/test-fixtures/${fixtureName}`);
  if (!resp.ok) {
    console.error(`[parity] fixture ${fixtureName} missing — run the matching scripts/gen_*_fixture.py`);
    return;
  }
  const fx = await resp.json();
  const png = Uint8Array.from(atob(fx.png_base64), c => c.charCodeAt(0));
  const pre = await preprocessImage(new Blob([png], { type: 'image/png' }), activeVisionDesc);

  const diff = (name: string, ours: Float32Array | number[], ref: number[]) => {
    if (ours.length !== ref.length) {
      console.error(`[parity] ${name}: LENGTH ${ours.length} vs ${ref.length}`);
      return;
    }
    let maxAbs = 0, sumAbs = 0, at = -1;
    for (let i = 0; i < ref.length; i++) {
      const d = Math.abs((ours as any)[i] - ref[i]);
      if (d > maxAbs) { maxAbs = d; at = i; }
      sumAbs += d;
    }
    console.log(`[parity] ${name}: maxAbsDiff=${maxAbs.toExponential(3)} (at ${at}) meanAbsDiff=${(sumAbs / ref.length).toExponential(3)}`);
  };

  diff('pixel_values (preprocess)', pre.patches, fx.pixel_values);
  const encoder = await ensureVisionEncoder();
  const out = await encoder.encode(pre);
  diff('image_embeds (tower+merger)', out.embeddings, fx.image_embeds);
  out.deepstack.forEach((ds, i) => diff(`deepstack[${i}]`, ds, fx.deepstack[i] ?? []));
  console.log('[parity] done — preprocessing should be ~1e-6; tower f32-vs-f32 ~1e-3 or better');
};

// ─── TurboQuant parity harness (console: __TQ_PARITY__("prompt", N)) ───────
// Greedy-decodes the same prompt twice — exact f32 KV vs TurboQuant
// compressed KV — and reports the first divergent token. The honest gate
// before trusting compression on a given model: a few divergences late in a
// long decode is expected (lossy KV); early/frequent divergence means the
// compressed path is wrong for this family.
(globalThis as any).__TQ_PARITY__ = async (prompt = 'Explain how a rainbow forms, step by step.', maxNew = 64) => {
  if (!session) { console.error('Load a model first'); return; }
  const greedy = { temperature: 0, maxNewTokens: maxNew } as any;
  const collect = async (useCompressedKV: boolean) => {
    const ids: number[] = [];
    const messages = [{ role: 'user', content: prompt }];
    const h = (session as any).chat(messages, { ...greedy, useCompressedKV },
      (_t: string, id: number) => ids.push(id));
    await h.result;
    return ids;
  };
  console.log('[tq-parity] decoding exact (f32 KV)...');
  const exact = await collect(false);
  console.log('[tq-parity] decoding compressed (TurboQuant KV)...');
  let comp: number[];
  try {
    comp = await collect(true);
  } catch (err) {
    console.error('[tq-parity] compressed decode refused/failed:', err);
    return;
  }
  const n = Math.min(exact.length, comp.length);
  let firstDiff = -1;
  for (let i = 0; i < n; i++) if (exact[i] !== comp[i]) { firstDiff = i; break; }
  if (firstDiff === -1 && exact.length === comp.length) {
    console.log(`[tq-parity] IDENTICAL across all ${n} tokens — TurboQuant is exact-match on ${session.config.modelType}`);
  } else {
    const at = firstDiff === -1 ? n : firstDiff;
    console.log(`[tq-parity] first divergence at token ${at}/${n} (${((at / n) * 100).toFixed(0)}% in) — `
      + `exact="${session.tokenizer.decode(exact.slice(Math.max(0, at - 2), at + 1))}" `
      + `comp="${session.tokenizer.decode(comp.slice(Math.max(0, at - 2), at + 1))}"`);
    console.log('[tq-parity] late single-token divergence = acceptable lossy KV; early/frequent = compressed path bug');
  }
};

/** Resolve the VRAM budget for model loading. Priority: manual override
 *  (localStorage 'vramBudgetGB') → driver-reported free memory on the GPU
 *  matching the ACTIVE WebGPU adapter's vendor, minus a compositor/safety
 *  reserve → undefined (loader's conservative default). /api/gpu-info is
 *  nvidia-smi-based, so when WebGPU runs on a non-NVIDIA adapter (e.g. the
 *  Radeon via #gpu-select) there is no matching entry — fall back to the
 *  loader default rather than budgeting against the wrong card. */
async function resolveVRAMBudget(): Promise<number | undefined> {
  const manual = Number(localStorage.getItem('vramBudgetGB'));
  if (manual > 0) return manual * 1e9;
  try {
    const resp = await fetch('/api/gpu-info');
    const isJson = (resp.headers.get('content-type') ?? '').includes('json');
    if (resp.ok && isJson) {
      const { gpus } = await resp.json() as { gpus: Array<{ name: string; freeMB: number; displayActive: boolean }> };
      const vendor = (gpu?.adapterInfo?.vendor ?? '').toLowerCase();
      const vendorRe: Record<string, RegExp> = {
        nvidia: /nvidia|geforce|rtx|gtx|quadro/i,
        amd: /amd|radeon/i,
        intel: /intel|arc/i,
      };
      const re = vendorRe[vendor];
      if (re && !gpus.some(g => re.test(g.name))) {
        console.warn(`[VRAM] no /api/gpu-info entry matches active adapter vendor "${vendor}" — using loader default budget`);
        return undefined;
      }
      // Cards matching the active adapter's vendor. On a multi-GPU box with two
      // same-vendor cards (e.g. a display GPU + a discrete inference GPU) the
      // vendor regex matches BOTH, and a plain `.find()` grabs whichever
      // nvidia-smi lists first — often the display card at index 0, budgeting
      // the load against the wrong, smaller GPU. WebGPU never reports VRAM, but
      // it DOES report the adapter architecture — use it to pin the budget to
      // the card WebGPU actually chose, falling back to the same-vendor card
      // with the most free VRAM (the idle discrete inference target).
      const candidates = re ? gpus.filter(g => re.test(g.name)) : gpus.slice();
      const arch = (gpu?.adapterInfo?.architecture ?? '').toLowerCase();
      const archToName: Array<[RegExp, RegExp]> = [
        [/lovelace|ada/, /rtx\s*40\d\d|\bada\b/i],         // 40-series / Ada
        [/blackwell/,    /rtx\s*50\d\d/i],                 // 50-series
        [/ampere/,       /rtx\s*30\d\d|\ba\d{3,4}\b/i],    // 30-series / A-series
        [/turing/,       /rtx\s*20\d\d|gtx\s*16\d\d/i],
        [/pascal/,       /gtx\s*10\d\d|titan\s*x|\bp\d{3,4}\b/i], // 10-series
      ];
      const nameRe = archToName.find(([a]) => a.test(arch))?.[1];
      const pick =
        // 1) exact card by architecture (authoritative on multi-GPU boxes)
        (candidates.length > 1 && nameRe ? candidates.find(g => nameRe.test(g.name)) : undefined)
        // 2) else the same-vendor card with the most free VRAM
        ?? candidates.slice().sort((a, b) => b.freeMB - a.freeMB)[0]
        // 3) last resort: display-active, then first entry
        ?? gpus.find(g => g.displayActive) ?? gpus[0];
      if (!pick) {
        console.warn(`[VRAM] no /api/gpu-info entry matches active adapter vendor "${vendor}" — using loader default budget`);
        return undefined;
      }
      if (pick.freeMB > 0) {
        const budget = Math.max(2e9, (pick.freeMB - 1024) * 1e6);
        console.log(`[VRAM] auto budget: ${(budget / 1e9).toFixed(1)} GB on "${pick.name}" (arch="${arch}", driver-reported free minus 1 GB reserve)`);
        return budget;
      }
      console.warn('[VRAM] /api/gpu-info returned no usable GPUs — using loader default budget');
    } else {
      // A non-JSON 200 means the request never reached the dev server
      // (e.g. vite served its SPA fallback because the path isn't proxied).
      console.warn(`[VRAM] /api/gpu-info unavailable (status ${resp.status}, json=${isJson}) — using loader default budget`);
    }
  } catch (err) {
    console.warn('[VRAM] gpu-info fetch failed — using loader default budget:', err);
  }
  return undefined;
}

/** Fetch preprocessor_config.json — optional. Machine-local repos go to the
 *  dev-server cache directly (resolveFileUrl may point at the CDN by now). */
async function fetchPreprocessorConfig(repo: string): Promise<Record<string, any> | undefined> {
  const url = repo.startsWith('local/') || repo.startsWith('ollama/')
    ? `/api/hf-cache/${repo}/raw/main/preprocessor_config.json`
    : resolveFileUrl(repo, 'preprocessor_config.json');
  try {
    const resp = await fetch(url);
    if (resp.ok) return await resp.json();
  } catch { /* optional file */ }
  return undefined;
}

// ─── Chat ─────────────────────────────────────────────────────────────────────

sendBtn.addEventListener('click', () => {
  const typed = promptEl.value.trim();
  if (!typed && pendingFiles.length === 0) return;

  promptEl.value = '';
  promptEl.style.height = 'auto';
  // Attached text files ride inside the message; the window shows a compact
  // 📄 line while history carries the full content.
  const { sendText, displayText } = typed.startsWith('/raw ')
    ? { sendText: typed, displayText: typed }
    : consumePendingFiles(typed);
  const userDiv = addMessage('user', displayText);

  if (imageGenRepo) {
    void runImageGeneration(sendText, userDiv);
    return;
  }
  if (!session) {
    addMessage('system', 'No model loaded. Load a model first.');
    return;
  }
  void runGeneration(sendText, userDiv);
});

/** How many tool hops one Send can trigger before the loop is cut. */
const MAX_TOOL_HOPS = 5;

/** One chat turn: window history, stream the response with markdown rendering,
 *  write the turn back to history. When tools are enabled and the model ends
 *  with a ```tool_call block, the tool runs and the model continues — up to
 *  MAX_TOOL_HOPS times. Shared by Send, ⟳ retry, and ✎ edit. */
async function runGeneration(text: string, userDiv: HTMLElement): Promise<void> {
  if (!session || activeHandle) return;

  setStatus('Generating...');
  setGenerating(true);

  // /raw prefix: skip chat template, use raw text completion (for debugging)
  const isRaw = text.startsWith('/raw ');
  const userMsg: ChatMessage = { role: 'user', content: text };
  if (!isRaw) {
    linkMessage(userDiv, userMsg);
    attachActions(userDiv, 'user');
  }

  try {
    const temperature = parseFloat(tempSlider.value);
    const topP = parseFloat(toppSlider.value);
    const topK = parseInt(topkInput.value) || 0;
    const minP = parseFloat(minpSlider.value);
    const repetitionPenalty = parseFloat(reppenSlider.value);
    const dryMultiplier = parseFloat(drySlider.value);
    const maxNewTokens = parseInt(($('max-tokens') as HTMLInputElement).value) || 512;
    const useCompressedKV = ($('turboquant') as HTMLInputElement).checked;

    const sampling = {
      temperature, topP, topK, minP, repetitionPenalty, dryMultiplier,
      maxNewTokens, useCompressedKV,
    };
    // Gemma emits its thinking opener in the generation; Qwen-style templates
    // put the opener in the prompt, so marker-less text counts as thinking.
    const assumeThinkingOpen = !(session.config.modelType ?? '').toLowerCase().includes('gemma');

    /** Stream one model response into a fresh assistant bubble. */
    const streamTurn = async (buildHandle: (onToken: (t: string) => void) => GenerationHandle) => {
      const responseDiv = addMessage('assistant', '');
      let fullText = '';
      // Markdown-render the stream at most every 80 ms — a full re-parse per
      // token is wasteful at high tok/s and imperceptible below it.
      let lastRender = 0;
      const onToken = (token: string) => {
        fullText += token;
        const now = performance.now();
        if (now - lastRender > 80) {
          lastRender = now;
          renderAssistant(responseDiv, fullText, assumeThinkingOpen);
          scrollMessages();
        }
      };
      const handle = buildHandle(onToken);
      activeHandle = handle;
      const result = await handle.result;
      return { responseDiv, result };
    };

    if (isRaw) {
      const rawText = text.slice(5); // strip "/raw "
      console.log(`[Raw mode] Sending: "${rawText}"`);
      const { responseDiv, result } = await streamTurn(
        (onToken) => session!.run(rawText, sampling, onToken));
      responseDiv.textContent = result.text;
      attachMeta(responseDiv,
        `${result.numTokens} tokens | ${result.tokensPerSecond.toFixed(1)} tok/s | `
        + `${(result.totalMs / 1000).toFixed(1)}s | ${result.stopReason}`);
      setStatus(`Generated ${result.numTokens} tokens at ${result.tokensPerSecond.toFixed(1)} tok/s`);
      return;
    }

    const toolsOn = ($('tools-enabled') as HTMLInputElement).checked;
    // web_search is offered only when the user opted in AND the local server
    // is present (the hosted static build has no /api/search).
    const webSearchOn = toolsOn
      && ($('web-search-enabled') as HTMLInputElement).checked
      && await devApiAvailable;

    // The message that enters history on the next hop: the user's text first,
    // then tool results on later hops (already pushed to chatHistory directly).
    let pendingUser: ChatMessage | null = userMsg;

    for (let hop = 0; hop <= MAX_TOOL_HOPS; hop++) {
      let systemPrompt = ($('system-prompt') as HTMLTextAreaElement).value.trim();
      if (toolsOn) systemPrompt += toolSystemPreamble(webSearchOn);

      // Window the conversation: system pinned at [0] (context.ts contract),
      // recent turns kept within the KV budget, older turns compressed to
      // key points. The possibly-compressed history is written back as the
      // new source of truth — same flow as the Artifex GUI.
      const fullHistory: ChatMessage[] = [
        { role: 'system', content: systemPrompt },
        ...chatHistory,
        ...(pendingUser ? [pendingUser] : []),
      ];
      const countTokens = (t: string) => session!.tokenizer.encode(t).length;
      const { history: updated, active } = buildActiveMessages(fullHistory, {
        engineCtx: session.kvSession?.capacity ?? 8192,
        countTokens,
      });
      chatHistory = updated.slice(1);  // write back, minus the system slot
      pendingUser = null;
      const sendMessages = active[0].content ? active : active.slice(1);  // drop empty system

      let turn: { responseDiv: HTMLElement; result: any };
      if (hop === 0 && pendingImages.length > 0 && activeVisionDesc && (session as any).chatMM) {
        // Multimodal turn: encode pending images, prefix the user message
        // with placeholder spans, and send via the MultimodalPrompt path.
        // History keeps the plain text — images are per-turn in v1.
        const encoder = await ensureVisionEncoder();
        const images = pendingImages;
        pendingImages = [];
        renderChips();
        const encoded: VisionEncodeResult[] = [];
        for (const img of images) {
          setStatus(`Encoding ${img.name}...`);
          encoded.push(await encoder.encode(img.pre));
        }
        const ph = activeVisionDesc.placeholder;
        const startT = ph.startText ?? '<|vision_start|>';
        const padT = ph.padText ?? '<|image_pad|>';
        const endT = ph.endText ?? '<|vision_end|>';
        const padBlock = encoded.map(e => `${startT}${padT.repeat(e.numTokens)}${endT}`).join('');
        const mmMessages = sendMessages.map((m, idx) =>
          idx === sendMessages.length - 1 && m.role === 'user'
            ? { ...m, content: padBlock + m.content }
            : m);
        turn = await streamTurn((onToken) =>
          (session as any).chatMM(mmMessages, encoded, sampling, onToken, { enableThinking: true }));
      } else {
        turn = await streamTurn((onToken) =>
          (session as any).chat(sendMessages, sampling, onToken, { enableThinking: true }));
      }
      const { responseDiv, result } = turn;

      // Final settle: a non-thinking model never emits think markers, so the
      // stream rendered everything dimmed under assumeThinkingOpen. On a clean
      // EOS with no markers anywhere, treat it all as the answer. (On
      // max_length/abort with assumeOpen, it really was mid-thought.)
      const sawMarker = THINK_MARKERS.some(
        m => result.text.includes(m.open) || result.text.includes(m.close));
      const finalAssumeOpen = assumeThinkingOpen && (sawMarker || result.stopReason !== 'eos');
      renderAssistant(responseDiv, result.text, finalAssumeOpen);
      scrollMessages();

      const asstMsg: ChatMessage = { role: 'assistant', content: stripThinking(result.text) };
      chatHistory.push(asstMsg);
      linkMessage(responseDiv, asstMsg);
      attachActions(responseDiv, 'assistant');
      autoSave(chatHistory, { model: currentModel?.repo ?? 'unknown', backend: 'webgpu' });

      attachMeta(responseDiv,
        `${result.numTokens} tokens | ${result.tokensPerSecond.toFixed(1)} tok/s | `
        + `${(result.totalMs / 1000).toFixed(1)}s | ${result.stopReason}`);
      setStatus(result.stopReason === 'aborted'
        ? `Stopped after ${result.numTokens} tokens`
        : `Generated ${result.numTokens} tokens at ${result.tokensPerSecond.toFixed(1)} tok/s`);

      // ── Tool hop? ──────────────────────────────────────────────────────
      if (!toolsOn || result.stopReason !== 'eos') break;
      const call = parseToolCall(asstMsg.content);
      if (!call) break;
      if (hop === MAX_TOOL_HOPS) {
        addMessage('system', `Tool-call limit (${MAX_TOOL_HOPS}) reached — stopping. Send a message to continue.`);
        break;
      }
      setStatus(`Running ${call.tool || 'tool call'}...`);
      const toolResult = await executeToolCall(call, { webSearchEnabled: webSearchOn });
      // [TOOL RESULT ...] rides as a user message — the compressor already
      // knows how to summarize that prefix when the chat grows long.
      chatHistory.push({ role: 'user', content: toolResult });
      const preview = toolResult.length > 500 ? toolResult.slice(0, 500) + ' …' : toolResult;
      addMessage('system', `🔧 ${preview}`);
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    addMessage('system', `Generation error: ${msg}`);
    setStatus('Generation failed');
    reportError('generate', err);
  } finally {
    activeHandle = null;
    setGenerating(false);
  }
}

// ─── Auto-test polling ──────────────────────────────────────────────────────
// Polls /api/test for queued prompts/kernel-test requests while a session
// exists. Shared by the safetensors and GGUF load paths.
function startAutoTestPoller() {
      // Intercept console.log to capture debug output from forward pass
      const _origLog = console.log;
      let debugLogs: string[] = [];

      (async function pollTests() {
        if (!(await devApiAvailable)) return;
        while (session) {
          try {
            const resp = await fetch('/api/test');
            const test = await resp.json();
            if (test && test.kernelTests === true && gpu) {
              await runKernelTestsToDebugAPI();
            } else if (test && test.prompt) {
              _origLog(`[AutoTest] Running: "${test.prompt}"`);

              // Enable debug mode for this run's first forward pass
              (globalThis as any).__DEBUG_FORWARD_PASS__ = true;

              // Divergence probe: when the test request asks for it, arm the
              // full per-layer dump on the first forward pass. Dump lands in
              // globalThis.__DEBUG_DUMP_RESULT__ and rides back with the POST.
              if (test.dumpStats === true) {
                (globalThis as any).__DEBUG_DUMP_RESULT__ = undefined;
                (globalThis as any).__DEBUG_DUMP_STATS__ = true;
                // Optional: caller may request additional dumps at specific decode
                // steps. Default schedule covers early/mid/late decode drift.
                const decodeSteps = Array.isArray(test.decodeDumpSteps)
                  ? test.decodeDumpSteps
                  : [1, 10, 50];
                (globalThis as any).__DEBUG_DUMP_DECODE_STEPS__ = decodeSteps;
              }
              // Diagnostic: caller may force a non-default prefill chunk size.
              if (typeof test.prefillChunk === 'number' && test.prefillChunk > 0) {
                (globalThis as any).__DEBUG_PREFILL_CHUNK__ = test.prefillChunk;
              } else {
                (globalThis as any).__DEBUG_PREFILL_CHUNK__ = undefined;
              }
              // Per-tensor audit: list of layer indices to dump every linear
              // projection output for. Generic by design — labels are
              // L${l}-${proj}-out so future model families need only a
              // per-family JSON name-map (no engine changes).
              if (Array.isArray(test.auditLayers) && test.auditLayers.length > 0) {
                (globalThis as any).__DEBUG_AUDIT_LAYERS__ = test.auditLayers;
              } else {
                (globalThis as any).__DEBUG_AUDIT_LAYERS__ = undefined;
              }
              // SSM state-drift probe: read back hidden-state h from each
              // linear_attn layer every N forward() calls and log ‖h‖, max|h|,
              // nonzero%. Cost is ~48 MB readback per sample on Qwen3.5 so use
              // a coarse interval for long decodes (50-200).
              if (typeof test.ssmProbeInterval === 'number' && test.ssmProbeInterval > 0) {
                (globalThis as any).__DEBUG_SSM_STATE__ = test.ssmProbeInterval;
              } else {
                (globalThis as any).__DEBUG_SSM_STATE__ = undefined;
              }
              // Channel-saturation probe: extends SSM-PROBE line with top-K
              // channel indices by per-VD max|h|. A frozen attractor shows the
              // SAME top channel indices at every sample; fluctuating ones = noise.
              if (typeof test.ssmChannels === 'number' && test.ssmChannels > 0) {
                (globalThis as any).__DEBUG_SSM_CHANNELS__ = test.ssmChannels;
              } else {
                (globalThis as any).__DEBUG_SSM_CHANNELS__ = undefined;
              }
              // Attention KV-cache probe: reads ‖K‖, ‖V‖ for the 8 softmax-attn
              // layers at every N forward() calls. Cost grows with pos; use
              // coarse intervals (100-200) on long decodes.
              if (typeof test.attnKvInterval === 'number' && test.attnKvInterval > 0) {
                (globalThis as any).__DEBUG_ATTN_KV__ = test.attnKvInterval;
              } else {
                (globalThis as any).__DEBUG_ATTN_KV__ = undefined;
              }
              // Softpick (rectified softmax) — replaces exp normalization in
              // the 8 softmax attention layers to prevent sink saturation.
              // A/B flag for evaluating whether it fixes long-decode collapse.
              if (test.useSoftpick === true) {
                (globalThis as any).__USE_SOFTPICK__ = true;
              } else {
                (globalThis as any).__USE_SOFTPICK__ = false;
              }
              // Logit-distribution probe — logs top-5 raw logits + stats
              // every N decode steps. Used to tell concentrated collapse
              // (top-1 gap >> rest) from broad uncertainty (flat distribution).
              if (typeof test.logitProbeInterval === 'number' && test.logitProbeInterval > 0) {
                (globalThis as any).__DEBUG_LOGIT_TOPK__ = test.logitProbeInterval;
              } else {
                (globalThis as any).__DEBUG_LOGIT_TOPK__ = undefined;
              }

              // Capture console.log output during inference
              debugLogs = [];
              console.log = (...args: any[]) => {
                const line = args.map(a => typeof a === 'string' ? a : JSON.stringify(a)).join(' ');
                debugLogs.push(line);
                _origLog(...args);
              };

              const messages = [
                { role: 'system', content: 'You are Artifex, a helpful AI assistant.' },
                { role: 'user', content: test.prompt },
              ];
              const t0 = performance.now();
              const handle = session.chat(
                messages,
                {
                  temperature: test.temperature ?? 0,
                  topP: test.topP ?? 0.9,
                  maxNewTokens: test.maxTokens ?? 50,
                  repetitionPenalty: test.repetitionPenalty ?? 1.15,
                  // New sampler params (2026). Pass-through — when undefined,
                  // generate.ts applies its own defaults (minP=0.05, DRY=0.8).
                  minP: test.minP,
                  dryMultiplier: test.dryMultiplier,
                  dryBase: test.dryBase,
                  dryAllowedLength: test.dryAllowedLength,
                  dryRangeLastN: test.dryRangeLastN,
                },
                () => {},
              );
              const result = await handle.result;
              const elapsed = performance.now() - t0;

              // Restore console.log
              console.log = _origLog;

              const dumpResult = (globalThis as any).__DEBUG_DUMP_RESULT__;
              const debugData = {
                prompt: test.prompt,
                output: result.text,
                tokens: result.numTokens,
                tokPerSec: result.tokensPerSecond,
                stopReason: result.stopReason,
                elapsedMs: Math.round(elapsed),
                promptTokens: result.promptTokens,
                tokenIds: result.tokenIds,
                consoleLogs: debugLogs,
                layerDump: dumpResult ?? null,
              };
              (globalThis as any).__DEBUG_DUMP_RESULT__ = undefined;
              (globalThis as any).__DEBUG_DUMP_DECODE_STEPS__ = undefined;
              (globalThis as any).__DEBUG_AUDIT_LAYERS__ = undefined;
              (globalThis as any).__DEBUG_SSM_STATE__ = undefined;
              (globalThis as any).__DEBUG_SSM_CHANNELS__ = undefined;
              (globalThis as any).__DEBUG_ATTN_KV__ = undefined;
              (globalThis as any).__USE_SOFTPICK__ = false;
              (globalThis as any).__DEBUG_LOGIT_TOPK__ = undefined;
              console.log(`[AutoTest] Result: "${result.text}" (${result.numTokens} tok, ${result.tokensPerSecond.toFixed(1)} tok/s)`);
              await fetch('/api/debug', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(debugData),
              });
            }
          } catch { /* server not available */ }
          await new Promise(r => setTimeout(r, 2000));
        }
      })();
}

// ─── GGUF Load Path ──────────────────────────────────────────────────────────
// Loads a GGUF-only model dir (no config.json). K-quant weights stay in native
// llama.cpp block format on GPU (matmul_gguf kernels); token_embd stays in RAM
// and row-gathers on CPU per token.
async function buildGGUFSession(repo: string, ggufFile: string, progressEl: HTMLElement, mmprojFile?: string): Promise<void> {
  const { loadGGUFModel } = await import('./model/gguf-loader');
  const { descriptorFromGGUF, applyRopeFreqFactors } = await import('./model/model-descriptor');
  const { ggufArchitecture } = await import('./model/gguf');
  const { createGGUFLocator } = await import('./model/tensor-locator');
  const { createForwardPassEngine } = await import('./engine/forward-pass');
  const { createTokenizer, applyChatTemplate } = await import('./model/tokenizer');
  const { generate, createKVSession } = await import('./engine/generate');

  addMessage('system', `Loading GGUF: ${repo}/${ggufFile} ...`);
  const vramBudgetBytes = await resolveVRAMBudget();
  const model = await loadGGUFModel(gpu!.device, repo, ggufFile, (p) => {
    progressEl.textContent = p.message;
    if (p.overallProgress !== undefined) {
      setStatus(`Loading ${repo}... ${Math.round(p.overallProgress * 100)}%`);
    }
  }, vramBudgetBytes !== undefined ? { vramBudgetBytes } : undefined);
  currentModel = model as unknown as LoadedModel;

  addMessage('system',
    `Weights loaded: ${model.tensors.size} GPU tensors, ${formatBytes(model.totalGPUBytes)} GPU memory\n` +
    `Load time: ${(model.loadTimeMs / 1000).toFixed(1)}s (GGUF native)`,
    'weights loaded');

  setStatus('Building inference engine...');
  const config = descriptorFromGGUF(model.file);
  // Gemma 4 proportional RoPE: derive per-layer rotatedPairs from the
  // rope_freqs sentinel tensor ([1.0]×N rotating, ~1e30 frozen).
  const ropeFreqsT = model.cpuTensors.get('rope_freqs.weight');
  if (ropeFreqsT) {
    applyRopeFreqFactors(config, new Float32Array(
      ropeFreqsT.data.buffer, ropeFreqsT.data.byteOffset, ropeFreqsT.data.byteLength / 4));
  }
  const loc = createGGUFLocator(model.file.tensors, ggufArchitecture(model.file));

  const requireBuf = (role: TensorRole, l?: number): GPUBuffer => {
    const n = loc.locate(role, l);
    const t = n ? model.tensors.get(n) : undefined;
    if (!t) {
      throw new Error(`GGUF bridge: missing tensor for role "${role}"${l !== undefined ? ` (layer ${l})` : ''} (name: ${n ?? 'unmapped'})`);
    }
    return t.buffer;
  };
  const roleBuf = (role: TensorRole, l?: number): GPUBuffer | undefined => {
    const n = loc.locate(role, l);
    return n ? model.tensors.get(n)?.buffer : undefined;
  };
  /** Quantized tensor → `{slot}_gg` (matmul_gguf); F32/F16 → plain slot (matmul_bt). */
  const assignProj = (lw: any, slot: string, role: TensorRole, l: number) => {
    const n = loc.locate(role, l);
    const t = n ? model.tensors.get(n) : undefined;
    if (!t) throw new Error(`GGUF bridge: missing tensor for role "${role}" (layer ${l})`);
    if (t.isQuantized) lw[`${slot}_gg`] = { data: t.buffer, ggmlType: t.ggmlType };
    else lw[slot] = t.buffer;
  };

  // Global weights. embedTokens/lmHead get a dummy f32 buffer when the real
  // weight rides the embedGG / lmHeadGG path (mirrors the GPTQ-embed pattern).
  const embedCpu = model.cpuTensors.get('token_embd.weight');
  if (!embedCpu) throw new Error('GGUF bridge: token_embd.weight missing from CPU store');
  const finalNorm = requireBuf('finalNorm');
  const lmHeadName = loc.locate('lmHead')!; // falls back to token_embd.weight when tied
  const lmHeadT = model.tensors.get(lmHeadName);
  if (!lmHeadT) throw new Error(`GGUF bridge: lm_head tensor "${lmHeadName}" not on GPU`);
  const global: any = {
    embedTokens: finalNorm,  // dummy — embedGG path is used
    finalNorm,
    lmHead: lmHeadT.isQuantized ? finalNorm : lmHeadT.buffer,
    embedGG: { data: embedCpu.data, ggmlType: embedCpu.ggmlType, rowBytes: embedCpu.rowBytes },
  };
  if (lmHeadT.isQuantized) {
    global.lmHeadGG = { data: lmHeadT.buffer, ggmlType: lmHeadT.ggmlType };
  }
  console.log(`[GGUF] embed: CPU row-gather (${embedCpu.rowBytes} B/row); lm_head: ${lmHeadT.dtype}${config.tieWordEmbeddings ? ' (tied)' : ''}`);

  // Gemma 4 PLE: token table stays CPU-resident (row-gather per token);
  // model_proj + proj_norm live on GPU.
  if (config.perLayerEmbed) {
    const pleCpu = model.cpuTensors.get('per_layer_token_embd.weight');
    if (!pleCpu) throw new Error('GGUF bridge: per_layer_token_embd.weight missing from CPU store');
    global.pleTokenEmbedGG = {
      data: pleCpu.data, parts: pleCpu.parts, rowsPerPart: pleCpu.rowsPerPart,
      ggmlType: pleCpu.ggmlType, rowBytes: pleCpu.rowBytes,
    };
    const mpName = loc.locate('pleModelProj');
    const mpT = mpName ? model.tensors.get(mpName) : undefined;
    if (!mpT) throw new Error(`GGUF bridge: missing tensor for role "pleModelProj" (name: ${mpName ?? 'unmapped'})`);
    if (mpT.isQuantized) global.pleModelProj_gg = { data: mpT.buffer, ggmlType: mpT.ggmlType };
    else global.pleModelProj = mpT.buffer;
    global.pleProjNorm = requireBuf('pleProjNorm');
    console.log(`[GGUF] PLE: token table CPU (${pleCpu.rowBytes} B/row), model_proj ${mpT.dtype}`);
  }

  // MoE (Phase C): experts live in CPU RAM across a wasm worker fleet; the
  // shared expert rides the dense FFN slots (ffnDim === expertFFNDim).
  const moeSpec = config.layers.find((d) => d.moe)?.moe;
  let moe: { backend: import('./engine/moe-cpu').MoEBackend; sharedGateVecs: Float32Array[] } | undefined;
  if (moeSpec) {
    const { createMoECPUBackend } = await import('./engine/moe-cpu');
    setStatus('Loading MoE experts into CPU workers...');
    // ?moeWorkers=8|16|32 overrides the hardwareConcurrency-based default
    // (must divide numExperts) — for A/B perf testing across machines.
    const moeWorkersParam = new URLSearchParams(window.location.search).get('moeWorkers');
    const moeWorkersOverride = moeWorkersParam ? parseInt(moeWorkersParam, 10) : undefined;
    const backend = await createMoECPUBackend({
      url: model.url,
      expertTensors: model.expertTensors,
      numLayers: config.numLayers,
      numExperts: moeSpec.numExperts,
      hiddenSize: config.hiddenSize,
      ffnDim: moeSpec.expertFFNDim,
      numWorkers: moeWorkersOverride,
      onProgress: (message, frac) => {
        progressEl.textContent = message;
        setStatus(`Loading experts... ${Math.round(frac * 100)}%`);
      },
    });
    // Warm-up: Windows trims/compresses the 22 GB of expert pages during the
    // long load, leaving them cold (~0.15 GB/s on first touch → decode ran
    // 2-3 tok/s instead of warm speed). Warm in WAVES of 8 workers: all-NW
    // passes thrash the pagefile when commit is near RAM capacity (observed
    // 0.01 GB/s first pass × 4 retries at 32 workers, minutes pinned at max
    // RAM). Each wave gets full disk bandwidth, finishes fast, and its pages
    // stay resident while the next wave runs.
    // Two rounds: warming later waves can re-trim earlier waves' pages
    // (observed: first decode after a clean round 1 still ran ms/expert=0.36
    // vs 0.13 warm). Round 2 re-sweeps — resident waves pass in ~0.2 s, only
    // the re-trimmed delta gets re-faulted.
    const nw = backend.numWorkers;
    const waveSize = Math.min(8, nw);
    let warmMinGbps = Infinity;
    const warmT0 = performance.now();
    for (let round = 1; round <= 2; round++) {
      warmMinGbps = Infinity;
      for (let w0 = 0; w0 < nw; w0 += waveSize) {
        const count = Math.min(waveSize, nw - w0);
        const label = `Warming expert RAM (round ${round}, workers ${w0 + 1}-${w0 + count} of ${nw}`;
        setStatus(`${label})...`);
        let warm = await backend.touchTest!(w0, count);
        for (let pass = 2; pass <= 4 && Math.min(...warm.gbps) < 1.5; pass++) {
          setStatus(`${label}, pass ${pass})...`);
          warm = await backend.touchTest!(w0, count);
        }
        warmMinGbps = Math.min(warmMinGbps, Math.min(...warm.gbps));
      }
    }
    console.log(
      `[MoE] warm-up done in ${((performance.now() - warmT0) / 1000).toFixed(1)}s `
      + `(slowest worker ${warmMinGbps.toFixed(2)} GB/s${warmMinGbps < 1.5 ? ' — STILL COLD, expect a slow first generation' : ''})`,
    );
    const sharedGateVecs: Float32Array[] = [];
    for (let l = 0; l < config.numLayers; l++) {
      if (!config.layers[l].moe) { sharedGateVecs.push(new Float32Array(0)); continue; }
      const t = model.cpuTensors.get(`blk.${l}.ffn_gate_inp_shexp.weight`);
      if (!t) throw new Error(`GGUF bridge: blk.${l}.ffn_gate_inp_shexp.weight missing from CPU store`);
      sharedGateVecs.push(new Float32Array(t.data.buffer, t.data.byteOffset, config.hiddenSize));
    }
    moe = { backend, sharedGateVecs };
    addMessage('system',
      `MoE experts loaded: ${moeSpec.numExperts} experts × ${config.layers.filter(d => d.moe).length} layers `
      + `(${formatBytes(model.expertBytes)}) across ${backend.numWorkers} CPU workers`,
      'experts loaded');
  }

  const layers: any[] = [];
  for (let l = 0; l < config.numLayers; l++) {
    const lw: any = {
      inputNorm: requireBuf('inputNorm', l),
      postAttnNorm: requireBuf('postAttnNorm', l),
      // Gemma 4 sandwich norms (undefined on other archs)
      attnPostNorm: roleBuf('attnPostNorm', l),
      ffnPostNorm: roleBuf('ffnPostNorm', l),
    };
    if (config.layers[l].kind === 'linear_attention') {
      assignProj(lw, 'linearInProjQKV', 'linInProjQKV', l);
      assignProj(lw, 'linearInProjA', 'linInProjA', l);
      assignProj(lw, 'linearInProjB', 'linInProjB', l);
      assignProj(lw, 'linearInProjZ', 'linInProjZ', l);
      assignProj(lw, 'linearOutProj', 'linOutProj', l);
      lw.linearALog = requireBuf('linALog', l);
      lw.linearConv1dWeight = requireBuf('linConv1dWeight', l);
      lw.linearDtBias = requireBuf('linDtBias', l);
      lw.linearNormWeight = requireBuf('linNormWeight', l);
    } else {
      assignProj(lw, 'qProj', 'qProj', l);
      // KV-sharing layers (Gemma 4 ≥24) read another layer's cache; their
      // k/v weights are dead in the GGUF and the loader never uploaded them.
      if (config.layers[l].kvSourceLayer === undefined) {
        assignProj(lw, 'kProj', 'kProj', l);
        assignProj(lw, 'vProj', 'vProj', l);
      }
      assignProj(lw, 'oProj', 'oProj', l);
      lw.qNorm = roleBuf('qNorm', l);
      lw.kNorm = roleBuf('kNorm', l);
      if (config.attentionBias) {
        lw.qBias = roleBuf('qBias', l);
        lw.kBias = roleBuf('kBias', l);
        lw.vBias = roleBuf('vBias', l);
        lw.oBias = roleBuf('oBias', l);
      }
    }
    if (config.layers[l].moe) {
      // Shared expert in the dense FFN slots; router marks the layer as MoE.
      assignProj(lw, 'gateProj', 'moeSharedGateProj', l);
      assignProj(lw, 'upProj', 'moeSharedUpProj', l);
      assignProj(lw, 'downProj', 'moeSharedDownProj', l);
      lw.moeRouter = requireBuf('moeRouter', l);
    } else {
      assignProj(lw, 'gateProj', 'gateProj', l);
      assignProj(lw, 'upProj', 'upProj', l);
      assignProj(lw, 'downProj', 'downProj', l);
    }
    if (config.perLayerEmbed) {
      assignProj(lw, 'pleInpGate', 'pleInpGate', l);
      assignProj(lw, 'pleProj', 'pleProj', l);
      lw.plePostNorm = requireBuf('plePostNorm', l);
      lw.layerOutScale = roleBuf('layerOutScale', l);
    }
    layers.push(lw);
  }
  if (config.isHybrid) {
    const linCount = config.layers.filter(d => d.kind === 'linear_attention').length;
    console.log(`[GGUF] Hybrid model: ${linCount} linear + ${config.numLayers - linCount} full attention layers`);
  }

  // MTP / next-token-prediction head. The loader parks blk.{block_count}.*
  // in model.mtpTensors (not the trunk `tensors` map) only when ?spec=1 /
  // ?mtpHead=1. The head is structurally a full_attention trunk layer, so
  // reuse the same role→GGUF-name locator at layer index = numLayers (the
  // first MTP block) plus the four nextn.* tensors. Shared embed/lm_head.
  let mtp: any;
  if (model.mtpTensors.size > 0) {
    const ML = config.numLayers; // = block_count - nextn_predict_layers
    const mtpAssign = (lw: any, slot: string, role: TensorRole) => {
      const n = loc.locate(role, ML);
      const t = n ? model.mtpTensors.get(n) : undefined;
      if (!t) throw new Error(`MTP head: missing tensor for role "${role}" (blk.${ML}, name: ${n ?? 'unmapped'})`);
      if (t.isQuantized) lw[`${slot}_gg`] = { data: t.buffer, ggmlType: t.ggmlType };
      else lw[slot] = t.buffer;
    };
    const mtpNorm = (role: TensorRole): GPUBuffer => {
      const n = loc.locate(role, ML);
      const t = n ? model.mtpTensors.get(n) : undefined;
      if (!t) throw new Error(`MTP head: missing norm "${role}" (blk.${ML})`);
      return t.buffer;
    };
    const mtpRoleBuf = (role: TensorRole): GPUBuffer | undefined => {
      const n = loc.locate(role, ML);
      return n ? model.mtpTensors.get(n)?.buffer : undefined;
    };
    const nextnTensor = (suffix: string) => {
      const t = model.mtpTensors.get(`blk.${ML}.nextn.${suffix}`);
      if (!t) throw new Error(`MTP head: missing blk.${ML}.nextn.${suffix}`);
      return t;
    };
    const mlw: any = {
      inputNorm: mtpNorm('inputNorm'),       // blk.L.attn_norm
      postAttnNorm: mtpNorm('postAttnNorm'), // blk.L.ffn_norm (pre-FFN norm)
    };
    mtpAssign(mlw, 'qProj', 'qProj');
    mtpAssign(mlw, 'kProj', 'kProj');
    mtpAssign(mlw, 'vProj', 'vProj');
    mtpAssign(mlw, 'oProj', 'oProj');
    mlw.qNorm = mtpRoleBuf('qNorm');
    mlw.kNorm = mtpRoleBuf('kNorm');
    mtpAssign(mlw, 'gateProj', 'gateProj');
    mtpAssign(mlw, 'upProj', 'upProj');
    mtpAssign(mlw, 'downProj', 'downProj');

    const ehT = nextnTensor('eh_proj.weight');
    mtp = {
      layer: mlw,
      ehProj: ehT.isQuantized ? { data: ehT.buffer, ggmlType: ehT.ggmlType } : ehT.buffer,
      enorm: nextnTensor('enorm.weight').buffer,
      hnorm: nextnTensor('hnorm.weight').buffer,
      sharedHeadNorm: nextnTensor('shared_head_norm.weight').buffer,
    };
    console.log(`[GGUF] MTP head loaded (blk.${ML}.*): drafter available for spec decode`);
  }

  const engine = createForwardPassEngine(gpu!.device, config, { global, layers, moe, mtp });
  const tokenizer = await createTokenizer({ modelId: repo });
  // Guard: a tokenizer from the wrong model family silently produces garbage
  // (e.g. Qwen3-8B fallback vocab 151936 vs Qwen3.6 vocab 248320).
  if (tokenizer.vocabSize && Math.abs(tokenizer.vocabSize - config.vocabSize) > 1024) {
    throw new Error(
      `Tokenizer/model vocab mismatch: tokenizer ${tokenizer.modelId} has ${tokenizer.vocabSize} tokens ` +
      `but model expects ${config.vocabSize}. Wrong tokenizer fallback?`);
  }
  // Vision: a sibling mmproj GGUF carries the tower (Qwen3-VL-style clip).
  // Parse it while hf-hub still points at the local cache.
  let visionConfigured = false;
  if (mmprojFile) {
    try {
      const { visionDescriptorFromGGUF } = await import('./vision/vision-descriptor');
      const { parseGGUF } = await import('./model/gguf');
      const mmUrl = resolveFileUrl(repo, mmprojFile);
      const mmFile = await parseGGUF((s, e) => fetchRange(mmUrl, s, e));
      const vdesc = visionDescriptorFromGGUF(mmFile);
      if (vdesc) {
        // The mmproj knows nothing about the text vocab — resolve the
        // placeholder ids through the tokenizer's special tokens.
        const padIds = tokenizer.encode('<|image_pad|>');
        const startIds = tokenizer.encode('<|vision_start|>');
        if (padIds.length === 1 && startIds.length === 1) {
          vdesc.placeholder.imageTokenId = padIds[0];
          activeVisionSource = { kind: 'gguf', file: mmprojFile };
          setVisionDesc(vdesc);
          visionConfigured = true;
        } else {
          console.warn('[Vision] tokenizer lacks <|image_pad|>/<|vision_start|> special tokens — vision disabled');
        }
      }
    } catch (err) {
      console.warn('[Vision] mmproj parse failed:', err);
    }
  } else {
    // No sibling mmproj — Ollama-packed multimodal blobs (Gemma 4) carry the
    // tower inline in the text GGUF.
    try {
      const { visionDescriptorFromGGUF } = await import('./vision/vision-descriptor');
      const vdesc = visionDescriptorFromGGUF(model.file);
      if (vdesc) {
        const padIds = tokenizer.encode(vdesc.placeholder.padText ?? '<|image_pad|>');
        if (padIds.length === 1) {
          vdesc.placeholder.imageTokenId = padIds[0];
          activeVisionSource = { kind: 'gguf', file: ggufFile };
          setVisionDesc(vdesc);
          visionConfigured = true;
        } else {
          console.warn('[Vision] tokenizer lacks the image placeholder special token — vision disabled');
        }
      }
    } catch (err) {
      console.warn('[Vision] inline vision probe failed:', err);
    }
  }
  resetToRemote();
  if (!visionConfigured) setVisionDescQuiet(null);

  const kvSession = createKVSession(Math.min(config.maxPositionEmbeddings || 8192, 8192, MAX_ATTN_SEQ_LEN));
  const resetKV = () => {
    if (kvSession.kvCache) engine.destroyKVCache(kvSession.kvCache);
    kvSession.kvCache = null;
    kvSession.cachedTokenIds = [];
  };
  session = {
    run: (prompt, sampling, onToken) => generate(gpu!.device, engine, tokenizer, prompt, sampling, onToken),
    chat: (messages: Array<{role: string; content: string}>, sampling: any, onToken: any, opts?: { enableThinking?: boolean }) => {
      const tokenIds = applyChatTemplate(tokenizer, messages, opts);
      return generate(gpu!.device, engine, tokenizer, tokenIds, sampling, onToken, { kvSession });
    },
    chatMM: (
      messages: Array<{role: string; content: string}>,
      encoded: VisionEncodeResult[],
      sampling: any, onToken: any, opts?: { enableThinking?: boolean },
    ) => {
      const tokenIds = applyChatTemplate(tokenizer, messages, opts);
      const padId = activeVisionDesc!.placeholder.imageTokenId;
      const spans: Array<{ start: number; count: number }> = [];
      let i = 0;
      while (i < tokenIds.length) {
        if (tokenIds[i] === padId) {
          let j = i;
          while (j < tokenIds.length && tokenIds[j] === padId) j++;
          spans.push({ start: i, count: j - i });
          i = j;
        } else i++;
      }
      if (spans.length !== encoded.length) {
        throw new Error(`Image span mismatch: ${spans.length} pad runs in prompt vs ${encoded.length} encoded images`);
      }
      const images = spans.map((s, k) => {
        if (s.count !== encoded[k].numTokens) {
          throw new Error(`Image ${k}: ${s.count} pad tokens vs ${encoded[k].numTokens} embedding rows`);
        }
        return {
          start: s.start, count: s.count,
          embeddings: encoded[k].embeddings,
          deepstack: encoded[k].deepstack,
          bidirectional: activeVisionDesc!.placeholder.bidirectional,
        };
      });
      return generate(gpu!.device, engine, tokenizer, { tokenIds, images }, sampling, onToken, { kvSession });
    },
    kvSession,
    resetKV,
    config,
    tokenizer,
    gpu: gpu!,
    vramEstimate: estimateVRAM(config),
    destroy: () => { resetVision(); resetKV(); moe?.backend.destroy(); unloadModel(currentModel!); session = null; },
  } as InferenceSession;

  // FLUX.2 TE debug/parity hook: lets Playwright harnesses drive the
  // hidden-tap prompt embedder against a loaded klein-TE GGUF.
  (globalThis as any).__flux2Embed = async (prompt: string) => {
    const { embedFlux2Prompt } = await import('./diffusion/text-embedder');
    return embedFlux2Prompt(engine, tokenizer, prompt);
  };

  promptEl.disabled = false;
  sendBtn.disabled = false;

  if ((config as any).experimentalArch) {
    addMessage('system',
      `⚠️ EXPERIMENTAL architecture (${(model.file.kv.get('general.architecture') as string) ?? '?'}): `
      + `recognized and attempted via the standard-transformer path, but not yet verified end-to-end. `
      + `It may work perfectly, error out, or produce off output — judge the results.`,
      'experimental arch');
  }
  addMessage('system',
    `Inference engine ready! (GGUF native)\n` +
    `Model: ${config.modelType} — ${config.numLayers} layers, ${config.numAttentionHeads} heads, d=${config.hiddenSize}\n` +
    `Vocab: ${config.vocabSize} | RoPE θ=${config.ropeTheta}\n` +
    `Type a message to chat!`,
    'engine ready');
  setStatus(`Ready: ${repo}`);
  markModelReady(repo);
  startAutoTestPoller();
}

// ─── FLUX.2 image generation (Phase 4) ───────────────────────────────────────

/** Enter image-gen mode: nothing heavyweight loads up front — each Send runs
 *  the sequential TE → DiT → VAE pipeline with per-stage VRAM lifecycle. */
function buildFlux2ImageSession(repo: string): void {
  session = null;
  imageGenRepo = repo;
  promptEl.disabled = false;
  sendBtn.disabled = false;
  addMessage('system',
    `FLUX.2 klein image generation ready!\n` +
    `Type a prompt to generate a 512×512 image (4-step distilled flow, no CFG).\n` +
    `Flags: start with "/256 " for a smaller/faster image or "/1024 " for full ` +
    `resolution (slow — several minutes of denoising); "/seed N " pins the ` +
    `noise seed (otherwise random — the seed is shown under each image).\n` +
    `v1 loads each stage on demand (TE → DiT → VAE, ~12 GB total streamed), so ` +
    `expect a few minutes per image; rerolling the same prompt skips the text encoder.`,
    'image gen ready');
  setStatus(`Ready: ${repo} (image gen)`);
  updateFooter({ model: 'FLUX.2-klein (image gen)' });
  markModelReady(repo);
}

/** One image-gen turn: parse inline flags, run the pipeline with live stage
 *  progress in an assistant bubble, then swap in the PNG (markdown data-URL —
 *  persists through history/autosave/restore like any other message). */
async function runImageGeneration(text: string, userDiv: HTMLElement): Promise<void> {
  if (imageGenBusy || !gpu) return;

  let px = 512;
  let seed = (Math.random() * 0x7fffffff) | 0;
  let prompt = text.trim();
  const sizeM = prompt.match(/^\/(256|512|1024)\s+/);
  if (sizeM) { px = parseInt(sizeM[1]); prompt = prompt.slice(sizeM[0].length).trim(); }
  const seedM = prompt.match(/^\/seed\s+(\d+)\s+/);
  if (seedM) { seed = parseInt(seedM[1]); prompt = prompt.slice(seedM[0].length).trim(); }
  if (!prompt) { addMessage('system', 'Empty prompt.'); return; }

  const userMsg: ChatMessage = { role: 'user', content: text };
  linkMessage(userDiv, userMsg);
  attachActions(userDiv, 'user');

  imageGenBusy = true;
  sendBtn.disabled = true;
  setStatus('Generating image...');
  const progressDiv = addMessage('assistant', '');
  const t0 = performance.now();
  try {
    const { generateFlux2Image } = await import('./diffusion/flux2-image');
    const res = await generateFlux2Image(gpu.device, {
      prompt, px, seed,
      onProgress: (stage, detail) => {
        progressDiv.textContent = `[${stage}] ${detail}`;
        setStatus(`Image gen: ${detail}`);
        scrollMessages();
      },
    });

    const canvas = document.createElement('canvas');
    canvas.width = res.width;
    canvas.height = res.height;
    canvas.getContext('2d')!.putImageData(new ImageData(res.rgba, res.width, res.height), 0, 0);
    const dataUrl = canvas.toDataURL('image/png');
    // Brackets/parens would break the markdown image syntax in the alt text.
    const md = `![${prompt.slice(0, 100).replace(/[[\]()]/g, ' ')}](${dataUrl})`;
    renderAssistant(progressDiv, md, false);

    const totalS = (performance.now() - t0) / 1000;
    const t = res.timings;
    attachMeta(progressDiv,
      `${px}×${px} | seed ${seed} | ${totalS.toFixed(0)}s`
      + ` (TE ${(t.teMs / 1000).toFixed(0)}s · DiT ${(t.ditMs / 1000).toFixed(0)}s · VAE ${(t.vaeMs / 1000).toFixed(0)}s)`);
    const asstMsg: ChatMessage = { role: 'assistant', content: md };
    linkMessage(progressDiv, asstMsg);
    chatHistory.push(userMsg, asstMsg);
    autoSave(chatHistory, { model: imageGenRepo ?? 'flux2', backend: 'webgpu' });
    setStatus(`Image generated in ${totalS.toFixed(0)}s (seed ${seed})`);
    scrollMessages(true);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    progressDiv.textContent = `Image generation failed: ${msg}`;
    setStatus('Image gen failed');
    reportError('image-gen', err);
  } finally {
    imageGenBusy = false;
    sendBtn.disabled = false;
  }
}

// ─── Load Model ──────────────────────────────────────────────────────────────

loadBtn.addEventListener('click', async () => {
  const repo = ($('model-repo') as HTMLInputElement).value.trim();
  if (!repo) return;
  if (activeHandle) {
    addMessage('system', 'Stop the current generation before loading a model.');
    return;
  }

  // Set HF auth token (for gated models like Qwen3.5)
  const tokenInput = $('hf-token') as HTMLInputElement;
  const hfToken = tokenInput.value.trim();
  if (hfToken) {
    setAuthToken(hfToken);
    localStorage.setItem('hf-token', hfToken);
  }

  if (!gpu) {
    addMessage('system', 'No GPU — initialize WebGPU first.');
    return;
  }

  // Unload previous model if loaded
  if (currentModel) {
    resetVision();
    unloadModel(currentModel);
    currentModel = null;
  }
  imageGenRepo = null;

  const progressEl = $('load-progress');
  loadBtn.disabled = true;
  setStatus(`Loading ${repo}...`);

  // Diagnostic: VRAM upload audit. Enable via ?vramAudit=prefix1,prefix2 in URL
  // or localStorage.setItem('vramAuditPrefixes', 'prefix1,prefix2'). After each
  // tensor whose name starts with any prefix is uploaded, the GPU buffer is
  // read back and byte-compared to the CPU source. Logs OK/MISMATCH per tensor.
  const urlParams = new URLSearchParams(window.location.search);
  const vramAuditFromUrl = urlParams.get('vramAudit');
  const vramAuditFromStorage = localStorage.getItem('vramAuditPrefixes');
  const vramAuditRaw = vramAuditFromUrl || vramAuditFromStorage;
  if (vramAuditRaw) {
    (globalThis as any).__DEBUG_VRAM_AUDIT_PREFIXES__ =
      vramAuditRaw.split(',').map(s => s.trim()).filter(s => s.length > 0);
    console.log(`[VRAM-AUDIT] enabled for prefixes:`, (globalThis as any).__DEBUG_VRAM_AUDIT_PREFIXES__);
  } else {
    (globalThis as any).__DEBUG_VRAM_AUDIT_PREFIXES__ = [];
  }

  try {
    // Check if model is available in local HF cache (50-100x faster than CDN)
    let usingLocalCache = false;
    let localFiles: string[] | null = null;
    try {
      const cacheResp = await fetch('/api/hf-cache/models');
      if (cacheResp.ok) {
        const cached = await cacheResp.json() as Array<{ repo: string; files: string[] }>;
        const localModel = cached.find(m => m.repo === repo);
        if (localModel) {
          localFiles = localModel.files;
          useLocalCache();
          usingLocalCache = true;
          addMessage('system', `Loading ${repo} from local HF cache (fast)...`);
        }
      }
    } catch { /* dev server not running, use CDN */ }

    // Diffusers image-gen pipeline (FLUX.2 klein): the pipeline dir carries
    // model_index.json instead of top-level weight files, so it is invisible
    // to the models listing above — probe the file directly.
    if (!localFiles) {
      try {
        const probe = await fetch(`/api/hf-cache/${repo}/resolve/main/model_index.json`);
        if (probe.ok) {
          const mi = await probe.json() as { _class_name?: string };
          if ((mi._class_name ?? '').includes('Flux2')) {
            useLocalCache();
            buildFlux2ImageSession(repo);
            return;
          }
          addMessage('system',
            `${repo} is a diffusers pipeline (${mi._class_name ?? 'unknown'}) — `
            + `only FLUX.2 klein image generation is supported.`);
          return;
        }
      } catch { /* no dev server — fall through to the HF flow */ }
    }

    if (!usingLocalCache) {
      addMessage('system', `Connecting to HuggingFace: ${repo}...`);
    }

    // GGUF-only model dir (no config.json): native GGUF load path
    const ggufFiles = (localFiles ?? []).filter(f => f.endsWith('.gguf') && !f.toLowerCase().includes('mmproj'));
    const mmprojFile = (localFiles ?? []).find(f => f.toLowerCase().includes('mmproj') && f.toLowerCase().endsWith('.gguf'));
    if (ggufFiles.length > 0 && !(localFiles ?? []).includes('config.json')) {
      await buildGGUFSession(repo, ggufFiles[0], progressEl, mmprojFile);
      updateFooter({ model: repo.split('/').pop() || repo });
      return;
    }

    // Preview first (just headers, fast)

    const preview = await previewModel(repo, (p) => {
      progressEl.textContent = p.message;
    });

    addMessage('system',
      `Model: ${repo}\n` +
      `Type: ${preview.config.model_type}\n` +
      `Layers: ${preview.config.num_hidden_layers} | Hidden: ${preview.config.hidden_size}\n` +
      `Tensors: ${preview.tensorCount} | Size: ${formatBytes(preview.totalBytes)}\n` +
      `Dtypes: ${preview.dtypes.join(', ')}\n` +
      `GPU needed: ~${formatBytes(preview.tensorCount * 4 * (preview.totalBytes / preview.tensorCount / 2))} (float32)`,
      'model preview'
    );

    // Decide whether to keep BF16 weights native on GPU (halves VRAM)
    // Enable when: (a) unquantized model too big for f32, OR (b) mixed-precision GPTQ with BF16 tensors
    const isQuantized = preview.dtypes.includes('I32'); // GPTQ models have I32 packed weights
    const hasBF16 = preview.dtypes.includes('BF16') || preview.dtypes.includes('F16');
    const f32Estimate = preview.totalBytes * 2;
    const VRAM_THRESHOLD = 6 * 1024 * 1024 * 1024; // 6 GB
    // Mixed-precision: GPTQ model that ALSO has BF16 tensors (e.g., our custom quantization)
    const isMixedPrecision = isQuantized && hasBF16;
    const keepBF16 = isMixedPrecision || (!isQuantized && f32Estimate > VRAM_THRESHOLD);
    console.log(`[Engine] keepBF16=${keepBF16}: isQuantized=${isQuantized}, hasBF16=${hasBF16}, isMixedPrecision=${isMixedPrecision}`);
    if (keepBF16) {
      console.log(`[Engine] Keeping BF16 weights native — halves VRAM usage`);
    }

    // SSM dequant strategy: on-the-fly GPU dequant at inference time.
    // Weights stay as Q4 on GPU (~12 GB), dequanted to a temp BF16 buffer
    // per-projection during the forward pass. No CPU-side dequant needed.
    const dequantToBF16 = new Set<string>();
    const maxDequantOverhead = 0;
    if (isQuantized) {
      const previewConfig = parseModelConfig(preview.config);
      if (previewConfig.isHybrid && previewConfig.layerTypes) {
        const ssmLayerCount = previewConfig.layerTypes.filter((t: string) => t === 'linear_attention').length;
        console.log(`[Engine] Hybrid model: ${ssmLayerCount} SSM layers will use on-the-fly GPU dequant (Q4 → BF16) during inference`);
      }
    }
    currentModel = await loadModel(gpu.device, repo, (p) => {
      progressEl.textContent = p.message;
      if (p.overallProgress !== undefined) {
        const pct = Math.round(p.overallProgress * 100);
        setStatus(`Loading ${repo}... ${pct}%`);
      }
    }, keepBF16, dequantToBF16, maxDequantOverhead);

    // Note: don't resetToRemote() here — tokenizer still needs local cache

    addMessage('system',
      `Weights loaded: ${currentModel.tensorCount} tensors, ${formatBytes(currentModel.totalGPUBytes)} GPU memory\n` +
      `Load time: ${(currentModel.loadTimeMs / 1000).toFixed(1)}s` +
      (usingLocalCache ? ' (from local cache)' : ''),
      'weights loaded'
    );

    // Build inference session from loaded weights
    setStatus('Building inference engine...');
    try {
      const config = descriptorFromHFConfig(currentModel.config);

      // Import bridgeWeights and engine builder
      const { createForwardPassEngine } = await import('./engine/forward-pass');
      const { createTokenizer } = await import('./model/tokenizer');
      const { createHFLocator } = await import('./model/tensor-locator');
      const { generate, createKVSession } = await import('./engine/generate');

      // Bridge weight tensors by canonical role (locator auto-detects prefix)
      const loc = createHFLocator(config.modelType, currentModel!.tensors);
      const embedName = loc.locate('embedTokens')!;
      const finalNormName = loc.locate('finalNorm')!;
      const lmHeadName = loc.locate('lmHead')!;
      const getTensor = (name: string) => {
        const t = currentModel!.tensors.get(name);
        if (!t) throw new Error(`Missing tensor: ${name}`);
        return t.buffer;
      };

      // Auto-detect if embedding is stored as packed 16-bit (BF16/F16)
      // This happens when: (a) keepBF16=true and tensor is >1MB, or (b) f32 would exceed 2GB buffer limit
      const embedTensor = currentModel!.tensors.get(embedName);
      const embedIsF16 = embedTensor ? (embedTensor.dtype === 'BF16' || embedTensor.dtype === 'F16') : false;
      console.log(
        `[Engine] embed lookup: embedTokens="${embedName}", `
        + `found=${!!embedTensor}, dtype=${embedTensor?.dtype}, `
        + `byteLength=${embedTensor?.byteLength}, shape=${embedTensor?.shape}, `
        + `embedIsF16=${embedIsF16}`
      );
      if (embedIsF16) {
        console.log(`[Engine] Embedding is ${embedTensor!.dtype}, using packed-16 embed shader`);
      }

      // Embedding may be f32, BF16, or GPTQ INT4
      const embedBuf = currentModel!.tensors.get(embedName)?.buffer ?? null;
      const lmHeadBuf = config.tieWordEmbeddings
        ? embedBuf
        : (currentModel!.tensors.get(lmHeadName)?.buffer ?? null);

      // Use a dummy buffer for embedTokens/lmHead when they're GPTQ (the Q4 path is used instead)
      const dummyBuf = embedBuf ?? lmHeadBuf ?? getTensor(finalNormName);

      const global: any = {
        embedTokens: embedBuf ?? dummyBuf,
        embedIsF16: embedBuf ? embedIsF16 : false,
        finalNorm: getTensor(finalNormName),
        lmHead: lmHeadBuf ?? dummyBuf,
        lmHeadIsBF16: lmHeadBuf ? (() => {
          const tensorName = config.tieWordEmbeddings ? embedName : lmHeadName;
          const t = currentModel!.tensors.get(tensorName);
          const isBF16 = t ? (t.dtype === 'BF16' || t.dtype === 'F16') : false;
          if (isBF16) console.log(`[Engine] LM head is ${t!.dtype}${config.tieWordEmbeddings ? ' (tied)' : ''}, using BF16 matmul`);
          return isBF16;
        })() : false,
      };

      // Helper for optional tensors (bias terms)
      const tryGetTensor = (name: string): GPUBuffer | undefined => {
        const t = currentModel!.tensors.get(name);
        return t?.buffer;
      };
      // Helper for optional per-layer roles (returns undefined if role unmapped)
      const tryGetRole = (role: TensorRole, l: number): GPUBuffer | undefined => {
        const n = loc.locate(role, l);
        return n ? tryGetTensor(n) : undefined;
      };

      // Helper to load GPTQ quad (qweight + scales + qzeros + g_idx) for a projection
      // If g_idx is missing (non-actorder model), generates trivial g_idx[k] = k / gs
      const trivialGIdxCache = new Map<number, GPUBuffer>();
      const getOrCreateTrivialGIdx = (K: number): GPUBuffer => {
        if (trivialGIdxCache.has(K)) return trivialGIdxCache.get(K)!;
        const gs = config.quantGroupSize || 128;
        const data = new Uint32Array(K);
        for (let k = 0; k < K; k++) data[k] = Math.floor(k / gs);
        const buf = gpu!.device.createBuffer({
          size: data.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
          label: `trivial-g_idx-K${K}`,
          mappedAtCreation: true,
        });
        new Uint32Array(buf.getMappedRange()).set(data);
        buf.unmap();
        trivialGIdxCache.set(K, buf);
        return buf;
      };
      const tryGetQ4 = (weightName: string) => {
        const base = weightName.replace('.weight', '');
        const qw = tryGetTensor(`${base}.qweight`);
        const sc = tryGetTensor(`${base}.scales`);
        const qz = tryGetTensor(`${base}.qzeros`);
        if (qw && sc && qz) {
          // Try to load actorder g_idx, fall back to trivial.
          // hasActOrder: true only when the file ships a g_idx tensor AND the
          // loader verified its values are non-trivial (actual column reorder).
          // If the tensor is present but values are floor(k / group_size) —
          // which many GPTQ exports ship for tool compatibility even without
          // desc_act=true — we treat it as non-actorder so the GEMV fast-path
          // kicks in.
          const gIdxTensor = currentModel!.tensors.get(`${base}.g_idx`);
          let gIdx = gIdxTensor?.buffer;
          const hasActOrder = gIdxTensor !== undefined && gIdxTensor.isTrivialGIdx !== true;
          if (!gIdx) {
            // Derive K from qweight: shape is [K/8, N], stored as int32
            const qwTensor = currentModel!.tensors.get(`${base}.qweight`);
            const K = qwTensor ? qwTensor.shape[0] * 8 : 0;
            if (K > 0) {
              console.log(`[Q4] ${base}: no g_idx tensor, generating trivial (K=${K})`);
              gIdx = getOrCreateTrivialGIdx(K);
            } else {
              console.warn(`[Q4] ${base}: no g_idx and cannot derive K from qweight!`);
            }
          }
          if (!gIdx) {
            console.error(`[Q4] ${base}: FAILED to get g_idx — q4 object will be skipped!`);
            return undefined;
          }
          return { qweight: qw, scales: sc, qzeros: qz, g_idx: gIdx, hasActOrder };
        }
        return undefined;
      };

      // Helper to load INT8 quad (qweight_q8 + scales_q8 + qzeros_q8 + g_idx_q8)
      // Helper to load INT8 quad (qweight_q8 + scales_q8 + qzeros_q8 + g_idx_q8)
      const tryGetQ8 = (weightName: string) => {
        const base = weightName.replace('.weight', '');
        const qw = tryGetTensor(`${base}.qweight_q8`);
        const sc = tryGetTensor(`${base}.scales_q8`);
        const qz = tryGetTensor(`${base}.qzeros_q8`);
        if (qw && sc && qz) {
          let gIdx = tryGetTensor(`${base}.g_idx_q8`);
          if (!gIdx) {
            const qwTensor = currentModel!.tensors.get(`${base}.qweight_q8`);
            const K = qwTensor ? qwTensor.shape[0] * 4 : 0;
            if (K > 0) gIdx = getOrCreateTrivialGIdx(K);
          }
          if (!gIdx) return undefined;
          return { qweight: qw, scales: sc, qzeros: qz, g_idx: gIdx };
        }
        return undefined;
      };

      // Helper: detect split BF16 buffers for a weight (embed_tokens or lm_head)
      const detectSplit = (baseName: string) => {
        const meta = currentModel!.tensors.get(`${baseName}.split_meta`);
        if (!meta) return undefined;
        const numParts = meta.shape[0];
        const splitPoints = meta.shape.slice(1);
        const buffers: GPUBuffer[] = [];
        for (let p = 0; p < numParts; p++) {
          const partTensor = currentModel!.tensors.get(`${baseName}.split${p}`);
          if (!partTensor?.buffer) {
            console.error(`[Engine] Missing split buffer: ${baseName}.split${p}`);
            return undefined;
          }
          buffers.push(partTensor.buffer);
        }
        return { buffers, splitPoints };
      };

      // Check if embedding is stored CPU-side (oversized BF16, saves ~2.4 GB VRAM)
      const embedBaseName = embedName.replace('.weight', '');
      const cpuEmbedTensor = currentModel!.tensors.get(`${embedBaseName}.cpu_embed`) as any;
      if (cpuEmbedTensor?.cpuEmbedData) {
        const { parts, splitPoint, isBF16 } = cpuEmbedTensor.cpuEmbedData;
        global.embedCPU = { parts, splitPoint, hiddenSize: config.hiddenSize, isBF16 };
        global.embedIsF16 = true;
        console.log(`[Engine] Embedding is CPU-side BF16: ${parts.length} parts, splitPoint=${splitPoint}, saves ~${(cpuEmbedTensor.shape[0] * cpuEmbedTensor.shape[1] * 2 / 1024 / 1024).toFixed(0)} MB VRAM`);
      }

      // Check if embedding was split across multiple GPU BF16 buffers
      const embedSplitData = !cpuEmbedTensor ? detectSplit(embedBaseName) : undefined;
      if (embedSplitData) {
        global.embedSplit = embedSplitData;
        global.embedIsF16 = true;
        console.log(`[Engine] Embedding is split BF16: ${embedSplitData.buffers.length} buffers, splitPoints=[${embedSplitData.splitPoints}]`);
      }

      // Check if embedding is GPTQ INT4 (saves ~1.4 GB VRAM)
      const embedQ4 = (!embedSplitData && !cpuEmbedTensor) ? tryGetQ4(embedName) : undefined;
      if (embedQ4) {
        global.embedQ4 = embedQ4;
        global.embedIsF16 = false;
        console.log(`[Engine] Embedding is GPTQ INT4, using Q4 embed shader`);
      }

      // Check if lm_head is stored CPU-side or split across GPU buffers
      if (!config.tieWordEmbeddings) {
        const lmHeadBaseName = lmHeadName.replace('.weight', '');
        const cpuLmHeadTensor = currentModel!.tensors.get(`${lmHeadBaseName}.cpu_lm_head`) as any;
        if (cpuLmHeadTensor?.cpuLmHeadData) {
          const { parts, splitPoint, isBF16 } = cpuLmHeadTensor.cpuLmHeadData;
          global.lmHeadCPU = {
            parts, splitPoint,
            hiddenSize: config.hiddenSize,
            vocabSize: config.vocabSize,
            isBF16,
          };
          console.log(`[Engine] LM head is CPU-side BF16: ${parts.length} parts, splitPoint=${splitPoint}, saves ~${(cpuLmHeadTensor.shape[0] * cpuLmHeadTensor.shape[1] * 2 / 1024 / 1024).toFixed(0)} MB VRAM`);
        } else {
          const lmHeadSplitData = detectSplit(lmHeadBaseName);
          if (lmHeadSplitData) {
            global.lmHeadSplit = lmHeadSplitData;
            global.lmHeadIsBF16 = false;
            console.log(`[Engine] LM head is split BF16: ${lmHeadSplitData.buffers.length} buffers, splitPoints=[${lmHeadSplitData.splitPoints}]`);
          } else {
            const lmHeadQ4 = tryGetQ4(lmHeadName);
            if (lmHeadQ4) {
              global.lmHeadQ4 = lmHeadQ4;
              global.lmHeadIsBF16 = false;
              console.log(`[Engine] LM head is GPTQ INT4, using Q4 matmul`);
            }
          }
        }
      } else if (embedQ4) {
        global.lmHeadQ4 = embedQ4;
        global.lmHeadIsBF16 = false;
        console.log(`[Engine] LM head is GPTQ INT4 (tied), using Q4 matmul`);
      }

      // Track BF16 weight buffers (for BF16 matmul dispatch)
      const bf16Buffers = new Set<GPUBuffer>();
      const isBF16Weight = (name: string): boolean => {
        const t = currentModel!.tensors.get(name);
        return t ? (t.dtype === 'BF16' || t.dtype === 'F16') : false;
      };
      const trackBF16 = (name: string, buf: GPUBuffer | undefined) => {
        if (buf && isBF16Weight(name)) bf16Buffers.add(buf);
      };

      // Helper: CPU-side GPTQ dequant to f32 GPU buffer (for mixed-precision SSM)
      const { dequantGPTQ } = await import('./model/weight-loader');
      const dequantToF32Buffer = (weightName: string): GPUBuffer | undefined => {
        const base = weightName.replace('.weight', '');
        const qwTensor = currentModel!.tensors.get(`${base}.qweight`);
        const scTensor = currentModel!.tensors.get(`${base}.scales`);
        const qzTensor = currentModel!.tensors.get(`${base}.qzeros`);
        if (!qwTensor || !scTensor || !qzTensor) return undefined;

        // Read raw data from GPU buffers back — we need CPU arrays
        // Actually, the raw tensor data is in the GPU buffer. We need the original CPU data.
        // For now, skip CPU dequant — the raw safetensors data isn't easily accessible here.
        // TODO: implement by keeping raw GPTQ arrays during loading
        return undefined;
      };

      // Debug: log tensor names at layer 0 (helps diagnose weight mapping issues)
      const l0Keys = [...currentModel!.tensors.keys()].filter(k => k.includes('layers.0.'));
      if (l0Keys.length > 0) console.log(`[Engine] Layer 0: ${l0Keys.length} tensors`);

      const layers = [];
      for (let l = 0; l < config.numLayers; l++) {
        const isLinearLayer = config.layers[l].kind === 'linear_attention';

        // Shared weights (both layer types)
        const gateName = loc.locate('gateProj', l)!;
        const upName = loc.locate('upProj', l)!;
        const downName = loc.locate('downProj', l)!;
        const lw: any = {
          inputNorm: getTensor(loc.locate('inputNorm', l)!),
          postAttnNorm: getTensor(loc.locate('postAttnNorm', l)!),
          gateProj: tryGetTensor(gateName),
          upProj: tryGetTensor(upName),
          downProj: tryGetTensor(downName),
        };
        trackBF16(gateName, lw.gateProj);
        trackBF16(upName, lw.upProj);
        trackBF16(downName, lw.downProj);

        const linQKVName = loc.locate('linInProjQKV', l);
        if (isLinearLayer && linQKVName) {
          // ── Linear attention layer weights ──────────────────────────
          const linNames = {
            qkv: linQKVName,
            a: loc.locate('linInProjA', l)!,
            b: loc.locate('linInProjB', l)!,
            z: loc.locate('linInProjZ', l)!,
            out: loc.locate('linOutProj', l)!,
          };
          lw.linearInProjQKV = tryGetTensor(linNames.qkv);
          lw.linearInProjA = tryGetTensor(linNames.a);
          lw.linearInProjB = tryGetTensor(linNames.b);
          lw.linearInProjZ = tryGetTensor(linNames.z);
          lw.linearOutProj = tryGetTensor(linNames.out);
          // Track BF16 SSM weights
          trackBF16(linNames.qkv, lw.linearInProjQKV);
          trackBF16(linNames.a, lw.linearInProjA);
          trackBF16(linNames.b, lw.linearInProjB);
          trackBF16(linNames.z, lw.linearInProjZ);
          trackBF16(linNames.out, lw.linearOutProj);

          // Non-quantized weights (BF16/F16 → f32)
          lw.linearALog = tryGetRole('linALog', l);
          lw.linearConv1dWeight = tryGetRole('linConv1dWeight', l);
          lw.linearDtBias = tryGetRole('linDtBias', l);
          lw.linearNormWeight = tryGetRole('linNormWeight', l);

          // GPTQ INT4 / INT8 for linear attention projections
          if (config.isQuantized) {
            const linQ4Keys = ['linearInProjQKV', 'linearInProjA', 'linearInProjB', 'linearInProjZ', 'linearOutProj'] as const;
            const linRoles = ['linInProjQKV', 'linInProjA', 'linInProjB', 'linInProjZ', 'linOutProj'] as const;
            for (let k = 0; k < linQ4Keys.length; k++) {
              const weightName = loc.locate(linRoles[k], l)!;
              // Try INT8 first, then INT4
              const q8 = tryGetQ8(weightName);
              if (q8) {
                (lw as any)[`${linQ4Keys[k]}_q8`] = q8;
              } else {
                const q4 = tryGetQ4(weightName);
                if (q4) {
                  lw[`${linQ4Keys[k]}_q4`] = q4;
                  if (l === 0) console.log(`[Q4] L0 ${linQ4Keys[k]}: GPTQ loaded`);
                }
              }
            }
          }

          if (l === 0) {
            console.log(`[Engine] L0: linear_attention layer`);
            console.log(`[Engine] L0 A_log: ${lw.linearALog ? 'FOUND' : 'MISSING'}`);
            console.log(`[Engine] L0 conv1d: ${lw.linearConv1dWeight ? 'FOUND' : 'MISSING'}`);
            console.log(`[Engine] L0 dt_bias: ${lw.linearDtBias ? 'FOUND' : 'MISSING'}`);
            console.log(`[Engine] L0 norm: ${lw.linearNormWeight ? 'FOUND' : 'MISSING'}`);
          }
        } else {
          // ── Standard softmax attention layer weights ────────────────
          const qName = loc.locate('qProj', l)!;
          const kName = loc.locate('kProj', l)!;
          const vName = loc.locate('vProj', l)!;
          const oName = loc.locate('oProj', l)!;
          lw.qProj = tryGetTensor(qName);
          lw.kProj = tryGetTensor(kName);
          lw.vProj = tryGetTensor(vName);
          lw.oProj = tryGetTensor(oName);
          trackBF16(qName, lw.qProj);
          trackBF16(kName, lw.kProj);
          trackBF16(vName, lw.vProj);
          trackBF16(oName, lw.oProj);

          // GPTQ INT4 / INT8 for standard attention projections
          if (config.isQuantized) {
            const nameKeys = ['qProj', 'kProj', 'vProj', 'oProj'] as const;
            for (const key of nameKeys) {
              const weightName = loc.locate(key, l)!;
              const q8 = tryGetQ8(weightName);
              if (q8) {
                (lw as any)[`${key}_q8`] = q8;
              } else {
                const q4 = tryGetQ4(weightName);
                if (q4) {
                  lw[`${key}_q4`] = q4;
                  if (l === 3) console.log(`[Q4] L3 ${key}: GPTQ loaded`);
                }
              }
            }
          }

          // Bias terms (only for full attention layers with bias)
          if (config.attentionBias) {
            lw.qBias = tryGetRole('qBias', l);
            lw.kBias = tryGetRole('kBias', l);
            lw.vBias = tryGetRole('vBias', l);
            lw.oBias = tryGetRole('oBias', l);
          }

          // Q/K per-head RMSNorm (Qwen3.5 full attention)
          if (loc.locate('qNorm', l) !== undefined) {
            lw.qNorm = tryGetRole('qNorm', l);
            lw.kNorm = tryGetRole('kNorm', l);
            if (l === 3) console.log(`[Engine] L3 qNorm: ${lw.qNorm ? 'FOUND' : 'MISSING'}, kNorm: ${lw.kNorm ? 'FOUND' : 'MISSING'}`);
          }
        }

        // FFN GPTQ INT4 / INT8 (shared for both layer types)
        if (config.isQuantized) {
          for (const key of ['gateProj', 'upProj', 'downProj'] as const) {
            const weightName = loc.locate(key, l)!;
            const q8 = tryGetQ8(weightName);
            if (q8) {
              (lw as any)[`${key}_q8`] = q8;
            } else {
              const q4 = tryGetQ4(weightName);
              if (q4) {
                lw[`${key}_q4`] = q4;
                if (l === 0) console.log(`[Q4] L0 ${key}: GPTQ loaded`);
              }
            }
          }
        }

        layers.push(lw);
      }

      if (config.isHybrid) {
        const linCount = config.layers.filter(d => d.kind === 'linear_attention').length;
        const fullCount = config.layers.filter(d => d.kind === 'full_attention').length;
        console.log(`[Engine] Hybrid model: ${linCount} linear + ${fullCount} full attention layers`);
      }
      if (l0Keys.length > 0) console.log(`[Engine] attentionBias=${config.attentionBias}`);

      if (bf16Buffers.size > 0) {
        console.log(`[Engine] ${bf16Buffers.size} weight buffers kept in BF16 (using BF16 matmul kernel)`);
      } else {
        console.log(`[Engine] All weights in f32 (bf16Buffers empty)`);
      }

      // Summarize g_idx trivialness probe — tells us whether the GEMV fast path
      // or actorder path will dominate during decode.
      let gidxTotal = 0, gidxTrivial = 0, gidxReal = 0;
      for (const [n, t] of currentModel!.tensors) {
        if (!n.endsWith('.g_idx')) continue;
        gidxTotal++;
        if (t.isTrivialGIdx) gidxTrivial++;
        else gidxReal++;
      }
      if (gidxTotal > 0) {
        console.log(`[Engine] g_idx probe: ${gidxTrivial}/${gidxTotal} trivial (fast path), ${gidxReal}/${gidxTotal} actorder (slow path)`);
      }
      const engine = createForwardPassEngine(gpu!.device, config, { global, layers, bf16Buffers });
      const tokenizer = await createTokenizer({ modelId: repo });

      // Reset to CDN now that weights + tokenizer are loaded from local cache
      if (usingLocalCache) resetToRemote();

      // Build chat template function
      const { applyChatTemplate } = await import('./model/tokenizer');

      // Store session for the send button
      const kvSession = createKVSession(Math.min(config.maxPositionEmbeddings || 8192, 8192, MAX_ATTN_SEQ_LEN));
      const resetKV = () => {
        if (kvSession.kvCache) engine.destroyKVCache(kvSession.kvCache);
        kvSession.kvCache = null;
        kvSession.cachedTokenIds = [];
      };
      session = {
        run: (prompt, sampling, onToken) => generate(gpu!.device, engine, tokenizer, prompt, sampling, onToken),
        chat: (messages: Array<{role: string; content: string}>, sampling: any, onToken: any, opts?: { enableThinking?: boolean }) => {
          const tokenIds = applyChatTemplate(tokenizer, messages, opts);
          return generate(gpu!.device, engine, tokenizer, tokenIds, sampling, onToken, { kvSession });
        },
        // Multimodal turn: template with placeholder spans already in the
        // user content, locate the pad runs, pair them with encoded images.
        chatMM: (
          messages: Array<{role: string; content: string}>,
          encoded: VisionEncodeResult[],
          sampling: any, onToken: any, opts?: { enableThinking?: boolean },
        ) => {
          const tokenIds = applyChatTemplate(tokenizer, messages, opts);
          const padId = activeVisionDesc!.placeholder.imageTokenId;
          const spans: Array<{ start: number; count: number }> = [];
          let i = 0;
          while (i < tokenIds.length) {
            if (tokenIds[i] === padId) {
              let j = i;
              while (j < tokenIds.length && tokenIds[j] === padId) j++;
              spans.push({ start: i, count: j - i });
              i = j;
            } else i++;
          }
          if (spans.length !== encoded.length) {
            throw new Error(`Image span mismatch: ${spans.length} pad runs in prompt vs ${encoded.length} encoded images`);
          }
          const images = spans.map((s, k) => {
            if (s.count !== encoded[k].numTokens) {
              throw new Error(`Image ${k}: ${s.count} pad tokens vs ${encoded[k].numTokens} embedding rows`);
            }
            return {
              start: s.start, count: s.count,
              embeddings: encoded[k].embeddings,
              deepstack: encoded[k].deepstack,
            };
          });
          return generate(gpu!.device, engine, tokenizer, { tokenIds, images }, sampling, onToken, { kvSession });
        },
        kvSession,
        resetKV,
        config,
        tokenizer,
        gpu: gpu!,
        vramEstimate: estimateVRAM(config),
        destroy: () => { resetVision(); resetKV(); unloadModel(currentModel!); session = null; },
      } as InferenceSession;

      // Vision support: multimodal checkpoints get a descriptor → 📎 enabled
      try {
        const pp = await fetchPreprocessorConfig(repo);
        activeVisionSource = { kind: 'hf' };
        setVisionDesc(visionDescriptorFromHFConfig(currentModel!.config as Record<string, any>, pp));
      } catch {
        setVisionDescQuiet(null);
      }

      // Enable chat input
      promptEl.disabled = false;
      sendBtn.disabled = false;

      const rotaryInfo = config.partialRotaryFactor
        ? `partial=${config.partialRotaryFactor} (${Math.floor(config.partialRotaryFactor * config.headDim)} of ${config.headDim} dims)`
        : `full (all ${config.headDim} dims)`;
      addMessage('system',
        `Inference engine ready!\n` +
        `Model: ${config.modelType} — ${config.numLayers} layers, ${config.numAttentionHeads} heads, d=${config.hiddenSize}\n` +
        `GQA: ${config.isGQA ? `${config.numAttentionHeads}Q/${config.numKVHeads}KV` : 'no'}\n` +
        `Vocab: ${config.vocabSize} | RoPE θ=${config.ropeTheta} | Rotary: ${rotaryInfo}\n` +
        `Type a message to chat!`,
        'engine ready'
      );

      setStatus(`Ready: ${repo}`);
      markModelReady(repo);

      startAutoTestPoller();

    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      addMessage('system', `Engine build failed: ${msg}\n\nWeights are loaded but inference is not available.`);
      setStatus(`Engine error: ${msg}`);
      reportError('engine-build', err);
    }

    updateFooter({ model: repo.split('/').pop() || repo });

  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    progressEl.textContent = `Error: ${msg}`;
    addMessage('system', `Failed to load ${repo}: ${msg}`);
    setStatus('Load failed');
    reportError('model-load', err);
  } finally {
    loadBtn.disabled = false;
  }
});

// ─── Clear / Export / Unload ─────────────────────────────────────────────────

clearBtn.addEventListener('click', () => {
  messagesEl.innerHTML = '';
  chatHistory = [];
  session?.resetKV?.();
  // Remember the clear so the boot-time restore doesn't resurrect autosaves
  // from before it.
  try { localStorage.setItem('artifex-ui:cleared-at', String(Date.now() / 1000)); } catch {}
  addMessage('system', 'Chat cleared.');
});

// ─── Sessions (save / load — session.py-compatible format) ──────────────────

function applyLoadedSession(state: SessionFile) {
  chatHistory = state.messages.filter(
    (m): m is ChatMessage => m.role === 'user' || m.role === 'assistant');
  session?.resetKV?.();  // history changed — next send does a full prefill
  messagesEl.innerHTML = '';
  for (const m of chatHistory) addChatMessage(m);
  const model = state.metadata?.model ? ` (saved with ${state.metadata.model})` : '';
  addMessage('system', `Session "${state.name}" loaded — ${chatHistory.length} messages${model}.`);
}

($('save-session-btn') as HTMLButtonElement)?.addEventListener('click', () => {
  if (chatHistory.length === 0) {
    addMessage('system', 'Nothing to save yet.');
    return;
  }
  const name = window.prompt('Session name:', 'chat');
  if (!name) return;
  const metadata = { model: currentModel?.repo ?? 'unknown', backend: 'webgpu' };
  const key = saveSession(name, chatHistory, metadata);
  exportSessionFile(buildSessionState(name, chatHistory, metadata));
  addMessage('system', `Session saved (${key.split(':')[1]}) and downloaded as .json — drop it in Artifex's sessions/ to continue there.`);
});

($('load-session-btn') as HTMLButtonElement)?.addEventListener('click', () => {
  const fileInput = $('session-file-input') as HTMLInputElement;
  const sessions = listSessions();
  if (sessions.length === 0) {
    fileInput.click();
    return;
  }
  const menu = sessions.slice(0, 10)
    .map((s, i) => `${i + 1}. ${s.name} — ${s.message_count} msgs (${s.timestamp})`)
    .join('\n');
  const choice = window.prompt(
    `Enter a session number to load, or leave empty to pick a .json file:\n${menu}`);
  if (choice === null) return;  // cancelled
  const idx = parseInt(choice, 10);
  if (!choice.trim() || Number.isNaN(idx)) {
    fileInput.click();
    return;
  }
  const picked = sessions[idx - 1];
  const state = picked ? loadSession(picked.key) : null;
  if (state) applyLoadedSession(state);
  else addMessage('system', 'Could not load that session.');
});

($('session-file-input') as HTMLInputElement)?.addEventListener('change', async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  const state = await importSessionFile(file);
  if (state) applyLoadedSession(state);
  else addMessage('system', `Could not parse ${file.name} as a session file.`);
  (e.target as HTMLInputElement).value = '';
});

exportBtn.addEventListener('click', () => {
  // Export from history, not the DOM — rendered markdown flattened to
  // textContent would lose code fences and formatting.
  let md = '# Artifex WebGPU Chat Export\n\n';
  const sysPrompt = ($('system-prompt') as HTMLTextAreaElement).value.trim();
  if (sysPrompt && chatHistory.length > 0) md += `## System\n\n${sysPrompt}\n\n`;
  for (const m of chatHistory) {
    md += `## ${m.role === 'user' ? 'User' : 'Assistant'}\n\n${m.content}\n\n`;
  }
  const blob = new Blob([md], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `artifex-chat-${Date.now()}.md`;
  a.click();
  URL.revokeObjectURL(url);
});

unloadBtn.addEventListener('click', async () => {
  if (imageGenRepo) {
    imageGenRepo = null;
    addMessage('system', 'Image-gen mode off (weights load per generation — nothing was resident).');
    setStatus('Model unloaded');
    updateFooter({ model: 'none' });
    return;
  }
  if (currentModel) {
    const name = currentModel.repo;
    const freed = formatBytes(currentModel.totalGPUBytes);
    resetVision();
    unloadModel(currentModel);
    currentModel = null;
    session = null;
    addMessage('system', `Model unloaded: ${name} — freed ${freed} GPU memory`);
    setStatus('Model unloaded');
    updateFooter({ model: 'none' });
  } else {
    addMessage('system', 'No model loaded.');
  }
});

// ─── Cache Management (Storage panel) ────────────────────────────────────────
// Model downloads land in the browser's Cache API — multi-GB blobs that are
// otherwise invisible and only reclaimable through DevTools. The panel lists
// them with sizes and per-model delete.

const cacheListEl = $('cache-list');

async function renderCachePanel(): Promise<void> {
  const stats = await getCacheStats();
  cacheListEl.innerHTML = '';
  if (stats.itemCount === 0) {
    cacheListEl.textContent = 'No cached downloads.';
    return;
  }
  for (const [repo, info] of [...stats.models.entries()].sort((a, b) => b[1].totalBytes - a[1].totalBytes)) {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;justify-content:space-between;align-items:center;gap:6px;padding:2px 0';
    const label = document.createElement('span');
    label.textContent = `${repo} — ${formatBytes(info.totalBytes)}`;
    label.style.cssText = 'overflow:hidden;text-overflow:ellipsis;white-space:nowrap';
    label.title = `${repo}: ${info.shardCount} file(s), ${formatBytes(info.totalBytes)}`;
    const del = document.createElement('button');
    del.textContent = '✕';
    del.title = `Delete cached download of ${repo} (re-downloads on next load)`;
    del.style.cssText = 'background:none;border:1px solid #333;border-radius:3px;color:var(--dim);cursor:pointer;font-size:10px;padding:1px 6px';
    del.addEventListener('click', async () => {
      del.disabled = true;
      await removeModelFromCache(repo);
      await renderCachePanel();
    });
    row.append(label, del);
    cacheListEl.appendChild(row);
  }
  const total = document.createElement('div');
  total.textContent = `total: ${formatBytes(stats.totalBytes)}`;
  total.style.cssText = 'margin-top:4px;color:var(--accent)';
  cacheListEl.appendChild(total);
}

($('cache-refresh-btn') as HTMLButtonElement).addEventListener('click', () => void renderCachePanel());
($('cache-clear-btn') as HTMLButtonElement).addEventListener('click', async () => {
  if (!window.confirm('Delete ALL cached model downloads? Models will re-download on next load.')) return;
  await clearCache();
  await renderCachePanel();
});

void renderCachePanel();

// ─── UI Settings Persistence ─────────────────────────────────────────────────
// Sampler preset (+ custom slider values), system prompt, max tokens,
// TurboQuant toggle, and the last loaded model survive reloads.

const UI_STORE = 'artifex-ui:';

function persistUISettings(): void {
  try {
    localStorage.setItem(UI_STORE + 'settings', JSON.stringify({
      preset: presetSelect.value,
      temperature: tempSlider.value,
      topP: toppSlider.value,
      topK: topkInput.value,
      minP: minpSlider.value,
      repPen: reppenSlider.value,
      dryMult: drySlider.value,
      maxTokens: ($('max-tokens') as HTMLInputElement).value,
      turboquant: ($('turboquant') as HTMLInputElement).checked,
      systemPrompt: ($('system-prompt') as HTMLTextAreaElement).value,
      toolsEnabled: ($('tools-enabled') as HTMLInputElement).checked,
      webSearchEnabled: ($('web-search-enabled') as HTMLInputElement).checked,
    }));
  } catch { /* quota — best-effort */ }
}

function restoreUISettings(): void {
  try {
    const raw = localStorage.getItem(UI_STORE + 'settings');
    if (!raw) return;
    const s = JSON.parse(raw);
    if (s.preset === 'custom') {
      suppressPresetFlip = true;
      if (s.temperature !== undefined) tempSlider.value = s.temperature;
      if (s.topP !== undefined) toppSlider.value = s.topP;
      if (s.topK !== undefined) topkInput.value = s.topK;
      if (s.minP !== undefined) minpSlider.value = s.minP;
      if (s.repPen !== undefined) reppenSlider.value = s.repPen;
      if (s.dryMult !== undefined) drySlider.value = s.dryMult;
      [tempSlider, toppSlider, minpSlider, reppenSlider, drySlider].forEach(el =>
        el.dispatchEvent(new Event('input')));
      suppressPresetFlip = false;
      presetSelect.value = 'custom';
    } else if (s.preset && s.preset in PRESETS) {
      presetSelect.value = s.preset;
      applyPreset(s.preset as PresetName);
    }
    if (s.maxTokens) ($('max-tokens') as HTMLInputElement).value = s.maxTokens;
    if (typeof s.turboquant === 'boolean') ($('turboquant') as HTMLInputElement).checked = s.turboquant;
    if (typeof s.toolsEnabled === 'boolean') ($('tools-enabled') as HTMLInputElement).checked = s.toolsEnabled;
    if (typeof s.webSearchEnabled === 'boolean') ($('web-search-enabled') as HTMLInputElement).checked = s.webSearchEnabled;
    if (typeof s.systemPrompt === 'string' && s.systemPrompt.trim()) {
      ($('system-prompt') as HTMLTextAreaElement).value = s.systemPrompt;
    }
  } catch { /* corrupt entry — defaults stand */ }
}

// Search is keyless now (local /api/search proxy) — clear the key slot any
// pre-release build may have written.
try { localStorage.removeItem(UI_STORE + 'tavily-key'); } catch {}

for (const id of ['max-tokens', 'turboquant', 'system-prompt', 'tools-enabled', 'web-search-enabled']) {
  $(id).addEventListener('change', persistUISettings);
  $(id).addEventListener('input', persistUISettings);
}
for (const el of [tempSlider, toppSlider, topkInput, minpSlider, reppenSlider, drySlider]) {
  el.addEventListener('change', persistUISettings);
}
presetSelect.addEventListener('change', persistUISettings);

/** Called by both load paths once the engine is ready. */
function markModelReady(repo: string): void {
  try { localStorage.setItem(UI_STORE + 'last-model', repo); } catch {}
  // Mid-chat model switch: history is text-level, so it carries over — the
  // new model just re-prefills it on the next send.
  if (chatHistory.length > 0) {
    addMessage('system',
      `Conversation continues with ${repo} — it re-reads the chat history on your next message.`);
  }
  void renderCachePanel();  // a fresh download may have landed in the cache
}

/** Boot-time restore of the most recent conversation (autosave or named),
 *  unless the user's last action was Clear Chat. The model is NOT auto-loaded
 *  — multi-GB downloads/uploads shouldn't happen on page open. */
function restoreLastConversation(): void {
  try {
    const latest = listSessions()[0];
    if (!latest || latest.message_count === 0) return;
    const clearedAt = Number(localStorage.getItem(UI_STORE + 'cleared-at') ?? 0);
    if (latest.saved_at <= clearedAt) return;
    const state = loadSession(latest.key);
    if (!state) return;
    chatHistory = state.messages.filter(
      (m): m is ChatMessage => m.role === 'user' || m.role === 'assistant');
    if (chatHistory.length === 0) return;
    for (const m of chatHistory) addChatMessage(m);
    addMessage('system',
      `Restored last conversation (${chatHistory.length} messages) — Clear Chat to start fresh.`);
  } catch { /* corrupt store — start clean */ }
}

// ─── Boot ────────────────────────────────────────────────────────────────────

// Restore HF auth token from localStorage
const savedToken = localStorage.getItem('hf-token');
if (savedToken) {
  setAuthToken(savedToken);
  ($('hf-token') as HTMLInputElement).value = savedToken;
}

restoreUISettings();
const lastModel = localStorage.getItem(UI_STORE + 'last-model');
if (lastModel) ($('model-repo') as HTMLInputElement).value = lastModel;
restoreLastConversation();

// PWA service worker — production builds only. In dev, vite owns module
// serving and the dev server sets the COOP/COEP headers itself.
if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`)
    .catch(err => console.warn('[PWA] service worker registration failed:', err));
}

init();
