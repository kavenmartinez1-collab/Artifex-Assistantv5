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
import { loadModel, unloadModel, previewModel, formatBytes, getCacheStats, clearCache, type LoadedModel } from './model';
import { setAuthToken, useLocalCache, resetToRemote } from './model/hf-hub';
import { createInferenceSession, type InferenceSession } from './engine/inference';
import { parseModelConfig, estimateVRAM } from './model/model-config';
import { descriptorFromHFConfig } from './model/model-descriptor';
import type { TensorRole } from './model/tensor-locator';

// ─── State ───────────────────────────────────────────────────────────────────

let gpu: GPUContext | null = null;
let currentModel: LoadedModel | null = null;
let session: InferenceSession | null = null;

// ─── DOM Elements ────────────────────────────────────────────────────────────

const $ = (id: string) => document.getElementById(id)!;
const gpuBadge = $('gpu-badge');
const gpuSelect = $('gpu-select') as HTMLSelectElement;
const statusEl = $('status');
const messagesEl = $('messages');
const promptEl = $('prompt') as HTMLTextAreaElement;
const sendBtn = $('send-btn') as HTMLButtonElement;
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

// Connect to hub (non-blocking — doesn't prevent app from working without hub)
setTimeout(connectOrchestrator, 1000);

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

  try {
    const resp = await fetch('/api/hf-cache/models');
    if (!resp.ok) throw new Error(`${resp.status}`);
    const models = await resp.json() as Array<{ repo: string; files: string[]; totalSize: number }>;

    if (models.length === 0) {
      modelList.innerHTML = '<div style="padding:8px;color:var(--dim)">No local models found</div>';
      return;
    }

    // Sort: local/ models first, then by size descending
    models.sort((a, b) => {
      const aLocal = a.repo.startsWith('local/') ? 0 : 1;
      const bLocal = b.repo.startsWith('local/') ? 0 : 1;
      if (aLocal !== bLocal) return aLocal - bLocal;
      return b.totalSize - a.totalSize;
    });

    modelList.innerHTML = '';
    for (const m of models) {
      const el = document.createElement('div');
      const sizeGB = (m.totalSize / 1024 / 1024 / 1024).toFixed(1);
      const safetensors = m.files.filter(f => f.endsWith('.safetensors')).length;
      const isLocal = m.repo.startsWith('local/');
      el.style.cssText = 'padding:6px 8px;cursor:pointer;border-bottom:1px solid #222;display:flex;justify-content:space-between;align-items:center';
      el.innerHTML = `
        <span style="color:${isLocal ? 'var(--accent)' : 'var(--text)'}">${isLocal ? '📁 ' : ''}${m.repo}</span>
        <span style="color:var(--dim);font-size:10px;white-space:nowrap;margin-left:8px">${sizeGB} GB · ${safetensors} shard${safetensors !== 1 ? 's' : ''}</span>
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

// ─── UI Helpers ──────────────────────────────────────────────────────────────

function setStatus(text: string) {
  statusEl.textContent = text;
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
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

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

// ─── Chat (placeholder — functional once model loading is implemented) ───────

sendBtn.addEventListener('click', async () => {
  const text = promptEl.value.trim();
  if (!text) return;

  promptEl.value = '';
  promptEl.style.height = 'auto';
  addMessage('user', text);

  if (!session) {
    addMessage('system', 'No model loaded. Load a model first.');
    return;
  }

  setStatus('Generating...');
  sendBtn.disabled = true;

  try {
    const responseDiv = addMessage('assistant', '');
    let fullText = '';

    const temperature = parseFloat(tempSlider.value);
    const topP = parseFloat(toppSlider.value);
    const topK = parseInt(topkInput.value) || 0;
    const minP = parseFloat(minpSlider.value);
    const repetitionPenalty = parseFloat(reppenSlider.value);
    const dryMultiplier = parseFloat(drySlider.value);
    const maxNewTokens = parseInt(($('max-tokens') as HTMLInputElement).value) || 512;
    const useCompressedKV = ($('turboquant') as HTMLInputElement).checked;

    // /raw prefix: skip chat template, use raw text completion (for debugging)
    const isRaw = text.startsWith('/raw ');
    const sampling = {
      temperature, topP, topK, minP, repetitionPenalty, dryMultiplier,
      maxNewTokens, useCompressedKV,
    };
    let thinkingDone = false;
    const onToken = (token: string) => {
      fullText += token;

      // Show thinking content dimmed, final answer bold
      if (fullText.includes('</think>')) {
        thinkingDone = true;
        const parts = fullText.split('</think>');
        const thinking = parts[0].replace('<think>', '').trim();
        const answer = parts.slice(1).join('</think>').trimStart();
        responseDiv.innerHTML = (thinking ? `<span style="opacity:0.4;font-size:0.85em">${thinking}</span><br><br>` : '') + answer;
      } else {
        // Still thinking — show content with dim styling
        const thinking = fullText.replace('<think>', '').trim();
        responseDiv.innerHTML = `<span style="opacity:0.4;font-size:0.85em">💭 ${thinking}</span>`;
      }
      messagesEl.scrollTop = messagesEl.scrollHeight;
    };

    let handle;
    if (isRaw) {
      const rawText = text.slice(5); // strip "/raw "
      console.log(`[Raw mode] Sending: "${rawText}"`);
      handle = session.run(rawText, sampling, onToken);
    } else {
      const systemPrompt = ($('system-prompt') as HTMLTextAreaElement).value.trim();
      const messages: Array<{ role: string; content: string }> = [];
      if (systemPrompt) {
        messages.push({ role: 'system', content: systemPrompt });
      }
      messages.push({ role: 'user', content: text });
      handle = (session as any).chat(messages, sampling, onToken, { enableThinking: true });
    }

    const result = await handle.result;

    // If no streaming happened (e.g., empty response), show the full text
    if (!fullText && result.text) {
      responseDiv.textContent = result.text;
    }

    const meta = `${result.numTokens} tokens | ${result.tokensPerSecond.toFixed(1)} tok/s | ${(result.totalMs / 1000).toFixed(1)}s | ${result.stopReason}`;
    const metaEl = document.createElement('div');
    metaEl.className = 'meta';
    metaEl.textContent = meta;
    responseDiv.appendChild(metaEl);

    setStatus(`Generated ${result.numTokens} tokens at ${result.tokensPerSecond.toFixed(1)} tok/s`);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    addMessage('system', `Generation error: ${msg}`);
    setStatus('Generation failed');
    reportError('generate', err);
  } finally {
    sendBtn.disabled = false;
  }
});

// ─── Auto-test polling ──────────────────────────────────────────────────────
// Polls /api/test for queued prompts/kernel-test requests while a session
// exists. Shared by the safetensors and GGUF load paths.
function startAutoTestPoller() {
      // Intercept console.log to capture debug output from forward pass
      const _origLog = console.log;
      let debugLogs: string[] = [];

      (async function pollTests() {
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
async function buildGGUFSession(repo: string, ggufFile: string, progressEl: HTMLElement): Promise<void> {
  const { loadGGUFModel } = await import('./model/gguf-loader');
  const { descriptorFromGGUF, applyRopeFreqFactors } = await import('./model/model-descriptor');
  const { ggufArchitecture } = await import('./model/gguf');
  const { createGGUFLocator } = await import('./model/tensor-locator');
  const { createForwardPassEngine } = await import('./engine/forward-pass');
  const { createTokenizer, applyChatTemplate } = await import('./model/tokenizer');
  const { generate } = await import('./engine/generate');

  addMessage('system', `Loading GGUF: ${repo}/${ggufFile} ...`);
  const model = await loadGGUFModel(gpu!.device, repo, ggufFile, (p) => {
    progressEl.textContent = p.message;
    if (p.overallProgress !== undefined) {
      setStatus(`Loading ${repo}... ${Math.round(p.overallProgress * 100)}%`);
    }
  });
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
      assignProj(lw, 'kProj', 'kProj', l);
      assignProj(lw, 'vProj', 'vProj', l);
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
    layers.push(lw);
  }
  if (config.isHybrid) {
    const linCount = config.layers.filter(d => d.kind === 'linear_attention').length;
    console.log(`[GGUF] Hybrid model: ${linCount} linear + ${config.numLayers - linCount} full attention layers`);
  }

  const engine = createForwardPassEngine(gpu!.device, config, { global, layers, moe });
  const tokenizer = await createTokenizer({ modelId: repo });
  // Guard: a tokenizer from the wrong model family silently produces garbage
  // (e.g. Qwen3-8B fallback vocab 151936 vs Qwen3.6 vocab 248320).
  if (tokenizer.vocabSize && Math.abs(tokenizer.vocabSize - config.vocabSize) > 1024) {
    throw new Error(
      `Tokenizer/model vocab mismatch: tokenizer ${tokenizer.modelId} has ${tokenizer.vocabSize} tokens ` +
      `but model expects ${config.vocabSize}. Wrong tokenizer fallback?`);
  }
  resetToRemote();

  session = {
    run: (prompt, sampling, onToken) => generate(gpu!.device, engine, tokenizer, prompt, sampling, onToken),
    chat: (messages: Array<{role: string; content: string}>, sampling: any, onToken: any, opts?: { enableThinking?: boolean }) => {
      const tokenIds = applyChatTemplate(tokenizer, messages, opts);
      return generate(gpu!.device, engine, tokenizer, tokenIds, sampling, onToken);
    },
    config,
    tokenizer,
    gpu: gpu!,
    vramEstimate: estimateVRAM(config),
    destroy: () => { moe?.backend.destroy(); unloadModel(currentModel!); session = null; },
  } as InferenceSession;

  promptEl.disabled = false;
  sendBtn.disabled = false;

  addMessage('system',
    `Inference engine ready! (GGUF native)\n` +
    `Model: ${config.modelType} — ${config.numLayers} layers, ${config.numAttentionHeads} heads, d=${config.hiddenSize}\n` +
    `Vocab: ${config.vocabSize} | RoPE θ=${config.ropeTheta}\n` +
    `Type a message to chat!`,
    'engine ready');
  setStatus(`Ready: ${repo}`);
  startAutoTestPoller();
}

// ─── Load Model ──────────────────────────────────────────────────────────────

loadBtn.addEventListener('click', async () => {
  const repo = ($('model-repo') as HTMLInputElement).value.trim();
  if (!repo) return;

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
    unloadModel(currentModel);
    currentModel = null;
  }

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

    if (!usingLocalCache) {
      addMessage('system', `Connecting to HuggingFace: ${repo}...`);
    }

    // GGUF-only model dir (no config.json): native GGUF load path
    const ggufFiles = (localFiles ?? []).filter(f => f.endsWith('.gguf') && !f.toLowerCase().includes('mmproj'));
    if (ggufFiles.length > 0 && !(localFiles ?? []).includes('config.json')) {
      await buildGGUFSession(repo, ggufFiles[0], progressEl);
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
      const { generate } = await import('./engine/generate');

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
      session = {
        run: (prompt, sampling, onToken) => generate(gpu!.device, engine, tokenizer, prompt, sampling, onToken),
        chat: (messages: Array<{role: string; content: string}>, sampling: any, onToken: any, opts?: { enableThinking?: boolean }) => {
          const tokenIds = applyChatTemplate(tokenizer, messages, opts);
          return generate(gpu!.device, engine, tokenizer, tokenIds, sampling, onToken);
        },
        config,
        tokenizer,
        gpu: gpu!,
        vramEstimate: estimateVRAM(config),
        destroy: () => { unloadModel(currentModel!); session = null; },
      } as InferenceSession;

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
  addMessage('system', 'Chat cleared.');
});

exportBtn.addEventListener('click', () => {
  const msgs = messagesEl.querySelectorAll('.message');
  let md = '# Artifex WebGPU Chat Export\n\n';
  msgs.forEach(m => {
    const role = m.classList.contains('user') ? '## User' :
                 m.classList.contains('assistant') ? '## Assistant' : '## System';
    md += `${role}\n\n${m.textContent}\n\n`;
  });
  const blob = new Blob([md], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `artifex-chat-${Date.now()}.md`;
  a.click();
  URL.revokeObjectURL(url);
});

unloadBtn.addEventListener('click', async () => {
  if (currentModel) {
    const name = currentModel.repo;
    const freed = formatBytes(currentModel.totalGPUBytes);
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

// ─── Cache Management ────────────────────────────────────────────────────────

// Show cache stats on boot (async, non-blocking)
getCacheStats().then(stats => {
  if (stats.itemCount > 0) {
    console.log(`[Cache] ${stats.itemCount} shards cached (${formatBytes(stats.totalBytes)})`);
    for (const [repo, info] of stats.models) {
      console.log(`  ${repo}: ${info.shardCount} shards, ${formatBytes(info.totalBytes)}`);
    }
  }
});

// ─── Boot ────────────────────────────────────────────────────────────────────

// Restore HF auth token from localStorage
const savedToken = localStorage.getItem('hf-token');
if (savedToken) {
  setAuthToken(savedToken);
  ($('hf-token') as HTMLInputElement).value = savedToken;
}

init();
