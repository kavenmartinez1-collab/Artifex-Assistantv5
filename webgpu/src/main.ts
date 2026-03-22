/**
 * Artifex WebGPU Engine — Main Entry Point
 *
 * Initializes WebGPU, runs kernel tests, and sets up the chat UI.
 * This is Phase 0-1: device detection, compute foundation, and UI shell.
 */

import { initWebGPU, type GPUContext } from './engine/gpu-device';
import { reportMetric, reportError, timed } from './utils/metrics';
import { runKernelTests } from './engine/kernel-tests';

// ─── State ───────────────────────────────────────────────────────────────────

let gpu: GPUContext | null = null;

// ─── DOM Elements ────────────────────────────────────────────────────────────

const $ = (id: string) => document.getElementById(id)!;
const gpuBadge = $('gpu-badge');
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
const tempVal = $('temp-val');
const toppVal = $('topp-val');

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

tempSlider.addEventListener('input', () => {
  tempVal.textContent = tempSlider.value;
});

toppSlider.addEventListener('input', () => {
  toppVal.textContent = toppSlider.value;
});

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

async function init() {
  setStatus('Detecting WebGPU...');

  try {
    gpu = await timed('perf', 'webgpu-init', () => initWebGPU());

    const info = gpu.adapterInfo;
    const label = info.device || info.description || info.architecture || 'Unknown GPU';
    const maxMB = Math.round(gpu.maxBufferSize / (1024 * 1024));

    gpuBadge.textContent = `${label} (${maxMB} MB max buffer)`;
    gpuBadge.classList.remove('error');

    updateFooter({
      gpu: label,
      vram: `${maxMB} MB max`,
    });

    setStatus('WebGPU ready — load a model or run kernel tests');

    addMessage('system',
      `GPU: ${label}\n` +
      `Vendor: ${info.vendor || 'unknown'} | Arch: ${info.architecture || 'unknown'}\n` +
      `Max buffer: ${maxMB} MB | Max storage bindings: ${gpu.limits.maxStorageBuffersPerShaderStage}\n` +
      `Max workgroup: ${gpu.limits.maxComputeWorkgroupSizeX}x${gpu.limits.maxComputeWorkgroupSizeY}x${gpu.limits.maxComputeWorkgroupSizeZ}`
    );

  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    gpuBadge.textContent = 'No WebGPU';
    gpuBadge.classList.add('error');
    setStatus(`WebGPU Error: ${msg}`);
    addMessage('system', `WebGPU initialization failed: ${msg}`);
    reportError('init', err);
  }
}

// ─── Kernel Tests ────────────────────────────────────────────────────────────

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

  if (!gpu) {
    addMessage('system', 'No GPU available. Cannot generate.');
    return;
  }

  // TODO: Phase 4+ — actual inference
  setStatus('Generating...');
  sendBtn.disabled = true;

  addMessage('assistant',
    '[Model not loaded yet]\n\n' +
    'The WebGPU compute foundation is ready. Next steps:\n' +
    '  - Phase 2: SafeTensors weight loader\n' +
    '  - Phase 3: Tokenizer\n' +
    '  - Phase 4: Qwen3.5 model implementation\n' +
    '  - Phase 5: INT4 quantization\n' +
    '  - Phase 6: Sampling & generation loop\n\n' +
    'Run kernel tests to verify GPU compute is working.',
    'placeholder response'
  );

  setStatus('Ready');
  sendBtn.disabled = false;
});

// ─── Load Model (placeholder) ────────────────────────────────────────────────

loadBtn.addEventListener('click', async () => {
  const repo = ($('model-repo') as HTMLInputElement).value.trim();
  if (!repo) return;

  const progress = $('load-progress');
  progress.textContent = 'Loading not yet implemented — Phase 2+3+4 required';

  addMessage('system',
    `Model loading (${repo}) requires:\n` +
    '  - SafeTensors parser (Phase 2)\n' +
    '  - Tokenizer (Phase 3)\n' +
    '  - Qwen3.5 forward pass (Phase 4)\n\n' +
    'The GPU compute layer is ready. Kernel tests verify correct operation.'
  );
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

unloadBtn.addEventListener('click', () => {
  addMessage('system', 'Model unloaded (no model was loaded).');
  setStatus('Model unloaded');
});

// ─── Boot ────────────────────────────────────────────────────────────────────

init();
