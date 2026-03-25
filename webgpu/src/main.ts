/**
 * Artifex WebGPU Engine — Main Entry Point
 *
 * Initializes WebGPU, runs kernel tests, and sets up the chat UI.
 * Phase 0-1: device detection, compute foundation, UI shell.
 * Phase 2: SafeTensors weight loading from HuggingFace.
 */

import { initWebGPU, type GPUContext } from './engine/gpu-device';
import { reportMetric, reportError, timed } from './utils/metrics';
import { runKernelTests } from './engine/kernel-tests';
import { loadModel, unloadModel, previewModel, formatBytes, getCacheStats, clearCache, type LoadedModel } from './model';
import { setAuthToken } from './model/hf-hub';
import { createInferenceSession, type InferenceSession } from './engine/inference';
import { parseModelConfig, estimateVRAM } from './model/model-config';

// ─── State ───────────────────────────────────────────────────────────────────

let gpu: GPUContext | null = null;
let currentModel: LoadedModel | null = null;
let session: InferenceSession | null = null;

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

    const systemPrompt = ($('system-prompt') as HTMLTextAreaElement).value.trim();
    const messages: Array<{ role: string; content: string }> = [];
    if (systemPrompt) {
      messages.push({ role: 'system', content: systemPrompt });
    }
    messages.push({ role: 'user', content: text });

    const handle = session.chat(
      messages,
      { temperature, topP, maxNewTokens: 512 },
      (token) => {
        fullText += token;
        responseDiv.textContent = fullText;
        messagesEl.scrollTop = messagesEl.scrollHeight;
      },
    );

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

  try {
    // Preview first (just headers, fast)
    addMessage('system', `Connecting to HuggingFace: ${repo}...`);

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

    // Full load — download weights and upload to GPU
    currentModel = await loadModel(gpu.device, repo, (p) => {
      progressEl.textContent = p.message;
      if (p.overallProgress !== undefined) {
        const pct = Math.round(p.overallProgress * 100);
        setStatus(`Loading ${repo}... ${pct}%`);
      }
    });

    addMessage('system',
      `Weights loaded: ${currentModel.tensorCount} tensors, ${formatBytes(currentModel.totalGPUBytes)} GPU memory\n` +
      `Load time: ${(currentModel.loadTimeMs / 1000).toFixed(1)}s`,
      'weights loaded'
    );

    // Build inference session from loaded weights
    setStatus('Building inference engine...');
    try {
      const config = parseModelConfig(currentModel.config);

      // Import bridgeWeights and engine builder
      const { createForwardPassEngine } = await import('./engine/forward-pass');
      const { createTokenizer } = await import('./model/tokenizer');
      const { getWeightNameMap, resolveLayerWeightName } = await import('./model/model-config');
      const { generate } = await import('./engine/generate');

      // Bridge weight tensors to structured format
      const nameMap = getWeightNameMap(config.modelType);
      const getTensor = (name: string) => {
        const t = currentModel!.tensors.get(name);
        if (!t) throw new Error(`Missing tensor: ${name}`);
        return t.buffer;
      };

      // Auto-detect if embedding needs F16 (>2GB at f32 hits WebGPU buffer limit)
      const embedTensor = currentModel!.tensors.get(nameMap.embedTokens);
      const embedF32Size = embedTensor ? embedTensor.elementCount * 4 : 0;
      const embedIsF16 = embedF32Size > 1.9 * 1024 * 1024 * 1024; // >1.9 GB → use F16
      if (embedIsF16) {
        console.log(`[Engine] Embedding too large for f32 (${(embedF32Size / (1024**3)).toFixed(1)} GB), using F16`);
      }

      const global = {
        embedTokens: getTensor(nameMap.embedTokens),
        embedIsF16,
        finalNorm: getTensor(nameMap.finalNorm),
        lmHead: config.tieWordEmbeddings
          ? getTensor(nameMap.embedTokens)
          : getTensor(nameMap.lmHead),
      };

      // Helper for optional tensors (bias terms)
      const tryGetTensor = (name: string): GPUBuffer | undefined => {
        const t = currentModel!.tensors.get(name);
        return t?.buffer;
      };

      // Helper to load GPTQ triplet (qweight + scales + qzeros) for a projection
      const tryGetQ4 = (weightName: string) => {
        const base = weightName.replace('.weight', '');
        const qw = tryGetTensor(`${base}.qweight`);
        const sc = tryGetTensor(`${base}.scales`);
        const qz = tryGetTensor(`${base}.qzeros`);
        if (qw && sc && qz) return { qweight: qw, scales: sc, qzeros: qz };
        return undefined;
      };

      const layers = [];
      for (let l = 0; l < config.numLayers; l++) {
        // For quantized models, .weight tensors won't exist for linear layers.
        // Use tryGetTensor for projection weights (may be null if GPTQ).
        const lw: any = {
          inputNorm: getTensor(resolveLayerWeightName(nameMap.layer.inputNorm, l)),
          qProj: tryGetTensor(resolveLayerWeightName(nameMap.layer.qProj, l)),
          kProj: tryGetTensor(resolveLayerWeightName(nameMap.layer.kProj, l)),
          vProj: tryGetTensor(resolveLayerWeightName(nameMap.layer.vProj, l)),
          oProj: tryGetTensor(resolveLayerWeightName(nameMap.layer.oProj, l)),
          postAttnNorm: getTensor(resolveLayerWeightName(nameMap.layer.postAttnNorm, l)),
          gateProj: tryGetTensor(resolveLayerWeightName(nameMap.layer.gateProj, l)),
          upProj: tryGetTensor(resolveLayerWeightName(nameMap.layer.upProj, l)),
          downProj: tryGetTensor(resolveLayerWeightName(nameMap.layer.downProj, l)),
        };

        // Load GPTQ quantized weights if available
        if (config.isQuantized) {
          const projNames = ['qProj', 'kProj', 'vProj', 'oProj', 'gateProj', 'upProj', 'downProj'];
          const nameKeys = ['qProj', 'kProj', 'vProj', 'oProj', 'gateProj', 'upProj', 'downProj'] as const;
          for (const key of nameKeys) {
            const weightName = resolveLayerWeightName(nameMap.layer[key], l);
            const q4 = tryGetQ4(weightName);
            if (q4) {
              lw[`${key}_q4`] = q4;
              if (l === 0) console.log(`[Q4] L0 ${key}: GPTQ loaded`);
            }
          }
        }
        // Add bias terms if model has them
        if (l === 0) console.log(`[Engine] attentionBias=${config.attentionBias}, raw=${currentModel!.config.attention_bias}`);
        if (config.attentionBias) {
          const qBiasName = resolveLayerWeightName(nameMap.layer.qBias, l);
          const kBiasName = resolveLayerWeightName(nameMap.layer.kBias, l);
          const vBiasName = resolveLayerWeightName(nameMap.layer.vBias, l);
          const oBiasName = resolveLayerWeightName(nameMap.layer.oBias, l);
          lw.qBias = tryGetTensor(qBiasName);
          lw.kBias = tryGetTensor(kBiasName);
          lw.vBias = tryGetTensor(vBiasName);
          lw.oBias = tryGetTensor(oBiasName);
          if (l === 0) {
            console.log(`[Bias] L0 qBias=${qBiasName}: ${lw.qBias ? 'FOUND' : 'MISSING'}`);
            console.log(`[Bias] L0 kBias=${kBiasName}: ${lw.kBias ? 'FOUND' : 'MISSING'}`);
            console.log(`[Bias] L0 vBias=${vBiasName}: ${lw.vBias ? 'FOUND' : 'MISSING'}`);
            console.log(`[Bias] L0 oBias=${oBiasName}: ${lw.oBias ? 'FOUND' : 'MISSING'}`);
          }
        }
        layers.push(lw);
      }

      const engine = createForwardPassEngine(gpu!.device, config, { global, layers });
      const tokenizer = await createTokenizer({ modelId: repo });

      // Build chat template function
      const { applyChatTemplate } = await import('./model/tokenizer');

      // Store session for the send button
      session = {
        run: (prompt, sampling, onToken) => generate(gpu!.device, engine, tokenizer, prompt, sampling, onToken),
        chat: (messages, sampling, onToken) => {
          const tokenIds = applyChatTemplate(tokenizer, messages);
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

      addMessage('system',
        `Inference engine ready!\n` +
        `Model: ${config.modelType} — ${config.numLayers} layers, ${config.numAttentionHeads} heads, d=${config.hiddenSize}\n` +
        `GQA: ${config.isGQA ? `${config.numAttentionHeads}Q/${config.numKVHeads}KV` : 'no'}\n` +
        `Vocab: ${config.vocabSize} | RoPE θ=${config.ropeTheta}\n` +
        `Type a message to chat!`,
        'engine ready'
      );

      setStatus(`Ready: ${repo}`);
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
