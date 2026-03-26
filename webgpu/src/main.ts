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
import { setAuthToken, useLocalCache, resetToRemote } from './model/hf-hub';
import { createInferenceSession, type InferenceSession } from './engine/inference';
import { parseModelConfig, estimateVRAM, getWeightNameMap, resolveLayerWeightName } from './model/model-config';

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
    const maxNewTokens = parseInt(($('max-tokens') as HTMLInputElement).value) || 512;
    const useCompressedKV = ($('turboquant') as HTMLInputElement).checked;

    // /raw prefix: skip chat template, use raw text completion (for debugging)
    const isRaw = text.startsWith('/raw ');
    const sampling = { temperature, topP, maxNewTokens, useCompressedKV, repetitionPenalty: 1.5 };
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
    // Check if model is available in local HF cache (50-100x faster than CDN)
    let usingLocalCache = false;
    try {
      const cacheResp = await fetch('/api/hf-cache/models');
      if (cacheResp.ok) {
        const cached = await cacheResp.json() as Array<{ repo: string }>;
        if (cached.some(m => m.repo === repo)) {
          useLocalCache();
          usingLocalCache = true;
          addMessage('system', `Loading ${repo} from local HF cache (fast)...`);
        }
      }
    } catch { /* dev server not running, use CDN */ }

    if (!usingLocalCache) {
      addMessage('system', `Connecting to HuggingFace: ${repo}...`);
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

    // Decide whether to keep BF16 weights (halves VRAM for unquantized models)
    // Enable when f32 conversion would exceed ~6 GB (leave room for KV cache + intermediates)
    const isQuantized = preview.dtypes.includes('I32'); // GPTQ models have I32 packed weights
    const f32Estimate = preview.totalBytes * 2; // BF16→F32 roughly doubles size
    const VRAM_THRESHOLD = 6 * 1024 * 1024 * 1024; // 6 GB
    const keepBF16 = !isQuantized && f32Estimate > VRAM_THRESHOLD;
    console.log(`[Engine] keepBF16=${keepBF16}: f32 would need ${formatBytes(f32Estimate)}, threshold ${formatBytes(VRAM_THRESHOLD)}`);
    if (keepBF16) {
      console.log(`[Engine] Keeping BF16 weights native — halves VRAM usage`);
    }

    // Mixed-precision GPTQ: dequant linear_attn weights to BF16 for hybrid models
    // This prevents INT4 quantization noise from compounding in the SSM recurrence
    const dequantToBF16 = new Set<string>();
    if (isQuantized) {
      const previewConfig = parseModelConfig(preview.config);
      if (previewConfig.isHybrid && previewConfig.layerTypes) {
        // Add both possible prefixes (multimodal uses model.language_model.*, text-only uses model.*)
        const projSuffixes = [
          'linear_attn.in_proj_qkv', 'linear_attn.in_proj_a',
          'linear_attn.in_proj_b', 'linear_attn.in_proj_z', 'linear_attn.out_proj',
        ];
        const prefixes = ['model.language_model.layers', 'model.layers'];
        for (let l = 0; l < previewConfig.numLayers; l++) {
          if (previewConfig.layerTypes[l] === 'linear_attention') {
            for (const prefix of prefixes) {
              for (const suffix of projSuffixes) {
                dequantToBF16.add(`${prefix}.${l}.${suffix}`);
              }
            }
          }
        }
        if (dequantToBF16.size > 0) {
          console.log(`[Engine] Mixed-precision GPTQ: ${dequantToBF16.size / 2} linear_attn projections → BF16 (${dequantToBF16.size} name variants)`);
        }
      }
    }

    // Full load — download weights and upload to GPU
    currentModel = await loadModel(gpu.device, repo, (p) => {
      progressEl.textContent = p.message;
      if (p.overallProgress !== undefined) {
        const pct = Math.round(p.overallProgress * 100);
        setStatus(`Loading ${repo}... ${pct}%`);
      }
    }, keepBF16, dequantToBF16);

    // Reset to CDN for future loads (if we were using local cache)
    if (usingLocalCache) resetToRemote();

    addMessage('system',
      `Weights loaded: ${currentModel.tensorCount} tensors, ${formatBytes(currentModel.totalGPUBytes)} GPU memory\n` +
      `Load time: ${(currentModel.loadTimeMs / 1000).toFixed(1)}s` +
      (usingLocalCache ? ' (from local cache)' : ''),
      'weights loaded'
    );

    // Build inference session from loaded weights
    setStatus('Building inference engine...');
    try {
      const config = parseModelConfig(currentModel.config);

      // Import bridgeWeights and engine builder
      const { createForwardPassEngine } = await import('./engine/forward-pass');
      const { createTokenizer } = await import('./model/tokenizer');
      const { autoDetectWeightNameMap, resolveLayerWeightName } = await import('./model/model-config');
      const { generate } = await import('./engine/generate');

      // Bridge weight tensors to structured format (auto-detects prefix)
      const nameMap = autoDetectWeightNameMap(config.modelType, currentModel!.tensors);
      const getTensor = (name: string) => {
        const t = currentModel!.tensors.get(name);
        if (!t) throw new Error(`Missing tensor: ${name}`);
        return t.buffer;
      };

      // Auto-detect if embedding is stored as packed 16-bit (BF16/F16)
      // This happens when: (a) keepBF16=true and tensor is >1MB, or (b) f32 would exceed 2GB buffer limit
      const embedTensor = currentModel!.tensors.get(nameMap.embedTokens);
      const embedIsF16 = embedTensor ? (embedTensor.dtype === 'BF16' || embedTensor.dtype === 'F16') : false;
      if (embedIsF16) {
        console.log(`[Engine] Embedding is ${embedTensor!.dtype}, using packed-16 embed shader`);
      }

      const global = {
        embedTokens: getTensor(nameMap.embedTokens),
        embedIsF16,
        finalNorm: getTensor(nameMap.finalNorm),
        lmHead: config.tieWordEmbeddings
          ? getTensor(nameMap.embedTokens)
          : getTensor(nameMap.lmHead),
        lmHeadIsBF16: (() => {
          // Check the actual tensor used for LM head (may be embed_tokens if tied)
          const tensorName = config.tieWordEmbeddings ? nameMap.embedTokens : nameMap.lmHead;
          const t = currentModel!.tensors.get(tensorName);
          const isBF16 = t ? (t.dtype === 'BF16' || t.dtype === 'F16') : false;
          if (isBF16) console.log(`[Engine] LM head is ${t!.dtype}${config.tieWordEmbeddings ? ' (tied)' : ''}, using BF16 matmul`);
          return isBF16;
        })(),
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
        const isLinearLayer = config.layerTypes?.[l] === 'linear_attention';

        // Shared weights (both layer types)
        const gateName = resolveLayerWeightName(nameMap.layer.gateProj, l);
        const upName = resolveLayerWeightName(nameMap.layer.upProj, l);
        const downName = resolveLayerWeightName(nameMap.layer.downProj, l);
        const lw: any = {
          inputNorm: getTensor(resolveLayerWeightName(nameMap.layer.inputNorm, l)),
          postAttnNorm: getTensor(resolveLayerWeightName(nameMap.layer.postAttnNorm, l)),
          gateProj: tryGetTensor(gateName),
          upProj: tryGetTensor(upName),
          downProj: tryGetTensor(downName),
        };
        trackBF16(gateName, lw.gateProj);
        trackBF16(upName, lw.upProj);
        trackBF16(downName, lw.downProj);

        if (isLinearLayer && nameMap.linearLayer) {
          // ── Linear attention layer weights ──────────────────────────
          const lin = nameMap.linearLayer;
          const linNames = {
            qkv: resolveLayerWeightName(lin.inProjQKV, l),
            a: resolveLayerWeightName(lin.inProjA, l),
            b: resolveLayerWeightName(lin.inProjB, l),
            z: resolveLayerWeightName(lin.inProjZ, l),
            out: resolveLayerWeightName(lin.outProj, l),
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
          lw.linearALog = tryGetTensor(resolveLayerWeightName(lin.aLog, l));
          lw.linearConv1dWeight = tryGetTensor(resolveLayerWeightName(lin.conv1dWeight, l));
          lw.linearDtBias = tryGetTensor(resolveLayerWeightName(lin.dtBias, l));
          lw.linearNormWeight = tryGetTensor(resolveLayerWeightName(lin.normWeight, l));

          // GPTQ for linear attention projections
          if (config.isQuantized) {
            const linQ4Keys = ['linearInProjQKV', 'linearInProjA', 'linearInProjB', 'linearInProjZ', 'linearOutProj'] as const;
            const linNameKeys = ['inProjQKV', 'inProjA', 'inProjB', 'inProjZ', 'outProj'] as const;
            for (let k = 0; k < linQ4Keys.length; k++) {
              const weightName = resolveLayerWeightName(lin[linNameKeys[k]], l);
              const q4 = tryGetQ4(weightName);
              if (q4) {
                lw[`${linQ4Keys[k]}_q4`] = q4;
                if (l === 0) console.log(`[Q4] L0 ${linQ4Keys[k]}: GPTQ loaded`);
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
          const qName = resolveLayerWeightName(nameMap.layer.qProj, l);
          const kName = resolveLayerWeightName(nameMap.layer.kProj, l);
          const vName = resolveLayerWeightName(nameMap.layer.vProj, l);
          const oName = resolveLayerWeightName(nameMap.layer.oProj, l);
          lw.qProj = tryGetTensor(qName);
          lw.kProj = tryGetTensor(kName);
          lw.vProj = tryGetTensor(vName);
          lw.oProj = tryGetTensor(oName);
          trackBF16(qName, lw.qProj);
          trackBF16(kName, lw.kProj);
          trackBF16(vName, lw.vProj);
          trackBF16(oName, lw.oProj);

          // GPTQ for standard attention projections
          if (config.isQuantized) {
            const nameKeys = ['qProj', 'kProj', 'vProj', 'oProj'] as const;
            for (const key of nameKeys) {
              const weightName = resolveLayerWeightName(nameMap.layer[key], l);
              const q4 = tryGetQ4(weightName);
              if (q4) {
                lw[`${key}_q4`] = q4;
                if (l === 3) console.log(`[Q4] L3 ${key}: GPTQ loaded`); // L3 is first full_attn
              }
            }
          }

          // Bias terms (only for full attention layers with bias)
          if (config.attentionBias) {
            lw.qBias = tryGetTensor(resolveLayerWeightName(nameMap.layer.qBias, l));
            lw.kBias = tryGetTensor(resolveLayerWeightName(nameMap.layer.kBias, l));
            lw.vBias = tryGetTensor(resolveLayerWeightName(nameMap.layer.vBias, l));
            lw.oBias = tryGetTensor(resolveLayerWeightName(nameMap.layer.oBias, l));
          }

          // Q/K per-head RMSNorm (Qwen3.5 full attention)
          if (nameMap.layer.qNorm) {
            lw.qNorm = tryGetTensor(resolveLayerWeightName(nameMap.layer.qNorm, l));
            lw.kNorm = tryGetTensor(resolveLayerWeightName(nameMap.layer.kNorm!, l));
            if (l === 3) console.log(`[Engine] L3 qNorm: ${lw.qNorm ? 'FOUND' : 'MISSING'}, kNorm: ${lw.kNorm ? 'FOUND' : 'MISSING'}`);
          }
        }

        // FFN GPTQ (shared for both layer types)
        if (config.isQuantized) {
          for (const key of ['gateProj', 'upProj', 'downProj'] as const) {
            const weightName = resolveLayerWeightName(nameMap.layer[key], l);
            const q4 = tryGetQ4(weightName);
            if (q4) {
              lw[`${key}_q4`] = q4;
              if (l === 0) console.log(`[Q4] L0 ${key}: GPTQ loaded`);
            }
          }
        }

        layers.push(lw);
      }

      if (config.isHybrid) {
        const linCount = config.layerTypes!.filter(t => t === 'linear_attention').length;
        const fullCount = config.layerTypes!.filter(t => t === 'full_attention').length;
        console.log(`[Engine] Hybrid model: ${linCount} linear + ${fullCount} full attention layers`);
      }
      if (l0Keys.length > 0) console.log(`[Engine] attentionBias=${config.attentionBias}`);

      if (bf16Buffers.size > 0) {
        console.log(`[Engine] ${bf16Buffers.size} weight buffers kept in BF16 (using BF16 matmul kernel)`);
      } else {
        console.log(`[Engine] All weights in f32 (bf16Buffers empty)`);
      }
      const engine = createForwardPassEngine(gpu!.device, config, { global, layers, bf16Buffers });
      const tokenizer = await createTokenizer({ modelId: repo });

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

      addMessage('system',
        `Inference engine ready!\n` +
        `Model: ${config.modelType} — ${config.numLayers} layers, ${config.numAttentionHeads} heads, d=${config.hiddenSize}\n` +
        `GQA: ${config.isGQA ? `${config.numAttentionHeads}Q/${config.numKVHeads}KV` : 'no'}\n` +
        `Vocab: ${config.vocabSize} | RoPE θ=${config.ropeTheta}\n` +
        `Type a message to chat!`,
        'engine ready'
      );

      setStatus(`Ready: ${repo}`);

      // ── Auto-test polling: check /api/test for queued prompts ─────
      // Intercept console.log to capture debug output from forward pass
      const _origLog = console.log;
      let debugLogs: string[] = [];

      (async function pollTests() {
        while (session) {
          try {
            const resp = await fetch('/api/test');
            const test = await resp.json();
            if (test && test.prompt) {
              _origLog(`[AutoTest] Running: "${test.prompt}"`);

              // Enable debug mode for this run's first forward pass
              (globalThis as any).__DEBUG_FORWARD_PASS__ = true;

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
                { temperature: test.temperature ?? 0, topP: 0.9, maxNewTokens: test.maxTokens ?? 50, repetitionPenalty: test.repetitionPenalty ?? 1.15 },
                () => {},
              );
              const result = await handle.result;
              const elapsed = performance.now() - t0;

              // Restore console.log
              console.log = _origLog;

              const debugData = {
                prompt: test.prompt,
                output: result.text,
                tokens: result.numTokens,
                tokPerSec: result.tokensPerSecond,
                stopReason: result.stopReason,
                elapsedMs: Math.round(elapsed),
                promptTokens: result.promptTokens,
                consoleLogs: debugLogs,
              };
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
