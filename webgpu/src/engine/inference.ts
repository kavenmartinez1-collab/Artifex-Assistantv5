/**
 * Inference — Top-level orchestrator that wires everything together.
 *
 * This is the single entry point for running a model in the browser:
 *   1. Initialize WebGPU
 *   2. Load tokenizer from HuggingFace
 *   3. Load model weights (SafeTensors → GPU buffers)
 *   4. Parse model config → build forward pass engine
 *   5. Accept prompts → generate text
 *
 * Usage:
 *   const session = await createInferenceSession({
 *     modelId: 'Qwen/Qwen3.5-0.6B',
 *     onProgress: (msg) => console.log(msg),
 *   });
 *   const result = await session.run('What is 2+2?');
 *   console.log(result.text);
 */

import { initWebGPU, type GPUContext } from './gpu-device';
import { createForwardPassEngine, MAX_ATTN_SEQ_LEN, type ModelWeights, type LayerWeights, type GlobalWeights } from './forward-pass';
import { generate, createKVSession, type KVSessionState, type SamplingConfig, type GenerationResult, type GenerationHandle, type OnTokenCallback } from './generate';
import { loadModel, unloadModel, type LoadedModel, type LoadProgress } from '../model/weight-loader';
import { createTokenizer, applyChatTemplate, type Tokenizer } from '../model/tokenizer';
import { getWeightNameMap, resolveLayerWeightName, estimateVRAM, type ModelConfig } from '../model/model-config';
import { descriptorFromHFConfig, type ModelDescriptor } from '../model/model-descriptor';
import type { GPUTensor } from '../model/weight-loader';

// ── Types ────────────────────────────────────────────────────────────────

export interface InferenceSessionConfig {
  /** HuggingFace model ID (e.g., 'Qwen/Qwen3.5-0.6B') */
  modelId: string;
  /** Progress callback for model loading */
  onProgress?: (progress: LoadProgress) => void;
  /** Status messages for UI */
  onStatus?: (message: string) => void;
}

export interface InferenceSession {
  /** Generate text from a plain prompt. */
  run(prompt: string, sampling?: SamplingConfig, onToken?: OnTokenCallback): GenerationHandle;

  /** Generate text from chat messages (applies chat template). Multi-turn
   *  calls share the session's persistent KV cache (prefix reuse). */
  chat(
    messages: Array<{ role: string; content: string }>,
    sampling?: SamplingConfig,
    onToken?: OnTokenCallback,
    opts?: { enableThinking?: boolean },
  ): GenerationHandle;

  /** Persistent KV session backing chat() turns. */
  readonly kvSession: KVSessionState;

  /** Drop the multi-turn KV cache (start a new conversation). */
  resetKV(): void;

  /** Get the model descriptor. */
  readonly config: ModelDescriptor;

  /** Get the tokenizer. */
  readonly tokenizer: Tokenizer;

  /** Get GPU context info. */
  readonly gpu: GPUContext;

  /** VRAM usage estimate. */
  readonly vramEstimate: ReturnType<typeof estimateVRAM>;

  /** Unload the model and free all GPU memory. */
  destroy(): void;
}

// ── Bridge: LoadedModel tensors → ModelWeights ───────────────────────────

/**
 * Maps the flat tensor map from the weight loader to the structured
 * ModelWeights interface expected by the forward pass engine.
 *
 * Uses the weight name map from model-config.ts to find tensors by name.
 * Throws if any required tensor is missing.
 */
function bridgeWeights(
  tensors: Map<string, GPUTensor>,
  config: ModelConfig,
): ModelWeights {
  const nameMap = getWeightNameMap(config.modelType);

  function getTensor(name: string): GPUBuffer {
    const tensor = tensors.get(name);
    if (!tensor) {
      throw new Error(`Missing weight tensor: "${name}". Available: ${[...tensors.keys()].slice(0, 10).join(', ')}...`);
    }
    return tensor.buffer;
  }

  // Global weights
  const global: GlobalWeights = {
    embedTokens: getTensor(nameMap.embedTokens),
    finalNorm: getTensor(nameMap.finalNorm),
    lmHead: config.tieWordEmbeddings
      ? getTensor(nameMap.embedTokens)  // shared with embedding
      : getTensor(nameMap.lmHead),
  };

  // Per-layer weights
  const layers: LayerWeights[] = [];
  for (let l = 0; l < config.numLayers; l++) {
    layers.push({
      inputNorm: getTensor(resolveLayerWeightName(nameMap.layer.inputNorm, l)),
      qProj: getTensor(resolveLayerWeightName(nameMap.layer.qProj, l)),
      kProj: getTensor(resolveLayerWeightName(nameMap.layer.kProj, l)),
      vProj: getTensor(resolveLayerWeightName(nameMap.layer.vProj, l)),
      oProj: getTensor(resolveLayerWeightName(nameMap.layer.oProj, l)),
      postAttnNorm: getTensor(resolveLayerWeightName(nameMap.layer.postAttnNorm, l)),
      gateProj: getTensor(resolveLayerWeightName(nameMap.layer.gateProj, l)),
      upProj: getTensor(resolveLayerWeightName(nameMap.layer.upProj, l)),
      downProj: getTensor(resolveLayerWeightName(nameMap.layer.downProj, l)),
    });
  }

  return { global, layers };
}

// ── Inference Session ────────────────────────────────────────────────────

/**
 * Create a full inference session — loads model, creates engine, ready to generate.
 */
export async function createInferenceSession(
  sessionConfig: InferenceSessionConfig,
): Promise<InferenceSession> {
  const { modelId, onProgress, onStatus } = sessionConfig;

  const status = (msg: string) => {
    if (onStatus) onStatus(msg);
    console.log(`[Inference] ${msg}`);
  };

  // ── Step 1: Initialize WebGPU ──────────────────────────────────────
  status('Initializing WebGPU...');
  const gpu = await initWebGPU();
  status(`GPU: ${gpu.adapterInfo.device || gpu.adapterInfo.description || 'unknown'} — ${Math.round(gpu.maxBufferSize / (1024 * 1024))} MB max buffer`);

  // ── Step 2: Load tokenizer ─────────────────────────────────────────
  status(`Loading tokenizer for ${modelId}...`);
  const tokenizer = await createTokenizer({ modelId });
  status(`Tokenizer loaded — vocab size: ${tokenizer.vocabSize}`);

  // ── Step 3: Load model weights ─────────────────────────────────────
  status(`Loading model weights for ${modelId}...`);
  const loadedModel = await loadModel(gpu.device, modelId, onProgress);
  status(`Model loaded: ${loadedModel.tensorCount} tensors, ${Math.round(loadedModel.totalGPUBytes / (1024 * 1024))} MB GPU memory`);

  // ── Step 4: Parse config and build engine ──────────────────────────
  status('Building inference engine...');
  const config = descriptorFromHFConfig(loadedModel.config);
  const modelWeights = bridgeWeights(loadedModel.tensors, config);
  const engine = createForwardPassEngine(gpu.device, config, modelWeights);
  const vramEstimate = estimateVRAM(config);
  status(`Engine ready — ${config.modelType} ${config.numLayers}L/${config.numAttentionHeads}H/${config.hiddenSize}D`);

  // ── API ────────────────────────────────────────────────────────────

  // Persistent multi-turn KV session — sized to the model's context (capped;
  // generate() grows it on demand). Chat turns that extend the previous token
  // sequence prefill only the delta.
  const kvSession = createKVSession(Math.min(config.maxPositionEmbeddings || 8192, 8192, MAX_ATTN_SEQ_LEN));

  function resetKV() {
    if (kvSession.kvCache) engine.destroyKVCache(kvSession.kvCache);
    kvSession.kvCache = null;
    kvSession.cachedTokenIds = [];
  }

  function run(
    prompt: string,
    sampling?: SamplingConfig,
    onToken?: OnTokenCallback,
  ): GenerationHandle {
    return generate(gpu.device, engine, tokenizer, prompt, sampling, onToken);
  }

  function chat(
    messages: Array<{ role: string; content: string }>,
    sampling?: SamplingConfig,
    onToken?: OnTokenCallback,
    opts?: { enableThinking?: boolean },
  ): GenerationHandle {
    const prompt = applyChatTemplate(tokenizer, messages, opts);
    return generate(gpu.device, engine, tokenizer, prompt, sampling, onToken, { kvSession });
  }

  function destroy() {
    resetKV();
    unloadModel(loadedModel);
    status('Model unloaded — GPU memory freed');
  }

  return {
    run,
    chat,
    kvSession,
    resetKV,
    config,
    tokenizer,
    gpu,
    vramEstimate,
    destroy,
  };
}
