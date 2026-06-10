/**
 * Validate descriptorFromGGUF + createGGUFLocator against the real GGUF files.
 *
 * Run (from webgpu/):
 *   npx tsc src/model/gguf.ts src/model/model-config.ts src/model/model-descriptor.ts \
 *       src/model/tensor-locator.ts --module commonjs --target es2022 --skipLibCheck --outDir .tmp-cjs
 *   node scripts/test-gguf-descriptor.cjs
 */
const { open } = require('node:fs/promises');
const path = require('node:path');

const { parseGGUF, ggufArchitecture } = require('../.tmp-cjs/gguf.js');
const { descriptorFromGGUF, applyRopeFreqFactors } = require('../.tmp-cjs/model-descriptor.js');
const { createGGUFLocator } = require('../.tmp-cjs/tensor-locator.js');

let failures = 0;
function check(cond, msg) {
  if (!cond) { console.error(`  FAIL: ${msg}`); failures++; }
}
function eq(got, want, msg) {
  check(JSON.stringify(got) === JSON.stringify(want), `${msg}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
}

async function parseFile(p) {
  const fh = await open(p, 'r');
  const readRange = async (start, end) => {
    const len = end - start;
    const buf = Buffer.alloc(len);
    const { bytesRead } = await fh.read(buf, 0, len, start);
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + bytesRead);
  };
  const file = await parseGGUF(readRange);
  await fh.close();
  return file;
}

/** Read a tensor's raw bytes from the file (absolute offset from the header). */
async function readTensorBytes(p, file, name) {
  const t = file.tensors.get(name);
  if (!t) throw new Error(`tensor "${name}" not in file`);
  const fh = await open(p, 'r');
  const buf = Buffer.alloc(t.byteLength);
  const { bytesRead } = await fh.read(buf, 0, t.byteLength, t.offset);
  await fh.close();
  if (bytesRead !== t.byteLength) throw new Error(`short read for ${name}`);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + t.byteLength);
}

/** Every tensor must be reachable through the locator (completeness check). */
function coverageCheck(label, file, desc) {
  const loc = createGGUFLocator(file.tensors, ggufArchitecture(file));
  const globalRoles = [
    'embedTokens', 'finalNorm', 'lmHead',
    'pleTokenEmbed', 'pleModelProj', 'pleProjNorm', 'ropeFreqs',
  ];
  const layerRoles = [
    'inputNorm', 'postAttnNorm', 'attnPostNorm', 'ffnPostNorm', 'layerOutScale',
    'qProj', 'kProj', 'vProj', 'oProj', 'qBias', 'kBias', 'vBias', 'oBias', 'qNorm', 'kNorm',
    'gateProj', 'upProj', 'downProj',
    'linInProjQKV', 'linInProjA', 'linInProjB', 'linInProjZ', 'linOutProj',
    'linALog', 'linConv1dWeight', 'linDtBias', 'linNormWeight',
    'moeRouter', 'moeExpertGate', 'moeExpertUp', 'moeExpertDown',
    'moeSharedGateProj', 'moeSharedUpProj', 'moeSharedDownProj', 'moeSharedExpertGate',
    'pleInpGate', 'pleProj', 'plePostNorm',
  ];
  const located = new Set();
  for (const role of globalRoles) {
    const n = loc.locate(role);
    if (n) located.add(n);
  }
  for (let l = 0; l < desc.numLayers; l++) {
    for (const role of layerRoles) {
      const n = loc.locate(role, l);
      if (n) {
        check(file.tensors.has(n), `${label}: locate(${role},${l}) -> "${n}" not in file`);
        located.add(n);
      }
    }
  }
  const unreached = [...file.tensors.keys()].filter(n => !located.has(n));
  check(unreached.length === 0, `${label}: ${unreached.length} tensors unreachable via locator: ${unreached.slice(0, 10).join(', ')}`);
  console.log(`${label}: locator coverage ${file.tensors.size - unreached.length}/${file.tensors.size} tensors`);
}

(async () => {
  // ── 35B MoE ──────────────────────────────────────────────────────────
  const f35 = await parseFile(path.join(__dirname, '../../models/qwen3.6-35b-a3b-gguf/Qwen3.6-35B-A3B-UD-Q5_K_S.gguf'));
  const d35 = descriptorFromGGUF(f35);
  eq(d35.modelType, 'qwen3_5_moe', '35B modelType');
  eq(d35.numLayers, 40, '35B numLayers');
  eq(d35.hiddenSize, 2048, '35B hiddenSize');
  eq(d35.numAttentionHeads, 16, '35B heads');
  eq(d35.numKVHeads, 2, '35B kvHeads');
  eq(d35.headDim, 256, '35B headDim');
  eq(d35.vocabSize, 248320, '35B vocabSize');
  eq(d35.isHybrid, true, '35B isHybrid');
  eq(d35.tieWordEmbeddings, false, '35B tied');
  eq(d35.attentionBias, false, '35B attnBias');
  eq(d35.partialRotaryFactor, 0.25, '35B partialRotary');
  eq(d35.ropeTheta, 10000000, '35B ropeTheta');
  eq(d35.attnOutputGate, true, '35B attnOutputGate');
  eq(d35.linearKeyHeadDim, 128, '35B linKeyDim');
  eq(d35.linearValueHeadDim, 128, '35B linValDim');
  eq(d35.linearNumKeyHeads, 16, '35B linKeyHeads');
  eq(d35.linearNumValueHeads, 32, '35B linValHeads');
  eq(d35.linearConvKernelDim, 4, '35B convKernel');
  eq(d35.sourceFormat, 'gguf', '35B sourceFormat');
  eq(d35.layers.length, 40, '35B layers.length');
  // layer kinds: full attention at l % 4 == 3
  for (let l = 0; l < 40; l++) {
    const want = (l + 1) % 4 === 0 ? 'full_attention' : 'linear_attention';
    if (d35.layers[l].kind !== want) check(false, `35B layer ${l} kind ${d35.layers[l].kind} != ${want}`);
  }
  // MoE spec on every layer
  const moe = d35.layers[0].moe;
  check(!!moe, '35B layer 0 has moe spec');
  if (moe) {
    eq(moe.numExperts, 256, '35B numExperts');
    eq(moe.numExpertsPerToken, 8, '35B topK');
    eq(moe.expertFFNDim, 512, '35B expertFFN');
    eq(moe.sharedExpertFFNDim, 512, '35B sharedFFN');
    eq(moe.sharedExpertGate, 'sigmoid', '35B sharedGate');
  }
  eq(d35.layers[0].rope?.dimensionSections, [11, 11, 10, 0], '35B mrope sections');
  coverageCheck('35B', f35, d35);

  // ── 9B dense hybrid ─────────────────────────────────────────────────
  const f9 = await parseFile(path.join(__dirname, '../../models/qwen3.5-9b-abliterated-gguf/Huihui-Qwen3.5-9B-abliterated.i1-Q4_K_M.gguf'));
  const d9 = descriptorFromGGUF(f9);
  eq(d9.modelType, 'qwen3_5', '9B modelType');
  eq(d9.numLayers, 32, '9B numLayers');
  eq(d9.hiddenSize, 4096, '9B hiddenSize');
  eq(d9.intermediateSize, 12288, '9B ffnDim');
  eq(d9.numAttentionHeads, 16, '9B heads');
  eq(d9.numKVHeads, 4, '9B kvHeads');
  eq(d9.headDim, 256, '9B headDim');
  eq(d9.isHybrid, true, '9B isHybrid');
  eq(d9.tieWordEmbeddings, false, '9B tied');
  eq(d9.layers[3].kind, 'full_attention', '9B layer 3 kind');
  eq(d9.layers[0].kind, 'linear_attention', '9B layer 0 kind');
  check(d9.layers[0].moe === undefined, '9B has no moe spec');
  coverageCheck('9B', f9, d9);

  // ── Gemma 4 E4B ──────────────────────────────────────────────────────
  const p4 = path.join(__dirname, '../../models/gemma-4-e4b-it-gguf/gemma-4-E4B-it-Q4_K_M.gguf');
  const f4 = await parseFile(p4);
  const d4 = descriptorFromGGUF(f4);
  eq(d4.modelType, 'gemma4_text', 'E4B modelType');
  eq(d4.numLayers, 42, 'E4B numLayers');
  eq(d4.hiddenSize, 2560, 'E4B hiddenSize');
  eq(d4.intermediateSize, 10240, 'E4B ffnDim');
  eq(d4.numAttentionHeads, 8, 'E4B heads');
  eq(d4.numKVHeads, 2, 'E4B kvHeads');
  eq(d4.headDim, 512, 'E4B headDim (full layers)');
  eq(d4.vocabSize, 262144, 'E4B vocabSize');
  eq(d4.isHybrid, false, 'E4B isHybrid');
  eq(d4.tieWordEmbeddings, true, 'E4B tied');
  eq(d4.attentionBias, false, 'E4B attnBias');
  eq(d4.activation, 'gelu_tanh', 'E4B activation');
  eq(d4.finalLogitSoftcap, 30, 'E4B finalLogitSoftcap');
  eq(d4.perLayerEmbed, true, 'E4B perLayerEmbed');
  eq(d4.perLayerEmbedDim, 256, 'E4B perLayerEmbedDim');
  eq(d4.embedScale, Math.sqrt(2560), 'E4B embedScale');
  eq(d4.maxPositionEmbeddings, 131072, 'E4B maxPos');
  eq(d4.ropeTheta, 1000000, 'E4B ropeTheta (full)');
  eq(d4.layers.length, 42, 'E4B layers.length');

  // SSSSSF ×7: full attention at 5,11,17,23,29,35,41; rest sliding
  const FULL4 = new Set([5, 11, 17, 23, 29, 35, 41]);
  for (let l = 0; l < 42; l++) {
    const layer = d4.layers[l];
    const full = FULL4.has(l);
    eq(layer.kind, full ? 'full_attention' : 'sliding_attention', `E4B layer ${l} kind`);
    eq(layer.headDim, full ? 512 : 256, `E4B layer ${l} headDim`);
    eq(layer.rope?.theta, full ? 1000000 : 10000, `E4B layer ${l} rope theta`);
    eq(layer.slidingWindow, full ? undefined : 512, `E4B layer ${l} slidingWindow`);
    // KV sharing: layers 24-41 read L22 (sliding) / L23 (full)
    const wantSrc = l < 24 ? undefined : (full ? 23 : 22);
    eq(layer.kvSourceLayer, wantSrc, `E4B layer ${l} kvSourceLayer`);
  }

  // rope_freqs tensor data → rotatedPairs on full layers only
  const freqRaw = await readTensorBytes(p4, f4, 'rope_freqs.weight');
  const freqs = new Float32Array(freqRaw);
  eq(freqs.length, 256, 'E4B rope_freqs length');
  applyRopeFreqFactors(d4, freqs);
  for (let l = 0; l < 42; l++) {
    const want = FULL4.has(l) ? 64 : undefined;
    eq(d4.layers[l].rope?.rotatedPairs, want, `E4B layer ${l} rotatedPairs`);
  }

  // Arch-aware norm-name override: gemma4's pre-FFN norm is ffn_norm, and
  // post_attention_norm is the sandwich post-attn norm (Qwen3.5 is opposite).
  const loc4 = createGGUFLocator(f4.tensors, ggufArchitecture(f4));
  eq(loc4.locate('postAttnNorm', 0), 'blk.0.ffn_norm.weight', 'E4B postAttnNorm (pre-FFN)');
  eq(loc4.locate('attnPostNorm', 0), 'blk.0.post_attention_norm.weight', 'E4B attnPostNorm (sandwich)');
  eq(loc4.locate('ffnPostNorm', 0), 'blk.0.post_ffw_norm.weight', 'E4B ffnPostNorm');
  eq(loc4.locate('layerOutScale', 0), 'blk.0.layer_output_scale.weight', 'E4B layerOutScale');
  eq(loc4.locate('pleInpGate', 0), 'blk.0.inp_gate.weight', 'E4B pleInpGate');
  eq(loc4.locate('pleProj', 0), 'blk.0.proj.weight', 'E4B pleProj');
  eq(loc4.locate('plePostNorm', 0), 'blk.0.post_norm.weight', 'E4B plePostNorm');
  eq(loc4.locate('pleTokenEmbed'), 'per_layer_token_embd.weight', 'E4B pleTokenEmbed');
  eq(loc4.locate('pleModelProj'), 'per_layer_model_proj.weight', 'E4B pleModelProj');
  eq(loc4.locate('pleProjNorm'), 'per_layer_proj_norm.weight', 'E4B pleProjNorm');
  eq(loc4.locate('ropeFreqs'), 'rope_freqs.weight', 'E4B ropeFreqs');
  eq(loc4.locate('lmHead'), 'token_embd.weight', 'E4B lmHead tied');
  // Qwen 9B (no override): post_attention_norm IS the pre-FFN norm
  const loc9 = createGGUFLocator(f9.tensors, ggufArchitecture(f9));
  eq(loc9.locate('postAttnNorm', 3), 'blk.3.post_attention_norm.weight', '9B postAttnNorm default chain');
  eq(loc9.locate('attnPostNorm', 3), undefined, '9B attnPostNorm absent');

  coverageCheck('E4B', f4, d4);

  if (failures > 0) {
    console.error(`\n${failures} FAILURE(S)`);
    process.exit(1);
  }
  console.log('\nALL DESCRIPTOR/LOCATOR CHECKS PASSED');
})();
