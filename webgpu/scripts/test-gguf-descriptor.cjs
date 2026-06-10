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

const { parseGGUF } = require('../.tmp-cjs/gguf.js');
const { descriptorFromGGUF } = require('../.tmp-cjs/model-descriptor.js');
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

/** Every tensor must be reachable through the locator (completeness check). */
function coverageCheck(label, file, desc) {
  const loc = createGGUFLocator(file.tensors);
  const roles = [
    'embedTokens', 'finalNorm', 'lmHead',
    'inputNorm', 'postAttnNorm',
    'qProj', 'kProj', 'vProj', 'oProj', 'qBias', 'kBias', 'vBias', 'oBias', 'qNorm', 'kNorm',
    'gateProj', 'upProj', 'downProj',
    'linInProjQKV', 'linInProjA', 'linInProjB', 'linInProjZ', 'linOutProj',
    'linALog', 'linConv1dWeight', 'linDtBias', 'linNormWeight',
    'moeRouter', 'moeExpertGate', 'moeExpertUp', 'moeExpertDown',
    'moeSharedGateProj', 'moeSharedUpProj', 'moeSharedDownProj', 'moeSharedExpertGate',
  ];
  const located = new Set();
  for (const role of ['embedTokens', 'finalNorm', 'lmHead']) {
    const n = loc.locate(role);
    if (n) located.add(n);
  }
  for (let l = 0; l < desc.numLayers; l++) {
    for (const role of roles.slice(3)) {
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

  if (failures > 0) {
    console.error(`\n${failures} FAILURE(S)`);
    process.exit(1);
  }
  console.log('\nALL DESCRIPTOR/LOCATOR CHECKS PASSED');
})();
