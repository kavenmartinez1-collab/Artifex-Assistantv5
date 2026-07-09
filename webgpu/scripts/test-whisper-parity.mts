/**
 * Whisper-base.en parity gate — STT port.
 *
 * Node-side (no GPU), grows per phase:
 *   W1 mel : logMelSpectrogram vs whisper_fixture/<clip>.mel (relL2 ≤ 1e-3)
 *   W2/W3  : encoder / decoder taps (added as those phases land)
 *
 * Fixtures are HF WhisperFeatureExtractor / model dumps from
 * gen_whisper_fixture.py; the audio samples are injected from the fixture so the
 * frontend math is isolated from any resampling/IO.
 *
 * Run: npx tsx scripts/test-whisper-parity.mts
 */
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { parseHeader, extractTensorData, tensorToFloat32 } from '../src/model/safetensors';
import { logMelSpectrogram, encode, decode, greedyDecode, N_MELS, N_FREQ, type WhisperWeights } from '../src/audio/whisper';

const here = dirname(fileURLToPath(import.meta.url));
const fixDir = resolve(here, 'whisper_fixture');
const modelDir = resolve(here, '../models/whisper-base-en');

if (!existsSync(resolve(fixDir, 'manifest.json'))) {
  console.error('no whisper_fixture/manifest.json — run gen_whisper_fixture.py first');
  process.exit(2);
}
const manifest = JSON.parse(readFileSync(resolve(fixDir, 'manifest.json'), 'utf8'));

function loadBin(file: string): Float32Array {
  const raw = readFileSync(resolve(fixDir, file));
  return new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
}
function loadFix(name: string): Float32Array {
  return loadBin(manifest.tensors[name].file);
}
function loadMelFilters(): Float32Array {
  const raw = readFileSync(resolve(modelDir, 'mel_filters.bin'));
  const f = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  if (f.length !== N_MELS * N_FREQ) throw new Error(`mel LUT len ${f.length} != ${N_MELS * N_FREQ}`);
  return f;
}
function loadWeights(): WhisperWeights {
  const buf = readFileSync(resolve(modelDir, 'model.safetensors'));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
  const header = parseHeader(ab);
  const w: WhisperWeights = new Map();
  for (const [name, info] of header.tensors) {
    const raw = extractTensorData(ab, info, header.headerByteLength);
    w.set(name, { shape: info.shape, data: tensorToFloat32(raw, info.dtype) });
  }
  return w;
}

function relL2(got: Float32Array, want: Float32Array): number {
  if (got.length !== want.length) return NaN;
  let num = 0, den = 0;
  for (let i = 0; i < want.length; i++) {
    const d = got[i] - want[i];
    num += d * d;
    den += want[i] * want[i];
  }
  return Math.sqrt(num) / (Math.sqrt(den) || 1);
}

let failed = 0;
const report = (name: string, err: number, tol: number) => {
  const ok = err <= tol && Number.isFinite(err);
  if (!ok) failed++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'} ${name.padEnd(20)} relL2 ${err.toExponential(3)} (tol ${tol.toExponential(0)})`);
};

const melFilters = loadMelFilters();
const clips = Object.keys(manifest.tensors)
  .filter((n) => n.endsWith('.mel'))
  .map((n) => n.slice(0, -'.mel'.length));

console.log('W1 mel: logMelSpectrogram vs HF WhisperFeatureExtractor (injected audio)');
const mels: Record<string, Float32Array> = {};
for (const c of clips) {
  const audio = loadFix(`${c}.audio`);
  const got = logMelSpectrogram(audio, melFilters);
  mels[c] = got;
  const want = loadFix(`${c}.mel`);
  console.log(`  ${c}: ${audio.length} samples → mel ${got.length} (fixture ${want.length})`);
  report(`${c}.mel`, relL2(got, want), 1e-3);
}

console.log('\nW2 encoder: encode(mel) vs HF model.encoder.last_hidden_state');
const w = loadWeights();
console.log(`  loaded ${w.size} whisper tensors`);
const encs: Record<string, Float32Array> = {};
for (const c of clips) {
  const got = encode(mels[c], w);
  encs[c] = got;
  const want = loadFix(`${c}.encoder`);
  console.log(`  ${c}: encoder ${got.length} (fixture ${want.length})`);
  report(`${c}.encoder`, relL2(got, want), 2e-3);
}

console.log('\nW3 decoder: decode(prefix, enc) logits vs HF proj_out(model.decoder(...))');
for (const c of clips) {
  const prefix: number[] = manifest.meta[`${c}.dec_prefix`];
  const got = decode(prefix, encs[c], w);
  const want = loadFix(`${c}.dec_logits`);
  console.log(`  ${c}: prefix ${JSON.stringify(prefix)} → logits ${got.length} (fixture ${want.length})`);
  report(`${c}.dec_logits`, relL2(got, want), 2e-3);
}

console.log('\nW4 greedy: greedyDecode(enc) generated tail vs HF generate() tokens (EXACT)');
const gcfg = JSON.parse(readFileSync(resolve(modelDir, 'generation_config.json'), 'utf8'));
const forcedPrefix = [gcfg.decoder_start_token_id, ...gcfg.forced_decoder_ids.map((p: number[]) => p[1])];
for (const c of clips) {
  const want: number[] = manifest.meta[`${c}.gen_tokens`];
  const full = greedyDecode(encs[c], w, {
    forcedPrefix,
    suppress: gcfg.suppress_tokens,
    beginSuppress: gcfg.begin_suppress_tokens,
    eosTokenId: gcfg.eos_token_id,
    maxNewTokens: 64,
  });
  const tail = full.slice(forcedPrefix.length);
  const ok = tail.length === want.length && tail.every((t, i) => t === want[i]);
  if (!ok) failed++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'} ${`${c}.gen_tokens`.padEnd(20)} got ${JSON.stringify(tail)} want ${JSON.stringify(want)}`);
}

console.log(failed ? `\n${failed} FAILED` : '\nall PASS');
process.exit(failed ? 1 : 0);
