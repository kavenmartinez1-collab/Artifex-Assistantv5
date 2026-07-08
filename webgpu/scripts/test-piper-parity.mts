/**
 * Piper VITS parity gate — Phase P6.
 *
 * Node-side (no GPU), grows per module as P6 lands:
 *   - enc_p: m_p / logs_p vs scripts/piper_fixture/s{0,1}.{m_p,logs_p} (relL2 ≤ 5e-4)
 *
 * Fixtures are the zero-noise (scales=[0,1,0]) onnxruntime dumps from
 * gen_piper_fixture.py; phoneme ids are injected from the manifest to isolate
 * the VITS math from G2P.
 *
 * Run: npx tsx scripts/test-piper-parity.mts
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { parseHeader, extractTensorData, tensorToFloat32 } from '../src/model/safetensors';
import { encP, expandByDuration, flowReverse, dpReverse, decForward, synthesize, type PiperWeights } from '../src/audio/piper';

const here = dirname(fileURLToPath(import.meta.url));
const fixDir = resolve(here, 'piper_fixture');
const modelPath = resolve(here, '../models/piper-en-us-joe-medium/model.safetensors');
const manifest = JSON.parse(readFileSync(resolve(fixDir, 'manifest.json'), 'utf8'));

// ── load the piper safetensors (all f32) into a PiperWeights map ──
function loadWeights(): PiperWeights {
  const buf = readFileSync(modelPath);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
  const header = parseHeader(ab);
  const w: PiperWeights = new Map();
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

function loadFix(name: string): Float32Array {
  const raw = readFileSync(resolve(fixDir, manifest.tensors[name].file));
  return new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
}

let failed = 0;
const report = (name: string, err: number, tol: number) => {
  const ok = err <= tol && Number.isFinite(err);
  if (!ok) failed++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'} ${name.padEnd(20)} relL2 ${err.toExponential(3)} (tol ${tol.toExponential(0)})`);
};

const w = loadWeights();
console.log(`loaded ${w.size} piper tensors\n`);

console.log('enc_p: m_p / logs_p vs fixture (injected phoneme ids)');
for (const s of ['s0', 's1']) {
  const ids: number[] = manifest.meta[`${s}.ids`];
  const out = encP(ids, w);
  const mp = loadFix(`${s}.m_p`);
  const logsp = loadFix(`${s}.logs_p`);
  console.log(`  ${s}: T=${out.T} (fixture ${manifest.tensors[`${s}.m_p`].shape.join('×')})`);
  report(`${s}.m_p`, relL2(out.m, mp), 5e-4);
  report(`${s}.logs_p`, relL2(out.logs, logsp), 5e-4);
}

console.log('\ndp: stochastic duration predictor reverse vs fixture durations (EXACT)');
for (const s of ['s0', 's1']) {
  const out = encP(manifest.meta[`${s}.ids`], w);
  const durF = loadFix(`${s}.durations`);
  const want = Array.from(durF, (d) => Math.round(d));
  const got = dpReverse(out.x, out.T, w);
  let mismatch = 0;
  for (let i = 0; i < want.length; i++) if (got[i] !== want[i]) mismatch++;
  const ok = mismatch === 0 && got.length === want.length;
  if (!ok) failed++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'} ${`${s}.durations`.padEnd(20)} ${mismatch}/${want.length} mismatched (sum got ${got.reduce((a, b) => a + b, 0)} want ${want.reduce((a, b) => a + b, 0)})`);
}

console.log('\nflow: reverse decoder vs fixture z (zero-noise ⇒ z_p = expand(m_p))');
for (const s of ['s0', 's1']) {
  const out = encP(manifest.meta[`${s}.ids`], w);
  const durF = loadFix(`${s}.durations`);
  const dur = Array.from(durF, (d) => Math.round(d));
  // zero-noise: z_p = expand(m_p, durations); logs contributes 0 (noise_scale=0)
  const { data: zp, F } = expandByDuration(out.m, 192, out.T, dur);
  const z = flowReverse(zp, F, w);
  const zFix = loadFix(`${s}.z`);
  console.log(`  ${s}: F=${F} (fixture ${manifest.tensors[`${s}.z`].shape.join('×')})`);
  report(`${s}.z`, relL2(z, zFix), 5e-4);
}

console.log('\ndec: HiFiGAN generator (CPU ref) vs fixture waveform (full pipeline)');
for (const s of ['s0', 's1']) {
  const out = encP(manifest.meta[`${s}.ids`], w);
  const durF = loadFix(`${s}.durations`);
  const dur = Array.from(durF, (d) => Math.round(d));
  const { data: zp, F } = expandByDuration(out.m, 192, out.T, dur);
  const z = flowReverse(zp, F, w);
  const wav = decForward(z, F, w);
  const wavFix = loadFix(`${s}.waveform`);
  console.log(`  ${s}: samples=${wav.length} (fixture ${wavFix.length})`);
  report(`${s}.waveform`, relL2(wav, wavFix), 1e-3);
}

console.log('\nsynthesize(): public API, zero-noise ⇒ fixture waveform (full wiring)');
for (const s of ['s0', 's1']) {
  const ids: number[] = manifest.meta[`${s}.ids`];
  const res = await synthesize(ids, w, { noiseScale: 0, noiseW: 0, lengthScale: 1 });
  const wavFix = loadFix(`${s}.waveform`);
  console.log(`  ${s}: samples=${res.audio.length} (fixture ${wavFix.length}), F=${res.F}`);
  report(`${s}.synth`, relL2(res.audio, wavFix), 1e-3);
}

console.log(failed ? `\n${failed} FAILED` : '\nall PASS');
process.exit(failed ? 1 : 0);
