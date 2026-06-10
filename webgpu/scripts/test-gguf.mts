/**
 * Validate gguf.ts (structure) and gguf-dequant.ts (values) against the
 * Python-generated fixture (official `gguf` package = independent reference).
 *
 * Usage: node webgpu/scripts/test-gguf.mts <model.gguf> <fixture.json>
 * (node >= 23 runs TS directly via type stripping)
 */
import { open, readFile } from 'node:fs/promises';
import { parseGGUF, type RangeReader } from '../src/model/gguf.ts';
import { dequantGGML } from '../src/model/gguf-dequant.ts';

const [ggufPath, fixturePath] = process.argv.slice(2);
if (!ggufPath || !fixturePath) {
  console.error('usage: node test-gguf.mts <model.gguf> <fixture.json>');
  process.exit(2);
}

let failures = 0;
function check(cond: boolean, msg: string) {
  if (!cond) { console.error(`  FAIL: ${msg}`); failures++; }
}

const fixture = JSON.parse(await readFile(fixturePath, 'utf8'));
const fh = await open(ggufPath, 'r');
const readRange: RangeReader = async (start, end) => {
  const len = end - start;
  const buf = Buffer.alloc(len);
  const { bytesRead } = await fh.read(buf, 0, len, start);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + bytesRead);
};

console.log(`Parsing ${ggufPath} ...`);
const t0 = performance.now();
const file = await parseGGUF(readRange);
console.log(`Parsed in ${(performance.now() - t0).toFixed(0)} ms: ${file.tensors.size} tensors, ${file.kv.size} KV entries`);

// ── Structure checks ────────────────────────────────────────────────────
check(file.version === fixture.version, `version ${file.version} != ${fixture.version}`);
check(file.tensorCount === fixture.tensor_count, `tensorCount ${file.tensorCount} != ${fixture.tensor_count}`);
check(file.alignment === fixture.alignment, `alignment ${file.alignment} != ${fixture.alignment}`);
check(file.dataOffset === fixture.data_offset, `dataOffset ${file.dataOffset} != ${fixture.data_offset}`);

let tensorMismatches = 0;
for (const ft of fixture.tensors) {
  const t = file.tensors.get(ft.name);
  if (!t) { check(false, `missing tensor ${ft.name}`); tensorMismatches++; continue; }
  const ok =
    t.ggmlType === ft.ggml_type &&
    t.offset === ft.abs_offset &&
    t.byteLength === ft.n_bytes &&
    t.elementCount === ft.n_elements &&
    t.ne.length === ft.ne.length &&
    t.ne.every((d: number, i: number) => d === ft.ne[i]);
  if (!ok) {
    tensorMismatches++;
    check(false, `tensor ${ft.name}: got type=${t.ggmlType} off=${t.offset} bytes=${t.byteLength} ne=[${t.ne}] ` +
      `want type=${ft.ggml_type} off=${ft.abs_offset} bytes=${ft.n_bytes} ne=[${ft.ne}]`);
  }
}
console.log(`Tensor table: ${fixture.tensors.length - tensorMismatches}/${fixture.tensors.length} match`);

// Spot-check a few KV scalars
for (const [key, want] of Object.entries(fixture.kv)) {
  if (key.startsWith('GGUF.')) continue; // pseudo-keys the python reader synthesizes from header fields
  if (typeof want !== 'string' && typeof want !== 'number' && typeof want !== 'boolean') continue;
  const got = file.kv.get(key);
  if (typeof want === 'number' && typeof got === 'number') {
    check(Math.abs(got - want) <= Math.abs(want) * 1e-6, `kv ${key}: ${got} != ${want}`);
  } else {
    check(got === want, `kv ${key}: ${JSON.stringify(got)} != ${JSON.stringify(want)}`);
  }
}

// ── Dequant value checks ────────────────────────────────────────────────
for (const s of fixture.dequant_samples) {
  const raw = Uint8Array.from(Buffer.from(s.bytes_hex, 'hex'));
  const got = dequantGGML(s.ggml_type, raw, s.n_elements);
  const want = s.expected_f32 as number[];
  check(got.length === want.length, `sample ${s.tensor}: length ${got.length} != ${want.length}`);
  let maxAbs = 0, maxVal = 0;
  for (let i = 0; i < want.length; i++) {
    // numpy reference computes in f32; emulate with fround for exactness
    maxAbs = Math.max(maxAbs, Math.abs(Math.fround(got[i]) - want[i]));
    maxVal = Math.max(maxVal, Math.abs(want[i]));
  }
  const tol = Math.max(1e-7, maxVal * 1e-6);
  check(maxAbs <= tol, `sample ${s.tensor} (type ${s.ggml_type}): maxAbsDiff ${maxAbs} > tol ${tol}`);
  console.log(`Dequant ${s.tensor} (type ${s.ggml_type}): maxAbsDiff=${maxAbs.toExponential(2)} over ${want.length} values ${maxAbs <= tol ? 'OK' : 'FAIL'}`);
}

await fh.close();
if (failures > 0) {
  console.error(`\n${failures} FAILURE(S)`);
  process.exit(1);
}
console.log('\nALL CHECKS PASSED');
