/**
 * CPU parity for the IQ3_XXS / IQ3_S / IQ2_S dequant against the synthetic
 * fixture from gen_iq_parity_fixture.py (official `gguf` package reference).
 *
 * Usage: node webgpu/scripts/test-iq-parity.mts <fixture.json>
 */
import { readFile } from 'node:fs/promises';
import { dequantGGML } from '../src/model/gguf-dequant.ts';

const [fixturePath] = process.argv.slice(2);
if (!fixturePath) {
  console.error('usage: node test-iq-parity.mts <fixture.json>');
  process.exit(2);
}

const fixture = JSON.parse(await readFile(fixturePath, 'utf8'));
let failures = 0;

for (const s of fixture.dequant_samples) {
  const raw = Uint8Array.from(Buffer.from(s.bytes_hex, 'hex'));
  const got = dequantGGML(s.ggml_type, raw, s.n_elements);
  const want = s.expected_f32 as number[];
  if (got.length !== want.length) {
    console.error(`  FAIL ${s.tensor}: length ${got.length} != ${want.length}`);
    failures++;
    continue;
  }
  let maxAbs = 0, maxVal = 0, firstBad = -1;
  for (let i = 0; i < want.length; i++) {
    const d = Math.abs(Math.fround(got[i]) - want[i]);
    if (d > maxAbs) maxAbs = d;
    maxVal = Math.max(maxVal, Math.abs(want[i]));
    if (firstBad < 0 && d > Math.max(1e-7, Math.abs(want[i]) * 1e-5)) firstBad = i;
  }
  const tol = Math.max(1e-6, maxVal * 1e-5);
  const ok = maxAbs <= tol;
  if (!ok) {
    failures++;
    console.error(`  FAIL ${s.tensor} (type ${s.ggml_type}): maxAbsDiff ${maxAbs.toExponential(3)} > tol ${tol.toExponential(3)}`);
    if (firstBad >= 0) console.error(`    first mismatch @${firstBad}: got ${got[firstBad]} want ${want[firstBad]}`);
  }
  console.log(`${s.tensor} (type ${s.ggml_type}): maxAbsDiff=${maxAbs.toExponential(2)} over ${want.length} values ${ok ? 'OK' : 'FAIL'}`);
}

if (failures > 0) {
  console.error(`\n${failures} FAILURE(S)`);
  process.exit(1);
}
console.log('\nALL PARITY CHECKS PASSED');
