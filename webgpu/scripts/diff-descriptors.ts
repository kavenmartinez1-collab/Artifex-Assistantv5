/**
 * Debug: diff descriptorFromGGUF (abliterated Q4_K_M GGUF) against
 * descriptorFromHFConfig (base Qwen3.5-9B config.json — same architecture).
 * Any hyperparameter mismatch is a candidate for the greedy divergence.
 *
 * Run: npx tsx scripts/diff-descriptors.ts
 */
import * as fs from 'fs';
import { parseGGUF } from '../src/model/gguf';
import { descriptorFromGGUF, descriptorFromHFConfig } from '../src/model/model-descriptor';

const GGUF = 'C:/Artifex-Assistant-V5/models/qwen3.5-9b-abliterated-gguf/Huihui-Qwen3.5-9B-abliterated.i1-Q4_K_M.gguf';
const HF_CONFIG = 'C:/Artifex-Assistant-V5/models/qwen3.5-9b/config.json';

async function main() {
  const fd = fs.openSync(GGUF, 'r');
  const readRange = async (start: number, end: number): Promise<ArrayBuffer> => {
    const len = end - start;
    const buf = Buffer.alloc(len);
    fs.readSync(fd, buf, 0, len, start);
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + len);
  };
  const file = await parseGGUF(readRange);
  const dGguf = descriptorFromGGUF(file) as Record<string, any>;

  const hfConfig = JSON.parse(fs.readFileSync(HF_CONFIG, 'utf-8'));
  const dHf = descriptorFromHFConfig(hfConfig) as Record<string, any>;

  // Print all GGUF KV keys that mention rope/attention/norm for reference
  console.log('── GGUF KV (rope/attn/norm/embed) ──');
  for (const [k, v] of file.kv) {
    if (/rope|attention|norm|embedding|expert|context/i.test(k)) {
      console.log(`  ${k} = ${JSON.stringify(v)}`);
    }
  }

  console.log('\n── Descriptor diff (gguf vs hf) ──');
  const keys = new Set([...Object.keys(dGguf), ...Object.keys(dHf)]);
  for (const k of [...keys].sort()) {
    const a = dGguf[k];
    const b = dHf[k];
    if (k === 'layers') continue; // compared separately
    const sa = JSON.stringify(a);
    const sb = JSON.stringify(b);
    if (sa !== sb) console.log(`  ${k}:\n    gguf: ${sa}\n    hf:   ${sb}`);
  }

  // Layer kinds + per-layer fields
  const la = dGguf.layers ?? [];
  const lb = dHf.layers ?? [];
  console.log(`\nlayers: gguf=${la.length} hf=${lb.length}`);
  for (let i = 0; i < Math.max(la.length, lb.length); i++) {
    const sa = JSON.stringify(la[i]);
    const sb = JSON.stringify(lb[i]);
    if (sa !== sb) console.log(`  L${i}:\n    gguf: ${sa}\n    hf:   ${sb}`);
  }
  fs.closeSync(fd);
}

main().catch(e => { console.error(e); process.exit(1); });
