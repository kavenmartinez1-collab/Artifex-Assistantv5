/**
 * Generate WGSL + TS constant tables for IQ3_XXS / IQ3_S / IQ2_S decode by
 * parsing the grids straight out of vendor/llama.cpp/ggml/src/ggml-common.h.
 * No hand transcription — the tables are bit-exact with llama.cpp.
 *
 *   iq3xxs_grid : uint32 × 256  (4 magnitude bytes per entry)
 *   iq3s_grid   : uint32 × 512  (4 magnitude bytes per entry)
 *   iq2s_grid   : uint64 × 1024 (8 magnitude bytes per entry)
 *   kmask_iq2xs : uint8  × 8    (per-element sign bit mask)
 *
 * Run: node webgpu/scripts/gen-iq3-iq2s-tables.mjs
 *   WGSL=1  emit only the WGSL block
 *   TS=1    emit only the TS block
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const HDR = resolve(here, '../../vendor/llama.cpp/ggml/src/ggml-common.h');
const src = readFileSync(HDR, 'utf8');

/** Extract the integer list of a GGML_TABLE_BEGIN(type, name, count) block. */
function table(name) {
  const re = new RegExp(
    `GGML_TABLE_BEGIN\\(\\s*\\w+\\s*,\\s*${name}\\s*,\\s*(\\d+)\\s*\\)([\\s\\S]*?)GGML_TABLE_END\\(\\)`);
  const m = src.match(re);
  if (!m) throw new Error(`table ${name} not found in ${HDR}`);
  const count = Number(m[1]);
  const body = m[2].replace(/\/\/[^\n]*/g, '').replace(/\/\*[\s\S]*?\*\//g, '');
  const vals = body.split(',').map(s => s.trim()).filter(Boolean).map(s => BigInt(s));
  if (vals.length !== count) throw new Error(`${name}: parsed ${vals.length} != ${count}`);
  return vals;
}

const iq3xxs = table('iq3xxs_grid');   // 256 × u32
const iq3s   = table('iq3s_grid');     // 512 × u32
const iq2s   = table('iq2s_grid');     // 1024 × u64
const kmask  = table('kmask_iq2xs');   // 8 × u8

const hexu = (x) => '0x' + (Number(x) >>> 0).toString(16).padStart(8, '0') + 'u';
const fmt = (arr, perLine) => {
  const lines = [];
  for (let i = 0; i < arr.length; i += perLine) {
    lines.push('  ' + arr.slice(i, i + perLine).map(hexu).join(', ') + ',');
  }
  return lines.join('\n');
};

// u64 grid → 2 u32 words per entry (lo, hi) for WGSL.
const u64words = (vals) => {
  const w = [];
  for (const v of vals) { w.push(v & 0xffffffffn); w.push((v >> 32n) & 0xffffffffn); }
  return w;
};
// u64 grid → flat little-endian bytes for the TS reference.
const u64bytes = (vals) => {
  const b = [];
  for (const v of vals) for (let j = 0; j < 8; j++) b.push(Number((v >> BigInt(8 * j)) & 0xffn));
  return b;
};
// u32 grid → flat little-endian bytes (4 per entry) for the TS reference.
const u32bytes = (vals) => {
  const b = [];
  for (const v of vals) for (let j = 0; j < 4; j++) b.push(Number((v >> BigInt(8 * j)) & 0xffn));
  return b;
};

// ── INJECT mode: splice table literals into source files between markers ──
if (process.env.INJECT) {
  const wgslGrids =
    `const IQ3XXS_GRID = array<u32, 256>(\n${fmt(iq3xxs.map(Number), 8)}\n);\n` +
    `const IQ3S_GRID = array<u32, 512>(\n${fmt(iq3s.map(Number), 8)}\n);\n` +
    `const IQ2S_GRID = array<u32, 2048>(\n${fmt(u64words(iq2s).map(Number), 8)}\n);`;
  const tsGrids =
    `const IQ3XXS_GRID = new Uint8Array([${u32bytes(iq3xxs).join(',')}]);\n` +
    `const IQ3S_GRID = new Uint8Array([${u32bytes(iq3s).join(',')}]);\n` +
    `const IQ2S_GRID = new Uint8Array([${u64bytes(iq2s).join(',')}]);\n` +
    `const KMASK_IQ2XS = new Uint8Array([${kmask.map(Number).join(',')}]);`;

  const splice = (path, tag, payload) => {
    const f = resolve(here, path);
    let txt = readFileSync(f, 'utf8');
    const open = `// <${tag}>`, close = `// </${tag}>`;
    const re = new RegExp(`${open}[\\s\\S]*?${close}`);
    if (!re.test(txt)) throw new Error(`marker ${tag} not found in ${path}`);
    txt = txt.replace(re, `${open}\n${payload}\n${close}`);
    writeFileSync(f, txt);
    console.log(`injected ${tag} into ${path}`);
  };
  splice('../src/model/gguf-dequant.ts', 'iq3-iq2s-tables-ts', tsGrids);
  splice('../src/shaders/matmul_gguf.wgsl', 'iq3-iq2s-tables-wgsl', wgslGrids);
  process.exit(0);
}

const emitWGSL = !process.env.TS || process.env.WGSL;
const emitTS = !process.env.WGSL || process.env.TS;

if (emitWGSL) {
  console.log('// ── WGSL: paste into matmul_gguf.wgsl ──');
  console.log(`const IQ3XXS_GRID = array<u32, 256>(`);
  console.log(fmt(iq3xxs.map(v => Number(v)), 8));
  console.log(');');
  console.log(`const IQ3S_GRID = array<u32, 512>(`);
  console.log(fmt(iq3s.map(v => Number(v)), 8));
  console.log(');');
  console.log(`const IQ2S_GRID = array<u32, 2048>(`);
  console.log(fmt(u64words(iq2s).map(v => Number(v)), 8));
  console.log(');');
}

if (emitTS) {
  console.log('\n// ── TS: paste into gguf-dequant.ts ──');
  console.log(`const IQ3XXS_GRID = new Uint8Array([${u32bytes(iq3xxs).join(',')}]);`);
  console.log(`const IQ3S_GRID = new Uint8Array([${u32bytes(iq3s).join(',')}]);`);
  console.log(`const IQ2S_GRID = new Uint8Array([${u64bytes(iq2s).join(',')}]);`);
  console.log(`const KMASK_IQ2XS = new Uint8Array([${kmask.map(Number).join(',')}]);`);
}
