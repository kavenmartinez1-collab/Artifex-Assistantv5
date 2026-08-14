/**
 * Generate WGSL + TS constant tables for IQ2_XS / IQ1_M decode by parsing the
 * grids straight out of vendor/llama.cpp/ggml/src/ggml-common.h. No hand
 * transcription — the tables are bit-exact with llama.cpp.
 *
 *   iq2xs_grid : uint64 x 512  (8 UNSIGNED magnitude bytes per entry)
 *   iq1s_grid  : uint64 x 2048 (8 SIGNED int8 values per entry, shared by
 *                               IQ1_S and IQ1_M)
 *
 * IQ2_XS reuses the ksigns_iq2xs / kmask_iq2xs sign tables already emitted by
 * gen-iq2xxs-tables.mjs. IQ1_M needs no sign table at all: its grid values are
 * already signed and the only sign-like term is a per-group delta bit in qh.
 *
 * Run: node webgpu/scripts/gen-iq2xs-iq1m-tables.mjs
 *   WGSL=1    emit only the WGSL block
 *   TS=1      emit only the TS block
 *   INJECT=1  splice both blocks into the source files between markers
 *   GGML_COMMON_H=<path>  read the header from somewhere other than vendor/
 */
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const ENV = process['env'];
const here = dirname(fileURLToPath(import.meta.url));
const HDR = ENV.GGML_COMMON_H
  ? resolve(ENV.GGML_COMMON_H)
  : resolve(here, '../../vendor/llama.cpp/ggml/src/ggml-common.h');
if (!existsSync(HDR)) {
  throw new Error(
    `ggml-common.h not found at ${HDR}. Point GGML_COMMON_H at a llama.cpp checkout.`);
}
const src = readFileSync(HDR, 'utf8');

/** Resolve a table count that may be written as a #define'd macro. */
function resolveCount(tok) {
  if (/^\d+$/.test(tok)) return Number(tok);
  const m = src.match(new RegExp(`#define\\s+${tok}\\s+(\\d+)`));
  if (!m) throw new Error(`cannot resolve table size macro ${tok}`);
  return Number(m[1]);
}

/** Extract the integer list of a GGML_TABLE_BEGIN(type, name, count) block. */
function table(name) {
  const re = new RegExp(
    `GGML_TABLE_BEGIN\\(\\s*\\w+\\s*,\\s*${name}\\s*,\\s*(\\w+)\\s*\\)([\\s\\S]*?)GGML_TABLE_END\\(\\)`);
  const m = src.match(re);
  if (!m) throw new Error(`table ${name} not found in ${HDR}`);
  const count = resolveCount(m[1]);
  const body = m[2].replace(/\/\/[^\n]*/g, '').replace(/\/\*[\s\S]*?\*\//g, '');
  const vals = body.split(',').map(s => s.trim()).filter(Boolean).map(s => BigInt(s));
  if (vals.length !== count) throw new Error(`${name}: parsed ${vals.length} != ${count}`);
  return vals;
}

const iq2xs = table('iq2xs_grid');   // 512 x u64 (unsigned magnitudes)
const iq1s = table('iq1s_grid');     // 2048 x u64 (signed int8 lanes)

const hexu = (x) => '0x' + (Number(x) >>> 0).toString(16).padStart(8, '0') + 'u';
const fmt = (arr, perLine) => {
  const lines = [];
  for (let i = 0; i < arr.length; i += perLine) {
    lines.push('  ' + arr.slice(i, i + perLine).map(hexu).join(', ') + ',');
  }
  return lines.join('\n');
};

// u64 grid -> 2 u32 words per entry (lo, hi) for WGSL.
const u64words = (vals) => {
  const w = [];
  for (const v of vals) { w.push(v & 0xffffffffn); w.push((v >> 32n) & 0xffffffffn); }
  return w;
};
// u64 grid -> flat little-endian UNSIGNED bytes for the TS reference.
const u64bytes = (vals) => {
  const b = [];
  for (const v of vals) for (let j = 0; j < 8; j++) b.push(Number((v >> BigInt(8 * j)) & 0xffn));
  return b;
};
// u64 grid -> flat little-endian SIGNED bytes (int8) for the TS reference.
const u64sbytes = (vals) => {
  const b = [];
  for (const v of vals) {
    for (let j = 0; j < 8; j++) {
      const u = Number((v >> BigInt(8 * j)) & 0xffn);
      b.push(u > 127 ? u - 256 : u);
    }
  }
  return b;
};

const wgslGrids =
  `const IQ2XS_GRID = array<u32, 1024>(\n${fmt(u64words(iq2xs).map(Number), 8)}\n);\n` +
  `const IQ1S_GRID = array<u32, 4096>(\n${fmt(u64words(iq1s).map(Number), 8)}\n);`;
const tsGrids =
  `const IQ2XS_GRID = new Uint8Array([${u64bytes(iq2xs).join(',')}]);\n` +
  `const IQ1S_GRID = new Int8Array([${u64sbytes(iq1s).join(',')}]);`;

// -- INJECT mode: splice table literals into source files between markers --
if (ENV.INJECT) {
  const splice = (path, tag, payload) => {
    const f = resolve(here, path);
    let txt = readFileSync(f, 'utf8');
    const open = `// <${tag}>`, close = `// </${tag}>`;
    const re = new RegExp(`${open}[\\s\\S]*?${close}`);
    if (!re.test(txt)) throw new Error(`marker ${tag} not found in ${path}`);
    // Source files are CRLF — normalize the injected block to match.
    const body = `${open}\n${payload}\n${close}`.replace(/\r?\n/g, '\r\n');
    txt = txt.replace(re, body);
    writeFileSync(f, txt);
    console.log(`injected ${tag} into ${path}`);
  };
  splice('../src/model/gguf-dequant.ts', 'iq2xs-iq1m-tables-ts', tsGrids);
  splice('../src/shaders/matmul_gguf.wgsl', 'iq2xs-iq1m-tables-wgsl', wgslGrids);
  process.exit(0);
}

const emitWGSL = !ENV.TS || ENV.WGSL;
const emitTS = !ENV.WGSL || ENV.TS;

if (emitWGSL) {
  console.log('// -- WGSL: paste into matmul_gguf.wgsl --');
  console.log(wgslGrids);
}
if (emitTS) {
  console.log('\n// -- TS: paste into gguf-dequant.ts --');
  console.log(tsGrids);
}
