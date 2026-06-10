/**
 * GGUF parser — header, KV metadata, and tensor table.
 *
 * Format (little-endian throughout):
 *   magic u32 'GGUF' | version u32 (v2/v3) | tensor_count u64 | kv_count u64
 *   kv_count × { key: string, value_type: u32, value }
 *   tensor_count × { name: string, n_dims: u32, ne[n_dims]: u64, ggml_type: u32, offset: u64 }
 *   <padding to general.alignment (default 32)>
 *   tensor data (offsets relative to data section start)
 *
 * Dependency-free: parses over a caller-supplied range reader so it works
 * with hf-hub fetchRange (browser), a local File, or a Buffer in node tests.
 *
 * Mirrors the Python reference parser in core/gpu_pool.py:78-190 (KV types)
 * and llama.cpp gguf.c (tensor table, alignment, byte-size math).
 *
 * NOTE on dims: GGUF stores ne[] innermost-first (ne[0] = contiguous axis).
 * `shape` is the HF/safetensors convention (reversed), e.g. a GGUF
 * [2048, 256] router is shape [256, 2048] = [out, in] like torch weights.
 */

// ── GGML tensor types ──────────────────────────────────────────────────

export const GGML_TYPES = {
  F32: 0, F16: 1,
  Q4_0: 2, Q4_1: 3,
  Q5_0: 6, Q5_1: 7,
  Q8_0: 8, Q8_1: 9,
  Q2_K: 10, Q3_K: 11, Q4_K: 12, Q5_K: 13, Q6_K: 14, Q8_K: 15,
  IQ2_XXS: 16, IQ2_XS: 17, IQ3_XXS: 18, IQ1_S: 19, IQ4_NL: 20,
  IQ3_S: 21, IQ2_S: 22, IQ4_XS: 23,
  I8: 24, I16: 25, I32: 26, I64: 27, F64: 28, IQ1_M: 29,
  BF16: 30,
} as const;

export type GGMLTypeName = keyof typeof GGML_TYPES;

/** [elements per block, bytes per block] for each ggml type we understand. */
const TYPE_TRAITS: Record<number, { blockSize: number; typeSize: number; name: GGMLTypeName }> = {
  [GGML_TYPES.F32]:  { blockSize: 1,   typeSize: 4,   name: 'F32' },
  [GGML_TYPES.F16]:  { blockSize: 1,   typeSize: 2,   name: 'F16' },
  [GGML_TYPES.BF16]: { blockSize: 1,   typeSize: 2,   name: 'BF16' },
  [GGML_TYPES.I8]:   { blockSize: 1,   typeSize: 1,   name: 'I8' },
  [GGML_TYPES.I16]:  { blockSize: 1,   typeSize: 2,   name: 'I16' },
  [GGML_TYPES.I32]:  { blockSize: 1,   typeSize: 4,   name: 'I32' },
  [GGML_TYPES.I64]:  { blockSize: 1,   typeSize: 8,   name: 'I64' },
  [GGML_TYPES.F64]:  { blockSize: 1,   typeSize: 8,   name: 'F64' },
  [GGML_TYPES.Q4_0]: { blockSize: 32,  typeSize: 18,  name: 'Q4_0' },
  [GGML_TYPES.Q4_1]: { blockSize: 32,  typeSize: 20,  name: 'Q4_1' },
  [GGML_TYPES.Q5_0]: { blockSize: 32,  typeSize: 22,  name: 'Q5_0' },
  [GGML_TYPES.Q5_1]: { blockSize: 32,  typeSize: 24,  name: 'Q5_1' },
  [GGML_TYPES.Q8_0]: { blockSize: 32,  typeSize: 34,  name: 'Q8_0' },
  [GGML_TYPES.Q8_1]: { blockSize: 32,  typeSize: 36,  name: 'Q8_1' },
  [GGML_TYPES.Q2_K]: { blockSize: 256, typeSize: 84,  name: 'Q2_K' },
  [GGML_TYPES.Q3_K]: { blockSize: 256, typeSize: 110, name: 'Q3_K' },
  [GGML_TYPES.Q4_K]: { blockSize: 256, typeSize: 144, name: 'Q4_K' },
  [GGML_TYPES.Q5_K]: { blockSize: 256, typeSize: 176, name: 'Q5_K' },
  [GGML_TYPES.Q6_K]: { blockSize: 256, typeSize: 210, name: 'Q6_K' },
  [GGML_TYPES.Q8_K]: { blockSize: 256, typeSize: 292, name: 'Q8_K' },
};

export function ggmlTypeTraits(ggmlType: number): { blockSize: number; typeSize: number; name: GGMLTypeName } {
  const t = TYPE_TRAITS[ggmlType];
  if (!t) throw new Error(`[GGUF] Unsupported ggml type ${ggmlType}`);
  return t;
}

/** Bytes for a tensor: rowSize(ne0) × product of outer dims (llama.cpp ggml_nbytes). */
export function ggmlTensorBytes(ggmlType: number, ne: number[]): number {
  const { blockSize, typeSize } = ggmlTypeTraits(ggmlType);
  const ne0 = ne[0] ?? 1;
  if (ne0 % blockSize !== 0) {
    throw new Error(`[GGUF] ne[0]=${ne0} not divisible by block size ${blockSize} for type ${ggmlType}`);
  }
  const rowBytes = (ne0 / blockSize) * typeSize;
  let outer = 1;
  for (let i = 1; i < ne.length; i++) outer *= ne[i];
  return rowBytes * outer;
}

// ── Parsed structures ──────────────────────────────────────────────────

export interface GGUFTensorInfo {
  name: string;
  /** GGUF-native dims, innermost first (ne[0] = contiguous axis). */
  ne: number[];
  /** HF/torch convention (reversed ne): [out, in] for matrices. */
  shape: number[];
  ggmlType: number;
  typeName: GGMLTypeName;
  /** Absolute byte offset of this tensor's data in the file. */
  offset: number;
  byteLength: number;
  elementCount: number;
}

export interface GGUFFile {
  version: number;
  tensorCount: number;
  /** All KV metadata, e.g. 'general.architecture', '<arch>.block_count'. */
  kv: Map<string, unknown>;
  tensors: Map<string, GGUFTensorInfo>;
  /** Absolute offset where the aligned tensor data section starts. */
  dataOffset: number;
  alignment: number;
  /** Total bytes consumed by header + KV + tensor table (pre-padding). */
  headerBytes: number;
}

export type RangeReader = (start: number, end: number) => Promise<ArrayBuffer>;

// ── Cursor over progressively-fetched header bytes ─────────────────────

const GGUF_MAGIC = 0x46554747; // 'GGUF' LE
const INITIAL_FETCH = 4 * 1024 * 1024;   // tokenizer KV arrays can be tens of MB
const MAX_HEADER = 512 * 1024 * 1024;    // sanity cap

class HeaderCursor {
  private buf: ArrayBuffer;
  private view: DataView;
  private bytes: Uint8Array;
  private readRange: RangeReader;
  pos = 0;

  // No parameter properties — keeps the file runnable under node --strip-types
  constructor(readRange: RangeReader, initial: ArrayBuffer) {
    this.readRange = readRange;
    this.buf = initial;
    this.view = new DataView(initial);
    this.bytes = new Uint8Array(initial);
  }

  /** Ensure at least `n` more bytes are available; grows the buffer by refetching. */
  private async ensure(n: number): Promise<void> {
    if (this.pos + n <= this.buf.byteLength) return;
    let newLen = this.buf.byteLength;
    while (newLen < this.pos + n) newLen *= 2;
    if (newLen > MAX_HEADER) throw new Error(`[GGUF] Header exceeds ${MAX_HEADER} bytes — corrupt file?`);
    const more = await this.readRange(this.buf.byteLength, newLen);
    const grown = new Uint8Array(this.buf.byteLength + more.byteLength);
    grown.set(this.bytes, 0);
    grown.set(new Uint8Array(more), this.buf.byteLength);
    // ArrayBuffer.slice truncation rule: verify we actually got the bytes
    if (grown.byteLength < this.pos + n) {
      throw new Error(`[GGUF] Short read growing header buffer: have ${grown.byteLength}, need ${this.pos + n}`);
    }
    this.buf = grown.buffer;
    this.view = new DataView(this.buf);
    this.bytes = grown;
  }

  async u8(): Promise<number>  { await this.ensure(1); return this.view.getUint8(this.pos++); }
  async i8(): Promise<number>  { await this.ensure(1); return this.view.getInt8(this.pos++); }
  async u16(): Promise<number> { await this.ensure(2); const v = this.view.getUint16(this.pos, true); this.pos += 2; return v; }
  async i16(): Promise<number> { await this.ensure(2); const v = this.view.getInt16(this.pos, true); this.pos += 2; return v; }
  async u32(): Promise<number> { await this.ensure(4); const v = this.view.getUint32(this.pos, true); this.pos += 4; return v; }
  async i32(): Promise<number> { await this.ensure(4); const v = this.view.getInt32(this.pos, true); this.pos += 4; return v; }
  async f32(): Promise<number> { await this.ensure(4); const v = this.view.getFloat32(this.pos, true); this.pos += 4; return v; }
  async f64(): Promise<number> { await this.ensure(8); const v = this.view.getFloat64(this.pos, true); this.pos += 8; return v; }

  /** u64 → Number. Throws if the value exceeds 2^53 (file offsets never should). */
  async u64(): Promise<number> {
    await this.ensure(8);
    const v = this.view.getBigUint64(this.pos, true);
    this.pos += 8;
    if (v > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`[GGUF] u64 ${v} exceeds safe integer`);
    return Number(v);
  }

  async i64(): Promise<number> {
    await this.ensure(8);
    const v = this.view.getBigInt64(this.pos, true);
    this.pos += 8;
    if (v > BigInt(Number.MAX_SAFE_INTEGER) || v < -BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error(`[GGUF] i64 ${v} exceeds safe integer`);
    }
    return Number(v);
  }

  async str(): Promise<string> {
    const len = await this.u64();
    await this.ensure(len);
    const s = new TextDecoder('utf-8').decode(this.bytes.subarray(this.pos, this.pos + len));
    this.pos += len;
    return s;
  }
}

// GGUF metadata value types (same numbering as core/gpu_pool.py _val).
// Plain const object (not enum) so node's TS type-stripping can run this file.
const GGUFValueType = {
  UINT8: 0, INT8: 1, UINT16: 2, INT16: 3, UINT32: 4, INT32: 5,
  FLOAT32: 6, BOOL: 7, STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12,
} as const;

async function readValue(c: HeaderCursor, type: number): Promise<unknown> {
  switch (type) {
    case GGUFValueType.UINT8:   return c.u8();
    case GGUFValueType.INT8:    return c.i8();
    case GGUFValueType.UINT16:  return c.u16();
    case GGUFValueType.INT16:   return c.i16();
    case GGUFValueType.UINT32:  return c.u32();
    case GGUFValueType.INT32:   return c.i32();
    case GGUFValueType.FLOAT32: return c.f32();
    case GGUFValueType.BOOL:    return (await c.u8()) !== 0;
    case GGUFValueType.STRING:  return c.str();
    case GGUFValueType.UINT64:  return c.u64();
    case GGUFValueType.INT64:   return c.i64();
    case GGUFValueType.FLOAT64: return c.f64();
    case GGUFValueType.ARRAY: {
      const elemType = await c.u32();
      const count = await c.u64();
      const out = new Array(count);
      for (let i = 0; i < count; i++) out[i] = await readValue(c, elemType);
      return out;
    }
    default:
      throw new Error(`[GGUF] Unknown KV value type ${type}`);
  }
}

// ── Parser ─────────────────────────────────────────────────────────────

/**
 * Parse a GGUF file's header, KV metadata, and tensor table.
 * Only fetches header bytes — tensor data is NOT downloaded.
 */
export async function parseGGUF(readRange: RangeReader): Promise<GGUFFile> {
  const initial = await readRange(0, INITIAL_FETCH);
  const c = new HeaderCursor(readRange, initial);

  const magic = await c.u32();
  if (magic !== GGUF_MAGIC) {
    throw new Error(`[GGUF] Bad magic 0x${magic.toString(16)} — not a GGUF file`);
  }
  const version = await c.u32();
  if (version < 2 || version > 3) {
    throw new Error(`[GGUF] Unsupported version ${version} (need 2 or 3)`);
  }
  const tensorCount = await c.u64();
  const kvCount = await c.u64();

  const kv = new Map<string, unknown>();
  for (let i = 0; i < kvCount; i++) {
    const key = await c.str();
    const vtype = await c.u32();
    kv.set(key, await readValue(c, vtype));
  }

  const alignment = Number(kv.get('general.alignment') ?? 32);

  // Tensor table — offsets are relative to the data section start
  const raw: Array<{ name: string; ne: number[]; ggmlType: number; relOffset: number }> = [];
  for (let i = 0; i < tensorCount; i++) {
    const name = await c.str();
    const nDims = await c.u32();
    if (nDims > 4) throw new Error(`[GGUF] Tensor "${name}" has ${nDims} dims (max 4)`);
    const ne: number[] = [];
    for (let d = 0; d < nDims; d++) ne.push(await c.u64());
    const ggmlType = await c.u32();
    const relOffset = await c.u64();
    raw.push({ name, ne, ggmlType, relOffset });
  }

  const headerBytes = c.pos;
  const dataOffset = Math.ceil(headerBytes / alignment) * alignment;

  const tensors = new Map<string, GGUFTensorInfo>();
  for (const t of raw) {
    const byteLength = ggmlTensorBytes(t.ggmlType, t.ne);
    let elementCount = 1;
    for (const d of t.ne) elementCount *= d;
    tensors.set(t.name, {
      name: t.name,
      ne: t.ne,
      shape: [...t.ne].reverse(),
      ggmlType: t.ggmlType,
      typeName: ggmlTypeTraits(t.ggmlType).name,
      offset: dataOffset + t.relOffset,
      byteLength,
      elementCount,
    });
  }

  return { version, tensorCount, kv, tensors, dataOffset, alignment, headerBytes };
}

// ── Convenience accessors ──────────────────────────────────────────────

/** 'general.architecture', e.g. 'qwen35moe', 'qwen3', 'llama', 'gemma3'. */
export function ggufArchitecture(file: GGUFFile): string {
  const arch = file.kv.get('general.architecture');
  if (typeof arch !== 'string') throw new Error('[GGUF] Missing general.architecture');
  return arch;
}

/** Read '<arch>.<key>' with optional fallback, e.g. archKV(f, 'block_count'). */
export function archKV<T>(file: GGUFFile, key: string, fallback?: T): T {
  const arch = ggufArchitecture(file);
  const v = file.kv.get(`${arch}.${key}`);
  if (v === undefined) {
    if (fallback !== undefined) return fallback;
    throw new Error(`[GGUF] Missing KV ${arch}.${key}`);
  }
  return v as T;
}
