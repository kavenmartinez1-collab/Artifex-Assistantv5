/**
 * SafeTensors Format Parser
 *
 * SafeTensors file format:
 *   [8 bytes] — header length as little-endian uint64
 *   [N bytes] — JSON header (tensor metadata)
 *   [remainder] — raw tensor data (contiguous, byte-aligned)
 *
 * The JSON header maps tensor names to their metadata:
 *   { "model.layers.0.self_attn.q_proj.weight": { "dtype": "F16", "shape": [3584, 3584], "data_offsets": [0, 25690112] } }
 *
 * Supported dtypes: F32, F16, BF16, I8, U8, I32, I16, U16, F64, BOOL
 */

// ─── Types ───────────────────────────────────────────────────────────────────

export interface TensorInfo {
  name: string;
  dtype: SafeTensorDtype;
  shape: number[];
  dataOffsets: [number, number]; // [start, end] byte offsets into the data section
  byteLength: number;
  elementCount: number;
}

export type SafeTensorDtype =
  | 'F32' | 'F16' | 'BF16'
  | 'I8' | 'U8' | 'I16' | 'U16' | 'I32'
  | 'F64' | 'BOOL';

export interface SafeTensorsHeader {
  tensors: Map<string, TensorInfo>;
  metadata?: Record<string, string>; // optional __metadata__ field
  headerByteLength: number; // 8 + JSON header length
}

// Bytes per element for each dtype
const DTYPE_SIZES: Record<SafeTensorDtype, number> = {
  F32: 4, F16: 2, BF16: 2,
  I8: 1, U8: 1, I16: 2, U16: 2, I32: 4,
  F64: 8, BOOL: 1,
};

// ─── Header Parsing ──────────────────────────────────────────────────────────

/**
 * Parse the SafeTensors header from raw bytes.
 * Only reads the header — does not load tensor data.
 *
 * @param headerBytes - The first (8 + headerLength) bytes of the file
 * @returns Parsed header with tensor metadata
 */
export function parseHeader(headerBytes: ArrayBuffer): SafeTensorsHeader {
  const view = new DataView(headerBytes);

  // First 8 bytes: header length as little-endian uint64
  // JavaScript can't do 64-bit ints natively, but headers are always < 4GB
  const headerLen = view.getUint32(0, true); // little-endian, low 32 bits
  const headerHigh = view.getUint32(4, true);
  if (headerHigh > 0) {
    throw new Error(`SafeTensors header too large: ${headerHigh * 4294967296 + headerLen} bytes`);
  }

  if (headerLen === 0) {
    throw new Error('SafeTensors header length is 0');
  }

  // Decode JSON header
  const headerJsonBytes = new Uint8Array(headerBytes, 8, headerLen);
  const headerText = new TextDecoder().decode(headerJsonBytes);
  let headerObj: Record<string, any>;

  try {
    headerObj = JSON.parse(headerText);
  } catch (e) {
    throw new Error(`Invalid SafeTensors header JSON: ${e}`);
  }

  // Parse tensor entries
  const tensors = new Map<string, TensorInfo>();
  let metadata: Record<string, string> | undefined;

  for (const [key, value] of Object.entries(headerObj)) {
    if (key === '__metadata__') {
      metadata = value as Record<string, string>;
      continue;
    }

    const dtype = value.dtype as SafeTensorDtype;
    if (!(dtype in DTYPE_SIZES)) {
      console.warn(`[SafeTensors] Unknown dtype "${dtype}" for tensor "${key}", skipping`);
      continue;
    }

    const shape = value.shape as number[];
    const dataOffsets = value.data_offsets as [number, number];
    const elementCount = shape.reduce((a: number, b: number) => a * b, 1);
    const byteLength = dataOffsets[1] - dataOffsets[0];

    // Sanity check
    const expectedBytes = elementCount * DTYPE_SIZES[dtype];
    if (byteLength !== expectedBytes) {
      console.warn(
        `[SafeTensors] Size mismatch for "${key}": ` +
        `expected ${expectedBytes} bytes (${elementCount} × ${DTYPE_SIZES[dtype]}), ` +
        `got ${byteLength} bytes`
      );
    }

    tensors.set(key, {
      name: key,
      dtype,
      shape,
      dataOffsets,
      byteLength,
      elementCount,
    });
  }

  return {
    tensors,
    metadata,
    headerByteLength: 8 + headerLen,
  };
}

/**
 * Parse just the 8-byte header length prefix.
 * Use this to know how many bytes to fetch for the full header.
 */
export function parseHeaderLength(first8Bytes: ArrayBuffer): number {
  const view = new DataView(first8Bytes);
  const len = view.getUint32(0, true);
  const high = view.getUint32(4, true);
  if (high > 0) {
    throw new Error('SafeTensors header exceeds 4GB');
  }
  return len;
}

/**
 * Extract raw tensor data from a SafeTensors file buffer.
 *
 * @param fileData - The full file (or relevant slice) as ArrayBuffer
 * @param tensor - TensorInfo from the parsed header
 * @param dataOffset - Byte offset where the data section starts (= headerByteLength)
 * @returns Raw bytes for this tensor
 */
export function extractTensorData(
  fileData: ArrayBuffer,
  tensor: TensorInfo,
  dataOffset: number,
): ArrayBuffer {
  const start = dataOffset + tensor.dataOffsets[0];
  const end = dataOffset + tensor.dataOffsets[1];
  return fileData.slice(start, end);
}

/**
 * Convert raw tensor bytes to a Float32Array.
 * Handles F32, F16, and BF16 conversion.
 */
export function tensorToFloat32(data: ArrayBuffer, dtype: SafeTensorDtype): Float32Array {
  if (dtype === 'F32') {
    return new Float32Array(data);
  }

  if (dtype === 'F16') {
    return float16ToFloat32(new Uint16Array(data));
  }

  if (dtype === 'BF16') {
    return bfloat16ToFloat32(new Uint16Array(data));
  }

  if (dtype === 'I8') {
    const src = new Int8Array(data);
    const out = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) out[i] = src[i];
    return out;
  }

  if (dtype === 'U8') {
    const src = new Uint8Array(data);
    const out = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) out[i] = src[i];
    return out;
  }

  throw new Error(`Unsupported dtype for float32 conversion: ${dtype}`);
}

/**
 * Extract raw tensor bytes as a typed array without conversion.
 * Used for GPTQ packed weights (I32) and scales (F16) that need
 * to stay in their original format for GPU-side dequantization.
 */
export function tensorToTypedArray(data: ArrayBuffer, dtype: SafeTensorDtype): ArrayBufferView {
  switch (dtype) {
    case 'I32': return new Int32Array(data);
    case 'U8': return new Uint8Array(data);
    case 'I8': return new Int8Array(data);
    case 'I16': return new Int16Array(data);
    case 'U16': return new Uint16Array(data);
    case 'F16': return new Uint16Array(data); // raw F16 bits
    case 'BF16': return new Uint16Array(data); // raw BF16 bits
    case 'F32': return new Float32Array(data);
    case 'F64': return new Float64Array(data);
    default: return new Uint8Array(data);
  }
}

// ─── Float16/BFloat16 Conversion ─────────────────────────────────────────────

/**
 * Convert IEEE 754 half-precision (float16) to float32.
 * This is the format used by most HuggingFace models.
 */
function float16ToFloat32(input: Uint16Array): Float32Array {
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const h = input[i];
    const sign = (h >> 15) & 0x1;
    const exponent = (h >> 10) & 0x1f;
    const mantissa = h & 0x3ff;

    let value: number;
    if (exponent === 0) {
      // Subnormal or zero
      value = mantissa === 0 ? 0 : Math.pow(2, -14) * (mantissa / 1024);
    } else if (exponent === 31) {
      // Infinity or NaN
      value = mantissa === 0 ? Infinity : NaN;
    } else {
      // Normalized
      value = Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
    }

    output[i] = sign ? -value : value;
  }
  return output;
}

/**
 * Convert Google Brain's bfloat16 to float32.
 * BF16 is just the upper 16 bits of a float32, so conversion is trivial.
 */
function bfloat16ToFloat32(input: Uint16Array): Float32Array {
  const output = new Float32Array(input.length);
  const view = new DataView(output.buffer);
  for (let i = 0; i < input.length; i++) {
    // BF16 is the top 16 bits of float32 — just shift left by 16
    view.setUint32(i * 4, input[i] << 16, true);
  }
  return output;
}

// ─── Utilities ───────────────────────────────────────────────────────────────

/**
 * Get the byte size per element for a dtype.
 */
export function dtypeSize(dtype: SafeTensorDtype): number {
  return DTYPE_SIZES[dtype] ?? 4;
}

/**
 * Format a byte count as a human-readable string.
 */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

/**
 * Summarize a parsed header for display.
 */
export function summarizeHeader(header: SafeTensorsHeader): string {
  const tensorCount = header.tensors.size;
  let totalBytes = 0;
  const dtypeCounts: Record<string, number> = {};
  let maxShape = '';
  let maxSize = 0;

  for (const t of header.tensors.values()) {
    totalBytes += t.byteLength;
    dtypeCounts[t.dtype] = (dtypeCounts[t.dtype] || 0) + 1;
    if (t.byteLength > maxSize) {
      maxSize = t.byteLength;
      maxShape = `${t.name} [${t.shape.join('×')}]`;
    }
  }

  const dtypeStr = Object.entries(dtypeCounts)
    .map(([d, c]) => `${d}:${c}`)
    .join(', ');

  return (
    `${tensorCount} tensors, ${formatBytes(totalBytes)} total\n` +
    `Dtypes: ${dtypeStr}\n` +
    `Largest: ${maxShape} (${formatBytes(maxSize)})`
  );
}
