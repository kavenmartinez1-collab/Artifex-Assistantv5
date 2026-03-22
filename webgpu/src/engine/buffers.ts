/**
 * GPU Buffer Management Utilities
 *
 * Create, read, write, and manage WebGPU buffers for tensor data.
 */

export interface TensorBuffer {
  buffer: GPUBuffer;
  size: number;       // in bytes
  shape: number[];    // tensor dimensions
  dtype: string;      // 'f32', 'f16', 'i32', 'u8', etc.
  label: string;
}

/**
 * Create a storage buffer and optionally upload data.
 */
export function createStorageBuffer(
  device: GPUDevice,
  data: ArrayBufferView | null,
  sizeBytes: number,
  label = '',
  readable = false,
): GPUBuffer {
  const usage = GPUBufferUsage.STORAGE
    | GPUBufferUsage.COPY_DST
    | (readable ? GPUBufferUsage.COPY_SRC : 0);

  const buffer = device.createBuffer({
    size: sizeBytes,
    usage,
    label,
    mappedAtCreation: data !== null,
  });

  if (data !== null) {
    const mapped = buffer.getMappedRange();
    new Uint8Array(mapped).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap();
  }

  return buffer;
}

/**
 * Create a uniform buffer (small, for passing params to shaders).
 */
export function createUniformBuffer(
  device: GPUDevice,
  data: ArrayBufferView,
  label = '',
): GPUBuffer {
  const buffer = device.createBuffer({
    size: Math.max(data.byteLength, 16), // min 16 bytes for alignment
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange(0, data.byteLength)).set(
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
  );
  buffer.unmap();
  return buffer;
}

/**
 * Read data back from a GPU buffer to CPU.
 */
export async function readBuffer(
  device: GPUDevice,
  source: GPUBuffer,
  sizeBytes?: number,
): Promise<ArrayBuffer> {
  const size = sizeBytes ?? source.size;

  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    label: 'staging-read',
  });

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const data = staging.getMappedRange().slice(0); // copy before unmap
  staging.unmap();
  staging.destroy();

  return data;
}

/**
 * Write data to a GPU buffer.
 */
export function writeBuffer(
  device: GPUDevice,
  target: GPUBuffer,
  data: ArrayBufferView,
  offset = 0,
): void {
  device.queue.writeBuffer(target, offset, data.buffer, data.byteOffset, data.byteLength);
}

/**
 * Calculate byte size for a tensor shape and dtype.
 */
export function tensorBytes(shape: number[], dtype: string): number {
  const elements = shape.reduce((a, b) => a * b, 1);
  const bytesPerElement: Record<string, number> = {
    f32: 4, f16: 2, bf16: 2, i32: 4, u32: 4, i8: 1, u8: 1, i4: 0.5,
  };
  return Math.ceil(elements * (bytesPerElement[dtype] ?? 4));
}
