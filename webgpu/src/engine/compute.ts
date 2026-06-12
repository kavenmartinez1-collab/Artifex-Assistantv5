/**
 * Shader Compilation & Compute Dispatch
 *
 * Compiles WGSL shaders, creates compute pipelines, and dispatches work.
 */

import { reportMetric, reportError } from '../utils/metrics';

// Cache compiled pipelines to avoid recompilation
const pipelineCache = new Map<string, GPUComputePipeline>();

/**
 * Compile a WGSL shader and create a compute pipeline.
 * Results are cached by shader source hash (and override constants if any).
 * `constants` supplies WGSL pipeline-override constant values, letting
 * specialized variants share one shader module.
 */
export function createComputePipeline(
  device: GPUDevice,
  wgslSource: string,
  entryPoint = 'main',
  label = '',
  constants?: Record<string, number>,
): GPUComputePipeline {
  const constantsKey = constants
    ? ':' + Object.keys(constants).sort().map(k => `${k}=${constants[k]}`).join(',')
    : '';
  const cacheKey = `${label}:${entryPoint}:${simpleHash(wgslSource)}${constantsKey}`;

  const cached = pipelineCache.get(cacheKey);
  if (cached) return cached;

  const module = device.createShaderModule({
    code: wgslSource,
    label: `${label}-module`,
  });

  // Log shader compilation errors (WebGPU silently fails otherwise)
  module.getCompilationInfo().then(info => {
    for (const msg of info.messages) {
      const prefix = msg.type === 'error' ? '❌' : msg.type === 'warning' ? '⚠️' : 'ℹ️';
      console.log(`[WGSL ${label}] ${prefix} ${msg.type} at line ${msg.lineNum}:${msg.linePos}: ${msg.message}`);
    }
  });

  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint, ...(constants ? { constants } : {}) },
    label,
  });

  pipelineCache.set(cacheKey, pipeline);
  return pipeline;
}

/**
 * Dispatch a compute shader with the given bind group(s).
 */
export function dispatch(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroups: GPUBindGroup[],
  workgroupCounts: [number, number?, number?],
  label = '',
): void {
  const encoder = device.createCommandEncoder({ label: `${label}-encoder` });
  const pass = encoder.beginComputePass({ label: `${label}-pass` });

  pass.setPipeline(pipeline);
  for (let i = 0; i < bindGroups.length; i++) {
    pass.setBindGroup(i, bindGroups[i]);
  }
  pass.dispatchWorkgroups(
    workgroupCounts[0],
    workgroupCounts[1] ?? 1,
    workgroupCounts[2] ?? 1,
  );
  pass.end();

  device.queue.submit([encoder.finish()]);
}

/**
 * Create a bind group from a pipeline layout and buffer entries.
 */
export function createBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  groupIndex: number,
  entries: Array<{ binding: number; resource: GPUBindingResource }>,
  label = '',
): GPUBindGroup {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(groupIndex),
    entries: entries.map(e => ({
      binding: e.binding,
      resource: e.resource,
    })),
    label,
  });
}

/**
 * Calculate workgroup count for a given total size and workgroup size.
 */
export function workgroupCount(totalSize: number, workgroupSize: number): number {
  return Math.ceil(totalSize / workgroupSize);
}

/**
 * Run a compute shader and wait for completion (synchronous for testing).
 */
export async function dispatchAndWait(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroups: GPUBindGroup[],
  workgroupCounts: [number, number?, number?],
  label = '',
): Promise<number> {
  const start = performance.now();

  dispatch(device, pipeline, bindGroups, workgroupCounts, label);
  await device.queue.onSubmittedWorkDone();

  const elapsed = performance.now() - start;
  return elapsed;
}

// Pass fusion: keep ONE compute pass open across consecutive dispatches
// instead of begin/end per dispatch (~950 open/close pairs per decode token).
// WebGPU inserts implicit barriers between dispatches within a single pass
// (each dispatch is its own usage scope), so ordering and memory visibility
// are identical to per-dispatch passes — this only removes encoding overhead.
// A/B toggle: ?fusePass=0 restores the old per-dispatch passes.
const passFusionEnabled = typeof window === 'undefined'
  || new URLSearchParams(window.location.search).get('fusePass') !== '0';

/**
 * Batched dispatcher — collects multiple compute dispatches into a single
 * command encoder and submits them all at once via flush().
 *
 * For a 32-layer model this reduces ~480 separate queue.submit() calls
 * down to 1, eliminating per-submit driver overhead. With pass fusion the
 * dispatches also share one compute pass (setPipeline only on change).
 */
export class BatchedDispatcher {
  private encoder: GPUCommandEncoder;
  private device: GPUDevice;
  private count = 0;
  private pass: GPUComputePassEncoder | null = null;
  private lastPipeline: GPUComputePipeline | null = null;

  constructor(device: GPUDevice, label?: string) {
    this.device = device;
    this.encoder = device.createCommandEncoder({ label: label ?? 'batched-encoder' });
  }

  /** End the open fused pass (before copies, query resolves, or finish). */
  private endPass(): void {
    if (this.pass) {
      this.pass.end();
      this.pass = null;
      this.lastPipeline = null;
    }
  }

  /** Add a compute dispatch to the batch (no GPU submission yet). */
  dispatch(
    pipeline: GPUComputePipeline,
    bindGroups: GPUBindGroup[],
    workgroupCounts: [number, number?, number?],
    label?: string,
    timestampWrites?: GPUComputePassTimestampWrites,
  ): void {
    // Timestamped dispatches need their own pass (timestampWrites are a
    // pass-level property); same for the ?fusePass=0 A/B fallback.
    if (timestampWrites || !passFusionEnabled) {
      this.endPass();
      const passLabel = label ? `${label}-pass` : `batch-pass-${this.count}`;
      const passDesc: GPUComputePassDescriptor = { label: passLabel };
      if (timestampWrites) passDesc.timestampWrites = timestampWrites;
      const pass = this.encoder.beginComputePass(passDesc);
      pass.setPipeline(pipeline);
      for (let i = 0; i < bindGroups.length; i++) {
        pass.setBindGroup(i, bindGroups[i]);
      }
      pass.dispatchWorkgroups(
        workgroupCounts[0],
        workgroupCounts[1] ?? 1,
        workgroupCounts[2] ?? 1,
      );
      pass.end();
      this.count++;
      return;
    }

    if (!this.pass) {
      this.pass = this.encoder.beginComputePass({ label: 'batch-fused-pass' });
      this.lastPipeline = null;
    }
    if (this.lastPipeline !== pipeline) {
      this.pass.setPipeline(pipeline);
      this.lastPipeline = pipeline;
    }
    for (let i = 0; i < bindGroups.length; i++) {
      this.pass.setBindGroup(i, bindGroups[i]);
    }
    this.pass.dispatchWorkgroups(
      workgroupCounts[0],
      workgroupCounts[1] ?? 1,
      workgroupCounts[2] ?? 1,
    );
    this.count++;
  }

  /** Encode a query-set resolve into the current batch (for timestamp profiling). */
  resolveQuerySet(
    querySet: GPUQuerySet,
    firstQuery: number,
    queryCount: number,
    destination: GPUBuffer,
    destinationOffset: number,
  ): void {
    this.endPass();
    this.encoder.resolveQuerySet(querySet, firstQuery, queryCount, destination, destinationOffset);
  }

  /** Add a buffer-to-buffer copy to the batch (no GPU submission yet). */
  copyBuffer(
    src: GPUBuffer, srcOffset: number,
    dst: GPUBuffer, dstOffset: number,
    size: number,
  ): void {
    this.endPass();
    this.encoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, size);
    this.count++;
  }

  /** Submit all batched dispatches to the GPU in a single queue.submit(). */
  flush(): void {
    this.endPass();
    if (this.count === 0) return;
    this.device.queue.submit([this.encoder.finish()]);
  }

  /** Reset the encoder for reuse after a flush (e.g. after debug reads). */
  reset(label?: string): void {
    this.pass = null;
    this.lastPipeline = null;
    this.encoder = this.device.createCommandEncoder({ label: label ?? 'batched-encoder' });
    this.count = 0;
  }

  /** Submit and wait for all GPU work to complete. Returns elapsed ms. */
  async flushAndWait(): Promise<number> {
    const start = performance.now();
    this.flush();
    await this.device.queue.onSubmittedWorkDone();
    return performance.now() - start;
  }

  /** Number of dispatches currently batched. */
  get size(): number {
    return this.count;
  }
}

// Simple string hash for cache keys
function simpleHash(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash + str.charCodeAt(i)) | 0;
  }
  return hash >>> 0;
}
