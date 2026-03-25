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
 * Results are cached by shader source hash.
 */
export function createComputePipeline(
  device: GPUDevice,
  wgslSource: string,
  entryPoint = 'main',
  label = '',
): GPUComputePipeline {
  const cacheKey = `${label}:${entryPoint}:${simpleHash(wgslSource)}`;

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
    compute: { module, entryPoint },
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

// Simple string hash for cache keys
function simpleHash(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash + str.charCodeAt(i)) | 0;
  }
  return hash >>> 0;
}
