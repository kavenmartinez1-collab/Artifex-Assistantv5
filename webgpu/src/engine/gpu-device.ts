/**
 * WebGPU Device Initialization
 *
 * Discovers available GPU adapters and creates a device with maximum limits
 * for LLM inference. Supports multi-GPU selection (e.g., NVIDIA + AMD).
 */

import { reportMetric, reportError } from '../utils/metrics';

export interface GPUContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  adapterInfo: GPUAdapterInfo;
  limits: GPUSupportedLimits;
  maxBufferSize: number;
}

/** Discovered GPU adapter with its metadata. */
export interface DiscoveredAdapter {
  adapter: GPUAdapter;
  info: GPUAdapterInfo;
  label: string;
  maxBufferMB: number;
}

/**
 * Discover all available GPU adapters by probing with different preferences.
 * Returns unique adapters (deduplicated by device name).
 */
export async function discoverAdapters(): Promise<DiscoveredAdapter[]> {
  if (!navigator.gpu) return [];

  const seen = new Map<string, DiscoveredAdapter>();
  const preferences: GPUPowerPreference[] = ['high-performance', 'low-power'];

  for (const pref of preferences) {
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: pref });
      if (!adapter) continue;

      const info = adapter.info ?? (await (adapter as any).requestAdapterInfo?.()) ?? {} as GPUAdapterInfo;
      const key = `${info.vendor}:${info.device || info.description || info.architecture || pref}`;

      if (!seen.has(key)) {
        const maxBuf = Math.min(adapter.limits.maxBufferSize, 2 * 1024 * 1024 * 1024);
        const label = info.device || info.description || info.architecture || `GPU (${pref})`;
        seen.set(key, { adapter, info, label, maxBufferMB: Math.round(maxBuf / (1024 * 1024)) });
      }
    } catch {
      // Adapter request failed for this preference, skip
    }
  }

  // Also try without preference hint (may return a different default)
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) {
      const info = adapter.info ?? (await (adapter as any).requestAdapterInfo?.()) ?? {} as GPUAdapterInfo;
      const key = `${info.vendor}:${info.device || info.description || info.architecture || 'default'}`;
      if (!seen.has(key)) {
        const maxBuf = Math.min(adapter.limits.maxBufferSize, 2 * 1024 * 1024 * 1024);
        const label = info.device || info.description || info.architecture || 'Default GPU';
        seen.set(key, { adapter, info, label, maxBufferMB: Math.round(maxBuf / (1024 * 1024)) });
      }
    }
  } catch { /* ignore */ }

  const results = [...seen.values()];
  console.log(`[GPU Discovery] Found ${results.length} unique adapter(s):`);
  for (const r of results) {
    console.log(`  - ${r.label} | vendor: ${r.info.vendor} | arch: ${r.info.architecture} | ${r.maxBufferMB} MB`);
  }
  if (results.length <= 1) {
    console.log(`[GPU Discovery] Note: WebGPU can only discover discrete vs integrated GPUs.`);
    console.log(`  To use a different discrete GPU, set it in Windows Settings > Display > Graphics > Chrome.`);
  }
  return results;
}

/**
 * Initialize WebGPU with a specific adapter (or auto-select high-performance).
 */
export async function initWebGPU(selectedAdapter?: GPUAdapter): Promise<GPUContext> {
  // Check WebGPU support
  if (!navigator.gpu) {
    const msg = 'WebGPU is not supported in this browser. Use Chrome 113+ or Edge 113+.';
    reportError('webgpu-init', new Error(msg));
    throw new Error(msg);
  }

  // Use selected adapter or request high-performance
  const adapter = selectedAdapter ?? await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance',
  });

  if (!adapter) {
    const msg = 'No WebGPU adapter found. Check that your GPU drivers are up to date.';
    reportError('webgpu-init', new Error(msg));
    throw new Error(msg);
  }

  const adapterInfo = adapter.info ?? (await (adapter as any).requestAdapterInfo?.()) ?? {} as GPUAdapterInfo;

  // Request device with maximum buffer sizes for model weights
  const adapterLimits = adapter.limits;

  // Request the largest buffer size the adapter supports (up to 2GB)
  const maxBufferSize = Math.min(
    adapterLimits.maxBufferSize,
    2 * 1024 * 1024 * 1024 // 2 GB cap
  );
  const maxStorageBinding = Math.min(
    adapterLimits.maxStorageBufferBindingSize,
    2 * 1024 * 1024 * 1024
  );

  // Request the timestamp-query feature if supported — enables per-dispatch
  // kernel timing for the diagnostic profiler. Graceful no-op if unavailable.
  const requiredFeatures: GPUFeatureName[] = [];
  if (adapter.features.has('timestamp-query')) {
    requiredFeatures.push('timestamp-query');
  }

  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxBufferSize,
      maxStorageBufferBindingSize: maxStorageBinding,
      maxComputeWorkgroupSizeX: adapterLimits.maxComputeWorkgroupSizeX,
      maxComputeWorkgroupSizeY: adapterLimits.maxComputeWorkgroupSizeY,
      maxComputeWorkgroupSizeZ: adapterLimits.maxComputeWorkgroupSizeZ,
      maxComputeInvocationsPerWorkgroup: adapterLimits.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupsPerDimension: adapterLimits.maxComputeWorkgroupsPerDimension,
      maxStorageBuffersPerShaderStage: adapterLimits.maxStorageBuffersPerShaderStage,
      maxUniformBufferBindingSize: adapterLimits.maxUniformBufferBindingSize,
      maxComputeWorkgroupStorageSize: adapterLimits.maxComputeWorkgroupStorageSize,
    },
  });

  // Handle device loss
  device.lost.then((info) => {
    reportError('device-lost', new Error(`GPU device lost: ${info.reason} — ${info.message}`));
    console.error('WebGPU device lost:', info);
  });

  // Report GPU info to dev server
  const gpuInfo = {
    vendor: adapterInfo.vendor,
    architecture: adapterInfo.architecture,
    device: adapterInfo.device,
    description: adapterInfo.description,
    maxBufferSize_MB: Math.round(maxBufferSize / (1024 * 1024)),
    maxStorageBinding_MB: Math.round(maxStorageBinding / (1024 * 1024)),
    maxComputeWorkgroupSizeX: device.limits.maxComputeWorkgroupSizeX,
    maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
    maxStorageBuffers: device.limits.maxStorageBuffersPerShaderStage,
  };

  await reportMetric('webgpu-init', gpuInfo);
  console.log('[WebGPU] Device initialized:', gpuInfo);

  return {
    adapter,
    device,
    adapterInfo,
    limits: device.limits,
    maxBufferSize,
  };
}
