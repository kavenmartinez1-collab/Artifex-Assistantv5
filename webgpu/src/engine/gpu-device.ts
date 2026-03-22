/**
 * WebGPU Device Initialization
 *
 * Requests adapter and device with maximum limits for LLM inference.
 * Reports GPU capabilities back via metrics.
 */

import { reportMetric, reportError } from '../utils/metrics';

export interface GPUContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  adapterInfo: GPUAdapterInfo;
  limits: GPUSupportedLimits;
  maxBufferSize: number;
}

export async function initWebGPU(): Promise<GPUContext> {
  // Check WebGPU support
  if (!navigator.gpu) {
    const msg = 'WebGPU is not supported in this browser. Use Chrome 113+ or Edge 113+.';
    reportError('webgpu-init', new Error(msg));
    throw new Error(msg);
  }

  // Request adapter (high-performance GPU preferred)
  const adapter = await navigator.gpu.requestAdapter({
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

  const device = await adapter.requestDevice({
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
