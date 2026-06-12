/**
 * GPU greedy argmax — replaces the per-token full-vocab logits readback
 * (~600 KB mapAsync, ~5.4 ms/token on the RX 6700 XT) with a two-pass
 * reduction on the GPU and a 4-byte pumped readback.
 *
 * Semantics match the CPU greedy sampler exactly (max value, lowest index
 * on ties) — see argmax.wgsl and the JS parity port in
 * scripts/test-argmax.mts. Only used when sampling is greedy-neutral
 * (temperature 0, no repetition/DRY penalties, no debug probes).
 */

import argmaxWGSL from '../shaders/argmax.wgsl?raw';

const NWG = 256; // pass-1 workgroups — must mirror argmax.wgsl

export interface GpuArgmax {
  /** Argmax over the first n f32 values of logitsBuf (a storage buffer). */
  run(logitsBuf: GPUBuffer, n: number): Promise<number>;
  destroy(): void;
}

export function createGpuArgmax(device: GPUDevice): GpuArgmax {
  const module = device.createShaderModule({ code: argmaxWGSL, label: 'argmax' });
  const partialPipe = device.createComputePipeline({
    layout: 'auto', compute: { module, entryPoint: 'argmax_partial' }, label: 'argmax-partial',
  });
  const finalPipe = device.createComputePipeline({
    layout: 'auto', compute: { module, entryPoint: 'argmax_final' }, label: 'argmax-final',
  });

  const partialsBuf = device.createBuffer({
    size: NWG * 8, usage: GPUBufferUsage.STORAGE, label: 'argmax-partials',
  });
  const outBuf = device.createBuffer({
    size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, label: 'argmax-out',
  });
  const paramsBuf = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'argmax-params',
  });
  const staging = device.createBuffer({
    size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: 'argmax-staging',
  });

  // Auto layouts only include statically-referenced bindings:
  // partial uses logits(0)/partials(1)/params(3); final uses partials(1)/out_idx(2).
  // The final-pass bind group never changes; the partial-pass one is cached
  // per logits buffer (the engine reuses one logits buffer across steps).
  const finalBG = device.createBindGroup({
    layout: finalPipe.getBindGroupLayout(0),
    entries: [
      { binding: 1, resource: { buffer: partialsBuf } },
      { binding: 2, resource: { buffer: outBuf } },
    ],
    label: 'argmax-final-bg',
  });
  const partialBGCache = new Map<GPUBuffer, GPUBindGroup>();
  let lastN = -1;

  // mapAsync pump (same trick as forward-pass.ts): Chrome's mapAsync has a
  // ~3 ms resolution floor when the event loop idles; pumping a 4-byte
  // writeBuffer + sub-ms MessageChannel yield while the map is pending
  // forces real queue ticks → ~0.2 ms. Never await a bare mapAsync.
  const pumpDstBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST, label: 'argmax-pump' });
  const pumpSrcArr = new Uint32Array(1);
  const pumpChan = new MessageChannel();
  pumpChan.port1.start();
  const fastYield = () => new Promise<void>((resolve) => {
    pumpChan.port1.addEventListener('message', () => resolve(), { once: true });
    pumpChan.port2.postMessage(0);
  });
  async function mapWithPump(buf: GPUBuffer, bytes: number): Promise<void> {
    let done = false;
    const p = buf.mapAsync(GPUMapMode.READ, 0, bytes).then(() => { done = true; });
    while (!done) {
      device.queue.writeBuffer(pumpDstBuf, 0, pumpSrcArr);
      await fastYield();
    }
    await p;
  }

  async function run(logitsBuf: GPUBuffer, n: number): Promise<number> {
    if (n !== lastN) {
      device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([n, 0, 0, 0]));
      lastN = n;
    }
    let partialBG = partialBGCache.get(logitsBuf);
    if (!partialBG) {
      partialBG = device.createBindGroup({
        layout: partialPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: logitsBuf } },
          { binding: 1, resource: { buffer: partialsBuf } },
          { binding: 3, resource: { buffer: paramsBuf } },
        ],
        label: 'argmax-partial-bg',
      });
      partialBGCache.set(logitsBuf, partialBG);
    }
    const enc = device.createCommandEncoder({ label: 'argmax' });
    const pass = enc.beginComputePass();
    pass.setPipeline(partialPipe);
    pass.setBindGroup(0, partialBG);
    pass.dispatchWorkgroups(NWG);
    pass.setPipeline(finalPipe);
    pass.setBindGroup(0, finalBG);
    pass.dispatchWorkgroups(1);
    pass.end();
    enc.copyBufferToBuffer(outBuf, 0, staging, 0, 4);
    device.queue.submit([enc.finish()]);
    await mapWithPump(staging, 4);
    const idx = new Uint32Array(staging.getMappedRange(0, 4))[0];
    staging.unmap();
    return idx;
  }

  function destroy() {
    partialsBuf.destroy();
    outBuf.destroy();
    paramsBuf.destroy();
    staging.destroy();
    pumpDstBuf.destroy();
    partialBGCache.clear();
  }

  return { run, destroy };
}
