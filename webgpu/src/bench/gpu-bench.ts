/**
 * GPU sync/transfer microbenchmarks — Phase 0a/0b.
 *
 * (a) Blocking 8 KB readback latency: dispatch → copy → submit → mapAsync.
 *     This is the structural per-layer cost of the CPU-expert MoE design
 *     (40 of these per decoded token). Measured two ways:
 *       - reused persistent staging buffer (the Phase C staging-ring design)
 *       - fresh staging buffer per call (current buffers.ts readBuffer behavior)
 * (b) writeBuffer host→GPU bandwidth at small (0.7 MB ≈ one expert matrix)
 *     and large (512 MB ≈ one layer's full expert set) granularity.
 */

export interface ReadbackResult {
  reused: { meanMs: number; p50Ms: number; p95Ms: number };
  fresh: { meanMs: number; p50Ms: number; p95Ms: number };
  iterations: number;
}

export interface WriteBwResult {
  smallChunkGBps: number; // 0.7 MB chunks
  largeChunkGBps: number; // single 512 MB write
  smallChunkMB: number;
  largeChunkMB: number;
}

const READBACK_BYTES = 8192; // one hidden state: 2048 f32
const SMALL_CHUNK = 720896; // one expert gate/up matrix at Q5_K
const LARGE_MB = 512;

function stats(times: number[]) {
  const sorted = [...times].sort((a, b) => a - b);
  return {
    meanMs: times.reduce((s, t) => s + t, 0) / times.length,
    p50Ms: sorted[Math.floor(sorted.length * 0.5)],
    p95Ms: sorted[Math.floor(sorted.length * 0.95)],
  };
}

export async function benchReadback(
  device: GPUDevice,
  iterations = 200,
  onStatus?: (s: string) => void
): Promise<ReadbackResult> {
  const module = device.createShaderModule({
    code: `
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < ${READBACK_BYTES / 4}u) { buf[id.x] = buf[id.x] + 1.0; }
}`,
  });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });
  const storage = device.createBuffer({
    size: READBACK_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: storage } }],
  });

  const oneIter = async (staging: GPUBuffer) => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(READBACK_BYTES / 4 / 64);
    pass.end();
    enc.copyBufferToBuffer(storage, 0, staging, 0, READBACK_BYTES);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    // touch the data like real code would
    const v = new Float32Array(staging.getMappedRange(), 0, 1)[0];
    staging.unmap();
    return v;
  };

  // ── reused persistent staging buffer ──
  onStatus?.('readback: reused staging buffer...');
  const persistent = device.createBuffer({
    size: READBACK_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  for (let i = 0; i < 20; i++) await oneIter(persistent); // warmup
  const reusedTimes: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const t0 = performance.now();
    await oneIter(persistent);
    reusedTimes.push(performance.now() - t0);
  }
  persistent.destroy();

  // ── fresh staging buffer per call (current readBuffer behavior) ──
  onStatus?.('readback: fresh staging buffer per call...');
  const freshTimes: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const staging = device.createBuffer({
      size: READBACK_BYTES,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const t0 = performance.now();
    await oneIter(staging);
    freshTimes.push(performance.now() - t0);
    staging.destroy();
  }

  storage.destroy();
  return { reused: stats(reusedTimes), fresh: stats(freshTimes), iterations };
}

export interface ReadbackVariantsResult {
  [variant: string]: { meanMs: number; p50Ms: number; p95Ms: number };
}

/**
 * Experiment: can Chrome's ~3 ms mapAsync floor be reduced?
 * Variants:
 *  - naive: submit → mapAsync → await (baseline, same as benchReadback)
 *  - workDoneFirst: await onSubmittedWorkDone() before mapAsync
 *  - pumpEmpty: while map pending, queue.submit([]) + fast MessageChannel yield
 *  - pumpWrite: while map pending, 4-byte writeBuffer + fast yield (forces real ticks)
 */
export async function benchReadbackVariants(
  device: GPUDevice,
  iterations = 200,
  onStatus?: (s: string) => void
): Promise<ReadbackVariantsResult> {
  const module = device.createShaderModule({
    code: `
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < ${READBACK_BYTES / 4}u) { buf[id.x] = buf[id.x] + 1.0; }
}`,
  });
  const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });
  const storage = device.createBuffer({
    size: READBACK_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: storage } }],
  });
  const staging = device.createBuffer({
    size: READBACK_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const pumpDst = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST });
  const pumpSrc = new Uint32Array(1);

  // Sub-millisecond yield (setTimeout clamps to ~1 ms — useless here).
  const chan = new MessageChannel();
  chan.port1.start();
  const fastYield = () =>
    new Promise<void>((r) => {
      chan.port1.addEventListener('message', () => r(), { once: true });
      chan.port2.postMessage(0);
    });

  const submitWork = () => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(READBACK_BYTES / 4 / 64);
    pass.end();
    enc.copyBufferToBuffer(storage, 0, staging, 0, READBACK_BYTES);
    device.queue.submit([enc.finish()]);
  };

  const finishMap = () => {
    const v = new Float32Array(staging.getMappedRange(), 0, 1)[0];
    staging.unmap();
    return v;
  };

  const variants: Record<string, () => Promise<void>> = {
    naive: async () => {
      submitWork();
      await staging.mapAsync(GPUMapMode.READ);
      finishMap();
    },
    workDoneFirst: async () => {
      submitWork();
      await device.queue.onSubmittedWorkDone();
      await staging.mapAsync(GPUMapMode.READ);
      finishMap();
    },
    pumpEmpty: async () => {
      submitWork();
      let done = false;
      const p = staging.mapAsync(GPUMapMode.READ).then(() => {
        done = true;
      });
      while (!done) {
        device.queue.submit([]);
        await fastYield();
      }
      await p;
      finishMap();
    },
    pumpWrite: async () => {
      submitWork();
      let done = false;
      const p = staging.mapAsync(GPUMapMode.READ).then(() => {
        done = true;
      });
      while (!done) {
        device.queue.writeBuffer(pumpDst, 0, pumpSrc);
        await fastYield();
      }
      await p;
      finishMap();
    },
  };

  const out: ReadbackVariantsResult = {};
  for (const [name, fn] of Object.entries(variants)) {
    onStatus?.(`readback variant: ${name}...`);
    for (let i = 0; i < 20; i++) await fn(); // warmup
    const times: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const t0 = performance.now();
      await fn();
      times.push(performance.now() - t0);
    }
    out[name] = stats(times);
  }

  staging.destroy();
  storage.destroy();
  pumpDst.destroy();
  return out;
}

export async function benchWriteBandwidth(
  device: GPUDevice,
  maxBufferSize: number,
  onStatus?: (s: string) => void
): Promise<WriteBwResult> {
  const largeBytes = Math.min(LARGE_MB * 1024 * 1024, maxBufferSize);
  const dst = device.createBuffer({ size: largeBytes, usage: GPUBufferUsage.COPY_DST });

  // ── small chunks: 512 × 0.7 MB at rotating offsets ──
  onStatus?.('writeBuffer: 0.7 MB chunks...');
  const small = new Uint8Array(SMALL_CHUNK);
  const nSmall = 512;
  const slots = Math.floor(largeBytes / SMALL_CHUNK);
  // warmup
  for (let i = 0; i < 16; i++) device.queue.writeBuffer(dst, (i % slots) * SMALL_CHUNK, small);
  await device.queue.onSubmittedWorkDone();
  let t0 = performance.now();
  for (let i = 0; i < nSmall; i++) {
    device.queue.writeBuffer(dst, (i % slots) * SMALL_CHUNK, small);
  }
  await device.queue.onSubmittedWorkDone();
  const smallGBps = (nSmall * SMALL_CHUNK) / 1e9 / ((performance.now() - t0) / 1000);

  // ── one large write ──
  onStatus?.(`writeBuffer: single ${Math.round(largeBytes / 1024 / 1024)} MB write...`);
  const large = new Uint8Array(largeBytes);
  device.queue.writeBuffer(dst, 0, large); // warmup
  await device.queue.onSubmittedWorkDone();
  let best = Infinity;
  for (let rep = 0; rep < 3; rep++) {
    t0 = performance.now();
    device.queue.writeBuffer(dst, 0, large);
    await device.queue.onSubmittedWorkDone();
    best = Math.min(best, performance.now() - t0);
  }
  const largeGBps = largeBytes / 1e9 / (best / 1000);

  dst.destroy();
  return {
    smallChunkGBps: smallGBps,
    largeChunkGBps: largeGBps,
    smallChunkMB: SMALL_CHUNK / 1024 / 1024,
    largeChunkMB: largeBytes / 1024 / 1024,
  };
}
