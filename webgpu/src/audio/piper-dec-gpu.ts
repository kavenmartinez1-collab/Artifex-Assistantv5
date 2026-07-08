/**
 * Piper HiFiGAN decoder — GPU path (Phase P6).
 *
 * Mirrors decForward() in piper.ts op-for-op on the GPU via piper_dec.wgsl.
 * The CPU decForward stays the reference; this is validated against it and the
 * onnxruntime fixture waveform through scripts/test-piper-dec-gpu.mts
 * (headed, relL2 <= 1e-3).
 *
 * Layout matches piper.ts: channels-first [C, T] flat f32; conv weights
 * PyTorch [Cout,Cin,K] flat, ConvTranspose weights [Cin,Cout,K] flat.
 */
import shader from '../shaders/piper_dec.wgsl?raw';
import { createComputePipeline, createBindGroup, dispatch, workgroupCount } from '../engine/compute';
import { createStorageBuffer, createUniformBuffer, readBuffer } from '../engine/buffers';
import type { PiperWeights } from './piper';

const RES_K = [3, 5, 7];
const RES_D = [[1, 2], [2, 6], [3, 12]];
const UPS_K = [16, 16, 8], UPS_S = [8, 8, 4];
const UPS_CIN = [256, 128, 64], UPS_COUT = [128, 64, 32];

function getPadding(k: number, d: number): number {
  return Math.floor((k * d - d) / 2);
}

interface P {
  cin?: number; cout?: number; tIn?: number; tOut?: number; k?: number;
  pad?: number; dilation?: number; stride?: number; hasBias?: number;
  n?: number; slope?: number;
}

export class PiperDecGpu {
  private device: GPUDevice;
  private w = new Map<string, GPUBuffer>();
  private zeroBias: GPUBuffer;
  private pipe: Record<string, GPUComputePipeline> = {};

  constructor(device: GPUDevice, weights: PiperWeights) {
    this.device = device;
    for (const [name, t] of weights) {
      if (!name.startsWith('dec.')) continue;
      this.w.set(name, createStorageBuffer(device, t.data, t.data.byteLength, name));
    }
    this.zeroBias = createStorageBuffer(device, new Float32Array(1), 4, 'zeroBias');
    for (const e of ['conv1d', 'convt1d', 'leaky', 'add_ip', 'scale', 'tanh_act']) {
      this.pipe[e] = createComputePipeline(device, shader, e, `piperdec-${e}`);
    }
  }

  private wb(name: string): GPUBuffer {
    const b = this.w.get(name);
    if (!b) throw new Error(`piper-dec-gpu: missing weight ${name}`);
    return b;
  }

  private params(p: P): GPUBuffer {
    const buf = new ArrayBuffer(48);
    const u = new Uint32Array(buf), f = new Float32Array(buf);
    u[0] = p.cin ?? 0; u[1] = p.cout ?? 0; u[2] = p.tIn ?? 0; u[3] = p.tOut ?? 0;
    u[4] = p.k ?? 0; u[5] = p.pad ?? 0; u[6] = p.dilation ?? 1; u[7] = p.stride ?? 1;
    u[8] = p.hasBias ?? 0; u[9] = p.n ?? 0; f[10] = p.slope ?? 0; u[11] = 0;
    return createUniformBuffer(this.device, new Uint8Array(buf), 'piperdec-params');
  }

  private run(
    entry: string,
    binds: Array<{ binding: number; buffer: GPUBuffer }>,
    wg: [number, number?, number?],
  ): void {
    const pl = this.pipe[entry];
    const bg = createBindGroup(this.device, pl, 0,
      binds.map(b => ({ binding: b.binding, resource: { buffer: b.buffer } })));
    dispatch(this.device, pl, [bg], wg, `piperdec-${entry}`);
  }

  /** Dense/dilated Conv1d: [cin,tIn] -> [cout,tOut]. bias | null (=> zeroBias). */
  private conv1d(
    src: GPUBuffer, dst: GPUBuffer, wName: string, bName: string | null,
    cin: number, cout: number, tIn: number, k: number, pad: number, dilation: number,
  ): number {
    const tOut = Math.floor((tIn + 2 * pad - (dilation * (k - 1) + 1)) / 1) + 1;
    const bias = bName ? this.wb(bName) : this.zeroBias;
    const pu = this.params({ cin, cout, tIn, tOut, k, pad, dilation, stride: 1, hasBias: bName ? 1 : 0 });
    this.run('conv1d', [
      { binding: 0, buffer: src }, { binding: 1, buffer: this.wb(wName) },
      { binding: 2, buffer: bias }, { binding: 3, buffer: dst }, { binding: 4, buffer: pu },
    ], [workgroupCount(tOut, 256), cout, 1]);
    return tOut;
  }

  /** ConvTranspose1d: [cin,tIn] -> [cout,tOut]. */
  private convt1d(
    src: GPUBuffer, dst: GPUBuffer, wName: string, bName: string,
    cin: number, cout: number, tIn: number, k: number, s: number, pad: number,
  ): number {
    const tOut = (tIn - 1) * s - 2 * pad + k;
    const pu = this.params({ cin, cout, tIn, tOut, k, pad, stride: s, hasBias: 1 });
    this.run('convt1d', [
      { binding: 0, buffer: src }, { binding: 1, buffer: this.wb(wName) },
      { binding: 2, buffer: this.wb(bName) }, { binding: 3, buffer: dst }, { binding: 4, buffer: pu },
    ], [workgroupCount(tOut, 256), cout, 1]);
    return tOut;
  }

  private leaky(src: GPUBuffer, dst: GPUBuffer, n: number, slope: number): void {
    const pu = this.params({ n, slope });
    this.run('leaky', [
      { binding: 0, buffer: src }, { binding: 3, buffer: dst }, { binding: 4, buffer: pu },
    ], [workgroupCount(n, 256), 1, 1]);
  }

  /** dst += b (in place). */
  private addIp(dst: GPUBuffer, b: GPUBuffer, n: number): void {
    const pu = this.params({ n });
    this.run('add_ip', [
      { binding: 1, buffer: b }, { binding: 3, buffer: dst }, { binding: 4, buffer: pu },
    ], [workgroupCount(n, 256), 1, 1]);
  }

  private scale(src: GPUBuffer, dst: GPUBuffer, n: number, slope: number): void {
    const pu = this.params({ n, slope });
    this.run('scale', [
      { binding: 0, buffer: src }, { binding: 3, buffer: dst }, { binding: 4, buffer: pu },
    ], [workgroupCount(n, 256), 1, 1]);
  }

  private tanh(src: GPUBuffer, dst: GPUBuffer, n: number): void {
    const pu = this.params({ n });
    this.run('tanh_act', [
      { binding: 0, buffer: src }, { binding: 3, buffer: dst }, { binding: 4, buffer: pu },
    ], [workgroupCount(n, 256), 1, 1]);
  }

  /** z [192, F] -> mono waveform Float32Array of length F*256. */
  async forward(z: Float32Array, F: number): Promise<Float32Array> {
    const d = this.device;
    const peak = 8192 * F; // 32ch * 256*F (last stage) is the largest tensor
    const mk = (label: string) => createStorageBuffer(d, null, peak * 4, label, true);
    const A = mk('A'), B = mk('B'), C = mk('C'), D = mk('D'), E = mk('E');
    const zBuf = createStorageBuffer(d, z, z.byteLength, 'z');

    // conv_pre: 192 -> 256, k7 pad3
    let T = this.conv1d(zBuf, A, 'dec.conv_pre.weight', 'dec.conv_pre.bias', 192, 256, F, 7, 3, 1);
    // T === F ; x lives in A [256, F]

    for (let i = 0; i < 3; i++) {
      const Cin = UPS_CIN[i], Cout = UPS_COUT[i], K = UPS_K[i], S = UPS_S[i];
      // leaky(x) then upsample. x is in A; write leaky to B, upsample B -> A.
      this.leaky(A, B, Cin * T, 0.1);
      T = this.convt1d(B, A, `dec.ups.${i}.weight`, `dec.ups.${i}.bias`,
        Cin, Cout, T, K, S, Math.floor((K - S) / 2));
      // A is now the shared MRF input [Cout, T]; accumulate resblocks into C.
      const n = Cout * T;
      for (let j = 0; j < 3; j++) {
        const prefix = `dec.resblocks.${i * 3 + j}`;
        const K2 = RES_K[j], dils = RES_D[j];
        // resBlock: out = copy(A) -> D ; then out += conv(leaky(out)) twice
        this.scale(A, D, n, 1.0);
        for (let ii = 0; ii < 2; ii++) {
          const dd = dils[ii], pp = getPadding(K2, dd);
          this.leaky(D, B, n, 0.1);
          this.conv1d(B, E, `${prefix}.convs.${ii}.weight`, `${prefix}.convs.${ii}.bias`,
            Cout, Cout, T, K2, pp, dd);
          this.addIp(D, E, n);
        }
        if (j === 0) this.scale(D, C, n, 1.0);
        else this.addIp(C, D, n);
      }
      this.scale(C, A, n, 1 / 3); // x = mean(resblocks) -> A for next stage
    }

    // final leaky uses F.leaky_relu default slope 0.01 (not 0.1)
    this.leaky(A, B, 32 * T, 0.01);
    // conv_post: 32 -> 1, k7 pad3, NO bias
    const Twav = this.conv1d(B, A, 'dec.conv_post.weight', null, 32, 1, T, 7, 3, 1);
    this.tanh(A, B, Twav);

    const bytes = await readBuffer(d, B, Twav * 4);
    const wav = new Float32Array(bytes.slice(0, Twav * 4));
    for (const b of [A, B, C, D, E, zBuf]) b.destroy();
    return wav;
  }

  destroy(): void {
    for (const b of this.w.values()) b.destroy();
    this.zeroBias.destroy();
  }
}
