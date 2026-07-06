/**
 * Phase 1 gate: FLUX.2 WGSL kernel parity.
 *
 * Node-side (no GPU):
 *   - f64 RoPE table generation (theta 2000, axes [32,32,32,32],
 *     repeat_interleave_real) vs fixture pos_embed cos/sin dumps
 *     (dit.cap.pos_embed.call0 = img ids, call1 = txt ids).
 *
 * GPU (headed Chrome on the 6700 XT by default; all shapes tiny, no TDR risk):
 *   - rope_pairs.wgsl vs CPU adjacent-pair rotation using the FIXTURE txt
 *     cos/sin table (512x128) at H=24, D=128
 *   - conv2d.wgsl: conv2d_3x3 (rect H!=W), conv2d_1x1, upsample_nearest_2x
 *   - adaln.wgsl: adaln_modulate, swiglu_gate, gate_add
 *   - attention.wgsl valid_len key-mask (padded-key softmax) + a
 *     valid_len=0 regression against the full non-causal reference
 *
 * Tolerance: rel-L2 <= 1e-5 per kernel (plan parity policy).
 *
 * Run: npx tsx scripts/test-flux2-kernels.mts
 *      HEADLESS=1 npx tsx scripts/test-flux2-kernels.mts
 */
import { chromium } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const shaders = {
  rope: readFileSync(resolve(here, '../src/shaders/rope_pairs.wgsl'), 'utf8'),
  conv: readFileSync(resolve(here, '../src/shaders/conv2d.wgsl'), 'utf8'),
  adaln: readFileSync(resolve(here, '../src/shaders/adaln.wgsl'), 'utf8'),
  attn: readFileSync(resolve(here, '../src/shaders/attention.wgsl'), 'utf8'),
};

// ---------- fixture loading ----------
const fixDir = resolve(here, 'flux2_fixture');
const manifest = JSON.parse(readFileSync(resolve(fixDir, 'manifest.json'), 'utf8'));
function loadTensor(name: string): { data: Float32Array | Int32Array; shape: number[] } {
  const t = manifest.tensors[name];
  if (!t) throw new Error(`fixture tensor missing: ${name}`);
  const raw = readFileSync(resolve(fixDir, t.file));
  const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  const n = t.shape.reduce((a: number, b: number) => a * b, 1);
  const data = t.dtype === 'i32' ? new Int32Array(buf) : new Float32Array(buf);
  if (data.length !== n) throw new Error(`${name}: got ${data.length} elems, want ${n}`);
  return { data, shape: t.shape };
}

const relL2 = (got: ArrayLike<number>, want: ArrayLike<number>) => {
  let num = 0, den = 0;
  for (let i = 0; i < want.length; i++) {
    const d = got[i] - want[i];
    num += d * d;
    den += want[i] * want[i];
  }
  return Math.sqrt(num / Math.max(den, 1e-30));
};

let failed = false;
const report = (name: string, err: number, tol = 1e-5) => {
  const ok = err <= tol && Number.isFinite(err);
  if (!ok) failed = true;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name}: relL2=${err.toExponential(2)} (tol ${tol.toExponential(0)})`);
};

// ---------- 1) CPU f64 RoPE table gen vs fixture ----------
// diffusers get_1d_rotary_pos_embed(dim, pos, theta, repeat_interleave_real=True):
//   freqs[k] = 1 / theta^(2k / dim), k = 0..dim/2-1; angle = pos * freq;
//   cos/sin repeat_interleave(2) -> dim values; concat over the 4 axes.
const AXES = [32, 32, 32, 32];
const THETA = 2000;
function genRopeTable(ids: Int32Array, S: number): { cos: Float64Array; sin: Float64Array } {
  const D = AXES.reduce((a, b) => a + b, 0);
  const cos = new Float64Array(S * D), sin = new Float64Array(S * D);
  for (let s = 0; s < S; s++) {
    let off = 0;
    for (let a = 0; a < AXES.length; a++) {
      const dim = AXES[a];
      const pos = ids[s * AXES.length + a];
      for (let k = 0; k < dim / 2; k++) {
        const ang = pos / Math.pow(THETA, (2 * k) / dim);
        const c = Math.cos(ang), si = Math.sin(ang);
        const i = s * D + off + 2 * k;
        cos[i] = c; cos[i + 1] = c;
        sin[i] = si; sin[i + 1] = si;
      }
      off += dim;
    }
  }
  return { cos, sin };
}

console.log('node-side: RoPE table generation vs fixture');
for (const [call, label] of [['call0', 'img ids 256pos'], ['call1', 'txt ids 512pos']] as const) {
  const ids = loadTensor(`dit.cap.pos_embed.${call}.ids`);
  const wantCos = loadTensor(`dit.cap.pos_embed.${call}.cos`);
  const wantSin = loadTensor(`dit.cap.pos_embed.${call}.sin`);
  const S = ids.shape[0];
  const t = genRopeTable(ids.data as Int32Array, S);
  report(`rope-table ${label} cos`, relL2(t.cos, wantCos.data));
  report(`rope-table ${label} sin`, relL2(t.sin, wantSin.data));
}

// ---------- 2) GPU kernel parity ----------
const txtCos = loadTensor('dit.cap.pos_embed.call1.cos').data as Float32Array;
const txtSin = loadTensor('dit.cap.pos_embed.call1.sin').data as Float32Array;

const HEADLESS = process.env.HEADLESS === '1';
const browser = await chromium.launch({
  channel: 'chrome',
  headless: HEADLESS,
  args: ['--enable-unsafe-webgpu', '--window-size=360,240', '--window-position=20,20'],
});

try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
  await page.route('**/*', (route) =>
    route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><meta charset=utf8><title>flux2-kernels</title>' }));
  await page.goto('http://127.0.0.1/');

  const res: any = await page.evaluate(async ({ shaders, txtCos, txtSin }) => {
    (globalThis as any).__name = (f: any) => f; // tsx/esbuild keepNames shim
    const g = (navigator as any).gpu;
    if (!g) return { error: 'no navigator.gpu' };
    const adapter = await g.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { error: 'no adapter' };
    const device = await adapter.requestDevice();
    let lost: string | null = null;
    device.lost.then((l: any) => { lost = `${l.reason}: ${l.message}`; });

    const pipelines: Record<string, any> = {};
    for (const [key, entries] of [
      ['rope', ['rope_pairs']],
      ['conv', ['conv2d_3x3', 'conv2d_1x1', 'upsample_nearest_2x']],
      ['adaln', ['adaln_modulate', 'swiglu_gate', 'gate_add']],
      ['attn', ['attention']],
    ] as const) {
      const module = device.createShaderModule({ code: (shaders as any)[key] });
      const ci = await module.getCompilationInfo();
      const errs = ci.messages.filter((m: any) => m.type === 'error').map((m: any) => `${key}:${m.lineNum}: ${m.message}`);
      if (errs.length) return { error: `shader errors:\n${errs.join('\n')}` };
      for (const e of entries) {
        pipelines[e] = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: e } });
      }
    }

    let seed = 0xF1A2C3 >>> 0;
    const rng = () => {
      seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const rand = (n: number, scale = 1) => {
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = (rng() * 2 - 1) * scale;
      return a;
    };

    const stor = (data: Float32Array | ArrayBuffer, extra = 0) => {
      const b = device.createBuffer({
        size: (data as any).byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extra,
      });
      device.queue.writeBuffer(b, 0, data as any);
      return b;
    };
    const outBuf = (bytes: number) => device.createBuffer({
      size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const uni = (data: ArrayBuffer | Uint32Array) => {
      const b = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(b, 0, data as any);
      return b;
    };
    const run = async (pipe: any, binds: { b: number; buf: any }[], gx: number, gy: number, read: any, n: number) => {
      const bg = device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: binds.map((e) => ({ binding: e.b, resource: { buffer: e.buf } })),
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(gx, gy, 1);
      pass.end();
      const rb = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      enc.copyBufferToBuffer(read, 0, rb, 0, n * 4);
      device.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(rb.getMappedRange()).slice();
      rb.unmap();
      rb.destroy();
      return out;
    };
    const relL2 = (got: ArrayLike<number>, want: ArrayLike<number>) => {
      let num = 0, den = 0;
      for (let i = 0; i < want.length; i++) {
        const d = got[i] - want[i];
        num += d * d;
        den += want[i] * want[i];
      }
      return Math.sqrt(num / Math.max(den, 1e-30));
    };

    const results: { name: string; err: number }[] = [];
    device.pushErrorScope('validation');

    // ---- rope_pairs: S=512, H=24, D=128, fixture txt table ----
    {
      const S = 512, H = 24, D = 128;
      const cos = new Float32Array(txtCos), sin = new Float32Array(txtSin);
      const X = rand(S * H * D);
      const want = new Float64Array(S * H * D);
      for (let s = 0; s < S; s++) {
        for (let h = 0; h < H; h++) {
          for (let p = 0; p < D / 2; p++) {
            const i = (s * H + h) * D + 2 * p;
            const tb = s * D + 2 * p;
            want[i] = X[i] * cos[tb] - X[i + 1] * sin[tb];
            want[i + 1] = X[i + 1] * cos[tb + 1] + X[i] * sin[tb + 1];
          }
        }
      }
      const xB = stor(X, GPUBufferUsage.COPY_SRC);
      const got = await run(pipelines.rope_pairs, [
        { b: 0, buf: xB },
        { b: 1, buf: stor(cos) },
        { b: 2, buf: stor(sin) },
        { b: 3, buf: uni(new Uint32Array([H, D, S, 0])) },
      ], Math.ceil((H * D / 2) / 256), S, xB, S * H * D);
      results.push({ name: 'rope_pairs 512x24hx128d (fixture table)', err: relL2(got, want) });
    }

    // ---- conv2d_3x3: rectangular to catch H/W swaps ----
    {
      const Ci = 8, Co = 5, H = 7, W = 9;
      const X = rand(Ci * H * W), Wt = rand(Co * Ci * 9), B = rand(Co);
      const want = new Float64Array(Co * H * W);
      for (let oc = 0; oc < Co; oc++) {
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            let acc = B[oc];
            for (let ic = 0; ic < Ci; ic++) {
              for (let kh = 0; kh < 3; kh++) {
                for (let kw = 0; kw < 3; kw++) {
                  const iy = y + kh - 1, ix = x + kw - 1;
                  if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                  acc += X[ic * H * W + iy * W + ix] * Wt[((oc * Ci + ic) * 3 + kh) * 3 + kw];
                }
              }
            }
            want[oc * H * W + y * W + x] = acc;
          }
        }
      }
      const oB = outBuf(Co * H * W * 4);
      const got = await run(pipelines.conv2d_3x3, [
        { b: 0, buf: stor(X) },
        { b: 1, buf: stor(Wt) },
        { b: 2, buf: stor(B) },
        { b: 3, buf: oB },
        { b: 4, buf: uni(new Uint32Array([Ci, Co, H, W])) },
      ], Math.ceil((H * W) / 256), Co, oB, Co * H * W);
      results.push({ name: 'conv2d_3x3 8->5ch 7x9', err: relL2(got, want) });
    }

    // ---- conv2d_1x1 ----
    {
      const Ci = 16, Co = 6, H = 5, W = 11;
      const X = rand(Ci * H * W), Wt = rand(Co * Ci), B = rand(Co);
      const want = new Float64Array(Co * H * W);
      for (let oc = 0; oc < Co; oc++) {
        for (let p = 0; p < H * W; p++) {
          let acc = B[oc];
          for (let ic = 0; ic < Ci; ic++) acc += X[ic * H * W + p] * Wt[oc * Ci + ic];
          want[oc * H * W + p] = acc;
        }
      }
      const oB = outBuf(Co * H * W * 4);
      const got = await run(pipelines.conv2d_1x1, [
        { b: 0, buf: stor(X) },
        { b: 1, buf: stor(Wt) },
        { b: 2, buf: stor(B) },
        { b: 3, buf: oB },
        { b: 4, buf: uni(new Uint32Array([Ci, Co, H, W])) },
      ], Math.ceil((H * W) / 256), Co, oB, Co * H * W);
      results.push({ name: 'conv2d_1x1 16->6ch 5x11', err: relL2(got, want) });
    }

    // ---- upsample_nearest_2x ----
    {
      const C = 3, H = 5, W = 4;
      const X = rand(C * H * W);
      const want = new Float64Array(C * 4 * H * W);
      for (let c = 0; c < C; c++) {
        for (let oy = 0; oy < 2 * H; oy++) {
          for (let ox = 0; ox < 2 * W; ox++) {
            want[c * 4 * H * W + oy * 2 * W + ox] = X[c * H * W + (oy >> 1) * W + (ox >> 1)];
          }
        }
      }
      const oB = outBuf(C * 4 * H * W * 4);
      // conv entries reference Wt/Bias; upsample doesn't — bind only used
      const got = await run(pipelines.upsample_nearest_2x, [
        { b: 0, buf: stor(X) },
        { b: 3, buf: oB },
        { b: 4, buf: uni(new Uint32Array([C, 0, H, W])) },
      ], Math.ceil((4 * H * W) / 256), C, oB, C * 4 * H * W);
      results.push({ name: 'upsample_nearest_2x 3ch 5x4->10x8', err: relL2(got, want) });
    }

    // ---- adaln_modulate ----
    {
      const R = 10, D = 300; // dim > 256 exercises the 2-workgroup x path
      const X = rand(R * D);
      const mod = rand(1024);
      const shiftOff = 100, scaleOff = 600;
      const want = new Float64Array(R * D);
      for (let r = 0; r < R; r++) {
        for (let d = 0; d < D; d++) {
          want[r * D + d] = X[r * D + d] * (1 + mod[scaleOff + d]) + mod[shiftOff + d];
        }
      }
      const oB = outBuf(R * D * 4);
      const got = await run(pipelines.adaln_modulate, [
        { b: 0, buf: stor(X) },
        { b: 2, buf: oB },
        { b: 3, buf: stor(mod) },
        { b: 4, buf: uni(new Uint32Array([R, D, shiftOff, scaleOff])) },
      ], Math.ceil(D / 256), R, oB, R * D);
      results.push({ name: 'adaln_modulate 10x300', err: relL2(got, want) });
    }

    // ---- swiglu_gate ----
    {
      const R = 6, D = 288;
      const X = rand(R * 2 * D, 2);
      const want = new Float64Array(R * D);
      for (let r = 0; r < R; r++) {
        for (let d = 0; d < D; d++) {
          const a = X[r * 2 * D + d], b = X[r * 2 * D + D + d];
          want[r * D + d] = (a / (1 + Math.exp(-a))) * b;
        }
      }
      const oB = outBuf(R * D * 4);
      const got = await run(pipelines.swiglu_gate, [
        { b: 0, buf: stor(X) },
        { b: 2, buf: oB },
        { b: 4, buf: uni(new Uint32Array([R, D, 0, 0])) },
      ], Math.ceil(D / 256), R, oB, R * D);
      results.push({ name: 'swiglu_gate 6x288', err: relL2(got, want) });
    }

    // ---- gate_add ----
    {
      const R = 6, D = 300;
      const X = rand(R * D), Y = rand(R * D);
      const mod = rand(512);
      const gateOff = 40;
      const want = new Float64Array(R * D);
      for (let r = 0; r < R; r++) {
        for (let d = 0; d < D; d++) {
          want[r * D + d] = X[r * D + d] + mod[gateOff + d] * Y[r * D + d];
        }
      }
      const oB = outBuf(R * D * 4);
      const got = await run(pipelines.gate_add, [
        { b: 0, buf: stor(X) },
        { b: 1, buf: stor(Y) },
        { b: 2, buf: oB },
        { b: 3, buf: stor(mod) },
        { b: 4, buf: uni(new Uint32Array([R, D, gateOff, 0])) },
      ], Math.ceil(D / 256), R, oB, R * D);
      results.push({ name: 'gate_add 6x300', err: relL2(got, want) });
    }

    // ---- attention.wgsl valid_len mask ----
    {
      const H = 4, Dh = 32, S = 48;
      const Q = rand(S * H * Dh, 0.7), K = rand(S * H * Dh, 0.7), V = rand(S * H * Dh);
      const scale = 1 / Math.sqrt(Dh);
      const ref = (validLen: number) => {
        const nk = validLen > 0 ? validLen : S;
        const want = new Float64Array(S * H * Dh);
        for (let qi = 0; qi < S; qi++) {
          for (let h = 0; h < H; h++) {
            const sc = new Float64Array(nk);
            let m = -Infinity;
            for (let j = 0; j < nk; j++) {
              let dot = 0;
              for (let d = 0; d < Dh; d++) dot += Q[(qi * H + h) * Dh + d] * K[(j * H + h) * Dh + d];
              sc[j] = dot * scale;
              if (sc[j] > m) m = sc[j];
            }
            let l = 0;
            for (let j = 0; j < nk; j++) { sc[j] = Math.exp(sc[j] - m); l += sc[j]; }
            for (let d = 0; d < Dh; d++) {
              let acc = 0;
              for (let j = 0; j < nk; j++) acc += sc[j] * V[(j * H + h) * Dh + d];
              want[(qi * H + h) * Dh + d] = acc / l;
            }
          }
        }
        return want;
      };
      const attnRun = async (validLen: number) => {
        const p = new ArrayBuffer(48);
        const u = new Uint32Array(p), f = new Float32Array(p);
        u[0] = H; u[1] = H; u[2] = Dh; u[3] = S; u[4] = S;
        f[5] = scale;
        u[6] = 0; u[7] = 0; u[8] = 0; // non-causal, no offset, no window
        u[9] = validLen;
        const oB = outBuf(S * H * Dh * 4);
        return run(pipelines.attention, [
          { b: 0, buf: stor(Q) },
          { b: 1, buf: stor(K) },
          { b: 2, buf: stor(V) },
          { b: 3, buf: oB },
          { b: 4, buf: uni(p) },
        ], S, H, oB, S * H * Dh);
      };
      results.push({ name: 'attention valid_len=32 (48 keys, 16 pad)', err: relL2(await attnRun(32), ref(32)) });
      results.push({ name: 'attention valid_len=0 regression (full)', err: relL2(await attnRun(0), ref(0)) });
    }

    const vErr = await device.popErrorScope();
    if (vErr) return { error: `validation: ${vErr.message}` };
    if (lost) return { error: `device lost: ${lost}` };
    const info = adapter.info ?? {};
    return { adapter: `${info.vendor ?? '?'} ${info.architecture ?? '?'}`, results };
  }, { shaders, txtCos: Array.from(txtCos), txtSin: Array.from(txtSin) });

  if (res.error) {
    console.log(`FAIL: ${res.error}`);
    failed = true;
  } else {
    console.log(`\nGPU (${res.adapter}):`);
    for (const r of res.results) report(r.name, r.err);
  }
} finally {
  await browser.close();
}

console.log(`\n=== Phase 1 kernel gate: ${failed ? 'FAIL' : 'PASS'} ===`);
if (HEADLESS) console.log('NOTE: HEADLESS run — rerun headed on the 6700 XT before commit.');
process.exit(failed ? 1 : 0);
