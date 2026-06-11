// Gemma 4 vision tower ops (spec extracted from llama.cpp clip_graph_gemma4v):
//   clamp_op       — Gemma4ClippableLinear activation clamps (QAT calibration
//                    ranges ship as scalar tensors; values baked into params)
//   avgpool2d      — Gemma4VisionPooler: kernel×kernel average pool over the
//                    raster patch grid, with an output scale (√n_embd folded in)
//   vision_rope_xy — per-block 2D RoPE, NEOX pairing within each HALF of the
//                    head: dims [0, half) roped with pos_x phases, dims
//                    [half, 2·half) with pos_y phases (theta 100)
//
// Each entry point uses its own binding slots so the module validates as a
// whole (auto layout per entry point only sees its referenced bindings).

// ── clamp_op: bindings 0-2 ───────────────────────────────────────────────

struct ClampParams {
  n: u32,
  lo: f32,
  hi: f32,
}

@group(0) @binding(0) var<storage, read> cin: array<f32>;
@group(0) @binding(1) var<storage, read_write> cout: array<f32>;
@group(0) @binding(2) var<uniform> cparams: ClampParams;

@compute @workgroup_size(256)
fn clamp_op(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= cparams.n) { return; }
  cout[idx] = clamp(cin[idx], cparams.lo, cparams.hi);
}

// ── avgpool2d: bindings 3-5 ──────────────────────────────────────────────
// input:  [gridH * gridW, channels] raster rows
// output: [outH * outW, channels], out(y,x) = mean of kernel² window, ×scale

struct PoolParams {
  grid_w: u32,
  grid_h: u32,
  channels: u32,
  kernel: u32,
  out_w: u32,
  out_h: u32,
  scale: f32,
}

@group(0) @binding(3) var<storage, read> pool_in: array<f32>;
@group(0) @binding(4) var<storage, read_write> pool_out: array<f32>;
@group(0) @binding(5) var<uniform> pparams: PoolParams;

@compute @workgroup_size(256)
fn avgpool2d(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = pparams.out_w * pparams.out_h * pparams.channels;
  if (idx >= total) { return; }

  let c = idx % pparams.channels;
  let cell = idx / pparams.channels;
  let ox = cell % pparams.out_w;
  let oy = cell / pparams.out_w;

  var sum: f32 = 0.0;
  for (var ky: u32 = 0u; ky < pparams.kernel; ky++) {
    for (var kx: u32 = 0u; kx < pparams.kernel; kx++) {
      let gy = oy * pparams.kernel + ky;
      let gx = ox * pparams.kernel + kx;
      sum += pool_in[(gy * pparams.grid_w + gx) * pparams.channels + c];
    }
  }
  let k2 = f32(pparams.kernel * pparams.kernel);
  pool_out[idx] = (sum / k2) * pparams.scale;
}

// ── vision_rope_xy: bindings 6-9 ─────────────────────────────────────────
// Per patch row, phases hold [θx_0..θx_{q-1}, θy_0..θy_{q-1}] (q = head_dim/4).
// Each HALF of the head (width 2q) is NEOX-roped independently:
//   within a half, dim j pairs with j+q:
//     out_j     = x_j·cosθ_j − x_{j+q}·sinθ_j        (j <  q)
//     out_{j+q} = x_{j+q}·cosθ_j + x_j·sinθ_j
// First half uses θx, second half uses θy.

struct RopeXYParams {
  n_rows: u32,     // number of patches
  width: u32,      // heads * head_dim (row stride)
  head_dim: u32,
}

@group(0) @binding(6) var<storage, read> rin: array<f32>;
@group(0) @binding(7) var<storage, read_write> rout: array<f32>;
@group(0) @binding(8) var<storage, read> rphases: array<f32>;  // [n_rows, head_dim/2]
@group(0) @binding(9) var<uniform> rparams: RopeXYParams;

@compute @workgroup_size(256)
fn vision_rope_xy(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = rparams.n_rows * rparams.width;
  if (idx >= total) { return; }

  let row = idx / rparams.width;
  let within = idx % rparams.width;
  let d = within % rparams.head_dim;
  let half = rparams.head_dim / 2u;   // 32 for head_dim 64
  let q = half / 2u;                  // 16 pairs per half

  let inHalf = d % half;              // position within this half
  let isY = d >= half;                // second half ropes with pos_y
  let j = inHalf % q;                 // frequency index
  let phaseBase = row * half + select(0u, q, isY);
  let theta = rphases[phaseBase + j];
  let c = cos(theta);
  let s = sin(theta);

  let x = rin[idx];
  var outv: f32;
  if (inHalf < q) {
    outv = x * c - rin[idx + q] * s;
  } else {
    outv = x * c + rin[idx - q] * s;
  }
  rout[idx] = outv;
}
