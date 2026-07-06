// Adjacent-pair RoPE (FLUX.2 / diffusers apply_rotary_emb with
// use_real=True, use_real_unbind_dim=-1):
//
//   out[2p]   = x[2p]   * cos[2p]   - x[2p+1] * sin[2p]
//   out[2p+1] = x[2p+1] * cos[2p+1] + x[2p]   * sin[2p+1]
//
// This is NOT the Llama half-split rotation in rope.wgsl (pair i with
// i + d/2); FLUX.2 rotates ADJACENT elements (2p, 2p+1).
//
// cos/sin are precomputed [seq, head_dim] tables (repeat-interleaved, so
// cos[2p] == cos[2p+1]; the kernel still indexes both slots so the table
// is the single source of truth). Tables are built on the CPU in f64 from
// the 4-axis position ids (theta 2000, axes_dims [32,32,32,32]) and
// uploaded once per resolution. The same table applies to every head of a
// row, and to both Q and K.
//
// X: [seq, num_heads * head_dim], rotated IN PLACE.
// Dispatch: (ceil(num_heads * head_dim / 2 / 256), seq, 1)

struct Params {
  num_heads: u32,
  head_dim: u32,   // even; table row width
  seq: u32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> CosT: array<f32>;
@group(0) @binding(2) var<storage, read> SinT: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn rope_pairs(@builtin(workgroup_id) wid: vec3u,
              @builtin(local_invocation_id) lid: vec3u) {
  let half_d = params.head_dim / 2u;
  let pairs_per_row = params.num_heads * half_d;
  let pi = wid.x * 256u + lid.x;
  if (pi >= pairs_per_row) { return; }
  let s = wid.y;

  let head = pi / half_d;
  let p = pi % half_d;
  let base = (s * params.num_heads + head) * params.head_dim + 2u * p;
  let tb = s * params.head_dim + 2u * p;

  let x0 = X[base];
  let x1 = X[base + 1u];
  X[base]      = x0 * CosT[tb]      - x1 * SinT[tb];
  X[base + 1u] = x1 * CosT[tb + 1u] + x0 * SinT[tb + 1u];
}
