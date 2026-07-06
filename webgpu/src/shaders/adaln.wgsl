// FLUX.2 DiT modulation / activation kernels.
//
// The DiT computes all modulation vectors once per step from
// Linear(SiLU(temb)) into one flat buffer; entries here index into it via
// element offsets (off_a / off_b) so a single Mod binding serves every
// block without re-uploading.
//
// Entries (all: one thread per output element, X rows of width `dim`
// except swiglu_gate where X rows are 2*dim):
//   adaln_modulate  Out[r,d] = X[r,d] * (1 + Mod[off_b + d]) + Mod[off_a + d]
//                   (X = layer-normed input; off_a = shift, off_b = scale)
//   swiglu_gate     Out[r,d] = silu(X[r,d]) * X[r,dim+d]   (Flux2SwiGLU)
//   gate_add        Out[r,d] = X[r,d] + Mod[off_a + d] * Y[r,d]
//                   (gated residual; off_a = gate)
//   concat_cols     Out[r, off_a + d] = X[r, d] with Out rows of width off_b
//                   (column-band copy; builds the single-block [attn|mlp]
//                   concat without a strided GEMM)
//
// layout 'auto' drops bindings an entry doesn't reference: bind only what
// the entry uses (adaln_modulate/swiglu_gate skip Y; swiglu_gate skips Mod).
//
// Dispatch: (ceil(dim / 256), n_rows, 1)

struct Params {
  n_rows: u32,
  dim: u32,     // OUTPUT feature dim
  off_a: u32,   // modulate: shift offset | gate_add: gate offset
  off_b: u32,   // modulate: scale offset
}

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Y: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@group(0) @binding(3) var<storage, read> Mod: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn adaln_modulate(@builtin(workgroup_id) wid: vec3u,
                  @builtin(local_invocation_id) lid: vec3u) {
  let d = wid.x * 256u + lid.x;
  if (d >= params.dim) { return; }
  let r = wid.y;
  let i = r * params.dim + d;
  Out[i] = X[i] * (1.0 + Mod[params.off_b + d]) + Mod[params.off_a + d];
}

@compute @workgroup_size(256)
fn swiglu_gate(@builtin(workgroup_id) wid: vec3u,
               @builtin(local_invocation_id) lid: vec3u) {
  let d = wid.x * 256u + lid.x;
  if (d >= params.dim) { return; }
  let r = wid.y;
  let a = X[r * 2u * params.dim + d];
  let b = X[r * 2u * params.dim + params.dim + d];
  Out[r * params.dim + d] = (a / (1.0 + exp(-a))) * b;
}

@compute @workgroup_size(256)
fn gate_add(@builtin(workgroup_id) wid: vec3u,
            @builtin(local_invocation_id) lid: vec3u) {
  let d = wid.x * 256u + lid.x;
  if (d >= params.dim) { return; }
  let r = wid.y;
  let i = r * params.dim + d;
  Out[i] = X[i] + Mod[params.off_a + d] * Y[i];
}

@compute @workgroup_size(256)
fn concat_cols(@builtin(workgroup_id) wid: vec3u,
               @builtin(local_invocation_id) lid: vec3u) {
  let d = wid.x * 256u + lid.x;
  if (d >= params.dim) { return; }
  let r = wid.y;
  Out[r * params.off_b + params.off_a + d] = X[r * params.dim + d];
}
