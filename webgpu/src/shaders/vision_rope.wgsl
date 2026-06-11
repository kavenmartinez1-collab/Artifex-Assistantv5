// Vision 2D rotary embeddings (Qwen3-VL ViT).
//
// Each patch row has `half` precomputed phase values (row/col position ×
// inv_freq, already flattened [row*f_0..row*f_{half/2-1}, col*f_0..
// col*f_{half/2-1}]). The rotation duplicates phases across both halves of
// each head (GPT-NeoX rotate_half convention):
//   out_i = x_i * cos(θ_{i mod half}) + rot_i * sin(θ_{i mod half})
//   rot_i = -x_{i+half}  (i <  half)
//         =  x_{i-half}  (i >= half)
// Matches transformers apply_rotary_pos_emb_vision (cos/sin of the
// concatenated-then-duplicated frequency table).

struct Params {
  n_rows: u32,     // number of patches
  width: u32,      // heads * head_dim (row stride)
  head_dim: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> phases: array<f32>;  // [n_rows, head_dim/2]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn vision_rope(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let total = params.n_rows * params.width;
  if (idx >= total) { return; }

  let row = idx / params.width;
  let within = idx % params.width;
  let d = within % params.head_dim;
  let half = params.head_dim / 2u;

  let theta = phases[row * half + (d % half)];
  let c = cos(theta);
  let s = sin(theta);

  let x = input[idx];
  var rot: f32;
  if (d < half) {
    rot = -input[idx + half];
  } else {
    rot = input[idx - half];
  }
  output[idx] = x * c + rot * s;
}
