// LFM2/LFM3 gated short convolution — multi-token (prefill + decode).
//
// llama.cpp build_lfm_shortconv (src/models/lfm-base.cpp):
//   bcx = in_proj · x               [tokens, 3*dim], chunks B | C | X
//   bx  = B ⊙ X
//   y   = C ⊙ causal_depthwise_conv(bx)      (kernel K, no bias, no activation)
//   out = out_proj · y               (done by the caller)
// The conv is ggml_ssm_conv over [state | bx]: for channel ch and token t
//   conv[t] = Σ_{k=0..K-1} w[ch][k] · pad[t+k],  pad = state (K-1 oldest-first) ++ bx
// so w[ch][K-1] multiplies the CURRENT token. Accumulated in f32 in the same
// order as ggml-cpu (k ascending, starting from 0).
//
// One thread per channel walks the tokens in order with the K-1 window in
// registers, so a whole prefill chunk is one dispatch. The state persists
// across calls (zero-initialised = new sequence); the window after the last
// token is written back.
//
// Layouts: bcx row-major [seq_len, 3*dim]; weight [dim, K] (GGUF
// shortconv.conv.weight ne=[K, dim]); state [K-1, dim]; out [seq_len, dim].

struct Params {
  seq_len: u32,
  dim: u32,
  kernel: u32,   // K (GGUF shortconv.l_cache), 2..MAX_K
  _pad: u32,
}

const MAX_K: u32 = 16u;

@group(0) @binding(0) var<storage, read> bcx: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> state: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn short_conv(@builtin(global_invocation_id) gid: vec3u) {
  let ch = gid.x;
  let dim = params.dim;
  if (ch >= dim) { return; }
  let K = params.kernel;
  let hist = K - 1u;

  var win: array<f32, MAX_K>;
  var w: array<f32, MAX_K>;
  for (var k = 0u; k < K; k = k + 1u) {
    w[k] = weight[ch * K + k];
  }
  for (var k = 0u; k < hist; k = k + 1u) {
    win[k] = state[k * dim + ch];
  }

  for (var t = 0u; t < params.seq_len; t = t + 1u) {
    let row = t * 3u * dim;
    let b = bcx[row + ch];
    let c = bcx[row + dim + ch];
    let x = bcx[row + 2u * dim + ch];
    let bx = b * x;

    var acc: f32 = 0.0;
    for (var k = 0u; k < hist; k = k + 1u) {
      acc = acc + w[k] * win[k];
    }
    acc = acc + w[hist] * bx;
    out[t * dim + ch] = c * acc;

    for (var k = 0u; k + 1u < hist; k = k + 1u) {
      win[k] = win[k + 1u];
    }
    win[hist - 1u] = bx;
  }

  for (var k = 0u; k < hist; k = k + 1u) {
    state[k * dim + ch] = win[k];
  }
}
