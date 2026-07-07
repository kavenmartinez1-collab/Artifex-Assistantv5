// GroupNorm over an NCHW f32 tensor (batch 1) with per-channel affine —
// the FLUX.2 VAE decoder norm (torch nn.GroupNorm(32, C, eps=1e-6)).
//
// NCHW is channel-major, so group g's channels [g*cpg, (g+1)*cpg) occupy one
// contiguous span of cpg*hw elements: one workgroup per group reduces the
// span, then normalizes with weight[c]/bias[c].
//
// Dispatch: (num_groups, 1, 1).

struct Params {
  num_channels: u32,
  num_groups: u32,
  hw: u32,   // H * W
  eps: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> shmem: array<f32, 256>;

@compute @workgroup_size(256)
fn group_norm_nchw(@builtin(local_invocation_id) lid: vec3u,
                   @builtin(workgroup_id) wid: vec3u) {
  let cpg = params.num_channels / params.num_groups;
  let span = cpg * params.hw;
  let base = wid.x * span;
  let base_c = wid.x * cpg;
  let t = lid.x;

  var partial = 0.0;
  for (var i = t; i < span; i += 256u) { partial += input[base + i]; }
  shmem[t] = partial;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (t < stride) { shmem[t] += shmem[t + stride]; }
    workgroupBarrier();
  }
  let mean = shmem[0] / f32(span);
  workgroupBarrier();

  partial = 0.0;
  for (var i = t; i < span; i += 256u) {
    let d = input[base + i] - mean;
    partial += d * d;
  }
  shmem[t] = partial;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (t < stride) { shmem[t] += shmem[t + stride]; }
    workgroupBarrier();
  }
  let inv_std = 1.0 / sqrt(shmem[0] / f32(span) + params.eps);

  for (var i = t; i < span; i += 256u) {
    let c = base_c + i / params.hw;
    output[base + i] = weight[c] * (input[base + i] - mean) * inv_std + bias[c];
  }
}
