// LayerNorm — standard layer normalization with weight and bias.
// ViT towers (Qwen3-VL, SigLIP) use LayerNorm, not RMSNorm:
//   output = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
// One workgroup per row, mirroring rmsnorm.wgsl's reduction structure.

struct Params {
  hidden_size: u32,
  eps: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> sh_sum: array<f32, 256>;
var<workgroup> sh_sumsq: array<f32, 256>;

@compute @workgroup_size(256)
fn layernorm(@builtin(local_invocation_id) lid: vec3u,
             @builtin(workgroup_id) wid: vec3u) {
  let row = wid.x;
  let col = lid.x;
  let hidden = params.hidden_size;

  // Single pass: accumulate sum and sum of squares
  var sum: f32 = 0.0;
  var sumsq: f32 = 0.0;
  var i = col;
  while (i < hidden) {
    let v = input[row * hidden + i];
    sum += v;
    sumsq += v * v;
    i += 256u;
  }
  sh_sum[col] = sum;
  sh_sumsq[col] = sumsq;
  workgroupBarrier();

  // Parallel reduction (both accumulators together)
  var stride = 128u;
  while (stride > 0u) {
    if (col < stride) {
      sh_sum[col] += sh_sum[col + stride];
      sh_sumsq[col] += sh_sumsq[col + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  let n = f32(hidden);
  let mean = sh_sum[0] / n;
  let variance = max(sh_sumsq[0] / n - mean * mean, 0.0);
  let inv_std = inverseSqrt(variance + params.eps);

  i = col;
  while (i < hidden) {
    let v = input[row * hidden + i];
    output[row * hidden + i] = (v - mean) * inv_std * weight[i] + bias[i];
    i += 256u;
  }
}
