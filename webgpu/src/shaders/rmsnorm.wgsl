// RMSNorm — Root Mean Square Layer Normalization
// Qwen3.5 uses RMSNorm (not LayerNorm): output = (x / rms(x)) * weight
// where rms(x) = sqrt(mean(x^2) + eps)

struct Params {
  hidden_size: u32,
  eps: f32,
  use_residual_weight: u32, // 1 = use (1+weight), 0 = use weight directly
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> shmem: array<f32, 256>;

@compute @workgroup_size(256)
fn rmsnorm(@builtin(global_invocation_id) gid: vec3u,
           @builtin(local_invocation_id) lid: vec3u,
           @builtin(workgroup_id) wid: vec3u) {
  let row = wid.x;
  let col = lid.x;
  let hidden = params.hidden_size;

  // Each thread computes partial sum of squares
  var sum_sq: f32 = 0.0;
  var i = col;
  while (i < hidden) {
    let val = input[row * hidden + i];
    sum_sq += val * val;
    i += 256u;
  }

  shmem[col] = sum_sq;
  workgroupBarrier();

  // Unrolled parallel reduction for sum
  if (col < 128u) { shmem[col] = shmem[col] + shmem[col + 128u]; }
  workgroupBarrier();
  if (col < 64u) { shmem[col] = shmem[col] + shmem[col + 64u]; }
  workgroupBarrier();
  if (col < 32u) { shmem[col] = shmem[col] + shmem[col + 32u]; }
  workgroupBarrier();
  if (col < 16u) { shmem[col] = shmem[col] + shmem[col + 16u]; }
  workgroupBarrier();
  if (col < 8u) { shmem[col] = shmem[col] + shmem[col + 8u]; }
  workgroupBarrier();
  if (col < 4u) { shmem[col] = shmem[col] + shmem[col + 4u]; }
  workgroupBarrier();
  if (col < 2u) { shmem[col] = shmem[col] + shmem[col + 2u]; }
  workgroupBarrier();
  if (col < 1u) { shmem[0] = shmem[0] + shmem[1]; }
  workgroupBarrier();

  // Compute 1/rms from the total sum of squares
  let rms_inv = 1.0 / sqrt(shmem[0] / f32(hidden) + params.eps);

  // Normalize and scale
  i = col;
  while (i < hidden) {
    let idx = row * hidden + i;
    // Qwen3_5 uses (1.0 + weight) — weight initialized to 0, effective scale ~1.0
    // Standard models use weight directly — weight initialized to 1.0
    let w = select(weight[i], 1.0 + weight[i], params.use_residual_weight == 1u);
    output[idx] = input[idx] * rms_inv * w;
    i += 256u;
  }
}
