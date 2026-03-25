// Group Normalization for Gated DeltaNet output
//
// Divides channels into groups, normalizes within each group:
//   y[i] = weight[i] * (x[i] - mean_g) / sqrt(var_g + eps)
// where mean_g and var_g are computed over the channels in group g.
//
// One workgroup per group. Workgroup size = 256.

struct Params {
  num_channels: u32,      // total channels (e.g., 4096)
  num_groups: u32,        // number of groups
  channels_per_group: u32,// = num_channels / num_groups
  eps: f32,               // normalization epsilon
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> shmem: array<f32, 256>;

@compute @workgroup_size(256)
fn group_norm(@builtin(local_invocation_id) lid: vec3u,
              @builtin(workgroup_id) wid: vec3u) {
  let group = wid.x;
  let tid = lid.x;
  let cpg = params.channels_per_group;
  let base = group * cpg;

  // Step 1: Compute mean via parallel reduction
  var partial_sum: f32 = 0.0;
  var i = tid;
  while (i < cpg) {
    partial_sum += input[base + i];
    i = i + 256u;
  }
  shmem[tid] = partial_sum;
  workgroupBarrier();

  // Reduction for sum
  if (tid < 128u) { shmem[tid] = shmem[tid] + shmem[tid + 128u]; } workgroupBarrier();
  if (tid < 64u)  { shmem[tid] = shmem[tid] + shmem[tid + 64u]; }  workgroupBarrier();
  if (tid < 32u)  { shmem[tid] = shmem[tid] + shmem[tid + 32u]; }  workgroupBarrier();
  if (tid < 16u)  { shmem[tid] = shmem[tid] + shmem[tid + 16u]; }  workgroupBarrier();
  if (tid < 8u)   { shmem[tid] = shmem[tid] + shmem[tid + 8u]; }   workgroupBarrier();
  if (tid < 4u)   { shmem[tid] = shmem[tid] + shmem[tid + 4u]; }   workgroupBarrier();
  if (tid < 2u)   { shmem[tid] = shmem[tid] + shmem[tid + 2u]; }   workgroupBarrier();
  if (tid < 1u)   { shmem[0] = shmem[0] + shmem[1]; }              workgroupBarrier();

  let mean = shmem[0] / f32(cpg);

  // Step 2: Compute variance via parallel reduction
  var partial_var: f32 = 0.0;
  i = tid;
  while (i < cpg) {
    let diff = input[base + i] - mean;
    partial_var += diff * diff;
    i = i + 256u;
  }
  shmem[tid] = partial_var;
  workgroupBarrier();

  if (tid < 128u) { shmem[tid] = shmem[tid] + shmem[tid + 128u]; } workgroupBarrier();
  if (tid < 64u)  { shmem[tid] = shmem[tid] + shmem[tid + 64u]; }  workgroupBarrier();
  if (tid < 32u)  { shmem[tid] = shmem[tid] + shmem[tid + 32u]; }  workgroupBarrier();
  if (tid < 16u)  { shmem[tid] = shmem[tid] + shmem[tid + 16u]; }  workgroupBarrier();
  if (tid < 8u)   { shmem[tid] = shmem[tid] + shmem[tid + 8u]; }   workgroupBarrier();
  if (tid < 4u)   { shmem[tid] = shmem[tid] + shmem[tid + 4u]; }   workgroupBarrier();
  if (tid < 2u)   { shmem[tid] = shmem[tid] + shmem[tid + 2u]; }   workgroupBarrier();
  if (tid < 1u)   { shmem[0] = shmem[0] + shmem[1]; }              workgroupBarrier();

  let variance = shmem[0] / f32(cpg);
  let inv_std = 1.0 / sqrt(variance + params.eps);

  // Step 3: Normalize and scale
  i = tid;
  while (i < cpg) {
    let idx = base + i;
    output[idx] = weight[idx] * (input[idx] - mean) * inv_std;
    i = i + 256u;
  }
}
