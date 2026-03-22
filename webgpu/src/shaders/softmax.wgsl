// Softmax — row-wise softmax for attention scores
// softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//
// Three-pass: find max, compute exp and sum, normalize
// Processes one row per workgroup

struct Params {
  cols: u32,  // number of elements per row
  rows: u32,  // number of rows
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shmem: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax(@builtin(local_invocation_id) lid: vec3u,
           @builtin(workgroup_id) wid: vec3u) {
  let row = wid.x;
  let col = lid.x;
  let cols = params.cols;

  if (row >= params.rows) { return; }

  let row_offset = row * cols;

  // Pass 1: Find row max for numerical stability
  var local_max: f32 = -1e30;
  var i = col;
  while (i < cols) {
    local_max = max(local_max, input[row_offset + i]);
    i += 256u;
  }
  // Unused threads (col >= cols) keep -1e30 which won't affect max
  shmem[col] = local_max;
  workgroupBarrier();

  // Parallel reduction for max
  if (col < 128u) { shmem[col] = max(shmem[col], shmem[col + 128u]); }
  workgroupBarrier();
  if (col < 64u) { shmem[col] = max(shmem[col], shmem[col + 64u]); }
  workgroupBarrier();
  if (col < 32u) { shmem[col] = max(shmem[col], shmem[col + 32u]); }
  workgroupBarrier();
  if (col < 16u) { shmem[col] = max(shmem[col], shmem[col + 16u]); }
  workgroupBarrier();
  if (col < 8u) { shmem[col] = max(shmem[col], shmem[col + 8u]); }
  workgroupBarrier();
  if (col < 4u) { shmem[col] = max(shmem[col], shmem[col + 4u]); }
  workgroupBarrier();
  if (col < 2u) { shmem[col] = max(shmem[col], shmem[col + 2u]); }
  workgroupBarrier();
  if (col < 1u) { shmem[0] = max(shmem[0], shmem[1]); }
  workgroupBarrier();

  let row_max = shmem[0];

  // Pass 2: Compute exp(x - max) and partial sums
  var local_sum: f32 = 0.0;
  i = col;
  while (i < cols) {
    let val = exp(input[row_offset + i] - row_max);
    output[row_offset + i] = val;
    local_sum += val;
    i += 256u;
  }
  // Unused threads contribute 0 to sum
  shmem[col] = local_sum;
  workgroupBarrier();

  // Parallel reduction for sum
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

  let inv_sum = 1.0 / shmem[0];

  // Pass 3: Normalize — read from output, write back
  i = col;
  while (i < cols) {
    let idx = row_offset + i;
    output[idx] = output[idx] * inv_sum;
    i += 256u;
  }
}
