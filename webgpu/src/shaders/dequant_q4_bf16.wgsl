// GPTQ INT4 → BF16 Dequantization Compute Shader
//
// Converts a GPTQ INT4 packed weight matrix to BF16 format in-place on GPU.
// Output layout matches matmul_bt_bf16: [N, K/2] u32, each u32 holds two BF16
// values (even k in low 16 bits, odd k in high 16 bits).
//
// Dispatch: one thread per output u32 = one thread per (n, k_pair).
// Total threads = N * (K / 2).

struct Params {
  N: u32,           // out_features (cols of weight)
  K: u32,           // in_features (rows of weight)
  group_size: u32,  // quantization group size (typically 128)
}

@group(0) @binding(0) var<storage, read> B_packed: array<i32>;
@group(0) @binding(1) var<storage, read> scales_raw: array<u32>;
@group(0) @binding(2) var<storage, read> qzeros: array<i32>;
@group(0) @binding(3) var<storage, read> g_idx: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_bf16: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

fn extract_q4(packed: i32, nibble_idx: u32) -> u32 {
  return u32((packed >> (nibble_idx * 4u)) & 0xF);
}

fn read_f16_scale(idx: u32) -> f32 {
  let word = scales_raw[idx / 2u];
  let half_bits = select(word & 0xFFFFu, word >> 16u, (idx & 1u) == 1u);
  let sign = (half_bits >> 15u) & 1u;
  let exp = (half_bits >> 10u) & 0x1Fu;
  let frac = half_bits & 0x3FFu;
  if (exp == 0u) {
    if (frac == 0u) { return select(0.0, -0.0, sign == 1u); }
    let f = f32(frac) * bitcast<f32>(0x33800000u);
    return select(f, -f, sign == 1u);
  }
  if (exp == 31u) {
    return select(1e30, -1e30, sign == 1u);
  }
  let f32_bits = (sign << 31u) | ((exp + 112u) << 23u) | (frac << 13u);
  return bitcast<f32>(f32_bits);
}

fn f32_to_bf16(v: f32) -> u32 {
  let bits = bitcast<u32>(v);
  // Round-to-nearest-even: add 0x7FFF + bit 16 (the lsb of bf16 mantissa)
  let rounded = bits + 0x7FFFu + ((bits >> 16u) & 1u);
  return rounded >> 16u;
}

@compute @workgroup_size(256)
fn dequant_q4_to_bf16(@builtin(global_invocation_id) gid: vec3u) {
  let thread_id = gid.x;
  let K_half = params.K / 2u;
  let total = params.N * K_half;
  if (thread_id >= total) { return; }

  let n = thread_id / K_half;
  let k_pair = thread_id % K_half;
  let k_even = k_pair * 2u;
  let k_odd = k_even + 1u;

  let N = params.N;
  let gs = params.group_size;

  // Dequant even k
  let packed_even = B_packed[(k_even / 8u) * N + n];
  let q4_even = extract_q4(packed_even, k_even % 8u);
  let group_even = g_idx[k_even];
  let scale_even = read_f16_scale(group_even * N + n);
  let zero_even = extract_q4(qzeros[group_even * ((N + 7u) / 8u) + n / 8u], n % 8u);
  let val_even = (f32(q4_even) - f32(zero_even)) * scale_even;

  // Dequant odd k
  let packed_odd = B_packed[(k_odd / 8u) * N + n];
  let q4_odd = extract_q4(packed_odd, k_odd % 8u);
  let group_odd = g_idx[k_odd];
  let scale_odd = read_f16_scale(group_odd * N + n);
  let zero_odd = extract_q4(qzeros[group_odd * ((N + 7u) / 8u) + n / 8u], n % 8u);
  let val_odd = (f32(q4_odd) - f32(zero_odd)) * scale_odd;

  // Pack two BF16 values into one u32: even in low 16, odd in high 16
  let bf16_lo = f32_to_bf16(val_even);
  let bf16_hi = f32_to_bf16(val_odd);
  out_bf16[n * K_half + k_pair] = bf16_lo | (bf16_hi << 16u);
}
