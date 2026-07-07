// Q8_0 -> bf16-packed-u32 dequant for the FLUX.2 DiT all-resident path (P7.2).
//
// Source layout (one storage buffer per tensor, produced by loadFlux2DitQ8):
//   [ i8 quants: N*K bytes ][ pad to 64 ][ f16 scales: N*K/32 halfwords ]
// Block b covers 32 consecutive weights along K (row-major), value = d * q.
//
// Output is the [N, K/2] u32 raw-bf16-pair layout matmul_bt_bf16* expects
// (even k in the low 16 bits). f32 -> bf16 uses round-to-nearest-even so the
// CPU reference in the parity gate can match bit-exactly: the product
// f32(d_f16) * q is exact in f32 (11-bit x 7-bit mantissa), RNE is
// deterministic.
//
// One thread per block: reads 8 quant words + 1 scale halfword, writes 16
// output words. Pure bandwidth; ~3 bytes moved per weight.

struct Params {
  num_blocks: u32,   // N*K/32
  s_word_off: u32,   // scale section offset in u32 words within src
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn i8_at(w: u32, i: u32) -> f32 {
  let b = (w >> (8u * i)) & 0xffu;
  return f32((i32(b) << 24u) >> 24u);
}

fn bf16_rne(x: f32) -> u32 {
  let u = bitcast<u32>(x);
  return (u + 0x7fffu + ((u >> 16u) & 1u)) >> 16u;
}

@compute @workgroup_size(256)
fn dequant_q8_bf16(@builtin(global_invocation_id) gid: vec3<u32>) {
  let blk = gid.x;
  if (blk >= params.num_blocks) { return; }
  let sw = src[params.s_word_off + blk / 2u];
  let d = unpack2x16float(sw)[blk & 1u];
  let q0 = blk * 8u;   // quant words for this block
  let o0 = blk * 16u;  // output words
  for (var w = 0u; w < 8u; w = w + 1u) {
    let qw = src[q0 + w];
    let a = bf16_rne(d * i8_at(qw, 0u));
    let b = bf16_rne(d * i8_at(qw, 1u));
    let c = bf16_rne(d * i8_at(qw, 2u));
    let e = bf16_rne(d * i8_at(qw, 3u));
    dst[o0 + w * 2u] = a | (b << 16u);
    dst[o0 + w * 2u + 1u] = c | (e << 16u);
  }
}
