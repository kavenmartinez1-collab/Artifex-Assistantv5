/**
 * q5k_gemv.c — freestanding wasm32 SIMD Q5_K GEMV kernel.
 *
 * Phase 0 benchmark kernel + seed of the Phase C expert-FFN kernel.
 * Implements llama.cpp-compatible Q5_K dequant-dot against Q8-quantized
 * activations (integer path, i16x8 widening dot products).
 *
 * Build: see build-wasm.sh (clang --target=wasm32, no libc, no emscripten).
 *
 * Q5_K superblock layout (ggml, 256 elements, 176 bytes):
 *   d (f16), dmin (f16), scales[12] (8x 6-bit scale/min pairs),
 *   qh[32] (5th bits), qs[128] (low nibbles).
 * Dequant: x[i] = d*sc*q5[i] - dmin*m   with q5 in 0..31.
 */

#include <stdint.h>
#include <wasm_simd128.h>

#define QK_K 256

typedef struct {
  uint16_t d;        // ggml fp16 super-scale
  uint16_t dmin;     // ggml fp16 super-min
  uint8_t scales[12];
  uint8_t qh[32];
  uint8_t qs[128];
} block_q5_K;

_Static_assert(sizeof(block_q5_K) == 176, "block_q5_K must be 176 bytes");

/** Q8 activation block: 256 elems, f32 scale, per-32 subgroup sums (for mins). */
typedef struct {
  float d;
  int8_t q[QK_K];
  int16_t bsums[QK_K / 32];
} block_q8_act;

_Static_assert(sizeof(block_q8_act) == 276, "block_q8_act must be 276 bytes");

static inline float f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t man = h & 0x3FF;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      // subnormal: normalize
      uint32_t e = 127 - 15 + 1;
      while (!(man & 0x400u)) { man <<= 1; e--; }
      man &= 0x3FFu;
      bits = sign | (e << 23) | (man << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (man << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  union { uint32_t u; float f; } u;
  u.u = bits;
  return u.f;
}

/** ggml get_scale_min_k4: unpack 6-bit (scale, min) pair j from 12 packed bytes. */
static inline void get_scale_min(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

/** Horizontal sum of 4 i32 lanes. */
static inline int32_t hsum_i32x4(v128_t v) {
  return wasm_i32x4_extract_lane(v, 0) + wasm_i32x4_extract_lane(v, 1) +
         wasm_i32x4_extract_lane(v, 2) + wasm_i32x4_extract_lane(v, 3);
}

/**
 * Dot of 32 unsigned 5-bit weights (two v128 of u8 in 0..31) with 32 signed
 * q8 activations. Products fit i16 (31*127), pairs accumulate to i32.
 */
static inline int32_t dot32(v128_t w0, v128_t w1, const int8_t *q8) {
  v128_t a0 = wasm_v128_load(q8);
  v128_t a1 = wasm_v128_load(q8 + 16);
  v128_t acc = wasm_i32x4_dot_i16x8(wasm_u16x8_extend_low_u8x16(w0),
                                    wasm_i16x8_extend_low_i8x16(a0));
  acc = wasm_i32x4_add(acc, wasm_i32x4_dot_i16x8(wasm_u16x8_extend_high_u8x16(w0),
                                                 wasm_i16x8_extend_high_i8x16(a0)));
  acc = wasm_i32x4_add(acc, wasm_i32x4_dot_i16x8(wasm_u16x8_extend_low_u8x16(w1),
                                                 wasm_i16x8_extend_low_i8x16(a1)));
  acc = wasm_i32x4_add(acc, wasm_i32x4_dot_i16x8(wasm_u16x8_extend_high_u8x16(w1),
                                                 wasm_i16x8_extend_high_i8x16(a1)));
  return hsum_i32x4(acc);
}

/** Dot of one Q5_K row (nb superblocks) with Q8 activations. */
static float q5k_dot_row(const block_q5_K *w, const block_q8_act *x, int nb) {
  float sumf = 0.0f;
  const v128_t mask4 = wasm_i8x16_splat(0x0F);
  const v128_t one = wasm_i8x16_splat(1);

  for (int i = 0; i < nb; i++) {
    const block_q5_K *b = &w[i];
    const block_q8_act *a = &x[i];
    const float d = f16_to_f32(b->d) * a->d;
    const float dmin = f16_to_f32(b->dmin) * a->d;

    uint8_t sc[8], mn[8];
    int32_t smin = 0;
    for (int j = 0; j < 8; j++) {
      get_scale_min(j, b->scales, &sc[j], &mn[j]);
      smin += (int32_t)mn[j] * (int32_t)a->bsums[j];
    }

    const uint8_t *ql = b->qs;
    const int8_t *q8 = a->q;
    v128_t qh0 = wasm_v128_load(b->qh);
    v128_t qh1 = wasm_v128_load(b->qh + 16);

    int32_t sumi = 0;
    for (int j = 0; j < 4; j++) { // 64 elements per iteration
      v128_t l0 = wasm_v128_load(ql);
      v128_t l1 = wasm_v128_load(ql + 16);

      // low nibbles get qh bit (2j), high nibbles get qh bit (2j+1), as +16
      v128_t hlo0 = wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(qh0, 2 * j), one), 4);
      v128_t hlo1 = wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(qh1, 2 * j), one), 4);
      v128_t hhi0 = wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(qh0, 2 * j + 1), one), 4);
      v128_t hhi1 = wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(qh1, 2 * j + 1), one), 4);

      v128_t wlo0 = wasm_v128_or(wasm_v128_and(l0, mask4), hlo0);
      v128_t wlo1 = wasm_v128_or(wasm_v128_and(l1, mask4), hlo1);
      v128_t whi0 = wasm_v128_or(wasm_u8x16_shr(l0, 4), hhi0);
      v128_t whi1 = wasm_v128_or(wasm_u8x16_shr(l1, 4), hhi1);

      sumi += (int32_t)sc[2 * j] * dot32(wlo0, wlo1, q8);
      sumi += (int32_t)sc[2 * j + 1] * dot32(whi0, whi1, q8 + 32);

      ql += 32;
      q8 += 64;
    }

    sumf += d * (float)sumi - dmin * (float)smin;
  }
  return sumf;
}

/**
 * Quantize f32 activations to Q8 blocks (n must be a multiple of 256).
 * llama.cpp quantize_row_q8_K equivalent (round-to-nearest, per-block scale).
 */
__attribute__((export_name("q8_quantize")))
void q8_quantize(const float *x, block_q8_act *y, int n) {
  for (int i = 0; i < n / QK_K; i++) {
    float amax = 0.0f;
    for (int j = 0; j < QK_K; j++) {
      float v = x[j] < 0 ? -x[j] : x[j];
      if (v > amax) amax = v;
    }
    float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    y[i].d = d;
    for (int g = 0; g < 8; g++) {
      int sum = 0;
      for (int j = 0; j < 32; j++) {
        float v = x[g * 32 + j] * id;
        int q = (int)(v + (v >= 0 ? 0.5f : -0.5f));
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        y[i].q[g * 32 + j] = (int8_t)q;
        sum += q;
      }
      y[i].bsums[g] = (int16_t)sum;
    }
    x += QK_K;
  }
}

/**
 * GEMV: y[rows] = W[rows x cols] @ x, W row-major Q5_K, x pre-quantized Q8.
 * cols must be a multiple of 256.
 */
__attribute__((export_name("q5k_gemv")))
void q5k_gemv(const block_q5_K *w, const block_q8_act *x, float *y, int rows, int cols) {
  int nb = cols / QK_K;
  for (int r = 0; r < rows; r++) {
    y[r] = q5k_dot_row(w + (uint32_t)r * (uint32_t)nb, x, nb);
  }
}

/*
 * Q6_K superblock layout (ggml, 256 elements, 210 bytes):
 *   ql[128] (low 4 bits), qh[64] (upper 2 bits), scales[16] (int8, per-16),
 *   d (f16). Dequant: x[i] = d * sc[i/16] * (q6[i] - 32), q6 in 0..63.
 *
 * Only a few ffn_down_exps tensors are Q6_K (k-quant bump layers), so this
 * is a straightforward scalar port of llama.cpp ggml_vec_dot_q6_K_q8_K —
 * <5% of expert bytes per token, clang -O3 auto-vectorizes the inner loop.
 */
typedef struct {
  uint8_t ql[128];
  uint8_t qh[64];
  int8_t scales[16];
  uint16_t d; // ggml fp16 super-scale
} block_q6_K;

_Static_assert(sizeof(block_q6_K) == 210, "block_q6_K must be 210 bytes");

static float q6k_dot_row(const block_q6_K *w, const block_q8_act *x, int nb) {
  float sumf = 0.0f;
  for (int i = 0; i < nb; i++) {
    const block_q6_K *b = (const block_q6_K *)((const uint8_t *)w + (uint32_t)i * 210u);
    const block_q8_act *a = &x[i];
    const float d = f16_to_f32(b->d) * a->d;

    const uint8_t *ql = b->ql;
    const uint8_t *qh = b->qh;
    const int8_t *sc = b->scales;
    const int8_t *q8 = a->q;

    int32_t isum = 0;
    for (int n = 0; n < 2; n++) { // two 128-element halves
      for (int l = 0; l < 32; l++) {
        int32_t q1 = (int32_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        int32_t q2 = (int32_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        int32_t q3 = (int32_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        int32_t q4 = (int32_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        isum += (int32_t)sc[l / 16 + 0] * q1 * (int32_t)q8[l]
              + (int32_t)sc[l / 16 + 2] * q2 * (int32_t)q8[l + 32]
              + (int32_t)sc[l / 16 + 4] * q3 * (int32_t)q8[l + 64]
              + (int32_t)sc[l / 16 + 6] * q4 * (int32_t)q8[l + 96];
      }
      ql += 64;
      qh += 32;
      sc += 8;
      q8 += 128;
    }
    sumf += d * (float)isum;
  }
  return sumf;
}

/**
 * GEMV: y[rows] = W[rows x cols] @ x, W row-major Q6_K, x pre-quantized Q8.
 * cols must be a multiple of 256. Note: block_q6_K is 210 bytes (unaligned —
 * row pointers are computed in bytes).
 */
__attribute__((export_name("q6k_gemv")))
void q6k_gemv(const block_q6_K *w, const block_q8_act *x, float *y, int rows, int cols) {
  int nb = cols / QK_K;
  uint32_t rowBytes = (uint32_t)nb * 210u;
  for (int r = 0; r < rows; r++) {
    y[r] = q6k_dot_row((const block_q6_K *)((const uint8_t *)w + (uint32_t)r * rowBytes), x, nb);
  }
}
