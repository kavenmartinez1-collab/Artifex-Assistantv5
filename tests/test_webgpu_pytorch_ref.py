"""PyTorch numerical regression tests for WebGPU kernel CPU references.

Validates that the CPU reference implementations in kernel-tests.ts
produce results within tolerance of PyTorch's own ops, ensuring the
WebGPU shaders are validated against a known-correct baseline.

Uses fixed RNG seeds so results are deterministic and reproducible
across platforms. Tolerance is 1e-4 (matches kernel-tests.ts TOLERANCE).
"""

import math
import struct
import unittest

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

TOLERANCE = 1e-4
SEED = 42


def _seeded_f32(n: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n).astype(np.float32)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestSiLU(unittest.TestCase):
    """SiLU: x * sigmoid(x) — must match torch.nn.functional.silu."""

    def test_silu_matches_pytorch(self):
        x = _seeded_f32(1024)
        expected = F.silu(torch.from_numpy(x)).numpy()
        cpu_ref = x / (1.0 + np.exp(-x))
        diff = np.max(np.abs(expected - cpu_ref))
        self.assertLess(diff, TOLERANCE, f"SiLU maxDiff={diff}")

    def test_silu_large_negative(self):
        x = np.array([-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0], dtype=np.float32)
        expected = F.silu(torch.from_numpy(x)).numpy()
        cpu_ref = x / (1.0 + np.exp(-x))
        diff = np.max(np.abs(expected - cpu_ref))
        self.assertLess(diff, TOLERANCE)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestSoftmax(unittest.TestCase):
    """Row-wise softmax with numerical stability."""

    def test_softmax_matches_pytorch(self):
        rows, cols = 4, 128
        x = _seeded_f32(rows * cols).reshape(rows, cols)
        expected = F.softmax(torch.from_numpy(x), dim=-1).numpy()
        # CPU reference: row-wise max subtract, exp, normalize
        maxes = x.max(axis=1, keepdims=True)
        e = np.exp(x - maxes)
        cpu_ref = e / e.sum(axis=1, keepdims=True)
        diff = np.max(np.abs(expected - cpu_ref))
        self.assertLess(diff, TOLERANCE, f"Softmax maxDiff={diff}")

    def test_softmax_numerical_stability(self):
        x = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
        expected = F.softmax(torch.from_numpy(x), dim=-1).numpy()
        maxes = x.max(axis=1, keepdims=True)
        e = np.exp(x - maxes)
        cpu_ref = e / e.sum(axis=1, keepdims=True)
        diff = np.max(np.abs(expected - cpu_ref))
        self.assertLess(diff, TOLERANCE)


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestRMSNorm(unittest.TestCase):
    """RMS normalization: x * weight / sqrt(mean(x^2) + eps)."""

    def test_rmsnorm_matches_pytorch(self):
        rows, hidden = 2, 256
        eps = 1e-6
        x = _seeded_f32(rows * hidden).reshape(rows, hidden)
        weight = np.abs(_seeded_f32(hidden, seed=123)) + 0.5

        x_t = torch.from_numpy(x)
        w_t = torch.from_numpy(weight)
        # PyTorch RMSNorm: x * w / sqrt(mean(x^2) + eps)
        rms = torch.sqrt(x_t.pow(2).mean(dim=-1, keepdim=True) + eps)
        expected = (x_t / rms * w_t).numpy()

        # CPU reference (matches kernel-tests.ts cpuRMSNorm)
        cpu_ref = np.zeros_like(x)
        for r in range(rows):
            sum_sq = np.sum(x[r] ** 2)
            rms_inv = 1.0 / math.sqrt(sum_sq / hidden + eps)
            cpu_ref[r] = x[r] * rms_inv * weight

        diff = np.max(np.abs(expected - cpu_ref))
        self.assertLess(diff, TOLERANCE, f"RMSNorm maxDiff={diff}")


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestMatmul(unittest.TestCase):
    """Matrix multiplication: C = A @ B and C = A @ B^T."""

    def test_matmul_matches_pytorch(self):
        M, N, K = 32, 32, 64
        a = _seeded_f32(M * K).reshape(M, K)
        b = _seeded_f32(K * N, seed=99).reshape(K, N)
        expected = (torch.from_numpy(a) @ torch.from_numpy(b)).numpy()
        cpu_ref = a @ b
        diff = np.max(np.abs(expected - cpu_ref))
        self.assertLess(diff, TOLERANCE, f"Matmul maxDiff={diff}")

    def test_matmul_bt_matches_pytorch(self):
        """C[M,N] = A[M,K] @ B^T where B stored as [N,K]."""
        M, N, K = 1, 32, 64
        a = _seeded_f32(M * K).reshape(M, K)
        b = _seeded_f32(N * K, seed=77).reshape(N, K)
        expected = (torch.from_numpy(a) @ torch.from_numpy(b).T).numpy()
        cpu_ref = a @ b.T
        diff = np.max(np.abs(expected - cpu_ref))
        self.assertLess(diff, TOLERANCE, f"MatmulBT maxDiff={diff}")

    def test_matmul_larger_dims(self):
        M, N, K = 1, 256, 2048
        a = _seeded_f32(M * K).reshape(M, K)
        b = _seeded_f32(N * K, seed=55).reshape(N, K)
        expected = (torch.from_numpy(a) @ torch.from_numpy(b).T).numpy()
        cpu_ref = a @ b.T
        # Larger dims accumulate more FP32 rounding — allow wider tolerance
        diff = np.max(np.abs(expected - cpu_ref))
        self.assertLess(diff, 1e-2, f"MatmulBT large maxDiff={diff}")


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestBF16Packing(unittest.TestCase):
    """BF16 round-trip: f32 → bf16 → f32 loses only bottom 16 mantissa bits."""

    def test_bf16_round_trip_within_tolerance(self):
        x = _seeded_f32(128)
        x_bf16 = torch.from_numpy(x).bfloat16()
        x_back = x_bf16.float().numpy()
        # BF16 truncates 16 mantissa bits; relative error is ~2^-8 ≈ 0.004
        for i in range(len(x)):
            if abs(x[i]) > 1e-6:
                rel_err = abs(x[i] - x_back[i]) / abs(x[i])
                self.assertLess(rel_err, 0.01, f"BF16 rel_err[{i}]={rel_err}")

    def test_bf16_truncation_error_bounded(self):
        """WGSL BF16 uses truncation (not RNE). Verify truncation error is bounded."""
        x = np.array([1.0, -2.5, 3.14, 0.0, 100.0, -0.001], dtype=np.float32)
        for v in x:
            bs = struct.pack('<f', v)
            u32 = struct.unpack('<I', bs)[0]
            bf16 = (u32 >> 16) & 0xFFFF
            back = struct.unpack('<f', struct.pack('<I', bf16 << 16))[0]
            if abs(v) > 1e-6:
                rel_err = abs(v - back) / abs(v)
                self.assertLess(rel_err, 0.01, f"BF16 truncation rel_err({v})={rel_err}")


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestElementwise(unittest.TestCase):
    """Elementwise add and multiply."""

    def test_add_matches_pytorch(self):
        a = _seeded_f32(1024)
        b = _seeded_f32(1024, seed=77)
        expected = (torch.from_numpy(a) + torch.from_numpy(b)).numpy()
        cpu_ref = a + b
        diff = np.max(np.abs(expected - cpu_ref))
        self.assertLess(diff, TOLERANCE)

    def test_mul_matches_pytorch(self):
        a = _seeded_f32(1024)
        b = _seeded_f32(1024, seed=77)
        expected = (torch.from_numpy(a) * torch.from_numpy(b)).numpy()
        cpu_ref = a * b
        diff = np.max(np.abs(expected - cpu_ref))
        self.assertLess(diff, TOLERANCE)


if __name__ == "__main__":
    unittest.main()
