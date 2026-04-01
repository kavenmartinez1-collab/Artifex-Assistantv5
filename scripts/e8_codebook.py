#!/usr/bin/env python3
"""
E8 Lattice Codebook Generator for 2-bit Vector Quantization
=============================================================

Generates a 256-entry codebook of 8-dimensional vectors from the E8 lattice
for use in 2-bit weight quantization (QuIP#-style).

The E8 lattice is the densest sphere packing in 8 dimensions. We construct
the 240 root vectors (shortest non-zero lattice points) and then select 256
entries optimized for Gaussian-distributed data (post-KLT rotation makes
weights approximately Gaussian).

Two modes:
  1. Lattice mode (default): 240 E8 roots + 16 scaled variants, unit-normalized
  2. Trained mode: k-means on calibration weight data for data-adaptive codebook

Output: codebook_e8.bin — 256 * 8 * float32 = 8192 bytes

Usage:
  python scripts/e8_codebook.py                          # lattice mode
  python scripts/e8_codebook.py --output codebook_e8.bin # custom output path
"""

import argparse
import itertools
from pathlib import Path

import numpy as np


def generate_e8_roots() -> np.ndarray:
    """Generate all 240 root vectors of the E8 lattice.

    The E8 root system consists of:
      Type A: all permutations of (+-1, +-1, 0, 0, 0, 0, 0, 0) — C(8,2)*4 = 112
      Type B: (+-1/2)^8 with an even number of minus signs — 2^7 = 128

    Total: 240 vectors.
    """
    roots = []

    # Type A: 2 non-zero coordinates that are +-1, rest zero
    for i, j in itertools.combinations(range(8), 2):
        for si in (-1.0, 1.0):
            for sj in (-1.0, 1.0):
                v = np.zeros(8, dtype=np.float32)
                v[i] = si
                v[j] = sj
                roots.append(v)

    # Type B: all coordinates +-0.5, even number of negatives
    base = np.full(8, 0.5, dtype=np.float32)
    for neg_count in range(0, 9, 2):  # 0, 2, 4, 6, 8 negatives
        for neg_positions in itertools.combinations(range(8), neg_count):
            v = base.copy()
            for p in neg_positions:
                v[p] = -0.5
            roots.append(v)

    roots = np.array(roots, dtype=np.float32)
    assert roots.shape[0] == 240, f"Expected 240 roots, got {roots.shape[0]}"
    return roots


def generate_e8_codebook(seed: int = 42) -> np.ndarray:
    """Generate 256-entry E8 lattice codebook for 2-bit vector quantization.

    Starts with the 240 E8 root vectors (unit-normalized), then adds 16
    additional scaled vectors selected for good coverage of Gaussian data.

    Returns: codebook [256, 8] float32 array, each row unit-norm.
    """
    rng = np.random.RandomState(seed)
    roots = generate_e8_roots()

    # Normalize all roots to unit norm
    norms = np.linalg.norm(roots, axis=1, keepdims=True)
    roots_normed = roots / norms

    # Remove exact duplicates (some roots may normalize to the same vector)
    # Use rounding to detect near-duplicates
    seen = set()
    unique = []
    for v in roots_normed:
        key = tuple(np.round(v, 6))
        if key not in seen:
            seen.add(key)
            unique.append(v)
    roots_normed = np.array(unique, dtype=np.float32)

    # We should have exactly 240 unique unit-norm roots
    assert roots_normed.shape[0] == 240, (
        f"Expected 240 unique roots, got {roots_normed.shape[0]}")

    # Add 16 additional vectors: scaled E8 lattice points optimized for
    # Gaussian coverage. Use diagonal-dominant directions (common in
    # post-KLT weight distributions) at various norms.
    extra = []

    # 8 axis-aligned unit vectors (capture single-channel outliers)
    for i in range(8):
        v = np.zeros(8, dtype=np.float32)
        v[i] = 1.0
        extra.append(v)

    # 8 vectors along (1,1,...,1) direction with varying signs
    # These capture correlated multi-channel patterns
    for idx in range(8):
        v = np.ones(8, dtype=np.float32)
        # Flip sign of one coordinate to get 8 distinct vectors
        v[idx] = -1.0
        v /= np.linalg.norm(v)
        extra.append(v)

    extra = np.array(extra, dtype=np.float32)

    # Remove any extras that duplicate existing roots
    final_extra = []
    for v in extra:
        key = tuple(np.round(v, 6))
        if key not in seen:
            seen.add(key)
            final_extra.append(v)

    # Fill remaining slots (if we have fewer than 16 unique extras) with
    # random unit-norm Gaussian vectors for maximum coverage
    while len(final_extra) < 16:
        v = rng.randn(8).astype(np.float32)
        v /= np.linalg.norm(v)
        key = tuple(np.round(v, 6))
        if key not in seen:
            seen.add(key)
            final_extra.append(v)

    final_extra = np.array(final_extra[:16], dtype=np.float32)
    codebook = np.concatenate([roots_normed, final_extra], axis=0)
    assert codebook.shape == (256, 8), f"Expected (256, 8), got {codebook.shape}"

    # Verify unit norms
    cb_norms = np.linalg.norm(codebook, axis=1)
    assert np.allclose(cb_norms, 1.0, atol=1e-5), (
        f"Codebook norms not unit: min={cb_norms.min()}, max={cb_norms.max()}")

    return codebook


def generate_trained_codebook(weight_data: np.ndarray, seed: int = 42,
                              max_iter: int = 100) -> np.ndarray:
    """Generate a data-adaptive 256-entry codebook via k-means.

    Args:
        weight_data: [num_vectors, 8] float32 — groups of 8 weight values
        seed: random seed for k-means initialization
        max_iter: maximum k-means iterations

    Returns: codebook [256, 8] float32 array, each row unit-norm.
    """
    rng = np.random.RandomState(seed)
    n = weight_data.shape[0]
    k = 256

    # Normalize input vectors to unit norm (quantization uses separate scale)
    norms = np.linalg.norm(weight_data, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    data_normed = weight_data / norms

    # Initialize centroids with k-means++ style selection
    centroids = np.zeros((k, 8), dtype=np.float32)
    idx = rng.randint(n)
    centroids[0] = data_normed[idx]

    for i in range(1, k):
        # Distance to nearest existing centroid
        dists = np.min(np.sum((data_normed[:, None, :] - centroids[None, :i, :]) ** 2,
                              axis=2), axis=1)
        probs = dists / dists.sum()
        idx = rng.choice(n, p=probs)
        centroids[i] = data_normed[idx]

    # K-means iterations
    for iteration in range(max_iter):
        # Assign each vector to nearest centroid
        # Batch computation to avoid memory explosion
        batch_size = 100000
        assignments = np.zeros(n, dtype=np.int32)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = data_normed[start:end]
            dists = np.sum((batch[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            assignments[start:end] = np.argmin(dists, axis=1)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.int64)
        for i in range(k):
            mask = assignments == i
            count = mask.sum()
            if count > 0:
                new_centroids[i] = data_normed[mask].mean(axis=0)
                norm = np.linalg.norm(new_centroids[i])
                if norm > 1e-8:
                    new_centroids[i] /= norm
                counts[i] = count

        # Re-seed empty clusters
        empty = counts == 0
        if empty.any():
            for i in np.where(empty)[0]:
                new_centroids[i] = data_normed[rng.randint(n)]
                new_centroids[i] /= np.linalg.norm(new_centroids[i])

        # Check convergence
        shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
        centroids = new_centroids
        if shift < 1e-6:
            break

    return centroids.astype(np.float32)


def save_codebook(codebook: np.ndarray, path: str):
    """Save codebook as raw binary (256 * 8 * float32 = 8192 bytes)."""
    assert codebook.shape == (256, 8)
    assert codebook.dtype == np.float32
    with open(path, 'wb') as f:
        f.write(codebook.tobytes())
    print(f"Saved codebook to {path} ({Path(path).stat().st_size} bytes)")


def load_codebook(path: str) -> np.ndarray:
    """Load codebook from raw binary file."""
    with open(path, 'rb') as f:
        data = f.read()
    assert len(data) == 256 * 8 * 4, f"Expected 8192 bytes, got {len(data)}"
    return np.frombuffer(data, dtype=np.float32).reshape(256, 8)


def main():
    parser = argparse.ArgumentParser(
        description="Generate E8 lattice codebook for 2-bit vector quantization")
    parser.add_argument("--output", default="codebook_e8.bin",
                        help="Output path for codebook binary (default: codebook_e8.bin)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--verify", action="store_true",
                        help="Run verification tests on generated codebook")
    args = parser.parse_args()

    print("Generating E8 lattice codebook (256 entries, 8D, unit-norm)...")
    codebook = generate_e8_codebook(seed=args.seed)

    if args.verify:
        print(f"\nCodebook shape: {codebook.shape}")
        norms = np.linalg.norm(codebook, axis=1)
        print(f"Norm range: [{norms.min():.6f}, {norms.max():.6f}]")

        # Check minimum distance between codewords
        min_dist = float('inf')
        for i in range(256):
            for j in range(i + 1, 256):
                d = np.linalg.norm(codebook[i] - codebook[j])
                if d < min_dist:
                    min_dist = d
        print(f"Minimum inter-codeword distance: {min_dist:.6f}")

        # Check coverage on random Gaussian data
        test_data = np.random.randn(10000, 8).astype(np.float32)
        test_norms = np.linalg.norm(test_data, axis=1, keepdims=True)
        test_normed = test_data / np.maximum(test_norms, 1e-8)
        dists = np.sum((test_normed[:, None, :] - codebook[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(dists, axis=1)
        used = len(np.unique(assignments))
        avg_dist = np.sqrt(np.min(dists, axis=1).mean())
        print(f"Gaussian coverage: {used}/256 codewords used, avg dist = {avg_dist:.4f}")

    save_codebook(codebook, args.output)


if __name__ == "__main__":
    main()
