"""
TurboQuant+ KV Cache — PyTorch implementation.

Ports the TurboQuant algorithm (Google, ICLR 2026) with improvements from
TurboQuant+ (TheTom) to a transformers-compatible Cache wrapper.

Architecture:
  TurboQuantCache wraps the model's native cache (DynamicCache, Qwen3.5
  hybrid cache, etc.) and intercepts update() to compress old tokens.
  All other attributes and methods proxy transparently to the inner cache,
  so SSM state (conv_states, recurrent_states, has_previous_state) and
  any model-specific cache features work unchanged.

Algorithm:
  Stage 1 (PolarQuant): normalize → WHT rotate → Lloyd-Max scalar quantize
  Stage 2 (QJL):        residual → JL project → 1-bit sign correction

TurboQuant+ improvements over base paper:
  - Walsh-Hadamard rotation: O(n log n) vs O(n²) random orthogonal matrix
  - Asymmetric K/V: keys at 3-bit, values at 2-bit ("V compression is free")
  - Boundary layer protection: first/last 2 layers stay full precision

Acknowledgement:
  TurboQuant+ findings from github.com/TheTom/turboquant_plus

Usage:
    from core.turboquant_cache import TurboQuantCache
    cache = TurboQuantCache(inner_cache, key_bits=3, value_bits=2)
    # Pass as past_key_values to model.generate()
"""

import math
import torch
from transformers.cache_utils import Cache


# ═══════════════════════════════════════════════════════════════════════════
# LLOYD-MAX OPTIMAL CODEBOOKS FOR N(0,1)
# ═══════════════════════════════════════════════════════════════════════════

CODEBOOK = {
    1: {
        "centroids": [0.7979],
        "thresholds": [],
    },
    2: {
        "centroids": [0.4528, 1.5104],
        "thresholds": [0.9816],
    },
    3: {
        "centroids": [0.2451, 0.7560, 1.3440, 2.1520],
        "thresholds": [0.5006, 1.0500, 1.7480],
    },
    4: {
        "centroids": [0.1284, 0.3881, 0.6568, 0.9423, 1.2562, 1.6180, 2.0690, 2.7326],
        "thresholds": [0.2582, 0.5224, 0.7996, 1.0993, 1.4371, 1.8435, 2.4008],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# FAST WALSH-HADAMARD TRANSFORM
# ═══════════════════════════════════════════════════════════════════════════

def fast_wht(x):
    """Vectorized Fast Walsh-Hadamard Transform on the last dimension.

    O(n log n) — replaces the O(n²) random orthogonal matrix multiply.
    WHT is self-inverse when normalized: WHT(WHT(x)) = x.

    Args:
        x: (..., d) tensor where d is a power of 2
    Returns:
        (..., d) WHT-transformed tensor, normalized by 1/sqrt(d)
    """
    d = x.shape[-1]
    result = x.clone()
    h = 1
    while h < d:
        r = result.reshape(*result.shape[:-1], -1, 2 * h)
        a = r[..., :h].clone()
        b = r[..., h:].clone()
        r[..., :h] = a + b
        r[..., h:] = a - b
        result = r.reshape(*x.shape)
        h *= 2
    return result / math.sqrt(d)


def _generate_jl_matrix(d, seed=137, device="cpu"):
    """Generate a d×d Johnson-Lindenstrauss projection matrix."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(d, d, generator=gen, dtype=torch.float32, device="cpu").to(device) / math.sqrt(d)


# ═══════════════════════════════════════════════════════════════════════════
# TURBOQUANT+ CODEC
# ═══════════════════════════════════════════════════════════════════════════

class TurboQuantCodec:
    """Encoder/decoder for TurboQuant+ KV compression."""

    def __init__(self, head_dim, bits=3, device="cpu", dtype=torch.float16):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device
        self.dtype = dtype

        cb = CODEBOOK[bits]
        self.centroids = torch.tensor(cb["centroids"], dtype=torch.float32, device=device)
        self.thresholds = torch.tensor(cb["thresholds"], dtype=torch.float32, device=device)
        self.num_centroids = len(cb["centroids"])
        self.sqrt_d = math.sqrt(head_dim)

        self.jl_matrix = _generate_jl_matrix(head_dim, seed=137, device=device)

    def to(self, device):
        self.device = device
        self.centroids = self.centroids.to(device)
        self.thresholds = self.thresholds.to(device)
        self.jl_matrix = self.jl_matrix.to(device)
        return self

    def encode(self, vectors, compute_qjl=True):
        """Encode vectors to compressed representation."""
        d = self.head_dim
        flat = vectors.reshape(-1, d).float()

        norms = torch.linalg.norm(flat, dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = flat / norms

        rotated = fast_wht(normalized)

        scaled = rotated * self.sqrt_d
        abs_val = scaled.abs()
        sign = scaled.sign()

        if self.thresholds.numel() > 0:
            bins = (abs_val.unsqueeze(-1) > self.thresholds.view(1, 1, -1)).sum(dim=-1)
        else:
            bins = torch.zeros_like(abs_val, dtype=torch.long)
        bins = bins.long()

        dequant_abs = self.centroids[bins] / self.sqrt_d
        dequantized = sign * dequant_abs

        indices = torch.where(sign >= 0, bins, bins + self.num_centroids)

        result = {
            "indices": indices.to(torch.uint8),
            "norms": norms.squeeze(-1).to(self.dtype),
        }

        if compute_qjl:
            residual = rotated - dequantized
            residual_norms = torch.linalg.norm(residual, dim=-1, keepdim=True).clamp(min=1e-8)

            jl_projected = residual @ self.jl_matrix.T
            sign_raw = (jl_projected >= 0).to(torch.uint8)

            sign_grouped = sign_raw.reshape(-1, d // 8, 8)
            bit_weights = (2 ** torch.arange(8, device=flat.device)).to(torch.uint8)
            sign_packed = (sign_grouped * bit_weights).sum(dim=-1).to(torch.uint8)

            result["sign_packed"] = sign_packed
            result["residual_norms"] = residual_norms.squeeze(-1).to(self.dtype)

        return result

    def _unpack_sign_bits(self, sign_packed):
        d = self.head_dim
        bit_weights = (2 ** torch.arange(8, device=sign_packed.device)).to(torch.uint8)
        unpacked = ((sign_packed.unsqueeze(-1) & bit_weights) > 0).float()
        return (2.0 * unpacked.reshape(-1, d) - 1.0)

    def decode(self, encoded):
        """Decode compressed representation back to approximate vectors."""
        indices = encoded["indices"].long()
        norms = encoded["norms"].float()

        is_negative = indices >= self.num_centroids
        bins = torch.where(is_negative, indices - self.num_centroids, indices)
        sign = torch.where(is_negative, -1.0, 1.0)

        dequant_abs = self.centroids[bins] / self.sqrt_d
        dequantized_rotated = sign * dequant_abs

        reconstructed = fast_wht(dequantized_rotated)
        return (norms.unsqueeze(-1) * reconstructed).to(self.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# TRANSPARENT CACHE WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

class TurboQuantCache(Cache):
    """Transparent wrapper that adds TurboQuant+ compression to any cache.

    Wraps the model's native cache object and intercepts update() to
    compress old KV tokens. All other attribute access (conv_states,
    recurrent_states, has_previous_state, key_cache, value_cache, etc.)
    proxies transparently to the inner cache.

    This design works with any model's cache:
      - DynamicCache (standard transformers)
      - Qwen3.5 hybrid cache (with SSM conv/recurrent states)
      - Jamba, Zamba, or any future hybrid architecture

    Args:
        inner: The model's native cache instance (created by the model).
        key_bits: Quantization bits for keys (1-4). Default 3.
        value_bits: Quantization bits for values (1-4). Default 2.
        residual_length: Recent tokens kept at full precision.
        boundary_layers: First/last N layers stay at full precision.
        num_layers: Total model layers (for last-N boundary detection).
    """

    def __init__(self, inner, key_bits=3, value_bits=2, residual_length=128,
                 boundary_layers=2, num_layers=None):
        # Don't call super().__init__() — we proxy everything to inner
        self._inner = inner
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.residual_length = residual_length
        self.boundary_layers = boundary_layers
        self.num_layers = num_layers
        self._k_codecs = {}
        self._v_codecs = {}
        self._compressed = {}
        self._compressed_len = {}
        self._max_layer_seen = 0

    # ── Transparent proxy ─────────────────────────────────────────────────
    # Any attribute not on TurboQuantCache falls through to the inner cache.
    # This is what makes conv_states, recurrent_states, has_previous_state,
    # key_cache, value_cache, layers, etc. all work without patching.

    def __getattr__(self, name):
        # __getattr__ is only called when normal lookup fails,
        # so our own attributes (key_bits, _inner, etc.) resolve normally.
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        # Our own attributes go on self, everything else on inner
        if name.startswith('_') or name in (
            'key_bits', 'value_bits', 'residual_length',
            'boundary_layers', 'num_layers',
        ):
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)

    def __len__(self):
        return len(self._inner)

    def __bool__(self):
        return bool(self._inner)

    def __iter__(self):
        return iter(self._inner)

    # ── Cache interface ───────────────────────────────────────────────────

    def _is_boundary_layer(self, layer_idx):
        if self.boundary_layers <= 0:
            return False
        if layer_idx < self.boundary_layers:
            return True
        self._max_layer_seen = max(self._max_layer_seen, layer_idx)
        total = self.num_layers or (self._max_layer_seen + 1)
        if layer_idx >= total - self.boundary_layers:
            return True
        return False

    def _get_k_codec(self, head_dim, device):
        if head_dim not in self._k_codecs:
            self._k_codecs[head_dim] = TurboQuantCodec(
                head_dim=head_dim, bits=self.key_bits, device=device
            )
        codec = self._k_codecs[head_dim]
        if codec.device != device:
            codec.to(device)
        return codec

    def _get_v_codec(self, head_dim, device):
        if head_dim not in self._v_codecs:
            self._v_codecs[head_dim] = TurboQuantCodec(
                head_dim=head_dim, bits=self.value_bits, device=device
            )
        codec = self._v_codecs[head_dim]
        if codec.device != device:
            codec.to(device)
        return codec

    def _compress_overflow(self, layer_idx, key_states, value_states):
        seq_len = key_states.shape[2]
        if seq_len <= self.residual_length:
            return key_states, value_states

        n_compress = seq_len - self.residual_length
        k_old, k_recent = key_states[:, :, :n_compress], key_states[:, :, n_compress:]
        v_old, v_recent = value_states[:, :, :n_compress], value_states[:, :, n_compress:]

        batch, heads, old_len, head_dim = k_old.shape
        device = key_states.device
        k_codec = self._get_k_codec(head_dim, device)
        v_codec = self._get_v_codec(head_dim, device)

        k_flat = k_old.reshape(-1, head_dim)
        v_flat = v_old.reshape(-1, head_dim)

        k_enc = k_codec.encode(k_flat, compute_qjl=True)
        v_enc = v_codec.encode(v_flat, compute_qjl=False)

        prev = self._compressed.get(layer_idx)
        if prev is not None:
            for key in k_enc:
                k_enc[key] = torch.cat([prev["keys"][key], k_enc[key]], dim=0)
            for key in v_enc:
                v_enc[key] = torch.cat([prev["values"][key], v_enc[key]], dim=0)
            self._compressed_len[layer_idx] = prev["shape"][2] + old_len
        else:
            self._compressed_len[layer_idx] = old_len

        self._compressed[layer_idx] = {
            "keys": k_enc,
            "values": v_enc,
            "shape": (batch, heads, self._compressed_len[layer_idx], head_dim),
        }

        return k_recent, v_recent

    def _reconstruct_full(self, layer_idx, key_states, value_states):
        compressed = self._compressed.get(layer_idx)
        if compressed is None:
            return key_states, value_states

        batch, heads, comp_len, head_dim = compressed["shape"]
        device = key_states.device
        k_codec = self._get_k_codec(head_dim, device)
        v_codec = self._get_v_codec(head_dim, device)

        k_decoded = k_codec.decode(compressed["keys"]).reshape(batch, heads, comp_len, head_dim)
        v_decoded = v_codec.decode(compressed["values"]).reshape(batch, heads, comp_len, head_dim)

        full_k = torch.cat([k_decoded, key_states], dim=2)
        full_v = torch.cat([v_decoded, value_states], dim=2)

        return full_k, full_v

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Intercept update: delegate to inner cache, then compress overflow."""
        # Let the inner cache do its native update (handles SSM state etc.)
        keys, values = self._inner.update(key_states, value_states, layer_idx, cache_kwargs)

        # Boundary layers: no compression
        if self._is_boundary_layer(layer_idx):
            return keys, values

        # Only compress attention layers (skip SSM layers that don't return 4D tensors)
        if keys.ndim != 4:
            return keys, values

        # Compress tokens beyond residual_length
        k_recent, v_recent = self._compress_overflow(layer_idx, keys, values)

        # Update inner cache to only hold recent tokens
        if hasattr(self._inner, 'key_cache') and self._inner.key_cache is not None:
            if isinstance(self._inner.key_cache, list) and layer_idx < len(self._inner.key_cache):
                self._inner.key_cache[layer_idx] = k_recent
                self._inner.value_cache[layer_idx] = v_recent

        # Reconstruct full K/V for this attention computation
        full_k, full_v = self._reconstruct_full(layer_idx, k_recent, v_recent)

        return full_k, full_v

    def get_seq_length(self, layer_idx=0):
        base_len = self._inner.get_seq_length(layer_idx)
        compressed_len = self._compressed_len.get(layer_idx, 0)
        return base_len + compressed_len

    def reset(self):
        self._inner.reset()
        self._compressed.clear()
        self._compressed_len.clear()

    # Forward remaining Cache interface methods to inner
    def reorder_cache(self, beam_idx):
        return self._inner.reorder_cache(beam_idx)

    def get_max_cache_shape(self, layer_idx=0):
        return self._inner.get_max_cache_shape(layer_idx)


def wrap_cache_with_turboquant(cache, num_layers=None, key_bits=3, value_bits=2,
                                residual_length=128, boundary_layers=2):
    """Convenience: wrap any cache instance with TurboQuant+ compression.

    Usage in generate():
        # After model creates its cache, wrap it:
        cache = wrap_cache_with_turboquant(model_cache, num_layers=32)
    """
    return TurboQuantCache(
        inner=cache,
        key_bits=key_bits,
        value_bits=value_bits,
        residual_length=residual_length,
        boundary_layers=boundary_layers,
        num_layers=num_layers,
    )
