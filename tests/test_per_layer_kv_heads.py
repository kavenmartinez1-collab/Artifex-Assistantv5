"""Hybrid models that store attention.head_count_kv PER LAYER.

LFM2/LFM3 interleave short-convolution layers (no KV cache) with attention
layers, and write head_count_kv as a list with 0 on the conv layers. The
KV estimators multiplied attn_layers * head_count_kv, so a list raised
"can't multiply sequence by non-int" and every chat with LFM3 was a 500.
"""

import struct

import pytest

from core.gpu_pool import per_layer_kv_heads, read_gguf_kv_params
from core.engine_llama_cpp import _read_gguf_kv_params
from core.gguf_meta import read_gguf

# LFM3-1.7B's real layout: 24 blocks, attention on every other layer from 2.
LFM3_KV = [0, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0]


def _write_gguf(path, arch, head_count_kv):
    """Minimal GGUF v3: header + metadata, no tensors (all the readers need)."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    def u32(key, v):
        return s(key) + struct.pack("<II", 4, v)

    kv = [s("general.architecture") + struct.pack("<I", 8) + s(arch),
          u32(f"{arch}.block_count", len(LFM3_KV)),
          u32(f"{arch}.context_length", 262144),
          u32(f"{arch}.embedding_length", 2048),
          u32(f"{arch}.attention.head_count", 16),
          u32(f"{arch}.attention.key_length", 128),
          u32(f"{arch}.attention.value_length", 128)]
    if isinstance(head_count_kv, list):     # ARRAY (9) of INT32 (5)
        kv.append(s(f"{arch}.attention.head_count_kv") + struct.pack("<IIQ", 9, 5, len(head_count_kv))
                  + b"".join(struct.pack("<i", h) for h in head_count_kv))
    else:
        kv.append(u32(f"{arch}.attention.head_count_kv", head_count_kv))
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(kv))
    return str(path)


def test_normaliser():
    assert per_layer_kv_heads(LFM3_KV, 24) == (4, 11)
    assert per_layer_kv_heads([0, 0, 0], 3) == (0, 0)
    assert per_layer_kv_heads(8, 64) == (8, 64)      # scalar: untouched


@pytest.mark.parametrize("reader", [read_gguf_kv_params, _read_gguf_kv_params])
def test_readers_count_only_kv_layers(tmp_path, reader):
    p = _write_gguf(tmp_path / "lfm3.gguf", "lfm3", LFM3_KV)
    params = reader(p)
    assert params["head_count_kv"] == 4
    assert params["attn_layer_count"] == 11


@pytest.mark.parametrize("reader", [read_gguf_kv_params, _read_gguf_kv_params])
def test_scalar_models_unchanged(tmp_path, reader):
    p = _write_gguf(tmp_path / "llama.gguf", "llama", 4)
    params = reader(p)
    assert params["head_count_kv"] == 4
    assert params["attn_layer_count"] == 24


def test_gguf_meta_kv_bytes(tmp_path):
    info = read_gguf(_write_gguf(tmp_path / "lfm3.gguf", "lfm3", LFM3_KV),
                     tensors=False)
    assert info.attn_layer_count == 11
    # 11 layers x 4 heads x (128 + 128) dims x 2 bytes (f16)
    assert info.kv_bytes_per_token(2.0, 2.0) == 11 * 4 * 256 * 2
