"""Read what a GGUF file says about itself, without loading any weights.

Used by the model inventory to describe models that are not configured yet:
architecture, size, native context, the KV-cache layout (including hybrid
models where only some layers keep KV), an MTP draft head, sliding-window
attention, the quant mix, and whether the file is a vision projector.

It parses only the header: the metadata key/values and the tensor table.
Unknown value or tensor types are tolerated (forks add their own), so one odd
file never breaks a scan. The KV-sizing readers in core.gpu_pool and
core.engine_llama_cpp are left as they are; this module is additive.
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field

GGUF_MAGIC = 0x46554747  # b"GGUF"

# ggml tensor types -> (name, approximate bits per weight).  Bits/weight
# only feed the "share of bytes" in the quant mix, so approximations are fine.
TENSOR_TYPES = {
    0: ("F32", 32.0), 1: ("F16", 16.0), 2: ("Q4_0", 4.5), 3: ("Q4_1", 5.0),
    6: ("Q5_0", 5.5), 7: ("Q5_1", 6.0), 8: ("Q8_0", 8.5), 9: ("Q8_1", 9.0),
    10: ("Q2_K", 2.625), 11: ("Q3_K", 3.4375), 12: ("Q4_K", 4.5), 13: ("Q5_K", 5.5),
    14: ("Q6_K", 6.5625), 15: ("Q8_K", 9.125), 16: ("IQ2_XXS", 2.0625),
    17: ("IQ2_XS", 2.3125), 18: ("IQ3_XXS", 3.0625), 19: ("IQ1_S", 1.5625),
    20: ("IQ4_NL", 4.5), 21: ("IQ3_S", 3.4375), 22: ("IQ2_S", 2.5), 23: ("IQ4_XS", 4.25),
    24: ("I8", 8.0), 25: ("I16", 16.0), 26: ("I32", 32.0), 27: ("I64", 64.0),
    28: ("F64", 64.0), 29: ("IQ1_M", 1.75), 30: ("BF16", 16.0),
    34: ("TQ1_0", 1.6875), 35: ("TQ2_0", 2.0625), 39: ("MXFP4", 4.25),
}


@dataclass
class GGUFInfo:
    path: str
    file_size: int
    architecture: str = ""
    name: str = ""
    file_type: int | None = None
    block_count: int = 0
    context_length: int = 0
    head_count: int = 0
    head_count_kv: int = 0
    key_length: int = 0
    value_length: int = 0
    full_attention_interval: int = 1
    nextn_predict_layers: int = 0        # MTP draft head layers (0 = none)
    sliding_window: int = 0              # >0: some layers use sliding-window attention
    is_projector: bool = False           # a vision projector (mmproj), not a language model
    has_chat_template: bool = False
    tensor_types: dict = field(default_factory=dict)  # type name -> share of weight bytes
    metadata: dict = field(default_factory=dict)      # scalar keys only (no arrays)

    @property
    def attn_layer_count(self) -> int:
        """Layers that hold a KV cache (hybrid models keep KV on every Nth layer)."""
        if not self.block_count:
            return 0
        # nextn (MTP) layers are counted in block_count but carry their own draft KV.
        base = self.block_count - self.nextn_predict_layers
        if self.full_attention_interval > 1:
            return base // self.full_attention_interval
        return base

    def kv_bytes_per_token(self, bpe_k: float, bpe_v: float, draft_bpe: float | None = None) -> float:
        """KV-cache bytes for one token of context (main model plus any MTP draft head).

        bpe_* are bytes per element (q8_0 = 1.0625, q4_0 = 0.5625, f16 = 2.0).
        Returns 0 when the file lacks the attention keys.
        """
        kv_heads = self.head_count_kv or self.head_count
        if not (kv_heads and self.attn_layer_count):
            return 0.0
        k_dim = self.key_length or self._default_head_dim()
        v_dim = self.value_length or k_dim
        main = self.attn_layer_count * kv_heads * (k_dim * bpe_k + v_dim * bpe_v)
        draft = 0.0
        if self.nextn_predict_layers and draft_bpe is not None:
            draft = self.nextn_predict_layers * kv_heads * (k_dim + v_dim) * draft_bpe
        return main + draft

    def _default_head_dim(self) -> int:
        emb = self.metadata.get(f"{self.architecture}.embedding_length") or 0
        return int(emb // self.head_count) if self.head_count else 0

    @property
    def iq_share(self) -> float:
        """Share of weight bytes in IQ codebook formats (slow on some GPU backends)."""
        return round(sum(v for k, v in self.tensor_types.items() if k.startswith("IQ")), 3)


class GGUFReadError(Exception):
    pass


def _reader(f):
    def u32():
        return struct.unpack("<I", f.read(4))[0]

    def u64():
        return struct.unpack("<Q", f.read(8))[0]

    def string():
        n = u64()
        if n > 1 << 24:
            raise GGUFReadError(f"implausible string length {n}")
        return f.read(n).decode("utf-8", errors="replace")

    scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
              7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def value(t):
        if t in scalar:
            fmt = scalar[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        if t == 8:
            return string()
        if t == 9:  # array: skip the payload of big ones, keep small numeric ones
            at, n = u32(), u64()
            if at == 8:
                for _ in range(n):
                    string()
                return None
            if at in scalar and n <= 512:
                return [value(at) for _ in range(n)]
            if at in scalar:
                f.seek(struct.calcsize(scalar[at]) * n, os.SEEK_CUR)
                return None
            raise GGUFReadError(f"unsupported array element type {at}")
        raise GGUFReadError(f"unknown value type {t}")

    return u32, u64, string, value


def read_gguf(path: str, tensors: bool = True) -> GGUFInfo:
    """Parse a GGUF header. Raises GGUFReadError on a file that is not GGUF."""
    info = GGUFInfo(path=path, file_size=os.path.getsize(path))
    with open(path, "rb") as f:
        u32, u64, string, value = _reader(f)
        try:
            if u32() != GGUF_MAGIC:
                raise GGUFReadError("not a GGUF file")
            version = u32()
            if version < 2:
                raise GGUFReadError(f"GGUF version {version} unsupported")
            n_tensors, n_kv = u64(), u64()
            meta = {}
            for _ in range(n_kv):
                key = string()
                v = value(u32())
                if v is not None:
                    meta[key] = v
        except struct.error as e:
            raise GGUFReadError(f"truncated header: {e}") from None
        info.metadata = {k: v for k, v in meta.items() if not isinstance(v, list)}
        info.has_chat_template = "tokenizer.chat_template" in meta
        arch = str(meta.get("general.architecture", ""))
        info.architecture = arch
        info.name = str(meta.get("general.name", "") or "")
        info.file_type = meta.get("general.file_type")
        info.is_projector = arch == "clip" or str(meta.get("general.type", "")) == "mmproj"

        def a(key, default=0):
            v = meta.get(f"{arch}.{key}", default)
            if isinstance(v, list):  # per-layer arrays: use the largest value
                v = max(v) if v else default
            return v

        info.block_count = int(a("block_count"))
        info.context_length = int(a("context_length"))
        info.head_count = int(a("attention.head_count"))
        info.head_count_kv = int(a("attention.head_count_kv")) or info.head_count
        info.key_length = int(a("attention.key_length"))
        info.value_length = int(a("attention.value_length"))
        info.full_attention_interval = int(a("full_attention_interval", 1)) or 1
        info.nextn_predict_layers = int(a("nextn_predict_layers"))
        info.sliding_window = int(a("attention.sliding_window"))

        if tensors and n_tensors:
            bits = {}
            try:
                for _ in range(n_tensors):
                    string()
                    n_dims = u32()
                    n_el = 1
                    for _ in range(n_dims):
                        n_el *= u64()
                    t = u32()
                    u64()  # data offset
                    name, bpw = TENSOR_TYPES.get(t, (f"type{t}", 8.0))
                    bits[name] = bits.get(name, 0.0) + n_el * bpw
            except struct.error:
                bits = {}
            total = sum(bits.values())
            if total:
                info.tensor_types = {k: round(v / total, 3)
                                     for k, v in sorted(bits.items(), key=lambda kv: -kv[1])}
    return info


def find_projector(model_path: str) -> str | None:
    """A vision projector next to a model: the model's own folder, mmproj*.gguf."""
    folder = os.path.dirname(os.path.abspath(model_path))
    try:
        names = sorted(n for n in os.listdir(folder)
                       if n.lower().endswith(".gguf") and "mmproj" in n.lower())
    except OSError:
        return None
    if not names:
        return None
    stem = os.path.basename(model_path).lower().split("-q")[0].split(".gguf")[0]
    # Prefer a projector that shares the model's name stem, then any.
    for n in names:
        if stem and stem[:12] in n.lower():
            return os.path.join(folder, n)
    return os.path.join(folder, names[0])
