"""
Convert piper en_US-joe-medium.onnx -> safetensors + config.json for the
WebGPU engine. NO onnx package needed: a minimal pure-Python protobuf reader
parses the model (validated approach — f32 LE lives in TensorProto raw_data).

Many initializers are anonymous (fused weight-norm at export), so tensors are
named by graph traversal: the consumer node's name (e.g.
"/flow/flows.0/enc/in_layers.0/Conv") plus the input-slot role (weight/bias).

Outputs into webgpu/models/piper-en-us-joe-medium/:
  - model.safetensors   (all f32 initializers, graph-derived names)
  - config.json         (phoneme_id_map, scales, sample_rate, dims + marker)
  - graph.json          (full node census: op, inputs, outputs, attrs — the
                         gold reference for the TS runtime, avoids re-parsing)

Run:   ./venv/Scripts/python.exe webgpu/scripts/convert-piper-onnx.py
       add --inspect to only print the census.
Gate:  asserts total f32 param count == 15,650,475.
"""
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = ROOT / "models/piper-voices/en_US-joe-medium.onnx"
VOICE_JSON = ROOT / "models/piper-voices/en_US-joe-medium.onnx.json"
OUT_DIR = ROOT / "webgpu/models/piper-en-us-joe-medium"
EXPECT_PARAMS = 15_650_475

# ---------------------------------------------------------------- protobuf --

def read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def iter_fields(buf, start=0, end=None):
    """Yield (field_number, wire_type, value) over a protobuf message."""
    pos = start
    if end is None:
        end = len(buf)
    while pos < end:
        key, pos = read_varint(buf, pos)
        field, wt = key >> 3, key & 7
        if wt == 0:
            val, pos = read_varint(buf, pos)
        elif wt == 1:
            val = buf[pos:pos + 8]
            pos += 8
        elif wt == 2:
            ln, pos = read_varint(buf, pos)
            val = buf[pos:pos + ln]
            pos += ln
        elif wt == 5:
            val = buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wt} at {pos}")
        yield field, wt, val


def parse_packed_varints(val, wt):
    if wt == 0:
        return [val]
    out = []
    pos = 0
    while pos < len(val):
        v, pos = read_varint(val, pos)
        out.append(v)
    return out


def zigzag_to_signed(v):  # dims are int64 but always small/positive here
    return v


DTYPE_NAMES = {1: "f32", 2: "u8", 3: "i8", 6: "i32", 7: "i64", 9: "bool",
               10: "f16", 11: "f64"}


def parse_tensor(buf):
    """TensorProto: 1=dims 2=data_type 8=name 9=raw_data 4=float_data 7=int64_data"""
    t = {"dims": [], "dtype": 0, "name": "", "raw": b"", "floats": None}
    for field, wt, val in iter_fields(buf):
        if field == 1:
            t["dims"].extend(zigzag_to_signed(v) for v in parse_packed_varints(val, wt))
        elif field == 2:
            t["dtype"] = val
        elif field == 8:
            t["name"] = bytes(val).decode("utf-8")
        elif field == 9:
            t["raw"] = bytes(val)
        elif field == 4 and wt == 2:  # packed float_data fallback
            t["floats"] = val
    return t


def parse_attribute(buf):
    """AttributeProto: 1=name 2=f 3=i 4=s 5=t 7=floats 8=ints 20=type"""
    a = {"name": "", "f": None, "i": None, "s": None, "ints": None, "floats": None}
    for field, wt, val in iter_fields(buf):
        if field == 1:
            a["name"] = bytes(val).decode("utf-8")
        elif field == 2:
            a["f"] = struct.unpack("<f", val)[0]
        elif field == 3:
            a["i"] = val
        elif field == 4:
            try:
                a["s"] = bytes(val).decode("utf-8")
            except UnicodeDecodeError:
                a["s"] = repr(bytes(val))
        elif field == 7:
            if wt == 2:
                a["floats"] = list(struct.unpack(f"<{len(val)//4}f", val))
            else:
                a["floats"] = [struct.unpack("<f", val)[0]]
        elif field == 8:
            a["ints"] = parse_packed_varints(val, wt)
    return a


def parse_node(buf):
    """NodeProto: 1=input 2=output 3=name 4=op_type 7=attribute"""
    n = {"inputs": [], "outputs": [], "name": "", "op": "", "attrs": {}}
    for field, wt, val in iter_fields(buf):
        if field == 1:
            n["inputs"].append(bytes(val).decode("utf-8"))
        elif field == 2:
            n["outputs"].append(bytes(val).decode("utf-8"))
        elif field == 3:
            n["name"] = bytes(val).decode("utf-8")
        elif field == 4:
            n["op"] = bytes(val).decode("utf-8")
        elif field == 7:
            a = parse_attribute(val)
            v = a["ints"] if a["ints"] is not None else \
                a["i"] if a["i"] is not None else \
                a["floats"] if a["floats"] is not None else \
                a["f"] if a["f"] is not None else a["s"]
            n["attrs"][a["name"]] = v
    return n


def parse_graph(buf):
    """GraphProto: 1=node 5=initializer 11=input 12=output"""
    nodes, inits, g_in, g_out = [], [], [], []
    for field, wt, val in iter_fields(buf):
        if field == 1:
            nodes.append(parse_node(val))
        elif field == 5:
            inits.append(parse_tensor(val))
        elif field == 11 or field == 12:
            # ValueInfoProto: 1=name
            name = ""
            for f2, w2, v2 in iter_fields(val):
                if f2 == 1:
                    name = bytes(v2).decode("utf-8")
                    break
            (g_in if field == 11 else g_out).append(name)
    return nodes, inits, g_in, g_out


def parse_model(path):
    buf = memoryview(path.read_bytes())
    for field, wt, val in iter_fields(buf):
        if field == 7:  # ModelProto.graph
            return parse_graph(val)
    raise ValueError("no graph in model")

# ------------------------------------------------------------------ naming --

ROLE_BY_OP = {
    # op -> {input_slot: role}
    "Conv": {1: "weight", 2: "bias"},
    "ConvTranspose": {1: "weight", 2: "bias"},
    "Gemm": {1: "weight", 2: "bias"},
    "MatMul": {1: "weight"},
    "Gather": {0: "weight"},  # embedding table
    "Add": {0: "add_const", 1: "add_const"},
    "Sub": {0: "sub_const", 1: "sub_const"},
    "Mul": {0: "mul_const", 1: "mul_const"},
    "Div": {0: "div_const", 1: "div_const"},
}


def derive_names(nodes, inits):
    """Name every f32 initializer from its original name or first consumer."""
    consumers = {}  # tensor name -> (node, slot); first consumer wins
    for n in nodes:
        for slot, inp in enumerate(n["inputs"]):
            if inp and inp not in consumers:
                consumers[inp] = (n, slot)

    named = {}
    taken = set()
    report = []
    for t in inits:
        orig = t["name"]
        if t["dtype"] != 1:  # not f32 — shape consts etc.; TS doesn't need them
            continue
        if orig == "sid":
            # exporter reused the dangling speaker-id input name for the
            # phoneme embedding table (consumed by /enc_p/emb/Gather)
            name = "enc_p.emb.weight"
        elif orig and not orig.startswith("onnx::") and "/" not in orig:
            name = orig
        else:
            c = consumers.get(orig)
            if c is None:
                name = f"unconsumed.{orig or id(t)}"
            else:
                n, slot = c
                base = (n["name"] or n["op"]).strip("/").replace("/", ".")
                role = ROLE_BY_OP.get(n["op"], {}).get(slot, f"in{slot}")
                # trailing token often repeats the op (".../Conv") — keep it out
                parts = base.split(".")
                if parts and parts[-1].split("_")[0] == n["op"]:
                    parts = parts[:-1]
                name = ".".join(parts + [role])
        while name in taken:
            name += "_x"
        taken.add(name)
        named[name] = t
        report.append((name, t["dims"], orig))
    return named, report

# -------------------------------------------------------------------- main --

def main():
    inspect = "--inspect" in sys.argv
    nodes, inits, g_in, g_out = parse_model(ONNX_PATH)
    print(f"nodes: {len(nodes)}   initializers: {len(inits)}")
    print(f"graph inputs: {g_in}   outputs: {g_out}")

    from collections import Counter
    ops = Counter(n["op"] for n in nodes)
    print("op census:", dict(ops.most_common()))

    named, report = derive_names(nodes, inits)
    n_params = sum(
        (len(t["raw"]) // 4) if t["raw"] else (len(t["floats"] or b"") // 4)
        for t in named.values()
    )
    print(f"f32 initializers: {len(named)}   total params: {n_params:,}")

    if inspect:
        anon = sum(1 for _, _, orig in report if not orig or "/" in orig or orig.startswith("onnx::"))
        print(f"anonymous/graph-named: {anon}")
        for name, dims, orig in report:
            print(f"  {name:70s} {str(dims):20s} <- {orig[:40]}")
        return

    assert n_params == EXPECT_PARAMS, f"param count {n_params:,} != {EXPECT_PARAMS:,}"

    import numpy as np
    from safetensors.numpy import save_file

    tensors = {}
    for name, t in named.items():
        raw = t["raw"] if t["raw"] else bytes(t["floats"])
        arr = np.frombuffer(raw, dtype="<f4")
        dims = t["dims"] or [len(arr)]
        assert arr.size == int(np.prod(dims)), f"{name}: {arr.size} vs {dims}"
        tensors[name] = arr.reshape(dims)

    voice = json.loads(VOICE_JSON.read_text(encoding="utf-8"))
    config = {
        "model_type": "piper-vits",
        "voice": "en_US-joe-medium",
        "sample_rate": voice["audio"]["sample_rate"],
        "hop_length": 256,
        "phoneme_type": voice["phoneme_type"],
        "phoneme_id_map": voice["phoneme_id_map"],
        "inference": voice["inference"],  # default noise/length/noise_w scales
        "num_symbols": voice["num_symbols"],
        "dims": {
            "hidden": 192, "enc_layers": 6, "enc_heads": 2, "enc_ffn": 768,
            "rel_window": 4, "flow_layers": 4, "wn_kernel": 5,
            "upsample_kernels": [16, 16, 8], "upsample_strides": [8, 8, 4],
            "resblock_kernels": [3, 5, 7],
            "resblock_dilations": [[1, 2], [2, 6], [3, 12]],
        },
    }

    graph = [{"name": n["name"], "op": n["op"], "inputs": n["inputs"],
              "outputs": n["outputs"], "attrs": n["attrs"]} for n in nodes]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUT_DIR / "model.safetensors")
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    (OUT_DIR / "graph.json").write_text(json.dumps(graph, indent=1), encoding="utf-8")
    size = (OUT_DIR / "model.safetensors").stat().st_size
    print(f"wrote {OUT_DIR}  (safetensors {size/1e6:.1f} MB, {len(tensors)} tensors)")


if __name__ == "__main__":
    main()
