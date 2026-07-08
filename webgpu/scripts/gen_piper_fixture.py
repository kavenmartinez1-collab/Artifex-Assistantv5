"""
Piper VITS parity fixture generator (Phase W0/P6 ground truth).

1. Protobuf surgery on en_US-joe-medium.onnx: append four internal tensors as
   graph outputs (no onnx package needed — reuses the pure-Python protobuf
   reader from convert-piper-onnx.py and a tiny writer here):
     /enc_p/Split_output_0   m_p     [1,192,P]
     /enc_p/Split_output_1   logs_p  [1,192,P]
     /Ceil_output_0          durations (post ceil(d*length_scale)) [1,1,P]
     /Mul_7_output_0         z into dec.conv_pre (post-flow, masked) [1,192,F]
2. onnxruntime on the modified model, scales=[0,1,0] (verified deterministic),
   two fixture sentences; dumps all taps + waveform (+ default-scales waveform
   for listening, not gated).
3. g2p_corpus.json: espeak ground-truth phonemization of ~200 sentences via
   piper's bundled statically-linked espeak-ng (EspeakPhonemizer) for the
   Phase P5 G2P gate (>=90% exact word-level match).

Run: PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe webgpu/scripts/gen_piper_fixture.py
Out: webgpu/scripts/piper_fixture/  (gitignored)
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ONNX_PATH = ROOT / "models/piper-voices/en_US-joe-medium.onnx"
VOICE_JSON = ROOT / "models/piper-voices/en_US-joe-medium.onnx.json"
OUT_DIR = HERE / "piper_fixture"

spec = importlib.util.spec_from_file_location("conv", HERE / "convert-piper-onnx.py")
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)

TAPS = {
    "m_p": "/enc_p/Split_output_0",
    "logs_p": "/enc_p/Split_output_1",
    "durations": "/Ceil_output_0",
    "z": "/Mul_7_output_0",
}

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming how we interact with computers.",
]

# ------------------------------------------------------- protobuf surgery --

def enc_varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def enc_field(field: int, wt: int, payload) -> bytes:
    key = enc_varint((field << 3) | wt)
    if wt == 2:
        return key + enc_varint(len(payload)) + bytes(payload)
    if wt == 0:
        return key + enc_varint(payload)
    return key + bytes(payload)  # wt 1/5: fixed-width raw bytes


def value_info_f32(name: str) -> bytes:
    """ValueInfoProto{name, type: TypeProto{tensor_type: {elem_type: 1}}}"""
    tensor_type = enc_field(1, 0, 1)          # elem_type = FLOAT
    type_proto = enc_field(1, 2, tensor_type)  # TypeProto.tensor_type
    return enc_field(1, 2, name.encode()) + enc_field(2, 2, type_proto)


def add_graph_outputs(model_bytes: bytes, names: list[str]) -> bytes:
    """Re-emit ModelProto with extra GraphProto.output (field 12) entries."""
    out = bytearray()
    for field, wt, val in conv.iter_fields(memoryview(model_bytes)):
        if field == 7 and wt == 2:  # ModelProto.graph
            graph = bytes(val)
            for n in names:
                graph += enc_field(12, 2, value_info_f32(n))
            out += enc_field(7, 2, graph)
        else:
            out += enc_field(field, wt, val)
    return bytes(out)

# --------------------------------------------------------------- phonemes --

def make_phonemizer():
    from piper.phonemize_espeak import EspeakPhonemizer
    return EspeakPhonemizer()


def to_ids(phonemes: list[str], id_map) -> list[int]:
    from piper.phoneme_ids import phonemes_to_ids
    return phonemes_to_ids(phonemes, id_map)

# ------------------------------------------------------------------- main --

def main():
    voice = json.loads(VOICE_JSON.read_text(encoding="utf-8"))
    id_map = voice["phoneme_id_map"]
    OUT_DIR.mkdir(exist_ok=True)

    modified = add_graph_outputs(ONNX_PATH.read_bytes(), list(TAPS.values()))
    tmp_model = OUT_DIR / "_tapped.onnx"
    tmp_model.write_bytes(modified)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(tmp_model), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    print("session outputs:", out_names)

    phonemizer = make_phonemizer()
    manifest = {"meta": {"voice": "en_US-joe-medium",
                         "sample_rate": voice["audio"]["sample_rate"],
                         "scales": [0.0, 1.0, 0.0]},
                "tensors": {}}

    def dump(name: str, arr: np.ndarray):
        f = name.replace("/", "_").replace(".", "_") + ".bin"
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        (OUT_DIR / f).write_bytes(arr.tobytes())
        manifest["tensors"][name] = {"file": f, "shape": list(arr.shape), "dtype": "f32"}

    for si, text in enumerate(SENTENCES):
        sents = phonemizer.phonemize("en-us", text)
        phonemes = [p for s in sents for p in s]
        ids = to_ids(phonemes, id_map)
        px = f"s{si}"
        manifest["meta"][f"{px}.text"] = text
        manifest["meta"][f"{px}.phonemes"] = phonemes
        manifest["meta"][f"{px}.ids"] = ids

        feed = {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }
        res = sess.run(["output", *TAPS.values()], feed)
        wav, m_p, logs_p, durs, z = res
        dump(f"{px}.waveform", wav.reshape(-1))
        dump(f"{px}.m_p", m_p[0])          # [192, P]
        dump(f"{px}.logs_p", logs_p[0])    # [192, P]
        dump(f"{px}.durations", durs.reshape(-1))
        dump(f"{px}.z", z[0])              # [192, F]

        # determinism spot-check
        wav2 = sess.run(["output"], feed)[0]
        assert np.array_equal(wav, wav2), "zero-noise output not deterministic!"

        # default-scales waveform for listening (not a parity target)
        feed["scales"] = np.array([0.667, 1.0, 0.8], dtype=np.float32)
        wav_d = sess.run(["output"], feed)[0].reshape(-1)
        write_wav(OUT_DIR / f"{px}_default.wav", wav_d, voice["audio"]["sample_rate"])
        write_wav(OUT_DIR / f"{px}_zero.wav", wav.reshape(-1), voice["audio"]["sample_rate"])

        print(f"{px}: {len(ids)} ids, durations sum {int(durs.sum())} frames, "
              f"wav {wav.size} samples ({wav.size/voice['audio']['sample_rate']:.2f}s)")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    # ---------------- G2P corpus (Phase P5 gate ground truth) ----------------
    corpus_path = HERE / "g2p_corpus_sentences.txt"
    sentences = [ln.strip() for ln in corpus_path.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
    corpus = []
    for text in sentences:
        sents = phonemizer.phonemize("en-us", text)
        phonemes = [p for s in sents for p in s]
        # word-level split on the space phoneme for the word-match gate
        words, cur = [], []
        for p in phonemes:
            if p == " ":
                if cur:
                    words.append("".join(cur))
                cur = []
            else:
                cur.append(p)
        if cur:
            words.append("".join(cur))
        corpus.append({"text": text, "phonemes": phonemes, "words": words,
                       "ids": to_ids(phonemes, id_map)})
    (OUT_DIR / "g2p_corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"g2p_corpus.json: {len(corpus)} sentences")

    tmp_model.unlink()  # 63 MB temp — not needed after the run


def write_wav(path: Path, samples: np.ndarray, rate: int):
    import soundfile as sf
    sf.write(str(path), samples.astype(np.float32), rate)


if __name__ == "__main__":
    main()
