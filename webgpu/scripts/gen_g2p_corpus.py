"""
Regenerate the Piper G2P parity corpus (Phase P5 ground truth).

Phonemizes scripts/g2p_corpus_sentences.txt with piper's bundled, statically
linked espeak-ng (EspeakPhonemizer, en-us) and writes
scripts/piper_fixture/g2p_corpus.json:
  [{ text, phonemes, words, ids }]   — words split on the espeak space phoneme.

The TypeScript G2P (src/audio/g2p.ts) is gated against this file (>=90% exact
word-level phoneme match). This is the G2P-only slice of gen_piper_fixture.py
(no ONNX / onnxruntime), so it is cheap to rerun.

Run: PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe webgpu/scripts/gen_g2p_corpus.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VOICE_JSON = ROOT / "models/piper-voices/en_US-joe-medium.onnx.json"
SENT_PATH = HERE / "g2p_corpus_sentences.txt"
OUT_DIR = HERE / "piper_fixture"
OUT_PATH = OUT_DIR / "g2p_corpus.json"


def main():
    from piper.phonemize_espeak import EspeakPhonemizer
    from piper.phoneme_ids import phonemes_to_ids

    id_map = json.loads(VOICE_JSON.read_text(encoding="utf-8"))["phoneme_id_map"]
    phonemizer = EspeakPhonemizer()

    sentences = [ln.strip() for ln in SENT_PATH.read_text(encoding="utf-8").splitlines()
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
                       "ids": phonemes_to_ids(phonemes, id_map)})

    OUT_DIR.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(corpus, ensure_ascii=False, indent=1), encoding="utf-8")
    n_words = sum(len(c["words"]) for c in corpus)
    print(f"g2p_corpus.json: {len(corpus)} sentences, {n_words} words")


if __name__ == "__main__":
    main()
