"""Regression tests for the transformers engine (serving-core review,
2026-11-09).  No model, tokenizer or GPU is touched: generate(), the
streamer and CUDA are doubles.
"""

import queue
import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

import core.engine_transformers as et   # noqa: E402
from core import inference              # noqa: E402


# ── #14: a Stop raised from on_token must end model.generate() ─────────

class FakeStreamer:
    """TextIteratorStreamer stand-in: a queue the generate thread feeds."""

    def __init__(self, *a, **k):
        self._q = queue.Queue()

    def put_text(self, text):
        self._q.put(text)

    def end(self):
        self._q.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get(timeout=10)
        if item is None:
            raise StopIteration
        return item


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeModel:
    device = "cpu"
    config = types.SimpleNamespace(num_hidden_layers=2)

    def __init__(self, max_steps=400):
        self.max_steps = max_steps
        self.steps = 0
        self.stopped_by_criteria = False
        self.finished = threading.Event()

    def generate(self, streamer=None, stopping_criteria=(), input_ids=None, **kw):
        try:
            for _ in range(self.max_steps):
                self.steps += 1
                streamer.put_text("tok ")
                time.sleep(0.002)
                if any(bool(torch.as_tensor(c(input_ids, None)).all())
                       for c in stopping_criteria):
                    self.stopped_by_criteria = True
                    return
        finally:
            self.finished.set()


class FakeTokenizer:
    eos_token_id = 1
    pad_token_id = 0
    unk_token_id = 3

    def convert_tokens_to_ids(self, token):
        return None

    def encode(self, text, add_special_tokens=False):
        return text.split()


@pytest.fixture
def engine(monkeypatch):
    e = et.TransformersEngine()
    e.model, e.tokenizer = FakeModel(), FakeTokenizer()
    e._gpu_tier = {"tier": "ABUNDANT", "compute_dtype": torch.float16}
    monkeypatch.setattr(e, "_build_inputs_for_messages",
                        lambda m, t: FakeInputs(input_ids=torch.zeros((1, 3), dtype=torch.long)))
    monkeypatch.setattr(et, "TextIteratorStreamer", FakeStreamer)
    monkeypatch.setattr(et, "StopStringCriteria",
                        lambda tokenizer, stop_strings: (lambda ids, scores, **k: False))
    monkeypatch.setattr(et, "StoppingCriteriaList", list)
    monkeypatch.setattr(et, "get_turboquant_kv", lambda: False)
    # never let the cleanup path initialise CUDA in a test process
    monkeypatch.setattr(et.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(inference, "_tokenizer", None)
    return e


class StopPressed(Exception):
    pass


def test_on_token_exception_stops_generate_and_is_reraised(engine):
    seen = []

    def on_token(text):
        seen.append(text)
        if len(seen) == 3:
            raise StopPressed("stop")

    with pytest.raises(StopPressed):
        engine.generate_streaming([{"role": "user", "content": "hi"}], 64, 0.7,
                                  on_token=on_token)
    model = engine.model
    # By the time the caller gets its exception back, generate() is OVER:
    # nothing keeps decoding on the GPU while the next request starts.
    assert model.finished.is_set(), "generate() was left running in the background"
    assert model.stopped_by_criteria and model.steps < model.max_steps
    assert len(seen) == 3


def test_normal_generation_is_unaffected(engine):
    engine.model = FakeModel(max_steps=5)
    seen = []
    text = engine.generate_streaming([{"role": "user", "content": "hi"}], 64, 0.7,
                                     on_token=seen.append, raw_output=True)
    assert seen == ["tok "] * 5 and text == "tok tok tok tok tok"
    assert not engine.model.stopped_by_criteria
    assert engine._last_gen_stats["completion_tokens"] == 5


# ── #18: token budgeting must not keep a stale tokenizer ───────────────

def test_generate_publishes_the_loaded_models_tokenizer(engine):
    stale = object()
    inference._tokenizer = stale            # left by a previously loaded model
    engine.model = FakeModel(max_steps=1)
    engine.generate_streaming([{"role": "user", "content": "hi"}], 8, 0.7)
    assert inference._tokenizer is engine.tokenizer


def test_unload_withdraws_its_tokenizer(engine):
    inference._tokenizer = engine.tokenizer
    engine.unload()
    assert inference._tokenizer is None
    assert inference._count_tokens([{"role": "user", "content": "x" * 40}]) == 10


def test_unload_leaves_another_engines_tokenizer_alone(engine):
    other = object()
    inference._tokenizer = other
    engine.unload()
    assert inference._tokenizer is other


# ── #19: load edge cases ───────────────────────────────────────────────

def test_no_model_folder_gives_the_help_text_not_a_type_error(monkeypatch):
    e = et.TransformersEngine()
    monkeypatch.setattr(et.torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))
    monkeypatch.setattr(et, "get_active_model_path", lambda: None)
    with pytest.raises(FileNotFoundError) as err:
        e.load()
    assert "download_model.py" in str(err.value)


def _cache(tmp_path, files):
    d = tmp_path / "model-nf4-cached"
    d.mkdir()
    for name in files:
        (d / name).write_bytes(b"x")
    return str(d)


def test_legacy_single_file_cache_without_meta_still_loads(tmp_path):
    e = et.TransformersEngine()
    assert e._has_cached_quantized(_cache(tmp_path, ["config.json", "model.safetensors"]))


def test_finished_sharded_cache_loads(tmp_path):
    e = et.TransformersEngine()
    files = ["config.json", "model-00001-of-00002.safetensors",
             "model-00002-of-00002.safetensors", "model.safetensors.index.json"]
    assert e._has_cached_quantized(_cache(tmp_path, files))


@pytest.mark.parametrize("files", [
    # killed while writing the shards
    ["config.json", "model-00001-of-00003.safetensors"],
    # all shards there, the index (written last) is not
    ["config.json", "model-00001-of-00002.safetensors",
     "model-00002-of-00002.safetensors"],
])
def test_interrupted_sharded_save_is_not_taken_for_a_cache(tmp_path, files):
    e = et.TransformersEngine()
    assert not e._has_cached_quantized(_cache(tmp_path, files))


def test_meta_file_marks_a_finished_save(tmp_path):
    e = et.TransformersEngine()
    files = ["config.json", "model-00001-of-00002.safetensors",
             "model-00002-of-00002.safetensors", "_quant_meta.json"]
    assert e._has_cached_quantized(_cache(tmp_path, files))


def test_config_only_folder_is_still_not_a_cache(tmp_path):
    e = et.TransformersEngine()
    assert not e._has_cached_quantized(_cache(tmp_path, ["config.json"]))
