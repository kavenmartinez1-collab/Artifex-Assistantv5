"""Model inventory: GGUF metadata, family defaults, scan statuses, proposals and config writes.

Every GGUF here is a tiny synthetic header (metadata plus a tensor table, no
weights), written into tmp_path. Nothing touches the real config or models.
"""
import json
import os
import struct

import pytest

from core import gguf_meta, model_families, model_inventory as mi

Q4_K, Q3_K, Q6_K, IQ3_S, BF16 = 12, 11, 14, 21, 30


def _gguf(path, kv, tensors=(), pad_to=0):
    """Write a GGUF v3 header: kv {key: int|str|float}, tensors [(name, n_elements, type)]."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, v in kv.items():
        if isinstance(v, str):
            out += s(k) + struct.pack("<I", 8) + s(v)
        elif isinstance(v, float):
            out += s(k) + struct.pack("<If", 6, v)
        elif isinstance(v, list):
            out += s(k) + struct.pack("<IIQ", 9, 4, len(v)) + b"".join(struct.pack("<I", x) for x in v)
        else:
            out += s(k) + struct.pack("<II", 4, v)
    for name, n_el, t in tensors:
        out += s(name) + struct.pack("<I", 1) + struct.pack("<Q", n_el) + struct.pack("<IQ", t, 0)
    out += b"\0" * max(0, pad_to - len(out))
    path.write_bytes(out)
    return str(path)


QWEN38 = {"general.architecture": "qwen35", "general.name": "Qwen3.8 27B",
          "qwen35.block_count": 65, "qwen35.context_length": 262144,
          "qwen35.attention.head_count": 24, "qwen35.attention.head_count_kv": 4,
          "qwen35.embedding_length": 5120, "qwen35.attention.key_length": 256,
          "qwen35.attention.value_length": 256, "qwen35.full_attention_interval": 4,
          "qwen35.nextn_predict_layers": 1, "tokenizer.chat_template": "{{ x }}"}
GEMMA = {"general.architecture": "gemma3", "gemma3.block_count": 34, "gemma3.context_length": 131072,
         "gemma3.attention.head_count": 8, "gemma3.attention.head_count_kv": [4, 4, 4],
         "gemma3.embedding_length": 2560, "gemma3.attention.sliding_window": 1024,
         "tokenizer.chat_template": "{{ x }}"}
CLIP = {"general.architecture": "clip", "general.type": "mmproj"}


# ---- GGUF metadata -------------------------------------------------------

def test_reads_hybrid_layout_mtp_and_quant_mix(tmp_path):
    p = _gguf(tmp_path / "q.gguf", QWEN38, [("a", 600, Q3_K), ("b", 300, Q4_K), ("c", 100, Q6_K)])
    i = gguf_meta.read_gguf(p)
    assert (i.architecture, i.block_count, i.context_length) == ("qwen35", 65, 262144)
    assert i.nextn_predict_layers == 1 and i.attn_layer_count == 16   # (65 - 1 MTP) / 4
    assert i.has_chat_template and not i.is_projector
    assert list(i.tensor_types) == ["Q3_K", "Q4_K", "Q6_K"] and i.iq_share == 0
    # 16 layers x 4 KV heads x 512 dims x q8_0, plus 1 MTP layer at q4_0
    assert i.kv_bytes_per_token(1.0625, 1.0625, 0.5625) == 16 * 4 * 512 * 1.0625 + 4 * 512 * 0.5625


def test_per_layer_kv_heads_sliding_window_and_iq_share(tmp_path):
    p = _gguf(tmp_path / "g.gguf", GEMMA, [("a", 700, IQ3_S), ("b", 300, Q4_K)])
    i = gguf_meta.read_gguf(p)
    assert i.head_count_kv == 4 and i.sliding_window == 1024
    assert i.iq_share > 0.6


def test_projector_and_bad_files(tmp_path):
    assert gguf_meta.read_gguf(_gguf(tmp_path / "mmproj-x-f16.gguf", CLIP, [("v", 10, BF16)])).is_projector
    (tmp_path / "junk.gguf").write_bytes(b"not a gguf at all")
    with pytest.raises(gguf_meta.GGUFReadError):
        gguf_meta.read_gguf(str(tmp_path / "junk.gguf"))


def test_unknown_tensor_type_is_tolerated(tmp_path):
    i = gguf_meta.read_gguf(_gguf(tmp_path / "fork.gguf", QWEN38, [("a", 10, 142)]))
    assert "type142" in i.tensor_types


def test_find_projector_prefers_matching_name(tmp_path):
    m = _gguf(tmp_path / "Qwen3.8-27B-Q4_K_S.gguf", QWEN38)
    _gguf(tmp_path / "mmproj-other-f16.gguf", CLIP)
    want = _gguf(tmp_path / "mmproj-Qwen3.8-27B-f16.gguf", CLIP)
    assert gguf_meta.find_projector(m) == want


# ---- Family defaults -----------------------------------------------------

def test_qwen_hybrid_gets_mtp_drafting(tmp_path):
    d = model_families.family_defaults(gguf_meta.read_gguf(_gguf(tmp_path / "q.gguf", QWEN38)))
    assert d.family == "qwen-hybrid" and d.vision
    assert "--spec-type" in d.flags and "draft-mtp" in d.flags and "-ctkd" in d.flags


def test_gemma_gets_swa_full_and_iq_warning(tmp_path):
    d = model_families.family_defaults(gguf_meta.read_gguf(
        _gguf(tmp_path / "g.gguf", GEMMA, [("a", 9, IQ3_S), ("b", 1, Q4_K)])))
    assert d.family == "gemma" and "--swa-full" in d.flags and "--spec-type" not in d.flags
    assert any("IQ codebook" in w for w in d.warnings)


# ---- Context estimate: anchored to measured loads (2026-10-01) -------------

@pytest.mark.parametrize("file_gb,measured_max", [
    (16.12, 57344),    # Q4_K_S: 57344 fit, the ceiling
    (14.61, 81920),    # OrcaRouter Q3_K_M: 81920 fit, 90112 spilled
    (13.40, 114688),   # bartowski Q3_K_M: 114688 fit (estimate may be lower, never higher)
])
def test_ctx_estimate_never_exceeds_measured_ceiling(tmp_path, file_gb, measured_max):
    i = gguf_meta.read_gguf(_gguf(tmp_path / "q.gguf", QWEN38))
    ctx, why = mi.estimate_ctx(i, int(file_gb * 1e9), 1.0625, 1.0625, 0.5625, [12272, 8151])
    assert ctx is not None and measured_max - 8192 <= ctx <= measured_max, (ctx, why)


def test_ctx_estimate_refuses_a_model_that_cannot_fit(tmp_path):
    i = gguf_meta.read_gguf(_gguf(tmp_path / "q.gguf", QWEN38))
    ctx, why = mi.estimate_ctx(i, int(30e9), 1.0625, 1.0625, 0.5625, [12272, 8151])
    assert ctx is None and "does not fit" in why


# ---- Scan ----------------------------------------------------------------

@pytest.fixture
def setup(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    server = tmp_path / "llama-server.exe"
    server.write_bytes(b"x")
    ready = _gguf(models / "Ready-Q4_K_M.gguf", QWEN38)
    loose = _gguf(models / "Loose-Q3_K_M.gguf", QWEN38)
    _gguf(models / "mmproj-Loose-f16.gguf", CLIP)
    sub = models / "split"
    sub.mkdir()
    shard1 = _gguf(sub / "Big-00001-of-00002.gguf", GEMMA, pad_to=1000)
    (sub / "Big-00002-of-00002.gguf").write_bytes(b"\0" * 500)
    cfg_path = tmp_path / "llama_cpp_config.json"
    cfg_text = (
        '{\n  "_comment": "hand-written, keep me",\n\n'
        f'  "server_path": {json.dumps(str(server))},\n\n'
        '  "models": {\n'
        f'    "ready": {{"path": {json.dumps(ready)}, "num_ctx": 32768, '
        '"extra_flags": ["-sm", "layer", "-ts", "1,2", "-ctk", "q8_0", "-ctv", "q8_0", "-lm", "dio", '
        '"--spec-type", "draft-mtp", "--jinja"]},\n\n'
        f'    "gone": {{"path": {json.dumps(str(models / "Deleted.gguf"))}, "num_ctx": 8192}}\n'
        '  }\n}\n')
    cfg_path.write_text(cfg_text, encoding="utf-8")
    import core.config as config
    monkeypatch.setattr(config, "BASE_DIR", str(tmp_path / "repo-without-models"))
    monkeypatch.setattr(config, "LLAMA_CPP_CONFIG_PATH", str(cfg_path))
    return {"cfg": str(cfg_path), "cfg_text": cfg_text, "ready": ready, "loose": loose,
            "shard": shard1, "models": models}


def test_scan_statuses(setup):
    inv = mi.scan(setup["cfg"], include_other_backends=False)
    by = {(m["id"] or os.path.basename(m["path"])): m for m in inv["models"]}
    assert by["ready"]["status"] == "ready" and by["ready"]["family"] == "qwen-hybrid"
    assert by["gone"]["status"] == "broken" and "model file missing" in by["gone"]["problems"][0]
    assert by["Loose-Q3_K_M.gguf"]["status"] == "unconfigured" and by["Loose-Q3_K_M.gguf"]["vision"]
    big = by["Big-00001-of-00002.gguf"]
    assert big["status"] == "unconfigured" and big["size_bytes"] == 1500   # both shards counted
    assert not any(os.path.basename(m["path"] or "").startswith("Big-00002") for m in inv["models"])  # later shard hidden
    assert len(inv["projectors"]) == 1                                      # mmproj is not a model
    assert inv["counts"] == {"ready": 1, "unconfigured": 2, "broken": 1}


def test_proposal_copies_machine_flags_and_adds_model_flags(setup):
    p = mi.propose_entry(setup["loose"], setup["cfg"], device_totals_mb=[12272, 8151])
    flags = p["entry"]["extra_flags"]
    assert p["template"] == "ready" and p["name"] == "loose-q3-k-m"
    for f in ("-ts", "1,2", "-lm", "dio", "-ctk", "q8_0", "draft-mtp", "--no-mmproj-offload", "--metrics"):
        assert f in flags, f
    assert flags.count("--jinja") == 1 and flags.count("--spec-type") == 1   # no duplicates
    assert p["entry"]["server_path"].endswith("llama-server.exe")
    assert MIN_OK <= p["entry"]["num_ctx"] <= 262144


MIN_OK = 4096


def test_proposal_rejects_projectors(setup):
    with pytest.raises(ValueError):
        mi.propose_entry(str(setup["models"] / "mmproj-Loose-f16.gguf"), setup["cfg"], device_totals_mb=[8000])


# ---- Writing the config --------------------------------------------------

def test_add_entry_inserts_keeps_formatting_and_backs_up(setup):
    p = mi.propose_entry(setup["loose"], setup["cfg"], device_totals_mb=[12272, 8151])
    backup = mi.add_entry(p["name"], p["entry"], setup["cfg"])
    text = open(setup["cfg"], encoding="utf-8").read()
    cfg = json.loads(text)
    assert list(cfg["models"])[0] == "loose-q3-k-m" and "ready" in cfg["models"]
    assert '"_comment": "hand-written, keep me",\n\n' in text      # untouched formatting
    assert open(backup, encoding="utf-8").read() == setup["cfg_text"]
    inv = mi.scan(setup["cfg"], include_other_backends=False)
    assert {mi._norm(m["path"]): m["status"] for m in inv["models"]}[mi._norm(setup["loose"])] == "ready"


@pytest.mark.parametrize("name", ["ready", "../evil", "", "has space"])
def test_add_entry_rejects_bad_or_duplicate_names(setup, name):
    with pytest.raises(ValueError):
        mi.add_entry(name, {"path": setup["loose"]}, setup["cfg"])
    assert open(setup["cfg"], encoding="utf-8").read() == setup["cfg_text"]


# ---- API -----------------------------------------------------------------

def _client(allow_write, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.inventory_api import register_inventory_routes
    monkeypatch.setattr(mi, "_device_totals_mb", lambda n: [12272, 8151])
    import core.model_discovery as md
    monkeypatch.setattr(md, "discover_all", lambda *a, **k: [
        {"id": "llama3:8b", "backend": "ollama", "size": 5, "context_length": 8192, "capabilities": ["text"]}])
    app = FastAPI()
    register_inventory_routes(app, check_auth=lambda r: r.headers.get("authorization") == "Bearer k",
                              allow_write=allow_write)
    return TestClient(app)


H = {"authorization": "Bearer k"}


def test_api_lists_proposes_and_adds(setup, monkeypatch):
    c = _client(True, monkeypatch)
    assert c.get("/v1/inventory").status_code == 401
    inv = c.get("/v1/inventory", headers=H).json()
    assert {"ollama", "llama_cpp"} <= {m["backend"] for m in inv["models"]}
    prop = c.get("/v1/inventory/proposal", params={"path": setup["loose"]}, headers=H)
    assert prop.status_code == 200 and prop.json()["name"] == "loose-q3-k-m"
    r = c.post("/v1/inventory/add", json={"path": setup["loose"], "name": "mine", "num_ctx": 16384}, headers=H)
    assert r.status_code == 200, r.text
    assert json.load(open(setup["cfg"], encoding="utf-8"))["models"]["mine"]["num_ctx"] == 16384


def test_api_refuses_paths_outside_the_inventory_and_writes_without_permission(setup, monkeypatch):
    c = _client(False, monkeypatch)
    assert c.get("/v1/inventory/proposal", params={"path": setup["ready"]}, headers=H).status_code == 404
    assert c.get("/v1/inventory/proposal", params={"path": r"C:\Windows\win.ini"}, headers=H).status_code == 404
    assert c.post("/v1/inventory/add", json={"path": setup["loose"]}, headers=H).status_code in (404, 405)
