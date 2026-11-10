"""Holo tab server side: avatar manifests, landmarks, uploads, Orpheus GPU claims."""
import io
import json
import math
import os
import threading

import pytest

from core.services import holo_avatars as ha


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(ha, "root", lambda: str(tmp_path))
    monkeypatch.setattr(ha, "_resumed", True)
    monkeypatch.setattr(ha, "_inflight", set())
    monkeypatch.setattr(ha, "_cancelled", set())
    return tmp_path


def _png(w=64, h=80, color=(200, 100, 50)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def _lm(**over):
    lm = {"eyes": [{"c": [20, 30], "dir": [1, 0], "hw": 5, "hh": 2},
                   {"c": [44, 30], "dir": [1, 0], "hw": 5, "hh": 2}],
          "mouth": {"c": [32, 55], "dir": [1, 0], "hw": 8}}
    lm.update(over)
    return lm


# ── landmarks ──────────────────────────────────────────────────

def test_clean_landmarks_rejects_nan_and_infinity():
    bad = _lm(mouth={"c": [32, float("nan")], "dir": [1, 0], "hw": 8})
    assert ha._clean_landmarks(bad, 64, 80) is None
    bad = _lm(mouth={"c": [32, 55], "dir": [1, 0], "hw": float("inf")})
    assert ha._clean_landmarks(bad, 64, 80) is None


def test_clean_landmarks_clamps_x_to_width_and_y_to_height():
    lm = _lm(mouth={"c": [500, 900], "dir": [1, 0], "hw": 8})
    out = ha._clean_landmarks(lm, 64, 80)
    assert out["mouth"]["c"] == [64.0, 80.0]


def test_clean_landmarks_orients_axes_toward_the_mouth():
    # A face lying on its side: mouth to the right of the eyes, axes given
    # pointing "up". The normal n = (-dy, dx) must point from eyes to mouth.
    lm = {"eyes": [{"c": [20, 20], "dir": [0, -1], "hw": 5, "hh": 2},
                   {"c": [20, 50], "dir": [0, -1], "hw": 5, "hh": 2}],
          "mouth": {"c": [55, 35], "dir": [0, -1], "hw": 8}}
    out = ha._clean_landmarks(lm, 64, 80)
    mid = [20, 35]
    for part in out["eyes"] + [out["mouth"]]:
        d = part["dir"]
        n = (-d[1], d[0])
        assert n[0] * (55 - mid[0]) + n[1] * (35 - mid[1]) > 0
        assert math.isclose(math.hypot(*d), 1.0)


def test_clean_landmarks_needs_two_eyes():
    assert ha._clean_landmarks(_lm(eyes=[{"c": [1, 1], "dir": [1, 0], "hw": 3, "hh": 2}]), 64, 80) is None


# ── manifests ──────────────────────────────────────────────────

def _manifest(store, aid="0123456789ab", **over):
    d = store / aid
    d.mkdir()
    (d / "avatar.jpg").write_bytes(b"x")
    m = {"id": aid, "name": "New avatar", "created": 1, "w": 64, "h": 80, "status": "processing",
         "step": "queued", "error": None, "landmarks": None, "landmark_source": None,
         "persona": "", "gender": "neutral", "attempts": 0, "user_edited": []}
    m.update(over)
    (d / "manifest.json").write_text(json.dumps(m))
    return aid


def test_update_marks_user_edits(store):
    aid = _manifest(store)
    m = ha.update(aid, {"name": "Tux", "gender": "male", "landmarks": _lm()})
    assert m["name"] == "Tux" and m["landmark_source"] == "user"
    assert set(m["user_edited"]) == {"name", "gender", "landmarks"}


def test_update_refuses_nan_landmarks(store):
    aid = _manifest(store)
    with pytest.raises(ValueError):
        ha.update(aid, {"landmarks": _lm(mouth={"c": [1, float("nan")], "dir": [1, 0], "hw": 8})})


def test_processing_keeps_edits_made_meanwhile(store, monkeypatch):
    """A PATCH while the thread works must survive the thread's final write."""
    aid = _manifest(store)
    (store / aid / "avatar.jpg").write_bytes(_png())
    from PIL import Image
    monkeypatch.setattr(ha, "_depth", lambda im: Image.new("L", im.size))
    monkeypatch.setattr(ha, "_landmarks_mediapipe", lambda im: (None, None))
    monkeypatch.setattr(ha, "_landmarks_vision", lambda im: (None, None))

    def persona(im, name):
        ha.update(aid, {"name": "Lucy", "persona": "You are Lucy.", "landmarks": _lm()})   # the user, mid-way
        return {"name": "Lumina", "persona": "A detector guess.", "gender": "female"}
    monkeypatch.setattr(ha, "_persona", persona)
    ha._process(aid)
    m = ha.get(aid)
    assert m["status"] == "ready"
    assert (m["name"], m["persona"], m["landmark_source"]) == ("Lucy", "You are Lucy.", "user")
    assert m["gender"] == "female"          # not edited by the user: the detector fills it


def test_delete_during_processing_leaves_no_zombie(store, monkeypatch):
    aid = _manifest(store)
    (store / aid / "avatar.jpg").write_bytes(_png())
    from PIL import Image

    def depth(im):
        ha.delete(aid)                      # the user deletes while depth runs
        return Image.new("L", im.size)
    monkeypatch.setattr(ha, "_depth", depth)
    ha._process(aid)                        # must not raise
    assert not (store / aid).exists()
    assert ha.list_avatars() == []


def test_list_skips_folders_without_a_picture(store):
    aid = _manifest(store)
    os.remove(store / aid / "avatar.jpg")
    assert ha.list_avatars() == []


def test_resume_skips_inflight_and_gives_up_after_max_attempts(store, monkeypatch):
    started = []
    monkeypatch.setattr(ha, "_resumed", False)
    monkeypatch.setattr(threading, "Thread", lambda target, args, **k: type("T", (), {"start": lambda self: started.append(args[0])})())
    a = _manifest(store, "aaaaaaaaaaaa", status="error", attempts=ha.MAX_ATTEMPTS)
    b = _manifest(store, "bbbbbbbbbbbb", status="processing", attempts=1)
    c = _manifest(store, "cccccccccccc", status="processing", attempts=0)
    ha._inflight.add(c)
    assert ha.resume_interrupted() == 1 and started == [b]


# ── uploads ────────────────────────────────────────────────────

def test_create_refuses_oversized_pictures_before_decoding(store, monkeypatch):
    monkeypatch.setattr(ha, "MAX_PIXELS", 100)
    with pytest.raises(ValueError):
        ha.create(_png(20, 20))


def test_create_refuses_non_images_and_odd_formats(store):
    with pytest.raises(ValueError):
        ha.create(b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\n")
    with pytest.raises(ValueError):
        ha.create(b"definitely not an image")


def test_create_stores_and_starts_once(store, monkeypatch):
    calls = []
    monkeypatch.setattr(ha, "_start", lambda aid: calls.append(aid))
    m = ha.create(_png(), "Tux")
    assert m["name"] == "Tux" and calls == [m["id"]]
    assert (store / m["id"] / "avatar.jpg").is_file()


def test_ids_and_file_names_are_whitelisted(store):
    for bad in ("../x", "ABCDEF012345", "0123456789a", "0123456789abc"):
        with pytest.raises(KeyError):
            ha._dir(bad)
    aid = _manifest(store)
    assert ha.file_path(aid, "manifest.json") is None
    assert ha.file_path(aid, "../../secret") is None


# ── Orpheus GPU claims ─────────────────────────────────────────

def test_orpheus_claims_block_until_the_model_unloads(monkeypatch):
    from core.services import orpheus_tts as ot
    stops = []
    monkeypatch.setattr(ot, "_holders", set())
    monkeypatch.setattr(ot, "stop", lambda reason="": stops.append(reason))
    ot.gpu_claimed("vision-helper", True)          # CPU-only / coexisting: no block
    assert not ot._holders and not stops
    ot.gpu_claimed("qwen3.8-27b", False)
    assert ot._holders == {"qwen3.8-27b"} and stops
    with pytest.raises(RuntimeError):
        ot.ensure_running()
    ot.gpu_released("qwen3.8-27b")
    assert not ot._holders


def test_cpu_only_engine_coexists_with_orpheus():
    from core.engine_llama_cpp import LlamaCppEngine
    e = LlamaCppEngine("vision-helper", {"path": "x.gguf", "num_gpu_layers": 0})
    assert e._coexists_with_orpheus
    e = LlamaCppEngine("big", {"path": "x.gguf"})
    assert not e._coexists_with_orpheus


# ── the built-in orb avatar (drawn by a shader in holo.js) ─────────────────────

def _holo_js():
    import re
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "api", "static", "holo", "holo.js"), encoding="utf-8").read()
    return src, re


def test_orb_shader_declares_every_uniform_the_page_sets():
    src, re = _holo_js()
    shader = re.search(r"const FS_ORB = `(.*?)`;", src, re.S).group(1)
    wanted = re.search(r'for \(const n of \[([^\]]+)\]\) H\.uo\[n\]', src).group(1)
    names = re.findall(r'"(\w+)"', wanted)
    assert len(names) >= 8
    declared = set(re.findall(r"\bu[A-Z]\w*", " ".join(re.findall(r"uniform [^;]+;", shader))))
    assert set(names) <= declared, set(names) - declared
    assert "${" not in shader            # a template literal: nothing may interpolate


def test_both_builtin_avatars_exist_and_one_is_first():
    src, re = _holo_js()
    assert 'id: "orb", kind: "orb"' in src and 'id: "default"' in src
    # a build without a shipped portrait must not point at picture files it doesn't have
    if "const HAS_PORTRAIT = false;" in src:
        holo_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "api", "static", "holo")
        assert not os.path.exists(os.path.join(holo_dir, "avatar.jpg"))
        assert "if (!HAS_PORTRAIT) return [orbAvatar()];" in src
    first = re.search(r'const BUILTIN_FIRST = "(\w+)";', src).group(1)
    assert first in ("orb", "default")
    # the picture avatar still loads its two textures; the orb needs none
    assert 'a.kind === "orb" ? { orb: true }' in src
