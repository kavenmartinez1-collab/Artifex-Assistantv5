"""
Artifex Assistant V5 — Holo avatars: any uploaded picture -> a talking hologram.

An upload is processed in the background, all on the CPU so the chat brain
keeps its GPU:

  1. normalize    EXIF-rotate, RGB, longest side <= 1400 px -> avatar.jpg
  2. depth        Depth-Anything-V2-Small, then grown + blurred so a big
                  parallax shift stretches instead of tearing -> depth.png
  3. face         MediaPipe Face Landmarker (photos / realistic faces), else
                  the LFM3-VL vision helper's boxes in 0-1000 coordinates
                  (measured on the anime default avatar: mouth and one eye
                  within ~10/1000 of hand-marked truth, the other eye ~35),
                  else a centred guess. The phone shows draggable dots to fix
                  any of these, so a miss costs a drag, not a re-upload.
  4. persona      the vision helper names the character, writes a one-line
                  description and says which voice set fits (female / male)
  5. manifest     output/holo_avatars/<id>/manifest.json

Landmarks are in image pixels, y down, in the shape holo.js uses:
  eyes:  [{c:[x,y], dir:[dx,dy], hw, hh}, ...]   (eye centre, eye axis, half sizes)
  mouth: {c:[x,y], dir:[dx,dy], hw}               (lip line centre, axis, half width)
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import shutil
import threading
import time
import uuid

from core.logging_config import get_logger

_log = get_logger(__name__)

MAX_SIDE = 1400
_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
_DEPTH_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"   # pinned, like the installer's copy
_ID_RE = re.compile(r"^[a-f0-9]{12}$")
_FILES = {"avatar.jpg": "image/jpeg", "depth.png": "image/png"}

_lock = threading.Lock()          # manifest read-modify-writes and deletes
_work = threading.Lock()          # one avatar processes at a time
_depth_pipe = None
_inflight: set[str] = set()       # avatars a thread is (about to be) processing
_cancelled: set[str] = set()      # deleted while processing: the thread stops writing
MAX_ATTEMPTS = 3                  # an upload that keeps failing stops being retried
MAX_PIXELS = 40_000_000           # ~40 MP; bigger images are refused before decoding
_FORMATS = ["JPEG", "PNG", "WEBP", "GIF", "BMP", "MPO", "TIFF"]   # no EPS & co.


class Cancelled(Exception):
    """The avatar was deleted while it was being processed."""


def root() -> str:
    from core.config import BASE_DIR
    d = os.path.join(BASE_DIR, "output", "holo_avatars")
    os.makedirs(d, exist_ok=True)
    return d


def _dir(avatar_id: str) -> str:
    if not _ID_RE.match(avatar_id or ""):
        raise KeyError(avatar_id)
    return os.path.join(root(), avatar_id)


def file_path(avatar_id: str, name: str) -> str | None:
    if name not in _FILES:
        return None
    p = os.path.join(_dir(avatar_id), name)
    return p if os.path.isfile(p) else None


def media_type(name: str) -> str:
    return _FILES[name]


def _read(avatar_id: str) -> dict:
    with open(os.path.join(_dir(avatar_id), "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def _write_locked(m: dict) -> None:
    p = os.path.join(_dir(m["id"]), "manifest.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1, allow_nan=False)
    os.replace(tmp, p)


def _write(m: dict) -> None:
    with _lock:
        _write_locked(m)


def _patch(avatar_id: str, fn) -> dict:
    """Read-modify-write one manifest under the lock, so a PATCH from the
    phone and the processing thread never overwrite each other."""
    with _lock:
        if avatar_id in _cancelled:
            raise Cancelled(avatar_id)
        try:
            m = _read(avatar_id)
        except (OSError, ValueError):
            raise KeyError(avatar_id)
        fn(m)
        _write_locked(m)
        return m


def get(avatar_id: str) -> dict:
    try:
        return _read(avatar_id)
    except (OSError, ValueError):
        raise KeyError(avatar_id)


def list_avatars() -> list[dict]:
    resume_interrupted()
    out = []
    for name in sorted(os.listdir(root())):
        # a folder without its picture is a half-deleted leftover
        if _ID_RE.match(name) and os.path.isfile(os.path.join(root(), name, "avatar.jpg")):
            try:
                out.append(_read(name))
            except (OSError, ValueError):
                continue
    return sorted(out, key=lambda m: m.get("created", 0))


def delete(avatar_id: str) -> None:
    d = _dir(avatar_id)
    with _lock:
        if not os.path.isdir(d):
            raise KeyError(avatar_id)
        _cancelled.add(avatar_id)          # a processing thread stops at its next write
        shutil.rmtree(d, ignore_errors=True)
    if os.path.isdir(d):                   # Windows: a file was still open; once more
        time.sleep(0.5)
        shutil.rmtree(d, ignore_errors=True)


def update(avatar_id: str, fields: dict) -> dict:
    cur = get(avatar_id)
    lm = None
    if "landmarks" in fields:
        lm = _clean_landmarks(fields["landmarks"], cur["w"], cur["h"])
        if lm is None:
            raise ValueError("landmarks need eyes[2] {c,dir,hw,hh} and mouth {c,dir,hw}, finite numbers")

    def apply(m):
        edited = set(m.get("user_edited") or [])
        if "name" in fields and isinstance(fields["name"], str):
            m["name"] = fields["name"].strip()[:40] or m["name"]
            edited.add("name")
        if "persona" in fields and isinstance(fields["persona"], str):
            m["persona"] = fields["persona"].strip()[:2500]
            edited.add("persona")
        if fields.get("gender") in ("female", "male", "neutral"):
            m["gender"] = fields["gender"]
            edited.add("gender")
        if lm is not None:
            m["landmarks"], m["landmark_source"] = lm, "user"
            edited.add("landmarks")
        m["user_edited"] = sorted(edited)    # processing never overwrites these

    try:
        return _patch(avatar_id, apply)
    except Cancelled:
        raise KeyError(avatar_id)


def _num(v) -> float:
    f = float(v)
    if not math.isfinite(f):
        raise ValueError("not a finite number")
    return f


def _clean_landmarks(lm, w, h):
    """Validate landmarks (from the phone or a detector); None if malformed.

    Every number must be finite (NaN in a manifest broke the avatar list),
    points are clamped to the picture, and the axes are oriented by the
    face: the shader opens the mouth along n = (-dir.y, dir.x), so every
    axis is turned to make n point from the eyes toward the mouth - a
    sideways face still blinks and talks the right way round.
    """
    def pt(v):
        return [min(max(_num(v[0]), 0.0), float(w)), min(max(_num(v[1]), 0.0), float(h))]
    try:
        raw = lm["eyes"][:2]
        if len(raw) != 2:
            return None
        eyes = [{"c": pt(e["c"]), "dir": _unit(e["dir"]),
                 "hw": min(max(_num(e["hw"]), 2.0), w / 3), "hh": min(max(_num(e["hh"]), 1.0), h / 4)}
                for e in raw]
        mo = lm["mouth"]
        mouth = {"c": pt(mo["c"]), "dir": _unit(mo["dir"]), "hw": min(max(_num(mo["hw"]), 3.0), w / 3)}
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
        return None
    mid = [(eyes[0]["c"][0] + eyes[1]["c"][0]) / 2, (eyes[0]["c"][1] + eyes[1]["c"][1]) / 2]
    down = [mouth["c"][0] - mid[0], mouth["c"][1] - mid[1]]
    for part in eyes + [mouth]:
        d = part["dir"]
        if -d[1] * down[0] + d[0] * down[1] < 0:      # n·down < 0: flip
            part["dir"] = [-d[0], -d[1]]
    return {"eyes": eyes, "mouth": mouth}


def _unit(v):
    x, y = _num(v[0]), _num(v[1])
    n = math.hypot(x, y)
    if n < 1e-6:
        return [1.0, 0.0]
    return [x / n, y / n]


# ── create + background processing ─────────────────────────────

def create(image_bytes: bytes, name: str = "") -> dict:
    """Store a picture as a new avatar and start processing it. Raises
    ValueError for anything that isn't a reasonable photo or drawing."""
    from PIL import Image, ImageOps
    try:
        im = Image.open(io.BytesIO(image_bytes), formats=_FORMATS)
        if im.width * im.height > MAX_PIXELS:          # checked before any decoding
            raise ValueError(f"image too large ({im.width}x{im.height}); 40 MP max")
        if im.format == "JPEG":
            im.draft("RGB", (MAX_SIDE * 2, MAX_SIDE * 2))   # decode big JPEGs at a reduced size
        im = ImageOps.exif_transpose(im).convert("RGB")
    except ValueError:
        raise
    except Exception as e:      # decompression bombs, odd EXIF, unknown formats
        raise ValueError(f"not a usable image ({type(e).__name__})")
    s = MAX_SIDE / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    aid = uuid.uuid4().hex[:12]
    os.makedirs(_dir(aid))
    im.save(os.path.join(_dir(aid), "avatar.jpg"), quality=88, optimize=True, progressive=True)
    m = {"id": aid, "name": (name or "").strip()[:40] or "New avatar", "created": time.time(),
         "w": im.width, "h": im.height, "status": "processing", "step": "queued", "error": None,
         "landmarks": None, "landmark_source": None, "persona": "", "gender": "neutral",
         "attempts": 0, "user_edited": []}
    _write(m)
    _start(aid)
    return m


def _start(aid: str) -> None:
    with _lock:
        if aid in _inflight:
            return
        _inflight.add(aid)
    threading.Thread(target=_process, args=(aid,), name=f"holo-avatar-{aid}", daemon=True).start()


def _step(aid: str, step: str) -> None:
    def f(m):
        m["step"] = step
    _patch(aid, f)
    _log.info("holo avatar %s: %s", aid, step)


def _process(aid: str) -> None:
    global _depth_pipe
    try:
        with _work:
            try:
                def begin(m):
                    m["status"], m["step"], m["error"] = "processing", "queued", None
                    m["attempts"] = int(m.get("attempts") or 0) + 1
                _patch(aid, begin)
                from PIL import Image
                im = Image.open(os.path.join(_dir(aid), "avatar.jpg")).convert("RGB")
                _step(aid, "depth")
                depth = _depth(im)
                tmp = os.path.join(_dir(aid), "depth.png.tmp")
                depth.save(tmp, format="PNG", optimize=True)
                os.replace(tmp, os.path.join(_dir(aid), "depth.png"))   # never served half-written
                _step(aid, "face")
                lm, src = _landmarks_mediapipe(im)
                if lm is None:
                    lm, src = _landmarks_vision(im)
                if lm is None:
                    lm, src = _landmarks_guess(im), "guess"
                _step(aid, "persona")
                cur = get(aid)
                found = _persona(im, cur.get("name", ""))

                def finish(m):
                    edited = set(m.get("user_edited") or [])
                    if "landmarks" not in edited:         # the phone's dots win
                        m["landmarks"], m["landmark_source"] = lm, src
                    for k, v in found.items():
                        if k not in edited:
                            m[k] = v
                    m["status"], m["step"] = "ready", "done"
                _patch(aid, finish)
            except (Cancelled, KeyError):
                _log.info("holo avatar %s: deleted while processing", aid)
            except Exception as e:      # the phone shows the reason; the upload stays
                _log.exception("holo avatar %s failed", aid)
                msg = f"{type(e).__name__}: {e}"[:300]

                def failed(m):
                    m["status"], m["error"] = "error", msg
                try:
                    _patch(aid, failed)
                except (Cancelled, KeyError):
                    pass
    finally:
        with _lock:
            _inflight.discard(aid)
            idle = not _inflight
        if idle:
            _depth_pipe = None      # ~0.5 GB of CPU model back between uploads


def _depth(im):
    """Depth-Anything-V2-Small on the CPU, normalized, foreground grown + softened."""
    global _depth_pipe
    import numpy as np
    from PIL import Image, ImageFilter
    import torch
    if _depth_pipe is None:
        from transformers import pipeline
        from core.config import BASE_DIR
        local = os.path.join(BASE_DIR, "models", ".holo", "depth-anything-v2-small")   # the holo pack's copy
        if os.path.isfile(os.path.join(local, "model.safetensors")):
            _depth_pipe = pipeline("depth-estimation", model=local, device="cpu")
        else:
            _depth_pipe = pipeline("depth-estimation", model=_DEPTH_MODEL, revision=_DEPTH_REVISION, device="cpu")
    # a few cores only: chat, the vision helper and CPU MoE layers share them
    threads = torch.get_num_threads()
    torch.set_num_threads(min(4, threads))
    try:
        d = np.array(_depth_pipe(im)["depth"].resize(im.size, Image.BICUBIC)).astype(np.float32)
    finally:
        torch.set_num_threads(threads)
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    out = Image.fromarray((d * 255).astype(np.uint8))
    return out.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(3))


def _landmarks_mediapipe(im):
    from core.config import BASE_DIR
    task = os.path.join(BASE_DIR, "models", ".holo", "face_landmarker.task")   # dot dir: not a "model"
    if not os.path.isfile(task):
        return None, None
    try:
        import numpy as np
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
    except ImportError:
        return None, None
    try:
        # model_asset_buffer: a path with non-ASCII characters fails on Windows
        with open(task, "rb") as f:
            det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=mpt.BaseOptions(model_asset_buffer=f.read()), num_faces=1))
    except Exception as e:
        _log.warning("MediaPipe unavailable: %s", e)
        return None, None
    try:
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(im)))
    except Exception as e:
        _log.warning("MediaPipe failed: %s", e)
        return None, None
    finally:
        det.close()
    if not res.face_landmarks:
        return None, None
    L, W, H = res.face_landmarks[0], im.width, im.height
    P = lambda i: (L[i].x * W, L[i].y * H)

    def eye(a, b, up, lo):
        (x1, y1), (x2, y2), (ux, uy), (lx, ly) = P(a), P(b), P(up), P(lo)
        hw = math.hypot(x2 - x1, y2 - y1) / 2 * 1.15
        return {"c": [(x1 + x2) / 2, (y1 + y2) / 2], "dir": _unit([x2 - x1, y2 - y1]),
                "hw": hw, "hh": max(math.hypot(ux - lx, uy - ly) * 0.8, hw * 0.35)}

    eyes = sorted([eye(33, 133, 159, 145), eye(362, 263, 386, 374)], key=lambda e: e["c"][0])
    (lx, ly), (rx, ry), (ux, uy), (dx, dy) = P(61), P(291), P(13), P(14)
    mouth = {"c": [(ux + dx) / 2, (uy + dy) / 2], "dir": _unit([rx - lx, ry - ly]),
             "hw": math.hypot(rx - lx, ry - ly) / 2}
    lm = _clean_landmarks({"eyes": eyes, "mouth": mouth}, W, H)
    return (lm, "mediapipe") if lm else (None, None)


def _vision_chat(im, prompt, max_tokens=200, temperature=0.0):
    from core import vision_helper
    if not vision_helper.config():
        return None
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": prompt}]}]
    # Optional steps: a helper that was killed mid-request (an API restart
    # stops every llama-server) gets one fresh try, then the step is skipped.
    for attempt in (1, 2):
        try:
            return vision_helper.chat(msgs, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            _log.warning("vision helper failed (%s, try %d): %s", prompt[:30], attempt, e)
            time.sleep(2)
    return None


def _json_in(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except ValueError:
        return None


def _landmarks_vision(im):
    """LFM3-VL boxes. Normalized 0-1000 is what it localizes well (pixel
    coordinates came back ~0.8x squashed in y)."""
    out = _vision_chat(im, (
        "Locate the face parts. Reply ONLY with JSON of bounding boxes in normalized 0-1000 "
        'coordinates [x1,y1,x2,y2]: {"eye_left":[...],"eye_right":[...],"mouth":[...]} '
        "where eye_left is the eye nearer the left edge of the image."))
    j = _json_in(out)
    try:
        W, H = im.width, im.height
        box = lambda k: [j[k][0] / 1000 * W, j[k][1] / 1000 * H, j[k][2] / 1000 * W, j[k][3] / 1000 * H]
        el, er, mo = box("eye_left"), box("eye_right"), box("mouth")
    except (TypeError, KeyError, IndexError, ValueError, ZeroDivisionError):
        return None, None
    cen = lambda b: [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
    cl, cr = cen(el), cen(er)
    roll = _unit([cr[0] - cl[0], cr[1] - cl[1]])
    eyes = [{"c": cen(b), "dir": roll, "hw": max((b[2] - b[0]) / 2, 3), "hh": max((b[3] - b[1]) / 2, 2)}
            for b in (el, er)]
    mouth = {"c": cen(mo), "dir": roll, "hw": max((mo[2] - mo[0]) / 2, 4)}
    if not (0 < cl[0] < cr[0] < W):
        return None, None
    lm = _clean_landmarks({"eyes": eyes, "mouth": mouth}, W, H)
    return (lm, "vision") if lm else (None, None)


def _landmarks_guess(im):
    W, H = im.width, im.height
    e = W * 0.07
    return _clean_landmarks({"eyes": [{"c": [W * 0.40, H * 0.38], "dir": [1, 0], "hw": e, "hh": e * 0.45},
                     {"c": [W * 0.60, H * 0.38], "dir": [1, 0], "hw": e, "hh": e * 0.45}],
            "mouth": {"c": [W * 0.50, H * 0.50], "dir": [1, 0], "hw": W * 0.06}}, W, H)


def _persona(im, name):
    out = _vision_chat(im, (
        "You are naming a character for a talking hologram. Reply ONLY with JSON: "
        '{"name": a short first name that fits, "gender": "female" | "male" | "neutral", '
        '"persona": one or two sentences, second person, describing who they are and how they talk}.'),
        max_tokens=220, temperature=0.4)
    j = _json_in(out) or {}
    res = {}
    if isinstance(j.get("name"), str) and j["name"].strip() and name in ("", "New avatar"):
        res["name"] = j["name"].strip()[:40]
    if j.get("gender") in ("female", "male", "neutral"):
        res["gender"] = j["gender"]
    if isinstance(j.get("persona"), str):
        res["persona"] = j["persona"].strip()[:2500]
    return res


_resumed = False


def resume_interrupted() -> int:
    """Re-run avatars a restart cut off (status processing, or an error).

    The work is a background thread of the API, so stopping the API mid-way
    leaves the manifest at "processing" forever. Called once per server
    start (first list); avatars already being processed are skipped, and
    one that failed MAX_ATTEMPTS times is left as an error.
    """
    global _resumed
    if _resumed:
        return 0
    _resumed = True
    n = 0
    for name in sorted(os.listdir(root())):
        if not _ID_RE.match(name) or not os.path.isfile(os.path.join(root(), name, "avatar.jpg")):
            continue
        try:
            m = _read(name)
        except (OSError, ValueError):
            continue
        if name in _inflight:
            continue
        if m.get("status") in ("processing", "error") and int(m.get("attempts") or 0) < MAX_ATTEMPTS:
            _start(name)
            n += 1
    if n:
        _log.info("holo avatars: resumed %d interrupted upload(s)", n)
    return n
