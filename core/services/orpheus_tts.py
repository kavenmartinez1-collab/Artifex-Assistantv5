"""
Artifex Assistant V5 — Orpheus 3B voices (Holo's "tara"), streamed.

Orpheus is a Llama-3.2-3B that speaks in SNAC codec tokens. It runs under
its own llama-server (default: the 6700 XT over Vulkan, port 8082), and the
tokens are decoded to 24 kHz audio here on the CPU. The protocol is the one
measured against a reference client before it was ported here:

  * prompt = [start_of_human, BOS] + tokenize("<voice>: <text>") + [eot, end_of_human]
    and the BOS is load-bearing: without it the model drops the front of
    the line. llama-server's /tokenize does not add it.
  * the model writes `<custom_token_N>`: N == 2 is end of speech (also the
    stop string, or it says the line again), N < 10 is preamble, else the
    audio code is N - 10 - (position_in_frame * 4096).
  * 7 codes = 1 frame = 2048 samples = 85.3 ms, so realtime is ~82 tok/s;
    the 6700 XT does ~1.25x.

stream() yields int16 PCM as frames arrive (first chunk after 4+2 frames,
~0.45 s), decoding each chunk with 2 frames of right context so the seams
match. The server is started on first use and stopped after IDLE_S, or
before a model that needs the card loads (see LlamaCppEngine.load).

Configured by the "orpheus" block of llama_cpp_config.json (the installer's
holo pack writes it: server_path, model, device, port, min_free_mb), or by
ARTIFEX_ORPHEUS_SERVER / _MODEL / _DEVICE / _PORT / _MIN_FREE_MB, which win.
Unconfigured = Orpheus voices are unavailable and callers fall back to Piper.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time

from core.logging_config import get_logger

_log = get_logger(__name__)

VOICES = ["tara", "leah", "jess", "mia", "zoe", "leo", "dan", "zac"]   # Orpheus stock speakers (5 female, 3 male)
SAMPLE_RATE = 24000
IDLE_S = 600

_BOS, _START, _END = 128000, 128259, [128009, 128260]
_EOS_TEXT, _EOS_N, _FIRST_AUDIO = "<custom_token_2>", 2, 10
_TOKEN_RE = re.compile(r"<custom_token_(\d+)>")
_FRAME = 2048
_RAMP, _CHUNK, _LOOK, _CTX = (4, 6), 8, 2, 4
_GAIN = 2.2      # fixed makeup gain; per-chunk normalizing pumped the hiss

_gen_lock = threading.Lock()      # one utterance at a time on the one slot
_proc_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_last_use = 0.0
_snac = None
_torch = None
_watchdog_started = False


SNAC_REPO = "hubertsiuzdak/snac_24khz"
SNAC_REVISION = "d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec"   # pinned, like the installer's copy


def _setting(key: str, default=None):
    """An environment override, else the config block, else the default."""
    v = os.environ.get("ARTIFEX_ORPHEUS_" + key.upper())
    if v:
        return v
    try:
        from core.config import _load_llama_cpp_config
        block = (_load_llama_cpp_config() or {}).get("orpheus") or {}
    except Exception:
        block = {}
    key = {"server": "server_path"}.get(key, key)
    return block.get(key, default)


def _port() -> int:
    return int(_setting("port", 8082))


def _url() -> str:
    return f"http://127.0.0.1:{_port()}"


def configured() -> bool:
    return bool(_setting("server") and _setting("model"))


def _healthy() -> bool:
    import requests
    try:
        return requests.get(_url() + "/health", timeout=1.5).status_code == 200
    except requests.RequestException:
        return False


_holders: set[str] = set()   # loaded models that may need Orpheus's card
_ALIAS = "artifex-orpheus"     # marks a server this app launched (even a previous run's)
_ready = threading.Event()     # set while the server we own/adopted answers


def gpu_claimed(model_name: str, coexists: bool) -> None:
    """Called as each llama.cpp model loads. A model that may need the
    6700 XT stops Orpheus and keeps it off until that model unloads."""
    with _proc_lock:
        if coexists:
            _holders.discard(model_name)
            return
        _holders.add(model_name)
    stop("loading " + model_name)


def gpu_released(model_name: str) -> None:
    """Called as a llama.cpp model unloads."""
    with _proc_lock:
        _holders.discard(model_name)


def _ours_running() -> bool:
    """A healthy server on the port that this app launched (by its alias)."""
    import requests
    try:
        r = requests.get(_url() + "/v1/models", timeout=1.5)
        return r.ok and any(m.get("id") == _ALIAS for m in r.json().get("data", []))
    except (requests.RequestException, ValueError):
        return False


def _orpheus_card_free_mb() -> float | None:
    """Free VRAM on the AMD card (Orpheus's default device), if known."""
    try:
        from core.gpu_pool import get_pool
        pool = get_pool()
        for d in list(getattr(pool, "_devices", []) or []):
            if getattr(d, "vendor_id", 0) == 0x1002:
                d = pool.refresh_device(d.index) or d
                return float(d.memory_free_mb)
    except Exception:
        return None
    return None


def ensure_running(timeout: float = 60.0) -> None:
    """Start (or adopt) the Orpheus llama-server; raises RuntimeError if it can't."""
    global _proc, _last_use
    _last_use = time.monotonic()
    with _proc_lock:
        if _holders:
            raise RuntimeError(f"Orpheus voices are off while {', '.join(sorted(_holders))} uses the GPUs")
        if _healthy():
            return
        if not configured():
            raise RuntimeError("Orpheus not configured (ARTIFEX_ORPHEUS_SERVER / ARTIFEX_ORPHEUS_MODEL)")
        if _proc is None or _proc.poll() is not None:
            need = float(_setting("min_free_mb", 3000))
            free = _orpheus_card_free_mb()
            if free is not None and free < need:
                raise RuntimeError(f"Orpheus needs ~{need:.0f} MB on the 6700 XT, {free:.0f} MB free")
            from core.config import BASE_DIR
            cmd = [str(_setting("server")), "-m", str(_setting("model")),
                   "--device", str(_setting("device", "Vulkan1")),
                   "-ngl", "99", "-c", "4096", "--host", "127.0.0.1", "--port", str(_port()),
                   "--alias", _ALIAS, "--no-webui"]
            _log.info("starting Orpheus: %s", " ".join(cmd))
            try:
                os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
                with open(os.path.join(BASE_DIR, "logs", "orpheus-server.log"), "ab") as log:
                    _proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except OSError as e:
                _proc = None
                raise RuntimeError(f"Couldn't start Orpheus: {e}")
            _start_watchdog()
        proc = _proc
    # wait outside the lock, so a model load's stop() never queues behind it
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if _holders:
            stop("a model claimed the GPU while Orpheus started")
            raise RuntimeError("Orpheus voices are off while a model uses the GPUs")
        if _healthy():
            _log.info("Orpheus ready in %.1fs", time.monotonic() - t0)
            return
        if proc is None or proc.poll() is not None:
            raise RuntimeError(f"Orpheus server exited ({proc.returncode if proc else '?'}); see logs/orpheus-server.log")
        time.sleep(0.5)
    stop("not healthy in time")
    raise RuntimeError("Orpheus server did not become healthy")


def stop(reason: str = "") -> None:
    """Stop the Orpheus server this app runs: the one it launched, or one a
    previous run left behind (found by its alias). A server someone else
    runs on the port (another app's Orpheus) is left alone."""
    global _proc
    with _proc_lock:
        p, _proc = _proc, None
    if p is not None and p.poll() is None:
        _log.info("stopping Orpheus%s", f" ({reason})" if reason else "")
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    elif _ours_running():
        _log.info("stopping a leftover Orpheus%s", f" ({reason})" if reason else "")
        try:
            from core.proc_tree import stop_port_tree
            stop_port_tree(_port())
        except Exception as e:
            _log.warning("couldn't stop the leftover Orpheus: %s", e)


def _start_watchdog() -> None:
    global _watchdog_started
    if _watchdog_started:
        return
    _watchdog_started = True

    def loop():
        while True:
            time.sleep(30)
            if _proc is not None and time.monotonic() - _last_use > IDLE_S:
                stop("idle")

    threading.Thread(target=loop, name="orpheus-idle", daemon=True).start()
    import atexit
    atexit.register(stop, "API exit")


_decoder_lock = threading.Lock()


def _decoder():
    global _snac, _torch
    with _decoder_lock:                   # two first calls must not load it twice
        if _snac is None:
            import torch
            from snac import SNAC
            _torch = torch
            from core.config import BASE_DIR
            local = os.path.join(BASE_DIR, "models", ".holo", "snac_24khz")   # the holo pack's pinned copy
            if os.path.isfile(os.path.join(local, "pytorch_model.bin")):
                _snac = SNAC.from_pretrained(local)
            else:
                _snac = SNAC.from_pretrained(SNAC_REPO, revision=SNAC_REVISION)
            _snac = _snac.eval().to("cpu")
    return _snac


def _decode(codes):
    import numpy as np
    n = len(codes) // 7
    c0, c1, c2 = [], [], []
    for i in range(n):
        f = codes[i * 7:(i + 1) * 7]
        c0.append(f[0]); c1 += (f[1], f[4]); c2 += (f[2], f[3], f[5], f[6])
    snac, torch = _decoder(), _torch
    layers = [torch.tensor([c], dtype=torch.int32) for c in (c0, c1, c2)]
    with torch.inference_mode():
        wav = snac.decode(layers)
    return wav.squeeze().cpu().numpy().astype(np.float32)


def _pcm(x):
    """Fixed gain + a soft knee above 0.9 instead of hard clipping."""
    import numpy as np
    y = x * _GAIN
    a = np.abs(y)
    knee = a > 0.9
    y[knee] = np.sign(y[knee]) * (0.9 + 0.1 * np.tanh((a[knee] - 0.9) / 0.1))
    return (np.clip(y, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def _emit(codes, emitted, end, avail):
    start = max(0, emitted - _CTX)
    stop_ = min(avail, end + _LOOK)
    mono = _decode(codes[start * 7:stop_ * 7])
    return _pcm(mono[(emitted - start) * _FRAME:(end - start) * _FRAME])


def _clean(text: str) -> str:
    t = re.sub(r"\s+", " ", text.replace("*", "")).strip()
    return t if re.search(r"[A-Za-z0-9]", t) else ""


_stream_seq = 0     # the newest stream; older ones see they are superseded


def _generate(text, voice, my_seq, emit):
    """Run one Orpheus utterance, calling emit(pcm) per chunk. Stops early
    when a newer stream starts (one speaker, one Orpheus slot)."""
    import requests
    global _last_use
    with _gen_lock:
        if my_seq != _stream_seq:
            return
        _last_use = time.monotonic()
        with requests.Session() as sess:
            _generate_with(sess, text, voice, my_seq, emit)
        _last_use = time.monotonic()


def _generate_with(sess, text, voice, my_seq, emit):
    toks = sess.post(_url() + "/tokenize", json={"content": f"{voice}: {text}"}, timeout=10).json()["tokens"]
    body = {"prompt": [_START, _BOS] + toks + _END, "n_predict": 1200, "temperature": 0.6,
            "top_p": 0.9, "repeat_penalty": 1.1, "cache_prompt": True, "stream": True,
            "stop": [_EOS_TEXT]}
    for attempt in (1, 2):
        codes, buf, emitted, chunk_i, sent = [], "", 0, 0, False
        # closing the response mid-way also stops llama-server generating
        with sess.post(_url() + "/completion", json=body, stream=True, timeout=(5, 120)) as r:
            r.raise_for_status()
            r.encoding = "utf-8"
            done = False
            for raw in r.iter_lines(decode_unicode=True):
                if my_seq != _stream_seq:
                    return
                if not raw or not raw.startswith("data: "):
                    continue
                try:
                    buf += json.loads(raw[6:]).get("content", "")
                except ValueError:
                    continue
                used = 0
                for m in _TOKEN_RE.finditer(buf):
                    used = m.end()
                    n = int(m.group(1))
                    if n == _EOS_N:
                        done = True
                        break
                    if n < _FIRST_AUDIO:
                        continue
                    code = n - 10 - (len(codes) % 7) * 4096
                    if 0 <= code < 4096:
                        codes.append(code)
                    else:
                        del codes[len(codes) - len(codes) % 7:]   # drop the broken frame
                buf = buf[used:]
                if done:
                    break
                ready = len(codes) // 7
                need = _RAMP[chunk_i] if chunk_i < len(_RAMP) else _CHUNK
                if ready - emitted >= need + _LOOK:
                    if not emit(_emit(codes, emitted, ready - _LOOK, ready)):
                        return
                    emitted, chunk_i, sent = ready - _LOOK, chunk_i + 1, True
        ready = len(codes) // 7
        if ready >= 6 or sent:           # a real utterance (Orpheus sometimes bails at once)
            if ready > emitted:
                emit(_emit(codes, emitted, ready, ready))
            break
        _log.info("Orpheus gave %d frames; retrying once", ready)


def warm() -> None:
    """Get the server and the SNAC decoder ready without speaking (a warm-up
    line would supersede, i.e. cut off, whatever is being said)."""
    ensure_running()
    _decoder()


def stream(text: str, voice: str = "tara"):
    """Yield int16 little-endian mono PCM @ 24 kHz chunks for one line.

    The generation runs in its own thread and hands chunks over a queue, so
    the Orpheus lock is never held across a yield: when the phone hangs up
    (Stop, a new turn) Starlette simply stops iterating this generator, and a
    lock held there stayed held until garbage collection, so every later
    line waited forever. A newer stream() supersedes this one, and a worker
    whose chunks nobody reads gives up after a few seconds.
    """
    import queue
    global _stream_seq
    text = _clean(text)[:600]
    if not text:
        return
    voice = voice if voice in VOICES else VOICES[0]
    ensure_running()
    _decoder()
    _stream_seq += 1
    my_seq = _stream_seq
    q: "queue.Queue" = queue.Queue(maxsize=16)
    END = object()

    def emit(pcm):
        try:
            q.put(pcm, timeout=8)        # nobody reading for 8 s: the client is gone
            return True
        except queue.Full:
            return False

    def work():
        try:
            _generate(text, voice, my_seq, emit)
        except Exception as e:
            _log.warning("Orpheus stream failed: %s", e)
        finally:
            try:
                q.put(END, timeout=8)
            except queue.Full:
                pass

    threading.Thread(target=work, name="orpheus-gen", daemon=True).start()
    try:
        while True:
            item = q.get()
            if item is END:
                return
            yield item
    finally:
        if my_seq == _stream_seq:      # we were the newest and the caller left: stop generating
            _stream_seq += 1
