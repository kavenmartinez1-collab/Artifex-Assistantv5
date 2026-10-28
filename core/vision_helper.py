"""Always-available vision helper: a small VL model on the CPU.

Two features need to look at a picture at moments when the GPU belongs to
something else:

- the Create tab's AI-assisted prompt, which runs right before Qwen-Image
  takes the card. It used to wake the 27B chat model (a ~25 s cold load,
  ~13 GB of VRAM) only to rewrite the request BLIND: told "make it winter"
  for a cat photo, it wrote a snowy forest with "no animals" in the
  negative prompt. LFM3-VL-2B on the CPU sees the photo and answers in ~7 s.
- the agent's @view_image / @describe_images, which only worked when the
  run's own model had been launched with a projector.

The helper is a second llama-server on its own port, started on first use
through the same LlamaCppEngine that runs the main model (with -ngl 0 the
VRAM gate is skipped: nothing lands on a GPU). It stops again after a quiet
spell so its ~1-2 GB of RAM comes back. Configured by a top-level
"vision_helper" block in llama_cpp_config.json, a model entry of its own
that is NOT listed as a chat model:

    "vision_helper": {
      "path": ".../LFM3-VL-2B-Q4_K_M.gguf",
      "server_path": ".../llama-server.exe",   (a CPU build is enough)
      "num_gpu_layers": 0, "num_ctx": 8192, "port": 8095,
      "idle_unload_s": 600, "assist_tools": ["edit", "image"],
      "extra_flags": ["--mmproj", "...", "--no-mmproj-offload", "-t", "10", "--jinja"]
    }
"""

import json
import logging
import threading
import time
from urllib.request import Request, urlopen

_log = logging.getLogger("artifex.vision_helper")

DEFAULT_PORT = 8095
DEFAULT_IDLE_UNLOAD_S = 600

_lock = threading.Lock()
_engine = None
_last_use = 0.0
_reaper = None


def config() -> dict | None:
    """The vision_helper block, or None when it is absent or incomplete."""
    from core.config import _load_llama_cpp_config
    cfg = _load_llama_cpp_config() or {}
    block = cfg.get("vision_helper")
    if not isinstance(block, dict) or not block.get("path"):
        return None
    return block


def handles_assist(tool: str) -> bool:
    block = config()
    return bool(block) and tool in (block.get("assist_tools") or ["edit", "image"])


def _port(block) -> int:
    return int(block.get("port") or DEFAULT_PORT)


def ensure_url() -> str:
    """Start the helper if it is not running; return its base URL."""
    global _engine, _last_use, _reaper
    block = config()
    if not block:
        raise RuntimeError("no vision_helper configured in llama_cpp_config.json")
    with _lock:
        _last_use = time.monotonic()
        if _engine is None or _engine.model_path != block["path"]:
            from core.engine_llama_cpp import LlamaCppEngine
            if _engine is not None:
                _engine.unload()
            entry = dict(block, port=_port(block))
            entry.setdefault("num_gpu_layers", 0)
            _engine = LlamaCppEngine("vision-helper", entry)
        if not (_engine.is_loaded() and _engine._is_server_healthy()):
            _log.info("Starting vision helper on port %d", _port(block))
            _engine.load()
        if _reaper is None:
            _reaper = threading.Thread(target=_reap_idle, daemon=True,
                                       name="vision-helper-reaper")
            _reaper.start()
        return f"http://127.0.0.1:{_port(block)}"


def _reap_idle():
    while True:
        time.sleep(30)
        block = config() or {}
        idle_s = float(block.get("idle_unload_s") or DEFAULT_IDLE_UNLOAD_S)
        with _lock:
            if (_engine is not None and _engine.is_loaded()
                    and time.monotonic() - _last_use > idle_s):
                _log.info("Vision helper idle for %.0f s; stopping it", idle_s)
                try:
                    _engine.unload()
                except Exception as e:
                    _log.warning("Vision helper unload failed: %s", e)


def chat(messages: list, max_tokens: int = 600, temperature: float = 0.7,
         timeout: float = 300) -> str:
    """One non-streamed reply from the helper, thinking off."""
    url = ensure_url()
    payload = {
        "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = Request(f"{url}/v1/chat/completions",
                  data=json.dumps(payload).encode("utf-8"),
                  headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    global _last_use
    _last_use = time.monotonic()
    return (data["choices"][0]["message"].get("content") or "").strip()
