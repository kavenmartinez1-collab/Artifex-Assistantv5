"""Model inventory: every model Artifex can reach, and whether it is ready.

The model dropdowns only know llama.cpp models that have an entry in
llama_cpp_config.json. A GGUF sitting in a model folder is invisible, and an
entry whose file was deleted shows up as available until it fails. This
module scans:

  - every configured llama.cpp entry (status "ready", or "broken" when its
    model, server binary or projector is missing);
  - every .gguf in the model folders: the folders that configured models live
    in, the repo's models/ folder, and an optional top-level "model_dirs" list
    in llama_cpp_config.json (status "unconfigured" when nothing points at it);
  - Ollama, Claude CLI and Transformers models, via core.model_discovery.

For an unconfigured GGUF, propose_entry() builds a config entry. Machine
flags come from a template entry that already works here; model flags come
from core.model_families; num_ctx comes from an estimate fitted to measured
loads. Nothing is written until add_entry() is called, which keeps a backup
and validates the result.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time

from core.gguf_meta import GGUFInfo, GGUFReadError, find_projector, read_gguf
from core.model_families import family_defaults

# Context estimate for a proposed entry. Two loads measured on a split
# RTX 5060 Ti 8 GB + RX 6700 XT 12 GB (CUDA + ROCm, 8-bit KV, -ub 128,
# 2026-10-01) fix the slope and intercept: VRAM beyond the weights grows by
# about 1.43x the KV-cache bytes per token (compute buffers grow with context
# too), plus about 581 MB fixed. Checked against a third model: it predicts
# 81920 where 81920 was the measured ceiling (90112 spilled). Other hardware
# differs; the engine's VRAM gate still has the final word at load time.
EFFECTIVE_KV_FACTOR = 1.43
FIXED_OVERHEAD_MB = 581
DESKTOP_RESERVE_MB = 1024
SAFETY_MARGIN_MB = 512
CTX_STEP = 8192
MIN_CTX = 4096

KV_BPE = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 1.0625, "q5_1": 0.75,
          "q5_0": 0.6875, "q4_1": 0.625, "q4_0": 0.5625, "iq4_nl": 0.5625}

# Template flags that describe the MACHINE (copied into proposals); the rest
# of a template's flags describe its own model and are not copied.
_HW_VALUE_FLAGS = {"-sm", "--split-mode", "-ts", "--tensor-split", "--device", "-dev",
                   "-ctk", "--cache-type-k", "-ctv", "--cache-type-v", "-ub", "--ubatch-size",
                   "-b", "--batch-size", "-fit", "--fit", "-lm", "--load-mode", "-np", "--parallel",
                   "-fa", "--flash-attn", "-t", "--threads"}
_HW_BOOL_FLAGS = {"--no-mmap", "--mlock"}
_SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)

_meta_cache: dict = {}
_cache_lock = threading.Lock()


def _info(path: str) -> GGUFInfo | None:
    """GGUF metadata, cached by (path, size, mtime) so rescans are cheap."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (os.path.normcase(os.path.abspath(path)), st.st_size, st.st_mtime)
    with _cache_lock:
        if key in _meta_cache:
            return _meta_cache[key]
    try:
        info = read_gguf(path)
    except (GGUFReadError, OSError):
        info = None
    with _cache_lock:
        _meta_cache[key] = info
    return info


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def _load_config(config_path: str | None = None) -> tuple[dict, str]:
    if config_path is None:
        from core.config import LLAMA_CPP_CONFIG_PATH
        config_path = LLAMA_CPP_CONFIG_PATH
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f), config_path
    except (OSError, json.JSONDecodeError):
        return {}, config_path


def _flag_value(flags: list, *names: str):
    for i, f in enumerate(flags):
        if f in names and i + 1 < len(flags):
            return flags[i + 1]
    return None


def model_dirs(cfg: dict) -> list[str]:
    """Folders to scan: configured models' folders, repo models/, and cfg["model_dirs"]."""
    from core.config import BASE_DIR
    dirs = [os.path.join(BASE_DIR, "models")]
    dirs += [os.path.dirname(m.get("path", "")) for m in cfg.get("models", {}).values() if m.get("path")]
    dirs += list(cfg.get("model_dirs") or [])
    out, seen = [], set()
    for d in dirs:
        if d and os.path.isdir(d) and _norm(d) not in seen:
            seen.add(_norm(d))
            out.append(d)
    return out


def _gguf_files(folder: str, depth: int = 2) -> list[str]:
    found = []
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return found
    for e in entries:
        if e.is_file() and e.name.lower().endswith(".gguf"):
            m = _SHARD.search(e.name)
            if m and m.group(1) != "00001":
                continue  # later shards belong to the first one
            found.append(e.path)
        elif e.is_dir() and depth > 1 and not e.name.startswith("."):
            found += _gguf_files(e.path, depth - 1)
    return found


def _file_size(path: str) -> int:
    m = _SHARD.search(path)
    if not m:
        return os.path.getsize(path)
    total, n = 0, int(m.group(2))
    for i in range(1, n + 1):
        try:
            total += os.path.getsize(_SHARD.sub(f"-{i:05d}-of-{n:05d}.gguf", path))
        except OSError:
            pass
    return total


def _describe(info: GGUFInfo | None) -> dict:
    if info is None:
        return {"readable": False}
    fam = family_defaults(info)
    return {
        "readable": True, "architecture": info.architecture, "family": fam.family,
        "native_ctx": info.context_length or None, "mtp": bool(info.nextn_predict_layers),
        "sliding_window": info.sliding_window or None,
        "quant": dict(list(info.tensor_types.items())[:3]), "iq_share": info.iq_share,
        "warnings": fam.warnings,
    }


def scan(config_path: str | None = None, include_other_backends: bool = True) -> dict:
    """Everything reachable, with a status per model. Cheap to call repeatedly."""
    cfg, path = _load_config(config_path)
    default_server = cfg.get("server_path", "")
    items, configured_paths, projectors = [], set(), []

    for name, m in cfg.get("models", {}).items():
        model_path = m.get("path", "")
        flags = m.get("extra_flags") or []
        server = m.get("server_path") or default_server
        mmproj = _flag_value(flags, "--mmproj", "-mm")
        problems = []
        if not (model_path and os.path.isfile(model_path)):
            problems.append(f"model file missing: {model_path or '(no path)'}")
        if not (server and (os.path.isfile(server) or shutil.which(server))):
            problems.append(f"llama-server missing: {server or '(no server_path)'}")
        if mmproj and not os.path.isfile(mmproj):
            problems.append(f"vision projector missing: {mmproj}")
        if model_path:
            configured_paths.add(_norm(model_path))
        info = _info(model_path) if not problems or os.path.isfile(model_path) else None
        items.append({
            "id": name, "backend": "llama_cpp", "status": "broken" if problems else "ready",
            "configured": True, "path": model_path,
            "size_bytes": _file_size(model_path) if os.path.isfile(model_path) else None,
            "num_ctx": m.get("num_ctx"), "vision": bool(mmproj), "problems": problems,
            **_describe(info),
        })

    seen_files = set(configured_paths)
    for folder in model_dirs(cfg):
        for f in _gguf_files(folder):
            if _norm(f) in seen_files:  # configured, or reached from an overlapping folder
                continue
            seen_files.add(_norm(f))
            info = _info(f)
            if info is not None and info.is_projector:
                projectors.append(f)
                continue
            items.append({
                "id": None, "backend": "llama_cpp", "status": "unconfigured", "configured": False,
                "path": f, "size_bytes": _file_size(f), "num_ctx": None,
                "vision": bool(find_projector(f)), "problems": [], **_describe(info),
            })

    if include_other_backends:
        try:
            from core.model_discovery import discover_all
            from core.config import MODELS as hf_dirs
            for m in discover_all():
                if m.get("backend") == "llama_cpp":
                    continue
                if m.get("backend") == "transformers":
                    # The transformers scan lists every folder under models/. Only
                    # folders with a config.json (or a diffusers model_index.json)
                    # are loadable; GGUF folders already appear as llama.cpp items
                    # and the rest (voices, assets) are not models.
                    folder = hf_dirs.get(m["id"], "")
                    if not any(os.path.isfile(os.path.join(folder, f))
                               for f in ("config.json", "model_index.json")):
                        continue
                items.append({
                    "id": m["id"], "backend": m["backend"], "status": "ready", "configured": True,
                    "path": None, "size_bytes": m.get("size") or None,
                    "num_ctx": m.get("context_length") or None,
                    "vision": "vision" in (m.get("capabilities") or []), "problems": [],
                    "readable": None,
                })
        except Exception as e:  # discovery is best-effort; never break the inventory
            items.append({"id": None, "backend": "discovery", "status": "broken", "configured": False,
                          "problems": [f"backend discovery failed: {e}"]})

    counts = {s: sum(1 for i in items if i["status"] == s) for s in ("ready", "unconfigured", "broken")}
    return {"models": items, "projectors": projectors, "dirs": model_dirs(cfg),
            "counts": counts, "config_path": path, "generated_at": time.time()}


def _template(cfg: dict) -> tuple[str | None, dict]:
    """The configured entry whose machine flags a proposal copies.

    cfg["template_entry"] if set; otherwise the first entry whose model and
    server both exist (the first entry is the one the start script defaults to).
    """
    models = cfg.get("models", {})
    name = cfg.get("template_entry")
    if name in models:
        return name, models[name]
    for n, m in models.items():
        server = m.get("server_path") or cfg.get("server_path", "")
        if os.path.isfile(m.get("path", "")) and server and os.path.isfile(server):
            return n, m
    return None, {}


def _hardware_flags(flags: list) -> list:
    out, i = [], 0
    while i < len(flags):
        f = flags[i]
        if f in _HW_VALUE_FLAGS and i + 1 < len(flags):
            out += [f, flags[i + 1]]
            i += 2
            continue
        if f in _HW_BOOL_FLAGS:
            out.append(f)
        i += 1
    return out


def _device_totals_mb(n_devices: int) -> list[float]:
    try:
        from core.gpu_pool import get_pool
        devs = sorted((float(d.get("memory_total_mb") or 0) for d in get_pool().get_device_status()), reverse=True)
    except Exception:
        return []
    return [d for d in devs if d > 0][:max(1, n_devices)]


def estimate_ctx(info: GGUFInfo, weights_bytes: int, bpe_k: float, bpe_v: float,
                 draft_bpe: float | None, device_totals_mb: list[float]) -> tuple[int | None, str]:
    """Largest context the fitted VRAM model allows, rounded down to CTX_STEP."""
    per_token = info.kv_bytes_per_token(bpe_k, bpe_v, draft_bpe)
    if not per_token:
        return None, "no attention metadata in the file; set num_ctx by hand"
    if not device_totals_mb:
        return None, "no GPU information available; set num_ctx by hand"
    budget_mb = (sum(device_totals_mb) - weights_bytes / 2**20 - FIXED_OVERHEAD_MB
                 - DESKTOP_RESERVE_MB - SAFETY_MARGIN_MB)
    tokens = budget_mb * 2**20 / (per_token * EFFECTIVE_KV_FACTOR)
    ctx = int(tokens // CTX_STEP * CTX_STEP)
    if info.context_length:
        ctx = min(ctx, info.context_length)
    if ctx < MIN_CTX:
        return None, (f"the weights leave about {budget_mb:.0f} MB for context on "
                      f"{len(device_totals_mb)} GPU(s); this model likely does not fit")
    return ctx, (f"estimated from {len(device_totals_mb)} GPU(s), {sum(device_totals_mb):.0f} MB total, "
                 f"{per_token / 1024:.1f} KB KV per token")


def _entry_name(path: str, taken: set) -> str:
    stem = os.path.basename(_SHARD.sub(".gguf", path))[:-5].lower()
    base = re.sub(r"[^a-z0-9.]+", "-", stem).strip("-") or "model"
    name, n = base, 2
    while name in taken:
        name, n = f"{base}-{n}", n + 1
    return name


def propose_entry(path: str, config_path: str | None = None,
                  device_totals_mb: list[float] | None = None) -> dict:
    """A config entry for an unconfigured GGUF. Shown to the user; never written here."""
    cfg, _ = _load_config(config_path)
    info = _info(path)
    if info is None:
        raise ValueError(f"not a readable GGUF file: {path}")
    if info.is_projector:
        raise ValueError(f"{os.path.basename(path)} is a vision projector, not a language model")
    fam = family_defaults(info)
    tname, tmpl = _template(cfg)
    hw = _hardware_flags(tmpl.get("extra_flags") or [])
    notes = list(fam.notes)
    if tname:
        notes.insert(0, f"Machine flags copied from the working entry '{tname}': " + " ".join(hw))
    else:
        notes.insert(0, "No working entry to copy machine flags from; using llama.cpp defaults")
    flags = list(hw)
    for f in fam.flags:  # model flags, skipping any the template already set
        if f.startswith("-") and f in flags:
            continue
        flags.append(f)
    projector = find_projector(path) if fam.vision else None
    if projector:
        flags += ["--mmproj", projector, "--no-mmproj-offload"]
        notes.append(f"Vision projector found ({os.path.basename(projector)}); kept on the CPU "
                     "(--no-mmproj-offload) so it costs no VRAM or context")
    if "--metrics" not in flags:
        flags.append("--metrics")

    kv_k = _flag_value(flags, "-ctk", "--cache-type-k") or "f16"
    kv_v = _flag_value(flags, "-ctv", "--cache-type-v") or "f16"
    draft = _flag_value(flags, "-ctkd", "--spec-draft-type-k")
    n_dev = len((_flag_value(flags, "-ts", "--tensor-split") or "x").replace("/", ",").split(","))
    totals = device_totals_mb if device_totals_mb is not None else _device_totals_mb(n_dev)
    weights = _file_size(path)
    ctx, why = estimate_ctx(info, weights, KV_BPE.get(kv_k, 2.0), KV_BPE.get(kv_v, 2.0),
                            KV_BPE.get(draft) if draft else None, totals)
    warnings = list(fam.warnings)
    if ctx is None:
        warnings.append(why)
        ctx = min(info.context_length or 32768, 32768)
        why = "fallback: " + why
    entry = {
        "_comment": (f"Added by the model inventory on {time.strftime('%Y-%m-%d')}: "
                     f"{fam.family} family, {info.architecture}; num_ctx {why}. "
                     "The VRAM gate still checks every load."),
        "path": path.replace("\\", "/"),
        "server_path": (tmpl.get("server_path") or cfg.get("server_path") or "llama-server"),
        "num_ctx": ctx,
        "extra_flags": flags,
    }
    taken = set(cfg.get("models", {}))
    return {"name": _entry_name(path, taken), "entry": entry, "family": fam.family,
            "template": tname, "notes": notes, "warnings": warnings, "ctx_reason": why}


def add_entry(name: str, entry: dict, config_path: str | None = None) -> str:
    """Insert one entry into llama_cpp_config.json. Returns the backup's path.

    The file is hand-formatted (comments, spacing), so the entry is inserted as
    text at the top of "models" instead of re-serialising the whole file. The
    result is parsed before it replaces the original; a backup is kept beside it.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}", name or ""):
        raise ValueError("entry names use letters, digits, '.', '_' and '-' only")
    if not entry.get("path"):
        raise ValueError("an entry needs a model path")
    cfg, path = _load_config(config_path)
    if name in cfg.get("models", {}):
        raise ValueError(f"an entry named '{name}' already exists")
    text = ""
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    backup = f"{path}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    if text:
        shutil.copy2(path, backup)
    anchor = re.search(r'^([ \t]*)"models"\s*:\s*\{[ \t]*\r?\n', text, re.M)
    if anchor and cfg.get("models"):
        indent = anchor.group(1) + "  "
        block = f"{indent}{json.dumps(name)}: {json.dumps(entry, ensure_ascii=False)},\n\n"
        new_text = text[:anchor.end()] + block + text[anchor.end():]
    else:  # no models yet (or an unusual layout): rewrite the JSON
        cfg.setdefault("models", {})[name] = entry
        new_text = json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"
    json.loads(new_text)  # never write a file the app can't read
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(new_text)
    os.replace(tmp, path)
    try:
        from core.model_discovery import invalidate_cache
        invalidate_cache()
    except Exception:
        pass
    return backup if text else ""
