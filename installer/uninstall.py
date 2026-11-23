"""Remove what python -m installer put in this folder, after showing the list.

Only things the installer itself made are offered, each found by a rule
that can't match your own files:
- runtime/ (llama.cpp builds, portable Node, downloads) and python/ (the
  portable Python setup.cmd fetched): folders only the installer writes;
- venv/ only when it carries the installer's marker (a venv you built by
  hand is never listed);
- model files in models/gguf/ whose names are pinned in the catalog;
- the launchers the installer writes, by their fixed names;
- installer-state.json and the phone's hashed key file;
- the llama_cpp_config.json entries whose comment says the installer added
  them (the file is backed up first; entries you wrote stay);
- the Tailscale forward to the API port, if one is set.

Nothing outside this folder is removed, a link or junction is never
followed, and nothing happens until you say yes. The Artifex code itself,
pack folders (strata/, games/, Create's models) and Docker are left; the
list says where they are, to delete by hand if you want them gone.
"""

import datetime
import fnmatch
import json
import os
import shutil
import stat

from installer import recipes

LAUNCHERS = ("Artifex CLI.cmd", "Artifex API (this PC only).cmd", "Artifex GUI.cmd",
             "Artifex API (phone via Tailscale).cmd", "A Tale of Yarn.cmd")
LAUNCHER_PATTERNS = ("Artifex CLI (*).cmd",)
LEFT_FOR_YOU = ("strata", "games", os.path.join("models", "qwen-image-2.1-turbo"))


def _is_link(path: str) -> bool:
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISLNK(st.st_mode) or bool(
        getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _inside(root: str, path: str) -> bool:
    root, path = os.path.abspath(root), os.path.abspath(path)
    return path != root and os.path.commonpath([root, path]) == root


def _size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for base, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not _is_link(os.path.join(base, d))]
        for n in files:
            try:
                total += os.lstat(os.path.join(base, n)).st_size
            except OSError:
                pass
    return total


def plan(root: str, catalog: dict, tailscale_forward: bool = False) -> dict:
    root = os.path.abspath(root)
    items = []

    def add(path, what):
        if os.path.lexists(path) and _inside(root, path) and not _is_link(path):
            items.append({"path": path, "what": what, "bytes": _size(path)})

    add(os.path.join(root, "runtime"), "llama.cpp builds, portable Node and downloads")
    add(os.path.join(root, "python"), "the portable Python setup.cmd fetched")
    venv = os.path.join(root, "venv")
    if os.path.isfile(os.path.join(venv, recipes.INSTALLER_MARKER)):
        add(venv, "the Python environment the installer made")
    pinned = {m["file"] for m in catalog.get("chat_models", [])}
    vh = catalog.get("vision_helper") or {}
    pinned |= {vh[k]["file"] for k in ("model", "mmproj") if isinstance(vh.get(k), dict)}
    gguf = os.path.join(root, "models", "gguf")
    if os.path.isdir(gguf):
        for n in sorted(os.listdir(gguf)):
            if n in pinned:
                add(os.path.join(gguf, n), "a pinned model file")
    for n in sorted(os.listdir(root)):
        if n in LAUNCHERS or any(fnmatch.fnmatch(n, p) for p in LAUNCHER_PATTERNS):
            add(os.path.join(root, n), "a launcher the installer wrote")
    add(os.path.join(root, recipes.STATE_FILE), "what the installer installed here")
    add(os.path.join(root, "artifex_api_keys.json"), "the phone's API key hashes")
    entries = []
    cfg_path = os.path.join(root, "llama_cpp_config.json")
    try:
        with open(cfg_path, encoding="utf-8-sig") as f:
            models = (json.load(f) or {}).get("models") or {}
        entries = sorted(mid for mid, e in models.items() if isinstance(e, dict) and str(
            e.get("_comment", "")).startswith("Added by the Artifex installer"))
    except (OSError, ValueError, AttributeError):
        pass
    left = [os.path.join(root, d) for d in LEFT_FOR_YOU if os.path.isdir(os.path.join(root, d))]
    return {"root": root, "items": items, "config_entries": entries, "config_path": cfg_path,
            "tailscale_forward": tailscale_forward, "left": left}


def describe(pl: dict) -> str:
    lines = []
    for it in pl["items"]:
        lines.append(f"  {it['bytes'] / (1 << 30):7.2f} GB  {os.path.relpath(it['path'], pl['root'])}"
                     f"  ({it['what']})")
    if pl["config_entries"]:
        lines.append(f"  config entries the installer added: {', '.join(pl['config_entries'])} "
                     "(llama_cpp_config.json is backed up first)")
    if pl["tailscale_forward"]:
        lines.append("  the Tailscale forward to the Artifex API (tailscale serve ... off)")
    total = sum(it["bytes"] for it in pl["items"])
    lines.append(f"  total: {total / (1 << 30):.2f} GB")
    if pl["left"]:
        lines.append("Left in place (delete by hand if you want them gone): "
                     + ", ".join(os.path.relpath(p, pl["root"]) for p in pl["left"]))
    return "\n".join(lines)


def remove(pl: dict, log=print, serve_off=None) -> list:
    """Delete exactly the planned items. Returns what could not be removed."""
    failed = []
    for it in pl["items"]:
        path = it["path"]
        if not _inside(pl["root"], path) or _is_link(path):     # re-checked at delete time
            failed.append((path, "not inside the folder, or a link"))
            continue
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            log(f"removed {os.path.relpath(path, pl['root'])}")
        except OSError as e:
            failed.append((path, str(e)))
    if pl["config_entries"]:
        path = pl["config_path"]
        try:
            with open(path, encoding="utf-8-sig") as f:
                cfg = json.load(f)
            backup = f"{path}.bak-uninstall-{datetime.datetime.now():%Y%m%d-%H%M%S}"
            with open(backup, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            for mid in pl["config_entries"]:
                cfg.get("models", {}).pop(mid, None)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
                f.write("\n")
            os.replace(tmp, path)
            log(f"removed config entries {', '.join(pl['config_entries'])} (backup: {os.path.basename(backup)})")
        except (OSError, ValueError) as e:
            failed.append((path, str(e)))
    if pl["tailscale_forward"] and serve_off:
        try:
            serve_off()
            log("Tailscale forward to the API turned off")
        except (RuntimeError, ValueError) as e:
            failed.append(("tailscale serve", str(e)))
    return failed
