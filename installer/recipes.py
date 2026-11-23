"""Install recipes. Each one is fixed and pinned: the probe picks WHICH
recipe and which pinned files; nothing here is improvised at install time.

Core chat = a prebuilt llama.cpp server for this PC's GPU + a chat model that
fits it + an entry in llama_cpp_config.json. No compiling, no admin rights,
no PATH changes, no services: everything lands under the install root.
"""

import datetime
import json
import os
import re
import subprocess

from installer import fetch, hint

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(HERE, "catalog.json")
_HF = "https://huggingface.co/{repo}/resolve/{revision}/{file}"


def load_catalog(path: str = CATALOG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _version(s: str) -> tuple:
    try:
        return tuple(int(x) for x in s.split(".")[:2])
    except ValueError:
        return (0, 0)


def _variant_ids(catalog) -> dict:
    """The catalog's build ids by kind, so a llama.cpp bump that renames a
    build (cuda-13.4 -> cuda-13.5) needs no code change."""
    ids = list(((catalog or {}).get("llama_cpp") or {}).get("variants") or
               ["cuda-13.4", "cuda-12.4", "vulkan", "cpu"])
    pick = lambda pre, dflt: next((v for v in ids if v.startswith(pre)), dflt)
    return {"cuda13": pick("cuda-13", "cuda-13.4"), "cuda12": pick("cuda-12", "cuda-12.4"),
            "vulkan": "vulkan", "cpu": "cpu"}


def _nvidia_variant(p: dict, g: dict, ids: dict) -> tuple:
    cuda = _version((p.get("nvidia") or {}).get("cuda_version", ""))
    cc = _version(g.get("compute_cap", "0.0"))
    if cuda >= (13, 0):
        return ids["cuda13"], f"NVIDIA driver supports CUDA {cuda[0]}.{cuda[1]}"
    # The CUDA 12.4 build has no Blackwell (RTX 50, compute 12.x) kernels.
    if cuda >= (12, 4) and cc < (12, 0):
        return ids["cuda12"], f"NVIDIA driver supports CUDA {cuda[0]}.{cuda[1]}"
    return "vulkan", ("NVIDIA driver too old for the CUDA builds; using Vulkan "
                      "(updating the driver would allow CUDA)")


def choose_variant(p: dict, catalog: dict = None) -> tuple:
    """(variant id, gpu dict or None, reason).

    NVIDIA runs on CUDA, everything else on Vulkan. With cards of both kinds
    the NVIDIA one wins (CUDA is faster), unless another card is big enough
    to hold a better model: a 12 GB AMD card next to an 8 GB NVIDIA one used
    to be ignored, even when it would have fit the next model up.
    """
    ids = _variant_ids(catalog)
    gpus = [g for g in p.get("gpus") or [] if g.get("vram_mb", 0) >= 2048]
    nvidia = max((g for g in gpus if g.get("vendor") == "NVIDIA"),
                 key=lambda g: g["vram_mb"], default=None)
    other = max((g for g in gpus if g.get("vendor") != "NVIDIA"),
                key=lambda g: g["vram_mb"], default=None)
    if nvidia and other and catalog and \
            _tier(catalog, other["vram_mb"], p) > _tier(catalog, nvidia["vram_mb"], p):
        return "vulkan", other, (f"{other['vendor']} GPU: its {other['vram_mb'] // 1024} GB "
                                 f"fit a bigger model than the {nvidia['name']}")
    if nvidia:
        v, why = _nvidia_variant(p, nvidia, ids)
        return v, nvidia, why
    if other:
        return "vulkan", other, f"{other['vendor']} GPU"
    return "cpu", None, "no GPU with 2 GB or more of video memory"


def _fits_ram(m: dict, p: dict, on_cpu: bool) -> bool:
    """Enough system RAM for the model: its own minimum if the catalog sets
    one, and on the CPU the whole file plus room for Windows and the app."""
    ram = (p or {}).get("ram_mb") or 0
    if not ram:
        return True                       # unknown: don't guess it away
    if ram < m.get("min_ram_mb", 0):
        return False
    return not on_cpu or ram >= (m["size"] >> 20) * 13 // 10 + 4096


def _auto_models(catalog: dict, vram: int, p: dict = None, on_cpu: bool = False) -> list:
    # "auto": false models (abliterated ones, ones that need a profile's
    # RAM-offload flags) are only installed when asked for by id or profile.
    return [m for m in catalog["chat_models"]
            if m.get("auto", True) and m["min_vram_mb"] <= vram and _fits_ram(m, p, on_cpu)]


def _tier(catalog: dict, vram: int, p: dict = None) -> int:
    return max((m["min_vram_mb"] for m in _auto_models(catalog, vram, p)), default=-1)


def choose_model(catalog: dict, variant: str, gpu, p: dict = None) -> dict:
    on_cpu = variant == "cpu" or not gpu
    vram = 0 if on_cpu else gpu["vram_mb"]
    fits = _auto_models(catalog, vram, p, on_cpu)
    if not fits:                              # a very small PC: the smallest model, with a warning
        fits = [min((m for m in catalog["chat_models"] if m.get("auto", True)),
                    key=lambda m: m["size"])]
    return max(fits, key=lambda m: (m["min_vram_mb"], m["size"] if on_cpu else 0))


def _catalog_model(catalog: dict, model_id: str) -> dict:
    model = next((m for m in catalog["chat_models"] if m["id"] == model_id), None)
    if not model:
        raise ValueError(f"unknown model {model_id!r}; options: "
                         + ", ".join(m["id"] for m in catalog["chat_models"]))
    return model


def profile_warnings(p: dict, prof: dict) -> list:
    """What about this PC falls short of the profile it is installing.
    Warnings, not refusals: the profile's flags may still work, slower."""
    out = []
    vram = max((g["vram_mb"] for g in p.get("gpus") or [] if g.get("vendor") == "NVIDIA"),
               default=0)
    if vram < prof.get("min_vram_mb", 0):
        out.append(f"this profile is tuned for an NVIDIA card with {prof['min_vram_mb'] // 1000} GB+; "
                   f"this PC has {vram / 1024:.1f} GB")
    if p.get("ram_mb", 0) and p["ram_mb"] < prof.get("min_ram_mb", 0):
        out.append(f"this profile needs about {prof['min_ram_mb'] // 1000} GB of RAM; "
                   f"this PC has {p['ram_mb'] / 1024:.1f} GB")
    return out


def _long_paths_enabled() -> bool:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\FileSystem") as k:
            return winreg.QueryValueEx(k, "LongPathsEnabled")[0] == 1
    except (ImportError, OSError):
        return os.name != "nt"


def path_warnings(root: str, models_dir: str = "", environ=None, long_paths=None) -> list:
    """Install-folder trouble that is common on a fresh Windows PC, said
    before the download rather than found after it."""
    environ = os.environ if environ is None else environ
    root = os.path.abspath(root)
    out = []
    synced = [environ.get(k) for k in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")]
    for where, path in (("Artifex", root), ("the models", models_dir)):
        if not path:
            continue
        p = os.path.normcase(os.path.abspath(path))
        if any(s and (p + os.sep).startswith(os.path.normcase(os.path.abspath(s)) + os.sep)
               for s in synced):
            out.append(f"{where} would go inside your OneDrive folder ({path}): OneDrive would try "
                       "to upload many GB and can lock files mid-install. Better: move the Artifex "
                       "folder to e.g. C:\\Artifex (or use --models-dir on another drive).")
    parent = os.path.dirname(root)
    if os.path.basename(parent) and os.path.basename(parent).lower() == os.path.basename(root).lower():
        out.append(f"the folder is nested in one of the same name ({os.path.basename(root)}\\"
                   f"{os.path.basename(root)}): Windows' Extract All does that. It works, but "
                   "moving the inner folder up keeps paths short.")
    long_ok = _long_paths_enabled() if long_paths is None else long_paths
    if not long_ok and len(root) > 90:
        out.append(f"the folder path is long ({len(root)} characters) and Windows long paths are "
                   "off: some packs (Node packages, the game) nest deep and can hit the 260-character "
                   "limit. A short folder such as C:\\Artifex avoids it.")
    return out


def plan(p: dict, catalog: dict, root: str, model_id: str = None,
         variant: str = None, profile_id: str = None, models_dir: str = "") -> dict:
    v, gpu, why = choose_variant(p, catalog)
    if variant:
        if variant not in catalog["llama_cpp"]["variants"]:
            raise ValueError(f"unknown backend variant {variant!r}")
        v, why = variant, "chosen by you"
    root = os.path.abspath(root)
    models_dir = os.path.abspath(models_dir) if models_dir else os.path.join(root, "models", "gguf")
    warnings = []
    if profile_id:
        if model_id:
            raise ValueError("pick a profile or a model, not both")
        prof = (catalog.get("profiles") or {}).get(profile_id)
        if not prof:
            raise ValueError(f"unknown profile {profile_id!r}; options: "
                             + ", ".join(catalog.get("profiles") or {}))
        # A profile entry may override the catalog's num_ctx / extra_flags
        # with ones measured for that hardware; everything pinned (repo,
        # revision, size, sha256) still comes from the catalog entry.
        models = [dict(_catalog_model(catalog, e["id"]),
                       **{k: e[k] for k in ("num_ctx", "extra_flags") if k in e})
                  for e in prof["models"]]
        warnings = profile_warnings(p, prof)
    elif model_id:
        ids = [model_id] if isinstance(model_id, str) else list(dict.fromkeys(model_id))
        models = [_catalog_model(catalog, i) for i in ids]
        have = gpu["vram_mb"] if (gpu and v != "cpu") else 0
        for m in models:
            if m["min_vram_mb"] > have:
                warnings.append(f"{m['label']} is meant for about {m['min_vram_mb'] // 1000} GB of "
                                f"video memory; {'this card has ' + format(have / 1024, '.1f') + ' GB' if have else 'this install runs on the CPU'}"
                                ", so it may not load or will run slowly")
    else:
        models = [choose_model(catalog, v, gpu, p)]
        if not _fits_ram(models[0], p, v == "cpu" or not gpu):
            warnings.append(f"this PC has {p.get('ram_mb', 0) / 1024:.1f} GB of RAM; even the smallest "
                            f"model ({models[0]['label']}) may run slowly or not at all")
    for m in models:
        engine_settings(m)          # a bad catalog entry stops here, before any download
    items = [{"model": m, "model_url": _HF.format(**m),
              "model_path": os.path.join(models_dir, m["file"])}
             for m in models]
    warnings += path_warnings(root, models_dir if models_dir != os.path.join(root, "models", "gguf") else "")
    lc = catalog["llama_cpp"]
    files = [dict(f, url=lc["base_url"] + f["name"]) for f in lc["variants"][v]["files"]]
    server_dir = os.path.join(root, "runtime", "llama.cpp", f"{lc['tag']}-{v}")
    return {
        "root": root,
        "variant": v, "variant_label": lc["variants"][v]["label"], "why": why,
        "gpu": gpu, "backend_files": files,
        "profile": profile_id, "warnings": warnings, "models_dir": models_dir,
        # The first model is the default the launchers start with.
        "models": items, "model": models[0],
        "server_dir": server_dir,
        "server_exe": os.path.join(server_dir, "llama-server.exe"),
        "download_dir": os.path.join(root, "runtime", "downloads"),
        "config_path": os.path.join(root, "llama_cpp_config.json"),
        "download_mb": (sum(f["size"] for f in files)
                        + sum(m["size"] for m in models)) // (1 << 20),
    }


BACKEND_STAMP = ".artifex-backend.json"


def _backend_stamp(pl: dict) -> dict:
    return {"tag": os.path.basename(pl["server_dir"]), "variant": pl["variant"],
            "files": {f["name"]: f["sha256"] for f in pl["backend_files"]}}


def backend_complete(pl: dict) -> bool:
    """Every zip of the planned llama.cpp build has been unpacked.

    The CUDA builds come in two zips (the server, then the CUDA runtime); a
    run that died between them left llama-server.exe in place, and checking
    for the exe alone then skipped the runtime for good. The stamp is written
    only after the last zip is unpacked. A folder unpacked by an older
    installer (no stamp) counts as complete when its exe is there and, for a
    CUDA build, the runtime DLL too; the stamp is then written for it.
    """
    stamp = os.path.join(pl["server_dir"], BACKEND_STAMP)
    try:
        with open(stamp, encoding="utf-8") as f:
            return json.load(f) == _backend_stamp(pl)
    except (OSError, ValueError):
        pass
    if not os.path.isfile(pl["server_exe"]):
        return False
    if pl["variant"].startswith("cuda") and not any(
            n.lower().startswith("cudart") and n.lower().endswith(".dll")
            for n in os.listdir(pl["server_dir"])):
        return False
    _write_backend_stamp(pl)
    return True


def _write_backend_stamp(pl: dict) -> None:
    with open(os.path.join(pl["server_dir"], BACKEND_STAMP), "w", encoding="utf-8") as f:
        json.dump(_backend_stamp(pl), f, indent=1)


def missing_mb(pl: dict) -> tuple:
    """(MB still to download, MB of disk the install needs now).

    Only what isn't on disk yet counts: a rerun with everything in place
    needs nothing, where the old check asked for twice the whole install.
    Zips need room twice (the archive and what it unpacks to); models don't.
    A model that --reuse would find elsewhere still counts here; that only
    makes the estimate cautious.
    """
    zips = 0 if backend_complete(pl) else sum(f["size"] for f in pl["backend_files"])
    models = sum(item["model"]["size"] for item in pl["models"]
                 if not (os.path.isfile(item["model_path"])
                         and os.path.getsize(item["model_path"]) == item["model"]["size"]))
    return (zips + models) >> 20, (2 * zips + models) >> 20


def preflight(pl: dict, env: bool = True) -> list:
    """Problems that would stop the install AFTER the downloads: found now,
    before any multi-GB file is fetched."""
    import sys
    errors = []
    if env:
        py = venv_python(pl["root"])
        venv = os.path.dirname(os.path.dirname(py))
        fresh = (not os.path.exists(venv)) or os.path.isfile(os.path.join(venv, UNFINISHED_MARKER))
        if fresh and not python_ok():
            errors.append(
                f"this Python is {sys.version_info[0]}.{sys.version_info[1]}; the Python "
                f"environment needs {SUPPORTED_PYTHON[0][0]}.{SUPPORTED_PYTHON[0][1]} to "
                f"{SUPPORTED_PYTHON[1][0]}.{SUPPORTED_PYTHON[1][1]} (setup.cmd brings a "
                "pinned one), or use --no-env")
    path = pl["config_path"]
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8-sig") as f:
                cfg = json.load(f)
            if not isinstance(cfg, dict) or not isinstance(cfg.get("models", {}), dict):
                raise ValueError("expected an object with a \"models\" object")
        except ValueError as e:
            errors.append(f"{path} can't be read ({e}); fix it or move it aside, then run "
                          "this again (nothing was downloaded)")
        except OSError as e:
            errors.append(f"{path} can't be opened ({e})")
    return errors


def find_existing(model: dict, search_dirs) -> str:
    """A file already on disk that IS the pinned model (size, then SHA-256)."""
    for d in search_dirs or []:
        for base, dirs, names in os.walk(d):
            if base[len(d):].count(os.sep) >= 2:
                dirs[:] = []
            for n in names:
                if not n.endswith(".gguf"):
                    continue
                path = os.path.join(base, n)
                try:
                    if os.path.getsize(path) != model["size"]:
                        continue
                except OSError:
                    continue
                if fetch.file_matches(path, model["size"], model["sha256"]):
                    return path
    return ""


# Keys a catalog model may hand to its llama_cpp_config.json entry through
# "engine": {...}. Each one is read by core/engine_llama_cpp.py (a test keeps
# the two in step); anything else in "engine" is refused, so a catalog typo
# can't silently write a setting the app never reads.
ENGINE_KEYS = ("port", "health_timeout", "sampling", "reasoning_effort", "reasoning_always",
               "native_tools", "coexists_with_orpheus")


def engine_settings(m: dict) -> dict:
    eng = dict(m.get("engine") or {})
    if m.get("native_tools"):
        eng.setdefault("native_tools", True)
    bad = sorted(set(eng) - set(ENGINE_KEYS))
    if bad:
        raise ValueError(f"catalog model {m.get('id')!r}: unknown engine setting(s) {bad}")
    return eng


def config_entry(pl: dict, item: dict) -> dict:
    m = item["model"]
    profile = f" Profile: {pl['profile']}." if pl.get("profile") else ""
    return {
        "_comment": (f"Added by the Artifex installer {datetime.date.today()}: {m['label']}. "
                     f"{m['repo']}@{m['revision'][:12]} {m['file']} (sha256 {m['sha256'][:16]}...), "
                     f"license: {m['license']}. Backend: llama.cpp "
                     f"{os.path.basename(pl['server_dir'])} ({pl['why']}).{profile}"),
        "path": item["model_path"].replace("\\", "/"),
        "server_path": pl["server_exe"].replace("\\", "/"),
        "num_gpu_layers": 0 if pl["variant"] == "cpu" else 99,
        "num_ctx": m["num_ctx"],
        "extra_flags": list(m["extra_flags"]),
        **engine_settings(m),
    }


def write_config(pl: dict, force: bool = False) -> str:
    """Add the plan's models to llama_cpp_config.json without touching other
    entries.

    An existing entry with the same id is left alone unless force=True; the
    previous file is kept as a dated backup either way.
    """
    path = pl["config_path"]
    cfg = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            cfg = json.load(f)
    notes, added, moved = [], [], []
    new_server = pl["server_exe"].replace("\\", "/")
    ours = os.path.join(pl["root"], "runtime", "llama.cpp").replace("\\", "/").lower() + "/"
    for item in pl["models"]:
        mid = item["model"]["id"]
        old = cfg.get("models", {}).get(mid)
        if old is not None and not force:
            # An entry this installer wrote, still on a llama.cpp it unpacked
            # earlier: a newer pinned build only reaches it by moving its
            # server_path. Everything else in it (hand-tuned ctx, flags) stays;
            # an entry someone wrote or repointed by hand is never touched.
            entry = old if isinstance(old, dict) else {}
            sp = str(entry.get("server_path", ""))
            if (str(entry.get("_comment", "")).startswith("Added by the Artifex installer")
                    and sp.lower().startswith(ours) and sp != new_server):
                moved.append((mid, sp))
            else:
                notes.append(f"kept your existing '{mid}' entry in {path} (use --force to replace it)")
        else:
            added.append(item)
    if not added and not moved:
        return "\n".join(notes)
    if os.path.exists(path):
        backup = f"{path}.bak-installer-{datetime.datetime.now():%Y%m%d-%H%M%S}"
        with open(backup, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    cfg.setdefault("default_port", 8081)
    cfg.setdefault("server_path", pl["server_exe"].replace("\\", "/"))
    for item in added:
        cfg.setdefault("models", {})[item["model"]["id"]] = config_entry(pl, item)
        notes.append(f"added '{item['model']['id']}' to {path}")
    for mid, sp in moved:
        cfg["models"][mid]["server_path"] = new_server
        notes.append(f"moved '{mid}' to the new llama.cpp build "
                     f"({os.path.basename(os.path.dirname(sp))} -> "
                     f"{os.path.basename(pl['server_dir'])}); the rest of the entry is unchanged")
    if moved and str(cfg.get("server_path", "")).lower().startswith(ours):
        cfg["server_path"] = new_server
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)
    return "\n".join(notes)


def _configured_models(config_path: str) -> dict:
    try:
        with open(config_path, encoding="utf-8-sig") as f:
            models = json.load(f).get("models")
    except (OSError, ValueError, AttributeError):
        return {}
    return models if isinstance(models, dict) else {}


def default_model_id(config_path: str) -> str:
    """The model the GUI and CLI start with: the first one configured
    (core.config.get_active_llama_cpp_model), or "" if there is none."""
    return next(iter(_configured_models(config_path)), "")


STATE_FILE = "installer-state.json"


def save_state(pl: dict, variant_chosen: bool = False) -> str:
    """What this install was, so "Run the install again" repeats exactly it
    (the models asked for by id, a backend picked by hand) instead of the
    generic plan, which could fetch a different multi-GB model and point the
    launchers at it."""
    os.makedirs(pl["root"], exist_ok=True)
    path = os.path.join(pl["root"], STATE_FILE)
    state = {"models": [item["model"]["id"] for item in pl["models"]],
             "profile": pl.get("profile") or "",
             "variant": pl["variant"] if variant_chosen else "",
             "models_dir": "" if pl.get("models_dir") in (None, os.path.join(pl["root"], "models", "gguf"))
             else pl["models_dir"],
             "date": datetime.date.today().isoformat()}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, path)
    return path


def rerun_args(root: str, catalog: dict) -> list:
    """The install arguments that repeat this folder's install."""
    root = os.path.abspath(root)
    try:
        with open(os.path.join(root, STATE_FILE), encoding="utf-8") as f:
            st = json.load(f)
    except (OSError, ValueError):
        st = None
    if isinstance(st, dict):
        known = {m["id"] for m in catalog["chat_models"]}
        args = []
        if st.get("profile") in (catalog.get("profiles") or {}):
            args = ["--profile", st["profile"]]
        else:
            for mid in st.get("models") or []:
                if mid in known:            # a model since dropped from the catalog is skipped
                    args += ["--model", mid]
        if st.get("variant") in catalog["llama_cpp"]["variants"]:
            args += ["--variant", st["variant"]]
        if st.get("models_dir") and isinstance(st["models_dir"], str):
            args += ["--models-dir", st["models_dir"]]
        if args:
            return args
    prof = installed_profile(root, catalog)          # installs from before the state file
    return ["--profile", prof] if prof else []


def installed_profile(root: str, catalog: dict) -> str:
    """The machine profile this folder was installed from, or "".

    config_entry() notes the profile in every entry it writes; a profile
    counts as installed when each of its models has such an entry. "Run the
    install again" uses this to repeat THAT install: planned without it, a
    profile PC is offered the generic pick instead (one more multi-GB model,
    and launchers pointed at it).
    """
    models = _configured_models(os.path.join(os.path.abspath(root), "llama_cpp_config.json"))
    for pid, prof in (catalog.get("profiles") or {}).items():
        entries = [models.get(e["id"]) for e in prof["models"]]
        if entries and all(isinstance(m, dict) and f"Profile: {pid}." in str(m.get("_comment"))
                           for m in entries):
            return pid
    return ""


LOCKS = {name: os.path.join(HERE, f"requirements-{name}.txt")
         for name in ("core", "gui", "voice", "create", "strata", "holo", "holo-torch")}
PYTORCH_INDEX = "https://download.pytorch.org/whl/cu128"
ALLOWED_EXTRA_INDEXES = (PYTORCH_INDEX,)
SUPPORTED_PYTHON = ((3, 11), (3, 13))   # wheels in the locks cover these
INSTALLER_MARKER = "artifex-installer.txt"
UNFINISHED_MARKER = "artifex-installer-unfinished.txt"
GATEWAY_URL = "http://127.0.0.1:8080"   # the web-search pack's gateway (loopback only)


def venv_python(root: str) -> str:
    venv = os.path.join(os.path.abspath(root), "venv")
    return (os.path.join(venv, "Scripts", "python.exe") if os.name == "nt"
            else os.path.join(venv, "bin", "python"))


def add_locked_packages(root: str, lock: str, needed_modules, log=print,
                        run=subprocess.run, extra_index: str = "", no_deps: bool = False) -> str:
    """Install a pack's hash-locked packages into the installer's venv.

    A venv the installer didn't create (a hand-built full install) is left
    alone: if it already has the modules, fine; if not, say what's missing
    instead of changing someone's environment behind their back.

    no_deps: install exactly the lock's packages and nothing they declare
    (the lock must then list every package the pack really imports; the
    hashes are still required).
    """
    py = venv_python(root)
    venv = os.path.dirname(os.path.dirname(py))
    if not os.path.isfile(py):
        raise RuntimeError(f"no Python environment yet: run {hint('install')} first")
    if os.path.isfile(os.path.join(venv, INSTALLER_MARKER)):
        log(f"installing the locked packages from {os.path.basename(lock)}")
        # An extra index (PyTorch's, for CUDA builds) only widens where pip
        # may look; every file must still match a hash in the committed lock.
        if extra_index and extra_index not in ALLOWED_EXTRA_INDEXES:
            raise ValueError(f"refusing package index {extra_index!r}")
        run([py, "-m", "pip", "install", "--isolated", "--require-hashes",
             "--only-binary=:all:", "--no-input", "--disable-pip-version-check"]
            + (["--no-deps"] if no_deps else [])
            + (["--extra-index-url", extra_index] if extra_index else [])
            + ["-r", LOCKS["core"], "-r", lock], check=True)
        return py
    probe = "import importlib.util,sys; sys.exit(0 if all(importlib.util.find_spec(m) for m in sys.argv[1:]) else 1)"
    if run([py, "-c", probe] + list(needed_modules)).returncode != 0:
        raise RuntimeError(
            f"{venv} wasn't created by the installer, so it isn't changed automatically, "
            f"and it lacks {', '.join(needed_modules)}. Install them yourself with: "
            f'"{py}" -m pip install --require-hashes --only-binary=:all: -r "{lock}"')
    log(f"your environment {venv} already has what this pack needs (not modified)")
    return py


def python_ok(version=None) -> bool:
    import sys
    v = tuple((version or sys.version_info)[:2])
    return SUPPORTED_PYTHON[0] <= v <= SUPPORTED_PYTHON[1]


def install_env(root: str, gui: bool = True, log=print, run=subprocess.run) -> str:
    """Create <root>/venv and install the hash-locked packages into it.

    An existing venv is never modified: it may be a full install with
    PyTorch, and replacing packages under it could break the Create tools.
    pip runs isolated (no user/site pip.ini, no PIP_* environment, so nothing
    can add an index or turn checks off), with --require-hashes and
    --only-binary=:all:, so any file whose SHA-256 isn't in the committed
    lock is refused and no package's setup code ever runs.
    """
    import sys
    venv = os.path.join(os.path.abspath(root), "venv")
    py = os.path.join(venv, "Scripts", "python.exe") if os.name == "nt" \
        else os.path.join(venv, "bin", "python")
    unfinished = os.path.join(venv, UNFINISHED_MARKER)
    # A venv this installer started but didn't finish (pip was interrupted, the
    # network dropped) is its own to complete; without this it would be "kept"
    # forever with packages missing and the launchers would never start.
    resuming = os.path.isfile(unfinished)
    if os.path.exists(venv) and not resuming:
        if os.path.isfile(os.path.join(venv, INSTALLER_MARKER)):
            return _resync_env(venv, py, log=log, run=run)
        log(f"keeping your existing Python environment {venv} (not modified)")
        return py
    if not python_ok():
        raise RuntimeError(
            f"Python {sys.version_info[0]}.{sys.version_info[1]} is not supported here; "
            f"use {SUPPORTED_PYTHON[0][0]}.{SUPPORTED_PYTHON[0][1]} to "
            f"{SUPPORTED_PYTHON[1][0]}.{SUPPORTED_PYTHON[1][1]}")
    # The marker goes in before anything else, so a run that dies at any
    # later point is still recognised as ours.
    os.makedirs(venv, exist_ok=True)
    with open(unfinished, "w", encoding="utf-8") as f:
        f.write("python -m installer started this environment and has not finished it\n")
    if resuming and os.path.isfile(py):
        log(f"finishing the Python environment {venv} that an earlier run started")
    else:
        log(f"creating {venv}")
        run([sys.executable, "-m", "venv", venv], check=True)
    locks = ["core"] + (["gui"] if gui else [])
    log("installing the locked packages (core" + (" + desktop GUI" if gui else "") + ")")
    run(_pip_cmd(py, locks), check=True)
    # Marks the venv as the installer's own: packs may add their locked
    # packages to it, but never to a venv someone built by hand.
    _write_marker(venv, locks)
    os.remove(unfinished)
    return py


def _pip_cmd(py: str, locks) -> list:
    cmd = [py, "-m", "pip", "install", "--isolated", "--require-hashes",
           "--only-binary=:all:", "--no-input", "--disable-pip-version-check"]
    for name in locks:
        cmd += ["-r", LOCKS[name]]
    return cmd


def _lock_digest(locks) -> dict:
    return {name: fetch.sha256_file(LOCKS[name]) for name in locks}


def _write_marker(venv: str, locks) -> None:
    with open(os.path.join(venv, INSTALLER_MARKER), "w", encoding="utf-8") as f:
        f.write("created by python -m installer; packs may add hash-locked packages here\n")
        f.write("locks " + json.dumps(_lock_digest(locks)) + "\n")


def _marker_locks(venv: str) -> dict:
    try:
        with open(os.path.join(venv, INSTALLER_MARKER), encoding="utf-8") as f:
            for line in f:
                if line.startswith("locks "):
                    return json.loads(line[6:])
    except (OSError, ValueError):
        pass
    return {}


def _resync_env(venv: str, py: str, log=print, run=subprocess.run) -> str:
    """Bring the installer's own venv up to the committed locks.

    Security fixes land as new pins in requirements-*.txt; an environment
    made by an older installer never saw them, because an existing venv was
    always kept as it was. Only a venv carrying the installer's marker is
    touched (a hand-built one never is), with the same isolated,
    hash-checked, wheels-only pip; packages already at the pinned version are
    left alone by pip, so an up-to-date venv costs one quick check.
    """
    had = _marker_locks(venv)
    locks = ["core"] + (["gui"] if had.get("gui") or _has_module(py, "PyQt6", run) else [])
    if had and had == _lock_digest(locks):
        log(f"Python environment {venv} matches the locks")
        return py
    log(f"updating {venv} to the committed package locks ({', '.join(locks)})")
    run(_pip_cmd(py, locks), check=True)
    _write_marker(venv, locks)
    return py


def _has_module(py: str, module: str, run=subprocess.run) -> bool:
    probe = f"import importlib.util,sys; sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"
    try:
        return run([py, "-c", probe], check=False).returncode == 0
    except (OSError, AttributeError):
        return False


def write_launchers(root: str, model_id: str, gui: bool = True) -> list:
    """Plain .cmd launchers. Fixed text, no user input in them; the API one
    binds to 127.0.0.1 only, so nothing is reachable from the network until
    the remote-access pack sets up Tailscale on purpose."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", model_id):
        raise ValueError(f"refusing to write a launcher for model id {model_id!r}")
    disk_root = os.path.abspath(root)
    # cmd.exe expands %VAR% even inside quotes; double every % in the text
    # (only in the text: the files themselves go in the real folder).
    root = disk_root.replace("%", "%%")
    py = os.path.join(root, "venv", "Scripts", "python.exe")
    common = ["@echo off", "chcp 65001 >nul", f'cd /d "{root}"', "set ARTIFEX_BACKEND=llama_cpp",
              f"set WEB_GATEWAY_URL={GATEWAY_URL}"]
    files = {
        "Artifex CLI.cmd": common + [f'"{py}" main.py', "pause"],
        "Artifex API (this PC only).cmd": common + [
            f'"{py}" main_api.py --host 127.0.0.1 --port 8000 --backend llama_cpp '
            f'--model {model_id} --gateway {GATEWAY_URL}', "pause"],
    }
    if gui:
        files["Artifex GUI.cmd"] = common + [f'start "" "{py[:-len("python.exe")]}pythonw.exe" main_gui_qt.py']
    written = []
    for name, lines in files.items():
        path = os.path.join(disk_root, name)
        with open(path, "w", encoding="utf-8", newline="\r\n") as f:
            f.write("\n".join(lines) + "\n")
        written.append(path)
    return written


def is_installed(root: str) -> bool:
    """Core chat is in place: a finished environment and at least one model."""
    root = os.path.abspath(root)
    py = venv_python(root)
    venv = os.path.dirname(os.path.dirname(py))
    if not os.path.isfile(py) or os.path.isfile(os.path.join(venv, UNFINISHED_MARKER)):
        return False
    try:
        with open(os.path.join(root, "llama_cpp_config.json"), encoding="utf-8-sig") as f:
            return bool(json.load(f).get("models"))
    except (OSError, ValueError):
        return False


def has_gui(root: str) -> bool:
    return os.path.isfile(os.path.join(os.path.abspath(root), "Artifex GUI.cmd"))


def start_app(root: str, gui: bool = True, popen=subprocess.Popen):
    """Start Artifex the way its launchers do (same folder, same two settings),
    as a direct process: no cmd.exe and nothing typed by the user in between."""
    root = os.path.abspath(root)
    py = venv_python(root)
    env = dict(os.environ, ARTIFEX_BACKEND="llama_cpp", WEB_GATEWAY_URL=GATEWAY_URL)
    if gui:
        pyw = os.path.join(os.path.dirname(py), "pythonw.exe")
        return popen([pyw if os.path.isfile(pyw) else py, "main_gui_qt.py"], cwd=root, env=env)
    # The terminal version gets a window of its own, so setup can close.
    return popen([py, "main.py"], cwd=root, env=env,
                 creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))


def write_phone_launcher(root: str, port: int = 8000, model_id: str = None) -> str:
    """The API for the phone app: checks keys against artifex_api_keys.json (hashes) and
    still binds to 127.0.0.1 (Tailscale serve forwards the tailnet to it).
    Agent and file tools stay off; turn them on deliberately with
    ARTIFEX_PHONE_FULL_TOOLS=1 if you want them from the phone."""
    if model_id is not None and not re.fullmatch(r"[A-Za-z0-9._-]+", model_id):
        raise ValueError(f"refusing to write a launcher for model id {model_id!r}")
    if not isinstance(port, int) or not 1024 <= port <= 65535:
        raise ValueError(f"bad port {port!r}")
    disk_root = os.path.abspath(root)
    croot = disk_root.replace("%", "%%")
    py = os.path.join(croot, "venv", "Scripts", "python.exe")
    model = f" --model {model_id}" if model_id else ""
    lines = ["@echo off", "chcp 65001 >nul", f'cd /d "{croot}"',
             "set ARTIFEX_BACKEND=llama_cpp",
             f'if not exist "artifex_api_keys.json" (echo Run: {hint("remote setup")} in this folder & pause & exit /b 1)',
             f'set "ARTIFEX_API_KEYS_FILE={croot}\\artifex_api_keys.json"',
             f"set WEB_GATEWAY_URL={GATEWAY_URL}",
             f'"{py}" main_api.py --host 127.0.0.1 --port {port} --backend llama_cpp --phone-app'
             f'{model} --gateway {GATEWAY_URL}',
             "pause"]
    path = os.path.join(disk_root, "Artifex API (phone via Tailscale).cmd")
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")
    return path


def install_core(pl: dict, catalog: dict, reuse_dirs=None, force=False, log=print,
                 progress=None) -> dict:
    hosts = catalog["allowed_hosts"]
    # 1. llama.cpp: download the pinned zips, verify, unpack.
    if backend_complete(pl):
        log(f"llama.cpp already unpacked in {pl['server_dir']}")
    else:
        for f in pl["backend_files"]:
            dest = os.path.join(pl["download_dir"], f["name"])
            log(f"downloading {f['name']} ({f['size'] // (1 << 20)} MB)")
            fetch.download(f["url"], dest, f["size"], f["sha256"], hosts, progress=progress)
            log("  checksum OK; unpacking")
            fetch.safe_unzip(dest, pl["server_dir"])
            os.remove(dest)
        _write_backend_stamp(pl)
    # 2. Models: reuse an identical file already on disk, else download it.
    for item in pl["models"]:
        m = item["model"]
        existing = find_existing(m, reuse_dirs)
        if existing:
            log(f"reusing {existing} (identical to the pinned {m['file']})")
            item["model_path"] = existing
        else:
            log(f"downloading {m['file']} ({m['size'] // (1 << 20)} MB) from {m['repo']}")
            fetch.download(item["model_url"], item["model_path"], m["size"], m["sha256"],
                           hosts, progress=progress)
            log("  checksum OK")
    # 3. Smoke-test the (verified) server binary, then register the model.
    try:
        out = subprocess.run([pl["server_exe"], "--version"], capture_output=True,
                             text=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise RuntimeError("llama-server didn't answer within a minute; the model was not "
                           "registered. Run this again, or try --variant vulkan or cpu.")
    if out.returncode != 0:
        # Registering a server that can't start would only move the failure
        # to the first chat, with a far less clear message.
        code = out.returncode & 0xFFFFFFFF
        why = ("Windows couldn't find a DLL it needs (0xC0000135). On a fresh Windows that "
               "is usually the Microsoft Visual C++ Redistributable (x64): install it from "
               "Microsoft, then run this again."
               if code == 0xC0000135 else
               f"it exited with code {code:#x}. Try --variant vulkan or --variant cpu.")
        raise RuntimeError(f"llama-server ({pl['variant']}) doesn't start on this PC: {why}")
    version = (out.stdout + out.stderr).strip().splitlines()
    log("llama-server: " + (next((l for l in version if "version" in l.lower()), "")
                            or "(no version line)"))
    log(write_config(pl, force=force))
    return pl
