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

from installer import fetch

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


def choose_variant(p: dict) -> tuple:
    """(variant id, gpu dict or None, reason)."""
    gpus = [g for g in p.get("gpus") or [] if g.get("vram_mb", 0) >= 2048]
    nvidia = sorted((g for g in gpus if g.get("vendor") == "NVIDIA"),
                    key=lambda g: g["vram_mb"], reverse=True)
    if nvidia:
        g = nvidia[0]
        cuda = _version((p.get("nvidia") or {}).get("cuda_version", ""))
        cc = _version(g.get("compute_cap", "0.0"))
        if cuda >= (13, 0):
            return "cuda-13.4", g, f"NVIDIA driver supports CUDA {cuda[0]}.{cuda[1]}"
        # The CUDA 12.4 build has no Blackwell (RTX 50, compute 12.x) kernels.
        if cuda >= (12, 4) and cc < (12, 0):
            return "cuda-12.4", g, f"NVIDIA driver supports CUDA {cuda[0]}.{cuda[1]}"
        return "vulkan", g, ("NVIDIA driver too old for the CUDA builds; using Vulkan "
                             "(updating the driver would allow CUDA)")
    if gpus:
        g = max(gpus, key=lambda g: g["vram_mb"])
        return "vulkan", g, f"{g['vendor']} GPU"
    return "cpu", None, "no GPU with 2 GB or more of video memory"


def choose_model(catalog: dict, variant: str, gpu) -> dict:
    vram = gpu["vram_mb"] if (gpu and variant != "cpu") else 0
    fits = [m for m in catalog["chat_models"] if m["min_vram_mb"] <= vram]
    return max(fits, key=lambda m: m["min_vram_mb"])


def plan(p: dict, catalog: dict, root: str, model_id: str = None,
         variant: str = None) -> dict:
    v, gpu, why = choose_variant(p)
    if variant:
        if variant not in catalog["llama_cpp"]["variants"]:
            raise ValueError(f"unknown backend variant {variant!r}")
        v, why = variant, "chosen by you"
    if model_id:
        model = next((m for m in catalog["chat_models"] if m["id"] == model_id), None)
        if not model:
            raise ValueError(f"unknown model {model_id!r}; options: "
                             + ", ".join(m["id"] for m in catalog["chat_models"]))
    else:
        model = choose_model(catalog, v, gpu)
    lc = catalog["llama_cpp"]
    files = [dict(f, url=lc["base_url"] + f["name"]) for f in lc["variants"][v]["files"]]
    root = os.path.abspath(root)
    server_dir = os.path.join(root, "runtime", "llama.cpp", f"{lc['tag']}-{v}")
    return {
        "root": root,
        "variant": v, "variant_label": lc["variants"][v]["label"], "why": why,
        "gpu": gpu, "model": model, "backend_files": files,
        "model_url": _HF.format(**model),
        "server_dir": server_dir,
        "server_exe": os.path.join(server_dir, "llama-server.exe"),
        "model_path": os.path.join(root, "models", "gguf", model["file"]),
        "download_dir": os.path.join(root, "runtime", "downloads"),
        "config_path": os.path.join(root, "llama_cpp_config.json"),
        "download_mb": (sum(f["size"] for f in files) + model["size"]) // (1 << 20),
    }


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


def config_entry(pl: dict) -> dict:
    m = pl["model"]
    return {
        "_comment": (f"Added by the Artifex installer {datetime.date.today()}: {m['label']}. "
                     f"{m['repo']}@{m['revision'][:12]} {m['file']} (sha256 {m['sha256'][:16]}...), "
                     f"license: {m['license']}. Backend: llama.cpp "
                     f"{os.path.basename(pl['server_dir'])} ({pl['why']})."),
        "path": pl["model_path"].replace("\\", "/"),
        "server_path": pl["server_exe"].replace("\\", "/"),
        "num_gpu_layers": 0 if pl["variant"] == "cpu" else 99,
        "num_ctx": m["num_ctx"],
        "extra_flags": list(m["extra_flags"]),
        **({"native_tools": True} if m.get("native_tools") else {}),
    }


def write_config(pl: dict, force: bool = False) -> str:
    """Add the model to llama_cpp_config.json without touching other entries.

    An existing entry with the same id is left alone unless force=True; the
    previous file is kept as a dated backup either way.
    """
    path, mid = pl["config_path"], pl["model"]["id"]
    cfg = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        if mid in cfg.get("models", {}) and not force:
            return f"kept your existing '{mid}' entry in {path} (use --force to replace it)"
        backup = f"{path}.bak-installer-{datetime.datetime.now():%Y%m%d-%H%M%S}"
        with open(backup, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    cfg.setdefault("default_port", 8081)
    cfg.setdefault("server_path", pl["server_exe"].replace("\\", "/"))
    cfg.setdefault("models", {})[mid] = config_entry(pl)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)
    return f"added '{mid}' to {path}"


LOCKS = {name: os.path.join(HERE, f"requirements-{name}.txt")
         for name in ("core", "gui", "voice", "create")}
PYTORCH_INDEX = "https://download.pytorch.org/whl/cu128"
ALLOWED_EXTRA_INDEXES = (PYTORCH_INDEX,)
SUPPORTED_PYTHON = ((3, 11), (3, 13))   # wheels in the locks cover these
INSTALLER_MARKER = "artifex-installer.txt"
GATEWAY_URL = "http://127.0.0.1:8080"   # the web-search pack's gateway (loopback only)


def venv_python(root: str) -> str:
    venv = os.path.join(os.path.abspath(root), "venv")
    return (os.path.join(venv, "Scripts", "python.exe") if os.name == "nt"
            else os.path.join(venv, "bin", "python"))


def add_locked_packages(root: str, lock: str, needed_modules, log=print,
                        run=subprocess.run, extra_index: str = "") -> str:
    """Install a pack's hash-locked packages into the installer's venv.

    A venv the installer didn't create (a hand-built full install) is left
    alone: if it already has the modules, fine; if not, say what's missing
    instead of changing someone's environment behind their back.
    """
    py = venv_python(root)
    venv = os.path.dirname(os.path.dirname(py))
    if not os.path.isfile(py):
        raise RuntimeError("no Python environment yet: run python -m installer install first")
    if os.path.isfile(os.path.join(venv, INSTALLER_MARKER)):
        log(f"installing the locked packages from {os.path.basename(lock)}")
        # An extra index (PyTorch's, for CUDA builds) only widens where pip
        # may look; every file must still match a hash in the committed lock.
        if extra_index and extra_index not in ALLOWED_EXTRA_INDEXES:
            raise ValueError(f"refusing package index {extra_index!r}")
        run([py, "-m", "pip", "install", "--isolated", "--require-hashes",
             "--only-binary=:all:", "--no-input", "--disable-pip-version-check"]
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
    if os.path.exists(venv):
        log(f"keeping your existing Python environment {venv} (not modified)")
        return py
    if not python_ok():
        raise RuntimeError(
            f"Python {sys.version_info[0]}.{sys.version_info[1]} is not supported here; "
            f"use {SUPPORTED_PYTHON[0][0]}.{SUPPORTED_PYTHON[0][1]} to "
            f"{SUPPORTED_PYTHON[1][0]}.{SUPPORTED_PYTHON[1][1]}")
    log(f"creating {venv}")
    run([sys.executable, "-m", "venv", venv], check=True)
    cmd = [py, "-m", "pip", "install", "--isolated", "--require-hashes",
           "--only-binary=:all:", "--no-input", "--disable-pip-version-check",
           "-r", LOCKS["core"]]
    if gui:
        cmd += ["-r", LOCKS["gui"]]
    log("installing the locked packages (core" + (" + desktop GUI" if gui else "") + ")")
    run(cmd, check=True)
    # Marks the venv as the installer's own: packs may add their locked
    # packages to it, but never to a venv someone built by hand.
    os.makedirs(venv, exist_ok=True)
    with open(os.path.join(venv, INSTALLER_MARKER), "w", encoding="utf-8") as f:
        f.write("created by python -m installer; packs may add hash-locked packages here\n")
    return py


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


def write_phone_launcher(root: str, port: int = 8000, model_id: str = None) -> str:
    """The API for the phone app: loads the key from artifex_api_key.txt and
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
             'if not exist "artifex_api_key.txt" (echo Run: python -m installer remote setup & pause & exit /b 1)',
             'set /p ARTIFEX_API_KEY=<"artifex_api_key.txt"',
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
    if os.path.isfile(pl["server_exe"]):
        log(f"llama.cpp already unpacked in {pl['server_dir']}")
    else:
        for f in pl["backend_files"]:
            dest = os.path.join(pl["download_dir"], f["name"])
            log(f"downloading {f['name']} ({f['size'] // (1 << 20)} MB)")
            fetch.download(f["url"], dest, f["size"], f["sha256"], hosts, progress=progress)
            log("  checksum OK; unpacking")
            fetch.safe_unzip(dest, pl["server_dir"])
            os.remove(dest)
    # 2. Model: reuse an identical file already on disk, else download it.
    m = pl["model"]
    existing = find_existing(m, reuse_dirs)
    if existing:
        log(f"reusing {existing} (identical to the pinned {m['file']})")
        pl["model_path"] = existing
    else:
        log(f"downloading {m['file']} ({m['size'] // (1 << 20)} MB) from {m['repo']}")
        fetch.download(pl["model_url"], pl["model_path"], m["size"], m["sha256"], hosts,
                       progress=progress)
        log("  checksum OK")
    # 3. Smoke-test the (verified) server binary, then register the model.
    out = subprocess.run([pl["server_exe"], "--version"], capture_output=True,
                         text=True, timeout=60)
    version = (out.stdout + out.stderr).strip().splitlines()
    log("llama-server: " + (next((l for l in version if "version" in l.lower()), "")
                            or "(no version line)"))
    log(write_config(pl, force=force))
    return pl
