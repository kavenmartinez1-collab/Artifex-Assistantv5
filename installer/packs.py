"""Optional packs on top of core chat: python -m installer packs / pack <id>.

Same rules as the core install: every download pinned in catalog.json (or a
hash lock), no improvised commands, nothing exposed beyond 127.0.0.1. Each
pack says what it will do and asks first. Things that need admin rights or
come with their own licence terms (Docker Desktop, Ollama) are never
installed silently: the pack explains where to get them and stops.
"""

import json
import os
import re
import shutil
import subprocess
import time
import urllib.request

from installer import fetch, hint, recipes, remote

GATEWAY_URL = "http://127.0.0.1:8080"


class PackUnavailable(Exception):
    """The pack can't be installed on this PC yet; the message says why/how."""


def _docker(args, root, timeout=1800, run=subprocess.run):
    return run(["docker"] + list(args), cwd=root, capture_output=True, text=True,
               timeout=timeout)


def docker_ready(run=subprocess.run) -> str:
    """Docker server version, or '' when Docker isn't installed or running."""
    if not shutil.which("docker"):
        return ""
    try:
        r = run(["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def gateway_health(url=GATEWAY_URL, timeout_s=180, sleep=time.sleep) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                return json.loads(r.read().decode("utf-8"))
        except (OSError, ValueError):
            sleep(3)
    return {}


# ── Web search ──────────────────────────────────────────────────────────────

def web_search_plan(root, catalog):
    return [
        "check that Docker is installed and running (it is not installed for you)",
        "build the web gateway from web-gateway/ (base image pinned by digest, "
        "every Python package hash-locked)",
        "start it with SearXNG (image pinned by digest) from docker-compose.yml: "
        "the gateway on 127.0.0.1:8080 only, SearXNG not published at all",
        "wait for the gateway's health check, then run the exposure check",
    ]


def install_web_search(root, catalog, log=print, run=subprocess.run):
    ver = docker_ready(run)
    if not ver:
        raise PackUnavailable(
            "Web search runs in Docker, which isn't installed or isn't running. "
            "Install Docker Desktop from https://www.docker.com/products/docker-desktop/ "
            "(it needs admin rights and has its own licence terms, so Artifex doesn't "
            "install it for you), start it, then run this pack again.")
    log(f"Docker {ver} is running")
    compose = os.path.join(root, "docker-compose.yml")
    if not os.path.isfile(compose):
        raise PackUnavailable(f"{compose} is missing; run this from an Artifex checkout")
    log("building and starting web-gateway + searxng (first time: a few minutes)")
    r = _docker(["compose", "-f", compose, "up", "-d", "--build", "web-gateway", "searxng"],
                root, run=run)
    if r.returncode != 0:
        raise RuntimeError("docker compose failed: " + (r.stderr or r.stdout)[-800:])
    health = gateway_health()
    if not health:
        raise RuntimeError(f"the gateway didn't answer on {GATEWAY_URL}/health")
    log(f"gateway: {health.get('status')} (searxng: {health.get('searxng')})")
    return health


# ── Ollama ──────────────────────────────────────────────────────────────────

def ollama_plan(root, catalog):
    return ["check that Ollama is installed (it is not installed for you)",
            "check that it only listens on this PC (Ollama has no password)",
            "list its models and write launchers that start Artifex on the Ollama backend"]


def _ollama_models(run=subprocess.run) -> list:
    exe = shutil.which("ollama")
    if not exe:
        return []
    try:
        r = run([exe, "list"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    rows = (r.stdout or "").strip().splitlines()[1:]
    return [row.split()[0] for row in rows if row.strip()]


def install_ollama(root, catalog, log=print, run=subprocess.run, env=os.environ):
    if not shutil.which("ollama"):
        raise PackUnavailable(
            "Ollama isn't installed. Get it from https://ollama.com/download (its own "
            "installer and licence), then run this pack again. Artifex uses it as is.")
    host = env.get("OLLAMA_HOST", "")
    if host and remote.classify_address(host.split("//")[-1].rsplit(":", 1)[0] or "0.0.0.0") in ("any", "lan"):
        raise PackUnavailable(
            f"OLLAMA_HOST is set to {host!r}, which opens Ollama (no password) to your "
            "network. Remove it or set it to 127.0.0.1, restart Ollama, then run this again.")
    exposed = [s for s in remote.listening_sockets(run)
               if s[1] == 11434 and remote.classify_address(s[0]) in ("any", "lan")]
    if exposed:
        raise PackUnavailable(f"Ollama is listening on {exposed[0][0]}:11434, open to your "
                              "network. Bind it to 127.0.0.1 first.")
    models = _ollama_models(run)
    log("Ollama models: " + (", ".join(models) if models else
                             "none yet (ollama pull <model>)"))
    return write_backend_launchers(root, "ollama")


def write_backend_launchers(root, backend):
    if backend not in ("ollama", "llama_cpp"):
        raise ValueError(backend)
    disk_root = os.path.abspath(root)
    croot = disk_root.replace("%", "%%")
    py = os.path.join(croot, "venv", "Scripts", "python.exe")
    label = "Ollama" if backend == "ollama" else "llama.cpp"
    common = ["@echo off", "chcp 65001 >nul", f'cd /d "{croot}"',
              f"set ARTIFEX_BACKEND={backend}", f"set WEB_GATEWAY_URL={GATEWAY_URL}"]
    files = {f"Artifex CLI ({label}).cmd": common + [f'"{py}" main.py', "pause"]}
    written = []
    for name, lines in files.items():
        path = os.path.join(disk_root, name)
        with open(path, "w", encoding="utf-8", newline="\r\n") as f:
            f.write("\n".join(lines) + "\n")
        written.append(path)
    return written


# ── Voice ───────────────────────────────────────────────────────────────────

def voice_plan(root, catalog):
    v = catalog["voice"]
    mb = sum(f["size"] for k in ("whisper", "piper") for f in v[k]["files"]) // (1 << 20)
    return ["add the hash-locked voice packages (faster-whisper, Piper; ~200 MB) to "
            "the installer's Python environment",
            f"download Whisper small.en and the Piper 'joe' voice ({mb} MB, pinned to a "
            "commit and checksum-verified) into models/",
            "test it: speak a sentence with Piper, transcribe it back with Whisper"]


def _fetch_hf_files(spec, root, hosts, log=print):
    dest_dir = os.path.join(root, spec["dest"])
    for f in spec["files"]:
        name = os.path.basename(f["path"])
        dest = os.path.join(dest_dir, name)
        if not fetch.file_matches(dest, f["size"], f["sha256"]):
            log(f"downloading {name} ({max(1, f['size'] // (1 << 20))} MB)")
        fetch.download(f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{f['path']}",
                       dest, f["size"], f["sha256"], hosts)


def install_voice(root, catalog, log=print, run=subprocess.run):
    py = recipes.add_locked_packages(root, recipes.LOCKS["voice"], ["faster_whisper", "piper"],
                                     log=log, run=run)
    for key in ("whisper", "piper"):
        _fetch_hf_files(catalog["voice"][key], root, catalog["allowed_hosts"], log)
    log("testing: Piper speaks, Whisper transcribes it back")
    test = ("from core.services import voice_io as v; w = v.tts('The voice pack works.'); "
            "print(v.stt(w))")
    r = run([py, "-c", test], cwd=root, capture_output=True, text=True, timeout=900)
    heard = (r.stdout or "").strip().splitlines()[-1:] if r.returncode == 0 else []
    if not heard or "voice pack" not in heard[0].lower():
        raise RuntimeError("voice self-test failed: " + ((r.stderr or r.stdout) or "")[-600:])
    log(f"heard back: {heard[0]!r}")
    return heard[0]


# ── Vision ──────────────────────────────────────────────────────────────────

def _config(root):
    path = os.path.join(root, "llama_cpp_config.json")
    if not os.path.isfile(path):
        raise PackUnavailable(f"no llama_cpp_config.json yet: run {hint('install')} first")
    with open(path, encoding="utf-8-sig") as f:
        return path, json.load(f)


def _save_config(path, cfg):
    backup = f"{path}.bak-installer-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(path, backup)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def _hf_file(spec, root, hosts, log=print):
    dest = os.path.join(root, "models", "gguf", spec["file"])
    if not fetch.file_matches(dest, spec["size"], spec["sha256"]):
        log(f"downloading {spec['file']} ({spec['size'] // (1 << 20)} MB)")
    fetch.download(recipes._HF.format(**spec), dest, spec["size"], spec["sha256"], hosts)
    return dest.replace("\\", "/")


def vision_plan(root, catalog):
    return ["for each chat model you installed that can see, download its pinned image "
            "projector (~0.9 GB) and add a '<model>-vision' entry (your entries are not changed)",
            "download Qwen3.5 2B + its image encoder (about 1.9 GB, pinned; the model file is "
            "shared with the small chat model) and the CPU llama.cpp build if missing, and "
            "register the CPU vision helper (port 8095, this PC only)",
            "test it: the helper describes a test picture"]


def install_vision(root, catalog, log=print, run=subprocess.run):
    hosts = catalog["allowed_hosts"]
    path, cfg = _config(root)
    models = cfg.setdefault("models", {})
    added = []
    for m in catalog["chat_models"]:
        vid = m["id"] + "-vision"
        if m["id"] in models and m.get("mmproj") and vid not in models:
            mm = _hf_file(m["mmproj"], root, hosts, log)
            entry = json.loads(json.dumps(models[m["id"]]))
            entry["extra_flags"] = list(entry.get("extra_flags", [])) + ["--mmproj", mm]
            entry["_comment"] = f"Added by the installer's vision pack: {m['id']} + {m['mmproj']['file']}."
            models[vid] = entry
            added.append(vid)
    # The CPU vision helper: its own llama.cpp CPU build, model and projector.
    vh = catalog["vision_helper"]
    cpu_plan = recipes.plan({"gpus": []}, catalog, root, variant="cpu",
                            model_id=catalog["chat_models"][0]["id"])
    if not recipes.backend_complete(cpu_plan):
        for f in cpu_plan["backend_files"]:
            dest = os.path.join(cpu_plan["download_dir"], f["name"])
            log(f"downloading {f['name']} (CPU llama.cpp for the helper)")
            fetch.download(f["url"], dest, f["size"], f["sha256"], hosts)
            fetch.safe_unzip(dest, cpu_plan["server_dir"])
            os.remove(dest)
        recipes._write_backend_stamp(cpu_plan)
    # The helper's model is the small chat model's file: when an entry already
    # points at a verified copy (installed with --reuse, say), use that one.
    model = next((str(e["path"]).replace("\\", "/") for e in models.values()
                  if isinstance(e, dict) and e.get("path")
                  and os.path.basename(str(e["path"])) == vh["model"]["file"]
                  and fetch.file_matches(str(e["path"]), vh["model"]["size"], vh["model"]["sha256"])), "")
    if model:
        log(f"vision helper: using the verified {vh['model']['file']} the chat model already has")
    else:
        model = _hf_file(vh["model"], root, hosts, log)
    mmproj = _hf_file(vh["mmproj"], root, hosts, log)
    old = cfg.get("vision_helper")
    if (isinstance(old, dict) and str(old.get("_comment", "")).startswith("Added by the installer's vision pack")
            and os.path.basename(str(old.get("path", ""))) != vh["model"]["file"]):
        # A helper this pack set up earlier with a model the catalog no longer
        # ships (LFM3-VL until 2026-11): replace it. One you wrote yourself
        # is left alone.
        log(f"replacing the vision helper's old model {os.path.basename(str(old.get('path')))}")
        del cfg["vision_helper"]
    if "vision_helper" not in cfg:
        cfg["vision_helper"] = {
            "_comment": f"Added by the installer's vision pack. {vh['license']}.",
            "path": model, "server_path": cpu_plan["server_exe"].replace("\\", "/"),
            "num_gpu_layers": 0, "num_ctx": vh["num_ctx"], "port": vh["port"],
            "idle_unload_s": vh["idle_unload_s"], "assist_tools": ["edit", "image"],
            "extra_flags": ["--mmproj", mmproj, "--no-mmproj-offload", "-t",
                            str(max(2, (os.cpu_count() or 4) - 2)), "-np", "1", "--jinja"]}
        if vh.get("sampling"):
            cfg["vision_helper"]["sampling"] = dict(vh["sampling"])
        added.append("vision_helper")
    if added:
        _save_config(path, cfg)
        log("added to llama_cpp_config.json: " + ", ".join(added))
    log("testing: the vision helper describes a test picture")
    test = ("import base64, io; from PIL import Image, ImageDraw; from core import vision_helper as vh\n"
            "im = Image.new('RGB', (320, 200), 'white'); d = ImageDraw.Draw(im)\n"
            "d.rectangle([20, 20, 140, 180], fill='red'); d.ellipse([170, 30, 300, 160], fill='blue')\n"
            "b = io.BytesIO(); im.save(b, 'PNG'); u = 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()\n"
            "try:\n"
            "    print(vh.chat([{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': u}},"
            " {'type': 'text', 'text': 'Name the two shapes and their colours in one line.'}]}], max_tokens=80))\n"
            "finally:\n"
            "    vh._engine and vh._engine.unload()\n")
    py = recipes.venv_python(root)
    r = run([py, "-c", test], cwd=root, capture_output=True, text=True, timeout=600)
    answer = (r.stdout or "").strip().splitlines()[-1:] if r.returncode == 0 else []
    if not answer or not re.search(r"red|blue", answer[0], re.I):
        raise RuntimeError("vision self-test failed: " + ((r.stderr or r.stdout) or "")[-600:])
    log(f"helper saw: {answer[0]!r}")
    return answer[0]


# ── Create ──────────────────────────────────────────────────────────────────

QWEN_MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen-image-manifest.json")
QWEN_DIR = os.path.join("models", "qwen-image-2.1-turbo")


def _create_gpu(probe_fn=None):
    from installer import probe as probe_mod
    p = (probe_fn or probe_mod.probe)()
    nv = [g for g in p.get("gpus") or [] if g.get("vendor") == "NVIDIA"]
    cuda = recipes._version((p.get("nvidia") or {}).get("cuda_version", ""))
    if not nv:
        raise PackUnavailable(
            "Create needs an NVIDIA GPU: its image and music models run on PyTorch with "
            "CUDA, which has no Windows build for AMD or Intel cards.")
    g = max(nv, key=lambda x: x["vram_mb"])
    if cuda < (12, 8):
        raise PackUnavailable(
            f"Create's PyTorch build needs an NVIDIA driver that supports CUDA 12.8 "
            f"(yours: {'.'.join(map(str, cuda)) or 'unknown'}). Update the driver, then run "
            "this pack again.")
    if g["vram_mb"] < 7500:
        raise PackUnavailable(f"Create needs 8 GB of video memory ({g['name']} has "
                              f"{g['vram_mb'] / 1024:.1f} GB).")
    return g


def _free_ram_gb() -> float:
    if os.name != "nt":
        return 0.0
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return 0.0
    return m.ullAvailPhys / (1 << 30)


def create_plan(root, catalog):
    have = os.path.isdir(os.path.join(root, QWEN_DIR))
    return ["check for an NVIDIA GPU with 8 GB+ and a CUDA 12.8 driver",
            "add the hash-locked Create packages (PyTorch 2.7.1 CUDA 12.8, diffusers, "
            "transformers; ~3.5 GB) to the installer's Python environment",
            ("keep your existing Qwen-Image folder" if have else
             "download the official Qwen-Image 2.1 Turbo (32 GB, pinned to a commit, every "
             "file checksum-verified), compress it to 4-bit (~11.5 GB), delete the 32 GB copy"),
            "music, 3D and video download on first use, each pinned to a fixed commit",
            "test it: generate one picture"]


def install_create(root, catalog, log=print, run=subprocess.run, probe_fn=None):
    g = _create_gpu(probe_fn)
    log(f"GPU: {g['name']} ({g['vram_mb'] / 1024:.1f} GB)")
    py = recipes.add_locked_packages(root, recipes.LOCKS["create"], ["torch", "diffusers"],
                                     log=log, run=run, extra_index=recipes.PYTORCH_INDEX)
    out = os.path.join(root, QWEN_DIR)
    src = os.path.join(root, "runtime", "downloads", "qwen-image-src")
    if os.path.isdir(out):
        log(f"keeping {out}")
        if os.path.isdir(src):          # left over from an interrupted earlier run
            shutil.rmtree(src, ignore_errors=True)
            log("removed the leftover 32 GB source copy")
    else:
        m = json.load(open(QWEN_MANIFEST, encoding="utf-8"))
        need = [f for f in m["files"] if not f["path"].startswith("assets/")]
        free = shutil.disk_usage(root).free
        if free < sum(f["size"] for f in need) + 12 * (1 << 30):
            raise PackUnavailable(f"not enough disk space: ~45 GB needed, "
                                  f"{free / (1 << 30):.0f} GB free")
        for f in need:
            parts = f["path"].split("/")
            if any(p in ("", ".", "..") for p in parts):
                raise fetch.VerifyError(f"refused manifest path {f['path']!r}")
            dest = os.path.join(src, *parts)
            if not fetch.file_matches(dest, f["size"], f["sha256"]):
                log(f"downloading {f['path']} ({max(1, f['size'] // (1 << 20))} MB)")
            fetch.download(f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{f['path']}",
                           dest, f["size"], f["sha256"], catalog["allowed_hosts"])
        # Loading the 17.5 GB bf16 text encoder to compress it needs most of a
        # 32 GB PC: with Artifex and Docker also running, the first live run
        # died with an access violation (0xC0000005); with them closed it
        # finished in ~4 min. Downloads are kept, so a rerun picks up here.
        free_gb = _free_ram_gb()
        if free_gb and free_gb < 18:
            raise PackUnavailable(
                f"compressing the image model needs about 18 GB of free memory and "
                f"{free_gb:.0f} GB is free. Close other programs (Artifex, Docker Desktop, "
                "browsers), then run this pack again; the download is kept.")
        log("compressing to 4-bit with scripts/quantize_qwen_image.py (needs the GPU)")
        run([py, os.path.join(root, "scripts", "quantize_qwen_image.py"), "--repo", src,
             "--out", out], cwd=root, check=True)
        shutil.rmtree(src, ignore_errors=True)
        log("removed the 32 GB source copy")
    log("testing: generating one picture (about a minute and a half)")
    test = ("import sys; from core.services import get_service, get_file_manager; "
            "from core.services import media_tools as m\n"
            "mp = m._resolve_model('image', None)\n"
            "kw = m._pipeline_kwargs('image', mp, {'prompt': 'a red apple on a wooden table', "
            "'size': 'square'}, None)\n"
            "r = get_service().run_pipeline('text-to-image', mp, kwargs=kw, store_output=True)\n"
            "print('OK', get_file_manager().get_file_path(r.metadata['file_id'])) if r.success "
            "else print('FAIL', r.error)\n")
    r = run([py, "-c", test], cwd=root, capture_output=True, text=True, timeout=1800)
    line = ((r.stdout or "").strip().splitlines() or [""])[-1]
    if not line.startswith("OK "):
        raise RuntimeError("image self-test failed: " + (line or (r.stderr or "")[-600:]))
    log(f"test picture: {line[3:]}")
    return line[3:]


# ── Node.js projects: Artifex Web (WebGPU) and the game ─────────────────────
#
# npm is a supply-chain target of its own: install scripts run arbitrary
# code (how the 2026 "Auth Error" North Korean repos work). So:
#   - `npm ci`, never `npm install`: exact lockfile versions, every package
#     checked against the lockfile's integrity hash, and the lockfile is never
#     rewritten. The lockfile itself is audited first (below).
#   - --ignore-scripts: no package's install/postinstall code runs. Both apps
#     run without them (they only touch esbuild/onnxruntime/protobuf tooling).
#   - npm is started as `node npm-cli.js`, never through npm.cmd: running a
#     .cmd from code goes through cmd.exe and its argument parsing
#     (the "BatBadBut" class of bugs).
#   - Node itself: an installed Node 20+ is used as is; otherwise the pinned
#     portable zip from nodejs.org (catalog.json "node") goes in runtime/node.

NPM_REGISTRY = "https://registry.npmjs.org/"


def ensure_node(root, catalog, log=print, run=subprocess.run, prefer_portable=False):
    """(node.exe, npm-cli.js) to use, installing the pinned portable Node if needed."""
    spec = catalog["node"]
    if not prefer_portable:
        node = shutil.which("node")
        if node:
            try:
                ver = run([node, "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                ver = ""
            m = re.match(r"v(\d+)\.", ver)
            cli = os.path.join(os.path.dirname(node), "node_modules", "npm", "bin", "npm-cli.js")
            if m and int(m.group(1)) >= spec["min_major"] and os.path.isfile(cli):
                log(f"using the installed Node.js {ver}")
                return node, cli
    base = os.path.join(root, "runtime", "node")
    home = os.path.join(base, spec["file"][:-len(".zip")])
    node, cli = os.path.join(home, "node.exe"), os.path.join(home, "node_modules", "npm", "bin", "npm-cli.js")
    if not os.path.isfile(node):
        dest = os.path.join(root, "runtime", "downloads", spec["file"])
        log(f"downloading Node.js {spec['version']} ({spec['size'] // (1 << 20)} MB, pinned)")
        fetch.download(spec["base_url"] + spec["file"], dest, spec["size"], spec["sha256"],
                       catalog["allowed_hosts"])
        fetch.safe_unzip(dest, base)
        os.remove(dest)
    return node, cli


def audit_lockfile(project_dir, allow_git=()):
    """Refuse a lockfile that would let npm fetch something unverified."""
    path = os.path.join(project_dir, "package-lock.json")
    if not os.path.isfile(path):
        raise PackUnavailable(f"{path} is missing: without it npm can't verify anything")
    lock = json.load(open(path, encoding="utf-8"))
    if lock.get("lockfileVersion", 0) < 2:
        raise PackUnavailable("package-lock.json is too old (lockfileVersion < 2)")
    problems = []
    for name, p in lock.get("packages", {}).items():
        if not name or p.get("link"):
            continue
        resolved = p.get("resolved", "")
        if resolved.startswith(NPM_REGISTRY):
            if not p.get("integrity", "").startswith("sha512-"):
                problems.append(f"{name}: no sha512 integrity")
            continue
        git = re.fullmatch(r"git\+(?:ssh://git@|https://)github\.com/([\w.-]+/[\w.-]+?)(?:\.git)?#([0-9a-f]{40})",
                           resolved)
        if git and git.group(1) in allow_git:
            continue        # pinned to a full commit; git checks the content by hash
        problems.append(f"{name}: resolved from {resolved or 'nowhere'}")
    if problems:
        raise fetch.VerifyError("package-lock.json has unverifiable entries: " + "; ".join(problems[:5]))


def _npm_env(node):
    env = {k: v for k, v in os.environ.items() if not k.lower().startswith("npm_config_")}
    env.update({
        "npm_config_registry": NPM_REGISTRY, "npm_config_ignore_scripts": "true",
        "npm_config_audit": "false", "npm_config_fund": "false",
        "npm_config_update_notifier": "false",
        "PATH": os.path.dirname(node) + os.pathsep + env.get("PATH", ""),
        # A git dependency is fetched over https only (no ssh keys needed) and
        # git may use no other transport.
        "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_COUNT": "4",
        "GIT_CONFIG_KEY_0": "url.https://github.com/.insteadOf", "GIT_CONFIG_VALUE_0": "ssh://git@github.com/",
        "GIT_CONFIG_KEY_1": "url.https://github.com/.insteadOf", "GIT_CONFIG_VALUE_1": "git@github.com:",
        "GIT_CONFIG_KEY_2": "protocol.allow", "GIT_CONFIG_VALUE_2": "never",
        "GIT_CONFIG_KEY_3": "protocol.https.allow", "GIT_CONFIG_VALUE_3": "always",
    })
    return env


def npm(node, cli, args, cwd, run=subprocess.run, timeout=3600):
    r = run([node, cli] + list(args), cwd=cwd, env=_npm_env(node), capture_output=True,
            text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"npm {' '.join(args)} failed: " + ((r.stderr or r.stdout) or "")[-800:])
    return r


def check_loopback_only(project_dir):
    """Both apps' dev servers must bind 127.0.0.1; refuse to set up one that doesn't."""
    vite = open(os.path.join(project_dir, "vite.config.ts"), encoding="utf-8").read()
    srv = open(os.path.join(project_dir, "server", "dev-server.ts"), encoding="utf-8").read()
    if not re.search(r"host:\s*'127\.0\.0\.1'", vite) or not re.search(
            r"\.listen\(\s*PORT\s*,\s*'127\.0\.0\.1'", srv):
        raise fetch.VerifyError(f"{project_dir}: dev server isn't bound to 127.0.0.1 only")


def _web_launcher(root, name, project_dir, node, url):
    disk = os.path.join(os.path.abspath(root), name)
    q = lambda p: os.path.abspath(p).replace("%", "%%")
    lines = ["@echo off", "chcp 65001 >nul", f'cd /d "{q(project_dir)}"',
             f'set "PATH={q(os.path.dirname(node))};%PATH%"',
             "set npm_config_ignore_scripts=true",
             f'start "" "{url}"',
             f'"{q(node)}" "{q(os.path.join(os.path.dirname(node), "node_modules", "npm", "bin", "npm-cli.js"))}" run dev',
             "pause"]
    with open(disk, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")
    return disk


def webgpu_plan(root, catalog):
    w = catalog["webgpu"]
    spec = _webgpu_spec(catalog)
    return ["use your Node.js 20+ (or download the pinned portable Node.js)",
            "check webgpu/package-lock.json: every package from the npm registry with a "
            "sha512 integrity hash",
            "npm ci --ignore-scripts (exact versions, hashes checked, no install scripts)",
            f"put {spec['label'].split(' (')[0]} in webgpu/models/{w['model_dir']}: the core "
            f"install's verified copy when it has one, else a pinned download "
            f"({spec['size'] // (1 << 20)} MB), plus its pinned tokenizer files",
            "check its servers bind 127.0.0.1 only, build it as a test, write 'Artifex Web.cmd' "
            f"(it opens with local/{w['model_dir']} filled in)"]


def _webgpu_spec(catalog):
    w = catalog["webgpu"]
    return next(m for m in catalog["chat_models"] if m["id"] == w["model"])


def _webgpu_model(root, catalog, log=print):
    """Artifex Web's model: the pinned GGUF + tokenizer in webgpu/models/<dir>."""
    w, spec, hosts = catalog["webgpu"], _webgpu_spec(catalog), catalog["allowed_hosts"]
    dest_dir = os.path.join(root, "webgpu", "models", w["model_dir"])
    os.makedirs(dest_dir, exist_ok=True)
    gguf = os.path.join(dest_dir, spec["file"])
    core = os.path.join(root, "models", "gguf", spec["file"])
    if (not fetch.file_matches(gguf, spec["size"], spec["sha256"])
            and fetch.file_matches(core, spec["size"], spec["sha256"])):
        # Same verified file: hard-link it (no second copy on disk), else copy;
        # either way it lands under a temporary name and is re-verified first.
        tmp = gguf + ".part"
        if os.path.exists(tmp):
            os.remove(tmp)
        try:
            os.link(core, tmp)
            log(f"{spec['file']}: linked the core install's verified copy")
        except OSError:
            log(f"{spec['file']}: copying the core install's verified copy")
            shutil.copyfile(core, tmp)
        if not fetch.file_matches(tmp, spec["size"], spec["sha256"]):
            os.remove(tmp)
            raise fetch.VerifyError(f"{spec['file']}: the copy does not match its pinned hash")
        os.replace(tmp, gguf)
    if not fetch.file_matches(gguf, spec["size"], spec["sha256"]):
        log(f"downloading {spec['file']} ({spec['size'] // (1 << 20)} MB) for Artifex Web")
    fetch.download(recipes._HF.format(**spec), gguf, spec["size"], spec["sha256"], hosts)
    tok = w["tokenizer"]
    for f in tok["files"]:
        fetch.download(recipes._HF.format(repo=tok["repo"], revision=tok["revision"], file=f["file"]),
                       os.path.join(dest_dir, f["file"]), f["size"], f["sha256"], hosts)
    log(f"Artifex Web model ready: local/{w['model_dir']}")
    return "local/" + w["model_dir"]


def install_webgpu(root, catalog, log=print, run=subprocess.run, prefer_portable=False):
    proj = os.path.join(root, "webgpu")
    if not os.path.isfile(os.path.join(proj, "package.json")):
        raise PackUnavailable(f"{proj} isn't here; run this from an Artifex checkout")
    check_loopback_only(proj)
    audit_lockfile(proj)
    node, cli = ensure_node(root, catalog, log, run, prefer_portable)
    log("npm ci --ignore-scripts (Artifex Web)")
    npm(node, cli, ["ci", "--ignore-scripts"], proj, run)
    log("building it as a test (tsc + vite build)")
    npm(node, cli, ["run", "build"], proj, run)
    model = _webgpu_model(root, catalog, log)
    return [_web_launcher(root, "Artifex Web.cmd", proj, node,
                          "http://127.0.0.1:5173/?model=" + model)]


def game_plan(root, catalog):
    g = catalog["game"]
    mb = sum(m["size"] for m in g["models"]) // (1 << 20)
    return [f"git clone {g['repo']} over https and check out commit {g['commit'][:12]} "
            f"(verified) into {g['dest']}",
            "check its package-lock.json and dev servers (127.0.0.1 only)",
            "npm ci --ignore-scripts; steamworks.js comes from its pinned git commit over https",
            f"download its voices and NPC model ({mb} MB, each file pinned and checksum-verified)",
            "build the pronunciation lexicon from a pinned CMUdict and check the result's hash",
            "build it as a test and write 'A Tale of Yarn.cmd'"]


def _git(args, cwd=None, run=subprocess.run):
    git = shutil.which("git")
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    r = run([git, "-c", "protocol.allow=never", "-c", "protocol.https.allow=always",
             "-c", "core.hooksPath=NUL" if os.name == "nt" else "core.hooksPath=/dev/null"] + list(args),
            cwd=cwd, env=env, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: " + (r.stderr or r.stdout)[-600:])
    return r.stdout.strip()


def install_game(root, catalog, log=print, run=subprocess.run, prefer_portable=False):
    g = catalog["game"]
    if not shutil.which("git"):
        raise PackUnavailable("The game is installed with git. Get Git for Windows from "
                              "https://git-scm.com/download/win, then run this pack again.")
    dest = os.path.join(root, *g["dest"].split("/"))
    if not os.path.isdir(os.path.join(dest, ".git")):
        log(f"cloning {g['repo']}")
        _git(["clone", "--no-checkout", g["repo"], dest], run=run)
    elif _git(["status", "--porcelain"], cwd=dest, run=run):
        raise PackUnavailable(f"{dest} has local changes; it is left alone")
    else:
        _git(["fetch", "origin"], cwd=dest, run=run)
    _git(["checkout", "--detach", g["commit"]], cwd=dest, run=run)
    head = _git(["rev-parse", "HEAD"], cwd=dest, run=run)
    if head != g["commit"]:
        raise fetch.VerifyError(f"checked out {head}, expected {g['commit']}")
    log(f"game source at commit {head[:12]} (verified)")
    check_loopback_only(dest)
    audit_lockfile(dest, allow_git=("ceifa/steamworks.js",))
    node, cli = ensure_node(root, catalog, log, run, prefer_portable)
    log("npm ci --ignore-scripts (A Tale of Yarn)")
    npm(node, cli, ["ci", "--ignore-scripts"], dest, run)
    hosts = catalog["allowed_hosts"]
    for m in g["models"]:
        target = os.path.join(dest, *m["dest"].split("/"))
        if not fetch.file_matches(target, m["size"], m["sha256"]):
            log(f"downloading {m['dest']} ({max(1, m['size'] // (1 << 20))} MB)")
        fetch.download(f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{m['path']}",
                       target, m["size"], m["sha256"], hosts)
    cmu = g["cmudict"]
    for f in cmu["files"]:
        fetch.download(cmu["base_url"] + f["name"],
                       os.path.join(dest, *cmu["cache_dir"].split("/"), f["name"]),
                       f["size"], f["sha256"], hosts)
    log("building the pronunciation lexicon (from the pinned CMUdict, offline)")
    tsx = os.path.join(dest, "node_modules", "tsx", "dist", "cli.mjs")
    r = run([node, tsx, "scripts/build-lexicon.mts"], cwd=dest, env=_npm_env(node),
            capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError("lexicon build failed: " + (r.stderr or r.stdout)[-600:])
    for rel, want in cmu["expected_output"].items():
        got = fetch.sha256_file(os.path.join(dest, *rel.split("/")))
        if got != want:
            raise fetch.VerifyError(f"{rel} came out different from the pinned build")
    log("building it as a test (tsc + vite build)")
    npm(node, cli, ["run", "build"], dest, run)
    return [_web_launcher(root, "A Tale of Yarn.cmd", dest, node, "http://127.0.0.1:5173/game.html")]


# ── Strata: Qwen3.8-Flash-Next on one GPU + system RAM ──────────────────────
#
# Strata (github.com/Niko1221/Strata, MIT) runs the 125B-total / 6B-active MoE
# from GPU + RAM + SSD. Since 0.2.20 nothing of Strata's own installer runs:
# every input is pinned in catalog.json and fetched by fetch.py (allow-listed
# hosts, size + SHA-256), and only Strata's offline tools run:
#   - the source: git (https) sparse checkout of terminal/ at the pinned commit;
#   - Python: its own venv with OUR hash-locked packages (requirements-strata.txt);
#   - the engine: the archive for this GPU (sha256 pinned), unpacked with the
#     tar-slip checks, then its Ed25519-signed manifest verified offline by
#     Strata's own code against the pinned key id, every file re-hashed;
#   - the model: two pinned shards (an older pack's identical shard 2 is linked);
#   - the expert pack: tools/iq_pack.py --experts-bin, offline. A 32 GB PC runs
#     Strata's low-RAM resident mode, which reads the experts from that ~40 GB
#     experts.bin (upstream maps the GGUF in place only with ~60 GB of RAM; the
#     resident mode on a pack without experts.bin crashes the 0.1.66 engine);
#   - the MTP draft layer: 31 tensors fetched as pinned byte ranges of the base
#     model's shards (each SHA-256 pinned), packed by tools/mtp_pack.py and
#     tools/mtp_rt.py, offline.
# The server binds 127.0.0.1 only, runs --offline (no remote image URLs, no MCP
# subprocesses), keeps its state under strata\state, and Artifex starts and
# stops it (external_launcher).

STRATA_LOG_DIR = "logs"

# Strata's own table at the pinned commit (terminal/strata_control/engine_component.py
# WIN_AMD_DEVICE_ARCH): the architecture a Windows AMD card runs as, by PCI device id.
_AMD_DEVICE_ARCH = {0x744C: "gfx1100", 0x7448: "gfx1100", 0x745E: "gfx1100", 0x747E: "gfx1101",
                    0x7480: "gfx1102", 0x7590: "gfx1200", 0x7550: "gfx1201", 0x7551: "gfx1201",
                    0x73BF: "gfx1030", 0x73AF: "gfx1030", 0x73A5: "gfx1030", 0x73A1: "gfx1030",
                    0x73DF: "gfx1031", 0x73FF: "gfx1032", 0x73EF: "gfx1032", 0x73E3: "gfx1032"}
_GFX_ENGINE = {"gfx1030": "hip-rdna2", "gfx1031": "hip-rdna2", "gfx1032": "hip-rdna2",
               "gfx1100": "hip-rdna3", "gfx1101": "hip-rdna3", "gfx1102": "hip-rdna3",
               "gfx1200": "hip-rx9070", "gfx1201": "hip-rx9070"}


def _amd_gfx(g) -> str:
    """The card's gfx arch: by PCI device id, else from its name (Strata's gfx_from_name rules)."""
    gfx = _AMD_DEVICE_ARCH.get(int(g.get("device_id") or 0), "")
    if gfx:
        return gfx
    name = re.sub(r"\s+", " ", str(g.get("name", ""))).upper()
    if re.search(r"\bR9700\b|\bR97\d\d\b|\bRX 9070\b", name):
        return "gfx1201"
    if re.search(r"\bRX 906\d\b", name):
        return "gfx1200"
    m = re.search(r"\b(?:RX |W)(7[5-9]\d\d)\b", name)
    if m:
        return {79: "gfx1100", 78: "gfx1101", 77: "gfx1101", 76: "gfx1102", 75: "gfx1102"}[int(m.group(1)) // 100]
    m = re.search(r"\bRX (6\d{3})\b", name)
    if m:
        n = int(m.group(1))
        if n >= 6800:
            return "gfx1030"
        if n in (6700, 6750):
            return "gfx1031"
        if 6600 <= n < 6700:
            return "gfx1032"
    return ""


def _strata_gpu(probe_fn=None):
    """(backend, engine variant, gpu, ram_gb) for the card Strata should use: the one with the most VRAM."""
    from installer import probe as probe_mod
    p = (probe_fn or probe_mod.probe)()
    ram_gb = (p.get("ram_mb") or 0) / 1024
    gpus = [g for g in p.get("gpus") or [] if g.get("vendor") in ("NVIDIA", "AMD")]
    if not gpus:
        raise PackUnavailable("Strata needs an NVIDIA (RTX 20 or newer) or AMD Radeon RX 6000, "
                              "RX 7000 or RX 9000 graphics card.")
    g = max(gpus, key=lambda x: x.get("vram_mb") or 0)
    if (g.get("vram_mb") or 0) < 11000:
        raise PackUnavailable(f"Strata needs a card with 12 GB of video memory or more ({g['name']} has "
                              f"{(g.get('vram_mb') or 0) / 1024:.0f} GB).")
    if ram_gb and ram_gb < 30:
        raise PackUnavailable(f"Flash-Next IQ3_XXS needs 32 GB of RAM or more (this PC has {ram_gb:.0f} GB).")
    if g["vendor"] == "NVIDIA":
        cc = recipes._version(g.get("compute_cap", ""))
        drv = recipes._version((p.get("nvidia") or {}).get("driver", ""))
        if cc and cc < (7, 5):
            raise PackUnavailable(f"{g['name']} is older than RTX 20 (compute capability {g.get('compute_cap')}); "
                                  "Strata's ready-made NVIDIA engines start at sm_75.")
        # Strata's ENGINE-COMPONENT.md at the pinned commit: CUDA 13 needs 580.65+, CUDA 12 575.51+.
        if drv >= (580, 65):
            return "cuda", "cuda-cuda13", g, ram_gb
        if drv >= (575, 51):
            return "cuda", "cuda-cuda12", g, ram_gb
        raise PackUnavailable("Strata's NVIDIA engine needs driver 575.51 or newer (580.65+ for CUDA 13). "
                              "Update the driver.")
    gfx = _amd_gfx(g)
    if gfx in _GFX_ENGINE:
        return "hip", _GFX_ENGINE[gfx], g, ram_gb
    raise PackUnavailable(f"{g['name']}: Strata has no ready-made Windows engine for this AMD card "
                          "(RX 6600-6900, RX 7000 and RX 9000 / AI PRO R9700 are supported).")


def _strata_paths(root, s):
    strata = os.path.join(root, *s["dest"].split("/"))
    data = os.path.join(root, *s["data_dest"].split("/"))
    return {"strata": strata, "terminal": os.path.join(strata, "terminal"),
            "models": os.path.join(root, *s["models_dest"].split("/")),
            "data": data, "pack": os.path.join(data, "packs", "iq3_xxs"), "mtp": os.path.join(data, "mtp"),
            "state": os.path.join(root, "strata", "state"), "downloads": os.path.join(root, "strata", "downloads")}


def _strata_artifex_config(s, paths, engine_dir, backend, log_path):
    """Artifex's run config for Strata's server: loopback only, the measured settings."""
    t = s["tuning"]
    shard = lambda i: os.path.join(paths["models"], os.path.basename(s["model"]["files"][i]["path"]))
    args = ["--pack", paths["pack"], "--native", shard(0), "--ple-gguf", shard(1),
            "--expert-profile", os.path.join(paths["terminal"], "data", "expert-profile.bin"),
            "--expert-cache", "auto", "--prefill", str(t["prefill"]),
            "--spec", str(t["spec"]), "--spec-min-p", str(t["spec_min_p"]),
            "--mtp", os.path.join(paths["mtp"], "rt"), "--max-context", str(t["default_ctx"]),
            "--kv", t["kv"], "--resident-experts", "--conversation-cache-mib", str(t["conversation_cache_mib"]),
            # Strata pins RAM for experts up to "all but 4 GiB" by default; Artifex and the
            # rest of the PC need more room than that.
            "--ram-keep-gib", str(t["ram_keep_gib"])]
    env = {"STRATA_ROUTE_PRIOR": str(t["route_prior"]), "STRATA_IO_THREADS": str(t["io_threads"])}
    lib_dirs = []
    if backend == "hip":
        env["STRATA_RESIDENT_PIN"] = "0"          # Strata's own rule for HIP on Windows (setup_config.py)
        rocm = os.path.join(engine_dir, "rocm", "bin")
        if os.path.isdir(rocm):
            lib_dirs.append(rocm)
    return {"exe": os.path.join(engine_dir, "strata.exe"), "args": args, "cwd": paths["terminal"],
            "tokenizer": os.path.join(paths["pack"], "tokenizer"), "model_name": s["model_id"],
            "lib_dirs": lib_dirs, "backend": backend, "env": env, "log": log_path,
            "host": "127.0.0.1", "port": s["port"], "open_browser": False}


def _strata_launcher(root, strata_dir, vpy, cfg_path, port, max_ctx):
    path = os.path.join(os.path.abspath(root), "strata", "start-strata-artifex.ps1")
    state = os.path.join(os.path.abspath(root), "strata", "state")
    q = lambda p: os.path.abspath(p).replace("'", "''")
    text = f"""# Strata (Qwen3.8-Flash-Next) for Artifex: 127.0.0.1:{port} ONLY. Written by python -m installer
# (pack strata); Artifex starts it through its external_launcher entry and passes -Context <tier>.
param([int]$Context = 32768)
$ErrorActionPreference = 'Stop'
$strata = '{q(strata_dir)}'
$term = Join-Path $strata 'terminal'
$Context = [Math]::Max(4096, [Math]::Min($Context, {max_ctx}))
$cfg = Get-Content '{q(cfg_path)}' -Raw -Encoding UTF8 | ConvertFrom-Json
$a = [System.Collections.ArrayList]@($cfg.args)
$i = $a.IndexOf('--max-context')
if ($i -ge 0) {{ $a[$i + 1] = "$Context" }} else {{ [void]$a.Add('--max-context'); [void]$a.Add("$Context") }}
$cfg.args = @($a)
$cfg.host = '127.0.0.1'
$run = Join-Path $strata 'strata-artifex-run.json'
$cfg | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $run
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = $term
# Strata keeps its state (process records, update cache) under LOCALAPPDATA\\Strata; this
# install gets its own, so it never shares records with (or ends) another Strata.
$env:LOCALAPPDATA = '{q(state)}'
$env:STRATA_KEEP_LEFTOVERS = '1'
$env:STRATA_OFFLINE = '1'
foreach ($p in $cfg.env.PSObject.Properties) {{ Set-Item -Path ("env:" + $p.Name) -Value ([string]$p.Value) }}
Set-Location $term
& '{q(vpy)}' -u (Join-Path $term 'serve\\server.py') --engine strata --config $run --host 127.0.0.1 --port {port} --offline --no-api-key
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.makedirs(state, exist_ok=True)
    # UTF-8 with a BOM: Windows PowerShell 5.1 reads a script without one in
    # the ANSI code page, so a non-ASCII folder name would break it (the old
    # ascii-only write crashed outright, after the whole download).
    with open(path, "w", encoding="utf-8-sig", newline="\r\n") as f:
        f.write(text)
    return path


def _strata_smoke_test(launcher, port, model_id, log=print, run=subprocess.run, timeout_s=900):
    """Start the server like Artifex would, ask one question, stop it (and its children)."""
    import psutil
    base = f"http://127.0.0.1:{port}"

    def healthy():
        try:
            with urllib.request.urlopen(base + "/health", timeout=3) as r:
                return json.loads(r.read()).get("status") == "ok"
        except (OSError, ValueError):
            return False
    if healthy():
        log(f"something already answers on 127.0.0.1:{port} (Artifex running Flash-Next?) - test skipped")
        return "skipped"
    proc = subprocess.Popen([remote.system_exe("powershell"), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", launcher,
                             "-Context", "8192"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        deadline = time.monotonic() + timeout_s
        while not healthy():
            if proc.poll() is not None:
                raise RuntimeError("the Strata server stopped while starting (see logs/strata-flash-next.log)")
            if time.monotonic() > deadline:
                raise RuntimeError(f"the Strata server was not ready after {timeout_s} s")
            time.sleep(3)
        listen = [c for c in psutil.net_connections(kind="tcp")
                  if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN]
        if any(c.laddr.ip not in ("127.0.0.1", "::1") for c in listen):
            raise fetch.VerifyError(f"the Strata server listens beyond 127.0.0.1: {[c.laddr.ip for c in listen]}")
        body = json.dumps({"model": model_id, "max_tokens": 40, "temperature": 0,
                           "chat_template_kwargs": {"enable_thinking": False},
                           "messages": [{"role": "user", "content": "What is 17*23? Answer with the number only."}]})
        req = urllib.request.Request(base + "/v1/chat/completions", data=body.encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            reply = json.loads(r.read())["choices"][0]["message"]["content"].strip()
        if "391" not in reply:
            raise RuntimeError(f"Strata answered {reply!r} to 17*23")
        return reply
    finally:
        try:
            p = psutil.Process(proc.pid)
            procs = p.children(recursive=True) + [p]
        except psutil.NoSuchProcess:
            procs = []
        for p in procs:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(procs, timeout=20)


def _strata_tool(vpy, paths, args, log_name, root, run, timeout):
    """Run one of Strata's offline tools from terminal/, output to logs/<log_name>."""
    log_path = os.path.join(root, STRATA_LOG_DIR, log_name)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=paths["terminal"], STRATA_OFFLINE="1",
               LOCALAPPDATA=paths["state"])
    with open(log_path, "w", encoding="utf-8") as lf:
        r = run([vpy, "-u"] + args, cwd=paths["terminal"], env=env, stdout=lf, stderr=subprocess.STDOUT,
                timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"Strata's {args[0]} failed (exit {r.returncode}); see {log_path}")
    return log_path


_ENGINE_CHECK = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from strata_control import engine_component as ec
c = ec.read_manifest(Path(sys.argv[2]))
ec.verify(c, full=True)
status = ec.verify_signature(c, {"STRATA_REQUIRE_SIGNED_COMPONENTS": "1"})
print(json.dumps({"status": status, "version": c.version, "key_id": (c.signature or {}).get("key_id", "")}))
"""


def _strata_engine(s, variant, vpy, paths, hosts, log, run):
    """This GPU's engine: pinned archive, slip-checked unpack, signed manifest verified offline."""
    spec = s["engines"][variant]
    engine_dir = os.path.join(paths["strata"], "..", f"engine-{s['engine_version']}-{variant}")
    engine_dir = os.path.abspath(engine_dir)
    stamp = os.path.join(engine_dir, ".artifex-verified")
    if not os.path.isfile(stamp):
        archive = os.path.join(paths["downloads"], spec["url"].rsplit("/", 1)[-1])
        log(f"downloading the engine {os.path.basename(archive)} ({spec['size'] // (1 << 20)} MB)")
        fetch.download(spec["url"], archive, spec["size"], spec["sha256"], hosts)
        if os.path.isdir(engine_dir):
            shutil.rmtree(engine_dir)
        fetch.safe_untar(archive, engine_dir)
        os.remove(archive)
    r = run([vpy, "-c", _ENGINE_CHECK, paths["terminal"], engine_dir], capture_output=True, text=True,
            timeout=900, env=dict(os.environ, PYTHONUTF8="1"))
    try:
        got = json.loads((r.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise fetch.VerifyError("the Strata engine failed its signature/file check: "
                                + ((r.stderr or r.stdout) or "")[-400:])
    if r.returncode != 0 or got.get("status") != "verified" or got.get("key_id") != s["engine_key_id"] \
            or got.get("version") != s["engine_version"]:
        raise fetch.VerifyError(f"the Strata engine package is not the pinned, signed build: {got}")
    with open(stamp, "w", encoding="utf-8") as f:
        json.dump(got, f)
    log(f"engine {got['version']} ({variant}): signed by {got['key_id']}, every file re-hashed")
    return engine_dir


def _strata_models(s, paths, root, hosts, log):
    os.makedirs(paths["models"], exist_ok=True)
    for f in s["model"]["files"]:
        name = os.path.basename(f["path"])
        dest = os.path.join(paths["models"], name)
        if not fetch.file_matches(dest, f["size"], f["sha256"]):
            # An older pack's identical file (IQ2_XS shared shard 2): link it, don't fetch 29 GB again.
            for older in s.get("older_models_dest", []):
                odir = os.path.join(root, *older.split("/"))
                for cand in (os.listdir(odir) if os.path.isdir(odir) else []):
                    src = os.path.join(odir, cand)
                    if os.path.getsize(src) == f["size"] and fetch.file_matches(src, f["size"], f["sha256"]):
                        try:
                            os.link(src, dest)
                            log(f"{name}: linked the identical file {cand} from {older}")
                        except OSError:
                            pass
                        break
        if not fetch.file_matches(dest, f["size"], f["sha256"]):
            log(f"downloading {name} ({f['size'] / 1e9:.1f} GB)")
        fetch.download(f"https://huggingface.co/{s['model']['repo']}/resolve/{s['model']['revision']}/{f['path']}",
                       dest, f["size"], f["sha256"], hosts)


def _strata_mtp(s, vpy, paths, root, hosts, log, run):
    """The MTP draft layer: pinned tensor byte ranges, then Strata's offline packers."""
    mtp, m = paths["mtp"], s["mtp"]
    rt_done = os.path.join(mtp, "rt", ".artifex-done")
    if os.path.isfile(rt_done):
        log("MTP draft layer: already prepared")
        return
    manifest = []
    for t in m["tensors"]:
        rel = os.path.join("tensors", t["name"] + ".bin")
        fetch.download_range(f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{t['shard']}",
                             os.path.join(mtp, rel), t["start"], t["end"], t["sha256"], hosts)
        manifest.append(dict(t, file=rel))
    log(f"MTP draft layer: {len(manifest)} tensors fetched, each matching its pinned SHA-256")
    with open(os.path.join(mtp, "mtp-manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    gguf = os.path.join(mtp, "mtp-q2_0.gguf")
    _strata_tool(vpy, paths, ["tools/mtp_pack.py", "--src", mtp, "--experts", "q2_0", "--out", gguf],
                 "strata-mtp-pack.log", root, run, 3600)
    _strata_tool(vpy, paths, ["tools/mtp_rt.py", "--gguf", gguf, "--out", os.path.join(mtp, "rt")],
                 "strata-mtp-rt.log", root, run, 3600)
    shutil.copy2(os.path.join(paths["terminal"], "data", "draft_vocab.bin"), os.path.join(mtp, "rt", "draft_vocab.bin"))
    with open(rt_done, "w", encoding="utf-8") as f:
        f.write(f"{m['repo']}@{m['revision']}\n")


def _strata_pack(s, vpy, paths, root, log, run):
    done = os.path.join(paths["pack"], ".artifex-done")
    shard1 = s["model"]["files"][0]
    want = shard1["sha256"] + " experts-bin"
    if os.path.isfile(done) and open(done, encoding="utf-8").read().strip() == want \
            and os.path.isfile(os.path.join(paths["pack"], "experts.bin")):
        log("expert pack: already built")
        return
    log("building the expert pack from the model (offline, one time: ~40 GB written; 10-40 min, a lot of "
        "RAM while it runs, so close big programs first)")
    _strata_tool(vpy, paths, ["tools/iq_pack.py", "--gguf",
                              os.path.join(paths["models"], os.path.basename(shard1["path"])), "--out", paths["pack"],
                              "--experts-bin"],
                 "strata-pack.log", root, run, 4 * 3600)
    with open(done, "w", encoding="utf-8") as f:
        f.write(want + "\n")


def strata_plan(root, catalog):
    s = catalog["strata"]
    gb = sum(f["size"] for f in s["model"]["files"]) / 1e9
    mtp_gb = sum(t["bytes"] for t in s["mtp"]["tensors"]) / 1e9
    return ["pick the GPU with the most video memory (12 GB+) and its Strata engine package; check RAM (32 GB+)",
            f"git sparse checkout of Strata's terminal/ folder at commit {s['commit'][:12]} (v{s['version']}, verified)",
            "give it its own Python environment with the hash-locked packages (requirements-strata.txt)",
            f"download the model ({gb:.0f} GB, 2 files; an older pack's identical shard is linked instead) and the "
            "engine package, each pinned and checksum-verified; the engine's signed manifest is checked offline",
            f"download the MTP draft layer ({mtp_gb:.1f} GB: 31 tensors, each pinned) and build it, then build "
            "the expert pack (offline, Strata's own tools; ~42 GB on disk, a lot of RAM while it builds)",
            f"write a launcher for 127.0.0.1:{s['port']} only and add '{s['model_id']}' to llama_cpp_config.json "
            "(Artifex starts and stops it; it uses the whole GPU while it runs)",
            "test it: start it, ask 17*23, stop it"]


def install_strata(root, catalog, log=print, run=subprocess.run, probe_fn=None):
    import sys
    s = catalog["strata"]
    hosts = catalog["allowed_hosts"]
    backend, variant, g, ram_gb = _strata_gpu(probe_fn)
    log(f"GPU for Strata: {g['name']} ({(g.get('vram_mb') or 0) / 1024:.0f} GB) -> engine {variant}")
    if not shutil.which("git"):
        raise PackUnavailable("The Strata pack is installed with git. Get Git for Windows from "
                              "https://git-scm.com/download/win, then run this pack again.")
    cfg_path, cfg = _config(root)
    paths = _strata_paths(root, s)
    need = sum(f["size"] for f in s["model"]["files"]
               if not os.path.isfile(os.path.join(paths["models"], os.path.basename(f["path"]))))
    extra = 54 * (1 << 30)              # experts.bin (~40 GiB), draft layer, dense pack, engine, venv
    free = shutil.disk_usage(root).free
    if free < need + extra:
        raise PackUnavailable(f"not enough disk space: about {(need + extra) / (1 << 30):.0f} GB needed, "
                              f"{free / (1 << 30):.0f} GB free")
    # 1. source: terminal/ only, at the pinned commit
    strata = paths["strata"]
    if not os.path.isdir(os.path.join(strata, ".git")):
        log(f"cloning {s['repo']} (terminal/ only)")
        _git(["clone", "--filter=blob:none", "--no-checkout", "--sparse", s["repo"], strata], run=run)
    else:
        _git(["fetch", "--filter=blob:none", "origin"], cwd=strata, run=run)
    _git(["sparse-checkout", "set", "--cone"] + list(s["sparse"]), cwd=strata, run=run)
    _git(["checkout", "--detach", s["commit"]], cwd=strata, run=run)
    head = _git(["rev-parse", "HEAD"], cwd=strata, run=run)
    if head != s["commit"]:
        raise fetch.VerifyError(f"checked out {head}, expected {s['commit']}")
    log(f"Strata source at commit {head[:12]} (verified)")
    # 2. its venv, hash-locked
    venv = os.path.join(strata, ".venv")
    vpy = os.path.join(venv, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv, "bin", "python")
    if not os.path.isfile(vpy):
        if not recipes.python_ok():
            raise RuntimeError(f"Python {sys.version_info[0]}.{sys.version_info[1]} is not supported for Strata")
        log(f"creating {venv}")
        run([sys.executable, "-m", "venv", venv], check=True)
    log("installing the locked packages (requirements-strata.txt)")
    run([vpy, "-m", "pip", "install", "--isolated", "--require-hashes", "--only-binary=:all:", "--no-input",
         "--disable-pip-version-check", "-r", recipes.LOCKS["strata"]], check=True)
    # 3. pinned downloads, the engine and its signature check
    _strata_models(s, paths, root, hosts, log)
    engine_dir = _strata_engine(s, variant, vpy, paths, hosts, log, run)
    # 4. offline preparation with Strata's own tools
    _strata_mtp(s, vpy, paths, root, hosts, log, run)
    _strata_pack(s, vpy, paths, root, log, run)
    # 5. Artifex's run config, launcher and model entry
    art = _strata_artifex_config(s, paths, engine_dir, backend,
                                 os.path.join(os.path.abspath(root), STRATA_LOG_DIR, "strata-flash-next.log"))
    art_path = os.path.join(strata, "strata-artifex.json")
    with open(art_path, "w", encoding="utf-8") as f:
        json.dump(art, f, indent=1)
    launcher = _strata_launcher(root, strata, vpy, art_path, s["port"], s["tuning"]["max_ctx"])
    models = cfg.setdefault("models", {})
    for mid, e in list(models.items()):
        # the previous pack's entry (IQ2_XS on 0.2.7): its model and launcher are gone
        if mid != s["model_id"] and isinstance(e, dict) and "(pack strata)" in str(e.get("_comment", "")):
            log(f"removing the old Strata entry '{mid}'")
            del models[mid]
    models[s["model_id"]] = {
        "_comment": (f"Qwen3.8-Flash-Next IQ3_XXS on Strata v{s['version']} ({variant}, {g['name']}), "
                     f"127.0.0.1:{s['port']} only; added by python -m installer (pack strata). "
                     "Uses the whole GPU while loaded; context follows Artifex's tiers."),
        "path": os.path.join(paths["models"], os.path.basename(s["model"]["files"][0]["path"])).replace("\\", "/"),
        "port": s["port"], "num_ctx": s["tuning"]["max_ctx"], "health_timeout": 600,
        "external_launcher": ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                              launcher.replace("\\", "/"), "-Context", "{ctx}"]}
    _save_config(cfg_path, cfg)
    log(f"added '{s['model_id']}' to {cfg_path}")
    # 6. test
    log("testing: start Strata, ask 17*23, stop it (the first start takes a minute)")
    reply = _strata_smoke_test(launcher, s["port"], s["model_id"], log=log, run=run)
    log(f"Strata answered: {reply}")
    older = [os.path.join(root, *d.split("/")) for d in s.get("older_models_dest", [])]
    for d in older:
        if os.path.isdir(d):
            log(f"note: {d} (an older pack's model) is no longer used; delete it by hand to free the space")
    return launcher


# ── Holo ────────────────────────────────────────────────────────────────────
# The phone app's Holo tab: a talking hologram avatar. Its voices are Piper
# (four female voices on top of the voice pack's) and Orpheus 3B, streamed
# from its own Vulkan llama.cpp server on 127.0.0.1:8082 (the second card when
# there are two, so the chat model keeps the first). Avatars from any photo:
# a depth map (Depth-Anything-V2) and face points (MediaPipe, or the vision
# helper for drawings) made on the CPU. Everything pinned in catalog.json /
# requirements-holo*.txt.

def holo_plan(root, catalog):
    h = catalog["holo"]
    mb = lambda keys: sum(f["size"] for k in keys for f in h[k]["files"]) // (1 << 20)
    return ["add the hash-locked Holo packages (SNAC decoder, transformers, MediaPipe; "
            "CPU PyTorch only if there is none) to the installer's Python environment",
            f"download 4 female Piper voices ({mb(['piper'])} MB), the Orpheus 3B voice model "
            f"({mb(['orpheus'])} MB), its SNAC decoder, the Depth-Anything-V2 depth model and "
            "MediaPipe's face landmarker (all pinned and checksum-verified) into models/",
            "download the Vulkan llama.cpp build for the Orpheus server if missing, and register "
            "it in llama_cpp_config.json (127.0.0.1:8082 only, started on demand, stopped when idle)",
            "mark the chat models that fit beside it, so the Tara voice stays on while they run",
            "test it: depth + face points on a test picture, then Orpheus speaks a line"]


def _orpheus_device(gpus):
    """Vulkan device for Orpheus: the second card when there are two or more
    (the chat model keeps the first), else the only one."""
    return "Vulkan1" if len(gpus) >= 2 else "Vulkan0"


def _coexists(entry, gpus, voice_mb):
    """Can this chat model run while Orpheus holds its card? Multi-card: a
    model that stays on one card. One card: a model small enough to share."""
    if entry.get("external_launcher"):
        return False
    flags = [str(f) for f in entry.get("extra_flags") or []]
    if len(gpus) >= 2:
        if any(f in ("-ts", "--tensor-split", "-sm", "--split-mode") for f in flags):
            return False
        for i, f in enumerate(flags):              # --device CUDA0,ROCm0 / Vulkan1: not just card 0
            if f in ("--device", "-dev") and i + 1 < len(flags):
                dev = flags[i + 1]
                if "," in dev or not dev.endswith("0"):
                    return False
        return True
    if not gpus:
        return False
    try:
        size_mb = os.path.getsize(entry["path"]) / (1 << 20)
    except (OSError, KeyError, TypeError):
        return False
    return size_mb + 1500 + voice_mb <= (gpus[0].get("vram_mb") or 0)


def install_holo(root, catalog, log=print, run=subprocess.run, probe_fn=None):
    h, hosts = catalog["holo"], catalog["allowed_hosts"]
    path, cfg = _config(root)
    # 1. packages (exactly the lock: --no-deps keeps MediaPipe's unused
    #    opencv-contrib / matplotlib / sounddevice out of the environment)
    py = recipes.add_locked_packages(root, recipes.LOCKS["holo"], ["snac", "mediapipe", "transformers"],
                                     log=log, run=run, no_deps=True)
    probe_torch = "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"
    if run([py, "-c", probe_torch]).returncode != 0:
        log("no PyTorch yet: adding the CPU build (the Create pack would bring CUDA)")
        recipes.add_locked_packages(root, recipes.LOCKS["holo-torch"], ["torch"], log=log, run=run,
                                    no_deps=True)
    # 2. pinned files
    for key in ("piper", "snac", "depth", "orpheus"):
        _fetch_hf_files(h[key], root, hosts, log)
    face = h["face"]
    face_dest = os.path.join(root, face["dest"])
    if not fetch.file_matches(face_dest, face["size"], face["sha256"]):
        log("downloading face_landmarker.task (4 MB)")
    fetch.download(face["url"], face_dest, face["size"], face["sha256"], hosts)
    # 3. the Vulkan llama.cpp build for the Orpheus server
    srv = h["orpheus_server"]
    vk = recipes.plan({"gpus": []}, catalog, root, variant=srv["variant"],
                      model_id=catalog["chat_models"][0]["id"])
    if not os.path.isfile(vk["server_exe"]):
        for f in vk["backend_files"]:
            dest = os.path.join(vk["download_dir"], f["name"])
            log(f"downloading {f['name']} (Vulkan llama.cpp for the Orpheus voice)")
            fetch.download(f["url"], dest, f["size"], f["sha256"], hosts)
            fetch.safe_unzip(dest, vk["server_dir"])
            os.remove(dest)
    # 4. config: the Orpheus block, and which chat models can share the GPUs
    from installer import probe as probe_mod
    gpus = [g for g in ((probe_fn or probe_mod.probe)().get("gpus") or [])
            if g.get("vendor") in ("NVIDIA", "AMD", "Intel")]
    changed = []
    model_file = os.path.join(root, h["orpheus"]["dest"], os.path.basename(h["orpheus"]["files"][0]["path"]))
    if "orpheus" not in cfg:
        cfg["orpheus"] = {
            "_comment": ("Added by the installer's holo pack: the Orpheus voices (tara & co.) of the "
                         f"Holo tab. {h['orpheus']['license']}. Started on first use on 127.0.0.1 only, "
                         "stopped after 10 idle minutes or when a model that needs this card loads. "
                         "device: the Vulkan device index (llama-server --list-devices)."),
            "server_path": vk["server_exe"].replace("\\", "/"),
            "model": model_file.replace("\\", "/"),
            "device": _orpheus_device(gpus), "port": srv["port"], "min_free_mb": srv["min_free_mb"]}
        changed.append("orpheus")
    for mid, entry in (cfg.get("models") or {}).items():
        if isinstance(entry, dict) and "coexists_with_orpheus" not in entry and _coexists(entry, gpus, srv["voice_mb"]):
            entry["coexists_with_orpheus"] = True
            changed.append(mid)
    if changed:
        _save_config(path, cfg)
        log("llama_cpp_config.json: " + ", ".join(changed))
    if not gpus:
        log("no graphics card found: the Orpheus voices need one (Vulkan); the Piper voices work anyway")
    # 5. test: an avatar's depth + face points, then Orpheus says a line
    log("testing: depth + face points on a test picture, then Orpheus speaks")
    test = ("from PIL import Image, ImageDraw\n"
            "from core.services import holo_avatars as ha, orpheus_tts as ot\n"
            "im = Image.new('RGB', (320, 400), (200, 170, 150)); d = ImageDraw.Draw(im)\n"
            "d.ellipse([90, 80, 230, 260], fill=(230, 200, 180))\n"
            "dm = ha._depth(im); lm, src = ha._landmarks_mediapipe(im); lm = lm or ha._landmarks_guess(im)\n"
            "assert dm.size == im.size and lm['mouth']['hw'] > 0\n"
            "print('avatar ok', src or 'guess')\n"
            "try:\n"
            "    n = sum(len(c) for c in ot.stream('The hologram voice works.', 'tara'))\n"
            "    print('orpheus ok', n)\n"
            "except Exception as e:\n"
            "    print('orpheus unavailable:', str(e)[:200])\n"
            "finally:\n"
            "    ot.stop('installer test')\n")
    r = run([py, "-c", test], cwd=root, capture_output=True, text=True, timeout=900)
    out = (r.stdout or "").strip().splitlines()
    if r.returncode != 0 or not any(l.startswith("avatar ok") for l in out):
        raise RuntimeError("holo self-test failed: " + ((r.stderr or r.stdout) or "")[-600:])
    for line in out:
        if line.startswith(("avatar ok", "orpheus")):
            log(line)
    return None      # (a list return is printed as launcher paths)


PACKS = {
    "web-search": {"label": "Web search (Docker: hardened gateway + SearXNG)",
                   "size": "~0.5 GB of Docker images",
                   "plan": web_search_plan, "install": install_web_search},
    "voice": {"label": "Voice: Whisper dictation + Piper read-aloud (CPU)",
              "size": "~0.75 GB",
              "plan": voice_plan, "install": install_voice},
    "vision": {"label": "Vision: image projectors for your chat model + CPU vision helper",
               "size": "~1.2-2.1 GB",
               "plan": vision_plan, "install": install_vision},
    "create": {"label": "Create: images, photo edit/restore, music, 3D, video (NVIDIA)",
               "size": "~3.5 GB of packages + 11.5 GB image model (32 GB download, once)",
               "plan": create_plan, "install": install_create},
    "webgpu": {"label": "Artifex Web: the in-browser WebGPU engine",
               "size": "~0.3 GB of npm packages (+ Node.js if you don't have it)",
               "plan": webgpu_plan, "install": install_webgpu},
    "game": {"label": "A Tale of Yarn: the WebGPU game, with its voices and NPC model",
             "size": "~0.6 GB of npm packages + 1.3 GB of models",
             "plan": game_plan, "install": install_game},
    "strata": {"label": "Qwen3.8-Flash-Next on Strata (one 12 GB+ GPU + 32 GB RAM)",
               "size": "~68 GB model + ~35 GB expert pack + ~6 GB draft layer and engine",
               "plan": strata_plan, "install": install_strata},
    "holo": {"label": "Holo: talking hologram avatars (Orpheus + Piper voices, photo avatars)",
             "size": "~2.9 GB (+0.2 GB CPU PyTorch if you have none)",
             "plan": holo_plan, "install": install_holo},
    "ollama": {"label": "Use an existing Ollama install as a backend",
               "size": "nothing to download",
               "plan": ollama_plan, "install": install_ollama},
}
