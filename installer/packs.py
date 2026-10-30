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
                             "none yet (ollama pull <model>, e.g. ollama pull lfm3)"))
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
    with open(path, encoding="utf-8") as f:
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
            "download LFM3-VL 2B + projector (1.2 GB, pinned) and the CPU llama.cpp build if "
            "missing, and register the CPU vision helper (port 8095, this PC only)",
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
    if not os.path.isfile(cpu_plan["server_exe"]):
        for f in cpu_plan["backend_files"]:
            dest = os.path.join(cpu_plan["download_dir"], f["name"])
            log(f"downloading {f['name']} (CPU llama.cpp for the helper)")
            fetch.download(f["url"], dest, f["size"], f["sha256"], hosts)
            fetch.safe_unzip(dest, cpu_plan["server_dir"])
            os.remove(dest)
    model = _hf_file(vh["model"], root, hosts, log)
    mmproj = _hf_file(vh["mmproj"], root, hosts, log)
    if "vision_helper" not in cfg:
        cfg["vision_helper"] = {
            "_comment": f"Added by the installer's vision pack. {vh['license']}.",
            "path": model, "server_path": cpu_plan["server_exe"].replace("\\", "/"),
            "num_gpu_layers": 0, "num_ctx": vh["num_ctx"], "port": vh["port"],
            "idle_unload_s": vh["idle_unload_s"], "assist_tools": ["edit", "image"],
            "extra_flags": ["--mmproj", mmproj, "--no-mmproj-offload", "-t",
                            str(max(2, (os.cpu_count() or 4) - 2)), "-np", "1", "--jinja"]}
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
# from GPU + RAM + SSD. Its own setup.py does the model preparation (expert
# pack, MTP draft layer); this pack feeds it only pinned inputs and keeps it
# offline where it can be:
#   - the source: git clone over https, checked out at the pinned commit;
#   - Python: its venv gets OUR hash-locked packages (requirements-strata.txt),
#     and setup's stamp file is written so setup's own unhashed pip step skips;
#   - the engine: the pinned v0.2.7 signed feed + the pinned package for this
#     GPU family in a local folder (--feed DIR); Strata checks the Ed25519
#     signature and the package's size + SHA-256 on top of ours;
#   - the model shards and the llama.cpp source: pinned size + SHA-256;
#   - the MTP draft layer: Strata's tools/mtp_fetch.py, which checks every
#     tensor against the SHA-256 pinned in that commit; verified again after.
# The server binds 127.0.0.1 only and Artifex starts/stops it (external_launcher).

STRATA_SETUP_LOG = os.path.join("logs", "strata-setup.log")


def _strata_gpu(probe_fn=None):
    """(backend, engine variant, gpu, ram_gb) for the card Strata should use: the one with the most VRAM."""
    from installer import probe as probe_mod
    p = (probe_fn or probe_mod.probe)()
    ram_gb = (p.get("ram_mb") or 0) / 1024
    gpus = [g for g in p.get("gpus") or [] if g.get("vendor") in ("NVIDIA", "AMD")]
    if not gpus:
        raise PackUnavailable("Strata needs an NVIDIA (RTX 20 or newer) or AMD Radeon RX 6800/6700 XT, "
                              "RX 7000 or RX 9000 graphics card.")
    g = max(gpus, key=lambda x: x.get("vram_mb") or 0)
    if (g.get("vram_mb") or 0) < 11000:
        raise PackUnavailable(f"Strata needs a card with 12 GB of video memory or more ({g['name']} has "
                              f"{(g.get('vram_mb') or 0) / 1024:.0f} GB).")
    if ram_gb and ram_gb < 30:
        raise PackUnavailable(f"Flash-Next IQ2_XS needs 32 GB of RAM or more (this PC has {ram_gb:.0f} GB).")
    name = g["name"].upper()
    if g["vendor"] == "NVIDIA":
        cc = recipes._version(g.get("compute_cap", ""))
        drv = recipes._version((p.get("nvidia") or {}).get("driver", ""))
        if cc and cc < (7, 5):
            raise PackUnavailable(f"{g['name']} is older than RTX 20 (compute capability {g.get('compute_cap')}); "
                                  "Strata's ready-made NVIDIA engines start at sm_75.")
        if drv >= (580,):
            return "cuda", "cuda-cuda13", g, ram_gb
        if drv >= (528,):
            return "cuda", "cuda-cuda12", g, ram_gb
        raise PackUnavailable("Strata's NVIDIA engine needs driver 528 or newer (580+ for CUDA 13). Update the driver.")
    if re.search(r"RX 6(7|8|9)\d\d", name):
        return "hip", "hip-rdna2", g, ram_gb
    if re.search(r"RX 7\d\d\d", name):
        return "hip", "hip-rdna3", g, ram_gb
    if re.search(r"RX 9\d\d\d|R9700", name):
        return "hip", "hip-rx9070", g, ram_gb
    raise PackUnavailable(f"{g['name']}: Strata has no ready-made Windows engine for this AMD card "
                          "(RX 6700 XT/6800/6900, RX 7000 and RX 9000 are supported).")


def _strata_runtime_packages(strata_dir):
    """The package list Strata's setup.py installs (setup.runtime_packages: requirements.txt +
    requirements-monitor.txt, comments dropped), so its stamp can say they are installed."""
    out = []
    for name in ("requirements.txt", "requirements-monitor.txt"):
        path = os.path.join(strata_dir, name)
        if not os.path.isfile(path):
            continue
        for raw in open(path, encoding="utf-8").read().splitlines():
            line = raw.split("#", 1)[0].strip()
            if line and line not in out:
                out.append(line)
    return out


def _strata_artifex_config(setup_cfg, s, ram_gb, backend):
    """Artifex's run config from setup's: loopback only, the tuned settings, and in the
    low-RAM mapped mode a resident RAM budget that leaves ~19.5 GiB for everything else."""
    t = s["tuning"]
    args = list(setup_cfg["args"])

    def put(flag, value=None):
        if flag in args:
            i = args.index(flag)
            if value is not None:
                args[i + 1] = str(value)
        else:
            args.append(flag)
            if value is not None:
                args.append(str(value))
    if "--mmap-experts" in args:
        args.remove("--mmap-experts")
        put("--resident-experts")
        put("--resident-budget-gib", f"{max(t['min_resident_budget_gib'], ram_gb - t['ram_left_for_others_gib']):g}")
    put("--spec", t["spec"])
    put("--vram-reserve-mib", t["vram_reserve_mib"])
    put("--conversation-cache-mib", t["conversation_cache_mib"])
    put("--max-context", t["default_ctx"])
    env = dict(setup_cfg.get("env") or {})
    env["STRATA_ROUTE_PRIOR"] = t["route_prior"]
    if backend == "hip":
        env["STRATA_HIP_PROMPT_F16"] = "1"          # Strata's own tip for RX 6000 prompt speed
    cfg = dict(setup_cfg, args=args, env=env, host="127.0.0.1", port=s["port"])
    return cfg


def _strata_launcher(root, strata_dir, vpy, cfg_path, port, max_ctx):
    path = os.path.join(os.path.abspath(root), "strata", "start-strata-artifex.ps1")
    q = lambda p: os.path.abspath(p).replace("'", "''")
    text = f"""# Strata (Qwen3.8-Flash-Next) for Artifex: 127.0.0.1:{port} ONLY. Written by python -m installer
# (pack strata); Artifex starts it through its external_launcher entry and passes -Context <tier>.
param([int]$Context = 32768)
$ErrorActionPreference = 'Stop'
$strata = '{q(strata_dir)}'
$Context = [Math]::Max(4096, [Math]::Min($Context, {max_ctx}))
$cfg = Get-Content '{q(cfg_path)}' -Raw | ConvertFrom-Json
$a = [System.Collections.ArrayList]@($cfg.args)
$i = $a.IndexOf('--max-context')
if ($i -ge 0) {{ $a[$i + 1] = "$Context" }} else {{ [void]$a.Add('--max-context'); [void]$a.Add("$Context") }}
$cfg.args = @($a)
$cfg.host = '127.0.0.1'
$run = Join-Path $strata 'strata-artifex-run.json'
$cfg | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $run
$env:PYTHONUTF8 = '1'
foreach ($p in $cfg.env.PSObject.Properties) {{ Set-Item -Path ("env:" + $p.Name) -Value ([string]$p.Value) }}
Set-Location $strata
& '{q(vpy)}' -u (Join-Path $strata 'serve\\server.py') --engine strata --config $run --host 127.0.0.1 --port {port}
"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\r\n") as f:
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
    proc = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", launcher,
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


def strata_plan(root, catalog):
    s = catalog["strata"]
    gb = sum(f["size"] for f in s["model"]["files"]) / 1e9
    return ["pick the GPU with the most video memory (12 GB+) and its Strata engine package; check RAM (32 GB+)",
            f"git clone {s['repo']} over https and check out commit {s['commit'][:12]} (v{s['version']}, verified)",
            "give it its own Python environment with the hash-locked packages (requirements-strata.txt)",
            f"download the model ({gb:.0f} GB, 2 files) and the engine package, each pinned and checksum-verified",
            "let Strata prepare the model offline (expert pack ~35 GB; the MTP draft layer ~5 GB, every tensor "
            "checked against Strata's pinned SHA-256)",
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
    strata = os.path.join(root, *s["dest"].split("/"))
    models = os.path.join(root, *s["models_dest"].split("/"))
    data = os.path.join(root, *s["data_dest"].split("/"))
    need = sum(f["size"] for f in s["model"]["files"]
               if not os.path.isfile(os.path.join(models, os.path.basename(f["path"]))))
    free = shutil.disk_usage(root).free
    if free < need + 45 * (1 << 30):
        raise PackUnavailable(f"not enough disk space: about {(need + 45 * (1 << 30)) / (1 << 30):.0f} GB "
                              f"needed (model + expert pack + draft layer), {free / (1 << 30):.0f} GB free")
    # 1. source at the pinned commit
    if not os.path.isdir(os.path.join(strata, ".git")):
        log(f"cloning {s['repo']}")
        _git(["clone", "--no-checkout", s["repo"], strata], run=run)
    else:
        _git(["fetch", "origin"], cwd=strata, run=run)
    _git(["checkout", "--detach", s["commit"]], cwd=strata, run=run)
    head = _git(["rev-parse", "HEAD"], cwd=strata, run=run)
    if head != s["commit"]:
        raise fetch.VerifyError(f"checked out {head}, expected {s['commit']}")
    log(f"Strata source at commit {head[:12]} (verified)")
    # 2. its venv, hash-locked; setup's own pip step is told it is done
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
    with open(os.path.join(venv, ".strata-pip.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(_strata_runtime_packages(strata)), f, indent=0)
    with open(os.path.join(venv, ".strata-pip.json.os"), "w", encoding="utf-8") as f:
        f.write(sys.platform)
    # 3. pinned downloads: model shards, llama.cpp source, the signed feed + this GPU's engine package
    for f in s["model"]["files"]:
        dest = os.path.join(models, os.path.basename(f["path"]))
        if not fetch.file_matches(dest, f["size"], f["sha256"]):
            log(f"downloading {os.path.basename(f['path'])} ({f['size'] / 1e9:.1f} GB)")
        fetch.download(f"https://huggingface.co/{s['model']['repo']}/resolve/{s['model']['revision']}/{f['path']}",
                       dest, f["size"], f["sha256"], hosts)
    lc = s["llama_cpp_source"]
    lz = os.path.join(strata, *lc["dest_in_strata"].split("/"))
    fetch.download(lc["url"], lz, lc["size"], lc["sha256"], hosts)
    with open(lz + ".done", "w", encoding="utf-8") as f:     # setup's finish mark: it uses this file, no download
        f.write("pinned by python -m installer\n")
    feed_dir = os.path.join(root, "strata", f"feed-v{s['version']}")
    for f in s["feed"] + [s["engines"][variant]]:
        fetch.download(f["url"], os.path.join(feed_dir, f["url"].rsplit("/", 1)[-1]), f["size"], f["sha256"], hosts)
    log(f"engine package {s['engines'][variant]['url'].rsplit('/', 1)[-1]} and the signed feed: verified")
    # 4. Strata's own preparation (expert pack, MTP draft layer), from the pinned inputs above
    # STRATA_UPDATE_FEED: the engine comes from the pinned local feed above, never the live one (setup's
    # `strata update` reads this; --feed alone only covers DeepSeek's engine).
    env = dict(os.environ, PYTHONUTF8="1", STRATA_UPDATE_FEED=feed_dir)
    if backend == "hip":            # the AMD card even when an NVIDIA card is present (#3892)
        env["STRATA_ENGINE_BACKEND"] = "hip"
        env["STRATA_FEED_BUILD"] = variant
    cmd = [vpy, "-u", "setup.py", "--backend", backend, "--family", "qwen", "--model", "IQ2_XS",
           "--gguf-dir", models, "--data-dir", data, "--feed", feed_dir, "--vision", "no", "--images", "no",
           "--host", "127.0.0.1", "--no-start", "--no-browser", "--yes"]
    if backend == "cuda":
        cmd += ["--cuda", "13" if variant == "cuda-cuda13" else "12"]
    log_path = os.path.join(root, STRATA_SETUP_LOG)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log(f"Strata setup: preparing the model (20-60 min; log: {log_path})")
    with open(log_path, "w", encoding="utf-8") as lf:
        r = run(cmd, cwd=strata, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=4 * 3600)
    if r.returncode != 0:
        raise RuntimeError(f"Strata setup failed (exit {r.returncode}); see {log_path}")
    build = json.load(open(os.path.join(strata, "engine", "BUILD.json"), encoding="utf-8-sig"))
    if build.get("version") != s["engine_version"]:
        raise fetch.VerifyError(f"Strata installed engine {build.get('version')}, not the pinned "
                                f"{s['engine_version']}; delete {os.path.join(strata, 'engine')} and run the pack again")
    log(f"engine {build['version']} ({build.get('backend')}) installed from the pinned feed")
    r = run([vpy, os.path.join(strata, "tools", "mtp_fetch.py"), "verify", "--out", os.path.join(data, "mtp")],
            cwd=strata, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise fetch.VerifyError("the MTP draft layer did not match Strata's pinned SHA-256: "
                                + (r.stdout or r.stderr)[-400:])
    log("MTP draft layer: every tensor matches the pinned SHA-256")
    # 5. Artifex's run config, launcher and model entry
    setup_cfg_path = os.path.join(strata, "strata-iq2_xs.json")
    if not os.path.isfile(setup_cfg_path):
        raise RuntimeError(f"Strata setup wrote no {setup_cfg_path}")
    setup_cfg = json.load(open(setup_cfg_path, encoding="utf-8-sig"))
    art = _strata_artifex_config(setup_cfg, s, ram_gb, backend)
    art["log"] = os.path.join(os.path.abspath(root), "logs", "strata-flash-next.log")
    art_path = os.path.join(strata, "strata-iq2_xs-artifex.json")
    with open(art_path, "w", encoding="utf-8") as f:
        json.dump(art, f, indent=1)
    launcher = _strata_launcher(root, strata, vpy, art_path, s["port"], s["tuning"]["max_ctx"])
    entry = {"_comment": (f"Qwen3.8-Flash-Next IQ2_XS on Strata v{s['version']} ({variant}, {g['name']}), "
                          f"127.0.0.1:{s['port']} only; added by python -m installer (pack strata). "
                          "Uses the whole GPU while loaded; context follows Artifex's tiers."),
             "path": os.path.join(models, os.path.basename(s["model"]["files"][0]["path"])).replace("\\", "/"),
             "port": s["port"], "num_ctx": s["tuning"]["max_ctx"], "health_timeout": 600,
             "external_launcher": ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                   launcher.replace("\\", "/"), "-Context", "{ctx}"]}
    cfg.setdefault("models", {})[s["model_id"]] = entry
    _save_config(cfg_path, cfg)
    log(f"added '{s['model_id']}' to {cfg_path}")
    # 6. test
    log("testing: start Strata, ask 17*23, stop it (the first start takes a minute)")
    reply = _strata_smoke_test(launcher, s["port"], s["model_id"], log=log, run=run)
    log(f"Strata answered: {reply}")
    return launcher


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
    "ollama": {"label": "Use an existing Ollama install as a backend",
               "size": "nothing to download",
               "plan": ollama_plan, "install": install_ollama},
}
