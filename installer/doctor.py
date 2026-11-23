"""python -m installer doctor: what this PC still needs, read-only.

The things a first install otherwise finds out the hard way (a llama-server
that won't start for want of the VC++ runtime, web search that silently does
nothing without Docker, a pack that needs Git), plus the state of this
folder's install. Nothing is changed or downloaded; each finding says what
to do about it.
"""

import json
import os
import shutil
import urllib.request

from installer import probe as probe_mod
from installer import recipes

OK, WARN, FAIL, INFO = "OK", "WARN", "FAIL", "INFO"


def _f(level, title, fix=""):
    return {"level": level, "title": title, "fix": fix}


def vc_runtime_installed() -> bool:
    """The Microsoft Visual C++ 2015-2022 runtime (x64): llama.cpp's Windows
    builds need it, and a fresh Windows may not have it."""
    if os.name != "nt":
        return True
    try:
        import winreg
        for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                    r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
                                    0, winreg.KEY_READ | view) as k:
                    if winreg.QueryValueEx(k, "Installed")[0] == 1:
                        return True
            except OSError:
                continue
    except ImportError:
        return True
    root = os.environ.get("SystemRoot") or r"C:\Windows"
    return os.path.isfile(os.path.join(root, "System32", "vcruntime140_1.dll"))


def _gateway_ok(url=recipes.GATEWAY_URL, timeout=3) -> bool:
    try:
        with urllib.request.urlopen(url + "/health", timeout=timeout) as r:
            return r.status == 200 and b"connected" in r.read()
    except OSError:
        return False


def check(root: str, p: dict = None, catalog: dict = None, gateway=_gateway_ok,
          vc=vc_runtime_installed, which=shutil.which) -> list:
    root = os.path.abspath(root)
    p = p or probe_mod.probe(root)
    catalog = catalog or recipes.load_catalog()
    out = []

    # The PC
    if vc():
        out.append(_f(OK, "Microsoft Visual C++ runtime (x64) is installed"))
    else:
        out.append(_f(FAIL, "The Microsoft Visual C++ runtime (x64) is missing; llama.cpp won't start without it",
                      "install \"Microsoft Visual C++ Redistributable 2015-2022 (x64)\" from Microsoft's site"))
    v, gpu, why = recipes.choose_variant(p, catalog)
    out.append(_f(OK if gpu else INFO, f"Backend: {catalog['llama_cpp']['variants'][v]['label']} ({why})"))
    nv = p.get("nvidia") or {}
    if any(g.get("vendor") == "NVIDIA" for g in p.get("gpus") or []) and not v.startswith("cuda"):
        out.append(_f(WARN, f"The NVIDIA driver only supports CUDA {nv.get('cuda_version') or '?'}; "
                            "Artifex falls back to the slower Vulkan build",
                      "update the NVIDIA driver (GeForce Experience / nvidia.com), then run the install again"))
    ram = p.get("ram_mb", 0)
    if ram and ram < 8192:
        out.append(_f(WARN, f"{ram / 1024:.1f} GB of RAM: only the smallest model will run, slowly"))
    for w in recipes.path_warnings(root):
        out.append(_f(WARN, w))
    if p.get("free_disk_mb", 0) and p["free_disk_mb"] < 10 * 1024:
        out.append(_f(WARN, f"Only {p['free_disk_mb'] / 1024:.1f} GB free on the install drive",
                      "a chat model needs 1-16 GB; free some space or use --models-dir on another drive"))
    if not recipes.python_ok():
        out.append(_f(WARN, "This Python is outside 3.11-3.13", "use setup.cmd: it brings a pinned portable Python"))

    # Optional extras and what they need
    if not p.get("docker"):
        out.append(_f(INFO, "Docker Desktop is not installed: web search in chat is off",
                      "install Docker Desktop yourself, then: setup pack web-search"))
    elif gateway():
        out.append(_f(OK, "Web search gateway is running (127.0.0.1:8080)"))
    else:
        out.append(_f(INFO, "Docker is installed but the web search gateway isn't answering",
                      "start Docker Desktop (the containers come back on their own), or: setup pack web-search"))
    if not which("git"):
        out.append(_f(INFO, "Git is not installed: the strata and game packs need it",
                      "install Git for Windows yourself if you want those packs"))

    # This folder's install
    if not recipes.is_installed(root):
        out.append(_f(INFO, "Artifex isn't installed in this folder yet", "run setup (or: python -m installer install)"))
        return out
    if not _installer_made(root):
        out.append(_f(INFO, "This folder was set up by hand, not by the installer: "
                            "its models and packages are yours to manage (install checks skipped)"))
        return out
    try:
        pl = recipes.plan(p, catalog, root, *(_rerun_model_args(root, catalog)))
        missing, _ = recipes.missing_mb(pl)
        if missing:
            out.append(_f(WARN, f"The install is incomplete or behind the catalog ({missing / 1024:.1f} GB to fetch)",
                          "menu option 6: Run the install again"))
        else:
            out.append(_f(OK, f"llama.cpp {catalog['llama_cpp']['tag']} and the model files are in place"))
    except ValueError as e:
        out.append(_f(WARN, f"Couldn't check the install against the catalog: {e}"))
    venv = os.path.join(root, "venv")
    if os.path.isfile(os.path.join(venv, recipes.INSTALLER_MARKER)):
        had = recipes._marker_locks(venv)
        if had and had == recipes._lock_digest(list(had)):
            out.append(_f(OK, "Python packages match the committed locks"))
        else:
            out.append(_f(WARN, "Python packages are behind the committed locks (security fixes may be waiting)",
                          "menu option 6: Run the install again"))
    try:
        with open(os.path.join(root, "llama_cpp_config.json"), encoding="utf-8-sig") as f:
            models = json.load(f).get("models") or {}
        gone = [mid for mid, e in models.items() if isinstance(e, dict) and e.get("path")
                and not e.get("external_launcher") and not os.path.isfile(e["path"])]
        if gone:
            out.append(_f(WARN, f"Model file missing for: {', '.join(gone)}",
                          "run the install again, or remove those entries from llama_cpp_config.json"))
    except (OSError, ValueError, AttributeError):
        pass
    return out


def _installer_made(root) -> bool:
    """installer-state.json, or a config entry the installer wrote."""
    if os.path.isfile(os.path.join(root, recipes.STATE_FILE)):
        return True
    try:
        with open(os.path.join(root, "llama_cpp_config.json"), encoding="utf-8-sig") as f:
            models = json.load(f).get("models") or {}
    except (OSError, ValueError, AttributeError):
        return False
    return any(isinstance(e, dict) and str(e.get("_comment", "")).startswith("Added by the Artifex installer")
               for e in models.values())


def _rerun_model_args(root, catalog):
    """plan() arguments that match this folder's install (see recipes.rerun_args)."""
    args = recipes.rerun_args(root, catalog)
    kw = {"model_id": [], "variant": None, "profile_id": None, "models_dir": ""}
    i = 0
    while i < len(args):
        k, v = args[i], args[i + 1]
        if k == "--model":
            kw["model_id"].append(v)
        elif k == "--variant":
            kw["variant"] = v
        elif k == "--profile":
            kw["profile_id"] = v
        elif k == "--models-dir":
            kw["models_dir"] = v
        i += 2
    return (kw["model_id"] or None, kw["variant"], kw["profile_id"], kw["models_dir"])


def report(findings: list) -> str:
    marks = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]", INFO: "[info]"}
    lines = []
    for x in findings:
        lines.append(f"{marks[x['level']]} {x['title']}")
        if x["fix"]:
            lines.append(f"         fix: {x['fix']}")
    return "\n".join(lines)
