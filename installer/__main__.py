import argparse
import json
import os
import re
import subprocess
import sys

from installer import probe as probe_mod
from installer import recipes
from installer.fetch import VerifyError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bar(got, total, _state={"last": -1}):
    pct = int(got * 100 / total) if total else 100
    if pct != _state["last"]:
        _state["last"] = pct
        sys.stdout.write(f"\r  {pct:3d}%  {got >> 20} / {total >> 20} MB")
        sys.stdout.flush()
        if pct == 100:
            sys.stdout.write("\n")
            _state["last"] = -1


def _describe(pl) -> str:
    g = pl["gpu"]
    lines = [f"Backend: llama.cpp, {pl['variant_label']} ({pl['why']})",
             f"GPU:     {g['name']}" if g else "GPU:     none (CPU)",
             f"Model:   {pl['model']['label']}",
             f"         {pl['model']['repo']} / {pl['model']['file']}",
             f"License: {pl['model']['license']}",
             f"Install: {pl['root']}",
             f"Download at most {pl['download_mb'] / 1024:.1f} GB "
             "(every file is checked against a SHA-256 pinned in installer/catalog.json)"]
    return "\n".join(lines)


def _remote(a) -> int:
    from installer import remote
    catalog = recipes.load_catalog()
    root = os.path.abspath(a.root)
    min_v = catalog["tailscale"]["min_version"]
    if a.action == "setup":
        if a.model and not re.fullmatch(r"[A-Za-z0-9._-]+", a.model):
            print(f"error: bad model id {a.model!r}")
            return 2
        steps = []
        if not remote.tailscale_exe():
            steps.append(f"install Tailscale {catalog['tailscale']['version']} (pinned and "
                         "signature-checked; Windows will ask you to approve it), then you sign in")
        steps += [f"create {remote.KEY_FILE} (a random 256-bit API key) if it doesn't exist",
                  f"forward your tailnet's port {a.port} to 127.0.0.1:{a.port} "
                  "(tailscale serve, tailnet only; never Funnel)",
                  "write 'Artifex API (phone via Tailscale).cmd' (still bound to 127.0.0.1)",
                  "run the exposure check"]
        print("Remote access setup will:\n" + "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(steps)))
        if not a.yes and input("\nGo ahead? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Nothing changed.")
            return 1
        try:
            if not remote.tailscale_exe():
                remote.install_tailscale(catalog, os.path.join(root, "runtime", "downloads"))
            if not remote.wait_connected():
                print("Tailscale isn't connected yet; sign in, then run this again.")
                return 5
            print("API key: " + remote.ensure_api_key(root))
            remote.serve_api(a.port)
            print("launcher: " + recipes.write_phone_launcher(root, a.port, a.model))
        except (VerifyError, RuntimeError, ValueError, subprocess.CalledProcessError) as e:
            print(f"\nSTOPPED: {e}")
            return 4
    findings = remote.check_exposure(root, a.port, min_version=min_v)
    print("\nExposure check:\n" + remote.report(findings))
    return 3 if any(f["level"] == remote.FAIL for f in findings) else 0


def _packs(a) -> int:
    from installer import packs, remote
    catalog = recipes.load_catalog()
    if a.cmd == "packs":
        for pid, p in packs.PACKS.items():
            print(f"  {pid:12} {p['label']}  ({p['size']})")
        print("\nInstall one with: python -m installer pack <id>")
        return 0
    if a.id not in packs.PACKS:
        print(f"error: no pack {a.id!r}; see python -m installer packs")
        return 2
    p, root = packs.PACKS[a.id], os.path.abspath(a.root)
    print(f"{p['label']} ({p['size']}) will:\n" + "\n".join(
        f"  {i + 1}. {s}" for i, s in enumerate(p["plan"](root, catalog))))
    if not a.yes and input("\nGo ahead? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Nothing changed.")
        return 1
    try:
        result = p["install"](root, catalog)
    except packs.PackUnavailable as e:
        print(f"\nNot installed: {e}")
        return 6
    except (VerifyError, RuntimeError, ValueError, subprocess.CalledProcessError) as e:
        print(f"\nSTOPPED: {e}")
        return 4
    if isinstance(result, list):
        for path in result:
            print(f"launcher: {path}")
    findings = remote.check_exposure(root, min_version=catalog["tailscale"]["min_version"])
    bad = [f for f in findings if f["level"] == remote.FAIL]
    print("\nDone." if not bad else "\nInstalled, but the exposure check FAILED:\n" + remote.report(bad))
    return 3 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m installer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("probe", help="show what this PC has")
    pr.add_argument("--json", action="store_true")
    for name in ("plan", "install"):
        s = sub.add_parser(name)
        s.add_argument("--root", default=REPO_ROOT, help="install root (default: this Artifex folder)")
        s.add_argument("--model", help="catalog model id (default: the best that fits)")
        s.add_argument("--variant", help="llama.cpp build: cuda-13.4, cuda-12.4, vulkan, cpu")
        if name == "install":
            s.add_argument("--reuse", action="append", default=[],
                           help="folder to look in for model files you already have")
            s.add_argument("--force", action="store_true",
                           help="replace an existing config entry with the same id")
            s.add_argument("--yes", action="store_true", help="don't ask before downloading")
            s.add_argument("--no-env", action="store_true",
                           help="skip the Python environment (backend and model only)")
            s.add_argument("--no-gui", action="store_true",
                           help="leave out the desktop GUI (PyQt6, ~100 MB)")
    rm = sub.add_parser("remote", help="phone access over Tailscale, and the exposure check")
    rm.add_argument("action", choices=["check", "setup"])
    rm.add_argument("--root", default=REPO_ROOT)
    rm.add_argument("--port", type=int, default=8000, help="Artifex API port (default 8000)")
    rm.add_argument("--model", help="model id the phone launcher starts with")
    rm.add_argument("--yes", action="store_true", help="don't ask before changing anything")
    sub.add_parser("packs", help="list the optional packs")
    pk = sub.add_parser("pack", help="install an optional pack")
    pk.add_argument("id")
    pk.add_argument("--root", default=REPO_ROOT)
    pk.add_argument("--yes", action="store_true", help="don't ask first")
    a = ap.parse_args(argv)

    if a.cmd in ("packs", "pack"):
        return _packs(a)
    if a.cmd == "probe":
        p = probe_mod.probe(REPO_ROOT)
        print(json.dumps(p, indent=2) if a.json else probe_mod.summary(p))
        return 0
    if a.cmd == "remote":
        return _remote(a)

    catalog = recipes.load_catalog()
    p = probe_mod.probe(a.root)
    try:
        pl = recipes.plan(p, catalog, a.root, model_id=a.model, variant=a.variant)
    except ValueError as e:
        print(f"error: {e}")
        return 2
    print(probe_mod.summary(p) + "\n\n" + _describe(pl))
    need_mb = pl["download_mb"] * 2 + 1024        # zips unpack next to themselves
    if p["free_disk_mb"] and p["free_disk_mb"] < need_mb:
        print(f"\nNot enough disk space: about {need_mb / 1024:.1f} GB needed, "
              f"{p['free_disk_mb'] / 1024:.1f} GB free.")
        return 3
    if a.cmd == "plan":
        return 0
    if not a.yes and input("\nInstall this? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Nothing installed.")
        return 1
    try:
        recipes.install_core(pl, catalog, reuse_dirs=a.reuse, force=a.force, progress=_bar)
        if not a.no_env:
            recipes.install_env(pl["root"], gui=not a.no_gui)
            for path in recipes.write_launchers(pl["root"], pl["model"]["id"], gui=not a.no_gui):
                print(f"launcher: {path}")
    except VerifyError as e:
        print(f"\nSTOPPED for safety: {e}")
        return 4
    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"\nSTOPPED: {e}")
        return 5
    print("\nDone. Start Artifex and pick "
          f"'{pl['model']['id']}' in the model list.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
