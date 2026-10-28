import argparse
import json
import os
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
    a = ap.parse_args(argv)

    if a.cmd == "probe":
        p = probe_mod.probe(REPO_ROOT)
        print(json.dumps(p, indent=2) if a.json else probe_mod.summary(p))
        return 0

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
    except VerifyError as e:
        print(f"\nSTOPPED for safety: {e}")
        return 4
    print("\nDone. Start Artifex and pick "
          f"'{pl['model']['id']}' in the model list.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
