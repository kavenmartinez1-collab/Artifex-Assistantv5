import argparse
import json
import os
import re
import subprocess
import sys
import time

from installer import probe as probe_mod
from installer import hint, recipes
from installer.fetch import VerifyError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bar(got, total, _state={"last": -1, "t0": 0.0, "g0": 0, "prev": 0}, clock=time.monotonic):
    pct = int(got * 100 / total) if total else 100
    if _state["last"] == -1 or got < _state["prev"]:   # a new file (or one starting over)
        _state.update(t0=clock(), g0=got, last=-1)
    _state["prev"] = got
    if pct != _state["last"]:
        _state["last"] = pct
        took = clock() - _state["t0"]
        rate = (got - _state["g0"]) / took if took > 1 else 0
        eta = ""
        if rate > 0 and got < total:
            left = int((total - got) / rate)
            eta = (f"  {rate / (1 << 20):.1f} MB/s, about "
                   + (f"{left // 60} min left" if left >= 90 else f"{left} s left"))
        sys.stdout.write(f"\r  {pct:3d}%  {got >> 20} / {total >> 20} MB{eta}".ljust(78))
        sys.stdout.flush()
        if pct == 100:
            sys.stdout.write("\n")
            _state["last"] = -1


def _ask(question: str, default: bool = False) -> bool:
    """A yes/no question. Just pressing Enter gives `default`; a closed or
    non-interactive input counts as no, never as a crash or a silent yes."""
    try:
        answer = input(f"{question} {'[Y/n]' if default else '[y/N]'} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return default if not answer else answer in ("y", "yes")


def _describe(pl) -> str:
    g = pl["gpu"]
    lines = [f"Backend: llama.cpp, {pl['variant_label']} ({pl['why']})",
             f"GPU:     {g['name']}" if g else "GPU:     none (CPU)"]
    if pl.get("profile"):
        lines.append(f"Profile: {pl['profile']}")
    for i, item in enumerate(pl["models"]):
        m = item["model"]
        tag = " (default)" if len(pl["models"]) > 1 and i == 0 else ""
        lines += [f"Model:   {m['label']}{tag}",
                  f"         {m['repo']} / {m['file']} ({m['size'] / (1 << 30):.1f} GB)",
                  f"License: {m['license']}"]
    lines += [f"Install: {pl['root']}",
              f"Download at most {pl['download_mb'] / 1024:.1f} GB "
              "(every file is checked against a SHA-256 pinned in installer/catalog.json)"]
    lines += [f"WARNING: {w}" for w in pl.get("warnings") or []]
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
        steps += [f"create {remote.KEYS_FILE} with a random 256-bit API key, stored as a "
                  f"hash, if there is none (an old {remote.KEY_FILE} is moved in, same key)",
                  f"forward your tailnet's port {a.port} to 127.0.0.1:{a.port} "
                  "(tailscale serve, tailnet only; never Funnel)",
                  "write 'Artifex API (phone via Tailscale).cmd' (still bound to 127.0.0.1)",
                  "run the exposure check"]
        print("Remote access setup will:\n" + "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(steps)))
        if not a.yes and not _ask("\nGo ahead?"):
            print("Nothing changed.")
            return 1
        try:
            if not remote.tailscale_exe():
                remote.install_tailscale(catalog, os.path.join(root, "runtime", "downloads"))
            if not remote.wait_connected():
                print("Tailscale isn't connected yet; sign in, then run this again.")
                return 5
            kpath, new_key, note = remote.ensure_api_keys(root)
            print(f"API keys: {kpath}" + (f" - {note}" if note else ""))
            if new_key:
                print("\nYour phone's API key (shown ONCE; only its hash is kept):\n\n"
                      f"    {new_key}\n\nPaste it into the phone app's settings. Lost it? "
                      "Regenerate it from another device's settings, or run\n"
                      f'    python -m core.api_keys --file "{kpath}" rotate phone\n')
            remote.serve_api(a.port)
            print("launcher: " + recipes.write_phone_launcher(root, a.port, a.model))
        except (VerifyError, RuntimeError, ValueError, subprocess.CalledProcessError) as e:
            print(f"\nSTOPPED: {e}")
            return 4
        except OSError as e:
            print(_NETWORK.format(e=e))
            return 5
    if a.action == "off":
        if not remote.tailscale_exe() or not remote.forward_is_on(a.port):
            print(f"Phone access is already off: no Tailscale forward for port {a.port}.")
            return 0
        if not a.yes and not _ask(f"Turn off the Tailscale forward to port {a.port} "
                                  "(the phone can't reach Artifex until you run setup again)?"):
            print("Nothing changed.")
            return 1
        try:
            remote.serve_off(a.port)
        except (RuntimeError, ValueError) as e:
            print(f"\nSTOPPED: {e}")
            return 4
        print("Phone access is off. Your API key file is kept; delete artifex_api_keys.json "
              "too if you won't turn it back on.")
    findings = remote.check_exposure(root, a.port, min_version=min_v)
    print("\nExposure check:\n" + remote.report(findings))
    return 3 if any(f["level"] == remote.FAIL for f in findings) else 0


def _packs(a) -> int:
    from installer import packs, remote
    catalog = recipes.load_catalog()
    if a.cmd == "packs":
        for pid, p in packs.PACKS.items():
            print(f"  {pid:12} {p['label']}  ({p['size']})")
        print("\nInstall one with: " + hint("pack <id>"))
        return 0
    if a.id not in packs.PACKS:
        print(f"error: no pack {a.id!r}; see {hint('packs')}")
        return 2
    p, root = packs.PACKS[a.id], os.path.abspath(a.root)
    print(f"{p['label']} ({p['size']}) will:\n" + "\n".join(
        f"  {i + 1}. {s}" for i, s in enumerate(p["plan"](root, catalog))))
    if not a.yes and not _ask("\nGo ahead?"):
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
    except OSError as e:
        print(_NETWORK.format(e=e))
        return 5
    if isinstance(result, list):
        for path in result:
            print(f"launcher: {path}")
    findings = remote.check_exposure(root, min_version=catalog["tailscale"]["min_version"])
    bad = [f for f in findings if f["level"] == remote.FAIL]
    print("\nDone." if not bad else "\nInstalled, but the exposure check FAILED:\n" + remote.report(bad))
    return 3 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=hint("").strip())
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("probe", help="show what this PC has")
    pr.add_argument("--json", action="store_true")
    for name in ("plan", "install"):
        s = sub.add_parser(name)
        s.add_argument("--root", default=REPO_ROOT, help="install root (default: this Artifex folder)")
        s.add_argument("--model", action="append",
                       help="catalog model id (default: the best that fits); repeat it for "
                            "several, the first is the default")
        s.add_argument("--profile", help="a machine profile: several models with flags "
                                         "measured for that hardware (see: profiles)")
        s.add_argument("--variant", help="llama.cpp build: cuda-13.5, cuda-12.4, vulkan, rocm-10.0, cpu")
        s.add_argument("--models-dir", default="",
                       help="put the model files in this folder (a second drive, say) "
                            "instead of models/gguf")
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
    rm.add_argument("action", choices=["check", "setup", "off"])
    rm.add_argument("--root", default=REPO_ROOT)
    rm.add_argument("--port", type=int, default=8000, help="Artifex API port (default 8000)")
    rm.add_argument("--model", help="model id the phone launcher starts with")
    rm.add_argument("--yes", action="store_true", help="don't ask before changing anything")
    sub.add_parser("profiles", help="list the machine profiles")
    sub.add_parser("packs", help="list the optional packs")
    pk = sub.add_parser("pack", help="install an optional pack")
    pk.add_argument("id")
    pk.add_argument("--root", default=REPO_ROOT)
    pk.add_argument("--yes", action="store_true", help="don't ask first")
    mn = sub.add_parser("menu", help="install if needed, then a numbered menu (what setup.cmd shows)")
    mn.add_argument("--root", default=REPO_ROOT)
    un = sub.add_parser("uninstall", help="list what the installer put here, then remove it (asks first)")
    un.add_argument("--root", default=REPO_ROOT)
    un.add_argument("--port", type=int, default=8000, help="API port whose Tailscale forward to turn off")
    un.add_argument("--yes", action="store_true", help="don't ask first")
    dr = sub.add_parser("doctor", help="check what this PC and this folder still need (read-only)")
    dr.add_argument("--root", default=REPO_ROOT)
    a = ap.parse_args(argv)

    if a.cmd == "menu":
        return _menu(a.root)
    if a.cmd == "doctor":
        from installer import doctor
        findings = doctor.check(a.root)
        print(doctor.report(findings))
        return 3 if any(f["level"] == doctor.FAIL for f in findings) else 0
    if a.cmd == "uninstall":
        return _uninstall(a)
    if a.cmd in ("packs", "pack"):
        return _packs(a)
    if a.cmd == "probe":
        p = probe_mod.probe(REPO_ROOT)
        print(json.dumps(p, indent=2) if a.json else probe_mod.summary(p))
        return 0
    if a.cmd == "remote":
        return _remote(a)
    return _install(a)


def _uninstall(a) -> int:
    from installer import remote, uninstall
    root = os.path.abspath(a.root)
    fwd = bool(remote.tailscale_exe()) and remote.forward_is_on(a.port)
    pl = uninstall.plan(root, recipes.load_catalog(), tailscale_forward=fwd)
    if not pl["items"] and not pl["config_entries"] and not fwd:
        print("Nothing the installer made is left in " + root)
        return 0
    print(f"Uninstall will remove, from {root}:\n" + uninstall.describe(pl))
    print("Close Artifex first (its files can't be deleted while it runs).")
    if not a.yes and not _ask("\nRemove all of the above?"):
        print("Nothing removed.")
        return 1
    failed = uninstall.remove(pl, serve_off=lambda: remote.serve_off(a.port))
    if failed:
        print("\nNot removed:\n" + "\n".join(f"  {p}: {why}" for p, why in failed))
        return 4
    print("\nDone. The Artifex code is still in this folder; delete the folder to remove it too.")
    return 0


def _start(root: str, gui: bool) -> None:
    try:
        recipes.start_app(root, gui=gui)
        print("Started. " + ("Its window opens in a few seconds." if gui
                             else "It opened in a window of its own."))
    except OSError as e:
        print(f"Couldn't start it ({e}). Double-click "
              f"'Artifex {'GUI' if gui else 'CLI'}.cmd' in this folder instead.")


def _menu(root: str, main_fn=None) -> int:
    """What a double-click on setup.cmd shows: install first if Artifex isn't
    there yet, then numbered choices, so nobody has to type a command. Every
    choice runs the same subcommand a person could type, with the same plan
    and the same question before anything is changed."""
    main_fn = main_fn or main
    root = os.path.abspath(root)
    from installer import packs
    if not recipes.is_installed(root):
        code = main_fn(["install", "--root", root])
        if code == 1:                      # they said no: that isn't a failure
            try:
                input("Press Enter to close. ")
            except (EOFError, KeyboardInterrupt):
                print()
            return 0
        if code:
            return code
        if _ask("\nStart Artifex now?", default=True):
            _start(root, gui=recipes.has_gui(root))
    ids = list(packs.PACKS)
    while True:
        gui = recipes.has_gui(root)
        print("\nArtifex setup\n"
              + ("  1  Start Artifex (desktop app)\n" if gui else "")
              + "  2  Start Artifex (in a terminal)\n"
                "  3  Add an extra (voice, pictures, image making, ...)\n"
                "  4  Use it from my phone (private, over Tailscale)\n"
                "  5  Check that nothing is reachable from the network\n"
                "  6  Run the install again (adds anything missing, changes nothing else)\n"
                "  7  Turn phone access off\n"
                "  8  Uninstall (lists what it would remove, asks first)\n"
                "  9  Check this PC: what's missing or out of date (changes nothing)\n"
                "  Enter  Close")
        try:
            pick = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if pick in ("", "q", "quit", "exit", "close"):
            return 0
        if pick == "1" and gui:
            _start(root, gui=True)
        elif pick == "2":
            _start(root, gui=False)
        elif pick == "3":
            print()
            for i, pid in enumerate(ids, 1):
                p = packs.PACKS[pid]
                print(f"  {i}  {p['label']}  ({p['size']})")
            try:
                which = input("Which one? (number, or Enter to go back) > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            pid = ids[int(which) - 1] if which.isdigit() and 1 <= int(which) <= len(ids) else which
            if pid in packs.PACKS:
                main_fn(["pack", pid, "--root", root])
            elif which:
                print("No such extra.")
        elif pick == "4":
            main_fn(["remote", "setup", "--root", root])
        elif pick == "5":
            main_fn(["remote", "check", "--root", root])
        elif pick == "6":
            # Repeat THIS folder's install (its profile, or the models it was
            # given by id, and a backend picked by hand); the generic plan would
            # add a different model and make it the default.
            main_fn(["install", "--root", root] + recipes.rerun_args(root, recipes.load_catalog()))
        elif pick == "7":
            main_fn(["remote", "off", "--root", root])
        elif pick == "8":
            if main_fn(["uninstall", "--root", root]) == 0:
                return 0
        elif pick == "9":
            main_fn(["doctor", "--root", root])
        else:
            print("Type one of the numbers above, or press Enter to close.")


def _done(ids: list, starts: str) -> str:
    """The last line of an install. `starts` is the model Artifex opens with
    (the first one in the config), which is not the one just installed when
    the config already had models."""
    if starts and starts != ids[0]:
        return (f"Done. Added {', '.join(repr(i) for i in ids)}. Your existing '{starts}' is "
                "still the one the app opens with; pick the new one in the model list.")
    return (f"Done. Artifex starts with '{ids[0]}'"
            + (f"; switch to {', '.join(repr(i) for i in ids[1:])} in the model list."
               if len(ids) > 1 else "."))


_NETWORK = ("\nSTOPPED: {e}\nIf that was the internet connection or a full disk, fix it and run "
            "this again: finished files are kept and a half-fetched one continues where it stopped.")


def _free_mb(path: str) -> int:
    import shutil
    path = os.path.abspath(path)
    while not os.path.exists(path) and os.path.dirname(path) != path:
        path = os.path.dirname(path)
    try:
        return shutil.disk_usage(path).free >> 20
    except OSError:
        return 0


def _install(a) -> int:
    catalog = recipes.load_catalog()
    if a.cmd == "profiles":
        for pid, prof in (catalog.get("profiles") or {}).items():
            print(f"  {pid:12} {prof['label']}")
            for e in prof["models"]:
                print(f"  {'':12}   {e['id']}")
        print("\nInstall one with: " + hint("install --profile <id>"))
        return 0
    p = probe_mod.probe(a.root)
    try:
        pl = recipes.plan(p, catalog, a.root, model_id=a.model, variant=a.variant,
                          profile_id=a.profile, models_dir=a.models_dir)
    except ValueError as e:
        print(f"error: {e}")
        return 2
    print(probe_mod.summary(p) + "\n\n" + _describe(pl))
    to_fetch_mb, disk_mb = recipes.missing_mb(pl)
    print("Everything is already downloaded." if not to_fetch_mb else
          f"Still to download: {to_fetch_mb / 1024:.1f} GB")
    need_mb = disk_mb + 1024 if to_fetch_mb else 0
    if need_mb and p["free_disk_mb"] and p["free_disk_mb"] < need_mb:
        print(f"\nNot enough disk space: about {need_mb / 1024:.1f} GB needed, "
              f"{p['free_disk_mb'] / 1024:.1f} GB free.")
        return 3
    if a.models_dir:
        # Models on another drive: that drive needs room for them.
        models_mb = sum(i["model"]["size"] for i in pl["models"]
                        if not os.path.isfile(i["model_path"])) >> 20
        free = _free_mb(pl["models_dir"])
        if models_mb and free and free < models_mb + 1024:
            print(f"\nNot enough space for the models in {pl['models_dir']}: about "
                  f"{(models_mb + 1024) / 1024:.1f} GB needed, {free / 1024:.1f} GB free.")
            return 3
    problems = recipes.preflight(pl, env=not getattr(a, "no_env", False))
    if problems:
        print("\n" + "\n".join(f"error: {e}" for e in problems))
        return 2
    if a.cmd == "plan":
        return 0
    if not a.yes and not _ask("\nInstall this?"):
        print("Nothing installed.")
        return 1
    try:
        recipes.install_core(pl, catalog, reuse_dirs=a.reuse, force=a.force, progress=_bar)
        recipes.save_state(pl, variant_chosen=bool(a.variant))
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
    except OSError as e:
        print(_NETWORK.format(e=e))
        return 5
    ids = [item["model"]["id"] for item in pl["models"]]
    print("\n" + _done(ids, recipes.default_model_id(pl["config_path"])))
    return 0


def run(argv=None) -> int:
    """main(), with Ctrl+C or a closed window turned into a plain message
    instead of a traceback. Downloads keep their .part files either way."""
    try:
        return main(argv)
    except KeyboardInterrupt:
        print("\n\nStopped (Ctrl+C). Finished downloads are kept, and a half-fetched file "
              "continues where it stopped: run the same command again.")
        return 130


if __name__ == "__main__":
    sys.exit(run())
