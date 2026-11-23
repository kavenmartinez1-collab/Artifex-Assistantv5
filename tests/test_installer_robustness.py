"""Installer reruns, repairs and early checks (2026-11 overhaul)."""

import json
import os

import pytest

from installer import __main__ as inst_main
from installer import packs, recipes, remote

NV8 = {"name": "RTX 5060 Ti", "vendor": "NVIDIA", "vram_mb": 8151, "compute_cap": "12.0"}


def _plan(tmp_path, **kw):
    p = {"gpus": [NV8], "nvidia": {"cuda_version": "13.3"}}
    pl = recipes.plan(p, recipes.load_catalog(), str(tmp_path), **kw)
    # Tiny stand-in sizes: _fill() writes the model files for real, and a
    # pinned 5.6 GB model would fill the temp folder.
    for item in pl["models"]:
        item["model"] = dict(item["model"], size=3 << 20)
    pl["backend_files"] = [dict(f, size=5 << 20) for f in pl["backend_files"]]
    return pl


def _fill(pl, cudart=True, stamp=False):
    """Put the planned backend and models on disk, as a finished install would."""
    os.makedirs(pl["server_dir"], exist_ok=True)
    open(pl["server_exe"], "wb").close()
    if cudart:
        open(os.path.join(pl["server_dir"], "cudart64_13.dll"), "wb").close()
    if stamp:
        recipes._write_backend_stamp(pl)
    for item in pl["models"]:
        os.makedirs(os.path.dirname(item["model_path"]), exist_ok=True)
        with open(item["model_path"], "wb") as f:
            f.truncate(item["model"]["size"])


def test_a_rerun_with_everything_in_place_needs_no_download_or_disk(tmp_path):
    pl = _plan(tmp_path)
    fetch_mb, disk_mb = recipes.missing_mb(pl)
    assert fetch_mb > 0 and disk_mb > fetch_mb          # zips count twice on disk
    _fill(pl)
    assert recipes.missing_mb(pl) == (0, 0)


def test_models_are_not_counted_twice_for_disk(tmp_path):
    pl = _plan(tmp_path)
    _fill(pl)
    os.remove(pl["models"][0]["model_path"])
    fetch_mb, disk_mb = recipes.missing_mb(pl)
    assert fetch_mb == disk_mb == pl["models"][0]["model"]["size"] >> 20


def test_a_cuda_backend_missing_its_runtime_zip_is_not_complete(tmp_path):
    pl = _plan(tmp_path)
    _fill(pl, cudart=False)              # the run died between the two zips
    assert not recipes.backend_complete(pl)
    assert recipes.missing_mb(pl)[0] > 0


def test_an_older_finished_backend_gets_a_stamp(tmp_path):
    pl = _plan(tmp_path)
    _fill(pl, cudart=True)
    assert recipes.backend_complete(pl)
    with open(os.path.join(pl["server_dir"], recipes.BACKEND_STAMP), encoding="utf-8") as f:
        assert json.load(f)["variant"] == pl["variant"]


def test_a_stamp_for_other_zips_does_not_count(tmp_path):
    pl = _plan(tmp_path)
    _fill(pl, stamp=True)
    with open(os.path.join(pl["server_dir"], recipes.BACKEND_STAMP), "w", encoding="utf-8") as f:
        json.dump({"tag": "old", "variant": pl["variant"], "files": {}}, f)
    os.remove(os.path.join(pl["server_dir"], "cudart64_13.dll"))
    assert not recipes.backend_complete(pl)


def test_install_core_stamps_the_backend_after_the_last_zip(tmp_path, monkeypatch):
    pl = _plan(tmp_path)
    unpacked = []

    def fake_download(url, dest, *a, **k):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, "wb").close()

    def fake_unzip(archive, dest):
        assert not os.path.exists(os.path.join(dest, recipes.BACKEND_STAMP))
        os.makedirs(dest, exist_ok=True)
        open(os.path.join(dest, "llama-server.exe"), "wb").close()
        unpacked.append(os.path.basename(archive))

    monkeypatch.setattr(recipes.fetch, "download", fake_download)
    monkeypatch.setattr(recipes.fetch, "safe_unzip", fake_unzip)
    monkeypatch.setattr(recipes, "find_existing", lambda m, d: "")
    monkeypatch.setattr(recipes.subprocess, "run", lambda *a, **k: recipes.subprocess.CompletedProcess(
        a, 0, "version: test", ""))
    recipes.install_core(pl, recipes.load_catalog(), log=lambda *a: None)
    assert unpacked == [f["name"] for f in pl["backend_files"]]
    assert recipes.backend_complete(pl)


def test_preflight_finds_a_broken_config_before_downloading(tmp_path):
    (tmp_path / "llama_cpp_config.json").write_text("{ not json", encoding="utf-8")
    errors = recipes.preflight(_plan(tmp_path), env=False)
    assert errors and "can't be read" in errors[0]


def test_preflight_accepts_a_config_with_a_bom(tmp_path):
    (tmp_path / "llama_cpp_config.json").write_bytes(b"\xef\xbb\xbf" + json.dumps({"models": {}}).encode())
    assert recipes.preflight(_plan(tmp_path), env=False) == []


def test_preflight_checks_python_only_when_a_venv_will_be_made(tmp_path, monkeypatch):
    monkeypatch.setattr(recipes, "python_ok", lambda version=None: False)
    assert recipes.preflight(_plan(tmp_path), env=True)
    assert recipes.preflight(_plan(tmp_path), env=False) == []
    (tmp_path / "venv").mkdir()                   # an existing venv is never rebuilt
    assert recipes.preflight(_plan(tmp_path), env=True) == []


def test_plan_stops_on_a_broken_config_without_downloading(tmp_path, monkeypatch, capsys):
    (tmp_path / "llama_cpp_config.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(inst_main.probe_mod, "probe", lambda root: {
        "gpus": [NV8], "nvidia": {"cuda_version": "13.3"}, "free_disk_mb": 10 ** 7})
    monkeypatch.setattr(inst_main.probe_mod, "summary", lambda p: "")
    monkeypatch.setattr(recipes, "install_core", lambda *a, **k: pytest.fail("downloaded"))
    assert inst_main.main(["install", "--root", str(tmp_path), "--yes", "--no-env"]) == 2
    assert "can't be read" in capsys.readouterr().out


def test_ctrl_c_is_a_message_not_a_traceback(monkeypatch, capsys):
    def interrupted(argv=None):
        raise KeyboardInterrupt
    monkeypatch.setattr(inst_main, "main", interrupted)
    assert inst_main.run([]) == 130
    assert "run the same command again" in capsys.readouterr().out


def test_strata_launcher_survives_a_non_ascii_folder(tmp_path):
    root = tmp_path / "Zoë's Artifex"
    path = packs._strata_launcher(str(root), str(root / "strata"), str(root / "strata" / "venv" / "python.exe"),
                                  str(root / "strata" / "cfg.json"), 8085, 131072)
    raw = open(path, "rb").read()
    assert raw.startswith(b"\xef\xbb\xbf")           # PowerShell 5.1 needs the BOM for UTF-8
    assert "Zoë" in raw.decode("utf-8-sig")
    assert b"\r\n" in raw


def test_exposure_check_watches_every_configured_model_port(tmp_path):
    (tmp_path / "llama_cpp_config.json").write_text(json.dumps({
        "default_port": 8081,
        "models": {"flash": {"port": 8087}, "inkling": {"port": 8093}, "plain": {"path": "x"}},
        "vision_helper": {"port": 8095}}), encoding="utf-8")
    ports = remote.watched_ports(str(tmp_path))
    assert {8000, 8081, 8085, 8087, 8093, 8095, 5173} <= set(ports)
    socks = [("0.0.0.0", 8093, "42"), ("127.0.0.1", 8087, "43")]
    findings = remote.check_exposure(str(tmp_path), run=_netstat(socks))
    fails = [f for f in findings if f["level"] == remote.FAIL]
    assert len(fails) == 1 and "8093" in fails[0]["title"] and "inkling" in fails[0]["title"]


def test_watched_ports_survive_a_missing_or_broken_config(tmp_path):
    assert remote.watched_ports(str(tmp_path)) == remote.WATCHED_PORTS
    (tmp_path / "llama_cpp_config.json").write_text("nope", encoding="utf-8")
    assert remote.watched_ports(str(tmp_path)) == remote.WATCHED_PORTS


def _netstat(socks):
    lines = "\n".join(f"  TCP    {a}:{p}    0.0.0.0:0    LISTENING    {pid}" for a, p, pid in socks)

    def run(cmd, **kw):
        out = lines if cmd[-1] == "TCP" else ""
        return recipes.subprocess.CompletedProcess(cmd, 0, out, "")
    return run


def test_windows_tools_are_called_by_full_path():
    src = open(remote.__file__, encoding="utf-8").read() + open(packs.__file__, encoding="utf-8").read()
    for name in ("netstat", "msiexec"):
        assert f'run(["{name}"' not in src
    assert 'Popen(["powershell"' not in src
    assert 'run(["powershell"' not in src


# ── catalog engine settings and the rerun memory ──────────────────────────

def test_every_catalog_engine_key_is_one_the_engine_reads():
    src = open(os.path.join(os.path.dirname(recipes.__file__), "..", "core", "engine_llama_cpp.py"),
               encoding="utf-8").read()
    for key in recipes.ENGINE_KEYS:
        assert f'model_config.get("{key}"' in src, key


def test_catalog_engine_settings_reach_the_config_entry(tmp_path):
    cat = recipes.load_catalog()
    m = dict(cat["chat_models"][0], id="x", engine={
        "sampling": {"temperature": 1.0, "top_k": 50}, "reasoning_always": True, "port": 8093})
    pl = _plan(tmp_path)
    item = dict(pl["models"][0], model=m)
    entry = recipes.config_entry(pl, item)
    assert entry["sampling"] == {"temperature": 1.0, "top_k": 50}
    assert entry["reasoning_always"] is True and entry["port"] == 8093


def test_an_unknown_engine_key_stops_the_plan_before_downloading(tmp_path, monkeypatch):
    cat = recipes.load_catalog()
    bad = dict(cat["chat_models"][0], id="typo", engine={"samplng": {}})
    cat = dict(cat, chat_models=cat["chat_models"] + [bad])
    monkeypatch.setattr(recipes, "load_catalog", lambda path=None: cat)
    p = {"gpus": [NV8], "nvidia": {"cuda_version": "13.3"}}
    with pytest.raises(ValueError, match="samplng"):
        recipes.plan(p, cat, str(tmp_path), model_id="typo")


def test_several_models_by_id_first_is_default(tmp_path):
    cat = recipes.load_catalog()
    ids = [m["id"] for m in cat["chat_models"][:2]]
    pl = _plan(tmp_path, model_id=ids + ids[:1])
    assert [i["model"]["id"] for i in pl["models"]] == ids


def test_rerun_repeats_models_asked_for_by_id_and_a_chosen_backend(tmp_path):
    cat = recipes.load_catalog()
    ids = [m["id"] for m in cat["chat_models"][:2]]
    pl = _plan(tmp_path, model_id=ids, variant="vulkan")
    recipes.save_state(pl, variant_chosen=True)
    assert recipes.rerun_args(str(tmp_path), cat) == [
        "--model", ids[0], "--model", ids[1], "--variant", "vulkan"]
    recipes.save_state(pl, variant_chosen=False)      # the probe picked it: let it pick again
    assert "--variant" not in recipes.rerun_args(str(tmp_path), cat)


def test_rerun_skips_models_no_longer_in_the_catalog(tmp_path):
    cat = recipes.load_catalog()
    (tmp_path / recipes.STATE_FILE).write_text(json.dumps(
        {"models": ["gone-model", cat["chat_models"][0]["id"]], "profile": "", "variant": ""}))
    assert recipes.rerun_args(str(tmp_path), cat) == ["--model", cat["chat_models"][0]["id"]]


def test_menu_rerun_uses_the_saved_state(tmp_path, monkeypatch):
    cat = recipes.load_catalog()
    mid = cat["chat_models"][1]["id"]
    recipes.save_state(_plan(tmp_path, model_id=mid))
    monkeypatch.setattr(recipes, "is_installed", lambda root: True)
    it = iter(["6", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))
    ran = []
    assert inst_main._menu(str(tmp_path), main_fn=lambda argv: ran.append(argv) or 0) == 0
    assert ran == [["install", "--root", str(tmp_path), "--model", mid]]


# ── updates reach existing installs ──────────────────────────────────────

def _fake_venv(tmp_path, marker_text):
    venv = tmp_path / "venv"
    (venv / "Scripts").mkdir(parents=True)
    (venv / "Scripts" / "python.exe").write_bytes(b"")
    (venv / recipes.INSTALLER_MARKER).write_text(marker_text, encoding="utf-8")
    return venv


def test_an_installer_venv_from_older_locks_is_brought_up_to_date(tmp_path):
    venv = _fake_venv(tmp_path, "created by python -m installer\n")      # pre-digest marker
    calls = []

    def run(cmd, check=False):
        calls.append(cmd)
        return recipes.subprocess.CompletedProcess(cmd, 1)           # no PyQt6 here
    recipes.install_env(str(tmp_path), log=lambda *a: None, run=run)
    pip = [c for c in calls if "pip" in c]
    assert len(pip) == 1 and recipes.LOCKS["core"] in pip[0] and recipes.LOCKS["gui"] not in pip[0]
    for flag in ("--isolated", "--require-hashes", "--only-binary=:all:"):
        assert flag in pip[0]
    assert "locks " in (venv / recipes.INSTALLER_MARKER).read_text(encoding="utf-8")
    calls.clear()
    recipes.install_env(str(tmp_path), log=lambda *a: None, run=run)  # now current: nothing runs
    assert not [c for c in calls if "pip" in c]


def test_a_venv_with_the_gui_keeps_getting_the_gui_lock(tmp_path):
    _fake_venv(tmp_path, "created by python -m installer\n")
    calls = []

    def run(cmd, check=False):
        calls.append(cmd)
        return recipes.subprocess.CompletedProcess(cmd, 0)           # PyQt6 is there
    recipes.install_env(str(tmp_path), log=lambda *a: None, run=run)
    pip = [c for c in calls if "pip" in c]
    assert recipes.LOCKS["gui"] in pip[0]


def test_a_hand_built_venv_is_still_never_touched(tmp_path):
    (tmp_path / "venv").mkdir()
    recipes.install_env(str(tmp_path), log=lambda *a: None,
                        run=lambda *a, **k: pytest.fail("must not run"))


def _entry(pl, server_exe, comment="Added by the Artifex installer 2026-10-28: x"):
    return {"_comment": comment, "path": "m.gguf", "server_path": server_exe.replace("\\", "/"),
            "num_ctx": 4096, "extra_flags": ["--keep-me"]}


def test_a_newer_llama_cpp_moves_only_installer_written_entries(tmp_path):
    pl = _plan(tmp_path)
    mid = pl["models"][0]["model"]["id"]
    old_exe = os.path.join(str(tmp_path), "runtime", "llama.cpp", "b11000-cuda-13.4", "llama-server.exe")
    (tmp_path / "llama_cpp_config.json").write_text(json.dumps({
        "server_path": old_exe.replace("\\", "/"),
        "models": {mid: _entry(pl, old_exe),
                   "mine": _entry(pl, old_exe, comment="my own entry")}}), encoding="utf-8")
    note = recipes.write_config(pl)
    cfg = json.loads((tmp_path / "llama_cpp_config.json").read_text(encoding="utf-8"))
    new = pl["server_exe"].replace("\\", "/")
    assert cfg["models"][mid]["server_path"] == new and "moved" in note
    assert cfg["models"][mid]["extra_flags"] == ["--keep-me"]          # hand tuning kept
    assert cfg["models"]["mine"]["server_path"] == old_exe.replace("\\", "/")
    assert cfg["server_path"] == new


def test_an_entry_pointed_elsewhere_by_hand_is_not_moved(tmp_path):
    pl = _plan(tmp_path)
    mid = pl["models"][0]["model"]["id"]
    (tmp_path / "llama_cpp_config.json").write_text(json.dumps({
        "models": {mid: _entry(pl, r"C:\my-llama\llama-server.exe")}}), encoding="utf-8")
    assert "kept your existing" in recipes.write_config(pl)


# ── uninstall and remote off ──────────────────────────────────────────────

from installer import uninstall


def _installed_tree(tmp_path):
    cat = recipes.load_catalog()
    root = tmp_path / "Artifex"
    (root / "runtime" / "llama.cpp" / "b1-cpu").mkdir(parents=True)
    (root / "runtime" / "llama.cpp" / "b1-cpu" / "llama-server.exe").write_bytes(b"x" * 10)
    venv = root / "venv"
    (venv / "Scripts").mkdir(parents=True)
    (venv / recipes.INSTALLER_MARKER).write_text("created by python -m installer\n")
    gguf = root / "models" / "gguf"
    gguf.mkdir(parents=True)
    pinned = cat["chat_models"][0]["file"]
    (gguf / pinned).write_bytes(b"m")
    (gguf / "my-own-model.gguf").write_bytes(b"mine")
    (root / "Artifex CLI.cmd").write_text("@echo off\n")
    (root / "Artifex Web.cmd").write_text("@echo off\n")          # ships with the code: kept
    (root / "main.py").write_text("print('the app')\n")
    (root / recipes.STATE_FILE).write_text("{}")
    (root / "llama_cpp_config.json").write_text(json.dumps({"models": {
        "installed": {"_comment": "Added by the Artifex installer 2026-11-23: x", "path": "a"},
        "mine": {"_comment": "my entry", "path": "b"}}}), encoding="utf-8")
    (root / "strata").mkdir()
    return str(root), cat, pinned


def test_uninstall_lists_only_what_the_installer_made(tmp_path):
    root, cat, pinned = _installed_tree(tmp_path)
    pl = uninstall.plan(root, cat)
    rel = {os.path.relpath(i["path"], root) for i in pl["items"]}
    assert rel == {"runtime", "venv", os.path.join("models", "gguf", pinned), "Artifex CLI.cmd",
                   recipes.STATE_FILE}
    assert pl["config_entries"] == ["installed"]
    assert [os.path.basename(p) for p in pl["left"]] == ["strata"]
    text = uninstall.describe(pl)
    assert "my-own-model" not in text and "main.py" not in text


def test_uninstall_removes_exactly_the_list_and_keeps_your_entries(tmp_path):
    root, cat, pinned = _installed_tree(tmp_path)
    assert uninstall.remove(uninstall.plan(root, cat), log=lambda *a: None) == []
    left = set(os.listdir(root))
    assert {"main.py", "Artifex Web.cmd", "strata", "models"} <= left
    assert not {"runtime", "venv", "Artifex CLI.cmd", recipes.STATE_FILE} & left
    assert os.listdir(os.path.join(root, "models", "gguf")) == ["my-own-model.gguf"]
    cfg = json.loads(open(os.path.join(root, "llama_cpp_config.json"), encoding="utf-8").read())
    assert list(cfg["models"]) == ["mine"]
    assert any(n.startswith("llama_cpp_config.json.bak-uninstall-") for n in left)


def test_a_hand_built_venv_is_not_offered_for_removal(tmp_path):
    root, cat, _ = _installed_tree(tmp_path)
    os.remove(os.path.join(root, "venv", recipes.INSTALLER_MARKER))
    assert "venv" not in {os.path.relpath(i["path"], root) for i in uninstall.plan(root, cat)["items"]}


def test_uninstall_never_leaves_the_folder(tmp_path):
    root, cat, _ = _installed_tree(tmp_path)
    pl = uninstall.plan(root, cat)
    outside = tmp_path / "precious.txt"
    outside.write_text("keep")
    pl["items"].append({"path": str(outside), "what": "x", "bytes": 4})
    failed = uninstall.remove(pl, log=lambda *a: None)
    assert outside.exists() and failed and failed[0][0] == str(outside)


def test_uninstall_says_no_by_default(tmp_path, monkeypatch, capsys):
    root, cat, _ = _installed_tree(tmp_path)
    monkeypatch.setattr(remote, "tailscale_exe", lambda: "")
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    assert inst_main.main(["uninstall", "--root", root]) == 1
    assert os.path.isdir(os.path.join(root, "runtime"))
    assert "Nothing removed" in capsys.readouterr().out


def test_remote_off_runs_only_the_fixed_undo(tmp_path):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd[1:])
        return recipes.subprocess.CompletedProcess(cmd, 0, "", "")
    remote.serve_off(8000, run=run) if remote.tailscale_exe() else None
    if calls:
        assert calls == [["serve", "--tcp=8000", "off"]]
    with pytest.raises(ValueError):
        remote.serve_off("8000; rm", run=run)


# ── install-folder guards and --models-dir ───────────────────────────────

def test_onedrive_folders_get_a_warning(tmp_path):
    od = tmp_path / "OneDrive"
    root = od / "Desktop" / "Artifex"
    w = recipes.path_warnings(str(root), environ={"OneDrive": str(od)}, long_paths=True)
    assert len(w) == 1 and "OneDrive" in w[0]
    assert recipes.path_warnings(str(tmp_path / "Artifex"), environ={"OneDrive": str(od)},
                                 long_paths=True) == []
    # a sibling named OneDriveBackup is not OneDrive
    assert recipes.path_warnings(str(tmp_path / "OneDriveBackup" / "A"),
                                 environ={"OneDrive": str(od)}, long_paths=True) == []


def test_models_dir_inside_onedrive_is_flagged_too(tmp_path):
    od = tmp_path / "OneDrive"
    w = recipes.path_warnings(str(tmp_path / "Artifex"), models_dir=str(od / "models"),
                              environ={"OneDrive": str(od)}, long_paths=True)
    assert len(w) == 1 and "the models" in w[0]


def test_extract_all_nesting_and_long_paths_are_flagged(tmp_path):
    nested = tmp_path / "Artifex-main" / "Artifex-main"
    assert any("nested" in x for x in recipes.path_warnings(str(nested), environ={}, long_paths=True))
    deep = tmp_path / ("x" * 100)
    assert any("long" in x for x in recipes.path_warnings(str(deep), environ={}, long_paths=False))
    assert not recipes.path_warnings(str(deep), environ={}, long_paths=True)


def test_models_dir_moves_the_models_and_is_remembered(tmp_path):
    other = tmp_path / "D-drive" / "models"
    pl = _plan(tmp_path / "Artifex", models_dir=str(other))
    assert all(os.path.dirname(i["model_path"]) == str(other) for i in pl["models"])
    recipes.save_state(pl)
    args = recipes.rerun_args(str(tmp_path / "Artifex"), recipes.load_catalog())
    assert args[-2:] == ["--models-dir", str(other)]
    plain = _plan(tmp_path / "Plain")
    recipes.save_state(plain)
    assert "--models-dir" not in recipes.rerun_args(str(tmp_path / "Plain"), recipes.load_catalog())


# ── which card and which model ────────────────────────────────────────────

AMD12 = {"name": "RX 6700 XT", "vendor": "AMD", "vram_mb": 12272}
AMD16 = {"name": "RX 7800 XT", "vendor": "AMD", "vram_mb": 16368}


def _pick(gpus, cuda="13.3", ram=32768):
    p = {"gpus": gpus, "nvidia": {"cuda_version": cuda} if any(
        g["vendor"] == "NVIDIA" for g in gpus) else None, "ram_mb": ram}
    pl = recipes.plan(p, recipes.load_catalog(), "C:/x")
    return pl["variant"], pl["gpu"]["name"] if pl["gpu"] else None, pl["model"]


def test_nvidia_wins_when_the_other_card_fits_the_same_model():
    v, gpu, _ = _pick([NV8, AMD12])
    assert v.startswith("cuda") and gpu == NV8["name"]


def test_a_bigger_non_nvidia_card_wins_when_it_fits_a_better_model():
    v, gpu, m = _pick([NV8, AMD16])
    assert (v, gpu) == ("vulkan", AMD16["name"]) and m["min_vram_mb"] > 7500


def test_cpu_model_must_fit_in_ram():
    cat = recipes.load_catalog()
    small = min((m for m in cat["chat_models"] if m.get("auto", True)), key=lambda m: m["size"])
    v, _, m = _pick([], ram=4096)
    assert v == "cpu" and m["id"] == small["id"]


def test_a_tiny_pc_still_gets_a_plan_with_a_warning():
    p = {"gpus": [], "nvidia": None, "ram_mb": 1024}
    pl = recipes.plan(p, recipes.load_catalog(), "C:/x")
    assert pl["warnings"] and "RAM" in pl["warnings"][0]


# ── doctor ────────────────────────────────────────────────────────────────

from installer import doctor

_PC = {"gpus": [NV8], "nvidia": {"cuda_version": "13.3"}, "ram_mb": 32768, "free_disk_mb": 10 ** 6,
       "docker": True}


def _doc(root, **kw):
    args = dict(p=_PC, gateway=lambda: True, vc=lambda: True, which=lambda n: "C:/x/" + n)
    args.update(kw)
    return doctor.check(str(root), **args)


def test_doctor_flags_a_missing_vc_runtime(tmp_path):
    f = _doc(tmp_path, vc=lambda: False)
    assert any(x["level"] == doctor.FAIL and "Visual C++" in x["title"] for x in f)


def test_doctor_explains_web_search_and_git(tmp_path):
    f = _doc(tmp_path, p=dict(_PC, docker=False), which=lambda n: None)
    titles = " ".join(x["title"] for x in f)
    assert "Docker Desktop is not installed" in titles and "Git is not installed" in titles


def test_doctor_on_an_old_nvidia_driver_suggests_an_update(tmp_path):
    f = _doc(tmp_path, p=dict(_PC, nvidia={"cuda_version": "11.8"}))
    assert any("update the NVIDIA driver" in x["fix"] for x in f)


def test_doctor_leaves_a_hand_built_folder_alone(tmp_path):
    (tmp_path / "venv" / "Scripts").mkdir(parents=True)
    (tmp_path / "venv" / "Scripts" / "python.exe").write_bytes(b"")
    (tmp_path / "llama_cpp_config.json").write_text(json.dumps({"models": {"mine": {"path": "x"}}}))
    f = _doc(tmp_path)
    assert any("set up by hand" in x["title"] for x in f)


def test_doctor_sees_an_install_that_is_behind(tmp_path):
    pl = _plan(tmp_path)
    (tmp_path / "venv" / "Scripts").mkdir(parents=True)
    (tmp_path / "venv" / "Scripts" / "python.exe").write_bytes(b"")
    recipes.write_config(pl)
    recipes.save_state(pl)
    f = _doc(tmp_path)
    assert any("incomplete or behind" in x["title"] for x in f)


def test_doctor_changes_nothing(tmp_path):
    before = sorted(os.listdir(tmp_path))
    _doc(tmp_path)
    assert sorted(os.listdir(tmp_path)) == before


def test_a_model_too_big_for_the_card_is_warned_about(tmp_path):
    pl = _plan(tmp_path, model_id="inkling-2-mini-iq4nl")
    assert any("video memory" in w for w in pl["warnings"])
    assert not _plan(tmp_path, model_id="qwen3.5-2b")["warnings"]


def test_inkling_entry_carries_its_engine_settings(tmp_path):
    pl = _plan(tmp_path, model_id="inkling-2-mini-iq4nl")
    entry = recipes.config_entry(pl, pl["models"][0])
    assert entry["reasoning_always"] is True and entry["native_tools"] is True
    assert entry["sampling"]["temperature"] == 1.0 and entry["sampling"]["top_k"] == 50
    assert pl["model"].get("auto") is False
