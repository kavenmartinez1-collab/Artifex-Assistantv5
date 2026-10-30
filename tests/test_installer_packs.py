"""Optional packs: each one installs only pinned, hash-locked pieces, refuses
to run where it can't work safely, and never exposes anything beyond
127.0.0.1. Docker/pip/Tailscale are replaced by fakes; the compose file and
gateway Dockerfile are checked as they are in the repo.
"""

import json
import os
import re
import types

import pytest

from core.pipelines import base, sr_arch
from core.services import media_tools
from installer import fetch, packs, recipes

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ok(stdout=""):
    return types.SimpleNamespace(returncode=0, stdout=stdout, stderr="")


# ── Docker: the compose file and gateway image, as committed ────────────────

def _compose():
    return open(os.path.join(REPO, "docker-compose.yml"), encoding="utf-8").read()


def test_every_published_port_is_loopback_only():
    published = re.findall(r'^\s*-\s*"([^"]*:\d+)"\s*$', _compose(), re.M)
    assert published, "no ports found; did the file format change?"
    for p in published:
        assert p.startswith("127.0.0.1:"), f"port {p!r} would be published on every interface"


def test_every_pulled_image_is_pinned_by_digest():
    for img in re.findall(r"^\s*image:\s*(\S+)", _compose(), re.M):
        assert re.search(r"@sha256:[0-9a-f]{64}$", img), f"{img} is not pinned by digest"


def test_gateway_image_is_pinned_and_hash_locked():
    df = open(os.path.join(REPO, "web-gateway", "Dockerfile"), encoding="utf-8").read()
    assert re.search(r"^FROM \S+@sha256:[0-9a-f]{64}\s*$", df, re.M)
    pip = next(l for l in df.splitlines() if "pip install" in l)
    assert "--require-hashes" in pip and "--only-binary=:all:" in pip
    lock = open(os.path.join(REPO, "web-gateway", "requirements.lock.txt"), encoding="utf-8").read()
    reqs = re.findall(r"^([A-Za-z0-9_.\-]+)==\S+ \\$", lock, re.M)
    assert len(reqs) > 10 and "--index-url" not in lock


def test_full_image_is_pinned_and_installs_only_pinned_packages():
    df = open(os.path.join(REPO, "Dockerfile"), encoding="utf-8").read()
    assert re.search(r"^FROM \S+@sha256:[0-9a-f]{64}( AS \w+)?\s*$", df, re.M)
    for line in df.splitlines():
        if "pip" in line and " install" in line and not line.lstrip().startswith("#"):
            assert "-r requirements.txt" in line, f"unpinned install: {line.strip()}"
            assert "|| true" not in line


# ── Web search ──────────────────────────────────────────────────────────────

def test_web_search_without_docker_explains_and_stops(monkeypatch, tmp_path):
    monkeypatch.setattr(packs.shutil, "which", lambda n: None)
    with pytest.raises(packs.PackUnavailable, match="Docker Desktop"):
        packs.install_web_search(str(tmp_path), {}, log=lambda *a: None)


def test_web_search_starts_only_the_gateway_and_searxng(monkeypatch, tmp_path):
    (tmp_path / "docker-compose.yml").write_text("services: {}")
    monkeypatch.setattr(packs.shutil, "which", lambda n: "docker")
    monkeypatch.setattr(packs, "gateway_health", lambda: {"status": "ok", "searxng": "ok"})
    calls = []
    def run(cmd, **kw):
        calls.append(cmd)
        return _ok("27.0.0")
    packs.install_web_search(str(tmp_path), {}, log=lambda *a: None, run=run)
    up = calls[-1]
    assert up[:2] == ["docker", "compose"] and up[-2:] == ["web-gateway", "searxng"]
    assert "--profile" not in up                  # never the full/ollama/lan profiles


# ── Ollama ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("host", ["0.0.0.0", "0.0.0.0:11434", "http://192.168.1.5:11434", "::"])
def test_ollama_open_to_the_network_is_refused(monkeypatch, tmp_path, host):
    monkeypatch.setattr(packs.shutil, "which", lambda n: "ollama")
    with pytest.raises(packs.PackUnavailable, match="OLLAMA_HOST"):
        packs.install_ollama(str(tmp_path), {}, log=lambda *a: None,
                             run=lambda *a, **k: _ok(), env={"OLLAMA_HOST": host})


def test_ollama_on_loopback_gets_launchers(monkeypatch, tmp_path):
    monkeypatch.setattr(packs.shutil, "which", lambda n: "ollama")
    def run(cmd, **kw):
        if cmd[0] == "netstat":
            return _ok("  TCP    127.0.0.1:11434  0.0.0.0:0  LISTENING  42\n")
        return _ok("NAME   ID   SIZE\nlfm3:latest  abc  1.7 GB\n")
    paths = packs.install_ollama(str(tmp_path), {}, log=lambda *a: None, run=run,
                                 env={"OLLAMA_HOST": "127.0.0.1"})
    text = open(paths[0], encoding="utf-8").read()
    assert "set ARTIFEX_BACKEND=ollama" in text


# ── Packages go only into the installer's own venv ──────────────────────────

def _venv(tmp_path, marker=True):
    # Both layouts: venv\Scripts\python.exe on Windows, venv/bin/python on
    # Linux (CI), which is what recipes.venv_python() looks for there.
    for sub, exe in (("Scripts", "python.exe"), ("bin", "python")):
        d = tmp_path / "venv" / sub
        d.mkdir(parents=True)
        (d / exe).write_text("")
    if marker:
        (tmp_path / "venv" / recipes.INSTALLER_MARKER).write_text("x")


def test_pack_packages_go_into_the_installer_venv_hash_locked(tmp_path):
    _venv(tmp_path)
    calls = []
    recipes.add_locked_packages(str(tmp_path), recipes.LOCKS["create"], ["torch"],
                                log=lambda *a: None, run=lambda c, **k: calls.append(c) or _ok(),
                                extra_index=recipes.PYTORCH_INDEX)
    cmd = calls[0]
    for flag in ("--isolated", "--require-hashes", "--only-binary=:all:"):
        assert flag in cmd
    assert cmd[cmd.index("--extra-index-url") + 1] == recipes.PYTORCH_INDEX


def test_unknown_package_index_is_refused(tmp_path):
    _venv(tmp_path)
    with pytest.raises(ValueError):
        recipes.add_locked_packages(str(tmp_path), recipes.LOCKS["voice"], ["x"],
                                    log=lambda *a: None, run=lambda c, **k: _ok(),
                                    extra_index="https://evil.example/simple")


def test_a_hand_built_venv_is_never_changed(tmp_path):
    _venv(tmp_path, marker=False)
    calls = []
    def run(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=1, stdout="", stderr="")   # modules missing
    with pytest.raises(RuntimeError, match="isn't changed automatically"):
        recipes.add_locked_packages(str(tmp_path), recipes.LOCKS["voice"], ["faster_whisper"],
                                    log=lambda *a: None, run=run)
    assert not any("install" in c for c in calls)


@pytest.mark.parametrize("lock", ["voice", "create", "strata"])
def test_pack_locks_are_fully_hashed_and_index_free(lock):
    text = open(recipes.LOCKS[lock], encoding="utf-8").read()
    assert "--index-url" not in text and "--extra-index-url" not in text
    pins = re.findall(r"^([A-Za-z0-9_.\-]+)==(\S+) \\$", text, re.M)
    assert pins
    for name, _ in pins:
        block = text.split(f"{name}==", 1)[1].split("\n\n", 1)[0]
        assert "--hash=sha256:" in block


def test_create_and_voice_locks_agree_on_shared_packages():
    def pins(lock):
        return dict(re.findall(r"^([A-Za-z0-9_.\-]+)==(\S+) \\$",
                               open(recipes.LOCKS[lock], encoding="utf-8").read(), re.M))
    v, c = pins("voice"), pins("create")
    clash = {k: (v[k], c[k]) for k in v.keys() & c.keys() if v[k] != c[k]}
    assert not clash, f"the voice and create packs would fight over {clash}"


# ── Vision ──────────────────────────────────────────────────────────────────

def test_vision_adds_entries_and_never_edits_yours(monkeypatch, tmp_path):
    cfg = {"models": {"qwen3.5-9b-q4km": {"path": "m.gguf", "extra_flags": ["-fa", "on"]},
                      "mine": {"path": "keep.gguf"}}}
    (tmp_path / "llama_cpp_config.json").write_text(json.dumps(cfg))
    monkeypatch.setattr(packs, "_hf_file", lambda spec, *a, **k: "models/gguf/" + spec["file"])
    cpu = tmp_path / "runtime" / "llama.cpp" / "b11945-cpu"
    cpu.mkdir(parents=True)
    (cpu / "llama-server.exe").write_text("")
    run = lambda cmd, **kw: _ok("Red rectangle, blue circle.")
    packs.install_vision(str(tmp_path), recipes.load_catalog(), log=lambda *a: None, run=run)
    new = json.loads((tmp_path / "llama_cpp_config.json").read_text())
    assert new["models"]["mine"] == {"path": "keep.gguf"}
    assert new["models"]["qwen3.5-9b-q4km"] == cfg["models"]["qwen3.5-9b-q4km"]
    vis = new["models"]["qwen3.5-9b-q4km-vision"]["extra_flags"]
    assert vis[-2] == "--mmproj" and vis[-1].endswith("mmproj-Qwen_Qwen3.5-9B-f16.gguf")
    vh = new["vision_helper"]
    assert vh["num_gpu_layers"] == 0 and vh["port"] == 8095


# ── Create ──────────────────────────────────────────────────────────────────

NV = {"name": "RTX 5060 Ti", "vendor": "NVIDIA", "vram_mb": 8151}


@pytest.mark.parametrize("probe, why", [
    ({"gpus": [{"name": "RX 6700 XT", "vendor": "AMD", "vram_mb": 12272}]}, "NVIDIA GPU"),
    ({"gpus": [NV], "nvidia": {"cuda_version": "12.4"}}, "CUDA 12.8"),
    ({"gpus": [dict(NV, vram_mb=6144)], "nvidia": {"cuda_version": "13.3"}}, "8 GB"),
    ({"gpus": []}, "NVIDIA GPU"),
])
def test_create_refuses_hardware_it_cant_run_on(probe, why):
    with pytest.raises(packs.PackUnavailable, match=why):
        packs._create_gpu(lambda: probe)


def test_qwen_manifest_is_pinned_and_path_safe():
    m = json.load(open(packs.QWEN_MANIFEST, encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{40}", m["revision"])
    for f in m["files"]:
        assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["size"] >= 0
        assert not set(f["path"].split("/")) & {"", ".", ".."}
    weights = [f["path"] for f in m["files"] if f["size"] > 100 << 20]
    assert weights and all(p.endswith(".safetensors") for p in weights)


# ── Create models: pinned commits and verified weights ──────────────────────

def test_hub_models_resolve_to_a_pinned_commit():
    for tool, models in media_tools._HUB_MODELS.items():
        for repo, _ in models:
            assert re.fullmatch(r"[0-9a-f]{40}", media_tools._HUB_REVISIONS[repo]), repo


@pytest.mark.parametrize("path, expect", [
    ("facebook/musicgen-small@" + "a" * 40, ("facebook/musicgen-small", "a" * 40)),
    ("org/repo@main", ("org/repo@main", None)),                 # a branch is not a pin
    (r"C:\models\qwen-image", (r"C:\models\qwen-image", None)),
    ("", ("", None)),
])
def test_hub_revision(path, expect):
    assert base.hub_revision(path) == expect


def test_restore_weights_have_pinned_hashes():
    for name, info in sr_arch.SR_MODELS.items():
        assert re.fullmatch(r"[0-9a-f]{64}", info["sha256"]), name


def test_tampered_restore_weights_are_refused(monkeypatch, tmp_path):
    def fake_retrieve(url, dest):
        open(dest, "wb").write(b"not the real weights")
    monkeypatch.setattr(sr_arch.urllib.request, "urlretrieve", fake_retrieve)
    dest = tmp_path / "w.pth"
    with pytest.raises(RuntimeError, match="does not match"):
        sr_arch.fetch_verified("https://github.com/x/w.pth", str(dest), "0" * 64)
    assert not dest.exists() and not (tmp_path / "w.pth.part").exists()


def test_a_swapped_cached_weight_file_is_replaced(monkeypatch, tmp_path):
    good = b"real weights"
    import hashlib
    h = hashlib.sha256(good).hexdigest()
    dest = tmp_path / "w.pth"
    dest.write_bytes(b"swapped on disk")
    monkeypatch.setattr(sr_arch.urllib.request, "urlretrieve",
                        lambda url, d: open(d, "wb").write(good))
    sr_arch.fetch_verified("https://github.com/x/w.pth", str(dest), h)
    assert dest.read_bytes() == good


# ── Node.js projects (Artifex Web, the game) ────────────────────────────────

def _lock(tmp_path, packages, version=3):
    (tmp_path / "package-lock.json").write_text(json.dumps(
        {"lockfileVersion": version, "packages": {"": {"name": "x"}, **packages}}))
    return str(tmp_path)


GOOD = {"resolved": "https://registry.npmjs.org/a/-/a-1.0.0.tgz", "integrity": "sha512-abc"}


def test_lockfile_audit_accepts_registry_packages_with_hashes(tmp_path):
    packs.audit_lockfile(_lock(tmp_path, {"node_modules/a": GOOD}))


@pytest.mark.parametrize("entry", [
    {"resolved": "https://registry.npmjs.org/a/-/a-1.0.0.tgz"},                     # no hash
    {"resolved": "https://registry.npmjs.org/a/-/a.tgz", "integrity": "sha1-abc"},   # weak hash
    {"resolved": "https://evil.example/a.tgz", "integrity": "sha512-abc"},           # other host
    {"resolved": "git+ssh://git@github.com/evil/thing.git#" + "a" * 40},             # git, not allowed
    {"resolved": "git+ssh://git@github.com/ceifa/steamworks.js.git#main"},           # branch, not commit
])
def test_lockfile_audit_refuses_unverifiable_entries(tmp_path, entry):
    with pytest.raises(fetch.VerifyError):
        packs.audit_lockfile(_lock(tmp_path, {"node_modules/a": entry}),
                             allow_git=("ceifa/steamworks.js",))


def test_lockfile_audit_allows_only_listed_git_deps_at_a_commit(tmp_path):
    entry = {"resolved": "git+ssh://git@github.com/ceifa/steamworks.js.git#" + "a" * 40}
    packs.audit_lockfile(_lock(tmp_path, {"node_modules/steamworks.js": entry}),
                         allow_git=("ceifa/steamworks.js",))


def test_missing_or_old_lockfile_is_refused(tmp_path):
    with pytest.raises(packs.PackUnavailable):
        packs.audit_lockfile(str(tmp_path))
    with pytest.raises(packs.PackUnavailable):
        packs.audit_lockfile(_lock(tmp_path, {}, version=1))


def test_npm_runs_without_scripts_from_the_registry_and_git_over_https(monkeypatch):
    monkeypatch.setenv("npm_config_registry", "https://evil.example/")
    monkeypatch.setenv("NPM_CONFIG_IGNORE_SCRIPTS", "false")
    env = packs._npm_env(r"C:\node\node.exe")
    assert env["npm_config_registry"] == packs.NPM_REGISTRY
    assert env["npm_config_ignore_scripts"] == "true"
    assert "NPM_CONFIG_IGNORE_SCRIPTS" not in env
    pairs = {(env[f"GIT_CONFIG_KEY_{i}"], env[f"GIT_CONFIG_VALUE_{i}"])
             for i in range(int(env["GIT_CONFIG_COUNT"]))}
    assert ("protocol.allow", "never") in pairs and ("protocol.https.allow", "always") in pairs
    calls = []
    packs.npm(r"C:\node\node.exe", r"C:\node\npm-cli.js", ["ci", "--ignore-scripts"], ".",
              run=lambda cmd, **kw: calls.append(cmd) or _ok())
    assert calls[0][0].endswith("node.exe") and calls[0][1].endswith("npm-cli.js")   # never npm.cmd


@pytest.mark.parametrize("vite, server, ok", [
    ("server: { host: '127.0.0.1', port: 5173 }", "server.listen(PORT, '127.0.0.1', () => {", True),
    ("server: { host: '0.0.0.0' }", "server.listen(PORT, '127.0.0.1', () => {", False),
    ("server: { host: '127.0.0.1' }", "server.listen(PORT, () => {", False),
])
def test_dev_servers_must_be_loopback_only(tmp_path, vite, server, ok):
    (tmp_path / "vite.config.ts").write_text(vite)
    (tmp_path / "server").mkdir()
    (tmp_path / "server" / "dev-server.ts").write_text(server)
    if ok:
        packs.check_loopback_only(str(tmp_path))
    else:
        with pytest.raises(fetch.VerifyError):
            packs.check_loopback_only(str(tmp_path))


def test_the_real_webgpu_project_passes_its_own_checks():
    packs.check_loopback_only(os.path.join(REPO, "webgpu"))
    packs.audit_lockfile(os.path.join(REPO, "webgpu"))


def test_node_and_game_are_pinned():
    cat = recipes.load_catalog()
    assert re.fullmatch(r"[0-9a-f]{64}", cat["node"]["sha256"])
    assert cat["node"]["base_url"].startswith("https://nodejs.org/dist/")
    g = cat["game"]
    assert re.fullmatch(r"[0-9a-f]{40}", g["commit"]) and g["repo"].startswith("https://github.com/")
    for m in g["models"]:
        assert re.fullmatch(r"[0-9a-f]{64}", m["sha256"]) and re.fullmatch(r"[0-9a-f]{40}", m["revision"])
        assert not set(m["dest"].split("/")) & {"", ".", ".."}
    assert "master" not in g["cmudict"]["base_url"]


def test_game_ships_lfm3_with_its_licence():
    g = recipes.load_catalog()["game"]
    dests = {m["dest"] for m in g["models"]}
    gguf = [d for d in dests if d.endswith(".gguf")]
    assert gguf == ["models/LiquidAI--LFM3-1.7B-GGUF/LFM3-1.7B-Q4_K_M.gguf"]
    folder = gguf[0].rsplit("/", 1)[0]
    # The LFM Open License must accompany the weights (§4a), and the
    # tokenizer must be local or the game would fetch one at runtime.
    for f in ("LICENSE", "tokenizer.json", "tokenizer_config.json"):
        assert f"{folder}/{f}" in dests


# ── Artifex Web's model (webgpu pack) ───────────────────────────────────────

def test_artifex_web_model_is_pinned_and_path_safe():
    cat = recipes.load_catalog()
    w = cat["webgpu"]
    spec = packs._webgpu_spec(cat)  # must name a chat_models entry
    assert re.fullmatch(r"[0-9a-f]{64}", spec["sha256"])
    assert re.fullmatch(r"[A-Za-z0-9._-]+", w["model_dir"]) and w["model_dir"] not in (".", "..")
    tok = w["tokenizer"]
    assert re.fullmatch(r"[0-9a-f]{40}", tok["revision"])
    for f in tok["files"]:
        assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["size"] > 0
        assert re.fullmatch(r"[A-Za-z0-9._-]+", f["file"])
        assert fetch.host_allowed(recipes._HF.format(repo=tok["repo"], revision=tok["revision"],
                                                     file=f["file"]), cat["allowed_hosts"])
    # A config.json beside the GGUF would make the app take the safetensors path.
    assert "config.json" not in {f["file"] for f in tok["files"]}


def _web_catalog(tmp_path, content=b"gguf bytes"):
    import hashlib
    spec = {"id": "m", "label": "Model M (small)", "repo": "org/M-GGUF", "revision": "a" * 40,
            "file": "M.gguf", "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    tok = {"repo": "org/M", "revision": "b" * 40,
           "files": [{"file": "tokenizer.json", "size": 2, "sha256": "c" * 64}]}
    return {"chat_models": [spec], "allowed_hosts": ["huggingface.co"],
            "webgpu": {"model": "m", "model_dir": "m-dir", "tokenizer": tok}}


def test_artifex_web_reuses_the_core_models_verified_copy(monkeypatch, tmp_path):
    cat = _web_catalog(tmp_path)
    core = tmp_path / "models" / "gguf" / "M.gguf"
    core.parent.mkdir(parents=True)
    core.write_bytes(b"gguf bytes")
    calls = []

    def fake_download(url, dest, size, sha256, hosts, **k):
        calls.append(url)
        if dest.endswith(".gguf"):  # already in place: download() returns without fetching
            assert fetch.file_matches(dest, size, sha256)
        return dest
    monkeypatch.setattr(packs.fetch, "download", fake_download)
    assert packs._webgpu_model(str(tmp_path), cat, log=lambda *a: None) == "local/m-dir"
    placed = tmp_path / "webgpu" / "models" / "m-dir" / "M.gguf"
    assert placed.read_bytes() == b"gguf bytes" and not (tmp_path / "webgpu/models/m-dir/M.gguf.part").exists()
    assert calls == ["https://huggingface.co/org/M-GGUF/resolve/" + "a" * 40 + "/M.gguf",
                     "https://huggingface.co/org/M/resolve/" + "b" * 40 + "/tokenizer.json"]


def test_artifex_web_ignores_a_core_copy_that_fails_its_hash(monkeypatch, tmp_path):
    cat = _web_catalog(tmp_path)
    core = tmp_path / "models" / "gguf" / "M.gguf"
    core.parent.mkdir(parents=True)
    core.write_bytes(b"tampered!!")  # same size, wrong content
    fetched = []
    monkeypatch.setattr(packs.fetch, "download", lambda url, dest, *a, **k: fetched.append(dest) or dest)
    packs._webgpu_model(str(tmp_path), cat, log=lambda *a: None)
    assert not (tmp_path / "webgpu/models/m-dir/M.gguf").exists()   # not linked/copied
    assert fetched[0].endswith("M.gguf")                              # the pinned download instead


def test_artifex_web_launcher_opens_with_the_model_filled_in(tmp_path):
    node = tmp_path / "node" / "node.exe"
    path = packs._web_launcher(str(tmp_path), "Artifex Web.cmd", str(tmp_path / "webgpu"), str(node),
                               "http://127.0.0.1:5173/?model=local/lfm3-1.7b")
    text = open(path, encoding="utf-8").read()
    assert 'start "" "http://127.0.0.1:5173/?model=local/lfm3-1.7b"' in text


# ── Strata (Qwen3.8-Flash-Next) ─────────────────────────────────────────────

def test_strata_is_pinned_end_to_end():
    s = recipes.load_catalog()["strata"]
    assert re.fullmatch(r"[0-9a-f]{40}", s["commit"]) and s["repo"].startswith("https://github.com/")
    assert re.fullmatch(r"[0-9a-f]{40}", s["model"]["revision"])
    pinned = s["model"]["files"] + list(s["engines"].values()) + s["feed"] + [s["llama_cpp_source"]]
    for f in pinned:
        assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["size"] > 0
    for f in list(s["engines"].values()) + s["feed"]:
        assert f["url"].startswith(f"https://github.com/Niko1221/Strata/releases/download/v{s['version']}/")
    assert s["llama_cpp_source"]["url"].endswith(".zip") and "/archive/" in s["llama_cpp_source"]["url"]
    for d in (s["dest"], s["models_dest"], s["data_dest"]):
        assert not set(d.split("/")) & {"", ".", ".."}


@pytest.mark.parametrize("probe, variant", [
    ({"ram_mb": 32556, "gpus": [{"name": "NVIDIA GeForce RTX 5060 Ti", "vendor": "NVIDIA", "vram_mb": 8151},
                                {"name": "AMD Radeon RX 6700 XT", "vendor": "AMD", "vram_mb": 12242}],
      "nvidia": {"driver": "610.88"}}, "hip-rdna2"),
    ({"ram_mb": 65536, "gpus": [{"name": "AMD Radeon RX 7900 XTX", "vendor": "AMD", "vram_mb": 24560}]}, "hip-rdna3"),
    ({"ram_mb": 65536, "gpus": [{"name": "AMD Radeon RX 9070 XT", "vendor": "AMD", "vram_mb": 16304}]}, "hip-rx9070"),
    ({"ram_mb": 32768, "gpus": [{"name": "NVIDIA GeForce RTX 3060", "vendor": "NVIDIA", "vram_mb": 12288,
                                 "compute_cap": "8.6"}], "nvidia": {"driver": "590.10"}}, "cuda-cuda13"),
    ({"ram_mb": 32768, "gpus": [{"name": "NVIDIA GeForce RTX 3060", "vendor": "NVIDIA", "vram_mb": 12288,
                                 "compute_cap": "8.6"}], "nvidia": {"driver": "560.00"}}, "cuda-cuda12"),
])
def test_strata_picks_the_biggest_card_and_its_engine(probe, variant):
    backend, got, g, _ = packs._strata_gpu(lambda: probe)
    assert got == variant and got in recipes.load_catalog()["strata"]["engines"]
    assert backend == ("hip" if got.startswith("hip") else "cuda")


@pytest.mark.parametrize("probe, why", [
    ({"ram_mb": 32768, "gpus": [{"name": "NVIDIA GeForce RTX 4060", "vendor": "NVIDIA", "vram_mb": 8188}]}, "12 GB"),
    ({"ram_mb": 16384, "gpus": [{"name": "NVIDIA GeForce RTX 3060", "vendor": "NVIDIA", "vram_mb": 12288}]}, "32 GB of RAM"),
    ({"ram_mb": 32768, "gpus": [{"name": "AMD Radeon RX 6650 XT 12GB", "vendor": "AMD", "vram_mb": 12288}]}, "no ready-made"),
    ({"ram_mb": 32768, "gpus": [{"name": "NVIDIA GeForce GTX 1080 Ti", "vendor": "NVIDIA", "vram_mb": 11264,
                                 "compute_cap": "6.1"}], "nvidia": {"driver": "590.10"}}, "older than RTX 20"),
    ({"ram_mb": 32768, "gpus": []}, "graphics card"),
])
def test_strata_refuses_hardware_it_cant_run_on(probe, why):
    with pytest.raises(packs.PackUnavailable, match=why):
        packs._strata_gpu(lambda: probe)


def test_strata_run_config_is_loopback_and_tuned():
    s = recipes.load_catalog()["strata"]
    setup_cfg = {"exe": "strata.exe", "port": 8080, "args": ["--pack", "p", "--spec", "4", "--mmap-experts",
                                                             "--max-context", "32768"]}
    cfg = packs._strata_artifex_config(setup_cfg, s, 31.8, "hip")
    a = cfg["args"]
    assert cfg["host"] == "127.0.0.1" and cfg["port"] == s["port"]
    assert "--mmap-experts" not in a and "--resident-experts" in a
    assert a[a.index("--resident-budget-gib") + 1] == "12.3"
    assert a[a.index("--spec") + 1] == "3" and a[a.index("--vram-reserve-mib") + 1] == "700"
    assert cfg["env"]["STRATA_ROUTE_PRIOR"] == "1.25" and cfg["env"]["STRATA_HIP_PROMPT_F16"] == "1"
    big = packs._strata_artifex_config(dict(setup_cfg, args=["--pack", "p"]), s, 128, "cuda")
    assert "--resident-budget-gib" not in big["args"] and "STRATA_HIP_PROMPT_F16" not in big["env"]


def test_strata_stamp_lists_what_strata_setup_would_install(tmp_path):
    (tmp_path / "requirements.txt").write_text("# c\nnumpy==2.4.6; python_version == \"3.11\"\npillow==12.3.0\n\n")
    (tmp_path / "requirements-monitor.txt").write_text("# m\ntextual==8.2.8\n")
    assert packs._strata_runtime_packages(str(tmp_path)) == [
        'numpy==2.4.6; python_version == "3.11"', "pillow==12.3.0", "textual==8.2.8"]


def test_strata_launcher_binds_loopback_only(tmp_path):
    p = packs._strata_launcher(str(tmp_path), str(tmp_path / "S"), str(tmp_path / "py.exe"),
                               str(tmp_path / "c.json"), 8085, 262144)
    text = open(p, encoding="ascii").read()
    assert "--host 127.0.0.1 --port 8085" in text and "$cfg.host = '127.0.0.1'" in text
    assert "0.0.0.0" not in text


def test_strata_lock_has_no_known_vulnerable_pillow():
    text = open(recipes.LOCKS["strata"], encoding="utf-8").read()
    assert "pillow==13.0.0" in text and "pillow==12." not in text
