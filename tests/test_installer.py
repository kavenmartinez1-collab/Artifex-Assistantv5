"""The installer's safety rules, attacked the way a bad server or archive would.

An installer is a supply-chain target: it downloads and unpacks code. These
cover the rules in installer/fetch.py (allow-listed https hosts, pinned size
and SHA-256 from the repo, nothing kept on a mismatch, zip-slip-proof
unpacking) and the choices in installer/recipes.py.
"""

import hashlib
import io
import json
import os
import re
import stat
import zipfile

import pytest

from installer import fetch, recipes
from installer.fetch import VerifyError

HOSTS = ["github.com", ".githubusercontent.com", "huggingface.co", ".huggingface.co", ".hf.co"]


# ── hosts ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url, ok", [
    ("https://github.com/ggml-org/llama.cpp/releases/download/x/a.zip", True),
    ("https://release-assets.githubusercontent.com/a", True),
    ("https://huggingface.co/x/y/resolve/abc/f.gguf", True),
    ("https://cas-bridge.xethub.hf.co/f", True),
    ("http://github.com/a", False),                      # not https
    ("https://evilgithub.com/a", False),                 # lookalike
    ("https://github.com.evil.io/a", False),             # suffix trick
    ("https://huggingface.co.attacker.net/a", False),
    ("https://githubusercontent.com/a", False),           # ".x" rules need a subdomain
    ("file:///C:/Windows/System32/x", False),
    ("https:///nohost", False),
])
def test_host_allow_list(url, ok):
    assert fetch.host_allowed(url, HOSTS) is ok


# ── download ────────────────────────────────────────────────────────────────

class _Resp(io.BytesIO):
    def __init__(self, data, url):
        super().__init__(data)
        self._url = url

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Opener:
    def __init__(self, data, final_url="https://github.com/x"):
        self.data, self.final_url = data, final_url

    def open(self, req, timeout=None):
        return _Resp(self.data, self.final_url)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def test_download_accepts_the_pinned_file(tmp_path):
    data = b"real release" * 100
    dest = str(tmp_path / "a.zip")
    fetch.download("https://github.com/x", dest, len(data), _sha(data), HOSTS,
                   opener=_Opener(data))
    assert open(dest, "rb").read() == data


def test_tampered_file_is_deleted_and_refused(tmp_path):
    good, evil = b"real release" * 100, b"evil payload" * 100
    dest = str(tmp_path / "a.zip")
    with pytest.raises(VerifyError, match="SHA-256"):
        fetch.download("https://github.com/x", dest, len(evil), _sha(good), HOSTS,
                       opener=_Opener(evil))
    assert not os.path.exists(dest) and not os.path.exists(dest + ".part")


def test_oversized_download_is_cut_off(tmp_path):
    dest = str(tmp_path / "a.zip")
    with pytest.raises(VerifyError, match="larger than"):
        fetch.download("https://github.com/x", dest, 10, _sha(b"x" * 10), HOSTS,
                       opener=_Opener(b"x" * 5000))
    assert not os.listdir(tmp_path)


def test_redirect_to_a_foreign_host_is_refused(tmp_path):
    data = b"x" * 10
    with pytest.raises(VerifyError, match="not allowed"):
        fetch.download("https://github.com/x", str(tmp_path / "a"), 10, _sha(data), HOSTS,
                       opener=_Opener(data, final_url="https://evil.example/x"))
    handler = fetch._AllowListRedirects(HOSTS)
    with pytest.raises(VerifyError):
        handler.redirect_request(None, None, 302, "", {}, "https://evil.example/payload")


def test_disallowed_url_is_never_fetched(tmp_path):
    class Boom:
        def open(self, *a, **k):
            raise AssertionError("must not connect")
    with pytest.raises(VerifyError):
        fetch.download("http://github.com/x", str(tmp_path / "a"), 1, "0" * 64, HOSTS,
                       opener=Boom())


def test_existing_identical_file_is_not_downloaded_again(tmp_path):
    data = b"model" * 50
    dest = tmp_path / "m.gguf"
    dest.write_bytes(data)

    class Boom:
        def open(self, *a, **k):
            raise AssertionError("must not download")
    fetch.download("https://huggingface.co/x", str(dest), len(data), _sha(data), HOSTS,
                   opener=Boom())


# ── unzip ───────────────────────────────────────────────────────────────────

def _zip(path, members):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members:
            if isinstance(name, zipfile.ZipInfo):
                z.writestr(name, data)
            else:
                z.writestr(name, data)
    return str(path)


def test_normal_archive_unpacks(tmp_path):
    a = _zip(tmp_path / "ok.zip", [("llama-server.exe", b"MZ"), ("lib/ggml.dll", b"MZ")])
    dest = tmp_path / "out"
    fetch.safe_unzip(a, str(dest))
    assert (dest / "llama-server.exe").exists() and (dest / "lib" / "ggml.dll").exists()


@pytest.mark.parametrize("bad", [
    "../evil.exe", "lib/../../evil.exe", "/abs/evil.exe", "C:/Windows/evil.exe",
    "..\\evil.exe",
])
def test_zip_slip_is_refused_and_nothing_is_written(tmp_path, bad):
    a = _zip(tmp_path / "bad.zip", [("ok.txt", b"x"), (bad, b"MZ")])
    dest = tmp_path / "out"
    with pytest.raises(VerifyError):
        fetch.safe_unzip(a, str(dest))
    assert not dest.exists()
    assert not (tmp_path / "evil.exe").exists()


def test_symlink_member_is_refused(tmp_path):
    info = zipfile.ZipInfo("link")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    a = _zip(tmp_path / "ln.zip", [(info, "../../Windows")])
    with pytest.raises(VerifyError, match="link"):
        fetch.safe_unzip(a, str(tmp_path / "out"))


# ── catalog and choices ─────────────────────────────────────────────────────

def test_catalog_is_fully_pinned():
    cat = recipes.load_catalog()
    files = [f for v in cat["llama_cpp"]["variants"].values() for f in v["files"]]
    for f in files:
        assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["size"] > 0
        assert fetch.host_allowed(cat["llama_cpp"]["base_url"] + f["name"], cat["allowed_hosts"])
    for m in cat["chat_models"]:
        assert re.fullmatch(r"[0-9a-f]{64}", m["sha256"]) and m["size"] > 0
        assert re.fullmatch(r"[0-9a-f]{40}", m["revision"]), "pin a commit, not a branch"
        assert fetch.host_allowed(recipes._HF.format(**m), cat["allowed_hosts"])


def _probe(gpus=(), cuda=""):
    return {"gpus": list(gpus), "nvidia": {"cuda_version": cuda} if cuda else None}


NV8 = {"name": "RTX 5060 Ti", "vendor": "NVIDIA", "vram_mb": 8151, "compute_cap": "12.0"}
NV24_OLD = {"name": "RTX 3090", "vendor": "NVIDIA", "vram_mb": 24576, "compute_cap": "8.6"}
AMD12 = {"name": "RX 6700 XT", "vendor": "AMD", "vram_mb": 12272}
IGPU = {"name": "Intel UHD", "vendor": "Intel", "vram_mb": 128}


@pytest.mark.parametrize("probe, variant, model", [
    (_probe([NV8], "13.3"), "cuda-13.4", "qwen3.5-9b-q4km"),
    (_probe([NV8], "12.6"), "vulkan", "qwen3.5-9b-q4km"),        # Blackwell needs CUDA 13 build
    (_probe([NV24_OLD], "12.4"), "cuda-12.4", "qwen3.8-27b-q3km"),
    (_probe([NV24_OLD], "11.8"), "vulkan", "qwen3.8-27b-q3km"),
    (_probe([AMD12]), "vulkan", "qwen3.5-9b-q4km"),
    (_probe([IGPU]), "cpu", "lfm3-1.7b"),
    (_probe([]), "cpu", "lfm3-1.7b"),
])
def test_choices(probe, variant, model, tmp_path):
    pl = recipes.plan(probe, recipes.load_catalog(), str(tmp_path))
    assert (pl["variant"], pl["model"]["id"]) == (variant, model)


def test_unknown_choices_are_refused(tmp_path):
    cat = recipes.load_catalog()
    with pytest.raises(ValueError):
        recipes.plan(_probe(), cat, str(tmp_path), model_id="../../evil")
    with pytest.raises(ValueError):
        recipes.plan(_probe(), cat, str(tmp_path), variant="cuda-99")


def test_config_is_added_to_not_overwritten(tmp_path):
    cfg_path = tmp_path / "llama_cpp_config.json"
    cfg_path.write_text(json.dumps({"server_path": "keep.exe",
                                    "models": {"mine": {"path": "keep.gguf"}}}))
    pl = recipes.plan(_probe([NV8], "13.3"), recipes.load_catalog(), str(tmp_path))
    recipes.write_config(pl)
    cfg = json.loads(cfg_path.read_text())
    assert cfg["models"]["mine"] == {"path": "keep.gguf"}
    assert cfg["server_path"] == "keep.exe"
    assert cfg["models"]["qwen3.5-9b-q4km"]["num_gpu_layers"] == 99
    assert any(n.startswith("llama_cpp_config.json.bak-installer-") for n in os.listdir(tmp_path))
    # A second run keeps the user's (possibly hand-tuned) entry.
    cfg["models"]["qwen3.5-9b-q4km"]["num_ctx"] = 4096
    cfg_path.write_text(json.dumps(cfg))
    assert "kept" in recipes.write_config(pl)
    assert json.loads(cfg_path.read_text())["models"]["qwen3.5-9b-q4km"]["num_ctx"] == 4096


def test_find_existing_matches_by_hash_not_name(tmp_path):
    data = b"gguf" * 100
    m = {"size": len(data), "sha256": _sha(data)}
    (tmp_path / "renamed.gguf").write_bytes(data)
    (tmp_path / "same-size-other.gguf").write_bytes(b"x" * len(data))
    assert recipes.find_existing(m, [str(tmp_path)]).endswith("renamed.gguf")
    assert recipes.find_existing({"size": 1, "sha256": "0" * 64}, [str(tmp_path)]) == ""


def test_installer_never_shells_out_or_evals():
    src = ""
    for name in ("fetch.py", "recipes.py", "probe.py", "__main__.py"):
        src += open(os.path.join(os.path.dirname(recipes.__file__), name), encoding="utf-8").read()
    assert "shell=True" not in src
    assert not re.search(r"\beval\(|\bexec\(|os\.system\(|os\.popen\(", src)


# ── Python environment, locks, launchers ────────────────────────────────────

def _lock_entries(path):
    """[(name==ver, [hashes])] from a pip requirements lock."""
    text = re.sub(r"\\\n", " ", open(path, encoding="utf-8").read())  # join continuations
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert not line.startswith("-"), f"option line in a lock: {line!r}"
        req, *rest = line.split()
        out.append((req, [r.split(":", 1)[1] for r in rest if r.startswith("--hash=sha256:")]))
    return out


@pytest.mark.parametrize("lock", ["core", "gui"])
def test_locks_are_exact_and_fully_hashed(lock):
    entries = _lock_entries(recipes.LOCKS[lock])
    assert entries
    text = open(recipes.LOCKS[lock], encoding="utf-8").read()
    for bad in ("--index-url", "--extra-index-url", "--trusted-host", "git+", "http://", " -e "):
        assert bad not in text
    for req, hashes in entries:
        assert re.fullmatch(r"[A-Za-z0-9_.\-]+==[A-Za-z0-9_.+\-]+", req), req
        assert hashes and all(re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes), req


def test_chat_only_lock_leaves_out_the_heavy_stack():
    names = {r.split("==")[0].lower() for r, _ in _lock_entries(recipes.LOCKS["core"])}
    assert {"fastapi", "uvicorn", "requests"} <= names
    assert not names & {"torch", "transformers", "diffusers", "bitsandbytes", "pyqt6"}


def test_env_runs_pip_locked_isolated_and_wheels_only(tmp_path):
    calls = []
    recipes.install_env(str(tmp_path), gui=True, log=lambda *a: None,
                        run=lambda cmd, check: calls.append(cmd))
    venv_cmd, pip_cmd = calls
    assert venv_cmd[1:3] == ["-m", "venv"]
    for flag in ("--isolated", "--require-hashes", "--only-binary=:all:", "--no-input"):
        assert flag in pip_cmd
    assert recipes.LOCKS["core"] in pip_cmd and recipes.LOCKS["gui"] in pip_cmd
    assert not any("index" in c for c in pip_cmd)


def test_env_never_touches_an_existing_venv(tmp_path):
    (tmp_path / "venv").mkdir()
    def boom(*a, **k):
        raise AssertionError("must not run")
    recipes.install_env(str(tmp_path), log=lambda *a: None, run=boom)


def test_python_version_gate():
    assert recipes.python_ok((3, 11)) and recipes.python_ok((3, 13))
    assert not recipes.python_ok((3, 9)) and not recipes.python_ok((3, 14))


def test_launchers_stay_local_and_escape_the_path(tmp_path):
    root = tmp_path / "100% Artifex"
    root.mkdir()
    paths = recipes.write_launchers(str(root), "qwen3.5-9b-q4km", gui=True)
    api = open(next(p for p in paths if "API" in p), encoding="utf-8").read()
    assert "--host 127.0.0.1" in api and "0.0.0.0" not in api
    assert "100%% Artifex" in api                      # cmd would expand a lone %
    assert len(paths) == 3
    with pytest.raises(ValueError):
        recipes.write_launchers(str(root), "x & calc.exe")
