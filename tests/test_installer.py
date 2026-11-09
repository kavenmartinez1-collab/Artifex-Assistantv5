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
import subprocess
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


# ── resume and retry ────────────────────────────────────────────────────────

class _Flaky:
    """Serves `data`, honouring Range, but drops each connection after `per`
    bytes; `fail_first` opens raise outright."""
    def __init__(self, data, per, ranges=True, fail_first=0):
        self.data, self.per, self.ranges, self.fail_first = data, per, ranges, fail_first
        self.requests = []

    def open(self, req, timeout=None):
        rng = req.get_header("Range")
        self.requests.append(rng)
        if self.fail_first:
            self.fail_first -= 1
            raise OSError("connection reset")
        start = int(rng[6:-1]) if (rng and self.ranges) else 0
        resp = _Resp(self.data[start:start + self.per], "https://huggingface.co/x")
        resp.status = 206 if (rng and self.ranges) else 200
        return resp


def test_a_dropped_download_continues_instead_of_starting_over(tmp_path):
    data = bytes(range(256)) * 40
    dest = str(tmp_path / "m.gguf")
    srv = _Flaky(data, per=4000)
    fetch.download("https://huggingface.co/x", dest, len(data), _sha(data), HOSTS,
                   opener=srv, sleep=lambda s: None)
    assert open(dest, "rb").read() == data and not os.path.exists(dest + ".part")
    assert srv.requests == [None, "bytes=4000-", "bytes=8000-"]


def test_a_download_that_keeps_failing_leaves_its_part_for_next_time(tmp_path):
    data = b"z" * 9000
    dest = str(tmp_path / "m.gguf")
    with pytest.raises(fetch.DownloadError, match="continues from there"):
        fetch.download("https://huggingface.co/x", dest, len(data), _sha(data), HOSTS,
                       opener=_Flaky(data, per=1000), sleep=lambda s: None)
    assert os.path.getsize(dest + ".part") == 1000 * fetch.RETRIES and not os.path.exists(dest)
    # The next run asks only for the rest, and the whole file still verifies.
    srv = _Flaky(data, per=9000)
    fetch.download("https://huggingface.co/x", dest, len(data), _sha(data), HOSTS,
                   opener=srv, sleep=lambda s: None)
    assert srv.requests == [f"bytes={1000 * fetch.RETRIES}-"]
    assert open(dest, "rb").read() == data


def test_a_tampered_part_file_cannot_get_past_the_hash(tmp_path):
    data = b"good" * 1000
    dest = str(tmp_path / "m.gguf")
    with open(dest + ".part", "wb") as f:
        f.write(b"evil" * 500)                       # same length as the real first half
    with pytest.raises(VerifyError, match="SHA-256"):
        fetch.download("https://huggingface.co/x", dest, len(data), _sha(data), HOSTS,
                       opener=_Flaky(data, per=len(data)), sleep=lambda s: None)
    assert not os.path.exists(dest) and not os.path.exists(dest + ".part")


def test_a_server_without_range_support_starts_the_file_over(tmp_path):
    data = b"abc" * 1000
    dest = str(tmp_path / "m.gguf")
    with open(dest + ".part", "wb") as f:
        f.write(b"stale bytes")
    fetch.download("https://huggingface.co/x", dest, len(data), _sha(data), HOSTS,
                   opener=_Flaky(data, per=len(data), ranges=False), sleep=lambda s: None)
    assert open(dest, "rb").read() == data


def test_a_refused_connection_is_retried(tmp_path):
    data = b"q" * 100
    dest = str(tmp_path / "a.zip")
    srv = _Flaky(data, per=100, fail_first=2)
    fetch.download("https://github.com/x", dest, len(data), _sha(data), HOSTS,
                   opener=srv, sleep=lambda s: None)
    assert len(srv.requests) == 3 and open(dest, "rb").read() == data


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


def test_an_interrupted_env_is_finished_on_the_next_run(tmp_path):
    def pip_dies(cmd, check):
        if "pip" in cmd:
            raise subprocess.CalledProcessError(1, cmd)
    with pytest.raises(subprocess.CalledProcessError):
        recipes.install_env(str(tmp_path), log=lambda *a: None, run=pip_dies)
    venv = tmp_path / "venv"
    assert (venv / recipes.UNFINISHED_MARKER).exists()
    assert not (venv / recipes.INSTALLER_MARKER).exists()
    assert not recipes.is_installed(str(tmp_path))
    calls = []
    recipes.install_env(str(tmp_path), log=lambda *a: None, run=lambda cmd, check: calls.append(cmd))
    assert any("pip" in c for c in calls)            # it was NOT "kept, not modified"
    assert (venv / recipes.INSTALLER_MARKER).exists()
    assert not (venv / recipes.UNFINISHED_MARKER).exists()


def test_a_server_that_cannot_start_is_not_registered(tmp_path, monkeypatch):
    pl = {"server_exe": str(tmp_path / "llama-server.exe"), "server_dir": str(tmp_path),
          "variant": "cuda-13.4", "backend_files": [], "download_dir": str(tmp_path),
          "model": {"file": "m.gguf", "size": 1, "sha256": "0" * 64, "repo": "r", "id": "m"},
          "model_url": "https://huggingface.co/x", "model_path": str(tmp_path / "m.gguf"),
          "config_path": str(tmp_path / "llama_cpp_config.json")}
    open(pl["server_exe"], "w").close()
    monkeypatch.setattr(recipes.fetch, "download", lambda *a, **k: None)
    monkeypatch.setattr(recipes.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 0xC0000135 - (1 << 32), "", ""))
    with pytest.raises(RuntimeError, match="Visual C\\+\\+ Redistributable"):
        recipes.install_core(pl, {"allowed_hosts": HOSTS}, log=lambda *a: None)
    assert not os.path.exists(pl["config_path"])


# ── the double-click menu ───────────────────────────────────

def _answers(monkeypatch, *lines):
    it = iter(lines)
    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    monkeypatch.setattr("builtins.input", fake_input)


def test_menu_installs_first_and_a_no_is_not_a_failure(tmp_path, monkeypatch):
    from installer import __main__ as cli
    ran = []
    _answers(monkeypatch)                            # nothing typed at all
    assert cli._menu(str(tmp_path), main_fn=lambda argv: ran.append(argv) or 1) == 0
    assert ran == [["install", "--root", str(tmp_path)]]
    # A real failure is passed on, so setup.cmd says it did not finish.
    assert cli._menu(str(tmp_path), main_fn=lambda argv: 5) == 5


def test_menu_choices_run_the_ordinary_commands(tmp_path, monkeypatch):
    from installer import __main__ as cli
    from installer import packs
    monkeypatch.setattr(recipes, "is_installed", lambda root: True)
    started = []
    monkeypatch.setattr(recipes, "start_app", lambda root, gui=True: started.append(gui))
    ran = []
    voice = str(list(packs.PACKS).index("voice") + 1)
    _answers(monkeypatch, "3", voice, "3", "nope", "5", "1", "2", "9", "")
    assert cli._menu(str(tmp_path), main_fn=lambda argv: ran.append(argv) or 0) == 0
    root = str(tmp_path)
    assert ran == [["pack", "voice", "--root", root], ["remote", "check", "--root", root]]
    assert started == [False]                        # no GUI launcher here, so "1" does nothing


def test_questions_survive_a_closed_input(monkeypatch):
    from installer import __main__ as cli
    _answers(monkeypatch)
    assert cli._ask("Install this?") is False and cli._ask("Start?", default=True) is False
    _answers(monkeypatch, "", "", "Y", "n")
    assert cli._ask("a") is False and cli._ask("b", default=True) is True
    assert cli._ask("c") is True and cli._ask("d", default=True) is False


def test_start_app_runs_the_venv_directly_and_locally(tmp_path):
    calls = []
    recipes.start_app(str(tmp_path), gui=False, popen=lambda cmd, **kw: calls.append((cmd, kw)))
    cmd, kw = calls[0]
    assert cmd == [recipes.venv_python(str(tmp_path)), "main.py"] and kw["cwd"] == str(tmp_path)
    assert kw["env"]["ARTIFEX_BACKEND"] == "llama_cpp" and "shell" not in kw
    assert kw["env"]["WEB_GATEWAY_URL"].startswith("http://127.0.0.1:")


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


# ── setup.cmd and its portable-Python fallback ─────────────────

def _repo_file(*parts):
    return open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), *parts),
                encoding="utf-8").read()


def test_bootstrap_python_is_pinned():
    cat = recipes.load_catalog()
    p = cat["bootstrap_python"]
    assert re.fullmatch(r"[0-9a-f]{128}", p["sha512"]) and p["size"] > 0
    assert p["url"] == (f"https://api.nuget.org/v3-flatcontainer/python/{p['version']}/"
                        f"python.{p['version']}.nupkg")
    assert fetch.host_allowed(p["url"], cat["allowed_hosts"])
    major, minor = (int(x) for x in p["version"].split(".")[:2])
    assert recipes.python_ok((major, minor))   # the locks have wheels for it
    assert re.fullmatch(r"[A-Za-z0-9_-]+", p["dest"]) and p["signer"] == "Python Software Foundation"


def test_bootstrap_scripts_verify_and_never_pipe_to_a_shell():
    cmd = _repo_file("setup.cmd")
    ps1 = _repo_file("installer", "get-python.ps1")
    for text in (cmd, ps1):
        low = text.lower()
        for bad in ("invoke-expression", "iex ", "| iex", "downloadstring", "curl ", "bitsadmin",
                    "-encodedcommand", "start-process -verb runas"):
            assert bad not in low, bad
    assert "-m installer menu" in cmd and "-m installer %*" in cmd and r'installer\get-python.ps1' in cmd
    # The ps1 takes every pin from catalog.json and checks all three gates.
    assert "bootstrap_python" in ps1
    assert "Get-FileHash -Algorithm SHA512" in ps1 and "$pin.size" in ps1
    assert "Get-AuthenticodeSignature" in ps1 and "'Valid'" in ps1
    # An existing python folder is refused, never deleted.
    assert "Remove-Item -Recurse -Force $dest" not in ps1


def test_bootstrap_cmd_keeps_crlf_everywhere():
    attrs = _repo_file(".gitattributes")
    assert "*.cmd text eol=crlf" in attrs and "installer/*.ps1 text eol=crlf" in attrs
    assert "/python/" in _repo_file(".gitignore").splitlines()


def test_hints_follow_how_the_installer_was_started(monkeypatch):
    import installer
    monkeypatch.delenv("ARTIFEX_SETUP", raising=False)
    assert installer.hint("pack voice") == "python -m installer pack voice"
    monkeypatch.setenv("ARTIFEX_SETUP", "1")
    assert installer.hint("pack voice") == "setup pack voice"
    assert 'set "ARTIFEX_SETUP=1"' in _repo_file("setup.cmd")
