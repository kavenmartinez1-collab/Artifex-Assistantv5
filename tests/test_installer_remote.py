"""Remote access pack: the exposure check must catch every way Artifex could
end up reachable from the internet, and setup must never create one.

Tailscale and netstat are replaced by a fake `run`, so these feed the check
the dangerous setups (a server on 0.0.0.0, Funnel on, no key) and assert it
fails them.
"""

import json
import os
import re
import types

import pytest

from installer import recipes, remote
from installer.fetch import VerifyError


def _netstat(*lines):
    return ("\n  Proto  Local Address  Foreign Address  State  PID\n"
            + "\n".join(lines) + "\n")


def fake_run(netstat="", version="1.104.4", backend="Running", serve=None, lock="ENABLED"):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        out = ""
        if cmd[0] == "netstat":
            out = netstat if cmd[-1] == "TCP" else ""
        elif cmd[1:] == ["version"]:
            out = version + "\n  tailscale commit: x\n"
        elif cmd[1:] == ["status", "--json"]:
            out = json.dumps({"BackendState": backend})
        elif cmd[1:] == ["serve", "status", "--json"]:
            out = json.dumps(serve if serve is not None else {})
        elif cmd[1:] == ["lock", "status"]:
            out = f"Tailnet Lock is {lock}."
        return types.SimpleNamespace(stdout=out, stderr="", returncode=0)
    run.calls = calls
    return run


@pytest.fixture
def ts_installed(monkeypatch):
    monkeypatch.setattr(remote, "tailscale_exe", lambda: r"C:\Program Files\Tailscale\tailscale.exe")


@pytest.fixture
def root(tmp_path):
    (tmp_path / remote.KEY_FILE).write_text("k" * 43)
    return str(tmp_path)


SAFE_SERVE = {"TCP": {"8000": {"TCPForward": "127.0.0.1:8000"}}}


def levels(findings):
    return {f["level"] for f in findings}


@pytest.mark.parametrize("addr, kind", [
    ("127.0.0.1", "loopback"), ("[::1]", "loopback"), ("100.100.10.20", "tailnet"),
    ("[fd7a:115c:a1e0::1234]", "tailnet"), ("0.0.0.0", "any"), ("[::]", "any"),
    ("192.168.1.20", "lan"), ("10.0.0.5", "lan"), ("203.0.113.9", "lan"),
])
def test_classify_address(addr, kind):
    assert remote.classify_address(addr) == kind


def test_safe_setup_passes(ts_installed, root):
    run = fake_run(_netstat("  TCP    127.0.0.1:8000   0.0.0.0:0   LISTENING   111",
                            "  TCP    100.101.1.2:8000 0.0.0.0:0   LISTENING   222"),
                   serve=SAFE_SERVE)
    f = remote.check_exposure(root, 8000, min_version="1.104.4", run=run)
    assert remote.FAIL not in levels(f) and remote.WARN not in levels(f)


@pytest.mark.parametrize("line", [
    "  TCP    0.0.0.0:8000      0.0.0.0:0   LISTENING   111",
    "  TCP    192.168.1.20:8081 0.0.0.0:0   LISTENING   111",
    "  TCP    0.0.0.0:11434     0.0.0.0:0   LISTENING   111",
])
def test_a_port_open_to_the_network_fails(ts_installed, root, line):
    f = remote.check_exposure(root, 8000, run=fake_run(_netstat(line), serve=SAFE_SERVE))
    assert any(x["level"] == remote.FAIL and "listening on" in x["title"] for x in f)


def test_funnel_fails(ts_installed, root):
    serve = dict(SAFE_SERVE, AllowFunnel={"pc.tail.ts.net:443": True})
    f = remote.check_exposure(root, 8000, run=fake_run(serve=serve))
    assert any(x["level"] == remote.FAIL and "Funnel is ON" in x["title"] for x in f)


def test_tailnet_api_without_a_key_fails(ts_installed, tmp_path):
    f = remote.check_exposure(str(tmp_path), 8000, run=fake_run(serve=SAFE_SERVE))
    assert any(x["level"] == remote.FAIL and "without a strong key" in x["title"] for x in f)


def test_forward_to_a_non_loopback_target_warns(ts_installed, root):
    serve = {"TCP": {"8000": {"TCPForward": "192.168.1.20:8000"}}}
    f = remote.check_exposure(root, 8000, run=fake_run(serve=serve))
    assert any(x["level"] == remote.WARN and "forwards to" in x["title"] for x in f)


def test_old_tailscale_and_no_lock_warn(ts_installed, root):
    f = remote.check_exposure(root, 8000, min_version="1.104.4",
                              run=fake_run(version="1.104.2", serve=SAFE_SERVE, lock="DISABLED"))
    titles = [x["title"] for x in f if x["level"] == remote.WARN]
    assert any("older than" in t for t in titles) and any("Tailnet Lock is off" in t for t in titles)


def test_no_tailscale_is_local_only(monkeypatch, root):
    monkeypatch.setattr(remote, "tailscale_exe", lambda: "")
    f = remote.check_exposure(root, 8000, run=fake_run())
    assert any("not installed" in x["title"] for x in f)
    assert remote.FAIL not in levels(f)


def test_check_never_changes_tailscale(ts_installed, root):
    run = fake_run(serve=SAFE_SERVE)
    remote.check_exposure(root, 8000, run=run)
    ts_cmds = [c[1:] for c in run.calls if c[0] != "netstat"]
    assert ts_cmds == [["version"], ["status", "--json"], ["serve", "status", "--json"],
                       ["lock", "status"]]


def test_serve_is_the_only_write_and_never_funnel(ts_installed):
    run = fake_run()
    remote.serve_api(8000, run=run)
    assert run.calls == [[r"C:\Program Files\Tailscale\tailscale.exe", "serve", "--bg",
                          "--tcp=8000", "tcp://127.0.0.1:8000"]]
    for bad in (80, 70000, "8000; calc", None):
        with pytest.raises(ValueError):
            remote.serve_api(bad, run=run)
    src = open(remote.__file__, encoding="utf-8").read()
    assert not re.search(r'_ts\(\s*\[\s*"funnel"', src)


@pytest.mark.parametrize("out, ok", [
    ("Valid|CN=Tailscale Inc., O=Tailscale Inc., L=Toronto", True),
    ("HashMismatch|CN=Tailscale Inc., O=Tailscale Inc.", False),
    ("NotSigned|", False),
    ("Valid|CN=Evil Corp, O=Evil Corp", False),
    ("Valid|CN=x, O=Tailscale Inc.Fake", False),           # lookalike organisation
    ("Valid|CN=O=Tailscale Inc., O=Evil", False),          # the name hidden in another field
])
def test_installer_signature(out, ok):
    run = lambda cmd, **kw: types.SimpleNamespace(stdout=out)
    if ok:
        assert remote.verify_signature("x.msi", run=run)
    else:
        with pytest.raises(VerifyError):
            remote.verify_signature("x.msi", run=run)


def test_api_key_is_strong_and_kept(tmp_path):
    path = remote.ensure_api_key(str(tmp_path))
    key = open(path).read().strip()
    assert len(key) >= 40 and re.fullmatch(r"[A-Za-z0-9_-]+", key)
    assert remote.ensure_api_key(str(tmp_path)) == path
    assert open(path).read().strip() == key                     # not regenerated
    (tmp_path / remote.KEY_FILE).write_text("short")
    remote.ensure_api_key(str(tmp_path))
    assert len(open(path).read().strip()) >= 40                 # a weak key is replaced


def test_phone_launcher_stays_on_loopback(tmp_path):
    path = recipes.write_phone_launcher(str(tmp_path), 8000, "qwen3.5-9b-q4km")
    text = open(path, encoding="utf-8").read()
    assert "--host 127.0.0.1" in text and "0.0.0.0" not in text
    assert 'set /p ARTIFEX_API_KEY=<"artifex_api_key.txt"' in text
    assert "ARTIFEX_PHONE_FULL_TOOLS" not in text
    with pytest.raises(ValueError):
        recipes.write_phone_launcher(str(tmp_path), 8000, "x & del /q *")
    with pytest.raises(ValueError):
        recipes.write_phone_launcher(str(tmp_path), 80)


def test_tailscale_installer_is_pinned():
    t = recipes.load_catalog()["tailscale"]
    assert re.fullmatch(r"[0-9a-f]{64}", t["sha256"]) and t["size"] > 0
    assert t["base_url"].startswith("https://pkgs.tailscale.com/")
