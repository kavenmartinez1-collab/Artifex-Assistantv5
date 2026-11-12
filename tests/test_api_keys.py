"""
Hashed API keys (core.api_keys) and the server's use of them: only hashes on
disk, one key per device, rotate/revoke take effect without a restart, the
old plaintext key migrates unchanged, and a store never fails open.
"""

import json
import os
import time

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from core import api_keys
from core.api_keys import KeyStore, KeyStoreError, hash_key, migrate_plaintext


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "keys.json")


def _bump(path):
    """Make the next read see a newer mtime (coarse clocks on some FS)."""
    st = os.stat(path)
    os.utime(path, (st.st_atime, st.st_mtime + 2))


def test_only_hashes_are_written(path):
    entry, key = KeyStore(path).add("phone")
    text = open(path, encoding="utf-8").read()
    assert key not in text
    assert hash_key(key) in text
    assert len(key) >= 43
    assert json.loads(text)["keys"][0]["label"] == "phone"


def test_match_finds_the_right_device(path):
    s = KeyStore(path)
    p, pk = s.add("phone")
    l, lk = s.add("laptop")
    assert s.match(pk)["id"] == p["id"]
    assert s.match(lk)["id"] == l["id"]
    assert s.match("nope" * 11) is None
    assert s.match("") is None


def test_rotate_replaces_only_that_key(path):
    s = KeyStore(path)
    p, pk = s.add("phone")
    l, lk = s.add("laptop")
    p2, pk2 = s.rotate("phone")
    assert p2["id"] == p["id"] and pk2 != pk
    assert s.match(pk) is None and s.match(pk2)["id"] == p["id"]
    assert s.match(lk) is not None


def test_revoke_and_last_key_guard(path):
    s = KeyStore(path)
    p, pk = s.add("phone")
    l, lk = s.add("laptop")
    s.revoke(l["id"])
    assert s.match(lk) is None
    with pytest.raises(KeyStoreError):
        s.revoke("phone")               # the last key: would lock everyone out


def test_ambiguous_label_needs_an_id(path):
    s = KeyStore(path)
    s.add("phone")
    s.add("phone")
    with pytest.raises(KeyStoreError):
        s.rotate("phone")


def test_changes_from_another_process_are_picked_up(path):
    server_side = KeyStore(path)
    _, k1 = server_side.add("phone")
    cli_side = KeyStore(path)
    _, k2 = cli_side.add("laptop")
    _bump(path)
    assert server_side.match(k2) is not None       # no restart needed
    cli_side.revoke("laptop")
    _bump(path)
    assert server_side.match(k2) is None


def test_fails_closed_when_the_file_disappears(path):
    s = KeyStore(path)
    _, k = s.add("phone")
    os.remove(path)
    assert s.configured() is True                  # still requires a key ...
    assert s.match(k) is None                      # ... and none matches


def test_broken_file_keeps_the_last_good_keys(path):
    s = KeyStore(path)
    _, k = s.add("phone")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{ half written")
    _bump(path)
    assert s.match(k) is not None


def test_no_file_no_env_means_auth_off(tmp_path):
    s = KeyStore(str(tmp_path / "missing.json"))
    assert s.configured() is False


def test_legacy_env_key_still_works(path):
    s = KeyStore(path, legacy_key="L" * 43)
    assert s.configured()
    assert s.match("L" * 43)["id"] == api_keys.LEGACY_ID
    with pytest.raises(KeyStoreError):
        s.rotate(api_keys.LEGACY_ID)


def test_migrate_keeps_the_same_key(path, tmp_path):
    txt = tmp_path / "artifex_api_key.txt"
    txt.write_text("M" * 64 + "\n")
    s = KeyStore(path)
    entry = migrate_plaintext(s, str(txt))
    assert s.match("M" * 64)["id"] == entry["id"]
    assert migrate_plaintext(s, str(txt))["id"] == entry["id"]   # idempotent
    assert len(s.list()) == 1


def test_cli_new_list_rotate_revoke(path, capsys):
    assert api_keys.main(["--file", path, "--no-clipboard", "new", "laptop"]) == 0
    out = capsys.readouterr().out
    key = [l.strip() for l in out.splitlines() if len(l.strip()) >= 43][0]
    assert KeyStore(path).match(key)["label"] == "laptop"
    assert api_keys.main(["--file", path, "list"]) == 0
    assert "laptop" in capsys.readouterr().out
    assert api_keys.main(["--file", path, "--no-clipboard", "new", "phone"]) == 0
    capsys.readouterr()
    assert api_keys.main(["--file", path, "revoke", "laptop"]) == 0
    assert KeyStore(path).match(key) is None
    assert api_keys.main(["--file", path, "revoke", "nobody"]) == 1


# ── the server ──────────────────────────────────────────────────────────────

class _Req:
    def __init__(self, key):
        self.headers = {"Authorization": "Bearer " + key} if key else {}
        self.client = type("C", (), {"host": "198.51.100.9"})()
        self.state = type("S", (), {})()


def test_server_check_auth_uses_the_store(path, monkeypatch):
    from api import server
    s = KeyStore(path)
    _, k = s.add("phone")
    monkeypatch.setattr(server, "_keystore", s)
    monkeypatch.setattr(server, "_auth_failures", {})
    req = _Req(k)
    assert server._check_auth(req) is True
    assert req.state.api_key_id.startswith("k_")
    assert server._check_auth(_Req("x" * 43)) is False
    assert server._check_auth(_Req("")) is False


@pytest.fixture
def phone_client(path, monkeypatch):
    from api import server
    s = KeyStore(path)
    _, k = s.add("phone")
    monkeypatch.setattr(server, "_keystore", s)
    monkeypatch.setattr(server, "_api_key", True)
    monkeypatch.setattr(server, "_auth_failures", {})
    monkeypatch.setenv("ARTIFEX_PHONE_APP", "1")
    monkeypatch.delenv("ARTIFEX_PHONE_FULL_TOOLS", raising=False)
    client = TestClient(server.create_app())
    return client, s, k


def _h(key):
    return {"Authorization": "Bearer " + key}


def test_phone_lists_makes_rotates_and_revokes_keys(phone_client):
    client, s, phone = phone_client
    r = client.get("/v1/auth/keys", headers=_h(phone))
    assert r.status_code == 200
    me = r.json()["current"]
    assert [k["label"] for k in r.json()["keys"]] == ["phone"]
    assert client.get("/v1/auth/keys", headers=_h("x" * 43)).status_code == 401

    r = client.post("/v1/auth/keys", json={"label": "laptop"}, headers=_h(phone))
    laptop = r.json()
    assert r.status_code == 200 and len(laptop["key"]) >= 43
    assert client.get("/v1/auth/keys", headers=_h(laptop["key"])).status_code == 200

    # revoking the key you are using is refused; another device's works
    assert client.delete(f"/v1/auth/keys/{me}", headers=_h(phone)).status_code == 400
    assert client.delete(f"/v1/auth/keys/{laptop['id']}", headers=_h(phone)).status_code == 200
    assert client.get("/v1/auth/keys", headers=_h(laptop["key"])).status_code == 401

    # regenerate this device's key: the old one stops at once
    r = client.post("/v1/auth/keys/rotate", json={}, headers=_h(phone))
    new = r.json()
    assert new["id"] == me and new["was_current"] is True
    assert client.get("/v1/auth/keys", headers=_h(phone)).status_code == 401
    assert client.get("/v1/auth/keys", headers=_h(new["key"])).status_code == 200
    assert new["key"] not in open(s.path, encoding="utf-8").read()
