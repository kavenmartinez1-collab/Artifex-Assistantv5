"""
API keys kept as hashes, one per device.

The server used to read one plaintext key (artifex_api_key.txt, passed in as
ARTIFEX_API_KEY). Anything that could read that file - an agent shell command
included - could drive the API. The key store keeps only a SHA-256 of each
key, so the file is useless to whoever reads it, and each device (the phone,
a laptop, a bench script) gets its own key that can be regenerated or revoked
without touching the others.

Keys are 256-bit random strings, so a plain SHA-256 is the right hash (the
same as GitHub's token storage): slow password hashes exist to stretch
guessable secrets, and these are not guessable.

File format (JSON, written atomically):

    {"version": 1,
     "keys": [{"id": "k_3f9a1c", "label": "phone",
               "hash": "sha256:<64 hex>", "created": "2026-11-12T20:15:00Z"}]}

Command line (prints a new key ONCE and copies it to the clipboard):

    python -m core.api_keys list
    python -m core.api_keys new laptop
    python -m core.api_keys rotate phone          # id or label
    python -m core.api_keys revoke laptop
    python -m core.api_keys migrate artifex_api_key.txt   # keeps the key

The file defaults to ARTIFEX_API_KEYS_FILE; --file overrides it.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import subprocess
import sys
import threading
from datetime import datetime, timezone
from typing import Optional

KEYS_FILE_ENV = "ARTIFEX_API_KEYS_FILE"
DEFAULT_FILE_NAME = "artifex_api_keys.json"
LEGACY_ID = "env"     # the ARTIFEX_API_KEY value, when one is still set

# A key someone types or pastes; anything shorter is not one of ours.
MIN_KEY_LEN = 32


def new_key() -> str:
    """43 url-safe characters = 256 random bits."""
    return secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return "sha256:" + hashlib.sha256(key.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class KeyStoreError(ValueError):
    """A key operation that cannot be done (unknown id, last key, ...)."""


class KeyStore:
    """The hashed keys in one JSON file, plus an optional legacy plaintext
    key from the environment.

    Reads are cheap: the file is re-read only when its modification time
    changes, so a key made or revoked from the command line takes effect on
    the running server without a restart.

    Fail closed: a store that had keys when it was opened never turns into
    "no auth" because its file went missing or empty later - every request
    is refused instead.
    """

    def __init__(self, path: str = "", legacy_key: str = ""):
        self.path = os.path.abspath(path) if path else ""
        self._legacy = legacy_key or ""
        self._legacy_hash = hash_key(self._legacy) if self._legacy else ""
        self._lock = threading.Lock()
        self._keys: list[dict] = []
        self._mtime: Optional[float] = None
        self._reload()
        self.required = bool(self._keys or self._legacy)

    # ── reading ──────────────────────────────────────────────────────────

    def _reload(self) -> None:
        if not self.path:
            return
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            self._keys, self._mtime = [], None
            return
        if mtime == self._mtime:
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            keys = [k for k in data.get("keys", [])
                    if isinstance(k, dict) and str(k.get("hash", "")).startswith("sha256:")]
        except (OSError, ValueError, AttributeError):
            # A half-written or hand-broken file: keep what we had rather
            # than locking everyone out (or worse, letting everyone in).
            return
        self._keys, self._mtime = keys, mtime

    def configured(self) -> bool:
        """Whether requests must carry a key."""
        return self.required or bool(self._legacy) or bool(self._current())

    def _current(self) -> list[dict]:
        with self._lock:
            self._reload()
            return list(self._keys)

    def match(self, token: str) -> Optional[dict]:
        """The key entry `token` belongs to, or None. Every stored hash is
        compared (constant time each), so timing says nothing about which."""
        if not token:
            return None
        h = hash_key(token)
        found = None
        for k in self._current():
            if hmac.compare_digest(h, k["hash"]):
                found = k
        if self._legacy_hash and hmac.compare_digest(h, self._legacy_hash):
            found = found or {"id": LEGACY_ID, "label": "ARTIFEX_API_KEY", "created": ""}
        return found

    def list(self) -> list[dict]:
        out = [{"id": k.get("id", ""), "label": k.get("label", ""),
                "created": k.get("created", "")} for k in self._current()]
        if self._legacy:
            out.append({"id": LEGACY_ID, "label": "ARTIFEX_API_KEY (environment)",
                        "created": "", "legacy": True})
        return out

    # ── writing ──────────────────────────────────────────────────────────

    def _write(self, keys: list[dict]) -> None:
        if not self.path:
            raise KeyStoreError(f"no key file configured (set {KEYS_FILE_ENV})")
        folder = os.path.dirname(self.path)
        os.makedirs(folder, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"version": 1, "keys": keys}, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, self.path)
        self._keys = keys
        try:
            self._mtime = os.path.getmtime(self.path)
        except OSError:
            self._mtime = None
        self.required = True

    def _find(self, keys: list[dict], ref: str) -> int:
        ref = (ref or "").strip()
        for i, k in enumerate(keys):
            if k.get("id") == ref:
                return i
        hits = [i for i, k in enumerate(keys) if k.get("label") == ref]
        if len(hits) == 1:
            return hits[0]
        if hits:
            raise KeyStoreError(f"more than one key is labelled {ref!r}; use its id")
        if ref == LEGACY_ID:
            raise KeyStoreError("the ARTIFEX_API_KEY key comes from the environment; "
                            "run `python -m core.api_keys migrate` to move it into "
                            "the key file first")
        raise KeyStoreError(f"no key {ref!r}")

    def _new_id(self, keys: list[dict]) -> str:
        taken = {k.get("id") for k in keys}
        while True:
            kid = "k_" + secrets.token_hex(3)
            if kid not in taken:
                return kid

    def add(self, label: str, key: Optional[str] = None) -> tuple[dict, str]:
        """A new key for `label`. Returns (entry, key); the key is not kept."""
        label = (label or "").strip()[:60] or "device"
        key = key or new_key()
        if len(key) < MIN_KEY_LEN:
            raise KeyStoreError(f"a key must be at least {MIN_KEY_LEN} characters")
        with self._lock:
            self._reload()
            keys = list(self._keys)
            if any(hmac.compare_digest(hash_key(key), k["hash"]) for k in keys):
                raise KeyStoreError("that key is already in the file")
            entry = {"id": self._new_id(keys), "label": label,
                     "hash": hash_key(key), "created": _now()}
            self._write(keys + [entry])
        return _public(entry), key

    def rotate(self, ref: str) -> tuple[dict, str]:
        """A new key in place of `ref` (same id and label); the old one stops
        working at once."""
        key = new_key()
        with self._lock:
            self._reload()
            keys = [dict(k) for k in self._keys]
            i = self._find(keys, ref)
            keys[i]["hash"] = hash_key(key)
            keys[i]["created"] = _now()
            self._write(keys)
            entry = keys[i]
        return _public(entry), key

    def revoke(self, ref: str) -> dict:
        with self._lock:
            self._reload()
            keys = list(self._keys)
            i = self._find(keys, ref)
            if len(keys) == 1 and not self._legacy:
                raise KeyStoreError("that is the last key; make a new one first "
                                "(or rotate this one)")
            gone = keys.pop(i)
            self._write(keys)
        return _public(gone)


def _public(entry: dict) -> dict:
    return {"id": entry.get("id", ""), "label": entry.get("label", ""),
            "created": entry.get("created", "")}


def migrate_plaintext(store: KeyStore, txt_path: str, label: str = "phone") -> dict:
    """Move the key in an old artifex_api_key.txt into the store, unchanged,
    so devices that already have it keep working. The text file is left for
    the caller to delete."""
    with open(txt_path, encoding="utf-8") as fh:
        key = fh.read().strip()
    if len(key) < MIN_KEY_LEN:
        raise KeyStoreError(f"{txt_path} does not hold a key")
    hit = store.match(key)
    if hit is not None and hit.get("id") != LEGACY_ID:
        return _public(hit)          # already moved: nothing to do
    entry, _ = store.add(label, key=key)
    return entry


def store_from_env() -> KeyStore:
    return KeyStore(os.environ.get(KEYS_FILE_ENV, ""),
                    legacy_key=os.environ.get("ARTIFEX_API_KEY", ""))


# ── command line ─────────────────────────────────────────────────────────

def _to_clipboard(text: str) -> bool:
    if os.name != "nt":
        return False
    try:
        subprocess.run(["clip"], input=text.encode("utf-16-le"), check=True,
                       timeout=10)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _show_key(entry: dict, key: str, clipboard: bool) -> None:
    print(f"Key {entry['id']} ({entry['label']}):\n\n    {key}\n")
    copied = clipboard and _to_clipboard(key)
    print("Copied to the clipboard. " if copied else "", end="")
    print("Paste it into the device's settings now: only its hash is kept, "
          "so it cannot be shown again (lost it? rotate it).")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m core.api_keys",
                                description="Manage Artifex API keys (stored as hashes).")
    p.add_argument("--file", default=os.environ.get(KEYS_FILE_ENV, ""),
                   help=f"key file (default: ${KEYS_FILE_ENV})")
    p.add_argument("--no-clipboard", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("new"); s.add_argument("label")
    s = sub.add_parser("rotate"); s.add_argument("ref", help="id or label")
    s = sub.add_parser("revoke"); s.add_argument("ref", help="id or label")
    s = sub.add_parser("migrate"); s.add_argument("txt", help="old artifex_api_key.txt")
    s.add_argument("--label", default="phone")
    a = p.parse_args(argv)
    if not a.file:
        p.error(f"no key file: pass --file or set {KEYS_FILE_ENV}")
    store = KeyStore(a.file)
    try:
        if a.cmd == "list":
            rows = store.list()
            if not rows:
                print(f"No keys in {store.path}.")
            for k in rows:
                print(f"{k['id']:10} {k['label']:24} {k['created']}")
        elif a.cmd == "new":
            entry, key = store.add(a.label)
            _show_key(entry, key, not a.no_clipboard)
        elif a.cmd == "rotate":
            entry, key = store.rotate(a.ref)
            _show_key(entry, key, not a.no_clipboard)
            print("The old key for it stopped working.")
        elif a.cmd == "revoke":
            gone = store.revoke(a.ref)
            print(f"Revoked {gone['id']} ({gone['label']}).")
        elif a.cmd == "migrate":
            entry = migrate_plaintext(store, a.txt, a.label)
            print(f"Moved the key in {a.txt} into {store.path} as "
                  f"{entry['id']} ({entry['label']}); devices that have it keep "
                  f"working. Delete {a.txt} once the server runs from the key file.")
    except (KeyStoreError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
