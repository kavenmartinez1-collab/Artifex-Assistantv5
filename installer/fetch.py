"""Download, verify and unpack — the parts of the installer that touch the network.

Rules, because an installer is exactly what a supply-chain attack aims at:

- HTTPS only, and only to hosts in the catalog's allow-list, checked again
  on every redirect (a release download hops github.com -> a CDN).
- Every file has a size and SHA-256 pinned in catalog.json (in the repo,
  not fetched from the server). It is written to a .part file, hashed while
  it streams, and only renamed into place when the hash matches. A mismatch
  deletes the file and stops.
- Nothing downloaded is executed here. Archives are unpacked with every
  member checked: no absolute paths, drive letters, ".." or links, and every
  target must resolve inside the destination folder ("zip slip").
"""

import hashlib
import http.client
import os
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

CHUNK = 1 << 20


class VerifyError(Exception):
    """A download or archive failed a safety check. Nothing was installed."""


def host_allowed(url: str, allowed_hosts) -> bool:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        return False
    host = parts.hostname.lower()
    for rule in allowed_hosts:
        rule = rule.lower()
        if rule.startswith("."):
            if host.endswith(rule) and len(host) > len(rule):
                return True
        elif host == rule:
            return True
    return False


class _AllowListRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_hosts):
        super().__init__()
        self.allowed_hosts = allowed_hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not host_allowed(newurl, self.allowed_hosts):
            raise VerifyError(f"refused redirect to a host that is not allowed: "
                              f"{urllib.parse.urlsplit(newurl).hostname}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def file_matches(path: str, size: int, sha256: str) -> bool:
    """True when `path` already holds exactly the pinned file."""
    try:
        if os.path.getsize(path) != size:
            return False
    except OSError:
        return False
    return sha256_file(path) == sha256.lower()


class DownloadError(RuntimeError):
    """The network gave out. The part already fetched is kept, so running the
    same command again continues from there."""


RETRIES = 5                 # tries per download before giving up for now
_NO_RETRY = (400, 401, 403, 404, 410, 451)


def _hash_prefix(part: str, size: int):
    """(sha256 of what is already in `part`, its length), or a fresh hash and
    0 when there is nothing usable to continue from."""
    h = hashlib.sha256()
    try:
        have = os.path.getsize(part)
    except OSError:
        return h, 0
    if not 0 < have < size:
        return h, 0
    with open(part, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h, have


def download(url: str, dest: str, size: int, sha256: str, allowed_hosts,
             progress=None, opener=None, sleep=time.sleep) -> str:
    """Fetch `url` to `dest` only if it matches the pinned size and SHA-256.

    A dropped connection is retried, and a download that still can't finish
    keeps its .part file: the next run asks the server for the rest (an HTTP
    Range request) instead of starting a multi-GB file over. The kept bytes
    are hashed again first, so the SHA-256 checked at the end always covers
    the whole file; a .part that was tampered with fails it and is deleted.
    """
    if not host_allowed(url, allowed_hosts):
        raise VerifyError(f"refused: {url} is not https on an allowed host")
    if file_matches(dest, size, sha256):
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    opener = opener or urllib.request.build_opener(_AllowListRedirects(allowed_hosts))
    name = os.path.basename(dest)
    try:
        last = None
        for attempt in range(RETRIES):
            if attempt:
                sleep(min(2 ** attempt, 30))
            h, got = _hash_prefix(part, size)
            headers = {"User-Agent": "Artifex-Installer/1"}
            if got:
                headers["Range"] = f"bytes={got}-"
            try:
                req = urllib.request.Request(url, headers=headers)
                with opener.open(req, timeout=60) as resp:
                    if not host_allowed(resp.geturl(), allowed_hosts):
                        raise VerifyError("refused: download ended on a host that is not allowed")
                    # 206 = the server is sending only the rest. Anything else
                    # is the whole file again: start the hash and file over.
                    if got and getattr(resp, "status", 200) != 206:
                        h, got = hashlib.sha256(), 0
                    with open(part, "ab" if got else "wb") as out:
                        if progress and got:
                            progress(got, size)
                        while True:
                            block = resp.read(CHUNK)
                            if not block:
                                break
                            got += len(block)
                            if got > size:
                                raise VerifyError(f"{name}: larger than the pinned "
                                                  f"{size} bytes; refused")
                            h.update(block)
                            out.write(block)
                            if progress:
                                progress(got, size)
            except VerifyError:
                raise
            except urllib.error.HTTPError as e:
                if e.code == 416:          # our .part doesn't fit the server's file
                    if os.path.exists(part):
                        os.remove(part)
                elif e.code in _NO_RETRY:
                    raise DownloadError(f"{name}: the server answered HTTP {e.code}") from e
                last = e
                continue
            except (OSError, http.client.HTTPException) as e:
                last = e                   # dropped connection, timeout, DNS...
                continue
            if got < size:                 # closed early without an error
                last = OSError(f"connection closed after {got} of {size} bytes")
                continue
            if h.hexdigest() != sha256.lower():
                raise VerifyError(f"{name}: SHA-256 does not match the pinned "
                                  "value. The file was deleted and nothing was installed.")
            os.replace(part, dest)
            return dest
        raise DownloadError(
            f"{name}: the download kept stopping ({last}). What was fetched so far is "
            "kept; run the same command again and it continues from there.")
    except VerifyError:
        # Only a failed safety check throws the partial file away. A dropped
        # network or a closed window (Ctrl+C) keeps it for the next run.
        if os.path.exists(part):
            os.remove(part)
        raise


def _member_is_link(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def safe_unzip(archive: str, dest: str) -> list:
    """Unpack `archive` into `dest` after checking every member. Returns names.

    Unpacks into a temporary folder next to `dest` first, so a refused archive
    leaves nothing half-written behind.
    """
    dest = os.path.abspath(dest)
    parent = os.path.dirname(dest)
    os.makedirs(parent, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names = []
        for info in z.infolist():
            name = info.filename.replace("\\", "/")
            if (not name or name.startswith("/") or ":" in name.split("/")[0]
                    or ".." in name.split("/")):
                raise VerifyError(f"refused archive member path: {info.filename!r}")
            if _member_is_link(info):
                raise VerifyError(f"refused link in archive: {info.filename!r}")
            target = os.path.abspath(os.path.join(dest, name))
            if os.path.commonpath([target, dest]) != dest:
                raise VerifyError(f"refused member outside the folder: {info.filename!r}")
            names.append(name)
        tmp = tempfile.mkdtemp(prefix=".unpack-", dir=parent)
        try:
            z.extractall(tmp)
            os.makedirs(dest, exist_ok=True)
            for entry in os.listdir(tmp):
                src, dst = os.path.join(tmp, entry), os.path.join(dest, entry)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return names
