"""download_range (one tensor out of a big shard) and safe_untar (Strata engine archives)."""

import hashlib
import io
import os
import tarfile

import pytest

from installer import fetch

HOSTS = ["huggingface.co", ".huggingface.co", ".hf.co"]
BLOB = bytes(range(256)) * 64          # the "shard"


def _sha(b):
    return hashlib.sha256(b).hexdigest()


class _Resp(io.BytesIO):
    def __init__(self, data, status, content_range):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range}

    def geturl(self):
        return "https://huggingface.co/x"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _RangeServer:
    def __init__(self, per=None, honour=True, fail_first=0):
        self.per, self.honour, self.fail_first, self.asked = per, honour, fail_first, []

    def open(self, req, timeout=None):
        rng = req.get_header("Range")
        self.asked.append(rng)
        if self.fail_first:
            self.fail_first -= 1
            raise OSError("reset")
        a, b = (int(x) for x in rng[6:].split("-"))
        if not self.honour:
            return _Resp(BLOB[:b - a + 1], 200, "")
        chunk = BLOB[a:b + 1][: self.per or None]
        return _Resp(chunk, 206, f"bytes {a}-{b}/{len(BLOB)}")


def test_range_download_takes_exactly_the_span(tmp_path):
    dest = str(tmp_path / "t.bin")
    want = BLOB[1000:5000]
    fetch.download_range("https://huggingface.co/r/x.safetensors", dest, 1000, 4999, _sha(want), HOSTS,
                         opener=_RangeServer(), sleep=lambda s: None)
    assert open(dest, "rb").read() == want


def test_range_download_resumes_after_a_drop(tmp_path):
    dest = str(tmp_path / "t.bin")
    srv = _RangeServer(per=1500)
    fetch.download_range("https://huggingface.co/x", dest, 1000, 4999, _sha(BLOB[1000:5000]), HOSTS,
                         opener=srv, sleep=lambda s: None)
    assert srv.asked == ["bytes=1000-4999", "bytes=2500-4999", "bytes=4000-4999"]


def test_a_server_that_ignores_range_is_refused(tmp_path):
    dest = str(tmp_path / "t.bin")
    with pytest.raises(fetch.VerifyError, match="did not send bytes"):
        fetch.download_range("https://huggingface.co/x", dest, 1000, 4999, _sha(BLOB[1000:5000]), HOSTS,
                             opener=_RangeServer(honour=False), sleep=lambda s: None)
    assert not os.path.exists(dest) and not os.path.exists(dest + ".part")


def test_range_download_checks_the_hash(tmp_path):
    with pytest.raises(fetch.VerifyError, match="SHA-256"):
        fetch.download_range("https://huggingface.co/x", str(tmp_path / "t.bin"), 0, 99, "0" * 64, HOSTS,
                             opener=_RangeServer(), sleep=lambda s: None)


def test_range_download_refuses_a_foreign_host(tmp_path):
    with pytest.raises(fetch.VerifyError):
        fetch.download_range("https://evil.example/x", str(tmp_path / "t.bin"), 0, 9, "0" * 64, HOSTS)


def _tar(tmp_path, members):
    path = tmp_path / "a.tar.gz"
    with tarfile.open(path, "w:gz") as t:
        for m, data in members:
            t.addfile(m, io.BytesIO(data) if data is not None else None)
    return str(path)


def _file(name, data=b"x"):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    return ti, data


def test_untar_unpacks_plain_files(tmp_path):
    arc = _tar(tmp_path, [_file("engine/strata.exe", b"MZ"), _file("engine/rocm/bin/a.dll", b"dll")])
    fetch.safe_untar(arc, str(tmp_path / "out"))
    assert (tmp_path / "out" / "engine" / "rocm" / "bin" / "a.dll").read_bytes() == b"dll"


@pytest.mark.parametrize("bad", ["../evil.txt", "/abs.txt", "C:/x.txt", "a/../../evil.txt"])
def test_untar_refuses_escaping_paths(tmp_path, bad):
    arc = _tar(tmp_path, [_file(bad)])
    with pytest.raises(fetch.VerifyError):
        fetch.safe_untar(arc, str(tmp_path / "out"))
    assert not (tmp_path / "evil.txt").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE])
def test_untar_refuses_links_and_devices(tmp_path, kind):
    ti = tarfile.TarInfo("engine/link")
    ti.type = kind
    ti.linkname = "../../etc/passwd"
    arc = _tar(tmp_path, [(ti, None)])
    with pytest.raises(fetch.VerifyError):
        fetch.safe_untar(arc, str(tmp_path / "out"))
