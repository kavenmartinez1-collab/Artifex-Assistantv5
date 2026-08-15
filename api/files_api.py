"""
Artifex Assistant V5 — Workspace file access over the REST API.

Lets the phone browse, read, download, and upload files without being at
the PC. Reach is EXACTLY the agent's filesystem sandbox: every path runs
through core.sandbox.fs_sandbox.check_path (deny list first, then allowed
roots), so the file panel can see precisely what an agent action could
touch and nothing more.

    GET /v1/files?path=<dir>            list a directory (default: repo root)
    GET /v1/files/read?path=<file>      text content, capped, truncation flagged
    GET /v1/files/download?path=<file>  raw file
    PUT /v1/files/upload?dir=&name=     raw request body -> dir/name

Upload takes the raw body (filename in the query) instead of multipart so
no new dependency (python-multipart) enters the tree.
"""
from __future__ import annotations

import os
import re
import time

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse

from core.sandbox.fs_sandbox import check_path
from core.logging_config import get_logger

_log = get_logger(__name__)

_MAX_LIST = 500
_MAX_READ = 512 * 1024          # text read cap
_MAX_UPLOAD = 50 * 1024 * 1024  # raw-body upload cap
_NAME_RE = re.compile(r"^[^\\/:*?\"<>|\x00-\x1f]{1,120}$")  # single component


def _checked(path: str) -> str:
    """abspath + sandbox check; raises 403 with the sandbox's reason."""
    p = os.path.abspath(os.path.expanduser(path or ""))
    reason = check_path(p)
    if reason:
        raise HTTPException(status_code=403, detail=reason)
    return p


def register_file_routes(app, check_auth, default_root: str):
    def _auth(request: Request):
        if not check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

    @app.get("/v1/files")
    async def list_dir(request: Request, path: str = ""):
        _auth(request)
        p = _checked(path or default_root)
        if not os.path.isdir(p):
            raise HTTPException(status_code=404, detail="Not a directory")
        entries = []
        try:
            with os.scandir(p) as it:
                for e in it:
                    try:
                        st = e.stat()
                        entries.append({
                            "name": e.name,
                            "dir": e.is_dir(),
                            "size": 0 if e.is_dir() else st.st_size,
                            "mtime": round(st.st_mtime, 1),
                        })
                    except OSError:
                        continue
        except OSError as e:
            raise HTTPException(status_code=500, detail=str(e))
        entries.sort(key=lambda x: (not x["dir"], x["name"].lower()))
        truncated = len(entries) > _MAX_LIST
        parent = os.path.dirname(p)
        return {
            "path": p,
            # Only offer a parent the sandbox would actually allow —
            # the client uses this for its ".." affordance.
            "parent": parent if (parent != p and not check_path(parent)) else None,
            "entries": entries[:_MAX_LIST],
            "truncated": truncated,
        }

    @app.get("/v1/files/read")
    async def read_file(request: Request, path: str):
        _auth(request)
        p = _checked(path)
        if not os.path.isfile(p):
            raise HTTPException(status_code=404, detail="No such file")
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            raw = f.read(_MAX_READ)
        binary = b"\x00" in raw
        return {
            "path": p,
            "size": size,
            "truncated": size > _MAX_READ,
            "binary": binary,
            "content": "" if binary else raw.decode("utf-8", errors="replace"),
        }

    @app.get("/v1/files/download")
    async def download_file(request: Request, path: str):
        _auth(request)
        p = _checked(path)
        if not os.path.isfile(p):
            raise HTTPException(status_code=404, detail="No such file")
        return FileResponse(p, filename=os.path.basename(p))

    @app.put("/v1/files/upload")
    async def upload_file(request: Request, dir: str, name: str):
        _auth(request)
        if not _NAME_RE.match(name or "") or name in (".", ".."):
            raise HTTPException(status_code=422, detail="Bad filename")
        d = _checked(dir)
        if not os.path.isdir(d):
            raise HTTPException(status_code=404, detail="Not a directory")
        target = _checked(os.path.join(d, name))  # re-check joined path (deny list)
        body = await request.body()
        if len(body) > _MAX_UPLOAD:
            raise HTTPException(status_code=413, detail="Upload too large (50 MB cap)")
        tmp = target + ".uploading"
        try:
            with open(tmp, "wb") as f:
                f.write(body)
            os.replace(tmp, target)
        except OSError as e:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise HTTPException(status_code=500, detail=str(e))
        _log.info("File uploaded via API: %s (%d bytes)", target, len(body))
        return {"ok": True, "path": target, "size": len(body),
                "mtime": round(time.time(), 1)}
