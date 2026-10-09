"""
Artifex Assistant V5 — Purge over the REST API (phone app, full tools).

    GET  /v1/admin/purge        what a purge would delete (labels, counts, sizes)
    POST /v1/admin/purge        {"confirm": "PURGE"} -> shut Artifex down and purge

The purge runs as a detached process (core.purge) that stops this API, so
the POST answers first and the server goes away a couple of seconds
later. Nothing restarts it: the PC has to run start-artifex again.
"""

from __future__ import annotations

import asyncio

from fastapi import HTTPException, Request
from pydantic import BaseModel

from core.logging_config import get_logger

_log = get_logger(__name__)


class PurgeRequest(BaseModel):
    confirm: str = ""


def register_purge_routes(app, check_auth):
    from core import purge

    def _auth(request: Request):
        if not check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

    @app.get("/v1/admin/purge")
    async def purge_preview(request: Request):
        _auth(request)
        rows = await asyncio.get_event_loop().run_in_executor(None, purge.preview)
        return {"targets": rows,
                "total_files": sum(r["files"] for r in rows),
                "total_bytes": sum(r["bytes"] for r in rows)}

    @app.post("/v1/admin/purge")
    async def purge_start(request: Request, body: PurgeRequest):
        _auth(request)
        if body.confirm != "PURGE":
            raise HTTPException(status_code=400, detail='Type PURGE to confirm.')
        _log.warning("Purge requested from %s; shutting down",
                     request.client.host if request.client else "?")
        pid = purge.spawn(delay=2.0)
        return {"purging": True, "pid": pid,
                "note": "Artifex is shutting down. Start it again on the PC."}
