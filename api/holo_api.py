"""
Artifex Assistant V5 — Holo avatars over the REST API (phone app Holo tab).

    GET    /v1/holo/avatars                 uploaded avatars (manifests)
    PUT    /v1/holo/avatars?name=           raw image body -> new avatar, processing
    GET    /v1/holo/avatars/{id}            one manifest (poll status/step)
    PATCH  /v1/holo/avatars/{id}            {name?, persona?, gender?, landmarks?}
    DELETE /v1/holo/avatars/{id}
    GET    /v1/holo/avatars/{id}/{file}     avatar.jpg | depth.png

Uploads take the raw body like /v1/media/upload. Processing (depth, face,
persona) runs on the CPU in the background; see core/services/holo_avatars.py.
Every route needs the bearer key: uploaded pictures are the user's own.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse

_MAX_UPLOAD = 20 * 1024 * 1024


def register_holo_routes(app, check_auth):
    from core.services import holo_avatars

    def _auth(request: Request):
        if not check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

    def _get(avatar_id):
        try:
            return holo_avatars.get(avatar_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="No such avatar")

    @app.get("/v1/holo/avatars")
    async def holo_avatars_list(request: Request):
        _auth(request)
        return {"data": holo_avatars.list_avatars()}

    @app.put("/v1/holo/avatars")
    async def holo_avatar_create(request: Request, name: Optional[str] = ""):
        _auth(request)
        try:
            declared = int(request.headers.get("content-length") or 0)
        except ValueError:
            declared = 0
        if declared > _MAX_UPLOAD:               # refused before reading 20+ MB
            raise HTTPException(status_code=413, detail="Image too large (20 MB max)")
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty upload")
        if len(body) > _MAX_UPLOAD:
            raise HTTPException(status_code=413, detail="Image too large (20 MB max)")
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(None, holo_avatars.create, body, name or "")
        except (OSError, ValueError) as e:      # not an image / too big / bad format
            raise HTTPException(status_code=400, detail=f"Couldn't use that image: {e}")

    @app.get("/v1/holo/avatars/{avatar_id}")
    async def holo_avatar_get(request: Request, avatar_id: str):
        _auth(request)
        return _get(avatar_id)

    @app.patch("/v1/holo/avatars/{avatar_id}")
    async def holo_avatar_update(request: Request, avatar_id: str):
        _auth(request)
        _get(avatar_id)
        try:
            fields = await request.json()
        except ValueError:
            raise HTTPException(status_code=400, detail="Expected a JSON body")
        try:
            return holo_avatars.update(avatar_id, fields if isinstance(fields, dict) else {})
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/v1/holo/avatars/{avatar_id}")
    async def holo_avatar_delete(request: Request, avatar_id: str):
        _auth(request)
        _get(avatar_id)
        holo_avatars.delete(avatar_id)
        return {"deleted": avatar_id}

    @app.get("/v1/holo/avatars/{avatar_id}/{name}")
    async def holo_avatar_file(request: Request, avatar_id: str, name: str):
        _auth(request)
        _get(avatar_id)
        path = holo_avatars.file_path(avatar_id, name)
        if not path:
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path, media_type=holo_avatars.media_type(name),
                            headers={"Cache-Control": "private, no-cache"})   # revalidate: a re-run rewrites depth.png
