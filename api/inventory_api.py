"""Model inventory over REST: what models exist, which are ready, and adding new ones.

    GET  /v1/inventory                 every reachable model with its status
                                       (ready / unconfigured / broken)
    GET  /v1/inventory/proposal?path=  the config entry "Add with defaults"
                                       would write for an unconfigured GGUF
    POST /v1/inventory/add             write that entry (optionally renamed or
                                       with another num_ctx); only registered
                                       when allow_write is set

The proposal and add routes accept only paths the scan itself reported as
unconfigured, so they can't be used to probe arbitrary files.
"""
# No "from __future__ import annotations": FastAPI must see the real Request type.
import asyncio
from typing import Optional


def register_inventory_routes(app, check_auth, allow_write: bool) -> None:
    from fastapi import HTTPException, Request
    from pydantic import BaseModel, Field

    from core import model_inventory as mi

    def _auth(request: Request):
        if not check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

    def _unconfigured(path: str) -> str:
        inv = mi.scan(include_other_backends=False)
        for m in inv["models"]:
            if m["status"] == "unconfigured" and m["path"] and mi._norm(m["path"]) == mi._norm(path):
                return m["path"]
        raise HTTPException(status_code=404, detail="not an unconfigured model in the inventory")

    @app.get("/v1/inventory")
    async def inventory(request: Request):
        _auth(request)
        return await asyncio.to_thread(mi.scan)

    @app.get("/v1/inventory/proposal")
    async def proposal(path: str, request: Request):
        _auth(request)
        real = await asyncio.to_thread(_unconfigured, path)
        try:
            return await asyncio.to_thread(mi.propose_entry, real)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    if not allow_write:
        return

    class AddRequest(BaseModel):
        path: str
        name: Optional[str] = Field(None, max_length=80)
        num_ctx: Optional[int] = Field(None, ge=2048, le=1048576)

    @app.post("/v1/inventory/add")
    async def add(body: AddRequest, request: Request):
        _auth(request)
        real = await asyncio.to_thread(_unconfigured, body.path)
        try:
            prop = await asyncio.to_thread(mi.propose_entry, real)
            name = body.name or prop["name"]
            if body.num_ctx:
                prop["entry"]["num_ctx"] = body.num_ctx
            backup = await asyncio.to_thread(mi.add_entry, name, prop["entry"])
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {"ok": True, "name": name, "entry": prop["entry"], "backup": backup}
