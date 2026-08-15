"""
Artifex Assistant V5 — Server-side chat sessions over the REST API.

Persistent conversation store so any device can list and resume previous
chats (the phone client shows them in a sidebar). Distinct from
core/session.py's named snapshots: those append a new timestamped file per
save (the GUI's explicit "save session" flow); this store is keyed by a
stable client-generated id and upserted after every exchange.

    GET    /v1/sessions          list metas, newest-updated first
    GET    /v1/sessions/{id}     full session (messages + meta)
    PUT    /v1/sessions/{id}     upsert {messages, title?} — client sync
    DELETE /v1/sessions/{id}     remove

Design notes:

- The chat endpoint stays stateless/OpenAI-shaped; clients own the sync
  (PUT after each completed exchange). Server-side truth, client-side
  writes — any client that syncs is resumable from any other.
- Ids are client-generated hex (8-32 chars), validated hard since they
  become filenames. Writes are atomic (tmp + os.replace) so a mid-write
  crash can't corrupt a session.
- System messages are stripped on write (they carry runtime state and get
  rebuilt per request); thinking never reaches the store because clients
  sync final content only.
- Bounded: message count and byte size are capped per session, and the
  store prunes to the newest _MAX_SESSIONS on write.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Optional

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from core.logging_config import get_logger

_log = get_logger(__name__)

_ID_RE = re.compile(r"^[a-f0-9]{8,32}$")
_MAX_SESSIONS = 200
_MAX_MESSAGES = 400
_MAX_BYTES = 2_000_000

_write_lock = threading.Lock()


class SessionUpsert(BaseModel):
    messages: list = Field(..., description="Full [{role, content}] history")
    title: Optional[str] = Field(None, max_length=120)
    model: Optional[str] = Field(None, max_length=80)


def _derive_title(messages: list) -> str:
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, list):  # multimodal shape
                content = " ".join(p.get("text", "") for p in content
                                   if isinstance(p, dict))
            text = " ".join(str(content or "").split())
            if text:
                return text[:60]
    return "New chat"


def register_session_routes(app, check_auth, session_dir: str):
    store = os.path.join(session_dir, "api")
    os.makedirs(store, exist_ok=True)

    def _auth(request: Request):
        if not check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

    def _path(session_id: str) -> str:
        if not _ID_RE.match(session_id or ""):
            raise HTTPException(status_code=422,
                                detail="session id must be 8-32 hex chars")
        return os.path.join(store, f"{session_id}.json")

    def _read(path: str) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="No such session")
        except (OSError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Session unreadable: {e}")

    def _prune():
        try:
            files = [os.path.join(store, n) for n in os.listdir(store)
                     if n.endswith(".json")]
            files.sort(key=os.path.getmtime, reverse=True)
            for stale in files[_MAX_SESSIONS:]:
                os.remove(stale)
        except OSError:
            pass

    @app.get("/v1/sessions")
    async def list_sessions(request: Request):
        _auth(request)
        metas = []
        try:
            names = [n for n in os.listdir(store) if n.endswith(".json")]
        except OSError:
            names = []
        for n in names:
            try:
                with open(os.path.join(store, n), "r", encoding="utf-8") as f:
                    data = json.load(f)
                metas.append({
                    "id": data.get("id", n[:-5]),
                    "title": data.get("title", "?"),
                    "updated": data.get("updated", 0),
                    "message_count": len(data.get("messages", [])),
                    "model": data.get("model"),
                })
            except (OSError, json.JSONDecodeError):
                continue
        metas.sort(key=lambda m: m["updated"], reverse=True)
        return {"sessions": metas}

    @app.get("/v1/sessions/{session_id}")
    async def get_session(session_id: str, request: Request):
        _auth(request)
        return _read(_path(session_id))

    @app.put("/v1/sessions/{session_id}")
    async def put_session(session_id: str, body: SessionUpsert, request: Request):
        _auth(request)
        path = _path(session_id)
        messages = [m for m in body.messages
                    if isinstance(m, dict) and m.get("role") != "system"]
        if len(messages) > _MAX_MESSAGES:
            messages = messages[-_MAX_MESSAGES:]
        data = {
            "id": session_id,
            "title": (body.title or "").strip() or _derive_title(messages),
            "model": body.model,
            "updated": round(time.time(), 2),
            "messages": messages,
        }
        raw = json.dumps(data, ensure_ascii=False)
        if len(raw.encode("utf-8")) > _MAX_BYTES:
            raise HTTPException(status_code=413, detail="Session too large")
        with _write_lock:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(raw)
            os.replace(tmp, path)
            _prune()
        return {"ok": True, "id": session_id, "title": data["title"],
                "message_count": len(messages)}

    @app.delete("/v1/sessions/{session_id}")
    async def delete_session(session_id: str, request: Request):
        _auth(request)
        path = _path(session_id)
        try:
            os.remove(path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="No such session")
        return {"ok": True, "deleted": session_id}
