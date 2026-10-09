"""
Artifex Assistant V5 — Media tools and voice over the REST API (phone app).

    GET  /v1/media/tools              tools, the models each can use, voices
    PUT  /v1/media/upload?name=       raw body -> stored file (source images)
    POST /v1/media/jobs               {tool, prompt, model?, file_id?, ...}
    GET  /v1/media/jobs[/{id}]        job status / result file_id
    GET  /v1/media/file/{file_id}     a stored input or output
    POST /v1/voice/tts                {text, voice?} -> audio/wav
    POST /v1/voice/stt?model=         raw recorded audio -> {text}

Uploads take the raw request body (name in the query) like files_api, so
python-multipart isn't needed. Generation runs as a background job that
holds the GPU (the chat LLM is unloaded and reloads on the next chat);
TTS/STT run on the CPU and answer directly.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

_MAX_UPLOAD = 30 * 1024 * 1024


class MediaJobRequest(BaseModel):
    tool: str = Field(..., example="image")
    prompt: Optional[str] = ""
    model: Optional[str] = None          # an id from GET /v1/media/tools
    file_id: Optional[str] = None        # source image for edit / restore
    size: Optional[str] = "square"       # image: square | portrait | landscape
    seed: Optional[int] = -1
    duration: Optional[int] = 10         # music seconds
    upscale: Optional[int] = 2           # restore: 2 or 4


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None


def register_media_routes(app, check_auth):
    from core.services import get_file_manager, media_tools, voice_io

    def _auth(request: Request):
        if not check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

    @app.get("/v1/media/tools")
    async def media_tools_catalog(request: Request):
        _auth(request)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, media_tools.catalog)

    @app.put("/v1/media/upload")
    async def media_upload(request: Request, name: str = "upload.jpg"):
        _auth(request)
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty upload")
        if len(body) > _MAX_UPLOAD:
            raise HTTPException(status_code=413, detail="Upload too large (30 MB max)")
        from core.services.file_manager import _detect_file_type
        if _detect_file_type(name) != "image":
            raise HTTPException(status_code=400, detail="Only images can be uploaded here")
        rec = get_file_manager().store_upload(body, os.path.basename(name))
        return {"file_id": rec.file_id, "filename": rec.original_name}

    @app.post("/v1/media/jobs")
    async def media_job_start(request: Request, body: MediaJobRequest):
        _auth(request)
        try:
            return media_tools.start_job(body.tool, body.model_dump())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/v1/media/jobs")
    async def media_job_list(request: Request):
        _auth(request)
        return {"data": media_tools.list_jobs()}

    @app.get("/v1/media/jobs/{job_id}")
    async def media_job_get(request: Request, job_id: str):
        _auth(request)
        job = media_tools.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job (the server may have restarted)")
        return job

    @app.get("/v1/media/file/{file_id}")
    async def media_file(request: Request, file_id: str):
        _auth(request)
        fm = get_file_manager()
        rec, path = fm.get_file(file_id), fm.get_file_path(file_id)
        if rec is None or path is None:
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path, media_type=rec.content_type,
                            filename=rec.original_name)

    @app.post("/v1/voice/tts")
    async def voice_tts(request: Request, body: TTSRequest):
        _auth(request)
        loop = asyncio.get_event_loop()
        try:
            wav = await loop.run_in_executor(None, voice_io.tts, body.text, body.voice)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return Response(content=wav, media_type="audio/wav")

    @app.post("/v1/voice/stt")
    async def voice_stt(request: Request, model: Optional[str] = None):
        _auth(request)
        audio = await request.body()
        if not audio:
            raise HTTPException(status_code=400, detail="No audio received")
        if len(audio) > _MAX_UPLOAD:
            raise HTTPException(status_code=413, detail="Recording too large (30 MB max)")
        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(None, voice_io.stt, audio, model)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Couldn't transcribe: {e}")
        return {"text": text}
