"""
Artifex Assistant V5 — Media tools for the phone app.

One place that knows, per tool (image, edit, restore, music, video, 3d):
which models can do the job, which one is the default, and the settings
each model family needs. Jobs run in the background (a generation takes
20-120 s; a phone screen may lock meanwhile) and hold the GPU through
ModelQueue.exclusive_gpu, which unloads the chat LLM first.

Jobs live in memory only, like agent runs: the last _KEEP_JOBS are kept,
and outputs themselves are stored by FileManager.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid

from core.logging_config import get_logger

_log = get_logger(__name__)

_KEEP_JOBS = 30

# tool -> pipeline type, label, whether it needs a source image
TOOLS = {
    "image":   {"label": "Create image", "pipeline": "text-to-image"},
    "edit":    {"label": "Edit image", "pipeline": "image-to-image", "needs_image": True},
    "restore": {"label": "Restore photo", "pipeline": "photo-restoration", "needs_image": True},
    "music":   {"label": "Music", "pipeline": "text-to-music"},
    "video":   {"label": "Video", "pipeline": "text-to-video"},
    "3d":      {"label": "3D model", "pipeline": "shap-e"},
}

# Built-in Hugging Face models for tools with no local folder; each
# downloads on first use.
_HUB_MODELS = {
    "music": [("facebook/musicgen-small", "MusicGen small (~2 GB download)"),
              ("facebook/musicgen-medium", "MusicGen medium (~4 GB download)")],
    "video": [("damo-vilab/text-to-video-ms-1.7b", "ModelScope 1.7B (download on first use)")],
    "3d":    [("openai/shap-e", "Shap-E (download on first use)")],
}

# Preferred defaults among local image models, best first
_IMAGE_PREFERENCE = ("qwen-image", "klein")

# Output sizes offered for 1024px-native models (Qwen-Image needs /32)
SIZES = {"square": (1024, 1024), "portrait": (896, 1152), "landscape": (1152, 896)}


def _models_dir() -> str:
    from core.config import BASE_DIR
    return os.path.join(BASE_DIR, "models")


def _family(path: str) -> str:
    name = os.path.basename(path or "").lower()
    if "qwen-image" in name:
        return "qwen-image"
    if "klein" in name:
        return "flux2-klein"
    return "diffusers"


def list_models(tool: str) -> list[dict]:
    """Models able to do `tool`, default first. Each: {id, label, default}."""
    if tool in _HUB_MODELS:
        models = [{"id": mid, "label": label} for mid, label in _HUB_MODELS[tool]]
    else:
        from core.model_registry import scan_local_models
        local = [m for m in scan_local_models(_models_dir())
                 if m.pipeline_type == "text-to-image"]

        def rank(m):
            n = m.name.lower()
            hits = [i for i, key in enumerate(_IMAGE_PREFERENCE) if key in n]
            return (hits[0] if hits else len(_IMAGE_PREFERENCE), n)

        local.sort(key=rank)
        if tool == "restore":
            # The restore stage loads through AutoPipelineForImage2Image,
            # which doesn't map Qwen-Image 2.1
            local = [m for m in local if _family(m.path) != "qwen-image"]
        models = [{"id": m.name, "label": f"{m.name} ({m.size_gb:.1f} GB)"}
                  for m in local]
        if tool == "restore":
            models.insert(0, {"id": "", "label": "Classic clean + upscale (no AI model)"})
    for i, m in enumerate(models):
        m["default"] = i == 0
    return models


def catalog() -> dict:
    """Everything the phone needs to draw the Create screen."""
    from core.services import voice_io
    return {
        "tools": [
            {"id": tid, "label": t["label"], "needs_image": bool(t.get("needs_image")),
             "models": list_models(tid),
             "sizes": list(SIZES) if tid == "image" else []}
            for tid, t in TOOLS.items()
        ],
        "voices": voice_io.list_voices(),
        "stt_models": voice_io.STT_MODELS,
    }


def _resolve_model(tool: str, requested: str | None) -> str:
    """Model id -> local path or hub id. Only ids list_models offers."""
    models = list_models(tool)
    ids = [m["id"] for m in models]
    if requested is None or requested not in ids:
        if requested:
            raise ValueError(f"'{requested}' can't do {tool}. Options: {ids}")
        if not models:
            raise ValueError(f"No model installed that can do {tool}.")
        requested = models[0]["id"]
    if tool in _HUB_MODELS or requested == "":
        return requested
    return os.path.join(_models_dir(), requested)


def _pipeline_kwargs(tool: str, model_path: str, params: dict,
                     image_path: str | None) -> dict:
    """Settings each model family needs, applied server-side."""
    prompt = (params.get("prompt") or "").strip()
    seed = int(params.get("seed", -1))
    family = _family(model_path)
    if tool == "image":
        w, h = SIZES.get(params.get("size") or "square", SIZES["square"])
        kw = {"prompt": prompt, "width": w, "height": h, "seed": seed}
        if family == "flux2-klein":
            kw.update(num_steps=4, guidance_scale=1.0)   # step-distilled
        elif family == "diffusers":
            kw.update(width=w // 2, height=h // 2, num_steps=30)  # SD-era: 512px
        return kw                                        # qwen: its own schedule
    if tool == "edit":
        kw = {"image_path": image_path, "prompt": prompt}
        if family == "flux2-klein":
            kw.update(num_steps=4, guidance_scale=1.0)
        return kw
    if tool == "restore":
        return {"image_path": image_path, "prompt": prompt,
                "upscale": int(params.get("upscale", 2))}
    if tool == "music":
        return {"prompt": prompt,
                "duration_seconds": max(1, min(30, int(params.get("duration", 10))))}
    if tool == "video":
        return {"prompt": prompt}
    if tool == "3d":
        return {"prompt": prompt}
    raise ValueError(f"Unknown tool: {tool}")


# ── Jobs ────────────────────────────────────────────────────────────────

_jobs: dict[str, dict] = {}


def get_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)


def list_jobs() -> list[dict]:
    return sorted(_jobs.values(), key=lambda j: j["created"], reverse=True)


def _remember(job: dict):
    _jobs[job["id"]] = job
    for old in sorted(_jobs.values(), key=lambda j: j["created"])[:-_KEEP_JOBS]:
        if old["status"] in ("done", "error"):
            _jobs.pop(old["id"], None)


def start_job(tool: str, params: dict) -> dict:
    """Validate, register, and schedule a job; returns it at once."""
    if tool not in TOOLS:
        raise ValueError(f"Unknown tool '{tool}'. Options: {list(TOOLS)}")
    spec = TOOLS[tool]
    model_path = _resolve_model(tool, params.get("model"))
    image_path = None
    if spec.get("needs_image"):
        from core.services import get_file_manager
        image_path = get_file_manager().get_file_path(params.get("file_id") or "")
        if not image_path:
            raise ValueError("This tool needs a source image (upload it first).")
    if tool != "restore" and not (params.get("prompt") or "").strip():
        raise ValueError("A prompt is required.")
    kwargs = _pipeline_kwargs(tool, model_path, params, image_path)

    job = {
        "id": uuid.uuid4().hex[:12], "tool": tool, "status": "queued",
        "stage": "Waiting for the GPU", "created": time.time(),
        "started": None, "finished": None, "error": None, "result": None,
        "model": os.path.basename(model_path) or "classic",
        "prompt": kwargs.get("prompt", ""),
    }
    _remember(job)
    asyncio.get_event_loop().create_task(_run(job, spec["pipeline"], model_path, kwargs))
    return job


async def _run(job: dict, pipeline_type: str, model_path: str, kwargs: dict):
    from core.model_queue import ModelBusyError, get_model_queue
    from core.services import get_service

    svc = get_service()

    def progress(_cur, _total, msg):
        if msg:
            job["stage"] = msg

    def work():
        # One media pipeline at a time: two Qwen-Image pipelines (gen +
        # edit) would hold ~23 GB of RAM and fight over the card
        for ptype in list(svc.get_loaded_pipelines()):
            if ptype != pipeline_type:
                svc.unload_pipeline(ptype)
        return svc.run_pipeline(pipeline_type, model_path, kwargs=kwargs,
                                progress_callback=progress, store_output=True)

    try:
        async with get_model_queue().exclusive_gpu(job["tool"]):
            job.update(status="running", started=time.time(), stage="Loading model")
            result = await asyncio.get_event_loop().run_in_executor(None, work)
        if not result.success:
            raise RuntimeError(result.error or "failed")
        file_id = (result.metadata or {}).get("file_id")
        if not file_id:
            raise RuntimeError("The output could not be stored.")
        from core.services import get_file_manager
        rec = get_file_manager().get_file(file_id)
        job["result"] = {"file_id": file_id, "output_type": result.output_type,
                         "content_type": rec.content_type if rec else None,
                         "filename": rec.original_name if rec else None}
        job.update(status="done", stage="Done")
    except ModelBusyError as e:
        job.update(status="error", error=str(e))
    except Exception as e:
        _log.exception("Media job %s (%s) failed", job["id"], job["tool"])
        job.update(status="error", error=str(e))
    finally:
        job["finished"] = time.time()
