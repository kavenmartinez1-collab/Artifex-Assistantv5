"""
Artifex Assistant V5 — Media tools for the phone app.

One place that knows, per tool (image, edit, restore, music, video, 3d):
which models can do the job, which one is the default, and the settings
each model family needs. Jobs run in the background (a generation takes
20-120 s; a phone screen may lock meanwhile) and hold the GPU through
ModelQueue.exclusive_gpu, which unloads the chat LLM first.

Jobs live in memory only, like agent runs: the last _KEEP_JOBS are kept,
and outputs themselves are stored by FileManager.

AI-assisted prompts (opt-in per job, image and edit only): the chat model
rewrites a short prompt into a detailed one before the job takes the GPU.
Measured on a "turn him into a South Park character" edit, a short prompt
only cartoonified the face; a detailed one turned the whole person, and
adding a "photograph, photorealistic" negative at true CFG 4 redrew the
background too (101 s vs 70 s, since CFG runs the model twice per step).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
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
    "video": [("ali-vilab/text-to-video-ms-1.7b", "ModelScope 1.7B (download on first use)")],
    "3d":    [("openai/shap-e", "Shap-E (download on first use)")],
}

# The exact commit each hub model above is loaded at (pinned 2026-10-28). The
# pipelines get "repo@commit" and pass revision= to from_pretrained, so the hub
# can only ever serve this snapshot. Bump deliberately, after checking the repo.
_HUB_REVISIONS = {
    "facebook/musicgen-small": "4c8334b02c6ec4e8664a91979669a501ec497792",
    "facebook/musicgen-medium": "d3bd7b00761b78ad7a8a05145ee31e7832e9916c",
    "ali-vilab/text-to-video-ms-1.7b": "8227dddca75a8561bf858d604cc5dae52b954d01",
    "openai/shap-e": "7bd337afdea1c17842e1c3cc45c4e268356dba40",
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


# ── AI-assisted prompts ─────────────────────────────────────────────────

ASSIST_TOOLS = ("image", "edit")
_ASSIST_CFG = 4.0      # true CFG for an edit that comes back with a negative

# Short and open on purpose. A/B on the South Park edit (same seed): this
# wording got the show's real look (big white eyes, round head) and kept
# the scene; a rule list for style changes gave a generic dot-eyed cartoon,
# and thinking on lost the person entirely (and took 3x as long).
_ASSIST_SYSTEM = {
    "image": (
        "The user wants a picture made by Qwen-Image, an AI image generator that "
        "follows long written prompts closely. Interpret the user's request to the "
        "best of your ability and write a specific, detailed prompt of 80-120 words "
        "that will get the generator to produce what they want.\n"
        'Reply with JSON only: {"prompt": "..."}'
    ),
    "edit": (
        "The user wants a photo edited by Qwen-Image-Edit, an AI editor that sees "
        "the photo and follows written instructions. Interpret the user's request "
        "to the best of your ability and write a specific, detailed instruction of "
        "80-120 words that will get the editor to carry out the request properly. "
        "Also write a short negative prompt listing what the result must not look "
        "like (empty if nothing).\n"
        'Reply with JSON only: {"prompt": "...", "negative": "..."}'
    ),
}

# async (messages) -> str: one reply from the chat model (api/server.py)
_prompt_llm = None


def set_prompt_llm(fn):
    global _prompt_llm
    _prompt_llm = fn


def parse_assist_reply(reply: str, fallback: str) -> tuple[str, str]:
    """The model's {"prompt", "negative"} JSON -> (prompt, negative).
    Prose instead of JSON is used as the prompt; nothing usable keeps the
    user's own prompt."""
    text = (reply or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    prompt, negative = "", ""
    if m:
        try:
            data = json.loads(m.group(0))
            prompt = str(data.get("prompt") or "").strip()
            negative = str(data.get("negative") or "").strip()
        except (ValueError, AttributeError):
            pass
    if not prompt and "{" not in text:      # prose, not cut-off JSON
        prompt = re.sub(r"^```\w*|```$", "", text).strip()
    return (prompt or fallback)[:2000], negative[:500]


def assist_available(tool: str) -> bool:
    """The vision helper or the chat model can write this tool's prompt."""
    if tool not in ASSIST_TOOLS:
        return False
    if _prompt_llm is not None:
        return True
    try:
        from core import vision_helper
        return vision_helper.handles_assist(tool)
    except Exception:
        return False


async def assist_prompt(tool: str, prompt: str,
                        image_path: str | None = None) -> tuple[str, str]:
    """Expand `prompt` for `tool` -> (prompt, negative).

    The CPU vision helper goes first when configured: it sees the photo an
    edit starts from, and leaves the GPU alone. The chat model, which only
    ever saw the words, is the fallback.
    """
    from core import vision_helper
    if vision_helper.handles_assist(tool):
        user = prompt
        if image_path:
            from tools.image_tools import encode_image
            user = [{"type": "image_url",
                     "image_url": {"url": encode_image(image_path, max_side=768)}},
                    {"type": "text", "text": prompt}]
        messages = [{"role": "system", "content": _ASSIST_SYSTEM[tool]},
                    {"role": "user", "content": user}]
        try:
            reply = await asyncio.get_event_loop().run_in_executor(
                None, lambda: vision_helper.chat(messages, max_tokens=600))
            return parse_assist_reply(reply, prompt)
        except Exception as e:
            if _prompt_llm is None:
                raise
            _log.warning("Vision helper assist failed (%s); using the chat model", e)
    reply = await _prompt_llm([
        {"role": "system", "content": _ASSIST_SYSTEM[tool]},
        {"role": "user", "content": prompt},
    ])
    return parse_assist_reply(reply, prompt)


def unavailable_reason(tool: str) -> str | None:
    """Why `tool` can't run on this machine at all, or None."""
    # Every Create pipeline runs on PyTorch, which a chat-only install
    # (python -m installer) leaves out. Say so up front instead of listing
    # models that would only fail once the job starts.
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        return ("Needs the Create pack (PyTorch and the image/music models), "
                "which this install doesn't have yet.")
    if tool == "video":
        from core.device import gpu_info
        if not gpu_info.is_available or gpu_info.total_gb < 16:
            return ("Video needs a GPU with 16 GB or more; the video pipeline "
                    "refuses smaller cards.")
    return None


def list_models(tool: str) -> list[dict]:
    """Models able to do `tool`, default first. Each: {id, label, default}."""
    if unavailable_reason(tool):
        return []
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
             "models": list_models(tid), "unavailable": unavailable_reason(tid),
             "sizes": list(SIZES) if tid == "image" else [],
             "assist": assist_available(tid)}
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
        if unavailable_reason(tool):
            raise ValueError(unavailable_reason(tool))
        if requested:
            raise ValueError(f"'{requested}' can't do {tool}. Options: {ids}")
        if not models:
            raise ValueError(f"No model installed that can do {tool}.")
        requested = models[0]["id"]
    if tool in _HUB_MODELS or requested == "":
        rev = _HUB_REVISIONS.get(requested)
        return f"{requested}@{rev}" if rev else requested
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
    assist = bool(params.get("assist")) and assist_available(tool)

    job = {
        "id": uuid.uuid4().hex[:12], "tool": tool, "status": "queued",
        "stage": "Writing a detailed prompt" if assist else "Waiting for the GPU",
        "created": time.time(), "assist": assist,
        "started": None, "finished": None, "error": None, "result": None,
        "model": os.path.basename(model_path) or "classic",
        "prompt": kwargs.get("prompt", ""),
    }
    _remember(job)
    asyncio.get_event_loop().create_task(
        _run(job, spec["pipeline"], model_path, kwargs, assist))
    return job


async def _assist(job: dict, kwargs: dict):
    """Swap in the chat model's detailed prompt. A failure keeps the user's
    prompt and says so on the job; the image still gets made."""
    job.update(status="running", started=time.time())
    try:
        prompt, negative = await assist_prompt(job["tool"], kwargs["prompt"],
                                               kwargs.get("image_path"))
        kwargs["prompt"] = job["assisted_prompt"] = prompt
        if job["tool"] == "edit" and negative:
            kwargs.update(negative_prompt=negative, true_cfg_scale=_ASSIST_CFG)
            job["negative_prompt"] = negative
    except Exception as e:
        _log.warning("Prompt assist for job %s failed: %s", job["id"], e)
        job["assist_error"] = f"AI prompt skipped ({e}); used your prompt."
    job["stage"] = "Waiting for the GPU"


async def _run(job: dict, pipeline_type: str, model_path: str, kwargs: dict,
               assist: bool = False):
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
        if assist:
            await _assist(job, kwargs)
        async with get_model_queue().exclusive_gpu(job["tool"]):
            job.update(status="running", started=job["started"] or time.time(),
                       stage="Loading model")
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
