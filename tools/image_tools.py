"""
Artifex-Assistant v5 — Image tools for the agent: @view_image and @describe_images.

The agent's own turns are text-only, so these tools send the picture to the
loaded llama-server themselves (OpenAI image_url format) and hand the model
back a text description. That only works when the loaded model was launched
with --mmproj (the `...-vision-...` config entries); the server's /props
`modalities.vision` flag is checked first so a text-only model gets a clear
"switch models" message instead of a silently hallucinated description.

@describe_images walks a whole folder INSIDE the tool, one request per
image, appending "- name — description" lines to a catalog file as it goes.
Doing the loop here rather than in the agent keeps thousands of images out of
a 16k context window, and the catalog doubles as the resume point: names
already in the file are skipped, so a call cut short by the time budget (or a
crash) picks up where it stopped.
"""

import base64
import io
import json
import logging
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_log = logging.getLogger(__name__)

IMAGE_EXTS = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff",
})

# Folders never worth walking for photos.
_SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", "venv", ".venv", ".tool_cache",
})

DEFAULT_CATALOG_NAME = "image_catalog.md"

DEFAULT_DESCRIBE_PROMPT = (
    "Say what this image is in ONE short line (under 25 words): the main "
    "subject, plus any visible text, place or detail that identifies it. "
    "No preamble, no markdown."
)

# Longest side sent to the model. The projector downsamples far below a
# camera's resolution anyway; sending the full 12 MP file only costs upload
# and decode time.
_MAX_SIDE = 1024

_SEP = " — "

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _vision_url() -> str:
    return os.environ.get("ARTIFEX_VISION_URL", "http://127.0.0.1:8081").rstrip("/")


def _time_budget() -> float:
    try:
        return float(os.environ.get("ARTIFEX_DESCRIBE_BUDGET_S", "900"))
    except ValueError:
        return 900.0


def check_vision_server(base_url=None):
    """(ok, message). ok=False means no request should be attempted."""
    base_url = base_url or _vision_url()
    try:
        with urlopen(Request(f"{base_url}/props"), timeout=5) as resp:
            props = json.loads(resp.read().decode("utf-8"))
    except (URLError, HTTPError, OSError, ValueError) as e:
        return False, f"No model server answering at {base_url} ({e})."
    if not (props.get("modalities") or {}).get("vision"):
        model = os.path.basename(str(props.get("model_path") or "the loaded model"))
        return False, (
            f"{model} was launched without vision (no --mmproj), so it cannot "
            "see images. Start the run on a vision model (the "
            "'...-vision-...' entry in the model list) and try again."
        )
    return True, ""


def encode_image(path, max_side=_MAX_SIDE) -> str:
    """Load any Pillow-readable image and return a JPEG data URL.

    Re-encoding normalizes formats llama.cpp's stb_image decoder does not
    read (webp, tiff) and applies the EXIF rotation phones store instead of
    rotating pixels.
    """
    from PIL import Image, ImageOps
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if getattr(img, "is_animated", False):
            img.seek(0)
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class VisionServerError(Exception):
    """The server itself failed (down, OOM, 5xx) — not this one image."""


def ask_about_image(path, prompt, max_tokens=200, base_url=None) -> str:
    """One image + one question -> the model's text answer.

    Raises VisionServerError when the server is the problem, and
    ValueError/OSError when the image is.
    """
    base_url = base_url or _vision_url()
    data_url = encode_image(path)
    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
        # Same switch engine_llama_cpp uses: Qwen3.x templates gate the think
        # block on this variable, and a thinking pass per image would cost
        # more than the description itself.
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_format": "none",
    }
    req = Request(f"{base_url}/v1/chat/completions",
                  data=json.dumps(payload).encode("utf-8"),
                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        if 400 <= e.code < 500:
            raise ValueError(f"server rejected the image (HTTP {e.code}) {detail}".strip())
        raise VisionServerError(f"HTTP {e.code} {detail}".strip())
    except (URLError, OSError) as e:
        raise VisionServerError(str(e))
    try:
        text = data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        raise VisionServerError(f"unexpected response: {str(data)[:200]}")
    # With thinking off, Qwen3.x templates still emit an EMPTY think block,
    # and reasoning_format "none" leaves it in the content.
    return _THINK_RE.sub("", text).strip()


def _one_line(text) -> str:
    return " ".join(str(text).split()).strip() or "(no description)"


def list_images(folder):
    """Every image under folder, recursively, as sorted relative paths."""
    found = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.startswith("."))
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                rel = os.path.relpath(os.path.join(root, name), folder)
                found.append(rel.replace("\\", "/"))
    return sorted(found, key=str.lower)


def catalogued_names(catalog_path):
    """Names already listed in a catalog written by run_describe_images."""
    done = set()
    try:
        with open(catalog_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("- ") and _SEP in line:
                    done.add(line[2:].split(_SEP, 1)[0].strip())
    except FileNotFoundError:
        pass
    return done


def run_view_image(content):
    """content: "path" or "path|question". Returns (success, output)."""
    path, _, question = content.partition("|")
    path = path.strip().strip("'\"")
    question = question.strip() or (
        "Describe this image: what it shows, any visible text, and notable details.")
    if not os.path.isfile(path):
        return False, f"Image not found: {path}"
    ok, msg = check_vision_server()
    if not ok:
        return False, msg
    try:
        answer = ask_about_image(path, question, max_tokens=600)
    except VisionServerError as e:
        return False, f"Vision server error: {e}"
    except Exception as e:
        return False, f"Could not read {os.path.basename(path)}: {e}"
    return True, f"[{os.path.basename(path)}]\n{answer}"


def run_describe_images(content, time_budget=None):
    """content: "folder" | "folder|catalog" | "folder|catalog|prompt".

    Appends one "- name — description" line per image to the catalog
    (default <folder>/image_catalog.md) and returns a progress summary.
    Stops at the time budget; calling again with the same arguments resumes.
    """
    parts = content.split("|", 2)
    folder = parts[0].strip().strip("'\"")
    catalog = parts[1].strip().strip("'\"") if len(parts) > 1 else ""
    prompt = parts[2].strip() if len(parts) > 2 else ""
    if not folder or not os.path.isdir(folder):
        return False, f"Folder not found: {folder or '(none given)'}"
    if not catalog:
        catalog = os.path.join(folder, DEFAULT_CATALOG_NAME)
    prompt = prompt or DEFAULT_DESCRIBE_PROMPT
    budget = _time_budget() if time_budget is None else time_budget

    images = list_images(folder)
    if not images:
        return True, f"No images found in {folder} (looked for {', '.join(sorted(IMAGE_EXTS))})."

    done = catalogued_names(catalog)
    todo = [rel for rel in images if rel not in done]
    if not todo:
        return True, (f"All {len(images)} images in {folder} are already in {catalog}. "
                      "Nothing left to do.")

    ok, msg = check_vision_server()
    if not ok:
        return False, msg

    new_file = not os.path.exists(catalog)
    parent = os.path.dirname(os.path.abspath(catalog))
    os.makedirs(parent, exist_ok=True)

    started = time.monotonic()
    described, failed, samples = 0, 0, []
    stopped_by = ""
    with open(catalog, "a", encoding="utf-8") as out:
        if new_file:
            out.write(f"# Images in {os.path.abspath(folder)}\n\n")
        for rel in todo:
            # At least one image per call, so every call makes progress.
            if (described or failed) and time.monotonic() - started > budget:
                stopped_by = "time budget"
                break
            try:
                desc = _one_line(ask_about_image(os.path.join(folder, rel), prompt))
                described += 1
            except VisionServerError as e:
                # Don't write a line: the image is fine, the server isn't, and
                # a retry should pick this one up again.
                stopped_by = f"vision server error: {e}"
                break
            except Exception as e:
                # The image itself is unreadable. Record it so the batch can
                # finish — otherwise "remaining" never reaches 0.
                desc = f"[could not read: {_one_line(e)[:120]}]"
                failed += 1
            out.write(f"- {rel}{_SEP}{desc}\n")
            out.flush()
            if len(samples) < 5:
                samples.append(f"- {rel}{_SEP}{desc}")

    remaining = len(todo) - described - failed
    elapsed = time.monotonic() - started
    lines = [
        f"Catalog: {catalog}",
        f"Described {described} image(s) this call"
        + (f", {failed} unreadable" if failed else "")
        + f", in {elapsed:.0f}s. {len(done)} were already listed; "
        f"{len(images)} images total.",
    ]
    if remaining:
        reason = f" (stopped: {stopped_by})" if stopped_by else ""
        lines.append(f"{remaining} remaining{reason}. Call @describe_images again "
                     "with the same arguments to continue.")
    else:
        lines.append("0 remaining. The catalog is complete.")
    if samples:
        lines.append("First lines written:")
        lines.extend(samples)
    success = not stopped_by.startswith("vision server error") or described > 0
    return success, "\n".join(lines)
