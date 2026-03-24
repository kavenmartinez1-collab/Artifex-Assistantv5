"""
Web Gateway — FastAPI service for safe web search, fetch, and download.
Runs in an isolated Docker container with internet access.
The main Artifex container (no internet) communicates through this gateway.
"""

import os
import re
import uuid
import shutil
import logging
from typing import Optional
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import (
    SEARXNG_URL, QUARANTINE_DIR, QUARANTINE_MAX_TOTAL,
    MAX_FETCH_BYTES, MAX_DOWNLOAD_BYTES, MAX_EXTRACTED_TEXT, MAX_SEARCH_RESULTS,
    FETCH_TIMEOUT, DOWNLOAD_TIMEOUT, SEARCH_TIMEOUT,
    RATE_LIMIT_SEARCH, RATE_LIMIT_FETCH, RATE_LIMIT_DOWNLOAD,
    GATEWAY_AUTH_TOKEN,
)
from sanitizer import extract_text, sanitize_url, check_download_extension

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("web-gateway")

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Artifex Web Gateway", version="1.0.0")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded. Please wait before retrying."},
    )


# ── Optional Auth Middleware ────────────────────────────────────────────────
# Only active when GATEWAY_AUTH_TOKEN is set. Health endpoint always open.
if GATEWAY_AUTH_TOKEN:
    import hmac

    @app.middleware("http")
    async def check_auth_token(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        token = request.headers.get("X-Gateway-Token", "")
        if not hmac.compare_digest(token, GATEWAY_AUTH_TOKEN):
            return JSONResponse(status_code=401, content={"error": "Invalid or missing gateway token"})
        return await call_next(request)

    log.info("Gateway auth enabled — requests require X-Gateway-Token header")


# Ensure quarantine directory exists
Path(QUARANTINE_DIR).mkdir(parents=True, exist_ok=True)

# Shared HTTP client
_client: Optional[httpx.AsyncClient] = None


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            follow_redirects=True,
            max_redirects=5,
            headers={"User-Agent": "Artifex-WebGateway/1.0 (AI research assistant)"},
        )
    return _client


@app.on_event("shutdown")
async def shutdown():
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()


# ── Request Models ───────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    max_results: int = 8
    categories: Optional[str] = "general"

class FetchRequest(BaseModel):
    url: str
    max_chars: Optional[int] = None  # override MAX_EXTRACTED_TEXT

class DownloadRequest(BaseModel):
    url: str
    filename: Optional[str] = None


# ── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    # Check SearXNG connectivity
    searxng_ok = False
    try:
        client = await get_client()
        resp = await client.get(f"{SEARXNG_URL}/healthz", timeout=3)
        searxng_ok = resp.status_code == 200
    except Exception:
        pass

    # Check quarantine dir
    quarantine_ok = os.path.isdir(QUARANTINE_DIR)
    quarantine_files = len(os.listdir(QUARANTINE_DIR)) if quarantine_ok else 0
    quarantine_bytes = sum(
        f.stat().st_size for f in Path(QUARANTINE_DIR).iterdir() if f.is_file()
    ) if quarantine_ok else 0

    return {
        "status": "healthy" if searxng_ok else "degraded",
        "searxng": "connected" if searxng_ok else "unreachable",
        "quarantine": {
            "available": quarantine_ok,
            "files": quarantine_files,
            "bytes": quarantine_bytes,
        },
    }


# ── Search ───────────────────────────────────────────────────────────────────

@app.post("/search")
@limiter.limit(RATE_LIMIT_SEARCH)
async def search(req: SearchRequest, request: Request):
    """Search via SearXNG and return sanitized results."""
    if not req.query.strip():
        raise HTTPException(400, "Empty search query")

    if len(req.query) > 500:
        raise HTTPException(400, "Search query too long (max 500 chars)")

    max_results = min(req.max_results, MAX_SEARCH_RESULTS)

    try:
        client = await get_client()
        resp = await client.get(
            f"{SEARXNG_URL}/search",
            params={
                "q": req.query,
                "format": "json",
                "categories": req.categories or "general",
                "pageno": 1,
            },
            timeout=SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.TimeoutException:
        raise HTTPException(504, "Search timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"SearXNG returned {e.response.status_code}")
    except Exception as e:
        log.error(f"Search error: {e}")
        raise HTTPException(502, f"Search failed: {e}")

    results = []
    for item in data.get("results", [])[:max_results]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", "")[:500],
            "engine": item.get("engine", ""),
        })

    return {
        "query": req.query,
        "results": results,
        "total": len(results),
    }


# ── Fetch ────────────────────────────────────────────────────────────────────

@app.post("/fetch")
@limiter.limit(RATE_LIMIT_FETCH)
async def fetch(req: FetchRequest, request: Request):
    """Fetch a URL, extract clean text, and return sanitized content."""
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "Empty URL")

    # Ensure scheme
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # URL safety check
    is_safe, reason = sanitize_url(url)
    if not is_safe:
        raise HTTPException(403, f"URL blocked: {reason}")

    # Fetch page
    try:
        client = await get_client()
        resp = await client.get(url, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
    except httpx.TimeoutException:
        raise HTTPException(504, f"Fetch timed out after {FETCH_TIMEOUT}s")
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"HTTP {e.response.status_code}: {url}")
    except Exception as e:
        raise HTTPException(502, f"Fetch failed: {e}")

    # Size check
    if len(resp.content) > MAX_FETCH_BYTES:
        raise HTTPException(413, f"Page too large ({len(resp.content) // 1024} KB, max {MAX_FETCH_BYTES // 1024} KB)")

    content_type = resp.headers.get("content-type", "")
    html = resp.text

    # PDF detection
    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        return {
            "url": url,
            "content_type": "application/pdf",
            "text": "[PDF content — use /download to save and process separately]",
            "title": None,
            "warnings": [],
            "injection_found": False,
        }

    # Extract and sanitize
    result = extract_text(html, url=url)

    # Truncate
    max_chars = req.max_chars or MAX_EXTRACTED_TEXT
    if len(result["text"]) > max_chars:
        result["text"] = result["text"][:max_chars] + "\n\n[...content truncated]"

    return {
        "url": str(resp.url),  # may differ from request URL after redirects
        "content_type": content_type,
        "text": result["text"],
        "title": result["title"],
        "author": result.get("author"),
        "date": result.get("date"),
        "warnings": result["warnings"],
        "injection_found": result["injection_found"],
    }


# ── Download ─────────────────────────────────────────────────────────────────

@app.post("/download")
@limiter.limit(RATE_LIMIT_DOWNLOAD)
async def download(req: DownloadRequest, request: Request):
    """Download a file to quarantine (tmpfs). Returns quarantine metadata."""
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "Empty URL")

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # URL safety check
    is_safe, reason = sanitize_url(url)
    if not is_safe:
        raise HTTPException(403, f"URL blocked: {reason}")

    # Determine filename
    if req.filename:
        filename = os.path.basename(req.filename)
    else:
        from urllib.parse import urlparse
        path = urlparse(url).path
        filename = os.path.basename(path) or "downloaded_file"

    # Sanitize filename
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = filename.replace("..", "").strip(". ")
    if not filename:
        filename = "downloaded_file"
    if len(filename) > 200:
        ext = os.path.splitext(filename)[1][:20]
        filename = filename[:200 - len(ext)] + ext

    # Extension check
    ext_safe, ext_reason = check_download_extension(filename)
    if not ext_safe:
        raise HTTPException(403, f"Download blocked: {ext_reason}")

    # Check quarantine capacity
    quarantine_size = sum(
        f.stat().st_size for f in Path(QUARANTINE_DIR).iterdir() if f.is_file()
    )
    if quarantine_size > QUARANTINE_MAX_TOTAL:
        raise HTTPException(507, "Quarantine full. Clear session or wait for cleanup.")

    # Download to quarantine
    quarantine_id = uuid.uuid4().hex[:12]
    quarantine_path = os.path.join(QUARANTINE_DIR, f"{quarantine_id}_{filename}")

    try:
        client = await get_client()
        async with client.stream("GET", url, timeout=DOWNLOAD_TIMEOUT) as resp:
            resp.raise_for_status()
            total = 0
            with open(quarantine_path, "wb") as f:
                async for chunk in resp.aiter_bytes(65536):
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        f.close()
                        os.unlink(quarantine_path)
                        raise HTTPException(413, f"File too large (>{MAX_DOWNLOAD_BYTES // (1024*1024)} MB)")
                    f.write(chunk)

        content_type = resp.headers.get("content-type", "unknown")

    except httpx.TimeoutException:
        if os.path.exists(quarantine_path):
            os.unlink(quarantine_path)
        raise HTTPException(504, f"Download timed out after {DOWNLOAD_TIMEOUT}s")
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(quarantine_path):
            os.unlink(quarantine_path)
        raise HTTPException(502, f"Download failed: {e}")

    file_size = os.path.getsize(quarantine_path)

    log.info(f"Downloaded {filename} ({file_size} bytes) to quarantine as {quarantine_id}")

    return {
        "quarantine_id": quarantine_id,
        "filename": filename,
        "size_bytes": file_size,
        "content_type": content_type,
        "url": url,
    }


# ── Quarantine Access ────────────────────────────────────────────────────────

@app.get("/quarantine/{quarantine_id}")
async def get_quarantined_file(quarantine_id: str):
    """Retrieve a quarantined file by ID. Only accessible from internal network."""
    # Find file with matching ID prefix
    for entry in os.listdir(QUARANTINE_DIR):
        if entry.startswith(quarantine_id + "_"):
            filepath = os.path.join(QUARANTINE_DIR, entry)
            filename = entry[len(quarantine_id) + 1:]  # strip ID prefix
            return FileResponse(filepath, filename=filename)

    raise HTTPException(404, f"Quarantined file not found: {quarantine_id}")


@app.get("/quarantine")
async def list_quarantine():
    """List all quarantined files."""
    files = []
    for entry in sorted(os.listdir(QUARANTINE_DIR)):
        filepath = os.path.join(QUARANTINE_DIR, entry)
        if os.path.isfile(filepath):
            parts = entry.split("_", 1)
            files.append({
                "quarantine_id": parts[0],
                "filename": parts[1] if len(parts) > 1 else entry,
                "size_bytes": os.path.getsize(filepath),
            })
    return {"files": files, "total": len(files)}


# ── Session Cleanup ──────────────────────────────────────────────────────────

@app.delete("/session")
async def cleanup_session():
    """Wipe all quarantined files. Called by Artifex on session end."""
    count = 0
    errors = []
    for entry in os.listdir(QUARANTINE_DIR):
        filepath = os.path.join(QUARANTINE_DIR, entry)
        try:
            if os.path.isfile(filepath):
                os.unlink(filepath)
                count += 1
        except Exception as e:
            errors.append(f"{entry}: {e}")

    log.info(f"Session cleanup: removed {count} quarantined files")

    return {
        "removed": count,
        "errors": errors if errors else None,
    }


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
