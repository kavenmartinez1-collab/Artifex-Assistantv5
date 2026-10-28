# Artifex Assistant V5 — Docker image
# Multi-stage build with CUDA runtime support

# Pinned by digest (2026-10-28) so a re-pushed tag can't change the build.
# Bump: docker buildx imagetools inspect nvidia/cuda:12.4.0-runtime-ubuntu22.04
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04@sha256:af8bd179ed3bf69d4b63b19a763662a6141f0f62ef099283f68d0b14b4bab0e3 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies (exact pins, fastapi/uvicorn included). The old
# extra `pip3 install fastapi uvicorn || true` pulled whatever was newest,
# unpinned, and hid failures; it is gone.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY core/ ./core/
COPY tools/ ./tools/
COPY ui/ ./ui/
COPY api/ ./api/
COPY knowledge/ ./knowledge/
COPY main.py main_gui.py main_api.py download_model.py setup.py setup_ollama.py ./

# Create required directories
RUN mkdir -p models sessions output logs .tool_cache

# Environment configuration
ENV CUDA_MODULE_LOADING=LAZY
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
# Web gateway URL (set in docker-compose, empty for local dev = fallback to direct search)
ENV WEB_GATEWAY_URL=

# Default: API server mode
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# 0.0.0.0 is the container's own interfaces, which Docker needs to forward
# into; docker-compose.yml publishes it on the host's 127.0.0.1 only.
CMD ["python3", "main_api.py", "--host", "0.0.0.0", "--port", "8000"]
