# Artifex Assistant V5 — Docker image
# Multi-stage build with CUDA runtime support

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Optional: API server dependencies
RUN pip3 install --no-cache-dir fastapi uvicorn || true

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

CMD ["python3", "main_api.py", "--host", "0.0.0.0", "--port", "8000"]
