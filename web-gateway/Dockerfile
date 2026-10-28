# Artifex Web Gateway — Lightweight content proxy with sanitization
# Runs in an isolated container with internet access.
# Downloads are quarantined in tmpfs (RAM-backed, never touches disk).

# Base image pinned by digest (python:3.12-slim on 2026-10-28), so a re-pushed
# tag can't change what gets built. Bump: docker buildx imagetools inspect python:3.12-slim
FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies: every package (dependencies included) pinned with its
# PyPI SHA-256, wheels only, so no package's setup code runs during the build.
# requirements.txt lists the direct pins; pins.txt -> requirements.lock.txt via
# python -m installer.make_locks gateway web-gateway/pins.txt web-gateway/requirements.lock.txt
COPY requirements.lock.txt .
RUN pip install --no-cache-dir --require-hashes --only-binary=:all: -r requirements.lock.txt

# Copy application
COPY main.py sanitizer.py config.py ./

# Create quarantine directory (will be overlaid with tmpfs in docker-compose)
RUN mkdir -p /quarantine

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8080/health', timeout=3); assert r.status_code == 200"

CMD ["python", "main.py"]
