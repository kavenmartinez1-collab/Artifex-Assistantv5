# Artifex Web Gateway — Lightweight content proxy with sanitization
# Runs in an isolated container with internet access.
# Downloads are quarantined in tmpfs (RAM-backed, never touches disk).

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY main.py sanitizer.py config.py ./

# Create quarantine directory (will be overlaid with tmpfs in docker-compose)
RUN mkdir -p /quarantine

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8080/health', timeout=3); assert r.status_code == 200"

CMD ["python", "main.py"]
