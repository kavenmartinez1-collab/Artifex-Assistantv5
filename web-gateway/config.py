"""
Web Gateway — Configuration.
URL filtering, rate limits, size limits, and safety thresholds.
"""

import os

# ── Service ──────────────────────────────────────────────────────────────────
HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
PORT = int(os.getenv("GATEWAY_PORT", "8080"))

# SearXNG base URL (internal Docker network)
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")

# ── Authentication (optional) ───────────────────────────────────────────────
# Set GATEWAY_AUTH_TOKEN on both the gateway and clients to require a shared
# secret on every request. Empty string = auth disabled (default).
GATEWAY_AUTH_TOKEN = os.getenv("GATEWAY_AUTH_TOKEN", "")

# ── Rate Limits ──────────────────────────────────────────────────────────────
# Requests per minute per session
RATE_LIMIT_SEARCH = os.getenv("RATE_LIMIT_SEARCH", "20/minute")
RATE_LIMIT_FETCH = os.getenv("RATE_LIMIT_FETCH", "30/minute")
RATE_LIMIT_DOWNLOAD = os.getenv("RATE_LIMIT_DOWNLOAD", "10/minute")

# ── Size Limits ──────────────────────────────────────────────────────────────
MAX_FETCH_BYTES = int(os.getenv("MAX_FETCH_BYTES", str(5 * 1024 * 1024)))       # 5 MB
MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", str(50 * 1024 * 1024))) # 50 MB
MAX_EXTRACTED_TEXT = int(os.getenv("MAX_EXTRACTED_TEXT", str(500_000)))            # 500 KB text
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))

# ── Timeouts ─────────────────────────────────────────────────────────────────
FETCH_TIMEOUT = int(os.getenv("FETCH_TIMEOUT", "15"))       # seconds
DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", "60"))  # seconds
SEARCH_TIMEOUT = int(os.getenv("SEARCH_TIMEOUT", "10"))      # seconds

# ── Quarantine ───────────────────────────────────────────────────────────────
QUARANTINE_DIR = os.getenv("QUARANTINE_DIR", "/quarantine")
# Max total quarantine size in bytes (256 MB default)
QUARANTINE_MAX_TOTAL = int(os.getenv("QUARANTINE_MAX_TOTAL", str(256 * 1024 * 1024)))

# ── URL Filtering ────────────────────────────────────────────────────────────
# Blocked URL schemes
BLOCKED_SCHEMES = {"file", "ftp", "data", "javascript", "vbscript", "blob"}

# Blocked TLDs (commonly associated with malware/phishing)
BLOCKED_TLDS = {
    ".zip", ".mov", ".php",   # abused new TLDs
    ".tk", ".ml", ".ga", ".cf", ".gq",  # Freenom TLDs (high abuse rate)
}

# Blocked hosts (exact match)
BLOCKED_HOSTS = {
    "localhost", "127.0.0.1", "0.0.0.0", "::1",
    "metadata.google.internal",           # cloud metadata endpoint
    "169.254.169.254",                    # AWS/GCP metadata
}

# Blocked host patterns (substring match)
BLOCKED_HOST_PATTERNS = [
    "10.0.",          # private ranges
    "172.16.",
    "172.17.",        # Docker bridge
    "192.168.",
]

# Allowed file extensions for downloads (empty = allow all)
ALLOWED_DOWNLOAD_EXTENSIONS = set()  # configured per-deployment if needed

# Blocked file extensions for downloads
BLOCKED_DOWNLOAD_EXTENSIONS = {
    ".exe", ".msi", ".bat", ".cmd", ".ps1", ".vbs", ".wsf",  # executables
    ".scr", ".pif", ".com", ".hta",                           # Windows danger
    ".dll", ".sys", ".drv",                                   # system files
    ".sh", ".csh", ".ksh",                                    # shell scripts
    ".jar", ".class",                                          # Java executables
}
