"""
Artifex Assistant V5 — Logging configuration.
Provides colored console logging and optional file logging.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_setup_done = False

# Color codes for log levels (using colorama-compatible ANSI)
_LEVEL_COLORS = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[35m",  # magenta
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI color codes based on log level."""

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        color = _LEVEL_COLORS.get(record.levelname, "")
        record.levelname_colored = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


def setup_logging(level="INFO", log_file=None, log_dir=None):
    """Initialize application-wide logging.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Explicit log file path. If None, uses log_dir/artifex.log
        log_dir: Directory for log files. Defaults to <project>/logs/
    """
    global _setup_done
    if _setup_done:
        return
    _setup_done = True

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Console handler with colors
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO)
    console_fmt = _ColorFormatter(
        fmt="[%(asctime)s %(levelname_colored)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    console.setFormatter(console_fmt)
    root.addHandler(console)

    # File handler (optional, rotates at 5 MB, keeps 3 backups)
    if log_file is None:
        if log_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_dir = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "artifex.log")

    try:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            fmt="[%(asctime)s %(levelname)s %(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        root.addHandler(file_handler)
    except (OSError, PermissionError):
        pass  # Can't write log file — continue without it

    # Quiet noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def get_logger(name):
    """Get a named logger. Ensures setup has run."""
    if not _setup_done:
        setup_logging()
    return logging.getLogger(name)
