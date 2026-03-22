"""
Artifex Assistant V5 — Session persistence.
Save and restore conversation state across sessions.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from core.config import SESSION_DIR
from core.logging_config import get_logger

_log = get_logger(__name__)


@dataclass
class SessionState:
    """Serializable snapshot of a conversation session."""
    messages: list = field(default_factory=list)
    session_map: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def _ensure_session_dir(session_dir=None):
    """Create session directory if it doesn't exist."""
    d = session_dir or SESSION_DIR
    os.makedirs(d, exist_ok=True)
    return d


def save_session(name, messages, session_map=None, metadata=None,
                 session_dir=None):
    """Save a conversation session to disk.

    Args:
        name: Session name (used in filename)
        messages: List of {role, content} message dicts
        session_map: Optional SessionMap data (serialized dict)
        metadata: Optional dict with model name, backend, workspace, etc.
        session_dir: Override session directory

    Returns:
        Path to saved session file
    """
    d = _ensure_session_dir(session_dir)

    # Sanitize name
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in name)[:60].strip()
    safe_name = safe_name.replace(" ", "_") or "session"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{timestamp}.json"
    path = os.path.join(d, filename)

    # Strip system messages from saved history (they contain runtime state)
    saved_messages = [
        m for m in messages
        if m.get("role") != "system"
    ]

    state = {
        "name": name,
        "timestamp": timestamp,
        "message_count": len(saved_messages),
        "messages": saved_messages,
        "session_map": session_map or {},
        "metadata": metadata or {},
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    _log.info("Session saved: %s (%d messages)", path, len(saved_messages))
    return path


def auto_save(messages, session_map=None, metadata=None, session_dir=None):
    """Auto-save session to a rotating autosave file.

    Overwrites the previous autosave. Keeps the last 3 autosaves.
    """
    d = _ensure_session_dir(session_dir)

    # Rotate: _autosave_3 -> delete, _autosave_2 -> 3, _autosave_1 -> 2, new -> 1
    for i in range(3, 1, -1):
        old = os.path.join(d, f"_autosave_{i - 1}.json")
        new = os.path.join(d, f"_autosave_{i}.json")
        if os.path.isfile(old):
            try:
                os.replace(old, new)
            except OSError:
                pass

    return save_session("_autosave_1", messages, session_map, metadata, d)


def load_session(path):
    """Load a session from a JSON file.

    Args:
        path: Path to session file

    Returns:
        SessionState or None if file not found/corrupt
    """
    if not os.path.isfile(path):
        _log.warning("Session file not found: %s", path)
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        state = SessionState(
            messages=data.get("messages", []),
            session_map=data.get("session_map", {}),
            metadata=data.get("metadata", {}),
        )
        _log.info("Session loaded: %s (%d messages)", path, len(state.messages))
        return state
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        _log.error("Failed to load session %s: %s", path, e)
        return None


def list_sessions(session_dir=None):
    """List all saved sessions, sorted by most recent first.

    Returns:
        List of dicts with: name, path, timestamp, message_count
    """
    d = _ensure_session_dir(session_dir)
    sessions = []

    for fname in os.listdir(d):
        if not fname.endswith(".json") or fname.startswith("_autosave"):
            continue

        path = os.path.join(d, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "name": data.get("name", fname),
                "path": path,
                "filename": fname,
                "timestamp": data.get("timestamp", ""),
                "message_count": data.get("message_count", 0),
                "metadata": data.get("metadata", {}),
            })
        except (json.JSONDecodeError, OSError):
            continue

    sessions.sort(key=lambda s: s["timestamp"], reverse=True)
    return sessions


def find_session(name_or_index, session_dir=None):
    """Find a session by name prefix or index number.

    Args:
        name_or_index: Session name (prefix match) or 1-based index from list

    Returns:
        Path to session file, or None
    """
    sessions = list_sessions(session_dir)
    if not sessions:
        return None

    # Try as index first
    try:
        idx = int(name_or_index) - 1
        if 0 <= idx < len(sessions):
            return sessions[idx]["path"]
    except (ValueError, TypeError):
        pass

    # Try as name prefix
    query = name_or_index.lower()
    for s in sessions:
        if s["name"].lower().startswith(query):
            return s["path"]
        if s["filename"].lower().startswith(query):
            return s["path"]

    return None
