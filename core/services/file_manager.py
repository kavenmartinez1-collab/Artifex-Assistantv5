"""
Artifex Assistant V5 — File manager for uploads and generated content.
Provides a unified file storage layer used by the GUI, CLI, and API.
"""

import json
import logging
import mimetypes
import os
import shutil
import tempfile
import time
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

_log = logging.getLogger(__name__)


# Allowed file extensions per category
ALLOWED_EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"},
    "audio": {".wav", ".mp3", ".flac", ".ogg", ".m4a"},
    "video": {".mp4", ".avi", ".webm", ".mov", ".mkv"},
    "document": {".pdf", ".txt", ".md", ".docx", ".csv", ".json"},
    "mesh": {".ply", ".obj", ".stl", ".glb", ".gltf", ".pt"},
}

# Reverse map: extension -> file_type
_EXT_TO_TYPE = {}
for _ftype, _exts in ALLOWED_EXTENSIONS.items():
    for _ext in _exts:
        _EXT_TO_TYPE[_ext] = _ftype


def _detect_file_type(filename: str) -> str:
    """Detect file type from extension. Returns 'unknown' for unrecognized."""
    ext = Path(filename).suffix.lower()
    return _EXT_TO_TYPE.get(ext, "unknown")


def _detect_content_type(filename: str) -> str:
    """Detect MIME type from filename."""
    ct, _ = mimetypes.guess_type(filename)
    return ct or "application/octet-stream"


@dataclass
class FileRecord:
    """Metadata for a stored file."""
    file_id: str
    original_name: str
    stored_path: str
    content_type: str
    file_type: str          # "image", "audio", "video", "document", "mesh"
    size_bytes: int
    created_at: float
    metadata: dict = field(default_factory=dict)
    purpose: str = "upload"  # "upload" or "generated"


class FileManager:
    """Manages file uploads and generated content with persistent index.

    Storage layout:
        {base_dir}/uploads/{file_id}.{ext}
        {base_dir}/generated/{file_id}.{ext}
        {base_dir}/file_index.json
    """

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            from core.config import BASE_DIR
            base_dir = os.path.join(BASE_DIR, "output")

        self._base_dir = base_dir
        self._upload_dir = os.path.join(base_dir, "uploads")
        self._generated_dir = os.path.join(base_dir, "generated")
        self._index_path = os.path.join(base_dir, "file_index.json")
        self._lock = threading.Lock()
        self._index: dict[str, FileRecord] = {}

        os.makedirs(self._upload_dir, exist_ok=True)
        os.makedirs(self._generated_dir, exist_ok=True)

        self._load_index()

    # --- Public API ---

    def store_upload(self, file_bytes: bytes, original_name: str,
                     content_type: str = None) -> FileRecord:
        """Store an uploaded file. Returns its FileRecord."""
        # Sanitize original_name to prevent path traversal in metadata
        original_name = Path(original_name).name
        ext = Path(original_name).suffix.lower()
        file_type = _detect_file_type(original_name)
        if content_type is None:
            content_type = _detect_content_type(original_name)

        file_id = uuid4().hex[:12]
        stored_name = f"{file_id}{ext}"
        stored_path = os.path.join(self._upload_dir, stored_name)

        with self._lock:
            with open(stored_path, "wb") as f:
                f.write(file_bytes)

            record = FileRecord(
                file_id=file_id,
                original_name=original_name,
                stored_path=stored_path,
                content_type=content_type,
                file_type=file_type,
                size_bytes=len(file_bytes),
                created_at=time.time(),
                purpose="upload",
            )

            self._index[file_id] = record
            self._save_index()

        return record

    def store_upload_from_path(self, source_path: str,
                               content_type: str = None) -> FileRecord:
        """Store a file from a local path (copies it into managed storage)."""
        if os.path.islink(source_path):
            raise ValueError(
                f"Refusing to copy symlink: {source_path}. "
                "Provide the real file path instead."
            )
        original_name = os.path.basename(source_path)
        with open(source_path, "rb") as f:
            data = f.read()
        return self.store_upload(data, original_name, content_type)

    def store_generated(self, content: Any, output_type: str,
                        metadata: dict = None, extension: str = None,
                        original_name: str = None) -> FileRecord:
        """Store generated pipeline output.

        Args:
            content: file bytes, file path (str/Path), or PIL Image
            output_type: "image", "audio", "video", "mesh"
            metadata: pipeline metadata dict
            extension: override file extension (e.g. ".wav", ".png")
            original_name: display name for the file
        """
        metadata = metadata or {}
        file_id = uuid4().hex[:12]

        # Resolve content to bytes + extension
        if isinstance(content, (str, Path)) and os.path.isfile(str(content)):
            # Content is a file path — copy it (reject symlinks for safety)
            source_path = str(content)
            if os.path.islink(source_path):
                raise ValueError(
                    f"Refusing to copy symlink: {source_path}. "
                    "Provide the real file path instead."
                )
            if extension is None:
                extension = Path(source_path).suffix.lower() or f".{output_type}"
            stored_name = f"{file_id}{extension}"
            stored_path = os.path.join(self._generated_dir, stored_name)
            shutil.copy2(source_path, stored_path)
            size_bytes = os.path.getsize(stored_path)
        elif isinstance(content, bytes):
            if extension is None:
                extension = f".{output_type}"
            stored_name = f"{file_id}{extension}"
            stored_path = os.path.join(self._generated_dir, stored_name)
            with open(stored_path, "wb") as f:
                f.write(content)
            size_bytes = len(content)
        else:
            # Assume PIL Image
            try:
                import io
                buf = io.BytesIO()
                content.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                extension = extension or ".png"
                stored_name = f"{file_id}{extension}"
                stored_path = os.path.join(self._generated_dir, stored_name)
                with open(stored_path, "wb") as f:
                    f.write(img_bytes)
                size_bytes = len(img_bytes)
            except Exception as e:
                raise TypeError(
                    f"Cannot store content of type {type(content).__name__}: {e}"
                )

        if original_name is None:
            original_name = stored_name

        record = FileRecord(
            file_id=file_id,
            original_name=original_name,
            stored_path=stored_path,
            content_type=_detect_content_type(stored_name),
            file_type=output_type,
            size_bytes=size_bytes,
            created_at=time.time(),
            metadata=metadata,
            purpose="generated",
        )

        with self._lock:
            self._index[file_id] = record
            self._save_index()

        return record

    def get_file(self, file_id: str) -> Optional[FileRecord]:
        """Look up a file record by ID. Returns None if not found."""
        with self._lock:
            return self._index.get(file_id)

    def get_file_path(self, file_id: str) -> Optional[str]:
        """Get the stored filesystem path for a file ID."""
        record = self.get_file(file_id)
        if record and os.path.isfile(record.stored_path):
            return record.stored_path
        return None

    def delete_file(self, file_id: str) -> bool:
        """Delete a file and its record. Returns True if found and removed."""
        with self._lock:
            record = self._index.pop(file_id, None)
            if record is None:
                return False
            if os.path.isfile(record.stored_path):
                os.remove(record.stored_path)
            self._save_index()
        return True

    def list_files(self, file_type: str = None,
                   purpose: str = None) -> list[FileRecord]:
        """List all files, optionally filtered by type or purpose."""
        with self._lock:
            records = list(self._index.values())
        if file_type:
            records = [r for r in records if r.file_type == file_type]
        if purpose:
            records = [r for r in records if r.purpose == purpose]
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def cleanup_old(self, max_age_hours: float = 24):
        """Remove generated files older than max_age_hours."""
        cutoff = time.time() - (max_age_hours * 3600)
        to_delete = []
        with self._lock:
            for fid, record in self._index.items():
                if record.purpose == "generated" and record.created_at < cutoff:
                    to_delete.append(fid)
        for fid in to_delete:
            self.delete_file(fid)
        return len(to_delete)

    def purge(self) -> int:
        """Delete ALL uploaded + generated files and reset the index.

        Returns the number of bytes reclaimed. Configs are untouched (this
        object only owns output/uploads + output/generated + file_index.json).
        """
        reclaimed = 0
        with self._lock:
            for d in (self._upload_dir, self._generated_dir):
                if not os.path.isdir(d):
                    continue
                for name in os.listdir(d):
                    p = os.path.join(d, name)
                    try:
                        if os.path.isfile(p):
                            reclaimed += os.path.getsize(p)
                            os.remove(p)
                    except OSError:
                        pass
            self._index = {}
            if os.path.isfile(self._index_path):
                try:
                    reclaimed += os.path.getsize(self._index_path)
                    os.remove(self._index_path)
                except OSError:
                    pass
        return reclaimed

    # --- Persistence ---

    def _load_index(self):
        """Load file index from disk."""
        if not os.path.isfile(self._index_path):
            return
        try:
            with open(self._index_path, "r") as f:
                data = json.load(f)
            for fid, rec_dict in data.items():
                self._index[fid] = FileRecord(**rec_dict)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            _log.warning(
                "File index corrupted (%s), starting fresh: %s",
                self._index_path, e,
            )

    def _save_index(self):
        """Persist file index to disk atomically. Caller must hold _lock."""
        data = {}
        for fid, record in self._index.items():
            data[fid] = asdict(record)
        # Write to temp file, then atomic rename to prevent corruption
        dir_name = os.path.dirname(self._index_path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            # On Windows, os.replace is atomic within the same volume
            os.replace(tmp_path, self._index_path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
