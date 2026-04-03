"""
Tests for core.services.file_manager — file upload, storage, and retrieval.
"""

import json
import os
import threading
import time

import pytest

from core.services.file_manager import (
    FileManager, FileRecord, _detect_file_type, _detect_content_type,
    ALLOWED_EXTENSIONS,
)


# --- Helpers ---

def _make_png_bytes():
    """Minimal valid PNG header (1x1 white pixel)."""
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx"
        b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00"
        b"\x00\x00\x00IEND\xaeB`\x82"
    )


def _make_wav_bytes():
    """Minimal WAV header (0 samples)."""
    return b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"


# --- File type detection ---

class TestFileTypeDetection:
    def test_image_types(self):
        assert _detect_file_type("photo.png") == "image"
        assert _detect_file_type("photo.JPG") == "image"
        assert _detect_file_type("photo.webp") == "image"

    def test_audio_types(self):
        assert _detect_file_type("recording.wav") == "audio"
        assert _detect_file_type("song.mp3") == "audio"
        assert _detect_file_type("track.flac") == "audio"

    def test_video_types(self):
        assert _detect_file_type("clip.mp4") == "video"
        assert _detect_file_type("movie.avi") == "video"

    def test_document_types(self):
        assert _detect_file_type("report.pdf") == "document"
        assert _detect_file_type("notes.txt") == "document"
        assert _detect_file_type("readme.md") == "document"

    def test_mesh_types(self):
        assert _detect_file_type("model.ply") == "mesh"
        assert _detect_file_type("model.obj") == "mesh"

    def test_unknown_type(self):
        assert _detect_file_type("data.xyz") == "unknown"
        assert _detect_file_type("noext") == "unknown"

    def test_content_type_detection(self):
        assert _detect_content_type("file.png") == "image/png"
        assert _detect_content_type("file.wav") in ("audio/wav", "audio/x-wav")
        ct = _detect_content_type("file.xyz123")
        assert ct == "application/octet-stream"


# --- FileManager ---

class TestFileManagerUpload:
    def test_store_and_retrieve(self, tmp_dir):
        fm = FileManager(base_dir=tmp_dir)
        data = _make_png_bytes()
        record = fm.store_upload(data, "test.png")

        assert record.file_id
        assert len(record.file_id) == 12
        assert record.original_name == "test.png"
        assert record.file_type == "image"
        assert record.content_type == "image/png"
        assert record.size_bytes == len(data)
        assert record.purpose == "upload"
        assert os.path.isfile(record.stored_path)

        # Retrieve by ID
        retrieved = fm.get_file(record.file_id)
        assert retrieved is not None
        assert retrieved.file_id == record.file_id

        # Get path
        path = fm.get_file_path(record.file_id)
        assert path == record.stored_path

    def test_store_from_path(self, tmp_dir):
        fm = FileManager(base_dir=tmp_dir)
        source = os.path.join(tmp_dir, "source.wav")
        with open(source, "wb") as f:
            f.write(_make_wav_bytes())

        record = fm.store_upload_from_path(source)
        assert record.file_type == "audio"
        assert record.original_name == "source.wav"
        assert os.path.isfile(record.stored_path)
        # Should be a copy, not the original
        assert record.stored_path != source

    def test_nonexistent_file(self, tmp_dir):
        fm = FileManager(base_dir=tmp_dir)
        assert fm.get_file("nonexistent") is None
        assert fm.get_file_path("nonexistent") is None


class TestFileManagerGenerated:
    def test_store_from_bytes(self, tmp_dir):
        fm = FileManager(base_dir=tmp_dir)
        data = _make_wav_bytes()
        record = fm.store_generated(
            content=data, output_type="audio", extension=".wav",
            metadata={"prompt": "hello"},
        )
        assert record.purpose == "generated"
        assert record.file_type == "audio"
        assert record.metadata["prompt"] == "hello"
        assert os.path.isfile(record.stored_path)

    def test_store_from_file_path(self, tmp_dir):
        fm = FileManager(base_dir=tmp_dir)
        source = os.path.join(tmp_dir, "gen_output.mp4")
        with open(source, "wb") as f:
            f.write(b"fake video content")

        record = fm.store_generated(content=source, output_type="video")
        assert record.purpose == "generated"
        assert record.file_type == "video"
        assert os.path.isfile(record.stored_path)

    def test_store_pil_image(self, tmp_dir):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not installed")

        fm = FileManager(base_dir=tmp_dir)
        img = Image.new("RGB", (64, 64), color="red")
        record = fm.store_generated(content=img, output_type="image")
        assert record.purpose == "generated"
        assert record.stored_path.endswith(".png")
        assert os.path.isfile(record.stored_path)


class TestFileManagerDelete:
    def test_delete_file(self, tmp_dir):
        fm = FileManager(base_dir=tmp_dir)
        record = fm.store_upload(_make_png_bytes(), "del_test.png")
        path = record.stored_path
        assert os.path.isfile(path)

        assert fm.delete_file(record.file_id) is True
        assert not os.path.isfile(path)
        assert fm.get_file(record.file_id) is None

    def test_delete_nonexistent(self, tmp_dir):
        fm = FileManager(base_dir=tmp_dir)
        assert fm.delete_file("nope") is False


class TestFileManagerList:
    def test_list_all(self, tmp_dir):
        fm = FileManager(base_dir=tmp_dir)
        fm.store_upload(_make_png_bytes(), "img.png")
        fm.store_upload(_make_wav_bytes(), "audio.wav")
        fm.store_generated(b"video", output_type="video", extension=".mp4")

        all_files = fm.list_files()
        assert len(all_files) == 3

    def test_list_by_type(self, tmp_dir):
        fm = FileManager(base_dir=tmp_dir)
        fm.store_upload(_make_png_bytes(), "img.png")
        fm.store_upload(_make_wav_bytes(), "audio.wav")

        images = fm.list_files(file_type="image")
        assert len(images) == 1
        assert images[0].file_type == "image"

    def test_list_by_purpose(self, tmp_dir):
        fm = FileManager(base_dir=tmp_dir)
        fm.store_upload(_make_png_bytes(), "up.png")
        fm.store_generated(b"data", output_type="audio", extension=".wav")

        uploads = fm.list_files(purpose="upload")
        assert len(uploads) == 1
        generated = fm.list_files(purpose="generated")
        assert len(generated) == 1


class TestFileManagerCleanup:
    def test_cleanup_old_generated(self, tmp_dir):
        fm = FileManager(base_dir=tmp_dir)
        record = fm.store_generated(b"old", output_type="audio",
                                    extension=".wav")
        # Backdate the record
        fm._index[record.file_id].created_at = time.time() - 90000  # 25 hours
        fm._save_index()

        removed = fm.cleanup_old(max_age_hours=24)
        assert removed == 1
        assert fm.get_file(record.file_id) is None

    def test_cleanup_preserves_uploads(self, tmp_dir):
        fm = FileManager(base_dir=tmp_dir)
        record = fm.store_upload(_make_png_bytes(), "keep.png")
        fm._index[record.file_id].created_at = time.time() - 90000
        fm._save_index()

        removed = fm.cleanup_old(max_age_hours=24)
        assert removed == 0  # uploads not cleaned
        assert fm.get_file(record.file_id) is not None


class TestFileManagerPersistence:
    def test_index_survives_reload(self, tmp_dir):
        fm1 = FileManager(base_dir=tmp_dir)
        record = fm1.store_upload(_make_png_bytes(), "persist.png")
        file_id = record.file_id

        # Create a new FileManager instance (simulates restart)
        fm2 = FileManager(base_dir=tmp_dir)
        retrieved = fm2.get_file(file_id)
        assert retrieved is not None
        assert retrieved.original_name == "persist.png"

    def test_corrupted_index_recovers(self, tmp_dir):
        fm = FileManager(base_dir=tmp_dir)
        fm.store_upload(_make_png_bytes(), "test.png")

        # Corrupt the index
        with open(fm._index_path, "w") as f:
            f.write("not json{{{")

        # Should recover gracefully (empty index)
        fm2 = FileManager(base_dir=tmp_dir)
        assert len(fm2._index) == 0


class TestFileManagerConcurrency:
    def test_concurrent_uploads(self, tmp_dir):
        fm = FileManager(base_dir=tmp_dir)
        results = []
        errors = []

        def upload(i):
            try:
                r = fm.store_upload(
                    _make_png_bytes(), f"concurrent_{i}.png"
                )
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=upload, args=(i,))
                   for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(results) == 10
        # All IDs should be unique
        ids = {r.file_id for r in results}
        assert len(ids) == 10
