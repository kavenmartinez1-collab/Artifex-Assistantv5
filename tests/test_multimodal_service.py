"""
Tests for core.services.multimodal_service — pipeline caching, VRAM management,
cancellation, and file storage integration.
"""

import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from core.pipelines.base import BasePipeline, PipelineResult
from core.services.file_manager import FileManager
from core.services.multimodal_service import MultimodalService


# --- Mock pipeline ---

class MockPipeline(BasePipeline):
    """Minimal pipeline for testing the service layer."""
    pipeline_type = "mock-pipeline"
    display_name = "Mock Pipeline"
    output_type = "text"

    def __init__(self):
        self._loaded = False
        self._model_path = ""
        self.load_count = 0
        self.unload_count = 0
        self.run_count = 0

    def load(self, model_path="", status_callback=None, **kwargs):
        self._loaded = True
        self._model_path = model_path
        self.load_count += 1
        if status_callback:
            status_callback("Mock loaded")

    def unload(self, status_callback=None):
        self._loaded = False
        self.unload_count += 1

    def is_loaded(self):
        return self._loaded

    def run(self, **kwargs):
        self.run_count += 1
        return PipelineResult(
            success=True, output_type=self.output_type,
            content="mock output", metadata={"test": True},
        )

    def get_vram_estimate(self, model_path=""):
        return 2.0


class MockImagePipeline(MockPipeline):
    pipeline_type = "mock-image"
    display_name = "Mock Image Gen"
    output_type = "image"

    def run(self, **kwargs):
        self.run_count += 1
        # Return a file path as content
        return PipelineResult(
            success=True, output_type="image",
            content=kwargs.get("output_path", "test.png"),
            metadata={"prompt": "test"},
        )


# --- Service tests ---

class TestMultimodalServiceLifecycle:
    def _make_service(self, tmp_dir):
        fm = FileManager(base_dir=tmp_dir)
        return MultimodalService(file_manager=fm)

    @patch("core.services.multimodal_service.create_pipeline")
    def test_load_and_cache(self, mock_create, tmp_dir):
        mock_pipe = MockPipeline()
        mock_create.return_value = mock_pipe
        svc = self._make_service(tmp_dir)

        # First load
        pipe = svc.load_pipeline("mock-pipeline", "/model/a")
        assert pipe is mock_pipe
        assert mock_pipe.load_count == 1

        # Second load — should reuse cache
        pipe2 = svc.load_pipeline("mock-pipeline", "/model/a")
        assert pipe2 is mock_pipe
        assert mock_pipe.load_count == 1  # NOT reloaded

    @patch("core.services.multimodal_service.create_pipeline")
    def test_different_model_reloads(self, mock_create, tmp_dir):
        mock_pipe = MockPipeline()
        mock_create.return_value = mock_pipe
        svc = self._make_service(tmp_dir)

        svc.load_pipeline("mock-pipeline", "/model/a")
        assert mock_pipe.load_count == 1

        # Different model — should unload and reload
        mock_pipe2 = MockPipeline()
        mock_create.return_value = mock_pipe2
        svc.load_pipeline("mock-pipeline", "/model/b")
        assert mock_pipe.unload_count == 1
        assert mock_pipe2.load_count == 1

    @patch("core.services.multimodal_service.create_pipeline")
    def test_unload_pipeline(self, mock_create, tmp_dir):
        mock_pipe = MockPipeline()
        mock_create.return_value = mock_pipe
        svc = self._make_service(tmp_dir)

        svc.load_pipeline("mock-pipeline")
        assert mock_pipe.is_loaded()

        svc.unload_pipeline("mock-pipeline")
        assert not mock_pipe.is_loaded()
        assert mock_pipe.unload_count == 1

    @patch("core.services.multimodal_service.create_pipeline")
    def test_unload_all(self, mock_create, tmp_dir):
        pipes = [MockPipeline(), MockPipeline()]
        call_count = [0]

        def create_side_effect(ptype):
            p = pipes[call_count[0]]
            call_count[0] += 1
            return p

        mock_create.side_effect = create_side_effect
        svc = self._make_service(tmp_dir)

        svc.load_pipeline("type-a")
        svc.load_pipeline("type-b")

        svc.unload_all()
        assert pipes[0].unload_count == 1
        assert pipes[1].unload_count == 1
        assert svc.get_loaded_pipelines() == {}

    @patch("core.services.multimodal_service.create_pipeline")
    def test_get_loaded_pipelines(self, mock_create, tmp_dir):
        mock_pipe = MockPipeline()
        mock_create.return_value = mock_pipe
        svc = self._make_service(tmp_dir)

        assert svc.get_loaded_pipelines() == {}

        svc.load_pipeline("mock-pipeline")
        loaded = svc.get_loaded_pipelines()
        assert "mock-pipeline" in loaded
        assert loaded["mock-pipeline"] == "Mock Pipeline"


class TestMultimodalServiceRun:
    def _make_service(self, tmp_dir):
        fm = FileManager(base_dir=tmp_dir)
        return MultimodalService(file_manager=fm)

    @patch("core.services.multimodal_service.create_pipeline")
    def test_run_pipeline_basic(self, mock_create, tmp_dir):
        mock_pipe = MockPipeline()
        mock_create.return_value = mock_pipe
        svc = self._make_service(tmp_dir)

        result = svc.run_pipeline("mock-pipeline", kwargs={"prompt": "test"},
                                  store_output=False)
        assert result.success is True
        assert result.content == "mock output"
        assert mock_pipe.run_count == 1

    @patch("core.services.multimodal_service.create_pipeline")
    def test_run_pipeline_with_progress(self, mock_create, tmp_dir):
        mock_pipe = MockPipeline()
        mock_create.return_value = mock_pipe
        svc = self._make_service(tmp_dir)

        progress_msgs = []
        def on_progress(cur, tot, msg):
            progress_msgs.append(msg)

        svc.run_pipeline("mock-pipeline", progress_callback=on_progress,
                         store_output=False)
        # Should have received at least loading and running messages
        assert any("loaded" in m.lower() or "loading" in m.lower()
                    for m in progress_msgs)

    @patch("core.services.multimodal_service.create_pipeline")
    def test_cancel_before_start(self, mock_create, tmp_dir):
        mock_pipe = MockPipeline()
        mock_create.return_value = mock_pipe
        svc = self._make_service(tmp_dir)

        cancel = threading.Event()
        cancel.set()  # Already cancelled

        result = svc.run_pipeline("mock-pipeline", cancel_event=cancel)
        assert result.success is False
        assert "cancel" in result.error.lower()
        assert mock_pipe.run_count == 0

    @patch("core.services.multimodal_service.create_pipeline")
    def test_cancel_after_load(self, mock_create, tmp_dir):
        cancel = threading.Event()

        class CancelOnLoadPipeline(MockPipeline):
            def load(self, model_path="", status_callback=None, **kw):
                super().load(model_path, status_callback, **kw)
                cancel.set()  # Cancel during load

        mock_pipe = CancelOnLoadPipeline()
        mock_create.return_value = mock_pipe
        svc = self._make_service(tmp_dir)

        result = svc.run_pipeline("mock-pipeline", cancel_event=cancel)
        assert result.success is False
        assert mock_pipe.load_count == 1
        assert mock_pipe.run_count == 0


class TestMultimodalServiceFileStorage:
    def _make_service(self, tmp_dir):
        fm = FileManager(base_dir=tmp_dir)
        return MultimodalService(file_manager=fm)

    @patch("core.services.multimodal_service.create_pipeline")
    def test_generated_image_stored(self, mock_create, tmp_dir):
        # Create a fake output file
        fake_output = os.path.join(tmp_dir, "output.png")
        with open(fake_output, "wb") as f:
            f.write(b"fake png")

        mock_pipe = MockImagePipeline()
        mock_create.return_value = mock_pipe
        svc = self._make_service(tmp_dir)

        result = svc.run_pipeline(
            "mock-image",
            kwargs={"output_path": fake_output},
            store_output=True,
        )
        assert result.success
        assert "file_id" in result.metadata
        assert "stored_path" in result.metadata

        # Verify file is in FileManager
        record = svc.file_manager.get_file(result.metadata["file_id"])
        assert record is not None
        assert record.purpose == "generated"

    @patch("core.services.multimodal_service.create_pipeline")
    def test_text_output_not_stored(self, mock_create, tmp_dir):
        mock_pipe = MockPipeline()  # output_type = "text"
        mock_create.return_value = mock_pipe
        svc = self._make_service(tmp_dir)

        result = svc.run_pipeline("mock-pipeline", store_output=True)
        assert result.success
        # Text output should NOT be stored as a file
        assert "file_id" not in result.metadata


class TestMultimodalServiceVRAM:
    @patch("core.services.multimodal_service.create_pipeline")
    def test_lru_eviction(self, mock_create, tmp_dir):
        """When VRAM is tight, LRU pipeline should be evicted."""
        pipe_a = MockPipeline()
        pipe_a.pipeline_type = "pipe-a"
        pipe_b = MockPipeline()
        pipe_b.pipeline_type = "pipe-b"
        pipe_c = MockPipeline()
        pipe_c.pipeline_type = "pipe-c"

        call_idx = [0]
        pipes = [pipe_a, pipe_b, pipe_c]

        def create_side_effect(ptype):
            p = pipes[call_idx[0]]
            call_idx[0] += 1
            return p

        mock_create.side_effect = create_side_effect

        fm = FileManager(base_dir=tmp_dir)
        svc = MultimodalService(file_manager=fm)

        # Load first two without VRAM pressure
        svc.load_pipeline("pipe-a", "/a")
        svc.load_pipeline("pipe-b", "/b")

        assert pipe_a.is_loaded()
        assert pipe_b.is_loaded()

        # Simulate VRAM pressure for third load
        fake_gpu = MagicMock()
        fake_gpu.is_available = True
        fake_gpu.total_gb = 12.0
        fake_gpu.free_gb = 1.0  # only 1GB free, pipeline needs 2GB

        with patch("core.device.gpu_info", fake_gpu), \
             patch("torch.cuda.empty_cache"), \
             patch("gc.collect"):

            svc.load_pipeline("pipe-c", "/c")

            # pipe_a (LRU) should have been evicted
            assert pipe_a.unload_count >= 1
