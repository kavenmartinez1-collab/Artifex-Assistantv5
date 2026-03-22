"""
Shared fixtures for Artifex Assistant V5 test suite.
"""

import os
import sys
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock

import pytest


# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory, cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="artifex_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def mock_no_cuda():
    """Patch torch.cuda.is_available to return False."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture
def mock_cuda_12gb():
    """Simulate a 12 GB GPU (TIGHT tier)."""
    props = MagicMock()
    props.total_memory = 12 * 1024 ** 3
    props.name = "Test GPU 12GB"
    props.major = 8
    props.minor = 6
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=props), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.memory_allocated", return_value=2 * 1024 ** 3):
        yield


@pytest.fixture
def mock_cuda_24gb():
    """Simulate a 24 GB GPU (ABUNDANT tier)."""
    props = MagicMock()
    props.total_memory = 24 * 1024 ** 3
    props.name = "Test GPU 24GB"
    props.major = 8
    props.minor = 9
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=props), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.memory_allocated", return_value=4 * 1024 ** 3):
        yield


@pytest.fixture
def sample_model_dir(tmp_dir):
    """Create a fake model directory with a config.json."""
    model_dir = os.path.join(tmp_dir, "test-model")
    os.makedirs(model_dir)
    config = {
        "model_type": "qwen2",
        "hidden_size": 3584,
        "num_hidden_layers": 28,
        "vocab_size": 152064,
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)
    return model_dir


@pytest.fixture
def sample_messages():
    """Provide a sample conversation history."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, what is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
        {"role": "user", "content": "How do I install it?"},
        {"role": "assistant", "content": "You can download it from python.org."},
        {"role": "user", "content": "What about pip?"},
        {"role": "assistant", "content": "pip is Python's package manager."},
        {"role": "user", "content": "How do I use it?"},
        {"role": "assistant", "content": "Run pip install <package>."},
    ]


@pytest.fixture
def knowledge_dir(tmp_dir):
    """Provide a temporary knowledge directory."""
    kd = os.path.join(tmp_dir, "knowledge")
    os.makedirs(os.path.join(kd, "reference"))
    os.makedirs(os.path.join(kd, "workspaces"))
    return kd
