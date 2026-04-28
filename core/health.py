"""
Artifex Assistant V5 — Health check and self-diagnostics.
Verifies system readiness: GPU, models, backends, dependencies, disk.
"""

import importlib
import os
import shutil

from core.config import (
    BASE_DIR, GPU_TIER, MODELS, OLLAMA_MODELS,
    get_active_backend, get_active_model_name,
)
from core.logging_config import get_logger

_log = get_logger(__name__)


def _check_cuda():
    """Check CUDA availability and GPU info."""
    try:
        from core.device import gpu_info
        if not gpu_info.is_available:
            return {
                "available": False,
                "reason": "CUDA not available (CPU-only mode)",
            }
        return {
            "available": True,
            "gpu_name": gpu_info.name,
            "gpu_count": gpu_info.device_count,
            "vram_total_gb": round(gpu_info.total_gb, 2),
            "vram_allocated_gb": round(gpu_info.allocated_gb, 2),
            "vram_free_gb": round(gpu_info.free_gb, 2),
            "compute_capability": gpu_info.compute_cap_str,
            "gpu_tier": GPU_TIER,
        }
    except Exception as e:
        _log.debug("GPU health check failed: %s", e)
        return {"available": False, "reason": str(e)}


def _check_models():
    """Check discovered local models."""
    return {
        "count": len(MODELS),
        "models": list(MODELS.keys())[:20],  # cap display
        "active_model": get_active_model_name(),
        "active_backend": get_active_backend(),
    }


def _check_ollama():
    """Check Ollama server reachability."""
    try:
        import urllib.request
        import json
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        model_count = len(data.get("models", []))
        return {"reachable": True, "model_count": model_count}
    except Exception as e:
        _log.debug("Ollama health check failed: %s", e)
        return {"reachable": False, "error": str(e)}


def _check_dependencies():
    """Check key package versions."""
    packages = [
        "torch", "transformers", "bitsandbytes", "accelerate",
        "diffusers", "safetensors", "Pillow", "FreeSimpleGUI",
        "colorama", "psutil", "huggingface_hub", "scipy",
        "sentence_transformers", "faiss",
    ]
    result = {}
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg.replace("-", "_"))
            version = getattr(mod, "__version__", "installed")
            result[pkg] = version
        except ImportError:
            result[pkg] = None  # not installed
    return result


def _check_disk():
    """Check disk space on the model directory drive."""
    model_dir = os.path.join(BASE_DIR, "models")
    try:
        usage = shutil.disk_usage(model_dir if os.path.exists(model_dir) else BASE_DIR)
        return {
            "total_gb": round(usage.total / (1024 ** 3), 1),
            "free_gb": round(usage.free / (1024 ** 3), 1),
            "used_percent": round((usage.used / usage.total) * 100, 1),
        }
    except Exception as e:
        _log.debug("Disk space check failed: %s", e)
        return {"error": str(e)}


def _check_knowledge():
    """Check knowledge base status."""
    from core.config import KNOWLEDGE_DIR, KNOWLEDGE_REFERENCE_DIR
    ref_count = 0
    if os.path.isdir(KNOWLEDGE_REFERENCE_DIR):
        ref_count = sum(
            1 for f in os.listdir(KNOWLEDGE_REFERENCE_DIR)
            if f.endswith(".json") and f != "_index.json"
        )
    workspace_count = 0
    ws_dir = os.path.join(KNOWLEDGE_DIR, "workspaces")
    if os.path.isdir(ws_dir):
        workspace_count = sum(1 for d in os.listdir(ws_dir) if os.path.isdir(os.path.join(ws_dir, d)))
    return {
        "reference_entries": ref_count,
        "workspace_dirs": workspace_count,
    }


def run_health_check():
    """Run all health checks and return a structured report.

    Returns:
        dict with keys: overall, cuda, models, ollama, dependencies, disk, knowledge
        overall is "healthy", "degraded", or "broken"
    """
    _log.info("Running health check...")

    cuda = _check_cuda()
    models = _check_models()
    ollama = _check_ollama()
    deps = _check_dependencies()
    disk = _check_disk()
    knowledge = _check_knowledge()

    # Determine overall health
    issues = []
    if not cuda.get("available"):
        issues.append("no_cuda")
    if models["count"] == 0:
        issues.append("no_models")
    if deps.get("torch") is None:
        issues.append("no_torch")
    if deps.get("transformers") is None:
        issues.append("no_transformers")
    disk_free = disk.get("free_gb", 999)
    if disk_free < 5:
        issues.append("low_disk")

    if "no_torch" in issues or "no_transformers" in issues:
        overall = "broken"
    elif len(issues) >= 2:
        overall = "degraded"
    elif issues:
        overall = "degraded"
    else:
        overall = "healthy"

    report = {
        "overall": overall,
        "issues": issues,
        "cuda": cuda,
        "models": models,
        "ollama": ollama,
        "dependencies": deps,
        "disk": disk,
        "knowledge": knowledge,
    }

    _log.info("Health check complete: %s (%d issues)", overall, len(issues))
    return report


def format_health_report(report):
    """Format a health check report as a human-readable string."""
    lines = []
    overall = report["overall"].upper()
    icon = {"HEALTHY": "+", "DEGRADED": "~", "BROKEN": "!"}
    lines.append(f"  [{icon.get(overall, '?')}] System: {overall}")

    if report["issues"]:
        lines.append(f"      Issues: {', '.join(report['issues'])}")

    # GPU
    cuda = report["cuda"]
    if cuda.get("available"):
        lines.append(
            f"  [+] GPU: {cuda['gpu_name']} "
            f"({cuda['vram_free_gb']:.1f}/{cuda['vram_total_gb']:.1f} GB free) "
            f"[{cuda['gpu_tier']}]"
        )
        if cuda.get("gpu_count", 1) > 1:
            lines.append(f"      GPUs detected: {cuda['gpu_count']}")
    else:
        lines.append(f"  [!] GPU: {cuda.get('reason', 'unavailable')}")

    # Models
    m = report["models"]
    lines.append(f"  [{'+'if m['count'] else '!'}] Models: {m['count']} found, active: {m['active_model']} ({m['active_backend']})")

    # Ollama
    o = report["ollama"]
    if o.get("reachable"):
        lines.append(f"  [+] Ollama: reachable ({o['model_count']} models)")
    else:
        lines.append(f"  [-] Ollama: not reachable")

    # Disk
    d = report["disk"]
    if "error" not in d:
        icon_d = "+" if d["free_gb"] >= 10 else ("~" if d["free_gb"] >= 5 else "!")
        lines.append(f"  [{icon_d}] Disk: {d['free_gb']:.1f} GB free ({d['used_percent']:.0f}% used)")

    # Key dependencies
    deps = report["dependencies"]
    missing = [k for k, v in deps.items() if v is None and k in ("torch", "transformers", "accelerate")]
    optional_missing = [k for k, v in deps.items() if v is None and k not in ("torch", "transformers", "accelerate")]
    if missing:
        lines.append(f"  [!] Missing required: {', '.join(missing)}")
    if optional_missing:
        lines.append(f"  [-] Optional not installed: {', '.join(optional_missing)}")
    torch_v = deps.get("torch", "?")
    tf_v = deps.get("transformers", "?")
    lines.append(f"  [i] torch={torch_v} transformers={tf_v}")

    # Knowledge
    kb = report["knowledge"]
    lines.append(f"  [i] Knowledge: {kb['reference_entries']} reference entries, {kb['workspace_dirs']} workspaces")

    return "\n".join(lines)
