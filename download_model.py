"""
Artifex Assistant V5 — Universal model downloader.
Downloads any model from HuggingFace or pulls Ollama models.
Auto-detects model type and shows VRAM requirements.

Usage:
    python download_model.py                              # Default: Qwen3.5-9B
    python download_model.py --repo stabilityai/stable-diffusion-xl-base-1.0
    python download_model.py --repo openai/shap-e
    python download_model.py --ollama qwen3.5:9b          # Pull Ollama model
    python download_model.py --list                       # List installed models
"""

import os
import shutil
import argparse
import sys

DEFAULT_REPO = "Qwen/Qwen3.5-9B"
DEFAULT_OUTPUT = "./models/qwen3.5-9b"


def download_hf_model(repo, output):
    """Download a model from HuggingFace."""
    from huggingface_hub import snapshot_download

    # Preserve patched config.json if it already exists
    config_path = os.path.join(output, "config.json")
    has_patched_config = os.path.exists(config_path)
    if has_patched_config:
        backup_path = config_path + ".dl_backup"
        shutil.copy2(config_path, backup_path)
        print(f"Backed up patched config.json")

    print(f"Downloading {repo} -> {output}")
    print("This may take a while for large models...\n")
    snapshot_download(repo_id=repo, local_dir=output)

    # Restore patched config.json
    if has_patched_config:
        shutil.copy2(backup_path, config_path)
        os.remove(backup_path)
        print(f"Restored patched config.json")

    # Auto-detect model type
    try:
        from core.model_registry import detect_model_type, _estimate_model_size, estimate_vram, ModelInfo
        pipeline_type = detect_model_type(output)
        size_gb = _estimate_model_size(output)

        print(f"\nModel: {repo}")
        print(f"Type:  {pipeline_type}")
        print(f"Size:  {size_gb:.1f} GB on disk")

        # VRAM estimate
        try:
            import torch
            if torch.cuda.is_available():
                gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                info = ModelInfo(name=os.path.basename(output), path=output,
                                pipeline_type=pipeline_type, size_gb=size_gb)
                est = estimate_vram(info, gpu_gb)
                print(f"VRAM:  ~{est['estimated_vram_gb']} GB needed ({est['recommended_quantization']})")
                if est['fits']:
                    print(f"GPU:   {gpu_gb:.0f} GB available — model will fit")
                else:
                    print(f"GPU:   {gpu_gb:.0f} GB available — WARNING: may not fit!")
                print(f"Note:  {est['notes']}")
        except Exception:
            pass  # VRAM estimate is informational ��� don't block the download

    except ImportError:
        pass  # torch not installed — skip GPU info

    print(f"\nDone. Model saved to: {output}")


def pull_ollama_model(model_name):
    """Pull a model using Ollama."""
    import subprocess
    ollama = shutil.which("ollama")
    if not ollama:
        print("Ollama is not installed.")
        print("Install: winget install Ollama.Ollama")
        print("Or visit: https://ollama.com")
        sys.exit(1)

    print(f"Pulling {model_name} via Ollama...")
    subprocess.run([ollama, "pull", model_name], check=True)
    print(f"\n{model_name} is ready for use with the Ollama backend.")


def list_models():
    """List all installed models (local + Ollama)."""
    print("=" * 60)
    print("  Artifex Assistant V5 — Installed Models")
    print("=" * 60)

    # Local models
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    if os.path.isdir(models_dir):
        try:
            from core.model_registry import scan_local_models
            local = scan_local_models(models_dir)
            if local:
                print(f"\n  LOCAL MODELS ({models_dir}):")
                print(f"  {'NAME':<30} {'TYPE':<20} {'SIZE'}")
                print(f"  {'-'*30} {'-'*20} {'-'*10}")
                for m in local:
                    print(f"  {m.name:<30} {m.pipeline_type:<20} {m.size_gb:.1f} GB")
            else:
                print(f"\n  No local models found in {models_dir}")
        except ImportError:
            entries = [d for d in os.listdir(models_dir)
                       if os.path.isdir(os.path.join(models_dir, d))
                       and not d.endswith("-nf4-cached") and not d.startswith(".")]
            if entries:
                print(f"\n  LOCAL MODELS ({models_dir}):")
                for name in entries:
                    print(f"    {name}")
            else:
                print(f"\n  No local models found in {models_dir}")
    else:
        print(f"\n  No models/ directory found")

    # Ollama models
    print()
    try:
        from core.model_registry import scan_ollama_models
        ollama = scan_ollama_models()
        if ollama:
            print(f"  OLLAMA MODELS:")
            print(f"  {'NAME':<30} {'SIZE'}")
            print(f"  {'-'*30} {'-'*10}")
            for m in ollama:
                print(f"  {m.name:<30} {m.size_gb:.1f} GB")
        else:
            print("  No Ollama models found (is Ollama running?)")
    except Exception:
        print("  Ollama not available")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Artifex Assistant V5 — Universal model downloader"
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"HuggingFace repo ID (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Local output directory (auto-detected from repo name if not set)",
    )
    parser.add_argument(
        "--ollama",
        default=None,
        help="Pull an Ollama model instead (e.g. qwen3.5:9b)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all installed models",
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.ollama:
        pull_ollama_model(args.ollama)
        return

    # HuggingFace download
    output = args.output
    if output is None:
        # Auto-generate output path from repo name
        repo_name = args.repo.split("/")[-1].lower()
        output = os.path.join("models", repo_name)

    download_hf_model(args.repo, output)


if __name__ == "__main__":
    main()
