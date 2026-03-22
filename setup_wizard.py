"""
Artifex Assistant V5 — Setup Wizard.
Detects GPU, installs correct requirements, and sets up starter models.

Usage:
    python setup_wizard.py               # Full interactive setup
    python setup_wizard.py --detect      # Just detect GPU and recommend config
    python setup_wizard.py --install     # Install requirements for detected GPU
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys


def detect_gpu():
    """Detect GPU and return info dict."""
    info = {
        "name": "Unknown",
        "vram_gb": 0,
        "compute_cap": 0,
        "arch": "unknown",
        "tier": "TIGHT",
        "requirements_file": "requirements.txt",
        "cuda_index": "cu124",
    }

    try:
        import torch
        if not torch.cuda.is_available():
            print("  WARNING: No CUDA GPU detected. PyTorch may need reinstalling.")
            return info

        props = torch.cuda.get_device_properties(0)
        info["name"] = props.name
        info["vram_gb"] = round(props.total_memory / (1024 ** 3), 1)
        info["compute_cap"] = f"{props.major}.{props.minor}"

        # Determine architecture using major.minor compute capability
        # Ada Lovelace = 8.9, Ampere = 8.0-8.6, Hopper = 9.0, Blackwell = 10.x
        cc = (props.major, props.minor)
        if props.major >= 10:
            info["arch"] = "Blackwell"
            info["cuda_index"] = "cu128"
        elif cc >= (8, 9):
            info["arch"] = "Ada Lovelace"
            info["cuda_index"] = "cu124"
        elif props.major >= 8:
            info["arch"] = "Ampere"
            info["cuda_index"] = "cu124"
        elif props.major >= 7:
            info["arch"] = "Turing"
            info["cuda_index"] = "cu121"
        else:
            info["arch"] = "Older"
            info["cuda_index"] = "cu121"

        # Determine tier
        if info["vram_gb"] <= 12:
            info["tier"] = "TIGHT"
        elif info["vram_gb"] <= 20:
            info["tier"] = "COMFORTABLE"
        else:
            info["tier"] = "ABUNDANT"

        # Recommend requirements file based on GPU name
        # Falls back to base requirements.txt for unrecognized GPUs
        name_lower = info["name"].lower()
        if any(x in name_lower for x in ("5060", "5070", "5080", "5090")):
            info["requirements_file"] = "requirements-rtx5060ti.txt"
        elif any(x in name_lower for x in ("4090", "4080")):
            info["requirements_file"] = "requirements-rtx4090.txt"
        elif "3060" in name_lower:
            info["requirements_file"] = "requirements-3060.txt"
        else:
            info["requirements_file"] = "requirements.txt"
            if info["name"] != "Unknown":
                print(f"  NOTE: No GPU-specific requirements for '{info['name']}'.")
                print(f"  Using base requirements.txt (works on any NVIDIA GPU).")

    except ImportError:
        print("  PyTorch not installed yet. Install it first:")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu124")

    return info


def recommend_models(gpu_info):
    """Recommend starter models based on GPU capabilities."""
    vram = gpu_info["vram_gb"]
    models = []

    # Text generation
    if vram >= 24:
        models.append(("Qwen/Qwen3.5-9B", "text-generation", "Best quality 9B chat model"))
    elif vram >= 12:
        models.append(("Qwen/Qwen3.5-9B", "text-generation", "9B model with 4-bit quantization"))
    elif vram >= 8:
        models.append(("Qwen/Qwen3.5-4B", "text-generation", "4B model for tight VRAM"))

    # Ollama alternative
    if shutil.which("ollama"):
        if vram >= 8:
            models.append(("qwen3.5:9b (Ollama)", "text-generation", "Pre-quantized, easy setup"))

    # Image generation
    if vram >= 8:
        models.append(("stabilityai/stable-diffusion-xl-base-1.0", "text-to-image", "SDXL image generation"))
    if vram >= 6:
        models.append(("runwayml/stable-diffusion-v1-5", "text-to-image", "SD 1.5 (smaller, faster)"))

    # 3D
    if vram >= 8:
        models.append(("openai/shap-e", "shap-e", "Text-to-3D mesh generation"))

    return models


def print_gpu_info(gpu_info):
    """Print detected GPU information."""
    print()
    print("=" * 60)
    print("  Artifex Assistant V5 — GPU Detection")
    print("=" * 60)
    print(f"  GPU:          {gpu_info['name']}")
    print(f"  VRAM:         {gpu_info['vram_gb']} GB")
    print(f"  Architecture: {gpu_info['arch']} (compute {gpu_info['compute_cap']}.x)")
    print(f"  Tier:         {gpu_info['tier']}")
    print(f"  CUDA index:   {gpu_info['cuda_index']}")
    print(f"  Requirements: {gpu_info['requirements_file']}")
    print()


def install_requirements(gpu_info):
    """Install the correct requirements for the detected GPU."""
    req_file = gpu_info["requirements_file"]
    cuda_index = gpu_info["cuda_index"]

    if not os.path.isfile(req_file):
        print(f"  Requirements file not found: {req_file}")
        print(f"  Using base requirements.txt")
        req_file = "requirements.txt"

    print(f"  Installing PyTorch with CUDA {cuda_index}...")
    torch_cmd = [
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", f"https://download.pytorch.org/whl/{cuda_index}",
    ]
    subprocess.run(torch_cmd, check=True)

    print(f"\n  Installing requirements from {req_file}...")
    pip_cmd = [sys.executable, "-m", "pip", "install", "-r", req_file]
    subprocess.run(pip_cmd, check=True)

    print("\n  Installation complete.")


def interactive_setup():
    """Full interactive setup wizard."""
    print()
    print("=" * 60)
    print("  Artifex Assistant V5 — Setup Wizard")
    print("  Universal Local AI Hosting Platform")
    print("=" * 60)
    print()
    print("  This wizard will:")
    print("    1. Detect your GPU")
    print("    2. Install the right dependencies")
    print("    3. Recommend models for your hardware")
    print()

    # Step 1: Detect GPU
    input("  Press Enter to start GPU detection...")
    gpu_info = detect_gpu()
    print_gpu_info(gpu_info)

    # Step 2: Install requirements
    print("  Step 2: Install Dependencies")
    print(f"  Recommended: {gpu_info['requirements_file']}")
    choice = input("  Install now? [Y/n]: ").strip().lower()
    if choice in ("", "y", "yes"):
        install_requirements(gpu_info)
    else:
        print("  Skipped. Install manually with:")
        print(f"    pip install torch --index-url https://download.pytorch.org/whl/{gpu_info['cuda_index']}")
        print(f"    pip install -r {gpu_info['requirements_file']}")

    # Step 3: Recommend models
    print()
    print("  Step 3: Recommended Models")
    print("  " + "-" * 50)
    models = recommend_models(gpu_info)
    for i, (name, ptype, desc) in enumerate(models, 1):
        print(f"    {i}. {name}")
        print(f"       Type: {ptype} — {desc}")
    print()
    print("  Download models with:")
    print("    python download_model.py --repo <repo_name>")
    print("    python download_model.py --ollama <model_name>")

    # Step 4: Check Ollama
    print()
    if shutil.which("ollama"):
        print("  Ollama: INSTALLED")
        print("    Use '/backend ollama' in the CLI to switch backends.")
    else:
        print("  Ollama: NOT INSTALLED (optional)")
        print("    Install for easy pre-quantized models:")
        if platform.system() == "Windows":
            print("      winget install Ollama.Ollama")
        else:
            print("      curl -fsSL https://ollama.com/install.sh | sh")

    print()
    print("=" * 60)
    print("  Setup complete! Start Artifex Assistant V5 with:")
    print("    python main.py       (CLI)")
    print("    python main_gui.py   (GUI)")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Artifex Assistant V5 — Setup wizard"
    )
    parser.add_argument("--detect", action="store_true",
                        help="Detect GPU and show recommendations")
    parser.add_argument("--install", action="store_true",
                        help="Install requirements for detected GPU")
    args = parser.parse_args()

    if args.detect:
        gpu_info = detect_gpu()
        print_gpu_info(gpu_info)
        models = recommend_models(gpu_info)
        if models:
            print("  Recommended models:")
            for name, ptype, desc in models:
                print(f"    - {name} ({ptype}): {desc}")
        return

    if args.install:
        gpu_info = detect_gpu()
        print_gpu_info(gpu_info)
        install_requirements(gpu_info)
        return

    interactive_setup()


if __name__ == "__main__":
    main()
