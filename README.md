# Artifex Assistant V5

Universal local AI hosting platform. Run any AI model locally — text generation, image generation, 3D modeling, vision, and audio — with automatic VRAM management and dual backend support.

## Features

- **Dual Backend** — Switch between Transformers and Ollama at runtime
- **Pipeline Architecture** — Text generation, image generation, 3D (ShapE), vision, and audio as equal pipeline modes
- **Auto GPU Detection** — Detects your GPU tier (Tight/Comfortable/Abundant) and scales token limits, context windows, and quantization automatically
- **Model Registry** — Auto-discovers models from HuggingFace, estimates VRAM requirements
- **4-bit Quantization** — NF4 quantization via bitsandbytes, with automatic 8-bit fallback
- **Knowledge System** — Web search, file reading, and workspace-aware context injection
- **CLI + GUI** — Terminal interface and a cyberpunk-themed GUI (FreeSimpleGUI)
- **Setup Wizard** — Interactive setup that detects your GPU and installs the right dependencies

## Requirements

- **OS:** Windows 10/11 (Linux should work but is untested)
- **GPU:** NVIDIA GPU with 8+ GB VRAM (CUDA required)
- **Python:** 3.10+

### Supported GPU Tiers

| Tier | VRAM | Example GPUs |
|------|------|-------------|
| Tight | <= 12 GB | RTX 3060 12GB, RTX 4060 8GB |
| Comfortable | 13-20 GB | RTX 4070 Ti 16GB |
| Abundant | > 20 GB | RTX 3090 24GB, RTX 4090, RTX 5090 |

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/kavenmartinez1-collab/Artifex-Assistantv5.git
cd Artifex-Assistantv5
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux
```

### 2. Run the setup wizard

```bash
python setup_wizard.py
```

This will detect your GPU, install the correct PyTorch + CUDA version, and recommend models.

### 3. Download a model

```bash
# From HuggingFace
python download_model.py --repo Qwen/Qwen3.5-9B

# Or use Ollama
python setup_ollama.py
```

### 4. Launch

```bash
python main.py          # CLI
python main_gui.py      # GUI
```

Or double-click `launch.bat` on Windows.

## Project Structure

```
Artifex-Assistantv5/
├── main.py                 # CLI entry point
├── main_gui.py             # GUI entry point
├── setup_wizard.py         # Setup wizard
├── setup_ollama.py         # Ollama backend setup
├── download_model.py       # Model downloader
├── launch.bat              # Windows launcher
├── core/
│   ├── config.py           # Centralized configuration, GPU tier detection
│   ├── engine_base.py      # Abstract base engine
│   ├── engine_factory.py   # Engine factory (create_engine)
│   ├── engine_transformers.py  # Transformers backend
│   ├── engine_ollama.py    # Ollama backend
│   ├── hardware.py         # Hardware detection utilities
│   ├── inference.py        # Inference orchestration
│   ├── model_loader.py     # Model loading with quantization
│   ├── model_registry.py   # Model auto-discovery and type detection
│   ├── knowledge.py        # Knowledge/RAG system
│   ├── prompts.py          # System prompt templates
│   └── pipelines/
│       ├── base.py         # BasePipeline ABC
│       ├── registry.py     # Pipeline factory
│       ├── text_gen.py     # Text generation pipeline
│       ├── image_gen.py    # Image generation pipeline
│       ├── shape_3d.py     # 3D generation (ShapE)
│       ├── vision.py       # Vision pipeline
│       └── audio.py        # Audio pipeline
├── tools/
│   ├── agent_tools.py      # Tool implementations
│   ├── codebase_tools.py   # Code analysis tools
│   └── tool_cache.py       # Tool output caching
├── ui/
│   ├── cli_assistant.py    # CLI interface
│   ├── cyber_gui.py        # GUI interface
│   ├── gui_theme.py        # GUI theming
│   └── terminal.py         # Terminal utilities
└── requirements*.txt       # GPU-specific dependency files
```

## Backends

### Transformers (default)
Loads models directly from HuggingFace with automatic 4-bit quantization. Best for maximum control and model variety.

### Ollama
Connects to a locally running Ollama server. Pre-quantized models, simpler setup, lower VRAM usage. Switch with `/backend ollama` in the CLI.

## Requirements Files

| File | Target |
|------|--------|
| `requirements.txt` | Base dependencies (any NVIDIA GPU) |
| `requirements-3060.txt` | RTX 3060 12GB (pinned versions) |
| `requirements-rtx4090.txt` | RTX 4090 24GB |
| `requirements-rtx5060ti.txt` | RTX 5060 Ti (Blackwell) |

## License

[MIT](LICENSE)
