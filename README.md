# Artifex Assistant V5

```
  ▄▀▀▀▀▀▀▀▀▀▄
  █ ·  ▲  · █
  █ ·╱ ◈ ╲· █
  █ ╱─────╲ █
  █ ╲ ╱·╲ ╱ █
  █ ·╲╱ ╲╱· █
  ▀▄▄▄▄▄▄▄▄▄▀
 Artifex-Assistant-v5
```

> *"Unless the LORD builds the house, the builders labor in vain."*
> — Psalm 127:1 (NIV)

All glory to **Jesus Christ**, through whom all things were made and in whom all wisdom dwells.

---

Universal Local AI Hosting Platform. Run any AI model locally — text generation, image generation, 3D, vision, audio, music, video, and more. Supports HuggingFace Transformers and Ollama backends with automatic VRAM management, plus an experimental WebGPU browser-based inference engine.

**Everything runs locally. All servers bind to 127.0.0.1 only. Nothing is exposed to the network.**

### GUI Demo

https://github.com/user-attachments/assets/91074fb1-1a53-48df-a627-071f3af519f0

## Table of Contents

- [Features](#features)
- [Verified Test Results](#verified-test-results-2026-03-22)
- [Supported GPU Tiers](#supported-gpu-tiers)
- [Prerequisites](#prerequisites)
- [Setup Guide](#setup-guide)
- [Using the CLI](#using-the-cli-python-mainpy)
- [Using the GUI](#using-the-gui-python-main_gui_qtpy)
- [Using the API Server](#using-the-api-server-python-main_apipy)
  - [Starting the Server](#starting-the-server)
  - [CLI Flags](#cli-flags)
  - [API Endpoints](#api-endpoints)
  - [Web Tools in the API](#web-tools-in-the-api)
  - [Authentication](#authentication)
- [Setting Up Ollama Backend](#setting-up-ollama-backend)
- [Web Gateway (Secure Web Search)](#web-gateway-secure-web-search)
  - [Architecture](#architecture)
  - [Quick Start](#quick-start)
  - [Safety Features](#safety-features)
  - [Port Configuration](#port-configuration)
- [Docker Deployment](#docker-deployment)
  - [Docker Profiles](#docker-profiles)
  - [Network Isolation](#network-isolation-full-profile)
- [Multi-Modal Pipelines](#multi-modal-pipelines)
- [Agent Tools](#agent-tools)
- [Starting Services](#starting-services)
  - [Control Center (Recommended)](#option-a-control-center-recommended)
  - [Manual Terminal Commands](#option-b-manual-terminal-commands)
- [WebGPU Engine](#webgpu-engine)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Context Profiles](#context-profiles)
  - [Modes](#modes)
- [Security](#security)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Features

- **Multi-modal inference** — 10 pipeline types: text, image, image editing, 3D mesh, vision, audio, speech recognition, music, video, embeddings
- **Two backends** — HuggingFace Transformers (GPU-accelerated) and Ollama (pre-quantized models)
- **Automatic VRAM management** — GPU tier detection, NF4/INT8 quantization, KV cache budgeting
- **Three interfaces** — CLI with agent tools and multimodal pipelines, PyQt6 GUI with inline media, OpenAI-compatible REST API with file management
- **Agent tools** — shell execution, Python runner, web search, codebase analysis (grep, glob, architecture), file I/O, edit-in-place
- **RAG knowledge base** — per-workspace knowledge entries with lifecycle classification, action keys, loop detection
- **Session persistence** — save/load conversations with full metadata (model, backend, mode)
- **WebGPU engine** — browser-based LLM inference with custom WGSL compute kernels (matmul, attention, RMSNorm, RoPE, softmax, SiLU, embedding), TurboQuant KV cache compression (3-4 bit, ~80% memory savings) with asymmetric attention for near-lossless compressed scores, batch prefill, GPTQ INT4 weight support, full transformer forward pass, and autoregressive generation with streaming. Loads any HuggingFace SafeTensors model directly in the browser with retry logic and parallel chunk prefetch.
- **Secure web search** — SearXNG + Web Gateway with content sanitization, prompt injection detection, tmpfs quarantine, and network isolation
- **Docker support** — GPU-enabled container with health checks, optional Ollama sidecar, and web gateway proxy
- **Multi-GPU** — device selection for multi-GPU systems
- **Security** — all servers localhost-only, dangerous command blocking, API key auth, web content sandboxing

## Verified Test Results (2026-03-22)

| Component | Status | Details |
|-----------|--------|---------|
| Unit tests (pytest) | **110/110 PASS** | config, health, inference, knowledge, model registry, pipelines, resilience, sessions, agent tools, tool cache |
| Hardware detection | PASS | Auto-detects GPU name, VRAM, compute capability, architecture, tier |
| Engine factory | PASS | Transformers + Ollama engines create and load correctly |
| Ollama live chat | PASS | Qwen3.5:9b responds locally (thinking + content) |
| Knowledge base CRUD | PASS | Add, search, find-by-key, remove, persistence, ContextEngine lifecycle |
| Session save/load | PASS | Round-trip with metadata, list, find-by-name/index |
| Pipeline registry | PASS | 10 pipelines implement full contract with capabilities |
| Agent tools | PASS | 7 extraction types + live grep, glob, read_file, python, shell |
| API /health | PASS | GPU, VRAM, Ollama, dependencies, disk diagnostics |
| API /v1/chat/completions | PASS | Routes through Ollama backend on localhost |
| WebGPU TypeScript | PASS | tsc --noEmit clean, 0 errors |
| WebGPU Vite build | PASS | 13 modules, 23KB bundle |
| WebGPU inference | PASS | Qwen2.5-0.5B-Instruct generates coherent English at ~20 tok/s in Chrome (f32), ~10 tok/s with TurboQuant 4-bit KV |
| WebGPU kernel tests | **15/15 PASS** | SiLU, Add, Mul, Matmul (naive, tiled, BT, BT-BF16 x3), Softmax, RMSNorm, TurboQuant 3-bit, TurboQuant 4-bit, Lloyd-Max codebook MSE, Asymmetric score |
| WebGPU batch prefill | PASS | 29-token prompt in 1 chunk at ~150 tok/s (vs one-by-one before) |
| Localhost binding | PASS | Confirmed NOT accessible on LAN IP |

## Supported GPU Tiers

| Tier | VRAM | Examples | Quantization |
|------|------|----------|-------------|
| TIGHT | <= 12 GB | RTX 3060 12GB, RTX 5060 Ti 8GB | NF4 all layers (dynamic VRAM planner auto-selects) |
| COMFORTABLE | 13-20 GB | RTX 4070 Ti 16GB | NF4 + BF16 SSM/lm_head (quality preserved) |
| ABUNDANT | > 20 GB | RTX 4090 24GB, RTX 5090 32GB | NF4 + BF16 SSM/lm_head/embeddings |

The transformers engine includes a **dynamic VRAM budget planner** that analyzes each model's weight map at load time and automatically selects which layer groups (SSM, lm_head, embeddings) can stay at BF16 vs must be NF4 to fit on your GPU. No manual configuration needed.

---

## Prerequisites

Before setting up Artifex, make sure you have these installed:

1. **Python 3.11+** — [python.org/downloads](https://www.python.org/downloads/)
2. **NVIDIA GPU drivers** — [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx) (Game Ready or Studio)
3. **Git** — [git-scm.com](https://git-scm.com/) (for cloning the repo)
4. **Node.js 18+** — [nodejs.org](https://nodejs.org/) (only needed for the WebGPU engine)
5. **Ollama** *(optional)* — [ollama.com](https://ollama.com/) or `winget install Ollama.Ollama` on Windows

You do **not** need to install CUDA separately — PyTorch bundles the CUDA runtime.

---

## Setup Guide

### Step 1: Clone and Create Virtual Environment

```bash
git clone <repo-url>
cd Artifex-Assistant-V5

# Create a Python virtual environment
python -m venv venv

# Activate it
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate          # Windows (cmd)
venv/Scripts/activate          # Windows (Git Bash)
```

### Step 2: Install Dependencies

**Option A — Automatic (recommended):**

```bash
python setup_wizard.py
```

The setup wizard will:
1. Detect your GPU (name, VRAM, architecture, compute capability)
2. Classify your GPU tier (TIGHT / COMFORTABLE / ABUNDANT)
3. Install PyTorch with the correct CUDA version for your GPU
4. Install the pinned requirements for your specific card
5. Recommend starter models to download

**Option B — Manual:**

```bash
# 1. Install PyTorch with CUDA (pick your GPU generation)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124   # RTX 30xx / 40xx
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx (Blackwell)

# 2. Install project dependencies (pick your GPU or use the base file)
pip install -r requirements.txt              # Any NVIDIA GPU
pip install -r requirements-rtx4090.txt      # RTX 4090 24GB (pinned versions)
pip install -r requirements-3060.txt         # RTX 3060 12GB (pinned versions)
pip install -r requirements-rtx5060ti.txt    # RTX 5060 Ti 8GB (pinned versions)

# 3. Optional extras
pip install fastapi uvicorn    # For the REST API server
pip install pytest             # For running the test suite
```

### Step 3: Download a Model

You need at least one model to use Artifex. There are two paths:

**Path A — HuggingFace models (Transformers backend):**

```bash
# Download the default model (Qwen3.5-9B — best quality for the size)
python download_model.py

# Or pick a specific model
python download_model.py --repo Qwen/Qwen3.5-4B                              # Smaller, for 8GB GPUs
python download_model.py --repo stabilityai/stable-diffusion-xl-base-1.0      # Image generation
python download_model.py --repo openai/shap-e                                 # 3D mesh generation

# See what you have installed
python download_model.py --list
```

The downloader auto-detects the model type and estimates VRAM requirements.

**Path B — Ollama models (pre-quantized, easier setup):**

```bash
# Set up Ollama (installs, starts server, pulls a model)
python setup_ollama.py                  # Default: qwen3.5:9b
python setup_ollama.py qwen3.5:4b      # Smaller model for tight VRAM
python setup_ollama.py --list           # List installed Ollama models
python setup_ollama.py --status         # Check Ollama server status

# Or use the general downloader
python download_model.py --ollama qwen3.5:9b
```

### Step 4: Launch

```bash
python main.py          # CLI — interactive assistant with agent tools
python main_gui.py      # GUI — cyberpunk desktop interface
python main_api.py      # API — OpenAI-compatible REST server on :8000
launch.bat              # Windows — double-click desktop shortcut

# API server with full options
python main_api.py --backend ollama --gateway http://localhost:8080 --port 8000
python main_api.py --backend transformers --model qwen3.5-27b-distilled --gateway http://localhost:8080
```

### Step 5: Verify Everything Works

```bash
# Quick GPU check
python setup_wizard.py --detect

# Full health report (from inside the CLI)
# Type /health after launching python main.py

# Run the test suite
python -m pytest tests/ -v
```

---

## Using the CLI (`python main.py`)

Interactive terminal assistant with streaming responses, thinking blocks, and agent tool execution.

https://github.com/user-attachments/assets/911f3f29-d70f-402c-960c-1c68f7d2de22

### Slash Commands

| Command | Description |
|---------|-------------|
| `/backend transformers` | Switch to HuggingFace Transformers backend |
| `/backend ollama` | Switch to Ollama backend |
| `/backend` | Show current backend and model |
| `/context STANDARD` | Switch to standard context profile (lower VRAM) |
| `/context HIGH` | Switch to high context profile (more tokens, needs 24GB+) |
| `/workspace <path>` | Set working directory for knowledge base |
| `/workspace scan` | Re-scan current workspace |
| `/health` | Full system diagnostics (GPU, VRAM, models, Ollama, disk) |
| `/compile on\|off` | Toggle torch.compile JIT (20-40% faster, slow first gen) |
| `/turboquant on\|off` | Toggle TurboQuant+ KV cache compression (longer context) |
| `/save <name>` | Save current conversation |
| `/load <name\|#index>` | Load a saved conversation |
| `/sessions` | List all saved sessions |
| `/export <path>` | Export conversation to a file |
| `/kb add <text>` | Add a knowledge entry |
| `/kb search <query>` | Search knowledge base |
| `/kb list` | List knowledge entries |
| `/index` | Show/rebuild the knowledge index |
| `/mode <name>` | Switch pipeline mode (chat, image_gen, vision, tts, stt, music, video, 3d) |
| `/attach <path>` | Attach a file for the next pipeline operation |
| `/output <dir>` | Set output directory for generated files |
| `/open <path>` | Open a file with the system default viewer |
| `/refresh` | Compress history and free VRAM |
| `/clear` | Reset conversation (keeps knowledge) |
| `/cleanup` | Deep clean — reset conversation + remove stale workspace data |
| `exit` or `quit` | Exit the assistant |

### How Agent Tools Work

When you ask the model a question, it can respond with tool markers. The assistant automatically extracts and executes them:

1. You ask: *"What Python files are in this project?"*
2. The model responds with: `@glob("**/*.py")`
3. The assistant executes the glob and feeds the results back to the model
4. The model gives you a human-readable answer based on the results

All tool execution requires your confirmation before running.

---

## Using the GUI (`python main_gui_qt.py`)

https://github.com/user-attachments/assets/91074fb1-1a53-48df-a627-071f3af519f0

The PyQt6 GUI provides a production-grade desktop interface with full multimodal support:

- **Model selector** — dropdown of all auto-discovered models
- **Backend toggle** — switch between Transformers and Ollama
- **Parameter controls** — temperature slider, max tokens, torch.compile, TurboQuant KV toggles
- **10 pipeline modes** — Chat, Code, Image Gen, Image Edit, Vision, 3D, Audio TTS, Audio STT, Music Gen, Video Gen
- **Drag-and-drop file input** — drop images, audio, video, documents onto the drop zone
- **Microphone recording** — record audio directly for STT
- **Rich chat view** — inline images, audio players, video players in chat bubbles
- **Token batching** — 50ms batch interval prevents GUI freezing during fast streaming
- **Cancel support** — interrupt generation mid-stream
- **Progress bars** — visual progress for pipeline operations
- **Knowledge base** — workspace-aware context with automatic knowledge extraction
- **Session management** — save/load conversations with model/backend metadata
- **5 themes** — Cyberpunk, Dark Blue, Blood Dragon, Forest, Light (instant hot-swap)
- **Resource monitor** — real-time VRAM/RAM/CPU in status bar
- **Keyboard shortcut** — Ctrl+Enter to execute

> **Note:** The legacy FreeSimpleGUI GUI (`main_gui.py`) is still available but deprecated. The PyQt6 GUI is the recommended interface.

---

## Using the API Server (`python main_api.py`)

OpenAI-compatible REST API. **Binds to 127.0.0.1 by default — localhost only.**

### Starting the Server

```bash
python main_api.py                              # Default: 127.0.0.1:8000, auto-detect backend
python main_api.py --port 8080                  # Custom port
python main_api.py --backend ollama             # Force Ollama backend
python main_api.py --backend transformers       # Force Transformers backend
python main_api.py --model qwen3.5-27b-distilled  # Select specific model (Transformers)
python main_api.py --model qwen3.5:27b          # Select specific model (Ollama)
python main_api.py --gateway http://localhost:8080  # Enable web search tools via gateway
python main_api.py --reload                     # Auto-reload on code changes
```

**Full example with all options:**

```bash
# Ollama backend with web search
python main_api.py --backend ollama --gateway http://localhost:8080 --port 8000

# Transformers backend with web search and specific model
python main_api.py --backend transformers --model qwen3.5-27b-distilled --gateway http://localhost:8080 --port 8000
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address (use `0.0.0.0` for network access) |
| `--port` | `8000` | Server port |
| `--backend` | auto-detect | `ollama` or `transformers` |
| `--model` | auto-detect | Model name (folder name for transformers, Ollama name for ollama) |
| `--gateway` | *(none)* | Web gateway URL for `@search`/`@web_read` tool execution |
| `--reload` | off | Auto-reload on code changes (development) |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System diagnostics (GPU, VRAM, models, backend, web gateway status) |
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | Chat completion (streaming or non-streaming, with optional web tools) |
| POST | `/v1/images/generations` | Image generation (returns file_id for download) |
| POST | `/v1/images/edits` | Image editing — img2img, inpaint, upscale |
| POST | `/v1/vision/analyze` | Image understanding (accepts file_id or base64) |
| POST | `/v1/audio/speech` | Text-to-speech (TTS) |
| POST | `/v1/audio/transcriptions` | Speech-to-text (STT) |
| POST | `/v1/audio/music` | Music generation from text prompt |
| POST | `/v1/video/generations` | Video generation from text prompt |
| POST | `/v1/3d/generations` | 3D mesh generation (ShapE) |
| POST | `/v1/embeddings` | Generate embeddings |
| POST | `/v1/files` | Upload file (image, audio, video, document) |
| GET | `/v1/files` | List uploaded and generated files |
| GET | `/v1/files/{file_id}` | Download a file by ID |
| DELETE | `/v1/files/{file_id}` | Delete a file |
| GET | `/docs` | Interactive Swagger API documentation |

**LAN access for KBot Web Suite:** Use `--host 0.0.0.0` to bind to all interfaces. CORS is configured to accept `192.168.x.x` origins. Set `ARTIFEX_CORS_ORIGINS` env var for additional origins.

### Example Requests

```bash
# Health check (includes backend and web gateway status)
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion (non-streaming)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:27b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 200,
    "temperature": 0.7,
    "stream": false
  }'

# Streaming chat (both Ollama and Transformers backends supported)
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:27b",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'

# Streaming with web search tools enabled
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:27b",
    "messages": [
      {"role": "system", "content": "You have @search and @web_read tools. Use them for current info."},
      {"role": "user", "content": "What are the latest Python release notes?"}
    ],
    "stream": true,
    "web_tools": true
  }'
```

### Web Tools in the API

When `web_tools: true` is set in a streaming request and the `--gateway` is configured, the API:

1. Streams the model's response (including thinking blocks) in real-time
2. Detects `@search("query")` and `@web_read(N)` markers in the response
3. Executes them through the web gateway (SearXNG search, content extraction)
4. Feeds results back to the model for a follow-up response
5. Streams the final answer with real web data

The SSE stream uses an extended OpenAI format with these additional fields:

| SSE Field | Description |
|-----------|-------------|
| `delta.x_thinking` | Model's reasoning/thinking tokens (separate from content) |
| `delta.content` | Standard response content tokens |
| `x_tool_status` | Array of tool execution labels (e.g., `["Searching: query"]`) |
| `x_tool_done` | Signals tool execution complete, follow-up response starting |
| `x_usage` | Token usage stats on the final chunk |

Standard OpenAI clients will ignore the `x_` fields and receive normal content. Clients that understand the extensions can render thinking blocks, tool indicators, and usage stats.

### Authentication

Set the `ARTIFEX_API_KEY` environment variable to require authentication:

```bash
# Set the key
export ARTIFEX_API_KEY=my-secret-key       # Linux/Mac
set ARTIFEX_API_KEY=my-secret-key          # Windows

# Requests must include the key
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer my-secret-key"

# Or via X-API-Key header
curl http://localhost:8000/v1/models \
  -H "X-API-Key: my-secret-key"
```

If `ARTIFEX_API_KEY` is not set, no authentication is required.

---

## Setting Up Ollama Backend

Ollama provides pre-quantized models that are easy to set up and use less VRAM.

### Install Ollama

```bash
# Windows
winget install Ollama.Ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Mac
brew install ollama
```

### Configure for Artifex

```bash
# Setup helper: finds Ollama, starts the server, pulls a model
python setup_ollama.py                  # Default: qwen3.5:9b
python setup_ollama.py qwen3.5:4b      # Smaller model
python setup_ollama.py --status         # Check what's installed
python setup_ollama.py --list           # List all models
```

### Switch to Ollama in the CLI

```
/backend ollama
```

### Pull More Models

```bash
ollama pull qwen3:8b             # Qwen3 8B
ollama pull llama3.3:8b          # Llama 3.3 8B
ollama pull mistral:7b           # Mistral 7B
ollama pull codellama:7b         # Code Llama 7B
```

All communication with Ollama stays on `localhost:11434` — nothing leaves your machine.

---

## Web Gateway (Secure Web Search)

The Web Gateway adds safe, sandboxed web search and content retrieval to Artifex. It uses a three-container Docker architecture that keeps your local AI isolated from the internet while still giving it access to web content through a sanitized proxy.

### Architecture

```
┌──────────────────────────────────────────────────────────┐
│  docker compose up (default — lightweight, ~500 MB)      │
│                                                          │
│  ┌──────────────────┐         ┌──────────────────────┐  │
│  │     searxng       │◄───────│    web-gateway        │  │
│  │  (search engine)  │        │  (content proxy)      │  │
│  │  Google, Bing,    │        │  trafilatura extract   │  │
│  │  DuckDuckGo, etc. │        │  injection detection  │  │
│  └──────────────────┘         │  tmpfs quarantine     │  │
│          │                    │  rate limiting         │  │
│          ▼                    └──────────┬────────────┘  │
│       Internet                     port 8080             │
│                                          │               │
└──────────────────────────────────────────┼───────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │  Artifex (local venv)    │
                              │  Your GPU, your models   │
                              │  WEB_GATEWAY_URL=        │
                              │   http://localhost:8080   │
                              └─────────────────────────┘
```

### Quick Start

```bash
# 1. Start the web gateway + search engine (~500 MB, no GPU needed)
docker compose up

# 2a. Run the CLI with gateway
$env:WEB_GATEWAY_URL = "http://localhost:8080"    # PowerShell
export WEB_GATEWAY_URL=http://localhost:8080       # Linux/Mac/bash
python main.py

# 2b. Or run the API server with --gateway flag (no env var needed)
python main_api.py --backend ollama --gateway http://localhost:8080
```

Now when the model uses `@search("query")` or `@web_read(url)`, requests are routed through the gateway. If Docker isn't running, the CLI falls back to direct DuckDuckGo search automatically. The API server requires the gateway for web tools (`web_tools: true` in requests).

### Important: Docker vs Local Gateway (Port 8080 Conflict)

There are **two** web gateways — a Docker container and a local Python script. **Do not run both.** They compete for port 8080 and the local one can't reach Docker's SearXNG.

| Mode | How to start | SearXNG access | Notes |
|------|-------------|----------------|-------|
| **Docker (recommended)** | `docker compose up` | Works (Docker internal network) | Gateway + SearXNG both in containers |
| **Local (no Docker)** | Control Center → start "Web Gateway" | Needs `SEARXNG_URL` configured | Only for setups without Docker |

**Order of operations:**

```
Docker mode:
  1. docker compose up          ← starts SearXNG + web gateway on port 8080
  2. Start API server           ← with --gateway http://localhost:8080
  3. Do NOT start "Web Gateway (local)" from the Control Center

Local mode (no Docker):
  1. Start SearXNG separately   ← or skip (no web search)
  2. Start "Web Gateway (local)" from Control Center
  3. Start API server           ← with --gateway http://localhost:8080
```

**Diagnosis if search tools fail:**
```bash
curl http://localhost:8080/health
# If "searxng": "unreachable" → the local gateway is running instead of Docker's
# Fix: stop the local gateway, ensure docker compose is running
```

### Safety Features

| Feature | Description |
|---------|-------------|
| **Content sanitization** | trafilatura extracts article text, strips scripts, iframes, ads, and tracking |
| **Prompt injection detection** | 20+ patterns detected (instruction override, role manipulation, delimiter attacks, tool injection, data exfiltration). Suspicious content wrapped in `[UNTRUSTED]` delimiters |
| **URL filtering** | Blocked schemes (file://, data://), blocked TLDs (.tk, .ml, etc.), blocked private/metadata IPs (169.254.169.254, 10.x, 172.x, 192.168.x) |
| **tmpfs quarantine** | Downloads stored in RAM-backed /quarantine/ — never touches disk, auto-deleted on container stop |
| **Rate limiting** | 20 searches/min, 30 fetches/min, 10 downloads/min per session |
| **Size limits** | 5 MB per page fetch, 50 MB per download, 500 KB extracted text |
| **Blocked extensions** | .exe, .bat, .ps1, .dll, .sh, .jar, and other executable formats blocked from download |
| **Session cleanup** | `/clear` and `/cleanup` commands wipe all quarantined files |
| **Optional auth token** | Set `GATEWAY_AUTH_TOKEN` to require a shared secret on all gateway requests (except `/health`). Disabled by default — enable if exposing port 8080 to the network. |
| **Auto-fallback** | If the gateway is unreachable, search/fetch falls back to direct DuckDuckGo (existing behavior) |

### Web Gateway API Endpoints

The gateway runs on port 8080 inside Docker (exposed to localhost).

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/search` | Search via SearXNG, return sanitized results |
| POST | `/fetch` | Fetch URL, extract clean text via trafilatura |
| POST | `/download` | Download file to tmpfs quarantine |
| GET | `/quarantine/{id}` | Retrieve a quarantined file |
| GET | `/quarantine` | List all quarantined files |
| DELETE | `/session` | Wipe all quarantined files (session cleanup) |
| GET | `/health` | Health check (SearXNG connectivity, quarantine status) |

### SearXNG Search Engines

The self-hosted SearXNG instance aggregates results from multiple engines:

- Google, Bing, DuckDuckGo (general web)
- Wikipedia, Wikidata (knowledge)
- GitHub (code and repositories)
- arXiv (research papers)

### Port Configuration

All ports are configurable. Default values and where to change them:

| Service | Default Port | Config Location | Environment Variable |
|---------|-------------|-----------------|---------------------|
| Artifex API | `8000` | `main_api.py --port` | — |
| Web Gateway | `8080` | `web-gateway/config.py` (`PORT`) | `GATEWAY_PORT` |
| SearXNG | `8080` (internal) | `web-gateway/searxng/settings.yml` (`server.port`) | — |
| Ollama | `11434` | Ollama default | `OLLAMA_HOST` |

**Connecting the API server to the web gateway:**

```bash
# Via CLI flag (recommended)
python main_api.py --gateway http://localhost:8080

# Via environment variable (CLI/GUI only)
$env:WEB_GATEWAY_URL = "http://localhost:8080"    # PowerShell
export WEB_GATEWAY_URL=http://localhost:8080       # Linux/Mac/bash
```

**Changing the web gateway port:**

1. Update `docker-compose.yml` — change the `ports` mapping under `web-gateway` (e.g., `"9090:8080"` to expose on host port 9090)
2. Set `WEB_GATEWAY_URL` to match (e.g., `http://localhost:9090`)

**Changing the Artifex API port:**

```bash
python main_api.py --port 9000
```

**Integrating with external applications:**

The Artifex API server is designed as a standalone backend that any frontend can connect to. To integrate:

1. Start the API: `python main_api.py --backend ollama --gateway http://localhost:8080`
2. POST to `/v1/chat/completions` with `stream: true` and `web_tools: true`
3. Parse the SSE stream — standard OpenAI format with optional `x_thinking`, `x_tool_status`, `x_tool_done`, and `x_usage` extensions
4. The API handles all backend selection, model management, and tool execution internally

Your application only needs to:
- Send messages to the Artifex API
- Parse the SSE response stream
- Render thinking blocks, tool indicators, and content tokens as desired

The API server includes a CORS allowlist in `api/server.py`. Add your application's origin to the `allow_origins` list if you need cross-origin browser access (not needed for server-to-server proxying).

### Stopping the Gateway

```bash
docker compose down          # Stop containers
docker compose down -v       # Stop and remove volumes
docker system prune          # Clean up unused images
```

---

## Docker Deployment

### Default: Web Gateway Only (Recommended)

```bash
# Start just the web gateway + SearXNG (~500 MB total)
docker compose up

# Option A: Run the CLI with env var
$env:WEB_GATEWAY_URL = "http://localhost:8080"    # PowerShell
export WEB_GATEWAY_URL=http://localhost:8080       # Linux/Mac/bash
python main.py

# Option B: Run the API server with --gateway flag (recommended for external apps)
python main_api.py --backend ollama --gateway http://localhost:8080
```

### Full Containerized Deployment

```bash
# Everything in Docker (requires NVIDIA Container Toolkit + GPU)
docker compose --profile full up

# With Ollama sidecar
docker compose --profile full --profile ollama up

# Custom environment
ARTIFEX_API_KEY=your-key CUDA_VISIBLE_DEVICES=0 docker compose --profile full up
```

### Docker Profiles

| Profile | Containers | Disk Usage | Use Case |
|---------|-----------|------------|----------|
| *(default)* | web-gateway + searxng | ~500 MB | Local dev with safe web search |
| `full` | + artifex (GPU) | ~8+ GB | Full isolated deployment |
| `ollama` | + ollama | ~2+ GB | Ollama backend in Docker |

### Network Isolation (Full Profile)

When using the `full` profile, network isolation is enforced:

- **ai_internal** network (`internal: true`): Artifex and Ollama — **no internet access**
- **ai_external** network: Web gateway and SearXNG — internet access
- The web gateway bridges both networks, proxying sanitized content to Artifex

### What Docker Provides

- **SearXNG** — self-hosted meta-search engine (API-only mode, no tracking)
- **Web Gateway** — content proxy with sanitization, rate limiting, and quarantine
- **CUDA 12.4 runtime** base image for full profile (no manual CUDA install needed)
- **GPU passthrough** via NVIDIA Container Toolkit (full profile)
- **Health checks** on all containers
- **tmpfs quarantine** — downloads in RAM, auto-deleted on stop
- **Persistent volumes** for models, sessions, output, and knowledge (full profile)

### Prerequisites for Docker

1. **Docker Desktop** — [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) (WSL 2 backend on Windows)
2. **NVIDIA Container Toolkit** *(only for full profile)* — [docs.nvidia.com/datacenter/cloud-native/container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

---

## Multi-Modal Pipelines

All 10 pipeline types are auto-discovered and implement a standard contract (load, run, unload, get_capabilities):

| Pipeline | Type Key | Description | Min VRAM |
|----------|----------|-------------|----------|
| Text Generation | `text-generation` | Chat/instruct models (Qwen, Llama, Mistral, etc.) | 6 GB |
| Image Generation | `text-to-image` | Stable Diffusion, SDXL, FLUX | 6 GB |
| Image Editing | `image-to-image` | Inpainting, upscaling, img2img | 6 GB |
| Vision | `image-text-to-text` | Image understanding (LLaVA, Qwen-VL) | 8 GB |
| 3D Generation | `shap-e` | Text-to-3D mesh (OpenAI ShapE) | 8 GB |
| Audio / TTS | `text-to-audio` | Text-to-speech (Bark) | 4 GB |
| Speech Recognition | `automatic-speech-recognition` | Speech-to-text (Whisper) | 4 GB |
| Music | `text-to-music` | Music generation (MusicGen) | 8 GB |
| Video | `text-to-video` | Video generation | 12 GB |
| Embeddings | `feature-extraction` | Embedding models for RAG (runs on CPU) | 2 GB |

### Adding Models

Models are auto-discovered from the `models/` directory. To add a new model:

1. Download it: `python download_model.py --repo <huggingface-repo-id>`
2. It saves to `models/<model-name>/`
3. It appears in the GUI dropdown and CLI automatically on next launch

You can also manually place any HuggingFace model folder in `models/` — the registry detects its type from `config.json`.

### Gemma 4 Support

Google's Gemma 4 models are supported with automatic detection and multimodal capabilities:

| Model | Params | Active | VRAM | Context | Multimodal |
|-------|--------|--------|------|---------|------------|
| gemma-4-E2B-it | 5.1B | 2.3B | ~7 GB | 128K | Text, Image, Video, Audio |
| gemma-4-E4B-it | 8B | 4.5B | ~10 GB | 128K | Text, Image, Video, Audio |
| gemma-4-26B-A4B-it | 25.2B | 3.8B (MoE) | ~18 GB | 256K | Text, Image, Video |
| gemma-4-31B-it | 30.7B | 30.7B | ~20 GB | 256K | Text, Image, Video |

**Setup:**
1. Requires `transformers >= 5.5.0` (for `AutoModelForMultimodalLM`)
2. Download: `huggingface-cli download google/gemma-4-E4B-it --local-dir models/gemma-4-e4b-it`
3. Select the model in the GUI or CLI — multimodal loading is automatic

The engine detects `model_type: gemma3n` in `config.json` and automatically uses `AutoModelForMultimodalLM` + `AutoProcessor`. Existing models (Qwen, Llama, Mistral) are unaffected — they continue using `AutoModelForCausalLM`.

Gemma 4 also works via Ollama: `ollama pull gemma4:e4b`

### Shared Service Layer

All pipelines are accessed through a shared `MultimodalService` layer (`core/services/`) used by the GUI, CLI, and API. This provides:

- **Pipeline caching** — loaded pipelines are reused across calls
- **VRAM management** — auto-evicts least-recently-used pipelines when VRAM is tight
- **File management** — uploads and generated files stored with persistent index (`output/file_index.json`)
- **Cancellation** — operations can be interrupted mid-stream
- **Progress reporting** — uniform callback interface for all pipeline types

---

## Agent Tools

The CLI assistant can execute these tool types, extracted from model responses:

| Tool | Syntax | Description |
|------|--------|-------------|
| Shell | `` ```bash ... ``` `` | Execute shell commands (with safety checks) |
| Python | `` ```python ... ``` `` | Run Python snippets in a sandboxed subprocess |
| Read File | `@read_file("path")` | Read file contents (chunked for large files) |
| Edit File | `` ```edit ... ``` `` | Find-and-replace edits with syntax validation |
| Grep | `@grep("pattern", "path")` | Regex search across files |
| Glob | `@glob("pattern")` | Find files by pattern |
| Architecture | `@architecture()` | Map project structure and symbols |
| Web Search | `@search("query")` | Search the web (SearXNG via gateway, DuckDuckGo fallback) |
| Web Read | `@web_read("url")` | Fetch and parse a webpage (sanitized via gateway with injection detection) |
| Download | `@download("url", "path")` | Download a file (quarantined in tmpfs via gateway) |
| Find Symbol | `@find_symbol("name")` | Locate function/class definitions |
| Read Function | `@read_function("name")` | Read a specific function's source code |

Dangerous commands (rm -rf /, format c:, etc.) are blocked by configurable safety patterns.

---

## WebGPU Engine

Experimental browser-based GPU inference using WebGPU compute shaders. Runs directly on your GPU through Chrome/Edge without Python or CUDA. **All servers bind to 127.0.0.1 only.**

https://github.com/user-attachments/assets/3e60ed18-4351-42c6-a846-effe862f3f84

### Prerequisites

- **Node.js 18+** — [nodejs.org](https://nodejs.org/)
- **Chrome 113+** or **Edge 113+** with WebGPU enabled
  - Navigate to `chrome://flags/#enable-unsafe-webgpu` and enable it
  - Or use Chrome Canary / Edge Canary where it's on by default

### Setup

```bash
cd webgpu
npm install
```

### Run

```bash
npm run dev       # Vite dev server (:5173) + metrics server (:3001)
npm run build     # Production build (13 modules, 23KB bundle)
npm run preview   # Preview production build
npm run server    # Metrics server only
```

### Running (Two Terminals)

```bash
cd webgpu
npx vite                      # Terminal 1: Vite dev server (:5173)
npx tsx server/dev-server.ts  # Terminal 2: metrics + local cache server (:3001)
```

### Testing the Kernels

1. Open `http://localhost:5173` in Chrome/Edge
2. The page auto-detects your GPU (vendor, architecture, limits)
3. Click **"Run GPU Tests"** in the sidebar
4. All 15 WGSL kernels run against CPU reference values
5. Results appear in the browser and log to the metrics server at `:3001`

### Architecture

- **GPU initialization** — WebGPU device/adapter detection, buffer negotiation up to 2GB
- **WGSL compute kernels**:
  - `matmul.wgsl` — tiled 16x16 matrix multiplication with shared memory + `matmul_bt` (B-transposed) + `matmul_bt_bf16` (BF16 native weights)
  - `attention.wgsl` — fused multi-head attention with GQA, causal masking, inline softmax
  - `attention_tq.wgsl` — asymmetric attention with QJL correction for TurboQuant-compressed KV cache (near-lossless scores despite 3-4 bit compression)
  - `rmsnorm.wgsl` — RMS layer normalization
  - `rope.wgsl` — rotary positional embeddings
  - `softmax.wgsl` — attention softmax
  - `elementwise.wgsl` — add, multiply, SiLU activation
  - `embed.wgsl` — parallel embedding table lookup (f32, BF16/F16 packed, and GPTQ INT4 with per-group dequant)
  - `turboquant_encode.wgsl` / `turboquant_decode.wgsl` — TurboQuant KV cache compression with residual norm output (Google, ICLR 2026)
  - `matmul_q4.wgsl` — fused INT4 GPTQ dequantization matmul (unpacks 4-bit nibbles, applies group scales/zeros on the fly)
- **Buffer management** — typed GPU buffer creation, read/write operations
- **Kernel test suite** — 15 correctness tests against CPU reference values (elementwise, matmul, BF16 matmul, softmax, RMSNorm, TurboQuant round-trip, Lloyd-Max codebook MSE validation, asymmetric inner product estimation)
- **Metrics collection** — browser-to-server event reporting with JSON logging
- **Dev server** — Express metrics endpoint, local HF cache file serving with Range support
- **Local model loading** — serves SafeTensors from `~/.cache/huggingface/hub/` and project `models/` directory via dev server (50-100x faster than CDN). Local-first with automatic CDN fallback for missing files.

### Status

Phases 0-6 complete plus Gated DeltaNet/Mamba-2 hybrid support, TurboQuant KV cache, mixed-precision quantization, and local model loading:

- **Standard transformer inference** — full forward pass with GQA, RoPE, KV cache, autoregressive generation
- **Gated DeltaNet (Mamba-2) hybrid** — Qwen3.5's 24 linear attention layers + 8 full attention layers. New WGSL kernels: conv1d, SSM delta rule recurrence, L2 norm, per-head RMSNormGated. Fixed-size SSM state (~50 MB) instead of growing KV cache. Token-by-token prefill for correct SSM recurrence.
- **TurboQuant+ KV cache with asymmetric attention** — Asymmetric K3/V2 compressed KV cache (keys 3-bit, values 2-bit) with boundary layer protection (first/last 2 layers at full precision). Walsh-Hadamard rotation replaces random orthogonal matrix. Uses the QJL unbiased inner product estimator for keys to correct attention scores directly from compressed data. Current token K/V is always exact; only cached tokens get compressed. Also available as a PyTorch cache wrapper for the transformers engine (`core/turboquant_cache.py`).
- **Batch prefill** — 512-token chunks for standard transformers, token-by-token for hybrid models (SSM recurrence requires sequential processing)
- **Mixed-precision quantization** — custom quantization pipeline (`scripts/quantize_mixed_precision.py`) that keeps SSM-critical linear_attn layers in original BF16 precision while quantizing FFN and attention to INT4. `dispatchProjection` auto-selects f32/BF16/INT4 kernel per weight buffer.
- **GPTQ INT4** — weight loader and fused `matmul_bt_q4` kernel for quantized models. INT4 embedding lookup (`embed_q4`) and INT4 LM head matmul.
- **BF16 native weights** — `keepBF16` auto-detection keeps large BF16 tensors in native format on GPU, halving VRAM usage. Works for both unquantized models and mixed-precision GPTQ. BF16 embedding lookup and BF16 LM head matmul (including tied-embedding models).
- **Local model loading** — dev server serves SafeTensors from local HF cache and project `models/` directory. Local-first with automatic CDN fallback. 50-100x faster than CDN downloads.
- **Thinking mode** — Qwen3.5 `<think>` reasoning with visible thinking display and `</think>` answer extraction
- **Resilient downloads** — exponential backoff retry, parallel chunk prefetch, per-chunk browser caching for 7GB+ models
- **Auto-detect weight names** — discovers tensor name prefixes (handles `model.language_model.*` for multimodal architectures)
- **Model-specific RMSNorm** — auto-detects `(1 + weight)` vs `weight` convention per model family

### Supported Models

Any HuggingFace model with a standard transformer decoder architecture works. Hybrid Mamba-2 models (Qwen3.5) have full support with calibrated GPTQ quantization. Weight name prefixes are auto-detected. Tested:

- **Qwen3.5-9B HailMary** (`local/qwen3.5-9b-HailMary`) — 32 layers (24 Gated DeltaNet + 8 full attention), 4096 hidden, 16Q/4KV GQA. **5.74 GB — fits 8 GB VRAM with ~2 GB headroom.** Coherent, accurate responses at 2.5 tok/s on RTX 5060 Ti 8GB. BF16 embed + INT4 GPTQ SSM/attention/FFN + INT4 RTN lm_head. Thinking mode with visible reasoning chain.
- **Qwen3.5-9B noact** (`local/qwen3.5-9b-GPTQv2-noact`) — Same architecture, 9.37 GB. BF16 embed/lm_head/SSM + INT4 attention/FFN. 1.9 tok/s. Higher quality (BF16 lm_head) but larger VRAM footprint.
- **Qwen3.5-2B** (`Qwen/Qwen3.5-2B`) — 24 layers (18 Gated DeltaNet + 6 full attention), 2048 hidden, GQA 8Q/2KV. Coherent English at 5.2 tok/s with native BF16 weights (4.18 GB VRAM).
- **Qwen2.5-0.5B-Instruct** — 24 layers, 896 hidden, GQA 14Q/2KV. Generates coherent English at ~20 tok/s (f32).
- **SmolLM2-135M-Instruct** — 30 layers, 576 hidden, GQA 9Q/3KV.
- **SmolLM2-360M-Instruct** — 32 layers, 960 hidden, GQA 15Q/5KV.

### Calibrated GPTQ Quantization

For hybrid models like Qwen3.5 that combine Gated DeltaNet (linear attention / SSM) with standard softmax attention, INT4 quantization of SSM layers causes recurrence noise to compound — each token's error accumulates in the hidden state via geometric amplification (`h_t = A*h_{t-1} + B*x_t`, error grows as `A^N * epsilon`). Two approaches: **HailMary** quantizes everything (including SSM) to INT4 for maximum compression — this works in practice despite the theoretical risk. **Conservative (noact)** keeps SSM weights in BF16 for higher quality at the cost of larger VRAM.

```bash
# HailMary model: 5.74 GB, fits 8 GB VRAM (recommended)
# INT4 everything except embed + norms — aggressive but proven to work
python scripts/quantize_gptq.py \
  --model ./models/qwen3.5-9b \
  --output models/qwen3.5-9b-HailMary \
  --keep-bf16 norm embed_tokens

# noact model: 9.37 GB, higher quality but larger
# BF16 SSM + embed + lm_head for maximum quality, INT4 FFN + attention
python scripts/quantize_gptq.py \
  --model ./models/qwen3.5-9b \
  --output models/qwen3.5-9b-GPTQv2-noact \
  --no-actorder \
  --keep-bf16 linear_attn norm embed_tokens lm_head
```

**How it works:**
1. Loads the unquantized BF16 model (18 GB for 9B)
2. Runs 128 wikitext calibration samples through the model to compute Hessian matrices
3. Layer-by-layer GPTQ: quantize each weight using Hessian-guided error propagation, then re-run calibration through quantized layer before proceeding to the next
4. BF16 weights for norms, embedding, and any `--keep-bf16` patterns are copied unchanged
5. lm_head is quantized with RTN (round-to-nearest) — GPTQ segfaults on [248K, 4096] due to ~12 GB intermediates
6. Produces GPTQ v2 SafeTensors with g_idx for actorder support

**SSM quantization tradeoffs:**
- DeltaNet recurrence `h_t = A*h_{t-1} + B*x_t` amplifies quantization error geometrically in theory
- INT8 SSM projections: 0.009% weight error, but 1-7% output error per projection, compounding through 24 layers
- Early testing with INT8 SSM and INT8+KLT SSM produced gibberish, but calibrated INT4 GPTQ with actorder (HailMary) produces coherent output — GPTQ's Hessian-guided error minimization compensates for the recurrence sensitivity
- BF16 SSM (noact) provides higher quality at the cost of ~3.6 GB additional VRAM
- The HailMary model proves aggressive INT4 SSM is viable when paired with proper calibration

**Why BF16 embed is required (not INT4):**
- INT4 embedding has 0.993 cosine similarity to BF16 — close but the 0.7% error is amplified through 24 recurrent SSM layers, causing gibberish by position 2

**INT4 lm_head: works without KLT rotation:**
- Earlier testing showed INT4 lm_head produced gibberish, but that was with KLT rotation applied
- Without KLT, RTN INT4 lm_head achieves 1.04% relative error and produces valid token rankings in the 248K-way softmax
- Saves 1.4 GB vs BF16 lm_head (1.89 GB -> ~0.5 GB)

**VRAM budget for Qwen3.5-9B HailMary on 8 GB:**

| Component | Format | Size |
|-----------|--------|------|
| Embedding (248K x 4096) | BF16 | 1.9 GB |
| 24 linear_attn layers (5 proj each) | INT4 GPTQ | 0.8 GB |
| 32 FFN layers (3 x 12288x4096 each) | INT4 GPTQ | 2.4 GB |
| 8 full attention layers (4 proj each) | INT4 GPTQ | 0.2 GB |
| LM head (248K x 4096) | INT4 RTN | 0.5 GB |
| Norms, biases, small weights | f32/BF16 | 0.1 GB |
| SSM state + KV cache + intermediates | f32 | 0.2 GB |
| **Total** | | **~5.7 GB (~2 GB headroom)** |

### Numerical Verification

All 15 WGSL compute kernels have been verified against PyTorch reference implementations (`scripts/audit_kernels.py`):

| Kernel | Max Error vs PyTorch |
|--------|---------------------|
| F16 decode | 0 (bit-exact, all 65536 values) |
| RMSNorm (1+w) | 4.77e-07 |
| RoPE (interleaved, partial rotary) | 1.73e-06 |
| Attention (GQA, causal, fused softmax) | 1.19e-07 |
| SiLU | 2.38e-07 |
| INT4 GPTQ dequant | 1.05e-03 (f16 scale precision) |
| INT4 matmul (Kahan summation) | 2.67e-03 |

---

## Starting Services

There are two ways to run Artifex — the **Control Center** (desktop app, manages everything) or **manual terminal commands** (run each service individually).

### Option A: Control Center (Recommended)

Desktop dashboard that manages all services from one place. No terminal commands needed.

https://github.com/user-attachments/assets/5b8aa53a-042e-4a1f-83b9-d030cb7964a7

```bash
cd control-center
npm install       # first time only
npm run dev       # or: npm start
```

The Services panel shows each service with start/stop/restart buttons and **configurable options**:

| Service | Configurable Options |
|---------|---------------------|
| **Python API Server** | Port, Backend (transformers/ollama), Model name, Gateway URL |
| **Web Gateway** | Port |
| **Vite / Dev Server** | Start/stop only |
| **Ollama** | Start/stop only |
| **Artifex CLI / GUI** | Start/stop only |

When a service is stopped, its config row shows input fields. Set your backend to `ollama`, pick a model name, then click Start — the options are passed as CLI arguments automatically.

**Process cleanup**: When you stop a service or close the Control Center, it:
1. Tree-kills the process and all child workers (`taskkill /T /F`)
2. Kills anything still on the service's port (safety net)
3. For Ollama: hunts and kills orphaned runner subprocesses that hold VRAM

### Option B: Manual Terminal Commands

Run each service in its own terminal:

```bash
# Activate the venv first
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/macOS

# ── Core services ──
python main.py                           # CLI assistant
python main_gui_qt.py                    # GUI (PyQt6 desktop — recommended)
python main_gui.py                       # GUI (legacy FreeSimpleGUI)
python main_api.py                       # API server (port 8000)
python main_api.py --backend ollama      # API with Ollama backend
python main_api.py --backend ollama --model qwen3.5-27b-iq2xxs --port 8000

# ── Web gateway (separate terminal) ──
python web-gateway/main.py               # Web search proxy (port 8080)

# ── Ollama (if not already running) ──
ollama serve                             # Starts on port 11434

# ── WebGPU frontend (separate terminal) ──
cd webgpu
npx vite --host 127.0.0.1               # Vite dev server (port 5173)
npx tsx server/dev-server.ts             # Dev server (port 3001)
```

**Environment variables** (alternative to CLI flags):
```bash
set ARTIFEX_BACKEND=ollama              # Windows
set ARTIFEX_MODEL=qwen3.5-27b-iq2xxs
python main_api.py
```

### 6 Control Center Panels

| Panel | What it does |
|-------|-------------|
| **Services** | Start/stop/restart all 7 services with configurable options. Auto-detects already-running services by scanning ports. |
| **Logs** | Unified chronological log stream from all services. Color-coded by source, filterable by service/severity, text search, export to file. |
| **Quantize** | 6-step wizard: select model, profile SSM, edit recipe, review config, run with progress bar, done. HailMary (5.7 GB) and Conservative (9.4 GB) presets. |
| **Models** | Browse models/ directory. Cards show name, size, quantization config, shard count. Delete with confirmation. |
| **Docker** | Manage Docker containers from docker-compose.yml. Compose Up/Down, per-container Start/Stop, view logs. Graceful "Docker not installed" fallback. |
| **Cluster** | GPU cluster monitor. Connects to WebSocket hub, shows worker cards with GPU/VRAM/status, task queue, tok/s sparklines. |

### Control Center Architecture

- **Electron** — vanilla TypeScript, no React/Vue/axios
- **Dependencies**: electron 34.2.0, ws 8.20.0, typescript 5.6.3 (pinned, no auto-updates)
- **Security**: `nodeIntegration: false`, `contextIsolation: true`, all IPC through `contextBridge`
- **Service management**: `child_process.spawn()` with `shell: false` (handles paths with spaces), tree-kill + port-kill cleanup
- **Auto-detect**: scans service ports on startup, adopts externally-running processes
- **Interactive apps**: CLI launches in its own cmd.exe terminal window via PowerShell `Start-Process`

### Diagnostic & Research Tools

| Script | Purpose |
|--------|---------|
| `scripts/diagnose_quant_quality.py` | Progressive layer-by-layer quantization diagnostic. Compares Original vs KLT GPTQ quality side-by-side. |
| `scripts/validate_klt.py` | Side-by-side KLT rotation fidelity check through all 32 layers. |
| `scripts/test_ppl.py` | Quick perplexity + generation quality test for quantized models. |
| `scripts/quantize_gptq.py --profile-ssm` | SSM activation profiler with per-channel variance analysis. |
| `scripts/e8_codebook.py` | E8 lattice 2-bit codebook generator for future compression. |

---

## Project Structure

```
Artifex-Assistant-V5/
  main.py                  # CLI entry point
  main_gui_qt.py           # PyQt6 GUI entry point (recommended)
  main_gui.py              # Legacy FreeSimpleGUI entry point
  main_api.py              # API server entry point
  setup_wizard.py          # GPU detection and setup wizard
  setup_ollama.py          # Ollama setup helper (install, start, pull)
  download_model.py        # Model downloader (HuggingFace + Ollama)
  launch.bat               # Windows desktop launcher
  Dockerfile               # CUDA Docker image (full profile)
  docker-compose.yml       # Docker Compose: web gateway (default) + full profile
  pyproject.toml           # Pytest configuration
  requirements.txt         # Base dependencies (GPU-agnostic)
  requirements-rtx4090.txt # RTX 4090 24GB (pinned versions)
  requirements-3060.txt    # RTX 3060 12GB (pinned versions)
  requirements-rtx5060ti.txt # RTX 5060 Ti 8GB (pinned versions)
  core/
    config.py              # GPU tier detection, model registry, modes, safety
    engine_base.py         # Abstract engine interface
    engine_factory.py      # Backend factory (Transformers / Ollama)
    engine_transformers.py # HuggingFace Transformers backend
    engine_ollama.py       # Ollama backend (localhost only)
    hardware.py            # GPU detection and VRAM management
    model_loader.py        # Model weight loading with quantization
    model_registry.py      # Model type detection and VRAM estimation
    inference.py           # Token counting, thinking blocks, response cleaning
    knowledge.py           # RAG knowledge base with ContextEngine
    rag.py                 # Retrieval-augmented generation pipeline
    health.py              # System health checks and diagnostics
    resilience.py          # OOM recovery and crash resilience
    session.py             # Session save/load with metadata
    prompts.py             # System prompt builders
    code_mode.py           # Code execution mode
    progress.py            # Progress tracking
    tool_protocol.py       # Agent tool execution protocol
    logging_config.py      # Structured logging
    services/              # Shared multimodal service layer
      __init__.py          # Singleton get_service() accessor
      multimodal_service.py # Pipeline caching, VRAM eviction, cancellation
      file_manager.py      # File upload/download/generated content management
    pipelines/             # Multi-modal model pipelines (10 types)
      base.py              # Abstract pipeline interface
      registry.py          # Pipeline discovery and factory
      text_gen.py          # Text generation / LLM chat
      image_gen.py         # Text-to-image (SD, SDXL, FLUX)
      image_edit.py        # Image editing (inpaint, upscale)
      vision.py            # Vision models (LLaVA, Qwen-VL)
      shape_3d.py          # 3D generation (ShapE)
      embedding.py         # Embedding models for RAG
      audio.py             # Audio (TTS + STT)
      music.py             # Music generation (MusicGen)
      video_gen.py         # Video generation
  ui/
    cli_assistant.py       # CLI assistant loop with tool execution + multimodal pipelines
    qt_gui.py              # PyQt6 main window (recommended GUI)
    qt_theme.py            # PyQt6 theme system (5 themes, QSS stylesheets)
    qt_workers.py          # QThread workers + TokenBatcher for smooth streaming
    qt_widgets.py          # DropZone, ImageViewer, AudioPlayer, VideoPlayer, MicRecorder, ChatView
    qt_launcher.py         # PyQt6 QApplication setup
    cyber_gui.py           # Legacy FreeSimpleGUI GUI
    gui_theme.py           # Legacy GUI theming
    terminal.py            # Terminal utilities
  api/
    server.py              # FastAPI OpenAI-compatible REST API (streaming for both backends, tool execution)
    web_tools.py           # Web search tool extraction and execution (@search, @web_read only)
  tools/
    agent_tools.py         # Shell, Python, web search, file ops, edit (gateway-aware)
    codebase_tools.py      # Code analysis, symbol search, architecture mapping
    tool_cache.py          # LRU cache for tool outputs + SessionMap
  web-gateway/
    Dockerfile             # Python 3.12-slim container (~150 MB)
    main.py                # FastAPI proxy: search, fetch, download, quarantine
    sanitizer.py           # Content extraction + prompt injection detection
    config.py              # URL filtering, rate limits, size limits, optional auth
    requirements.txt       # fastapi, trafilatura, httpx, slowapi
    searxng/
      settings.yml         # SearXNG API-only config (8 search engines)
      limiter.toml         # SearXNG rate limiting
  webgpu/
    src/
      main.ts              # WebGPU initialization, model loading, and chat UI
      engine/
        gpu-device.ts      # WebGPU adapter/device initialization
        buffers.ts         # GPU buffer create, read, write utilities
        compute.ts         # Shader compilation, pipeline caching, dispatch
        kernel-tests.ts    # GPU kernel correctness tests (15 tests)
        forward-pass.ts    # Full transformer forward pass orchestrator
        generate.ts        # Autoregressive generation loop with sampling
        inference.ts       # Top-level session orchestrator (load → chat)
        turboquant-pipeline.ts  # TurboQuant encode/decode pipeline manager
      model/
        model-config.ts    # HF config.json parser (Qwen/Llama/Mistral/Phi)
        tokenizer.ts       # BPE tokenizer via @huggingface/transformers
        weight-loader.ts   # SafeTensors download → parse → GPU upload
        safetensors.ts     # SafeTensors binary format parser (F32/F16/BF16)
        hf-hub.ts          # HuggingFace Hub API client
        cache.ts           # Browser IndexedDB cache for model weights
        turboquant.ts      # TurboQuant math (rotation matrix, codebook, CPU reference)
      shaders/             # WGSL compute kernels (15 kernels)
        matmul.wgsl        # f32/BF16 tiled matmul with Kahan summation
        matmul_q4.wgsl     # INT4 GPTQ dequant matmul (per-weight g_idx actorder)
        matmul_q8.wgsl     # INT8 dequant matmul (verified correct)
        matmul_e8.wgsl     # E8 2-bit codebook matmul (untested)
        hadamard.wgsl      # Fast Walsh-Hadamard transform
        rmsnorm.wgsl       # RMSNorm with (1+w) support, f32 accumulators
        attention.wgsl     # Fused QKV attention with stable softmax
        ssm_step.wgsl      # Gated DeltaNet recurrence kernel
        # + rope, embed, elementwise, conv1d, group_norm, l2norm, turboquant
      utils/               # Metrics reporting
    server/
      dev-server.ts        # Express metrics + WebSocket hub
  control-center/            # Electron desktop dashboard (6 panels)
    src/
      main/
        main.ts            # Electron window, tray, IPC registration
        preload.ts         # contextBridge typed API for renderer
        ipc-handlers.ts    # Central IPC router for all panels
        services/          # Service manager (spawn, kill, port-based adopt)
        logs/              # Log aggregator + ring buffer (10K lines)
        quantization/      # Quantize runner (drives quantize_gptq.py)
        models/            # Model scanner (reads models/ directory)
        docker/            # Docker CLI wrapper (no npm deps)
        cluster/           # WebSocket client + time-series state
        state/             # JSON persistence in userData
      renderer/
        panels/            # 6 panel UIs (services, logs, quantize, models, docker, cluster)
        components/        # Shared UI (modal, toast, progress bar, mini graph, wizard steps)
        styles/            # Dark theme (--bg: #0f111a, --accent: #00f0ff)
  scripts/                   # Quantization & diagnostic tools
    quantize_gptq.py       # GPTQ v2 quantizer (INT4/INT8/E8, KLT, recipes)
    diagnose_quant_quality.py  # Layer-by-layer quantization diagnostic
    validate_klt.py        # KLT rotation fidelity validator
    test_ppl.py            # Perplexity tester
    e8_codebook.py         # E8 lattice codebook generator
  recipes/                   # Quantization recipes (per-layer precision)
  tests/                     # Pytest test suite (12 modules, 110 tests)
  knowledge/                 # RAG knowledge bases (reference + workspaces)
  models/                    # Local model cache (auto-discovered)
  sessions/                  # Saved conversations
  output/                    # Generated outputs (images, audio, 3D, video)
  logs/                      # Application logs
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARTIFEX_API_KEY` | *(none)* | API authentication key (optional) |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device index (for multi-GPU systems) |
| `CUDA_MODULE_LOADING` | `LAZY` | Deferred CUDA kernel compilation (saves ~300-400 MB VRAM) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True,garbage_collection_threshold:0.8` | CUDA memory allocation tuning |
| `WEB_GATEWAY_URL` | *(none)* | Web gateway URL for CLI/GUI (e.g., `http://localhost:8080`). For the API server, use `--gateway` flag instead. Auto-set in Docker full profile. |
| `GATEWAY_AUTH_TOKEN` | *(none)* | Optional shared secret for web gateway authentication. When set, all gateway requests must include a matching `X-Gateway-Token` header. The `/health` endpoint is always open (for Docker healthchecks). Leave unset to disable (default). |
| `HF_TOKEN` | *(none)* | HuggingFace auth token for gated models (Qwen3.5, etc.). Enter in the WebGPU UI sidebar or set via browser localStorage. |

> **Note on environment variables in PowerShell:** Use `$env:VAR = "value"` syntax (not `set VAR=value` which is CMD-only). For the API server, the `--gateway` flag avoids env var hassles entirely.

### Context Profiles

| Profile | Max Input | Max Output | History | Knowledge Budget | Use Case |
|---------|-----------|------------|---------|-----------------|----------|
| STANDARD | 10,000 tokens | 1,536 tokens | 6,000 tokens | 550 tokens | TIGHT/COMFORTABLE GPUs |
| HIGH | 14,000 tokens | 2,048 tokens | 10,000 tokens | 750 tokens | ABUNDANT GPUs (24+ GB) |

Switch profiles in the CLI with `/context STANDARD` or `/context HIGH`.

### Modes

| Mode | Temperature | Max Tokens | Thinking | Use Case |
|------|-------------|------------|----------|----------|
| ASSISTANT | 0.7 | 2048-4096 (by tier) | Enabled | General conversation and tool use |
| CODE | 0.2 | 4096-8192 (by tier) | Enabled | Code generation and analysis |

---

## Security

- All servers bind to `127.0.0.1` — not accessible from LAN or internet
- Dangerous command patterns blocked (rm -rf, format, registry edits, shutdown, etc.)
- API key authentication via `ARTIFEX_API_KEY` environment variable
- Ollama communication stays on localhost:11434
- WebGPU Vite and metrics servers locked to localhost
- Docker network isolation — main AI container has no internet access (full profile)
- Optional gateway authentication via `GATEWAY_AUTH_TOKEN` — shared secret between Artifex and the web gateway. Disabled by default; enable if you expose port 8080 to the network.
- Edit operations validate syntax before applying changes
- **Web content sandboxing:**
  - Content sanitized via trafilatura before reaching the model
  - Prompt injection detection with 20+ patterns (instruction override, role manipulation, delimiter attacks, encoded content, data exfiltration)
  - Downloads quarantined in tmpfs (RAM-backed, never on disk, auto-deleted)
  - URL filtering blocks private IPs, metadata endpoints, dangerous schemes, and high-abuse TLDs
  - Executable file extensions (.exe, .bat, .ps1, .dll, etc.) blocked from download
  - Rate limiting on all web operations (search, fetch, download)
  - Session cleanup wipes quarantine on `/clear` and `/cleanup`

---

## Testing

```bash
# Run the full test suite (110 tests)
python -m pytest

# Verbose with short tracebacks
python -m pytest tests/ -v --tb=short

# Run specific modules
python -m pytest tests/test_config.py -v         # Config, GPU tiers, modes
python -m pytest tests/test_pipelines.py -v       # Pipeline registry and contracts
python -m pytest tests/test_knowledge.py -v       # Knowledge base CRUD + ContextEngine
python -m pytest tests/test_agent_tools.py -v     # Agent tool extraction
python -m pytest tests/test_inference.py -v       # Think filtering, history compression
python -m pytest tests/test_session.py -v         # Session save/load
python -m pytest tests/test_resilience.py -v      # OOM recovery
python -m pytest tests/test_health.py -v          # Health checks
python -m pytest tests/test_model_registry.py -v  # Model type detection
python -m pytest tests/test_tool_cache.py -v      # Cache and SessionMap

# WebGPU type-check
cd webgpu && npx tsc --noEmit

# WebGPU production build
cd webgpu && npx vite build

# WebGPU kernel tests — run in browser via "Run GPU Tests" button
cd webgpu && npm run dev
```

---

## Troubleshooting

### "No CUDA GPU detected"
- Make sure NVIDIA drivers are installed: `nvidia-smi` should show your GPU
- Reinstall PyTorch with the correct CUDA version for your card

### "Out of memory" (OOM)
- Try `/refresh` in the CLI to compress history and free VRAM
- Switch to a smaller model or use Ollama with a pre-quantized model
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (done automatically)

### "Ollama not reachable"
- Run `python setup_ollama.py --status` to check the server
- Start Ollama manually: `ollama serve`
- Make sure a model is pulled: `ollama pull qwen3.5:9b`

### Models not showing in GUI/CLI
- Models must be in the `models/` directory (or shared model paths)
- Each model needs a `config.json` for type detection
- Run `python download_model.py --list` to see what's installed

### WebGPU not working in browser
- Use Chrome 113+ or Edge 113+
- Enable `chrome://flags/#enable-unsafe-webgpu`
- Check that your GPU supports WebGPU (most discrete GPUs from 2018+ do)

### Web gateway not working
- Make sure Docker Desktop is running: the whale icon should be in your system tray
- Start the gateway: `docker compose up`
- Check health: `curl http://localhost:8080/health`
- If SearXNG shows "unreachable", wait 15-20 seconds for it to initialize
- Without the gateway, Artifex falls back to direct DuckDuckGo search automatically

### Windows Firewall popup
- All servers default to `127.0.0.1` — you should NOT see a firewall prompt
- If you do, deny the request — nothing needs network access

---

## Requirements

### Python (core)

- Python 3.11+
- PyTorch with CUDA (12.4 for RTX 30xx/40xx, 12.8 for RTX 50xx)
- transformers >= 5.2.0
- bitsandbytes >= 0.49.0
- accelerate >= 1.6.0
- numpy >= 1.24.0
- diffusers >= 0.30.0 (for image/3D/video pipelines)
- FreeSimpleGUI >= 5.0.0 (for GUI mode)
- FastAPI + uvicorn (for API server mode)
- pytest >= 8.0.0 (for testing)

### Node.js (WebGPU engine)

- Node.js 18+
- TypeScript 5.7+
- Vite 6+
- Chrome/Edge with WebGPU support

### Optional

- Ollama — for pre-quantized model support (qwen3.5:9b, qwen3:8b, etc.)
- librosa + soundfile — for audio processing
- scipy — for audio file output (WAV writing)
- sentence-transformers — for embedding models
- Docker + NVIDIA Container Toolkit — for container deployment

---

## Acknowledgments

> *"For from him and through him and for him are all things. To him be the glory forever! Amen."*
> — Romans 11:36 (NIV)

First and foremost, all praise and glory to **God** through **Jesus Christ**. Every good gift comes from above, and this work is no exception.

### Frameworks & Libraries

- **[PyTorch](https://pytorch.org/)** — Meta AI — deep learning framework powering all GPU inference
- **[HuggingFace Transformers](https://huggingface.co/docs/transformers)** — HuggingFace — model loading, tokenization, and inference
- **[HuggingFace Diffusers](https://huggingface.co/docs/diffusers)** — HuggingFace — image, video, and 3D generation pipelines
- **[HuggingFace Accelerate](https://huggingface.co/docs/accelerate)** — HuggingFace — multi-GPU support and memory-efficient loading
- **[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)** — bitsandbytes-foundation — NF4 and INT8 quantization
- **[safetensors](https://github.com/huggingface/safetensors)** — HuggingFace — safe model weight serialization
- **[Ollama](https://ollama.com/)** — Ollama — local model serving with pre-quantized GGUF models
- **[FastAPI](https://fastapi.tiangolo.com/)** — Sebastian Ramirez — REST API framework
- **[Uvicorn](https://www.uvicorn.org/)** — Encode — ASGI server
- **[FreeSimpleGUI](https://github.com/spyoungtech/FreeSimpleGUI)** — desktop GUI framework
- **[Vite](https://vite.dev/)** — Evan You / Vite team — frontend build tool for WebGPU engine
- **[TypeScript](https://www.typescriptlang.org/)** — Microsoft — typed JavaScript for WebGPU engine
- **[Express](https://expressjs.com/)** — OpenJS Foundation — metrics dev server
- **[NumPy](https://numpy.org/)** — NumPy team — numerical computing for embeddings and RAG
- **[Pillow](https://python-pillow.org/)** — Pillow contributors — image processing
- **[DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search)** — deedy5 — web search integration (fallback)
- **[SearXNG](https://github.com/searxng/searxng)** — SearXNG team — self-hosted meta-search engine (primary search via gateway)
- **[trafilatura](https://github.com/adbar/trafilatura)** — Adrien Barbaresi — web content extraction and sanitization
- **[HTTPX](https://github.com/encode/httpx)** — Encode — async HTTP client for the web gateway
- **[SlowAPI](https://github.com/laurentS/slowapi)** — Laurent Savaete — rate limiting for FastAPI
- **[Transformers.js](https://github.com/huggingface/transformers.js)** — HuggingFace — JavaScript tokenizer and model inference (used for BPE tokenization in WebGPU engine)

### AI Models & Research

- **[Qwen](https://github.com/QwenLM/Qwen)** — Alibaba Cloud / Qwen Team — Qwen3.5, Qwen3, Qwen2.5-Coder, Qwen-VL models
- **[Stable Diffusion / SDXL](https://stability.ai/)** — Stability AI — text-to-image generation
- **[FLUX](https://blackforestlabs.ai/)** — Black Forest Labs — text-to-image generation
- **[ShapE](https://github.com/openai/shap-e)** — OpenAI — text-to-3D mesh generation
- **[MusicGen](https://github.com/facebookresearch/audiocraft)** — Meta AI / FAIR — music generation
- **[Bark](https://github.com/suno-ai/bark)** — Suno AI — text-to-speech
- **[Whisper](https://github.com/openai/whisper)** — OpenAI — speech recognition
- **[LLaVA](https://llava-vl.github.io/)** — University of Wisconsin-Madison / Microsoft — vision-language models
- **[Llama](https://ai.meta.com/llama/)** — Meta AI — large language models
- **[Mistral](https://mistral.ai/)** — Mistral AI — language models
- **[Gemma](https://ai.google.dev/gemma)** — Google DeepMind — language models
- **[Phi](https://huggingface.co/microsoft/phi-3-mini-4k-instruct)** — Microsoft Research — small language models
- **[DeepSeek](https://www.deepseek.com/)** — DeepSeek AI — language models
- **[SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)** — HuggingFace — small language models (used for WebGPU testing)
- **[CogVideoX](https://github.com/THUDM/CogVideo)** — Tsinghua University / THUDM — video generation
- **[BERT](https://github.com/google-research/bert)** — Google Research — embedding models
- **[Sentence Transformers](https://www.sbert.net/)** — UKP Lab, TU Darmstadt — embedding models for RAG

### Algorithms & Research

- **[TurboQuant](https://arxiv.org/abs/2504.19874)** — Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni et al. (Google Research) — Online vector quantization for KV cache compression with near-optimal distortion rate (ICLR 2026). Our implementation includes the full two-stage pipeline (PolarQuant + QJL) with an asymmetric attention kernel that applies the QJL inner product correction directly during score computation, avoiding the quality loss of naive decode-then-attend.
- **[PolarQuant](https://arxiv.org/abs/2502.00527)** — Google Research — MSE-optimal scalar quantization via random orthogonal rotation (AISTATS 2026). Stage 1 of TurboQuant.
- **[QJL](https://github.com/amirzandieh/QJL)** — Amir Zandieh et al. — 1-bit quantized Johnson-Lindenstrauss transform for unbiased inner product estimation (AAAI 2025). Stage 2 of TurboQuant. The asymmetric estimator `<q, k̂> + ||k||·||r||·√(π/2)/√d · <S·Π·q, sign(S·r)>` is implemented as a dedicated WGSL attention kernel.
- **[Lloyd-Max quantizer](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm)** — Stuart Lloyd (Bell Labs, 1957/1982) — Optimal scalar quantization for known distributions. Our hardcoded codebook centroids and thresholds for N(0,1) match the Lloyd-Max optimal values.

### Validation & Reference Implementations

- **[tonbistudio/TurboQuant](https://github.com/tonbistudio/TurboQuant)** — tonbistudio — PyTorch reference implementation with Lloyd-Max codebook computation and real-model KV validation on Qwen2.5-3B. Their MSE measurements (3-bit: 0.034, 4-bit: 0.009) and the critical insight that naive decode-then-attend produces garbage (23-44% per-vector error) motivated our asymmetric attention kernel. Their `asymmetric_attention_scores()` in `compressors.py` served as algorithmic reference for the QJL inner product correction.

### Standards & Specifications

- **[WebGPU](https://www.w3.org/TR/webgpu/)** — W3C GPU for the Web Working Group — browser GPU compute API
- **[WGSL](https://www.w3.org/TR/WGSL/)** — W3C — WebGPU Shading Language
- **[@webgpu/types](https://github.com/gpuweb/types)** — W3C GPU for the Web CG — TypeScript type definitions

### Tools & Infrastructure

- **[NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit)** — NVIDIA — GPU computing platform
- **[Docker](https://www.docker.com/)** — Docker Inc. — containerization
- **[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)** — NVIDIA — GPU support in Docker
- **[pytest](https://docs.pytest.org/)** — Holger Krekel and pytest contributors — testing framework

---

## Acknowledgements

- **TurboQuant** (Google, ICLR 2026, arXiv:2504.19874) — PolarQuant + QJL KV cache compression algorithm. Our WebGPU and PyTorch implementations are based on the original paper.
- **TurboQuant+** ([TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)) — Key findings that improved our implementation: asymmetric K/V compression ("V compression is free"), boundary layer protection (first/last 2 layers at full precision), and Walsh-Hadamard rotation replacing random orthogonal matrices. These improvements are integrated into both the WebGPU WGSL shaders and the PyTorch `TurboQuantCache`.

---

## License

Artifex Assistant V5 — Universal Local AI Hosting Platform.
