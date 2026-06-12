# Artifex WebGPU

**Browser-based LLM inference on any GPU — from scratch, in WGSL.**

A from-the-ground-up transformer inference engine that runs entirely in your
browser. No server-side inference, no Python, no CUDA — just WebGPU compute
shaders. It runs dense models, DeltaNet hybrids, 35B-class Mixture-of-Experts
(experts streamed from CPU RAM via WASM workers), and vision-language models,
loading weights directly from HuggingFace, GGUF files, or your local Ollama
store.

> Part of the [Artifex](https://github.com/kavenmartinez1-collab/Artifex-Assistantv5)
> project; this engine also lives standalone here.

## Highlights

- **Pure-WGSL transformer** — handwritten kernels for matmul (f32/INT4/INT8/k-quant),
  attention, LayerNorm, RMSNorm, RoPE, softmax, SwiGLU, and embedding. No ML framework.
- **Runs real models** — dense Qwen2.5/3, DeltaNet hybrids (Qwen3.5/3.6), and
  Qwen3.6-35B-A3B MoE with experts held in CPU RAM across a WASM worker fleet.
- **Three weight sources** — HuggingFace SafeTensors (streamed via HTTP range),
  GGUF (Q4_K/Q5_K/Q6_K), and Ollama-store blobs — all auto-discovered.
- **Browser-native vision** — a from-scratch WGSL vision transformer encodes
  images for Qwen3-VL (parity-verified against HuggingFace transformers to ~1e-4)
  and Qwen3.6 multimodal. Drag-drop, paste, or attach.
- **Real chat** — multi-turn history with KV-prefix reuse, sliding-window context
  management, key-point compression, and session save/load.
- **TurboQuant KV** — 3–4 bit KV cache compression (~80% memory) on supported models.
- **Per-card auto-config** — VRAM budget and attention limits derived from the GPU.

## Setup — step by step

New to this? Follow these in order. You'll copy-paste a few commands; that's it.

### 1. Install the two things you need

- **Node.js** (version 18 or newer) — download the "LTS" installer from
  [nodejs.org](https://nodejs.org/) and run it (click Next through the prompts).
- **Git** — from [git-scm.com/downloads](https://git-scm.com/downloads) (also
  just click through). Git is how you copy the project to your computer.
- A **WebGPU browser** — **use Google Chrome** (it works most reliably). You
  almost certainly already have a recent enough version.

### 2. Open a terminal

- **Windows**: press the Start key, type `cmd`, press Enter.
- **Mac**: press Cmd+Space, type `terminal`, press Enter.

You'll paste commands here and press Enter after each.

### 3. Download the project (clone it)

```bash
git clone https://github.com/kavenmartinez1-collab/Artifex_Web.git
cd Artifex_Web
```

The first line copies the project into a folder; the second moves into it.

### 4. Install and start

```bash
npm install
npm run dev
```

`npm install` downloads the project's building blocks (takes a minute, one
time only). `npm run dev` starts it and **opens the app in your browser
automatically**. Leave this terminal window open while you use the app — it's
the local server. To stop it later, click the terminal and press `Ctrl+C`.

### 5. Load a model and chat

In the app, the easiest first run: click **Browse** to see models already on
your computer (from Ollama, if you have it), or type a small model name like
`Qwen/Qwen2.5-0.5B-Instruct` into the model box and click **Load**. Wait for
"inference engine ready," then type a message. The first load downloads the
model (small models are a few hundred MB); after that it's cached.

**Next time** you want to use it, you only need steps 2 and 4 — open a terminal,
`cd Artifex_Web`, and `npm run dev`.

> Everything runs on **your** computer, in your browser — no data is sent
> anywhere. The small local server only hands model files to the browser and
> reports your GPU's free memory; it does no AI itself.

## Using local models

Three sources are auto-discovered — none require configuration:

| Source | How |
|---|---|
| **Ollama** | Any pulled model appears as `ollama/<name>:<tag>`. |
| **HuggingFace cache** | Anything previously downloaded (`HF_HOME` respected). |
| **`models/` folder** | Drop GGUF files or model folders in `models/` (see `models/README.md`). |

To point at model directories elsewhere on disk, create
`model-dirs.local.json` next to `package.json` (gitignored):

```json
["D:/llm-models/gguf", "D:/my-quants"]
```

or set `ARTIFEX_MODEL_DIRS` (`;`-separated). Only aliases ever reach the
browser — absolute paths stay server-side.

## Choosing a model

The engine fits weights in VRAM (experts for MoE models stream from system
RAM). It tells you up front if a model won't fit on your GPU — pick a smaller
one or a lower quant. As a rough guide, a model needs a bit more free VRAM than
its file size; an ~8 GB card comfortably runs 7–9B models at Q4_K_M.

**GGUF quantization**: the engine runs `Q4_0`, `Q5_0`, `Q8_0`, and the
K-quants `Q2_K`/`Q3_K`/`Q4_K_M`/`Q5_K_M`/`Q6_K` (plus F16/F32/BF16). `IQ*`
(imatrix) and `Q4_1`/`Q5_1` aren't supported yet; the engine says so before
downloading. When in doubt, grab a `*-Q4_K_M.gguf`.

**Model families**: verified — Llama, Qwen3 / Qwen3.5 / Qwen3.6 (incl. the
35B MoE), Gemma 4. Experimental (recognized, attempted, not yet fully
verified — the app flags these on load) — Qwen2.5, Mistral. Not yet
supported: Gemma 2/3, Phi-3 (fused QKV), DeepSeek (MLA), and Mamba2 hybrids
(Nemotron-H, Granite-hybrid). Loading an unsupported architecture gives a
clear message naming it, never silent garbage.

## Vision

Load a vision model (e.g. `Qwen/Qwen3-VL-4B-Instruct`, or a Qwen3.6 GGUF with a
sibling `mmproj-*.gguf`) and the 📎 button activates. Attach, paste, or
drag-drop an image; supported formats are PNG, JPEG, WebP, GIF, BMP. The vision
transformer runs in WGSL alongside the language model.

Qwen3-VL and Qwen3.6 are parity-verified. Other families (Gemma 4) are marked
**EXPERIMENTAL** in the UI until they pass a reference parity run.

## How it works

```
HF SafeTensors ─┐
local GGUF       ├─→  ModelDescriptor  ─→  WGSL forward pass  ─→  streaming chat
Ollama blob     │    (per-layer kind,        (attention / DeltaNet
images ──────────┘    rope, MoE, vision)      SSM / MoE / vision tower)
```

A single descriptor parameterizes the whole forward pass, so one engine serves
many architectures and weight formats. Correctness is held by parity harnesses
that diff against reference implementations (llama.cpp for text, HuggingFace
transformers for vision) — see the `__VISION_PARITY__` and `__TQ_PARITY__`
console helpers.

## Developer scripts

```bash
npm run typecheck     # tsc --noEmit
npm run build         # production bundle
npm run test:e2e      # Playwright kernel tests (headless WebGPU)
```

## License

MIT — see [LICENSE](LICENSE).
