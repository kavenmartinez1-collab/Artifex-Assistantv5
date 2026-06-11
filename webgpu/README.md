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

## Quickstart

Requires **Node.js 18+** and a **WebGPU-capable browser**. **Chrome is the
reference target** (most reliable WebGPU); Edge usually matches but can lag,
and Firefox/Safari need WebGPU enabled.

```bash
npm install
npm run dev
```

This starts the Vite dev server and a small local file server, then opens the
app. Type a HuggingFace repo (e.g. `Qwen/Qwen2.5-0.5B-Instruct`) into the model
box and click **Load**, or pick a local model from the browser (see below).

> All weights are fetched and run client-side. The bundled Node server only
> serves local model files and reports GPU info — it performs no inference.

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
K-quants `Q4_K_M`/`Q5_K_M`/`Q6_K` (plus F16/F32/BF16). `IQ*` (imatrix) and
`Q4_1`/`Q5_1`/`Q2_K`/`Q3_K` aren't supported yet; the engine says so before
downloading. When in doubt, grab a `*-Q4_K_M.gguf`.

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
