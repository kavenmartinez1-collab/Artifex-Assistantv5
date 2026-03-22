# Artifex Assistant V5 — Learning Guide

> A self-taught developer's deep-dive into everything under the hood.
> No CS degree assumed. Every concept explained from the ground up.

---

## Table of Contents

1. [How AI Models Actually Work](#1-how-ai-models-actually-work)
2. [The Transformer Architecture](#2-the-transformer-architecture)
3. [Quantization — Fitting Big Models in Small GPUs](#3-quantization--fitting-big-models-in-small-gpus)
4. [Backend #1: HuggingFace Transformers](#4-backend-1-huggingface-transformers)
5. [Backend #2: Ollama](#5-backend-2-ollama)
6. [Backend #3: The API Server (OpenAI-Compatible)](#6-backend-3-the-api-server-openai-compatible)
7. [Backend #4: WebGPU — Running Models in the Browser](#7-backend-4-webgpu--running-models-in-the-browser)
8. [The GPU Compute Pipeline — Why Kernels Matter](#8-the-gpu-compute-pipeline--why-kernels-matter)
9. [Our WGSL Kernels Explained](#9-our-wgsl-kernels-explained)
10. [The WebGPU Inference Roadmap](#10-the-webgpu-inference-roadmap)
11. [How All the Pieces Connect](#11-how-all-the-pieces-connect)
12. [Glossary](#12-glossary)

---

## 1. How AI Models Actually Work

### What is a "model"?

A model is a giant file full of numbers — billions of them. These numbers are called **weights** or **parameters**. They were learned during **training**, where the model read enormous amounts of text and adjusted its weights to get better at predicting the next word.

When you ask "What is the capital of France?", the model doesn't look up the answer. It processes your question through layers of math operations and its weights collectively encode the pattern: "when someone asks about the capital of France, the most likely next tokens are 'The', 'capital', 'of', 'France', 'is', 'Paris'."

### Tokens — not words, not characters

Models don't work with words or letters. They work with **tokens** — chunks of text that the **tokenizer** breaks your input into. For example:

```
Input:  "Hello, how are you?"
Tokens: ["Hello", ",", " how", " are", " you", "?"]
Token IDs: [9906, 11, 1268, 527, 499, 30]
```

Each token maps to a number (token ID). The model only works with these numbers. Common words are single tokens; rare words get split into pieces ("unbelievable" might become ["un", "believ", "able"]).

Qwen3.5 has a vocabulary of ~152,000 tokens. Every token ID is an index into a 152,000-row table of learned vectors.

### The forward pass — input to output

When you send a prompt, the model does a **forward pass**:

1. **Tokenize** — convert your text to token IDs
2. **Embed** — look up each token ID in the embedding table to get a vector (a list of numbers, typically 3584 numbers long for Qwen3.5-9B)
3. **Process** — pass through 28-40 transformer layers (each layer refines the meaning)
4. **Project** — convert the final vector back to a probability distribution over all 152,000 tokens
5. **Sample** — pick the most likely next token (with some randomness controlled by **temperature**)
6. **Repeat** — append that token, run the forward pass again for the next token

This is why generation is slow — it produces **one token at a time**, and each token requires the entire model to run.

### Why GPUs?

Each layer involves multiplying matrices with millions of entries. A CPU does these one at a time. A GPU has thousands of small cores that do them in parallel. An RTX 5060 Ti has 4,608 CUDA cores — it can do 4,608 multiply-adds simultaneously. That's why a GPU is 10-100x faster than a CPU for AI inference.

---

## 2. The Transformer Architecture

The "transformer" is the specific math architecture that powers GPT, Qwen, Llama, and virtually every modern language model. Here's what happens inside each layer:

### The layer stack

A model like Qwen3.5-9B has **28 layers** stacked on top of each other. Each layer has the same structure:

```
Input vector (3584 numbers per token)
    │
    ├── RMSNorm (normalize the numbers)
    ├── Self-Attention (figure out which tokens relate to each other)
    ├── Add (residual connection — add the input back)
    │
    ├── RMSNorm (normalize again)
    ├── Feed-Forward Network (transform each token independently)
    ├── Add (residual connection again)
    │
Output vector (3584 numbers per token)
```

### RMSNorm — keeping numbers stable

As numbers flow through 28 layers of multiplication, they can explode (get huge) or vanish (get tiny). **RMSNorm** (Root Mean Square Normalization) rescales them at each layer:

```
rms = sqrt(mean(x²) + epsilon)
output = (x / rms) * learned_weight
```

This keeps the numbers in a reasonable range. Qwen uses RMSNorm instead of the older LayerNorm because it's simpler (no mean subtraction) and works just as well.

**Why we built an RMSNorm kernel**: Every single layer needs this operation. For 28 layers, that's 56 RMSNorm calls per token. If this is slow, everything is slow.

### Self-Attention — the magic ingredient

This is the core innovation of transformers. Attention lets each token "look at" every other token to understand context.

**Example**: In "The cat sat on the mat because it was tired", when processing "it", attention figures out that "it" refers to "the cat" (not "the mat") by computing similarity scores between all token pairs.

How it works:

1. Each token creates three vectors: **Query** (Q), **Key** (K), and **Value** (V)
   - Q = "What am I looking for?"
   - K = "What do I contain?"
   - V = "What information do I provide?"

2. For each token, compute **attention scores** = how much does my Query match each other token's Key?
   ```
   scores = Q × K^T / sqrt(dimension)    ← this is a matrix multiplication
   ```

3. Apply **softmax** to convert scores to probabilities (they sum to 1.0)

4. **Weighted sum**: multiply each token's Value by its attention score and add them up
   ```
   output = softmax(scores) × V    ← another matrix multiplication
   ```

**Grouped-Query Attention (GQA)**: Qwen3.5 uses a memory optimization where multiple Query heads share the same Key/Value heads. Instead of 28 Q heads each with their own K and V, there might be 28 Q heads but only 4 K/V heads. This reduces memory by ~7x for the KV-cache with minimal quality loss.

**KV-Cache**: During generation, each new token needs to attend to ALL previous tokens. Rather than recomputing K and V for old tokens, we cache them. This is why longer conversations use more VRAM — the KV-cache grows with every token.

**Causal mask**: The model can only look at tokens that came BEFORE the current position (not future tokens). This is enforced by setting future attention scores to negative infinity before softmax, which drives them to zero.

### RoPE — Rotary Position Embeddings

The attention mechanism has no built-in notion of token order — "cat sat mat" and "mat sat cat" would produce identical attention scores. **RoPE** (Rotary Position Embeddings) encodes position by rotating the Q and K vectors based on their position in the sequence:

```
Q_rotated[i] = Q[i] * cos(position * frequency) + Q[i+1] * sin(position * frequency)
```

Think of it like a clock — position 0 points at 12 o'clock, position 1 rotates slightly, position 1000 has rotated many times. The dot product between two rotated vectors depends on their *relative* position, which is exactly what the model needs.

**Why we built a RoPE kernel**: This rotation must happen to every Q and K vector at every layer. It's computationally simple but called thousands of times per generation.

### Feed-Forward Network (SwiGLU)

After attention, each token passes through a feed-forward network independently. Qwen uses **SwiGLU** (Swish-Gated Linear Unit):

```
gate = SiLU(x × gate_weights)     ← SiLU(x) = x * sigmoid(x)
up   = x × up_weights
output = (gate * up) × down_weights
```

This expands the 3584-dimensional vector to ~18,944 dimensions (the "intermediate size"), applies a nonlinear activation (SiLU), and projects back down. The "gating" mechanism lets the model learn which dimensions to activate.

**Why we built a SiLU kernel**: SiLU is the activation function used in every feed-forward layer. `SiLU(x) = x / (1 + exp(-x))`.

### Softmax — turning scores into probabilities

Softmax appears in two places:
1. **Attention scores** — converting raw similarity scores to probabilities
2. **Final output** — converting the 152,000 logits to token probabilities

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

The "trick" is subtracting the maximum value first for numerical stability:
```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

Without this, `exp(100)` would overflow to infinity. After subtracting max, the largest exponent is `exp(0) = 1`.

**Why we built a Softmax kernel**: Every attention head at every layer needs softmax. For a 28-layer model with 28 heads, that's 784 softmax operations per token.

### Matrix Multiplication — the bottleneck

If you look at the whole forward pass, almost every operation is either a matrix multiplication or something simple (add, multiply, normalize). The matmul operations are:

- Q, K, V projections (3 per layer)
- Attention score computation (1 per layer)
- Attention output projection (1 per layer)
- Feed-forward gate, up, down projections (3 per layer)
- Final LM head projection (1 total)

For 28 layers: `28 × 8 + 1 = 225 matrix multiplications per token`. This is why matmul speed determines overall inference speed.

**Why we built both naive and tiled matmul kernels**: The naive version is correct but slow. The tiled version uses **shared memory** (a small, fast cache on the GPU) to reduce the number of times data is fetched from slow global memory. This can improve performance by 10-50x.

---

## 3. Quantization — Fitting Big Models in Small GPUs

### The problem

Qwen3.5-9B has 9 billion parameters. In full precision (FP16 — 16 bits per number), that's:
```
9,000,000,000 × 2 bytes = 18 GB
```

Your RTX 5060 Ti has 8 GB of VRAM. The model literally doesn't fit.

### The solution: quantization

**Quantization** reduces the precision of each weight. Instead of 16 bits per number, we use 4 bits:
```
9,000,000,000 × 0.5 bytes = 4.5 GB
```

Now it fits, with room for the KV-cache and other overhead.

### NF4 — the quantization we use

**NF4** (4-bit NormalFloat) is the quantization format used by bitsandbytes. It works by:

1. Divide weights into groups of 128
2. For each group, find the min and max
3. Map each weight to one of 16 quantization levels (4 bits = 2⁴ = 16 possible values)
4. Store the scale factor and zero point for each group

The "NF" means the 16 levels are spaced according to a normal distribution (bell curve) rather than evenly. This matches how real model weights are distributed, reducing quantization error.

**Double quantization**: We also quantize the scale factors themselves (the "quantization parameters"). This saves additional memory with almost no quality loss.

### What you lose

Quantization is lossy — you're approximating 16-bit numbers with 4-bit numbers. In practice:
- 4-bit models lose ~1-3% quality compared to full precision
- For chat and coding, this is barely noticeable
- For math and reasoning, there can be more degradation

### INT4 for WebGPU

For the WebGPU engine, we'll use **INT4** quantization with **dequantization on the fly** — the weights are stored as 4-bit integers in GPU memory, and each matmul kernel unpacks them to float32 before multiplying. This means:
- Memory: 4 bits per weight (fits larger models)
- Compute: float32 (full precision math)
- The dequantization is **fused** into the matmul kernel (done in the same GPU operation) so it's nearly free

---

## 4. Backend #1: HuggingFace Transformers

### What it is

HuggingFace Transformers is a Python library that provides:
- A universal model loader (load any model from HuggingFace Hub)
- Tokenizers for every model family
- The forward pass implementation (the actual math)
- Integration with PyTorch for GPU acceleration

### How Artifex uses it

```
User prompt → Tokenizer → PyTorch model (on GPU via CUDA) → Token IDs → Detokenize → Response
```

When you use the Transformers backend:
1. `engine_transformers.py` loads the model using `AutoModelForCausalLM.from_pretrained()`
2. bitsandbytes applies NF4 quantization during loading
3. The model runs on your GPU via PyTorch's CUDA backend
4. Generation uses `model.generate()` with streaming via `TextIteratorStreamer`

### Pros and cons

**Pros:**
- Full control over the model
- Supports any HuggingFace model
- Can fine-tune, modify, and experiment
- Access to all PyTorch optimizations

**Cons:**
- Requires downloading full model weights (18 GB for 9B)
- First load is slow (quantization + compilation)
- Only works on NVIDIA GPUs (CUDA)
- Uses more VRAM than Ollama's optimized format

### The NF4 cache

The first time you load a model, bitsandbytes quantizes every weight from FP16 to NF4. This takes 2-5 minutes. Artifex saves the quantized weights to a cache directory (`models/qwen3.5-9b-nf4-cached/`) so subsequent loads are instant. This is the "fast path" in `engine_transformers.py`.

---

## 5. Backend #2: Ollama

### What it is

Ollama is a standalone program that serves AI models through a local HTTP API. Think of it like a database server, but for AI models — it runs in the background and responds to requests.

### Why it exists

The problem Ollama solves: running models via Transformers requires Python, PyTorch, CUDA, bitsandbytes, and careful VRAM management. Ollama wraps all of this into a single binary that:
- Downloads pre-quantized models (already compressed, no quantization step needed)
- Uses **GGUF** format (a highly optimized weight format created by the llama.cpp project)
- Manages GPU memory automatically
- Provides a simple HTTP API

### How it works

Ollama runs as a background service on `localhost:11434`. When Artifex uses the Ollama backend:

```
Artifex → HTTP POST to localhost:11434/api/chat → Ollama → GPU inference → Streamed response
```

The communication is:
1. Artifex sends a JSON payload with messages, temperature, max_tokens
2. Ollama streams back JSON chunks, each containing a piece of the response
3. Artifex assembles the chunks into the full response

### The GGUF format

GGUF (GPT-Generated Unified Format) is a weight format designed for efficient inference. Unlike SafeTensors (which stores raw float16 weights), GGUF stores:
- Pre-quantized weights in various formats (Q4_K_M, Q5_K_S, etc.)
- The tokenizer embedded in the file
- Model metadata (architecture, dimensions, etc.)

This means a single `.gguf` file contains everything needed to run the model — no separate tokenizer files, no config.json, no multiple shards.

### Ollama vs Transformers

| | Transformers | Ollama |
|---|---|---|
| Setup complexity | High (Python, PyTorch, CUDA) | Low (single binary) |
| Model format | SafeTensors (raw weights) | GGUF (pre-quantized) |
| GPU support | NVIDIA only (CUDA) | NVIDIA, AMD, Apple Silicon |
| First load | Slow (quantization needed) | Fast (pre-quantized) |
| Flexibility | Full (modify anything) | Limited (black box) |
| VRAM efficiency | Good with NF4 | Very good (optimized kernels) |
| Model selection | Any HuggingFace model | Ollama library models only |

### Why "13 layers on GPU"?

When you see `Ollama ready — model: qwen3.5:9b (GPU: 13 layers)`, that means Ollama split the model: 13 of the 28 layers run on the GPU, and the remaining 15 run on the CPU. This happens because 8 GB VRAM isn't enough for all layers. Ollama automatically decides how to split based on available memory.

This is called **CPU offloading** — layers that don't fit on the GPU are computed on the CPU instead. It's slower but allows running models that wouldn't otherwise fit.

---

## 6. Backend #3: The API Server (OpenAI-Compatible)

### What is an API?

API stands for **Application Programming Interface**. It's a way for programs to talk to each other. A **REST API** uses standard HTTP requests (the same protocol your browser uses to load web pages) to send and receive data.

### Why "OpenAI-compatible"?

OpenAI defined a REST API format for their GPT models. Because it became the industry standard, many tools (coding assistants, chatbots, apps) are built to work with this exact format. By making Artifex's API match OpenAI's format, any tool that works with OpenAI also works with Artifex — just change the URL from `api.openai.com` to `localhost:8000`.

### How REST APIs work

A REST API has **endpoints** — URLs that accept specific types of requests:

```
GET  /health                    → Returns system status
GET  /v1/models                 → Returns list of available models
POST /v1/chat/completions       → Send messages, get AI response
POST /v1/images/generations     → Send a prompt, get an image
POST /v1/embeddings             → Send text, get vector representation
```

**GET** requests retrieve information. **POST** requests send data and get a response.

### The chat completion flow

When a client sends a POST to `/v1/chat/completions`:

```json
{
  "model": "qwen3.5:9b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "max_tokens": 200,
  "temperature": 0.7
}
```

The server:
1. Validates the request (auth check, input validation)
2. Gets or creates the inference engine
3. Passes the messages to the engine
4. Waits for the response
5. Returns it in OpenAI's format:

```json
{
  "id": "chatcmpl-abc123",
  "choices": [{
    "message": {"role": "assistant", "content": "The capital of France is Paris."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 20, "completion_tokens": 8}
}
```

### Streaming

For long responses, waiting for the entire generation is slow. **Streaming** sends each token as it's generated using **Server-Sent Events (SSE)** — a standard web protocol where the server pushes data to the client as it becomes available:

```
data: {"choices":[{"delta":{"content":"The"}}]}
data: {"choices":[{"delta":{"content":" capital"}}]}
data: {"choices":[{"delta":{"content":" of"}}]}
...
data: [DONE]
```

### Swagger / OpenAPI docs

FastAPI automatically generates interactive documentation at `/docs`. This is **Swagger UI** — a web page where you can:
- See every endpoint
- Read what parameters they accept
- Try them out with real requests
- See the response format

The documentation is generated from **Pydantic models** — Python classes that define the expected request/response structure. When we added `ChatCompletionRequest` as a Pydantic model, Swagger knew exactly what fields to show.

### Authentication

The API supports optional key-based auth:
- Set `ARTIFEX_API_KEY=your-secret` environment variable
- Clients must send the key in the `Authorization: Bearer your-secret` header
- We use `hmac.compare_digest()` for comparison — this takes the same amount of time regardless of where the key differs, preventing **timing attacks** (where an attacker measures response time to guess the key character by character)

### CORS — Cross-Origin Resource Security

When a web page on `localhost:5173` (our WebGPU UI) tries to call an API on `localhost:8000`, the browser blocks it by default. This is a security feature called **Same-Origin Policy** — it prevents malicious websites from making requests to your local services.

**CORS** (Cross-Origin Resource Sharing) headers tell the browser which origins are allowed. We restrict it to localhost only:
```python
allow_origins=["http://localhost", "http://127.0.0.1", ...]
```

If we used `allow_origins=["*"]` (allow everything), any website you visit could secretly talk to your Artifex API server.

---

## 7. Backend #4: WebGPU — Running Models in the Browser

### What is WebGPU?

WebGPU is a new browser API (like WebGL, but for general computation). It gives JavaScript code direct access to the GPU for arbitrary math — not just graphics. It's supported in Chrome 113+, Edge, and Safari 18+.

### Why run AI in the browser?

| Traditional (Python/CUDA) | WebGPU (Browser) |
|---|---|
| Requires Python, PyTorch, CUDA | Just open a web page |
| NVIDIA GPUs only | Any GPU (NVIDIA, AMD, Intel, Apple) |
| Complex setup | Zero installation |
| Runs as a local service | Runs in a tab |
| Uses CUDA cores | Uses the same GPU through WebGPU |

The vision: you open a URL, it loads the model into your GPU through the browser, and you're chatting with a local AI. No Python. No installation. Works on any device with a GPU.

### How WebGPU compute works

WebGPU compute is fundamentally different from CPU programming. Here's the mental model:

**CPU approach** (sequential):
```
for each element in array:
    result[i] = input[i] * 2
```
One element at a time. 1 billion elements = 1 billion sequential operations.

**GPU approach** (parallel):
```
Launch 1 billion threads simultaneously.
Each thread: result[my_id] = input[my_id] * 2
```
All elements processed at once. In practice, the GPU doesn't have 1 billion cores, so it processes them in waves — but a GPU with 4,608 cores processes 4,608 elements per wave, finishing ~217,000x faster than a single CPU core.

### The WebGPU programming model

```
1. Create BUFFERS — memory on the GPU
   (input data, output data, parameters)

2. Write a SHADER — a small program in WGSL
   (the math each thread executes)

3. Create a PIPELINE — compiles the shader for the GPU
   (like compiling C code to machine code)

4. DISPATCH — launch thousands of threads
   (tell the GPU to run the shader on the data)

5. READ BACK — copy results from GPU to CPU
   (get the answer back into JavaScript)
```

### Buffers — GPU memory

The GPU has its own memory (VRAM), separate from system RAM. Data must be explicitly copied:

```
CPU RAM                    GPU VRAM
┌─────────┐               ┌─────────┐
│ input[] ─┼── upload ──→ │ input[] │
│          │               │ output[]│
│ result[] ←┼── readback ──┤         │
└─────────┘               └─────────┘
```

Buffer types in our code:
- **Storage buffers** — large, for model weights and activations (up to 2 GB each)
- **Uniform buffers** — small, for parameters (hidden size, epsilon, dimensions)

### Workgroups and invocations

When you dispatch a compute shader, you specify how many **workgroups** to launch, and each workgroup has a fixed number of **invocations** (threads).

```
@compute @workgroup_size(256)    ← 256 threads per workgroup
fn main(@builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
    // lid.x = 0..255 (which thread am I within my workgroup?)
    // wid.x = which workgroup am I?
}
```

If you dispatch 4 workgroups of 256 threads, you get 1,024 total threads. Each thread knows its own ID and can work on a different piece of data.

### Shared memory (workgroup memory)

Each workgroup has a small amount of **shared memory** — fast memory that all 256 threads in the workgroup can read/write. This is much faster than global GPU memory (storage buffers).

```
var<workgroup> shmem: array<f32, 256>;    ← shared within the workgroup
```

We use this for **reductions** — operations where many threads need to combine their results (like finding the maximum value or computing a sum). The pattern:

```
1. Each thread computes a partial result
2. Write to shared memory
3. workgroupBarrier()     ← wait for all threads to finish writing
4. Half the threads combine pairs
5. workgroupBarrier()
6. Quarter of the threads combine pairs
7. ... repeat until one thread has the final answer
```

This is called a **parallel reduction** and it's why `workgroupBarrier()` is critical — without it, threads might read values that other threads haven't written yet.

### The `shared` keyword bug

We named our shared memory variable `shared`, but `shared` is a **reserved keyword** in WGSL (the WebGPU shading language). The shaders failed to compile — the GPU returned all zeros because no computation ever ran. Renaming to `shmem` fixed it. The error was invisible in the test results (they showed "FAIL" not "COMPILE ERROR") because WebGPU doesn't throw exceptions for invalid shaders — it silently produces an invalid pipeline that returns garbage.

---

## 8. The GPU Compute Pipeline — Why Kernels Matter

### What is a "kernel"?

In GPU computing, a **kernel** is a function that runs on the GPU. It's called a "kernel" because it's the core (kernel) of the computation — the inner loop that processes data in parallel. Each of our WGSL shaders (`matmul.wgsl`, `softmax.wgsl`, etc.) contains one or more kernels.

### Why build our own kernels?

The transformer forward pass is made of ~8 operations repeated per layer. If we implement these 8 operations as efficient GPU kernels, we can run the entire model. The operations are:

1. **Matrix multiplication** (matmul) — for every linear layer (Q, K, V projections, FFN)
2. **RMSNorm** — normalization at each layer
3. **RoPE** — position encoding for attention
4. **Softmax** — converting attention scores to probabilities
5. **SiLU** — activation function in the feed-forward network
6. **Element-wise add** — residual connections
7. **Element-wise multiply** — gating in SwiGLU
8. **Embedding lookup** — converting token IDs to vectors

We've built kernels 1-7. Together with a weight loader and tokenizer, these are sufficient to run the complete Qwen3.5 forward pass in the browser.

### Why not use an existing library?

Libraries like WebLLM (by MLC) compile models using TVM (a compiler framework). This works but:
- Each model must be pre-compiled for WebGPU
- You can't load arbitrary HuggingFace models
- The compilation process is complex and opaque
- You're locked into their model format

By building our own kernels, we can load any SafeTensors model directly from HuggingFace and run it. This is what TensorBend (the project in those screenshots) does — pure WGSL kernels, no compilation step.

### Performance: CPU vs GPU

For a matrix multiplication of two 1024×1024 matrices:

| Platform | Operations | Time | Why |
|---|---|---|---|
| CPU (single core) | 1 billion multiplies, sequential | ~2 seconds | One at a time |
| CPU (16 cores) | 1 billion multiplies, 16 parallel | ~125ms | 16x faster |
| GPU (4,608 cores) | 1 billion multiplies, 4,608 parallel | ~0.5ms | 4,608x faster |
| GPU + shared memory | Same, but with tiled access | ~0.05ms | 10x faster due to memory efficiency |

The GPU isn't just "faster" — it's a fundamentally different architecture designed for exactly this type of parallel math.

---

## 9. Our WGSL Kernels Explained

### Kernel 1: Element-wise operations (`elementwise.wgsl`)

**What it does**: Applies a simple function to each element independently.

**Operations**:
- `add(a, b)` → `a[i] + b[i]`
- `mul(a, b)` → `a[i] * b[i]`
- `silu(x)` → `x[i] / (1 + exp(-x[i]))`

**Why it's parallel**: Each element is independent — thread 0 processes element 0, thread 1 processes element 1, etc. No thread needs to know what any other thread is doing.

**Used for**: Residual connections (add), SwiGLU gating (multiply), feed-forward activation (SiLU).

### Kernel 2: Matrix multiplication (`matmul.wgsl`)

**What it does**: Multiplies two matrices. `C = A × B`

**Naive version**: Each thread computes one element of the output:
```
C[row][col] = sum(A[row][k] * B[k][col]) for all k
```
Thread (row, col) reads an entire row of A and column of B from global memory.

**Tiled version**: Instead of each thread reading from slow global memory, a 16×16 tile of threads cooperatively loads chunks of A and B into fast shared memory, then computes from there:

```
1. Threads 0-255 collectively load a 16×16 tile of A into shared memory
2. Threads 0-255 collectively load a 16×16 tile of B into shared memory
3. workgroupBarrier()  ← everyone wait
4. Each thread computes a partial sum using the shared data
5. Move to the next tile, repeat
```

This reduces global memory reads by ~16x, which is why the tiled version is dramatically faster.

**Used for**: Every linear layer in the transformer (225 times per token).

### Kernel 3: Softmax (`softmax.wgsl`)

**What it does**: Converts a row of numbers to probabilities that sum to 1.

**The three passes**:
1. **Find max** — parallel reduction across the row (needed for numerical stability)
2. **Compute exp(x - max) and sum** — another parallel reduction
3. **Normalize** — divide each value by the sum

**Why three passes**: Softmax requires two global values (max and sum) before it can produce any output. Each global value requires a **parallel reduction** — where 256 threads combine their results down to a single number. This requires multiple rounds of synchronization (`workgroupBarrier()`).

**Used for**: Attention score normalization (784 times per token for a 28-layer, 28-head model).

### Kernel 4: RMSNorm (`rmsnorm.wgsl`)

**What it does**: Normalizes a vector by its root-mean-square, then scales by a learned weight.

```
rms = sqrt(mean(x²) + epsilon)
output[i] = (x[i] / rms) * weight[i]
```

**The parallel reduction**: Computing `mean(x²)` requires summing all squared values — this is a reduction across 3,584 elements (the hidden size). Same pattern as softmax: each thread computes a partial sum, then they combine via shared memory.

**Used for**: Normalization at every layer (56 times per token).

### Kernel 5: RoPE (`rope.wgsl`)

**What it does**: Rotates pairs of values in the Q and K vectors based on their position in the sequence.

```
For each pair (x[2i], x[2i+1]):
    cos_val = cos(position * frequency[i])
    sin_val = sin(position * frequency[i])
    output[2i]   = x[2i] * cos_val - x[2i+1] * sin_val
    output[2i+1] = x[2i] * sin_val + x[2i+1] * cos_val
```

This is a 2D rotation matrix applied to each pair. The frequency decreases for higher dimensions (lower dimensions rotate fast, encoding fine position differences; higher dimensions rotate slowly, encoding broad position).

**Used for**: Position encoding in attention (applied to Q and K at every layer).

---

## 10. The WebGPU Inference Roadmap

### What we've built (Phase 0-1)

```
✅ GPU device detection and capability reporting
✅ Buffer management (create, read, write GPU buffers)
✅ Shader compilation pipeline
✅ 7 compute kernels (all passing tests):
    ✅ SiLU, Add, Multiply (elementwise)
    ✅ Matmul naive, Matmul tiled
    ✅ Softmax
    ✅ RMSNorm
✅ RoPE shader (exists, needs test wiring)
✅ Metrics collection (browser → dev server)
✅ Chat UI shell
```

### What we need to build (Phase 2-6)

**Phase 2: Weight Loader** — Load SafeTensors files from HuggingFace directly into GPU buffers. SafeTensors is a simple format: an 8-byte header length, then a JSON header describing each tensor's name/shape/dtype/offset, then raw binary data. We'll use HTTP range requests to stream shards without downloading the full model.

**Phase 3: Tokenizer** — Convert text to token IDs and back. We'll either implement BPE (Byte Pair Encoding) in JavaScript or use the @huggingface/transformers.js tokenizer as a dependency.

**Phase 4: Forward Pass** — Wire all our kernels together to implement the Qwen3.5 transformer:
```
embed → [rmsnorm → attention → add → rmsnorm → ffn → add] × 28 → rmsnorm → lm_head
```

**Phase 5: INT4 Quantization** — Store weights as 4-bit integers, dequantize on the fly during matmul. This is essential for fitting larger models in the WebGPU 2 GB buffer limit.

**Phase 6: Generation Loop** — Implement the autoregressive decode loop: generate one token, append to KV-cache, repeat. Add sampling (temperature, top-k, top-p).

### The 2 GB buffer limit

WebGPU limits each individual buffer to ~2 GB. This means a 9B model can't be loaded into a single buffer. The solution: split weights across multiple buffers (one per layer, or one per weight matrix). The 2 GB limit is per-buffer, not total — you can allocate many buffers up to the GPU's total VRAM.

---

## 11. How All the Pieces Connect

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER                                      │
│                                                                  │
│  Types a message in one of three interfaces:                     │
│    CLI (terminal)  │  GUI (desktop window)  │  Browser (WebGPU)  │
└────────┬───────────┴──────────┬─────────────┴────────┬──────────┘
         │                      │                       │
         ▼                      ▼                       ▼
┌─────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│  cli_assistant   │  │   cyber_gui      │  │   WebGPU main.ts     │
│  (Python)        │  │   (Python)       │  │   (TypeScript)       │
│                  │  │                  │  │                      │
│  Extracts tools  │  │  Same engine     │  │  Currently calls     │
│  from AI output  │  │  via GUI         │  │  /v1/chat/completions│
│  and executes    │  │  controls        │  │  via HTTP proxy      │
└────────┬─────────┘  └────────┬─────────┘  │                      │
         │                      │            │  FUTURE: runs model  │
         ▼                      ▼            │  directly on GPU     │
┌──────────────────────────────────────┐     │  via WGSL kernels    │
│         Engine Layer                  │     └──────────┬───────────┘
│                                      │                 │
│  engine_factory.py                   │                 │
│    ├── TransformersEngine            │                 │
│    │     Uses: PyTorch + CUDA        │                 ▼
│    │     Format: SafeTensors + NF4   │     ┌──────────────────────┐
│    │     GPU: NVIDIA only            │     │  WebGPU Engine       │
│    │                                 │     │                      │
│    └── OllamaEngine                  │     │  Uses: WGSL shaders  │
│          Uses: HTTP to localhost     │     │  Format: SafeTensors  │
│          Format: GGUF                │     │  GPU: Any vendor     │
│          GPU: Any (via Ollama)       │     │                      │
│                                      │     │  matmul, softmax,    │
│  Both engines implement:             │     │  rmsnorm, rope, silu │
│    .load()                           │     │  → forward pass      │
│    .generate_streaming()             │     │  → token generation  │
│    .is_loaded()                      │     └──────────────────────┘
│    .unload()                         │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│         API Server (optional)        │
│                                      │
│  FastAPI on localhost:8000           │
│  OpenAI-compatible endpoints         │
│  Uses same engine layer above        │
│                                      │
│  Any OpenAI-compatible client can    │
│  connect (Cursor, Continue, etc.)    │
└──────────────────────────────────────┘
```

### The key insight

All four interfaces (CLI, GUI, API, WebGPU) are just different ways to call the same underlying operation: **send tokens to a model, get tokens back**. The intelligence is in the model weights. The code is plumbing.

- **Transformers** = load weights into GPU via Python/CUDA, run math in CUDA kernels
- **Ollama** = let Ollama handle the weights and CUDA, talk to it via HTTP
- **API Server** = wrap the engine layer in HTTP endpoints so other programs can use it
- **WebGPU** = load weights into GPU via JavaScript/WebGPU, run math in WGSL kernels

The WebGPU path is the most ambitious because we're building the inference engine from scratch — but it's also the most universal because it works on any GPU, in any browser, with zero installation.

---

## 12. Glossary

| Term | Definition |
|------|-----------|
| **Activation function** | A nonlinear function applied after linear layers. Without it, stacking linear layers would just be one big linear layer. SiLU, ReLU, and GELU are common choices. |
| **Attention** | The mechanism that lets each token "look at" other tokens to understand context. The core of the transformer architecture. |
| **BPE** | Byte Pair Encoding — the tokenization algorithm used by Qwen and most modern models. Builds a vocabulary by repeatedly merging the most common character pairs. |
| **Buffer** | A chunk of memory on the GPU. Data must be explicitly copied between CPU and GPU buffers. |
| **Causal mask** | A restriction that prevents tokens from attending to future positions. Enforced by setting future attention scores to -infinity. |
| **CORS** | Cross-Origin Resource Sharing — browser security headers that control which websites can call your API. |
| **CUDA** | NVIDIA's GPU computing platform. Only works on NVIDIA GPUs. The dominant platform for AI. |
| **Decode** | The token-by-token generation phase where the model produces one new token per forward pass. |
| **Dispatch** | The command that tells the GPU to launch threads and run a kernel. |
| **Embedding** | Converting a token ID (integer) to a dense vector (list of floats). The embedding table is the model's "vocabulary" of learned word meanings. |
| **Endpoint** | A specific URL path in an API that handles a particular type of request. |
| **FP16** | 16-bit floating point — the standard precision for model weights. 2 bytes per number. |
| **Forward pass** | Running input data through the model from beginning to end to produce an output. |
| **GQA** | Grouped-Query Attention — an optimization where multiple Q heads share K/V heads, reducing memory usage. |
| **GGUF** | The weight format used by llama.cpp and Ollama. Self-contained (includes tokenizer and metadata). |
| **INT4** | 4-bit integer quantization — stores weights as 4-bit numbers, reducing memory by 4x vs FP16. |
| **Kernel** | A function that runs on the GPU in parallel across thousands of threads. |
| **KV-cache** | Storage for previously computed Key and Value vectors in attention. Grows with conversation length. This is why long conversations use more VRAM. |
| **Logits** | Raw output scores from the model (before softmax). One score per vocabulary token (~152,000). |
| **LM Head** | The final linear layer that projects from hidden dimension to vocabulary size. |
| **NF4** | NormalFloat 4-bit — a quantization format that maps weights to 16 levels distributed on a normal curve. |
| **Parallel reduction** | A pattern where many threads combine their results into a single value (sum, max, etc.) using shared memory and barriers. |
| **Prefill** | The initial phase where the model processes all prompt tokens in parallel (faster than decode because no sequential dependency). |
| **Pydantic** | A Python library for data validation. FastAPI uses it to define request/response schemas and auto-generate API documentation. |
| **Quantization** | Reducing the precision of model weights (e.g., FP16 → INT4) to save memory at the cost of small quality loss. |
| **Residual connection** | Adding the input of a layer to its output (`output = layer(x) + x`). Helps information flow through deep networks. |
| **REST** | Representational State Transfer — an architecture for web APIs using HTTP methods (GET, POST, etc.). |
| **RMSNorm** | Root Mean Square Normalization — scales values so their RMS (root mean square) equals 1. Simpler than LayerNorm. |
| **RoPE** | Rotary Position Embeddings — encodes token position by rotating Q/K vectors. Allows the model to understand word order. |
| **SafeTensors** | A weight file format by HuggingFace. Simple, safe (no arbitrary code execution), and fast to load. |
| **Sampling** | Choosing the next token from the probability distribution. Temperature, top-k, and top-p control the randomness. |
| **Shader** | A program that runs on the GPU. In WebGPU, shaders are written in WGSL. |
| **Shared memory** | Fast, small memory shared among threads in a workgroup. ~100x faster than global memory. |
| **Softmax** | Converts a vector of numbers to probabilities that sum to 1. `softmax(x_i) = exp(x_i) / sum(exp(x_j))`. |
| **SSE** | Server-Sent Events — a protocol where the server pushes updates to the client (used for streaming chat). |
| **SwiGLU** | Swish-Gated Linear Unit — the feed-forward network variant used by Qwen. Uses gating (element-wise multiply) for better training. |
| **Swagger** | Interactive API documentation UI. Shows endpoints, lets you test them from the browser. |
| **Temperature** | Controls randomness in sampling. 0 = always pick the most likely token. 1 = sample proportionally. 2 = very random. |
| **Thinking blocks** | Qwen3.5 generates internal reasoning inside `<think>...</think>` tags before producing the visible response. |
| **Token** | A chunk of text (word, subword, or character) that the model processes. Models work with token IDs, not raw text. |
| **Top-k** | Only consider the k most likely tokens when sampling (e.g., top-k=50 means ignore all but the 50 most probable). |
| **Top-p** | Only consider tokens whose cumulative probability exceeds p (e.g., top-p=0.9 means keep adding tokens until their probabilities sum to 90%). |
| **VRAM** | Video RAM — the GPU's dedicated memory. Model weights, KV-cache, and intermediate computations all live here. |
| **WGSL** | WebGPU Shading Language — the programming language for WebGPU shaders. Similar to Rust syntax. |
| **Workgroup** | A group of threads that execute together and can share memory. Our kernels use 256 threads per workgroup. |
| **workgroupBarrier()** | A synchronization point — all threads in the workgroup must reach this line before any can proceed. Essential for correct shared memory operations. |

---

*"Unless the LORD builds the house, the builders labor in vain." — Psalm 127:1*
