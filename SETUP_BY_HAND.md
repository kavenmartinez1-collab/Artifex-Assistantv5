# Building `llama-server` by Hand for Artifex Assistant V5

A single, self-contained walkthrough for compiling the **mainline llama.cpp** `llama-server` binary that Artifex's `llama_cpp` backend drives. Every step is concrete; every command has been exercised on Windows 11 with an RTX 4090 and on smaller cards.

This guide covers **only the compile pipeline** — installing prerequisites, cloning llama.cpp, configuring CMake for your GPU, building the `llama-server` target, verifying the binary, and pointing Artifex's `llama_cpp_config.json` at the result. For the *runtime* side of `llama_cpp_config.json` (the `extra_flags` reference, model-specific tuning, KV-cache quantization tradeoffs, speculative decoding, vision wiring) see the **"Setting Up llama.cpp Backend"** section of `README.md`.

What this guide deliberately does **not** cover: environment variables, API auth, the web gateway, the sandbox, the Docker stack, model download scripting beyond the bare minimum, or anything in `api/`, `core/`, or `setup_wizard.py`. Those have their own homes.

---

## Table of contents

0. [Mental model — what you're building](#0-mental-model)
1. [Prerequisites (install these first)](#1-prerequisites-install-these-first)
2. [Open the right terminal (Windows)](#2-open-the-right-terminal-windows)
3. [Clone llama.cpp outside the Artifex repo](#3-clone-llamacpp-outside-the-artifex-repo)
4. [Pick a commit (latest vs pinned)](#4-pick-a-commit-latest-vs-pinned)
5. [Configure with CMake (CUDA on)](#5-configure-with-cmake-cuda-on)
6. [Build the `llama-server` target](#6-build-the-llama-server-target)
7. [Verify the binary](#7-verify-the-binary)
8. [Quick smoke run (optional but recommended)](#8-quick-smoke-run-optional-but-recommended)
9. [Wire the binary into Artifex](#9-wire-the-binary-into-artifex)
10. [Per-GPU substitution table](#10-per-gpu-substitution-table)
11. [Build-time troubleshooting](#11-build-time-troubleshooting)
12. [Linux and macOS notes](#12-linux-and-macos-notes)
13. [Quick checklist](#13-quick-checklist)

---

## 0. Mental model

Artifex's `core/engine_llama_cpp.py` does not embed llama.cpp. It **spawns** an external `llama-server.exe` process per model, talks to it over HTTP on a localhost port, and tears it down on unload. Everything in this guide produces that one binary.

```
   Artifex (Python)                         llama-server.exe
   ┌──────────────────┐    subprocess     ┌────────────────────┐
   │ engine_llama_cpp │ ────────────────▶ │ HTTP on 127.0.0.1  │
   │ load() / unload()│ ◀──────────────── │ /v1/chat/completions│
   └──────────────────┘    SSE / JSON     └────────────────────┘
            │                                       │
            └── reads llama_cpp_config.json          └── reads the GGUF
                (server_path + per-model flags)         from disk
```

You can build `llama-server` once and reuse the same binary for any GGUF you point Artifex at. You only need to rebuild when you want a newer llama.cpp commit (new features, bug fixes, CVE patches).

This guide assumes **mainline llama.cpp** (`https://github.com/ggml-org/llama.cpp`). Forks (TurboQuant, ik_llama, etc.) have their own quirks; see the README's "Path B" note for the deprecated TQ3 fork story.

---

## 1. Prerequisites (install these first)

Every item below must be installed and reachable on `PATH` before you touch llama.cpp. Each step has a one-line verification command. If any of them fails, fix it before continuing — diagnosing a CMake error caused by a missing prerequisite is a waste of an afternoon.

### 1.1 NVIDIA driver

- Install the latest **Game Ready** or **Studio** driver: <https://www.nvidia.com/Download/index.aspx>.
- Verify:
  ```powershell
  nvidia-smi
  ```
  Output should list your GPU(s), driver version, and a CUDA runtime version (e.g. `CUDA Version: 12.8`). The runtime version shown here is the *driver's* CUDA — it does **not** install `nvcc`. You still need the CUDA Toolkit (step 1.5).

### 1.2 Python 3.11.x

- Download from <https://www.python.org/downloads/> (Python 3.11; 3.12 may work but Artifex's pins are tested on 3.11).
- During install, check **"Add python.exe to PATH"**.
- Verify:
  ```powershell
  python --version
  # Python 3.11.x
  ```

Python isn't strictly required to build `llama-server`, but you need it for Artifex itself and for the HuggingFace CLI used to download GGUFs.

### 1.3 Git

- Install Git for Windows: <https://git-scm.com/>.
- Verify:
  ```powershell
  git --version
  ```

### 1.4 Visual Studio 2022 Build Tools (or full VS 2022)

You need the MSVC C++ toolchain. The lightweight option is **Build Tools** (no IDE):

- Download "Build Tools for Visual Studio 2022": <https://visualstudio.microsoft.com/downloads/>.
- In the installer, select the **"Desktop development with C++"** workload. Make sure these individual components are checked:
  - **MSVC v143 — VS 2022 C++ x64/x86 build tools**
  - **Windows 11 SDK** (or the latest 10 SDK)
  - **C++ CMake tools for Windows**
- After install, you get a Start menu shortcut named **"x64 Native Tools Command Prompt for VS 2022"**. You will build llama.cpp from this prompt (see section 2).
- Verify: open that prompt and run `cl` (no args). You should see the Microsoft optimizing compiler banner. If `cl` is "not recognized," the developer-environment script didn't run — re-launch from the Start menu shortcut.

### 1.5 CUDA Toolkit

Match the toolkit to your card:

| GPU family    | Cards                                  | CUDA Toolkit       |
|---------------|----------------------------------------|--------------------|
| Turing        | RTX 20xx, GTX 16xx                     | 12.4               |
| Ampere        | RTX 30xx (3060 / 3070 / 3080 / 3090)   | 12.4               |
| Ada Lovelace  | RTX 40xx (4060 / 4070 / 4080 / 4090)   | 12.4               |
| Hopper        | H100 / H200                            | 12.4 or 12.8       |
| **Blackwell** | **RTX 50xx (5060 / 5070 / 5080 / 5090)** | **12.8 (minimum)** |

- 12.4 download: <https://developer.nvidia.com/cuda-12-4-1-download-archive>
- 12.8 download: <https://developer.nvidia.com/cuda-12-8-0-download-archive>
- Pick **Windows / x86_64 / 11 / exe (local)**. The "express" installer is fine.
- After install, open a fresh terminal and verify:
  ```powershell
  nvcc --version
  # Cuda compilation tools, release 12.4 (or 12.8), V12.x.xxx
  ```
- If `nvcc` isn't on PATH, add the toolkit's `bin` directory to your user PATH and reopen the terminal:
  - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin` *or*
  - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin`

**Why Blackwell needs 12.8.** The RTX 50xx series uses compute capability `sm_120`, which first lands in CUDA 12.8. Building llama.cpp with CUDA 12.4 on a 50xx card produces a binary that errors at runtime with `CUDA error: no kernel image is available for execution on the device`. 12.8 is a correctness requirement, not a perf hint.

### 1.6 CMake 3.20+

The VS 2022 C++ workload already includes CMake. If you'd rather have a standalone copy (recommended for version control across machines):

- Download from <https://cmake.org/download/> (Windows x64 Installer).
- During install, choose **"Add CMake to the system PATH for all users."**
- Verify:
  ```powershell
  cmake --version
  ```
  Should print >= 3.20.

### 1.7 (Optional) HuggingFace CLI — for pulling GGUFs

You don't need this to *build* `llama-server`, but you'll want it once you want to run the binary against a real model:
```powershell
pip install -U "huggingface_hub[cli]"
huggingface-cli --version
```

---

## 2. Open the right terminal (Windows)

Open **"x64 Native Tools Command Prompt for VS 2022"** from the Start menu — *not* a regular PowerShell or `cmd` window. This shortcut runs `vcvarsall.bat x64` automatically, putting `cl.exe`, `link.exe`, the Windows SDK headers, and the MSVC runtime on PATH. CMake's MSVC generator will fail outside this prompt.

Sanity-check the environment is correct:
```cmd
cl 2>&1 | findstr /C:"Microsoft (R) C/C++"
nvcc --version
cmake --version
git --version
```
All four should respond. If `cl` is not recognized, you opened the wrong terminal — go back to the Start menu and re-launch.

> **PowerShell vs cmd.** If you prefer PowerShell, you can launch a developer PowerShell instead: in the Start menu, search for **"Developer PowerShell for VS 2022"**. It runs `Enter-VsDevShell` so `cl` / `link` are available. The build commands in this guide are written for the cmd-flavored Developer Prompt, but they all work in Developer PowerShell with minor quoting changes.

---

## 3. Clone llama.cpp outside the Artifex repo

Pick any directory you like — just **not inside** the Artifex repo. You don't want llama.cpp's source tree to pollute `git status` for Artifex.

A clean convention is `<your-user-home>\AI\llama-mainline\`:

```cmd
mkdir C:\Users\%USERNAME%\AI\llama-mainline
cd C:\Users\%USERNAME%\AI\llama-mainline
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

After this, the source is at `C:\Users\<you>\AI\llama-mainline\llama.cpp\`, and you're inside that directory ready to configure CMake.

---

## 4. Pick a commit (latest vs pinned)

You have two reasonable choices.

### Choice A — track mainline `master`

Just stay on `master`. Mainline gets steady improvements (faster kernels, new model families, CVE fixes); you re-pull and rebuild whenever you want them.

```cmd
git checkout master
git pull
git log -1 --format=%h
```

Be aware that `master` is a moving target. If a build that worked yesterday fails today, `git log --oneline` will tell you what changed. Reverting to the last commit you successfully built is always an option — see Choice B for how.

### Choice B — pin to a known-good commit

If you want byte-for-byte parity with a build that's known to work for your model + Artifex's resilience layer, check out a specific commit instead of tracking `master`:

```cmd
git fetch --all
git checkout <commit-sha-or-tag>
```

When pinning, note the commit hash and the build flags you used — that's enough to reproduce the binary on another machine.

> **Don't substitute a fork.** Artifex's `core/engine_llama_cpp.py` (resilience layer, slot-state assumptions, SSE parsing) is calibrated against mainline behavior. Forks like TurboQuant TQ3 or ik_llama can drift in subtle ways — they're useful for research, not for the supported text path. (The TQ3 fork was used historically for sub-4-BPW quants but has been deprecated; see `README.md` for the story.)

---

## 5. Configure with CMake (CUDA on)

The CUDA architecture flag has to match your card. Cross-reference:

| GPU family       | Cards                                | `CMAKE_CUDA_ARCHITECTURES` |
|------------------|--------------------------------------|----------------------------|
| Turing           | RTX 20xx, GTX 16xx                   | `75`                       |
| Ampere           | RTX 30xx (3060 / 3070 / 3080 / 3090) | `86`                       |
| Ada Lovelace     | RTX 40xx (4060 / 4070 / 4080 / 4090) | `89`                       |
| Hopper           | H100 / H200                          | `90`                       |
| Blackwell        | RTX 50xx (5060 / 5070 / 5080 / 5090) | `120`                      |

For an RTX 4090:
```cmd
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
```

For a 3060 / 3090, use `86`. For a 5060 Ti / 5090, use `120`. See section 10 for the full per-GPU table.

**Omitting the flag.** If you drop `-DCMAKE_CUDA_ARCHITECTURES`, CMake builds a generic multi-arch binary. That works on Ampere / Ada but takes longer to compile and produces a slightly larger binary. On **Blackwell** the omission is *not* harmless — without `120` in the arch list you'll see `CUDA error: no kernel image is available` at runtime even though the build succeeds.

**Expected output.** The configure step takes 30–60 seconds and ends with:
```
-- Build files have been written to: .../llama.cpp/build
```

**If CMake reports `CUDA not found`:**
- `nvcc --version` doesn't respond → install the CUDA Toolkit (step 1.5).
- `nvcc --version` works in a regular terminal but not the dev prompt → reopen the **x64 Native Tools Command Prompt for VS 2022** fresh; it should inherit the CUDA `bin` directory from your user PATH.
- `nvcc` works but CMake still misses it → pass `-DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe"` (adjust version) on the configure command.

---

## 6. Build the `llama-server` target

You only need `llama-server` for Artifex. Building it alone is meaningfully faster than building everything.

```cmd
cmake --build build --config Release --target llama-server -j
```

`-j` uses all available cores. On a modern desktop expect **10–25 minutes**; the CUDA kernels are the slow part. Be patient — there is no progress bar, just a stream of compilation lines.

Successful output ends with something like:
```
   Creating library .../build/bin/Release/llama-server.lib
   ...
   Generating Code...
llama-server.vcxproj -> .../build/bin/Release/llama-server.exe
```

If the build fails partway, the error is almost always either:
- a missing prerequisite (re-check section 1), or
- antivirus interference on the final link step (see section 11.1).

---

## 7. Verify the binary

The output path depends on the CMake generator. Both shapes are possible:

- **Visual Studio generator** (the default in the Developer Command Prompt):
  `build\bin\Release\llama-server.exe`
- **Ninja or Unix Makefiles**:
  `build\bin\llama-server.exe`

Note **which path applies on your machine** — you'll paste it into `llama_cpp_config.json` (section 9).

Sanity-check the binary:
```cmd
"C:\Users\%USERNAME%\AI\llama-mainline\llama.cpp\build\bin\Release\llama-server.exe" --version
```
Expected output looks like:
```
version: <build-number> (<commit-sha>)
built with MSVC <version> for x64-pc-windows-msvc
```

If `--version` errors out with a missing DLL, the most common cause is the CUDA runtime not being on PATH — make sure `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin` (or `v12.8\bin`) is in your user PATH and reopen the terminal.

---

## 8. Quick smoke run (optional but recommended)

Before wiring the binary into Artifex, prove it can serve a GGUF directly. If you haven't downloaded a GGUF yet, do that first:

```cmd
huggingface-cli download bartowski/Qwen3.5-4B-Instruct-GGUF Qwen3.5-4B-Instruct-Q4_K_M.gguf ^
  --local-dir "C:\Users\%USERNAME%\AI\models\Qwen3.5-4B-GGUF"
```

Then launch the server directly:
```cmd
"C:\Users\%USERNAME%\AI\llama-mainline\llama.cpp\build\bin\Release\llama-server.exe" ^
  -m "C:\Users\%USERNAME%\AI\models\Qwen3.5-4B-GGUF\Qwen3.5-4B-Instruct-Q4_K_M.gguf" ^
  --port 8081 -ngl 99 -c 8192
```

The first launch streams a long load trace, then settles into:
```
main: HTTP server is listening, hostname: 127.0.0.1, port: 8081
```

In a second terminal:
```powershell
curl http://localhost:8081/health
# {"status":"ok"}
```

A tiny chat completion:
```powershell
curl http://localhost:8081/v1/chat/completions `
  -Method POST `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{ "messages":[{"role":"user","content":"hello"}], "max_tokens":20 }'
```

If you get a response, your build is good. Ctrl-C the server when done.

> **Picking a port.** llama-server defaults to `8080`, which collides with all sorts of dev tooling (web proxies, Docker dashboards). `8081` is the next-most-obvious choice but is also a common collision target in dev environments — various IPC sockets, dashboards, and vendor desktop apps quietly bind it on launch. Pick a port nothing else on your machine wants; `8181` is a common safe alternative. The same port goes into `llama_cpp_config.json` (section 9).

---

## 9. Wire the binary into Artifex

Now that you have a working `llama-server.exe`, point Artifex at it.

```powershell
cd C:\path\to\Artifex-Assistantv5
copy llama_cpp_config.example.json llama_cpp_config.json
notepad llama_cpp_config.json     # or your editor of choice
```

At minimum, replace **every `/path/to/...` placeholder** with absolute paths:

- `server_path` → the path you noted in section 7 (e.g. `C:/Users/<you>/AI/llama-mainline/llama.cpp/build/bin/Release/llama-server.exe`).
- `models.<name>.path` → the absolute path to a GGUF on disk.
- `models.<name>.extra_flags` may reference an `--mmproj` projector — give that an absolute path too if you're using a vision model.

**Use forward slashes** in JSON paths even on Windows — JSON parses backslashes as escape sequences. Python's `subprocess.Popen` accepts forward slashes on Windows just fine.

For the meaning of every `extra_flags` entry, the per-model tuning advice (KV-cache quantization, speculative decoding, vision projector, `--swa-full` for hybrid models, and the *do-not-use* `--cache-reuse` warning), see the **"Setting Up llama.cpp Backend"** section of `README.md`. This guide doesn't repeat that material.

Start Artifex with the backend selected:
```powershell
python main_api.py --backend llama_cpp
```

The llama-server child process is launched **lazily** on the first chat-completion request, not at API server startup. Watch `logs/llama-server-port<N>.log` for the load trace on that first POST.

---

## 10. Per-GPU substitution table

The only parts of this guide that change per card are **CUDA Toolkit version** (section 1.5) and **`CMAKE_CUDA_ARCHITECTURES`** (section 5). Everything else is the same.

| Card                           | VRAM   | Arch        | `CMAKE_CUDA_ARCHITECTURES` | CUDA Toolkit |
|--------------------------------|--------|-------------|----------------------------|--------------|
| RTX 4090 / 4080                | 24/16  | Ada (sm_89) | `89`                       | 12.4         |
| RTX 4070 / 4070 Ti             | 12     | Ada (sm_89) | `89`                       | 12.4         |
| RTX 4060 / 4060 Ti             | 8      | Ada (sm_89) | `89`                       | 12.4         |
| RTX 3090 / 3090 Ti             | 24     | Ampere (sm_86) | `86`                    | 12.4         |
| RTX 3080 / 3080 Ti             | 10–12  | Ampere (sm_86) | `86`                    | 12.4         |
| RTX 3070 / 3070 Ti             | 8      | Ampere (sm_86) | `86`                    | 12.4         |
| RTX 3060 (12 GB / 8 GB)        | 12 / 8 | Ampere (sm_86) | `86`                    | 12.4         |
| RTX 5090                       | 32     | Blackwell (sm_120) | `120`               | **12.8**     |
| RTX 5080                       | 16     | Blackwell (sm_120) | `120`               | **12.8**     |
| RTX 5070 / 5070 Ti             | 12     | Blackwell (sm_120) | `120`               | **12.8**     |
| RTX 5060 / 5060 Ti             | 8      | Blackwell (sm_120) | `120`               | **12.8**     |
| RTX 20xx (Turing)              | 6–11   | Turing (sm_75) | `75`                    | 12.4         |
| H100 / H200                    | 80     | Hopper (sm_90) | `90`                    | 12.4 or 12.8 |

**Multi-card binaries.** If you build on one machine and copy the binary to another with a different card, generate a multi-arch build by passing a semicolon-separated list, e.g.:
```cmd
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES="86;89;120"
```
The binary gets bigger and the build takes longer, but it runs on Ampere, Ada, and Blackwell.

---

## 11. Build-time troubleshooting

Failures specific to compiling `llama-server`. (Runtime issues — VRAM gating, llama-server crashes mid-request, port collisions at launch — live in `README.md` and the engine's `/health` block, not here.)

### 11.1 Link step fails: `LNK1104: cannot open file 'bin\llama-server.exe'`

**Symptom.** The build compiles cleanly, then the final link step errors with `LNK1104: cannot open file 'bin\llama-server.exe'` (or `llama-server.pdb`).

**Cause.** A real-time antivirus scanner (Windows Defender, third-party endpoint products) is holding the freshly-produced `.exe` open for scanning while `link.exe` tries to overwrite it. This is a race between the build and the AV scanner.

**What to try, in order:**
1. **Just retry once.** From the same Developer Command Prompt, re-run:
   ```cmd
   cmake --build build --config Release --target llama-server -j
   ```
   Most of the time the second pass wins because the file is already partially produced and the AV scan has finished.
2. **Add an AV exclusion** *for the build directory* if you have permission to. On a personal machine, you can add `C:\Users\<you>\AI\llama-mainline\llama.cpp\build\` to Windows Defender's exclusion list (Settings → Privacy & security → Windows Security → Virus & threat protection → Manage settings → Exclusions). On a managed machine, ask IT first — do not disable AV.
3. **Clean and rebuild.** If retries keep failing, blow away the build directory and reconfigure:
   ```cmd
   rmdir /s /q build
   cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=<your-arch>
   cmake --build build --config Release --target llama-server -j
   ```
   Sometimes a stale partial link is the problem, not AV.

Do **not** disable AV without authority to do so, edit the registry, or try to race the scanner. Those are anti-patterns that don't fix the underlying race and create real security holes.

### 11.2 `cl` or `nvcc` "is not recognized"

You opened the wrong terminal. Close it, go back to the Start menu, and launch **"x64 Native Tools Command Prompt for VS 2022"**. Section 2 covers why.

### 11.3 CMake says `CUDA not found`

`nvcc --version` works in a normal terminal but CMake can't find it from the dev prompt. Fix:
- Confirm the CUDA `bin` directory is in your **user** PATH (not just system PATH).
- Reopen the dev prompt so it inherits the updated PATH.
- If that still fails, pass `-DCMAKE_CUDA_COMPILER=...` on the configure command pointing at `nvcc.exe` directly.

### 11.4 `nvcc fatal: Unsupported gpu architecture 'compute_120'`

Your CUDA Toolkit is older than the architecture you asked for. Blackwell (`120`) needs CUDA 12.8 minimum. Upgrade the toolkit (section 1.5), then reconfigure CMake.

### 11.5 Build runs out of memory

Lots of parallel CUDA compilation hammers RAM (4–8 GB per concurrent unit). On a 16 GB system, the `-j` default can OOM. Drop concurrency:
```cmd
cmake --build build --config Release --target llama-server -j 4
```
You can ratchet up from there if 4 succeeds.

### 11.6 Build succeeds but `--version` exits with a missing DLL

The CUDA runtime DLLs aren't on PATH. Add the toolkit's `bin` to your user PATH (`v12.4\bin` or `v12.8\bin`) and reopen the terminal.

### 11.7 Build succeeds but runtime says `no kernel image is available`

You built for an arch your card doesn't have. Most common on Blackwell when the toolkit is 12.4 (which can't emit `sm_120`) and CMake silently downgraded. Reinstall CUDA 12.8 and rebuild with `-DCMAKE_CUDA_ARCHITECTURES=120`.

### 11.8 Git clone fails inside a corporate proxy

If `git clone https://github.com/...` errors with `SSL certificate problem` or `Could not resolve host`, your network is behind a TLS-inspecting proxy. Set the appropriate `http.proxy` Git config or use a personal hotspot. Don't bypass the corporate proxy if there is one.

---

## 12. Linux and macOS notes

The same general flow works outside Windows, with three differences:

**Linux (Ubuntu / Debian).**
- Replace VS Build Tools with `build-essential` + `gcc-11` or newer:
  ```bash
  sudo apt update
  sudo apt install build-essential git cmake
  ```
- Install the CUDA Toolkit per <https://developer.nvidia.com/cuda-downloads> (pick the `.deb (local)` for your distro). Verify `nvcc --version`.
- Clone and build identically:
  ```bash
  git clone https://github.com/ggml-org/llama.cpp.git
  cd llama.cpp
  cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
  cmake --build build --config Release --target llama-server -j
  ```
- The binary lands at `build/bin/llama-server` (no `Release/` subdir, no `.exe`).

**macOS (Apple Silicon).**
- No CUDA. Build with Metal instead of `GGML_CUDA`:
  ```bash
  cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build build --config Release --target llama-server -j
  ```
- You'll need Xcode Command Line Tools (`xcode-select --install`). Homebrew CMake works (`brew install cmake git`).
- The binary lands at `build/bin/llama-server`. Performance on Apple Silicon depends on the chip — M-series GPUs share memory with the CPU, so VRAM math differs from discrete NVIDIA cards.

The Artifex side (`llama_cpp_config.json`, `python main_api.py --backend llama_cpp`) is identical on all three platforms; only the binary path changes.

---

## 13. Quick checklist

Print this and run it linearly on a new build box.

**Before starting**, look up the two values for your card from section 10:
- `CUDA_ARCH` — e.g. `89` (4090), `86` (3060), `120` (5060 Ti)
- `CUDA_TK_VER` — `12.4` for Turing/Ampere/Ada, **`12.8`** for Blackwell

Then:

1. [ ] NVIDIA driver installed — `nvidia-smi` lists your GPU
2. [ ] Python 3.11 installed — `python --version` shows `3.11.x`
3. [ ] Git installed — `git --version` works
4. [ ] VS 2022 Build Tools installed (Desktop development with C++)
5. [ ] CUDA Toolkit `CUDA_TK_VER` installed — `nvcc --version` reports the matching version
6. [ ] CMake 3.20+ installed — `cmake --version` works
7. [ ] Open **"x64 Native Tools Command Prompt for VS 2022"** — `cl` and `nvcc` both respond
8. [ ] `mkdir C:\Users\%USERNAME%\AI\llama-mainline && cd` into it
9. [ ] `git clone https://github.com/ggml-org/llama.cpp.git`
10. [ ] `cd llama.cpp`
11. [ ] (Optional) `git checkout <pinned-commit>` if you want byte-for-byte reproducibility
12. [ ] `cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=CUDA_ARCH`
13. [ ] `cmake --build build --config Release --target llama-server -j` (watch for `LNK1104` — see 11.1)
14. [ ] `llama-server.exe --version` works and reports the expected build/commit
15. [ ] `pip install -U "huggingface_hub[cli]"` if you haven't already
16. [ ] `huggingface-cli download <repo> <model>.gguf --local-dir <local-dir>` for any GGUF you want to test against
17. [ ] Smoke-launch the binary directly (`llama-server.exe -m <gguf> --port 8081 -ngl 99 -c 8192`) and hit `/health` — `{"status":"ok"}`
18. [ ] `cd` to the Artifex repo; `copy llama_cpp_config.example.json llama_cpp_config.json`
19. [ ] Fill `server_path` and `models.*.path` with the absolute paths from steps 14 and 16 (forward slashes in JSON)
20. [ ] `python main_api.py --backend llama_cpp` — first chat completion lazily launches `llama-server`; subsequent requests are fast

Done. When the build itself misbehaves, jump to section 11. For everything past "the binary works" — model tuning, VRAM math, runtime crashes, vision wiring — `README.md` is the source of truth.

---

*"Unless the LORD builds the house, the builders labor in vain." — Psalm 127:1*
