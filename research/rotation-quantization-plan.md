# Research Plan: Rotation-Based Near-Lossless Quantization for Hybrid SSM+Attention Models

## Goal
Develop a quantization pipeline that achieves near-zero accuracy loss for Qwen3.5's hybrid DeltaNet+attention architecture, enabling browser-based WebGPU inference at minimal VRAM with maximum quality.

## Core Insight
Standard INT4 quantization (GPTQ, AWQ) works for transformers but catastrophically fails on SSM/recurrent layers because quantization errors compound through the state recurrence. Rotation-based techniques can redistribute weight information to make quantization nearly lossless by smoothing outlier channels before compression.

---

## Research Topics

### Topic 1: MambaQuant — KLT-Enhanced Rotation for SSM Models
**Why**: Directly addresses our exact problem (SSM quantization failure). Achieved 77.8% accuracy vs SmoothQuant's 32.3% on SSM models at W8A8.
**Paper**: https://arxiv.org/abs/2501.13484 (ICLR 2025)
**Research questions**:
- How does the Karhunen-Loeve Transform (KLT) equalize channel variances in SSM weight matrices?
- What is the computational cost of computing KLT per layer? Is it a one-time offline step?
- How are the rotation matrices stored and applied at inference time? Can they be fused into adjacent weights?
- Does MambaQuant work at W4A16 (our target) or only W8A8?
- How does it handle hybrid models (SSM + attention layers in the same model)?

### Topic 2: Quamba2 — Sort-and-Cluster SSM Quantization
**Why**: Specifically designed for Mamba2-family models (DeltaNet is closely related). ICML 2025.
**Paper**: https://arxiv.org/abs/2503.22879
**Research questions**:
- What is the sort-and-cluster strategy and how does it differ from standard group quantization?
- How does per-state-group quantization work (8 groups of 128 channels)?
- What is the Hadamard smoothing technique and how does it differ from MambaQuant's KLT?
- Can the clustered quantization scheme be efficiently dequantized in a WGSL compute shader?
- What W4 results do they achieve vs baseline GPTQ?

### Topic 3: QuIP# — Hadamard Rotation for Weight Incoherence
**Why**: Achieves 2-bit weight quantization with reasonable quality. The Hadamard transform is computationally cheap (no learned parameters).
**Paper**: https://arxiv.org/abs/2402.04396
**Repository**: https://github.com/Cornell-RelaxML/quip-sharp
**Research questions**:
- How does the randomized Hadamard transform create "incoherent" weight matrices?
- What is the lattice codebook and how does it differ from uniform INT4 quantization?
- Can we use just the Hadamard rotation preprocessing with our existing GPTQ quantizer?
- What is the inference overhead of applying the inverse Hadamard at runtime?
- Does this work for non-square weight matrices (our projections are rectangular)?

### Topic 4: PCDVQ — Polar Coordinate Weight Decomposition
**Why**: Directly applies the polar coordinate idea (from TurboQuant) to weight quantization. Achieves 2-bit with 1.5%+ improvement over baselines.
**Paper**: https://arxiv.org/abs/2506.05432
**Research questions**:
- How does decomposing weights into direction + magnitude improve quantization?
- Finding: direction is 20x more sensitive than magnitude — can we keep direction at high precision and compress magnitude aggressively?
- How are the polar coordinates computed and stored? What's the VRAM overhead?
- Can the direction/magnitude decomposition be efficiently reconstructed in WGSL?

### Topic 5: ParoQuant — Pairwise Givens Rotations
**Why**: Uses learned pairwise rotations per layer. 2.4% accuracy improvement over AWQ at W4A16.
**Paper/Site**: https://z-lab.ai/projects/paroquant/
**Research questions**:
- How are the Givens rotation angles learned (calibration-based or gradient-based)?
- How many rotation parameters per layer? What's the storage cost?
- Can Givens rotations be fused into the weight matrix offline (zero inference cost)?
- Does this generalize to SSM weight matrices or only attention/FFN?

### Topic 6: WebGPU shader-f16 for Mixed-Precision SSM
**Why**: Chrome supports native f16 compute. The model was TRAINED in BF16 — matching training precision could reduce drift.
**References**:
- https://developer.chrome.com/blog/new-in-webgpu-120
- https://www.intel.com/content/www/us/en/developer/articles/community/revving-up-webgpu-applications-with-power-of-f16.html
**Research questions**:
- Can we use `enable f16;` in WGSL to run SSM state accumulation in f16?
- What is the perf difference (reported 2.1x prefill, 1.3x decode speedup)?
- BF16 vs F16: WebGPU only supports IEEE f16, not BF16. Is the precision difference significant for SSM stability?
- Which GPU vendors support shader-f16? (Qualcomm excluded — but we're on NVIDIA)
- Can we do mixed f16/f32 within a single shader (f16 for accumulation, f32 for final output)?

### Topic 7: Lyapunov Stability and Error Bounds
**Why**: Theoretical foundation for understanding WHY the drift happens and what the theoretical limits are.
**Paper**: https://arxiv.org/abs/2406.00209
**Research questions**:
- What is the proven bound on error amplification in Mamba's selective SSM?
- How does the Lyapunov exponent relate to the quantization bit-width needed?
- Is there a formula for: given X bits of weight precision, the max stable sequence length before drift exceeds threshold Y?
- Does this analysis extend to Gated DeltaNet (our architecture) or only vanilla Mamba?

---

## Research Execution Order

**Phase 1 — Core techniques (highest impact)**
1. MambaQuant (Topic 1) — most directly relevant to our problem
2. Quamba2 (Topic 2) — second-most relevant, different approach

**Phase 2 — Complementary methods**
3. QuIP# (Topic 3) — cheapest rotation (Hadamard = no learned params)
4. WebGPU shader-f16 (Topic 6) — orthogonal improvement, quick to test

**Phase 3 — Advanced techniques**
5. PCDVQ (Topic 4) — novel polar decomposition
6. ParoQuant (Topic 5) — learned rotations
7. Lyapunov stability (Topic 7) — theoretical bounds

---

## Output Format
For each topic, store research findings in `research/findings/` as individual markdown files:
- `mambaQuant.md`
- `quamba2.md`
- `quip-sharp.md`
- `pcdvq.md`
- `paroquant.md`
- `webgpu-f16.md`
- `lyapunov-ssm.md`

Each file should contain:
1. **Summary** — one paragraph explaining the technique
2. **Key algorithm** — step-by-step pseudocode of the core method
3. **Applicability** — how it maps to our Qwen3.5 hybrid model + WebGPU pipeline
4. **Implementation cost** — what changes to quantizer, weight loader, and shaders
5. **Expected benefit** — estimated quality/size improvement based on paper results
6. **Open questions** — what we still don't know

## Final Deliverable
An implementation plan in `research/implementation-plan.md` that synthesizes all findings into a concrete engineering roadmap: which techniques to combine, what to build first, expected VRAM budget, and estimated development effort.
