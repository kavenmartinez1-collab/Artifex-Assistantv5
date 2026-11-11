# Artifex-Assistant v5 — Core infrastructure

import os as _os

# PyTorch's Windows build allocates CPU tensors with a bundled mimalloc
# (v2.2.4 in torch 2.10), which hands freed memory back to Windows only on
# a delay, and runs that delayed release from its NEXT allocation. After a
# Create job nothing in the API allocates CPU tensors again, so an unloaded
# Qwen-Image kept ~18 GiB committed for good, and the Strata split, which
# needs that RAM, then failed to start ("bad allocation"). With no delay
# the release happens at free: 18.6 -> 4.0 GiB right after unload, measured
# 2026-11-10, generation no slower. mimalloc reads this when torch loads,
# so it has to be set before the first `import torch`; every entry point
# imports core first. A value already in the environment wins.
_os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")
