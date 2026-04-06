"""
Artifex Assistant V5 — Centralized GPU/device information.
Single source of truth for GPU detection, VRAM queries, and device properties.
Replaces the 15+ duplicated torch.cuda.get_device_properties() calls throughout
the codebase with a cached singleton.

Usage:
    from core.device import gpu_info
    gpu_info.total_gb       # 8.0
    gpu_info.free_gb        # 5.2
    gpu_info.name           # "NVIDIA GeForce RTX 5060 Ti"
    gpu_info.tier           # "TIGHT"
    gpu_info.is_available   # True
    gpu_info.compute_cap    # 12
"""

import threading


class GPUInfo:
    """Cached GPU information singleton.

    Probes the GPU once on first access, caches the static properties
    (name, total VRAM, compute capability, tier). Dynamic properties
    (allocated, free) are refreshed on each call.
    """

    def __init__(self):
        self._probed = False
        self._lock = threading.Lock()
        # Static (cached after first probe)
        self._is_available = False
        self._name = "NO_GPU"
        self._total_gb = 0.0
        self._total_bytes = 0
        self._compute_cap = 0
        self._compute_cap_minor = 0
        self._device_count = 0
        self._tier = "TIGHT"

    def _probe(self):
        """Probe GPU properties. Called once, results cached."""
        if self._probed:
            return
        with self._lock:
            if self._probed:
                return
            try:
                import torch
                if not torch.cuda.is_available():
                    self._probed = True
                    return

                self._is_available = True
                props = torch.cuda.get_device_properties(0)
                self._name = props.name
                self._total_bytes = props.total_memory
                self._total_gb = props.total_memory / (1024 ** 3)
                self._compute_cap = props.major
                self._compute_cap_minor = props.minor
                self._device_count = torch.cuda.device_count()

                # Tier classification
                if self._total_gb <= 12:
                    self._tier = "TIGHT"
                elif self._total_gb <= 20:
                    self._tier = "COMFORTABLE"
                else:
                    self._tier = "ABUNDANT"

            except Exception:
                pass  # No GPU / torch not installed
            self._probed = True

    # ── Static properties (cached) ──────────────────────────────────────

    @property
    def is_available(self) -> bool:
        self._probe()
        return self._is_available

    @property
    def name(self) -> str:
        self._probe()
        return self._name

    @property
    def total_gb(self) -> float:
        self._probe()
        return self._total_gb

    @property
    def total_bytes(self) -> int:
        self._probe()
        return self._total_bytes

    @property
    def compute_cap(self) -> int:
        """Major compute capability (e.g., 8 for Ampere, 12 for Blackwell)."""
        self._probe()
        return self._compute_cap

    @property
    def compute_cap_minor(self) -> int:
        self._probe()
        return self._compute_cap_minor

    @property
    def compute_cap_str(self) -> str:
        """Compute capability as 'major.minor' string."""
        self._probe()
        return f"{self._compute_cap}.{self._compute_cap_minor}"

    @property
    def device_count(self) -> int:
        self._probe()
        return self._device_count

    @property
    def tier(self) -> str:
        """GPU tier: TIGHT (<=12GB), COMFORTABLE (13-20GB), ABUNDANT (>20GB)."""
        self._probe()
        return self._tier

    # ── Dynamic properties (fresh each call) ────────────────────────────

    @property
    def allocated_gb(self) -> float:
        """Currently allocated VRAM in GB. Fresh on each access."""
        if not self.is_available:
            return 0.0
        try:
            import torch
            return torch.cuda.memory_allocated(0) / (1024 ** 3)
        except Exception:
            return 0.0

    @property
    def free_gb(self) -> float:
        """Estimated free VRAM in GB. Fresh on each access."""
        return self.total_gb - self.allocated_gb

    @property
    def usage_fraction(self) -> float:
        """VRAM usage as a fraction (0.0-1.0). Fresh on each access."""
        if self.total_gb <= 0:
            return 0.0
        return self.allocated_gb / self.total_gb

    # ── Utility methods ─────────────────────────────────────────────────

    @property
    def compute_dtype(self):
        """Recommended compute dtype based on compute capability.

        BF16 for compute_cap >= 8 (Ampere+), FP16 for older GPUs.
        """
        import torch
        return torch.bfloat16 if self.compute_cap >= 8 else torch.float16

    def summary(self) -> dict:
        """Return a summary dict for health checks / status display."""
        return {
            "available": self.is_available,
            "name": self.name,
            "total_gb": round(self.total_gb, 2),
            "allocated_gb": round(self.allocated_gb, 2),
            "free_gb": round(self.free_gb, 2),
            "tier": self.tier,
            "compute_capability": self.compute_cap_str,
            "device_count": self.device_count,
        }

    def reset(self):
        """Reset all cached state. Used by tests that mock torch.cuda."""
        with self._lock:
            self._probed = False
            self._is_available = False
            self._name = "NO_GPU"
            self._total_gb = 0.0
            self._total_bytes = 0
            self._compute_cap = 0
            self._compute_cap_minor = 0
            self._device_count = 0
            self._tier = "TIGHT"

    def __repr__(self):
        if not self.is_available:
            return "GPUInfo(available=False)"
        return (
            f"GPUInfo({self.name}, {self.total_gb:.1f}GB, "
            f"tier={self.tier}, cc={self.compute_cap_str})"
        )


# Module-level singleton — import this
gpu_info = GPUInfo()
