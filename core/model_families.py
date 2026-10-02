"""Family defaults for llama.cpp models, chosen from a GGUF's own metadata.

Rules key on general.architecture (and what the file contains: an MTP head,
sliding-window layers, a vision projector next to it). They supply the
flags that depend on the MODEL. Flags that depend on the MACHINE (backend
binary, split, KV type, load mode) come from a template entry that already
works here; see core.model_inventory.propose_entry.

Every value notes its source, so the UI can say why a flag is there.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.gguf_meta import GGUFInfo


@dataclass
class FamilyDefaults:
    family: str
    flags: list = field(default_factory=list)
    notes: list = field(default_factory=list)       # human-readable reasons, one per decision
    warnings: list = field(default_factory=list)
    vision: bool = False                            # the family can use an mmproj projector


QWEN_HYBRID = {"qwen35", "qwen3next"}       # Qwen3.5 / 3.6 / 3.8 hybrid (gated DeltaNet + attention)
QWEN = {"qwen2", "qwen3", "qwen3moe", "qwen2moe", "qwen2vl", "qwen25vl"}
GEMMA = {"gemma", "gemma2", "gemma3", "gemma3n", "gemma4"}


def family_defaults(info: GGUFInfo) -> FamilyDefaults:
    arch = info.architecture.lower()
    if arch in QWEN_HYBRID:
        d = FamilyDefaults("qwen-hybrid", vision=True)
        d.flags += ["--jinja", "--reasoning-format", "deepseek"]
        d.notes.append("Qwen hybrid thinking model: --jinja, with think blocks split out (--reasoning-format deepseek)")
        if info.nextn_predict_layers:
            d.flags += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "-ctkd", "q4_0", "-ctvd", "q4_0"]
            d.notes.append("Built-in MTP head: draft 3 tokens (measured +32% decode at temperature 1); "
                           "draft KV at q4_0 (otherwise f16)")
    elif arch in QWEN:
        d = FamilyDefaults("qwen", vision=arch.endswith("vl"))
        d.flags += ["--jinja", "--reasoning-format", "deepseek"]
        d.notes.append("Qwen chat template: --jinja, think blocks split out")
    elif arch in GEMMA:
        d = FamilyDefaults("gemma", vision=True)
        d.flags += ["--jinja"]
        d.notes.append("Gemma chat template: --jinja")
    elif arch == "llama":
        d = FamilyDefaults("llama")
        d.flags += ["--jinja"]
        d.notes.append("Llama chat template: --jinja")
    else:
        d = FamilyDefaults(arch or "unknown")
        d.flags += ["--jinja"]
        d.notes.append(f"No family rule for architecture '{arch or '?'}': generic --jinja only")
    if info.sliding_window:
        d.flags += ["--swa-full"]
        d.notes.append(f"Sliding-window attention ({info.sliding_window}): --swa-full keeps prompt reuse working")
    if not info.has_chat_template:
        d.warnings.append("The file has no chat template; --jinja will fall back to a generic one.")
    if "moe" in arch:
        d.warnings.append("Mixture-of-experts model: if it does not fit, add --cpu-moe or --n-cpu-moe N.")
    if info.iq_share >= 0.3:
        d.warnings.append(f"{info.iq_share:.0%} of the weights are IQ codebook formats; on the Vulkan/ROCm "
                          "GPUs measured here that cost 35-38% decode speed versus plain K-quants.")
    return d
