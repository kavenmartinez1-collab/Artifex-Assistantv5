"""Live forward-pass telemetry for the transformers backend.

Registers lightweight forward hooks on a HuggingFace CausalLM so the GUI can
visualize what the model is actually doing per generated token:

  * per-layer residual-stream L2 norm  (how "loud" each layer's output is)
  * for MoE models, the router's top-k expert selection + gate weights
    (which experts fired this token — the headline signal for Qwen3.5/3.6 MoE)

Everything captured here is tiny (a few floats/ints per layer per token) and is
reduced to scalars on-GPU before leaving the hook, so it costs ~3 device syncs
per token regardless of model depth. Attention is *deliberately untouched* —
requesting attention weights forces eager attention and disables SDPA/flash,
which is the only genuinely expensive telemetry. We skip it.

Hard rule: telemetry must NEVER break inference. Every hook body is wrapped so a
failure records ``None`` for that slot and is otherwise silent.

Frame schema (one dict per generated token, JSON-safe):

    {
        "step":     int,          # decode step, 0-based
        "n_layers": int,
        "moe":      bool,         # True if any layer reported expert routing
        "layers": [               # length == n_layers, input->output order
            {
                "norm":    float | None,                # residual L2 norm (last position)
                "experts": [[expert_id, gate_weight], ...] | None,  # MoE top-k, else None
            },
            ...
        ],
        # "token": str            # decoded text — filled in by the caller
    }
"""
from __future__ import annotations

import queue


def _module_by_path(root, path):
    obj = root
    for attr in path.split("."):
        if not hasattr(obj, attr):
            return None
        obj = getattr(obj, attr)
    return obj


def _find_decoder_layers(model):
    """Return the ordered list of transformer decoder layers, or []."""
    import torch.nn as nn

    # Common attribute paths across dense + multimodal HF wrappers.
    for path in (
        "model.layers",
        "model.model.layers",
        "model.language_model.layers",
        "model.model.language_model.layers",
        "language_model.model.layers",
        "transformer.h",
        "gpt_neox.layers",
    ):
        obj = _module_by_path(model, path)
        if isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return list(obj)

    # Fallback: the longest ModuleList whose children look like decoder blocks.
    best = None
    for _name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            cls = type(mod[0]).__name__
            if cls.endswith("DecoderLayer") or cls.endswith("Block"):
                if best is None or len(mod) > len(best):
                    best = mod
    return list(best) if best is not None else []


def _is_moe(model):
    cfg = getattr(model, "config", None)
    if cfg is None:
        return False
    for attr in ("num_experts", "num_local_experts", "n_routed_experts",
                 "moe_num_experts"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 1:
            return True
    return False


def _find_gate(layer):
    """Return the router/gate module of an MoE decoder layer, or None.

    Matches Qwen3-MoE (``mlp.gate``), Mixtral (``block_sparse_moe.gate``) and
    common variants. A dense MLP has no attribute literally named ``gate``
    (it has ``gate_proj``), so this returns None for dense layers.
    """
    import torch.nn as nn
    for mlp_attr in ("mlp", "block_sparse_moe", "moe", "feed_forward"):
        mlp = getattr(layer, mlp_attr, None)
        if mlp is None:
            continue
        for gate_attr in ("gate", "router"):
            gate = getattr(mlp, gate_attr, None)
            if isinstance(gate, nn.Module):
                return gate
    return None


class TelemetryCapture:
    """Hook a CausalLM and stream per-token forward-pass telemetry.

    Usage::

        cap = TelemetryCapture(model)        # registers hooks
        ...run model.generate(...)...        # hooks fill an internal queue
        for frame in cap.drain(): ...        # pull frames from the decode loop
        cap.close()                          # ALWAYS remove hooks

    ``close()`` must be called (use try/finally) or the hooks leak onto the
    model object and fire on the next generation.
    """

    def __init__(self, model, max_queue=512):
        self.q = queue.Queue(maxsize=max_queue)
        self._handles = []
        self._layers = _find_decoder_layers(model)
        self._n = len(self._layers)
        self._is_moe = _is_moe(model)
        try:
            self._topk = int(getattr(model.config, "num_experts_per_tok", 0)) or 8
        except Exception:
            self._topk = 8
        # Per-token scratch: GPU tensors, converted to Python in _flush().
        self._norms = [None] * self._n          # 0-d float tensors
        self._experts = [None] * self._n         # (idx_tensor, weight_tensor)
        self._step = 0
        if self._n:
            self._register()

    @property
    def n_layers(self):
        return self._n

    # ---- hook registration -------------------------------------------
    def _register(self):
        for i, layer in enumerate(self._layers):
            self._handles.append(layer.register_forward_hook(self._norm_hook(i)))
            if self._is_moe:
                gate = _find_gate(layer)
                if gate is not None:
                    self._handles.append(
                        gate.register_forward_hook(self._gate_hook(i))
                    )

    def _norm_hook(self, i):
        is_last = i == self._n - 1

        def hook(_module, _inputs, output):
            try:
                hs = output[0] if isinstance(output, (tuple, list)) else output
                # last token of the (flattened) sequence
                vec = hs.reshape(-1, hs.shape[-1])[-1]
                self._norms[i] = vec.detach().float().norm()
            except Exception:
                self._norms[i] = None
            # The final layer firing means the token's forward is complete.
            if is_last:
                self._flush()

        return hook

    def _gate_hook(self, i):
        import torch

        def hook(_module, _inputs, output):
            try:
                logits = output[0] if isinstance(output, (tuple, list)) else output
                row = logits.reshape(-1, logits.shape[-1])[-1].detach().float()
                k = min(self._topk, row.shape[-1])
                vals, idx = torch.topk(row, k)
                self._experts[i] = (idx, torch.softmax(vals, dim=-1))
            except Exception:
                self._experts[i] = None

        return hook

    # ---- frame assembly ----------------------------------------------
    def _flush(self):
        import torch
        n = self._n
        norms = [None] * n
        experts = [None] * n

        # One device sync for every layer's norm.
        try:
            idxs = [j for j in range(n) if self._norms[j] is not None]
            if idxs:
                vals = torch.stack([self._norms[j] for j in idxs]).tolist()
                for j, v in zip(idxs, vals):
                    norms[j] = float(v)
        except Exception:
            pass

        # Two device syncs for all MoE layers (uniform k => stackable).
        mj = [j for j in range(n) if self._experts[j] is not None]
        if mj:
            try:
                idx_l = torch.stack([self._experts[j][0] for j in mj]).tolist()
                w_l = torch.stack([self._experts[j][1] for j in mj]).tolist()
                for j, ids, ws in zip(mj, idx_l, w_l):
                    experts[j] = [[int(e), float(w)] for e, w in zip(ids, ws)]
            except Exception:
                # Ragged or odd shapes — fall back to per-layer conversion.
                for j in mj:
                    try:
                        ids = self._experts[j][0].tolist()
                        ws = self._experts[j][1].tolist()
                        experts[j] = [[int(e), float(w)] for e, w in zip(ids, ws)]
                    except Exception:
                        experts[j] = None

        frame = {
            "step": self._step,
            "n_layers": n,
            "moe": any(e is not None for e in experts),
            "layers": [{"norm": norms[j], "experts": experts[j]} for j in range(n)],
        }
        self._step += 1
        self._norms = [None] * n
        self._experts = [None] * n
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            pass  # viz is best-effort; drop under backpressure

    # ---- consumption --------------------------------------------------
    def drain(self):
        """Return and clear all queued frames (non-blocking)."""
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                break
        return out

    def close(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []
        self._layers = []


def iter_mock_frames(steps=10_000, n_layers=48, n_experts=128, k=8, seed=1):
    """Deterministic mock telemetry so the viz can be built without a model.

    Yields the exact frame schema TelemetryCapture produces. Uses a tiny LCG
    (no external deps, no torch) so the panel can be developed standalone.
    """
    state = (seed & 0x7fffffff) or 1

    def rnd():
        nonlocal state
        state = (state * 1103515245 + 12345) & 0x7fffffff
        return state / 0x7fffffff

    for s in range(steps):
        layers = []
        for li in range(n_layers):
            depth = li / max(1, n_layers - 1)
            bulge = 1.0 - abs(depth - 0.5) * 2.0          # peaks mid-stack
            norm = 3.0 + 20.0 * (0.4 + 0.6 * bulge) * (0.7 + 0.3 * rnd())
            picks, seen = [], set()
            while len(picks) < k:
                e = int(rnd() * n_experts)
                if e not in seen:
                    seen.add(e)
                    picks.append(e)
            raw = [rnd() for _ in picks]
            tot = sum(raw) or 1.0
            layers.append({
                "norm": norm,
                "experts": [[e, w / tot] for e, w in zip(picks, raw)],
            })
        yield {
            "step": s, "n_layers": n_layers, "moe": True,
            "layers": layers, "token": " ·",
        }
