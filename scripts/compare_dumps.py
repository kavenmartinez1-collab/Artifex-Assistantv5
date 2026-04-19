"""Compare engine dump vs PyTorch reference — find first-divergence layer.

Loads two JSON dumps (engine's __DEBUG_DUMP_RESULT__ and reference output of
divergence_probe.py) and reports, for each layer, cosine similarity and
relative L2 error on the last-token hidden state vector. Flags the first
layer where cosine drops below a configurable threshold.

Usage:
    ./venv/Scripts/python.exe scripts/compare_dumps.py \\
        --engine webgpu/debug-output.json --reference reference_dump.json

The engine dump is the full POST body from /api/debug — we pull .layerDump
out of it. If you pass a bare layer-dump JSON, it is used as-is.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    num = 0.0
    aa = 0.0
    bb = 0.0
    for i in range(n):
        num += a[i] * b[i]
        aa += a[i] * a[i]
        bb += b[i] * b[i]
    if aa == 0 or bb == 0:
        return float("nan")
    return num / (math.sqrt(aa) * math.sqrt(bb))


def l2_rel(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    num = 0.0
    den = 0.0
    for i in range(n):
        d = a[i] - b[i]
        num += d * d
        den += b[i] * b[i]
    if den == 0:
        return float("nan")
    return math.sqrt(num / den)


def topk_tokens(logits: list[float], k: int = 10) -> list[tuple[int, float]]:
    idx_val = sorted(enumerate(logits), key=lambda p: p[1], reverse=True)
    return idx_val[:k]


def kl_logits(p: list[float], q: list[float]) -> float:
    """Symmetric-KL-ish: KL(softmax(p) || softmax(q)). Uses log-sum-exp for stability."""
    n = min(len(p), len(q))
    if n == 0:
        return float("nan")
    pm = max(p[:n])
    qm = max(q[:n])
    ez_p = [math.exp(p[i] - pm) for i in range(n)]
    ez_q = [math.exp(q[i] - qm) for i in range(n)]
    sp = sum(ez_p)
    sq = sum(ez_q)
    kl = 0.0
    for i in range(n):
        pi = ez_p[i] / sp
        qi = ez_q[i] / sq
        if pi > 0 and qi > 0:
            kl += pi * math.log(pi / qi)
    return kl


def load_dump(path: Path) -> dict:
    """Return a dict of { tag: { layers: [...], ... } } regardless of source shape.

    Accepted shapes:
      - Engine POST body:    { layerDump: { tag1: {...}, tag2: {...} }, ... }
      - Bare engine dump:    { tag1: {...}, tag2: {...} }
      - Reference probe out: { tags: { tag1: {...}, tag2: {...} }, ... }
      - Legacy single-tag:   { layers: [...], ... }   -> wrapped under 'dump'
    """
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected JSON object")
    body = raw.get("layerDump") if "layerDump" in raw and raw["layerDump"] else raw
    if not isinstance(body, dict):
        raise ValueError(f"{path}: layerDump is not an object")
    if "tags" in body and isinstance(body["tags"], dict):
        return body["tags"]
    if "layers" in body and isinstance(body["layers"], list):
        # Legacy single-tag format.
        return {"dump": body}
    # Tag-keyed shape: each value should itself be a dump-with-layers.
    out: dict[str, dict] = {}
    for k, v in body.items():
        if isinstance(v, dict) and isinstance(v.get("layers"), list):
            out[k] = v
    if not out:
        raise ValueError(f"{path}: could not find any tag with .layers")
    return out


def index_by_label(dump: dict) -> dict[str, dict]:
    return {entry["label"]: entry for entry in dump["layers"]}


def load_layer_types(model_dir: Path) -> dict[int, str]:
    """Read layer_types from config.json so we can annotate each layer (FA vs SSM)."""
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return {}
    text_cfg = cfg.get("text_config", cfg)
    types = text_cfg.get("layer_types") or text_cfg.get("layer_type")
    if not isinstance(types, list):
        return {}
    return {i: str(t) for i, t in enumerate(types)}


def short_layer_type(t: str) -> str:
    if not t:
        return "?"
    if "linear" in t or "ssm" in t.lower() or "delta" in t.lower():
        return "SSM"
    if "full" in t or "attn" in t.lower():
        return "FA"
    return t[:3]


def maybe_load_tokenizer(model_dir: Path):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(model_dir))
    except Exception:
        return None


def load_label_map(map_path: Path) -> dict[str, str]:
    """Load engine-projection-shortname -> HF-dotted-path mapping.

    The map decouples engine names ('lin-qkv', 'q', 'gate') from per-family HF
    submodule paths ('linear_attn.in_proj_qkv', 'self_attn.q_proj', ...). To
    onboard a new model family, drop a new compare_label_map.<family>.json
    next to this script and pass --label-map. No engine or comparator code
    changes needed.
    """
    if not map_path.exists():
        return {}
    try:
        raw = json.loads(map_path.read_text())
    except Exception as e:
        print(f"[cmp] WARNING: label map {map_path} parse failed: {e}", file=sys.stderr)
        return {}
    projs = raw.get("projections")
    if not isinstance(projs, dict):
        return {}
    return {str(k): str(v) for k, v in projs.items()}


def resolve_engine_audit_label(label: str, label_map: dict[str, str]) -> str | None:
    """Engine 'L${l}-${proj}-out' -> reference 'L${l}-${dotted}-out' via map.

    Returns None if the label is not an audit label, or if the projection name
    isn't in the map (in which case we leave it unmatched and print a hint).
    """
    if not (label.startswith("L") and label.endswith("-out")):
        return None
    body = label[1:-len("-out")]  # strip leading 'L' and trailing '-out'
    # body is now '<l>-<proj>' where proj may itself contain hyphens
    dash = body.find("-")
    if dash < 0:
        return None
    li_str, proj = body[:dash], body[dash + 1:]
    if not li_str.isdigit():
        return None
    if proj not in label_map:
        return None
    return f"L{li_str}-{label_map[proj]}-out"


def compare_one_tag(tag: str, eng: dict, ref: dict, layer_types: dict[int, str],
                    tokenizer, threshold: float, top_k: int,
                    label_map: dict[str, str] | None = None) -> str | None:
    """Compare one tag pair. Returns first-divergence label (or None)."""
    print()
    print(f"==================== TAG: {tag} ====================")
    print(f"[cmp] engine:    seqLen={eng.get('seqLen')} pos={eng.get('pos')} "
          f"H={eng.get('H')} V={eng.get('V')} entries={len(eng['layers'])}")
    print(f"[cmp] reference: seqLen={ref.get('seqLen')} pos={ref.get('pos')} "
          f"H={ref.get('H')} V={ref.get('V')} entries={len(ref['layers'])}")
    if eng.get("H") and ref.get("H") and eng.get("H") != ref.get("H"):
        print(f"[cmp] WARNING: H mismatch ({eng.get('H')} vs {ref.get('H')})")

    eng_ix = index_by_label(eng)
    ref_ix = index_by_label(ref)

    first_divergence: str | None = None
    print()
    print(f"{'label':<32} {'type':>4} {'eng_abs':>10} {'ref_abs':>10} "
          f"{'cosine':>10} {'rel_L2':>10} {'NaN':>5} {'Inf':>5}")
    print("-" * 100)
    for entry in eng["layers"]:
        label = entry["label"]
        if label == "logits":
            continue
        ref_label = label
        # Engine audit labels ('L21-lin-qkv-out') map to per-family HF dotted
        # paths in the reference dump ('L21-linear_attn.in_proj_qkv-out').
        if label not in ref_ix and label_map:
            mapped = resolve_engine_audit_label(label, label_map)
            if mapped is not None and mapped in ref_ix:
                ref_label = mapped
        if ref_label not in ref_ix:
            print(f"{label:<32} (no reference)")
            continue
        es = entry["stats"]
        rs = ref_ix[ref_label]["stats"]
        cos = cosine(entry["sample"], ref_ix[ref_label]["sample"])
        rel = l2_rel(entry["sample"], ref_ix[ref_label]["sample"])
        # Layer-type annotation if label is "layer-N-out".
        ltype = "-"
        if label.startswith("layer-") and label.endswith("-out"):
            try:
                idx = int(label[len("layer-"):-len("-out")])
                ltype = short_layer_type(layer_types.get(idx, ""))
            except ValueError:
                pass
        flag = " <<<" if (not math.isnan(cos) and cos < threshold and first_divergence is None) else ""
        if flag:
            first_divergence = label
        print(f"{label:<32} {ltype:>4} {es['absMean']:>10.4f} {rs['absMean']:>10.4f} "
              f"{cos:>10.4f} {rel:>10.4f} {es['nanCount']:>5d} {es['infCount']:>5d}{flag}")

    print()
    if first_divergence:
        print(f"[cmp] [{tag}] FIRST DIVERGENCE (cosine < {threshold}): {first_divergence}")
    else:
        print(f"[cmp] [{tag}] no layer below threshold {threshold}")

    # Logits — full V-vector comparison + decoded top-K.
    if "logits" in eng_ix and "logits" in ref_ix:
        print()
        print(f"[cmp] [{tag}] === LOGITS ===")
        e_log = eng_ix["logits"]["sample"]
        r_log = ref_ix["logits"]["sample"]
        cos = cosine(e_log, r_log)
        rel = l2_rel(e_log, r_log)
        kl = kl_logits(e_log, r_log)
        print(f"  cosine={cos:.4f}  rel_L2={rel:.4f}  KL(eng||ref)={kl:.4f}")
        e_top = topk_tokens(e_log, top_k)
        r_top = topk_tokens(r_log, top_k)

        def fmt_top(top):
            out = []
            for i, v in top:
                tok_str = ""
                if tokenizer is not None:
                    try:
                        tok_str = tokenizer.decode([i]).replace("\n", "\\n")[:12]
                    except Exception:
                        tok_str = ""
                # Escape non-ASCII so Windows cp1252 stdout doesn't choke on CJK,
                # emoji, etc. that appear when the engine drifts into other-script
                # tokens — the very thing we're diagnosing.
                tok_str = tok_str.encode("ascii", "backslashreplace").decode("ascii")
                out.append(f"{i}({tok_str!r}={v:.2f})")
            return ", ".join(out)

        print(f"  engine    top-{top_k}: {fmt_top(e_top)}")
        print(f"  reference top-{top_k}: {fmt_top(r_top)}")
        overlap = len(set(i for i, _ in e_top) & set(i for i, _ in r_top))
        print(f"  top-{top_k} overlap: {overlap}/{top_k}")

    return first_divergence


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True, help="engine dump JSON (or /api/debug POST body)")
    ap.add_argument("--reference", required=True, help="PyTorch reference JSON from divergence_probe.py")
    ap.add_argument("--threshold", type=float, default=0.95,
                    help="cosine threshold below which a layer is flagged (default 0.95)")
    ap.add_argument("--top", type=int, default=10, help="top-K logits to compare (default 10)")
    ap.add_argument("--model-dir", default="models/qwen3.5-9b-HailMary",
                    help="model directory used for layer_types annotation + tokenizer decoding")
    ap.add_argument("--label-map", default="scripts/compare_label_map.qwen35.json",
                    help="per-family JSON mapping engine projection short-names to HF dotted "
                         "submodule paths. Required only when comparing audit dumps.")
    args = ap.parse_args()

    e_path = Path(args.engine)
    r_path = Path(args.reference)
    if not e_path.exists():
        print(f"[cmp] ERROR: engine dump not found: {e_path}", file=sys.stderr)
        return 1
    if not r_path.exists():
        print(f"[cmp] ERROR: reference dump not found: {r_path}", file=sys.stderr)
        return 1

    eng_tags = load_dump(e_path)
    ref_tags = load_dump(r_path)

    model_dir = Path(args.model_dir)
    layer_types = load_layer_types(model_dir)
    tokenizer = maybe_load_tokenizer(model_dir)
    label_map = load_label_map(Path(args.label_map))
    if label_map:
        print(f"[cmp] label-map: {args.label_map} ({len(label_map)} projections)")
    if layer_types:
        n_ssm = sum(1 for t in layer_types.values() if "linear" in t or "delta" in t.lower())
        n_fa = len(layer_types) - n_ssm
        print(f"[cmp] model: {model_dir}  layers={len(layer_types)} (SSM={n_ssm}, FA={n_fa})")
    else:
        print(f"[cmp] (no layer_types found in {model_dir}/config.json — annotations off)")
    print(f"[cmp] engine tags:    {list(eng_tags.keys())}")
    print(f"[cmp] reference tags: {list(ref_tags.keys())}")

    # Iterate intersection of tags in engine order.
    summary: list[tuple[str, str | None]] = []
    for tag in eng_tags:
        if tag not in ref_tags:
            print(f"[cmp] tag '{tag}' in engine but not in reference — skipping")
            continue
        first = compare_one_tag(tag, eng_tags[tag], ref_tags[tag], layer_types,
                                tokenizer, args.threshold, args.top, label_map)
        summary.append((tag, first))

    print()
    print("==================== SUMMARY ====================")
    for tag, first in summary:
        msg = first if first else f"all >= {args.threshold}"
        print(f"  {tag:<16} first divergence: {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
