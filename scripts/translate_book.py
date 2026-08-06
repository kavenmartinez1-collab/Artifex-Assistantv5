"""
Book translation pipeline — PDF -> English Markdown via the local
llama.cpp backend, fully offline. Generic over source books:

    extract    --pdf FILE -> work/pages.jsonl (+ structure report)
               Auto-detects and strips running heads (lines repeating at
               page tops document-wide), repairs end-of-line hyphenation,
               keeps page provenance for citation-accurate review.
    translate  pages A..B -> chunked <src-lang>->English via LlamaCppEngine
               with tail-overlap context; writes chapters/<name>.md
               (+ .meta.json with timing/token stats).
    stitch     chapters/*.md -> book.md

Usage (venv python, repo root):
    python scripts/translate_book.py extract --pdf path/to/book.pdf
    python scripts/translate_book.py translate --pages 129-150 --name ch5
    python scripts/translate_book.py stitch

Work dir: output/translation (override with BOOK_WORK). Sampling is
explicit (core.sampling contract): balanced chain at temp 0.25, thinking
OFF — translation is a fidelity task, not a reasoning task.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

# Runnable as `python scripts/translate_book.py` — put the repo root on the
# path so `core.*` imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORK = os.environ.get("BOOK_WORK", os.path.join("output", "translation"))
PAGES_JSONL = os.path.join(WORK, "pages.jsonl")
CHAPTER_DIR = os.path.join(WORK, "chapters")

# A top-of-page line whose digit-stripped form recurs on at least this
# fraction of pages is a running head (title/author/folio furniture).
RUNNING_HEAD_MIN_FRACTION = 0.08
_HEAD_ZONE_LINES = 3  # only the first N non-empty lines of a page qualify

CHUNK_TARGET_CHARS = 4800   # ~1200 tokens of Spanish
CHUNK_MAX_CHARS = 6400
OVERLAP_TAIL_CHARS = 400    # previous-chunk tail shown for continuity

SYSTEM_PROMPT_TEMPLATE = """You are a professional literary and academic translator.
Translate the user's {src_lang} text into clear, faithful English.

Rules:
- PRESERVE all personal names, place names, dates, document titles, and
  archival citations EXACTLY as written (never anglicize names).
- Historical honorifics and titles without a clean English equivalent keep
  the original term on first use with a brief English gloss in brackets,
  then the original term alone thereafter.
- Preserve paragraph breaks. Do not merge or split paragraphs.
- Footnote markers attached to words stay attached.
- Figure/table references become their English form ("see Figure 25").
- Translate completely and literally where prose allows; favor fidelity
  over elegance when they conflict.
- Output ONLY the translation — no preamble, no notes, no commentary."""


# ═══════════════════════════════════════════════════════════════════════════
# extract
# ═══════════════════════════════════════════════════════════════════════════

def _head_key(line: str) -> str:
    """Normalized form for running-head detection: digits stripped (folio
    numbers ride along with the head), whitespace collapsed, lowercased."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "", line)).strip().lower()


def _detect_running_heads(page_texts: list[str]) -> set[str]:
    """Digit-stripped top-zone lines recurring across many pages."""
    counts = Counter()
    for text in page_texts:
        seen_nonempty = 0
        page_keys = set()
        for line in text.splitlines():
            if not line.strip():
                continue
            seen_nonempty += 1
            key = _head_key(line)
            if key and len(key) > 3:
                page_keys.add(key)
            if seen_nonempty >= _HEAD_ZONE_LINES:
                break
        counts.update(page_keys)
    threshold = max(3, int(len(page_texts) * RUNNING_HEAD_MIN_FRACTION))
    return {key for key, n in counts.items() if n >= threshold}


def _clean_page(text: str, running_heads: set[str]) -> str:
    lines = [l.rstrip() for l in text.splitlines()]
    out = []
    seen_nonempty = 0
    for line in lines:
        if line.strip() and seen_nonempty < _HEAD_ZONE_LINES:
            seen_nonempty += 1
            # Bare folio number, or a line matching a detected running head.
            if re.fullmatch(r"\s*\d{1,4}\s*", line):
                continue
            if _head_key(line) in running_heads:
                continue
        out.append(line)
    text = "\n".join(out)
    # PDF extraction hyphens at line ends are soft breaks: join
    # lowercase-to-lowercase splits ("pe-\nqueno" -> "pequeno"); real
    # compound hyphens (capitalized/numeric neighbors) are left alone.
    text = re.sub(r"([^\W\dA-Z])-\n([^\W\dA-Z])", r"\1\2", text, flags=re.UNICODE)
    return text.strip()


def cmd_extract(pdf: str):
    from pypdf import PdfReader
    os.makedirs(WORK, exist_ok=True)
    reader = PdfReader(pdf)
    n = len(reader.pages)
    raw_pages = []
    for i, page in enumerate(reader.pages):
        raw_pages.append(page.extract_text() or "")
        if (i + 1) % 100 == 0:
            print(f"  extracted {i + 1}/{n} pages", flush=True)

    running_heads = _detect_running_heads(raw_pages)
    if running_heads:
        print("detected running heads (auto):")
        for key in sorted(running_heads):
            print(f"  {key[:76]!r}")

    heads = []
    with open(PAGES_JSONL, "w", encoding="utf-8") as f:
        for i, raw in enumerate(raw_pages):
            clean = _clean_page(raw, running_heads)
            f.write(json.dumps({"page": i + 1, "chars": len(clean),
                                "text": clean, "source": os.path.basename(pdf)},
                               ensure_ascii=False) + "\n")
            # Structure report: numbered section titles near page tops.
            for line in clean.splitlines()[:4]:
                s = line.strip()
                if re.match(r"^(\d+(\.\d+)*\.?)\s+[A-ZÀ-Ý]", s) and len(s) < 90:
                    heads.append((i + 1, s))
    print(f"extracted {n} pages -> {PAGES_JSONL}")
    print("\nheading candidates (page: title):")
    for pg, title in heads:
        print(f"  {pg:>4}: {title}")


# ═══════════════════════════════════════════════════════════════════════════
# translate
# ═══════════════════════════════════════════════════════════════════════════

def _load_pages(a: int, b: int) -> list[dict]:
    rows = []
    with open(PAGES_JSONL, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if a <= row["page"] <= b:
                rows.append(row)
    return rows


def _chunk(pages: list[dict]) -> list[dict]:
    """Greedy paragraph-aware chunks with page provenance."""
    chunks = []
    cur, cur_pages, cur_len = [], set(), 0
    for row in pages:
        for para in re.split(r"\n\s*\n", row["text"]):
            para = para.strip()
            if not para:
                continue
            plen = len(para)
            if cur and cur_len + plen > CHUNK_TARGET_CHARS and cur_len > 0:
                chunks.append({"text": "\n\n".join(cur),
                               "pages": (min(cur_pages), max(cur_pages))})
                cur, cur_pages, cur_len = [], set(), 0
            # Hard split pathological paragraphs.
            while plen > CHUNK_MAX_CHARS:
                cut = para.rfind(". ", 0, CHUNK_MAX_CHARS)
                cut = cut + 1 if cut > 0 else CHUNK_MAX_CHARS
                chunks.append({"text": para[:cut].strip(),
                               "pages": (row["page"], row["page"])})
                para = para[cut:].strip()
                plen = len(para)
            cur.append(para)
            cur_pages.add(row["page"])
            cur_len += plen + 2
    if cur:
        chunks.append({"text": "\n\n".join(cur),
                       "pages": (min(cur_pages), max(cur_pages))})
    return chunks


def cmd_translate(pages_arg: str, name: str, src_lang: str = "Spanish"):
    from core.engine_llama_cpp import LlamaCppEngine
    from core.config import get_llama_cpp_model_config
    from core.sampling import get_preset

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(src_lang=src_lang)
    a, b = (int(x) for x in pages_arg.split("-"))
    pages = _load_pages(a, b)
    if not pages:
        print(f"no pages in range {a}-{b}; run extract first")
        return 2
    chunks = _chunk(pages)
    total_chars = sum(len(c["text"]) for c in chunks)
    print(f"pages {a}-{b}: {len(pages)} pages, {len(chunks)} chunks, "
          f"{total_chars/1e3:.0f}K chars", flush=True)

    model = os.environ.get("BOOK_MODEL", "qwen3.6-35b-a3b")
    engine = LlamaCppEngine(model, get_llama_cpp_model_config(model))
    engine.load(status_callback=lambda s: print(f"  [engine] {s}", flush=True))

    sampling = get_preset("balanced", temperature=0.25, seed=42)
    os.makedirs(CHAPTER_DIR, exist_ok=True)
    out_md = os.path.join(CHAPTER_DIR, f"{name}.md")
    meta_path = os.path.join(CHAPTER_DIR, f"{name}.meta.json")

    results = []
    stats = []
    prev_tail = ""
    t_start = time.monotonic()
    for idx, chunk in enumerate(chunks, 1):
        user = chunk["text"]
        if prev_tail:
            user = (f"[Context — end of previous passage, already translated, "
                    f"for continuity only. Do NOT retranslate it.]\n"
                    f"...{prev_tail}\n\n[Translate the following]\n{user}")
        t0 = time.monotonic()
        out = engine.generate_streaming(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user}],
            max_tokens=3072, temperature=0.25,
            enable_thinking=False, sampling=sampling)
        dt = time.monotonic() - t0
        gen = dict(engine._last_gen_stats)
        results.append(out.strip())
        prev_tail = chunk["text"][-OVERLAP_TAIL_CHARS:]
        stats.append({"chunk": idx, "pages": chunk["pages"],
                      "src_chars": len(chunk["text"]),
                      "out_chars": len(out),
                      "seconds": round(dt, 1),
                      "completion_tokens": gen.get("completion_tokens"),
                      "tok_per_s": gen.get("predicted_per_second"),
                      "finish": gen.get("finish_reason")})
        done_chars = sum(s["src_chars"] for s in stats)
        rate = done_chars / max(time.monotonic() - t_start, 1)
        eta = (total_chars - done_chars) / max(rate, 1)
        print(f"  chunk {idx}/{len(chunks)} p{chunk['pages'][0]}-{chunk['pages'][1]} "
              f"{dt:.0f}s {gen.get('completion_tokens')} tok "
              f"({gen.get('predicted_per_second') or 0:.1f} tok/s) "
              f"ETA {eta/60:.0f}m", flush=True)

    body = "\n\n".join(results)
    src_name = pages[0].get("source", "source PDF")
    header = (f"<!-- Translated by Artifex ({model}) from pages {a}-{b} of\n"
              f"     {src_name} — machine translation, unreviewed. -->\n\n")
    with open(out_md, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + body + "\n")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"pages": [a, b], "chunks": stats,
                   "wall_s": round(time.monotonic() - t_start, 1)}, f, indent=1)
    wall = time.monotonic() - t_start
    print(f"\nwrote {out_md} ({len(body)/1e3:.0f}K chars) in {wall/60:.1f} min")
    print(f"meta -> {meta_path}")
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# stitch
# ═══════════════════════════════════════════════════════════════════════════

def cmd_stitch():
    parts = sorted(
        f for f in os.listdir(CHAPTER_DIR) if f.endswith(".md"))
    out = os.path.join(WORK, "book.md")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for part in parts:
            with open(os.path.join(CHAPTER_DIR, part), encoding="utf-8") as p:
                f.write(p.read().rstrip() + "\n\n---\n\n")
    print(f"stitched {len(parts)} parts -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--pdf", required=True, help="source PDF path")
    tr = sub.add_parser("translate")
    tr.add_argument("--pages", required=True, help="e.g. 129-150")
    tr.add_argument("--name", required=True, help="output chapter name")
    tr.add_argument("--src-lang", default="Spanish", help="source language")
    sub.add_parser("stitch")
    args = ap.parse_args()
    if args.cmd == "extract":
        return cmd_extract(args.pdf)
    if args.cmd == "translate":
        return cmd_translate(args.pages, args.name, args.src_lang)
    if args.cmd == "stitch":
        return cmd_stitch()


if __name__ == "__main__":
    sys.exit(main())
