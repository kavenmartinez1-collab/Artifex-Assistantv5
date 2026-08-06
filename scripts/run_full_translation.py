"""
Full-book translation driver — runs translate_book.py over the whole page
range in fixed segments, resumable (segments with existing output are
skipped, so a crash or reboot just needs a relaunch).

    python scripts/run_full_translation.py

Segment boundaries fall mid-chapter sometimes; that's fine — chunking is
paragraph-aware with tail-overlap context, and stitch concatenates the
sorted segment files back into reading order.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.translate_book import (  # noqa: E402
    CHAPTER_DIR, PAGES_JSONL, cmd_stitch, cmd_translate,
)

SEGMENT = int(os.environ.get("BOOK_SEGMENT", "40"))
SRC_LANG = os.environ.get("BOOK_SRC_LANG", "Spanish")


def _total_pages() -> int:
    n = 0
    with open(PAGES_JSONL, encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def main():
    t0 = time.monotonic()
    total = _total_pages()
    print(f"[driver] {total} pages, segment size {SEGMENT}, "
          f"source language {SRC_LANG}", flush=True)
    segments = []
    start = 1
    while start <= total:
        end = min(start + SEGMENT - 1, total)
        segments.append((start, end))
        start = end + 1

    for i, (a, b) in enumerate(segments, 1):
        name = f"seg{i:02d}_p{a:03d}-{b:03d}"
        out = os.path.join(CHAPTER_DIR, f"{name}.md")
        if os.path.isfile(out):
            print(f"[driver] {name} exists — skipping", flush=True)
            continue
        print(f"[driver] === segment {i}/{len(segments)}: pages {a}-{b} ===",
              flush=True)
        rc = cmd_translate(f"{a}-{b}", name, SRC_LANG)
        if rc not in (0, None):
            print(f"[driver] segment {name} failed (rc={rc}) — aborting so "
                  f"a relaunch can resume here", flush=True)
            return 1

    cmd_stitch()
    print(f"[driver] FULL BOOK DONE in {(time.monotonic() - t0)/60:.0f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
