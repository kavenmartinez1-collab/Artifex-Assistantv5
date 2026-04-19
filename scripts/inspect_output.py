"""inspect_output.py — pretty-print the engine's last autotest result.

Reads webgpu/debug-output.json (the file the engine POSTs after each autotest)
and prints the generated text in an ASCII-safe form, then flags any non-ASCII
characters with their position and hex code so we can see exactly where the
multilingual leak appears.

Usage:
    ./venv/Scripts/python.exe scripts/inspect_output.py
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

DEFAULT_PATH = Path("webgpu/debug-output.json")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        print(f"[inspect] no file at {path}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    text = data.get("output") or ""
    tokens = data.get("tokens", "?")
    tps = data.get("tokPerSec", 0.0)
    stop = data.get("stopReason", "?")
    elapsed = data.get("elapsedMs", "?")
    print(f"prompt:    {data.get('prompt', '?')!r}")
    print(f"tokens:    {tokens}  tok/s: {tps:.1f}  stop: {stop}  elapsed: {elapsed}ms")
    print("--- OUTPUT (ASCII-safe) ---")
    print(text.encode("ascii", "backslashreplace").decode("ascii"))
    print("--- END ---")
    non_ascii = [(i, c) for i, c in enumerate(text) if ord(c) > 127]
    if non_ascii:
        print(f"NON-ASCII characters: {len(non_ascii)} found in output of length {len(text)}")
        for i, c in non_ascii[:30]:
            esc = c.encode("unicode_escape").decode("ascii")
            try:
                name = unicodedata.name(c)
            except ValueError:
                name = "?"
            print(f"  pos={i:4d} {esc:<10} {name}")
        if len(non_ascii) > 30:
            print(f"  ... and {len(non_ascii) - 30} more")
    else:
        print("No non-ASCII characters in output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
