"""
PDF triage — byte-level risk scan with NO PDF parsing.

Reads raw bytes only (never renders or parses the document), so it is safe
to run on untrusted files. Reports the pdfid-style risk markers that matter
for the "is this just a document?" decision:

    /JavaScript /JS   — embedded scripts (the classic exploit vector)
    /OpenAction /AA   — actions that run automatically on open
    /Launch           — launches external programs
    /EmbeddedFile     — attached files inside the PDF
    /RichMedia /XFA   — legacy Flash / XML-forms exploit surfaces
    /Encrypt          — encrypted (limits what a raw scan can see)
    /ObjStm           — object streams (compressed containers a raw scan
                        cannot see inside — lowers confidence if present
                        alongside zero visible structure)

Name obfuscation via #xx hex escapes (e.g. /#4Aava#53cript) is normalized
before matching, same trick pdfid uses.

Usage:  python scripts/pdf_triage.py <file.pdf>
Exit codes: 0 = no active-content markers, 1 = markers found, 2 = not a PDF.
"""

import hashlib
import re
import sys


MARKERS = (
    ("/JavaScript", "embedded JavaScript"),
    ("/JS", "JavaScript entry (short form)"),
    ("/OpenAction", "auto-run action on open"),
    ("/AA", "additional (automatic) actions"),
    ("/Launch", "launches external program"),
    ("/EmbeddedFile", "file attachment"),
    ("/RichMedia", "embedded Flash/media"),
    ("/XFA", "XFA forms"),
    ("/AcroForm", "interactive form"),
    ("/URI", "external links (informational)"),
    ("/Encrypt", "encryption"),
    ("/ObjStm", "object streams (opaque to raw scan)"),
)

# /AA and /JS are short — require a non-name character after them so
# /AAX or /JSomething don't false-positive.
_SHORT = {"/JS", "/AA"}


def _normalize(data: bytes) -> bytes:
    """Resolve /#xx hex escapes in PDF names so obfuscation can't hide."""
    return re.sub(rb"#([0-9A-Fa-f]{2})",
                  lambda m: bytes([int(m.group(1), 16)]), data)


def triage(path: str) -> int:
    with open(path, "rb") as f:
        data = f.read()

    print(f"file:   {path}")
    print(f"size:   {len(data):,} bytes")
    print(f"sha256: {hashlib.sha256(data).hexdigest()}")

    head = data[:1024]
    if not head.lstrip().startswith(b"%PDF-"):
        print("VERDICT: NOT A PDF — header magic missing (first bytes: "
              f"{head[:16]!r})")
        return 2
    version = head.lstrip()[:8].decode("latin-1", "replace")
    print(f"header: {version}")
    if b"%%EOF" not in data[-2048:]:
        print("note:   no %%EOF near end (truncated download?)")

    norm = _normalize(data)
    findings = []
    for marker, meaning in MARKERS:
        pat = re.escape(marker.encode())
        if marker in _SHORT:
            pat += rb"(?![A-Za-z0-9])"
        n = len(re.findall(pat, norm))
        if n:
            findings.append((marker, meaning, n))

    active = [f for f in findings
              if f[0] in ("/JavaScript", "/JS", "/OpenAction", "/AA",
                          "/Launch", "/EmbeddedFile", "/RichMedia", "/XFA")]
    info = [f for f in findings if f not in active]

    print()
    if findings:
        for marker, meaning, n in findings:
            tag = "ACTIVE " if (marker, meaning, n) in active else "info   "
            print(f"  {tag}{marker:<14} x{n:<5} {meaning}")
    else:
        print("  (no markers at all)")

    print()
    if active:
        print("VERDICT: ACTIVE CONTENT PRESENT — do not open in a viewer; "
              "inspect the specific markers before proceeding.")
        return 1
    caveat = ""
    for marker, _, _ in info:
        if marker == "/ObjStm":
            caveat = (" (contains object streams a raw scan can't see "
                      "inside — pair this with an AV scan)")
        if marker == "/Encrypt":
            caveat = " (encrypted — raw scan confidence reduced)"
    print(f"VERDICT: no active-content markers{caveat}. "
          "Safe to text-extract in a subprocess.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(triage(sys.argv[1]))
