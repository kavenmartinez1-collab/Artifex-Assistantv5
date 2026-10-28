"""trust_remote_code stays off unless the user turns it on.

With it on, loading a model imports Python files from the model's own
Hugging Face repo: whoever controls that repo runs code on this PC. Every
model Artifex uses has a built-in transformers class, so it is off, and
only ARTIFEX_TRUST_REMOTE_CODE=1 turns it on.
"""

import importlib
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_no_code_hardcodes_remote_code_on():
    hits = []
    for top in ("core", "api", "tools", "ui", "scripts", "installer"):
        for base, _, files in os.walk(os.path.join(REPO, top)):
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(base, name)
                text = open(path, encoding="utf-8", errors="ignore").read()
                if re.search(r"""trust_remote_code["']?\s*[=:]\s*True""", text):
                    hits.append(os.path.relpath(path, REPO))
    assert not hits, f"trust_remote_code hard-coded on in: {hits}"


def test_off_by_default_and_on_only_when_asked(monkeypatch):
    import core.config as config
    monkeypatch.delenv("ARTIFEX_TRUST_REMOTE_CODE", raising=False)
    assert importlib.reload(config).TRUST_REMOTE_CODE is False
    for value in ("true", "yes", "0", " 2"):
        monkeypatch.setenv("ARTIFEX_TRUST_REMOTE_CODE", value)
        assert importlib.reload(config).TRUST_REMOTE_CODE is False, value
    monkeypatch.setenv("ARTIFEX_TRUST_REMOTE_CODE", "1")
    assert importlib.reload(config).TRUST_REMOTE_CODE is True
    monkeypatch.delenv("ARTIFEX_TRUST_REMOTE_CODE")
    importlib.reload(config)
