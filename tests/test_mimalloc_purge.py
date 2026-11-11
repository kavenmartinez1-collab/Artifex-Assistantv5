"""Importing core sets MIMALLOC_PURGE_DELAY=0 before torch can load.

Without it, PyTorch's bundled mimalloc kept an unloaded image model's
~18 GiB committed (its delayed release only runs on a later allocation),
and the Strata split then failed to start for lack of RAM.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code: str, env: dict) -> str:
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, text=True, check=True).stdout.strip()


def test_core_sets_purge_delay_before_torch():
    env = {k: v for k, v in os.environ.items() if k != "MIMALLOC_PURGE_DELAY"}
    out = _run("import sys, os, core; "
               "print(os.environ.get('MIMALLOC_PURGE_DELAY'), 'torch' in sys.modules)", env)
    assert out == "0 False"


def test_existing_value_wins():
    env = dict(os.environ, MIMALLOC_PURGE_DELAY="25")
    assert _run("import os, core; print(os.environ['MIMALLOC_PURGE_DELAY'])", env) == "25"
