"""Artifex installer: take stock of the PC, then install pinned pieces.

    python -m installer probe            what this PC has
    python -m installer plan             what "core chat" would install here
    python -m installer install          do it (asks first; --yes to skip)

On Windows, setup.cmd in the Artifex folder runs these too (double-click =
install; `setup pack voice` etc.).

See installer/recipes.py for the recipes and installer/fetch.py for the
download and unpack safety rules.
"""
import os


def hint(rest: str) -> str:
    """A command to show the user, the way they started the installer."""
    base = "setup" if os.environ.get("ARTIFEX_SETUP") == "1" else "python -m installer"
    return f"{base} {rest}"

