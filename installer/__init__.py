"""Artifex installer: take stock of the PC, then install pinned pieces.

    python -m installer probe            what this PC has
    python -m installer plan             what "core chat" would install here
    python -m installer install          do it (asks first; --yes to skip)

See installer/recipes.py for the recipes and installer/fetch.py for the
download and unpack safety rules.
"""
