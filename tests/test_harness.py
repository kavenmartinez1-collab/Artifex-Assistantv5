"""
Tests for core.harness — harness ingestion (Agent / Harness Mode).

Covers detection across multiple agent formats, the non-destructive .artifex
bundle, @import inlining with path-traversal protection, frontmatter handling,
hash-dedup, idempotency, include filtering, and the prompt-injection seam.
"""

import json
import os
import textwrap

import pytest

from core import harness
from core.prompts import build_assistant_prompt


def _w(root, rel, content):
    path = os.path.join(root, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(content).lstrip("\n"))
    return path


@pytest.fixture
def multi_harness(tmp_dir):
    """A folder seeded with several different agents' context files."""
    _w(tmp_dir, "CLAUDE.md", """
        # Claude Rules
        Indent with 4 spaces. See @docs/deploy.md for release steps.
        Escape attempt: @../secret.md
    """)
    _w(tmp_dir, "docs/deploy.md", "Run `make ship`. Never on Friday.")
    _w(os.path.dirname(tmp_dir), "secret.md", "TOP SECRET OUTSIDE ROOT")
    _w(tmp_dir, ".claude/agents/reviewer.md", "You are a strict reviewer.")
    _w(tmp_dir, ".claude/settings.json", '{"model": "claude-opus"}')
    _w(tmp_dir, "AGENTS.md", "# Universal\nAlways write tests.")
    _w(tmp_dir, ".cursor/rules/ts.mdc", """
        ---
        description: TS style
        globs: **/*.ts
        alwaysApply: true
        ---
        Prefer const. Ban any.
    """)
    _w(tmp_dir, ".github/copilot-instructions.md", "Keep PRs small.")
    _w(tmp_dir, ".windsurfrules", "House style applies.")
    # Noise that must be pruned and never detected:
    _w(tmp_dir, "node_modules/pkg/CLAUDE.md", "PRUNE ME")
    return tmp_dir


# ── detection ──────────────────────────────────────────────────────────────

def test_detect_finds_all_harnesses(multi_harness):
    report = harness.detect(multi_harness)
    ids = set(report.tool_ids)
    assert {"claude", "agents_md", "cursor", "copilot", "windsurf"} <= ids


def test_detect_prunes_heavy_dirs(multi_harness):
    report = harness.detect(multi_harness)
    all_rel = [f.relpath for h in report.hits for f in h.files]
    assert not any("node_modules" in r for r in all_rel)


def test_detect_empty_folder(tmp_dir):
    report = harness.detect(tmp_dir)
    assert report.is_empty
    assert report.tool_ids == []


def test_detect_orders_by_priority(multi_harness):
    report = harness.detect(multi_harness)
    # claude (10) before cursor (20) before windsurf (30)
    order = report.tool_ids
    assert order.index("claude") < order.index("cursor") < order.index("windsurf")


# ── adopt: bundle layout ───────────────────────────────────────────────────

def test_adopt_writes_bundle(multi_harness):
    harness.adopt(multi_harness)
    b = os.path.join(multi_harness, ".artifex")
    assert os.path.isfile(os.path.join(b, "ARTIFEX.md"))
    assert os.path.isfile(os.path.join(b, "ARTIFEX.local.md"))
    assert os.path.isfile(os.path.join(b, "manifest.json"))
    assert os.path.isfile(os.path.join(b, ".gitignore"))
    # verbatim source copies, namespaced by tool
    assert os.path.isfile(os.path.join(b, "sources", "claude", "CLAUDE.md"))
    assert os.path.isfile(os.path.join(b, "sources", "claude", ".claude", "settings.json"))


def test_manifest_records_tools_and_hashes(multi_harness):
    harness.adopt(multi_harness)
    man = harness.read_manifest(multi_harness)
    assert man["version"] == 1
    tool_ids = {t["id"] for t in man["tools"]}
    assert "claude" in tool_ids
    claude = next(t for t in man["tools"] if t["id"] == "claude")
    assert all(len(f["sha256"]) == 64 for f in claude["files"])


# ── injection content ──────────────────────────────────────────────────────

def test_injection_contains_instructions(multi_harness):
    harness.adopt(multi_harness)
    inj = harness.load_injection(multi_harness)
    assert "Claude Rules" in inj
    assert "Always write tests." in inj          # AGENTS.md
    assert "Prefer const. Ban any." in inj       # cursor body


def test_import_is_inlined(multi_harness):
    harness.adopt(multi_harness)
    inj = harness.load_injection(multi_harness)
    assert "make ship" in inj                    # @docs/deploy.md transcluded


def test_import_cannot_escape_root(multi_harness):
    harness.adopt(multi_harness)
    inj = harness.load_injection(multi_harness)
    assert "TOP SECRET OUTSIDE ROOT" not in inj


def test_frontmatter_body_stripped_metadata_surfaced(multi_harness):
    harness.adopt(multi_harness)
    inj = harness.load_injection(multi_harness)
    assert "Prefer const. Ban any." in inj             # body kept
    assert "---\ndescription: TS style" not in inj     # raw fence gone
    assert "globs: **/*.ts" in inj                      # surfaced in header


def test_settings_copied_but_not_injected(multi_harness):
    harness.adopt(multi_harness)
    inj = harness.load_injection(multi_harness)
    assert '"model": "claude-opus"' not in inj


def test_identical_files_deduped(tmp_dir):
    # Same content as AGENTS.md and AGENT.md → folded once.
    _w(tmp_dir, "AGENTS.md", "Shared guidance line.")
    _w(tmp_dir, "AGENT.md", "Shared guidance line.")
    harness.adopt(tmp_dir)
    inj = harness.load_injection(tmp_dir)
    assert inj.count("Shared guidance line.") == 1


# ── idempotency / non-destructive ──────────────────────────────────────────

def test_adopt_is_idempotent(multi_harness):
    harness.adopt(multi_harness)
    first = harness.load_injection(multi_harness)
    harness.adopt(multi_harness)
    second = harness.load_injection(multi_harness)
    # Only the volatile timestamp line differs.
    norm = lambda s: "\n".join(l for l in s.splitlines() if "Absorbed from" not in l)
    assert norm(first) == norm(second)


def test_local_notes_preserved_and_injected(multi_harness):
    harness.adopt(multi_harness)
    local = os.path.join(multi_harness, ".artifex", "ARTIFEX.local.md")
    with open(local, "a", encoding="utf-8") as f:
        f.write("\nPROJECT GOTCHA: db migrations are manual.\n")
    harness.adopt(multi_harness)  # re-adopt must not clobber
    with open(local, encoding="utf-8") as f:
        assert "PROJECT GOTCHA" in f.read()
    assert "PROJECT GOTCHA" in harness.load_injection(multi_harness)


def test_source_change_updates_manifest_hash(multi_harness):
    harness.adopt(multi_harness)
    before = harness.read_manifest(multi_harness)
    sha_before = next(f["sha256"] for t in before["tools"] if t["id"] == "claude"
                      for f in t["files"] if f["path"] == "CLAUDE.md")
    _w(multi_harness, "CLAUDE.md", "# Claude Rules\nTotally rewritten.")
    harness.adopt(multi_harness)
    after = harness.read_manifest(multi_harness)
    sha_after = next(f["sha256"] for t in after["tools"] if t["id"] == "claude"
                     for f in t["files"] if f["path"] == "CLAUDE.md")
    assert sha_before != sha_after


def test_bundle_excluded_from_redetect(multi_harness):
    harness.adopt(multi_harness)
    report = harness.detect(multi_harness)
    all_rel = [f.relpath for h in report.hits for f in h.files]
    assert not any(r.startswith(".artifex/") for r in all_rel)


# ── include filtering ──────────────────────────────────────────────────────

def test_include_filter_limits_injection_not_copies(multi_harness):
    harness.adopt(multi_harness, include={"claude"})
    inj = harness.load_injection(multi_harness)
    assert "Claude Rules" in inj
    assert "Prefer const. Ban any." not in inj   # cursor excluded from injection
    # …but cursor sources are still copied for provenance
    assert os.path.isfile(os.path.join(
        multi_harness, ".artifex", "sources", "cursor", ".cursor", "rules", "ts.mdc"))


# ── budget / load ──────────────────────────────────────────────────────────

def test_load_injection_respects_budget(tmp_dir):
    _w(tmp_dir, "CLAUDE.md", "# Big\n" + ("filler word " * 4000))
    harness.adopt(tmp_dir)
    inj = harness.load_injection(tmp_dir, token_budget=100)
    assert len(inj) <= 100 * 4 + 80
    assert "truncated" in inj


def test_load_injection_empty_when_not_adopted(tmp_dir):
    assert harness.load_injection(tmp_dir) == ""
    assert harness.is_adopted(tmp_dir) is False


# ── prompt seam ────────────────────────────────────────────────────────────

def test_prompt_includes_agent_context_block():
    p = build_assistant_prompt("sysinfo", "/cwd", agent_context="ABSORBED-XYZ")
    assert "AGENT CONTEXT" in p
    assert "ABSORBED-XYZ" in p


def test_prompt_omits_block_when_empty():
    p = build_assistant_prompt("sysinfo", "/cwd", agent_context="")
    assert "AGENT CONTEXT" not in p
