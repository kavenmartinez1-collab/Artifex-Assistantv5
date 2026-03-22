"""Tests for core/knowledge.py — KnowledgeStore CRUD, ContextEngine lifecycle."""

import os
import json
import tempfile
import shutil
import pytest

from core.knowledge import (
    KnowledgeEntry, KnowledgeStore, ContextEngine,
    VOLATILE, SNAPSHOT, DISCOVERY, REFERENCE,
)


@pytest.fixture
def store_dir():
    d = tempfile.mkdtemp(prefix="kb_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestKnowledgeEntry:
    """Test KnowledgeEntry dataclass."""

    def test_auto_id(self):
        entry = KnowledgeEntry()
        assert len(entry.id) == 12

    def test_auto_timestamps(self):
        entry = KnowledgeEntry()
        assert entry.created != ""
        assert entry.updated != ""

    def test_custom_fields(self):
        entry = KnowledgeEntry(
            category="finding",
            tags=["ssh", "port"],
            tier1_line="SSH on port 22",
            lifecycle=DISCOVERY,
        )
        assert entry.category == "finding"
        assert "ssh" in entry.tags
        assert entry.lifecycle == DISCOVERY


class TestKnowledgeStore:
    """Test KnowledgeStore persistence and CRUD."""

    def test_add_and_get(self, store_dir):
        store = KnowledgeStore(store_dir)
        entry = KnowledgeEntry(
            category="note",
            tier1_line="Test entry",
            tier2_summary="A test note",
        )
        store.add(entry)

        retrieved = store.get(entry.id)
        assert retrieved is not None
        assert retrieved.tier1_line == "Test entry"

    def test_count(self, store_dir):
        store = KnowledgeStore(store_dir)
        assert store.count == 0

        for i in range(3):
            store.add(KnowledgeEntry(tier1_line=f"Entry {i}"))
        assert store.count == 3

    def test_remove(self, store_dir):
        store = KnowledgeStore(store_dir)
        entry = KnowledgeEntry(tier1_line="To be removed")
        store.add(entry)
        assert store.count == 1

        assert store.remove(entry.id) is True
        assert store.count == 0
        assert store.get(entry.id) is None

    def test_remove_nonexistent(self, store_dir):
        store = KnowledgeStore(store_dir)
        assert store.remove("nonexistent") is False

    def test_persistence(self, store_dir):
        """Entries survive store reload."""
        store1 = KnowledgeStore(store_dir)
        store1.add(KnowledgeEntry(tier1_line="Persistent entry"))
        assert store1.count == 1

        # Create a new store pointing to the same dir
        store2 = KnowledgeStore(store_dir)
        assert store2.count == 1

    def test_list_entries(self, store_dir):
        store = KnowledgeStore(store_dir)
        store.add(KnowledgeEntry(category="finding", tier1_line="F1"))
        store.add(KnowledgeEntry(category="note", tier1_line="N1"))
        store.add(KnowledgeEntry(category="finding", tier1_line="F2"))

        all_entries = store.list_entries()
        assert len(all_entries) == 3

        findings = store.list_entries("finding")
        assert len(findings) == 2

    def test_search(self, store_dir):
        store = KnowledgeStore(store_dir)
        store.add(KnowledgeEntry(
            tier1_line="SSH port 22 open",
            tags=["ssh", "port"],
        ))
        store.add(KnowledgeEntry(
            tier1_line="HTTP server on 80",
            tags=["http", "web"],
        ))

        results = store.search("ssh")
        assert len(results) > 0
        # SSH entry should score higher
        assert "ssh" in results[0][2].get("tier1", "").lower() or \
               any("ssh" in t for t in results[0][2].get("tags", []))

    def test_find_by_action_key(self, store_dir):
        store = KnowledgeStore(store_dir)
        e1 = KnowledgeEntry(
            action_key="scan:port22",
            tier1_line="SSH port 22 open",
            lifecycle=SNAPSHOT,
        )
        store.add(e1)

        found_id = store.find_by_key("scan:port22")
        assert found_id == e1.id

    def test_remove_by_action_key(self, store_dir):
        store = KnowledgeStore(store_dir)
        e1 = KnowledgeEntry(
            action_key="scan:port22",
            tier1_line="SSH port 22 open",
            lifecycle=SNAPSHOT,
        )
        store.add(e1)
        assert store.count == 1

        assert store.remove_by_key("scan:port22") is True
        assert store.count == 0


class TestContextEngine:
    """Test the ContextEngine deterministic state machine."""

    def test_classify_lifecycle(self, store_dir):
        store = KnowledgeStore(store_dir)
        engine = ContextEngine(store)

        # Directory listings are volatile
        lc = engine._classify_lifecycle("shell", "ls -la")
        assert lc == VOLATILE

        # File reads are snapshots
        lc = engine._classify_lifecycle("read_file", "config.py")
        assert lc == SNAPSHOT

        # Web searches are references
        lc = engine._classify_lifecycle("search", "python tutorial")
        assert lc == REFERENCE

    def test_make_key(self, store_dir):
        store = KnowledgeStore(store_dir)
        engine = ContextEngine(store)
        key = engine._make_key("shell", "nmap -sV 10.10.10.1")
        assert key != ""
        # Same action should produce same key
        key2 = engine._make_key("shell", "nmap -sV 10.10.10.1")
        assert key == key2
