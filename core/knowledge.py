"""
Artifex-Lite v2 — Knowledge Tree & Context Engine.

Architecture:
  KnowledgeEntry   — one atomic knowledge item with lifecycle metadata
  KnowledgeStore   — manages one directory of entries + index (persistence)
  ContextEngine    — deterministic state machine for tool output processing
                     (lifecycle, staleness, upsert, loop detection, mutation tracking)
  KnowledgeManager — high-level bridge: session + reference stores, workspace, engine
"""

import json
import os
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Tuple

from core.config import KNOWLEDGE_DIR, KNOWLEDGE_REFERENCE_DIR, PHASES


# ===== CONSTANTS =====

CATEGORIES = ("finding", "credential", "topology", "technique", "note", "reference")
SOURCES = ("auto:command", "auto:ai", "manual")

# Lifecycle types — determine how entries are stored, updated, and evicted
VOLATILE = "volatile"      # Dir listings, process lists — replaced on re-run, staled on mutation
SNAPSHOT = "snapshot"      # File reads, meaningful commands — replaced on re-run
DISCOVERY = "discovery"    # CVEs, creds, ports — never auto-expire
REFERENCE = "reference"    # Web results, docs — long-lived, lower priority

# Budget allocation per lifecycle tier (fraction of total)
LIFECYCLE_BUDGETS = {
    DISCOVERY: 0.40,   # Always first: CVEs, creds, ports
    SNAPSHOT:  0.30,   # Recent file reads, meaningful commands
    VOLATILE:  0.15,   # Dir listings — only if fresh
    REFERENCE: 0.15,   # Web results, docs
}

CATEGORY_WEIGHTS = {
    "credential": 1.5,
    "finding": 1.0,
    "topology": 1.0,
    "technique": 0.8,
    "reference": 0.7,
    "note": 0.5,
}


# ===== ENTRY =====

@dataclass
class KnowledgeEntry:
    """One atomic knowledge item with lifecycle metadata."""
    id: str = ""
    category: str = "note"
    tags: List[str] = field(default_factory=list)
    phase: str = ""
    source: str = "manual"
    tier1_line: str = ""
    tier2_summary: str = ""
    tier3_full: str = ""
    created: str = ""
    updated: str = ""
    # Context engine fields
    lifecycle: str = SNAPSHOT       # volatile / snapshot / discovery / reference
    action_key: str = ""            # deterministic key for upsert
    workspace_gen: int = 0          # workspace generation when captured
    stale: bool = False             # marked stale when workspace mutates

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()
        if not self.created:
            self.created = now
        if not self.updated:
            self.updated = now


# ===== STORE =====

class KnowledgeStore:
    """Manages one directory of knowledge entries + a searchable index."""

    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        self.entries_dir = os.path.join(store_dir, "entries")
        self.index_file = os.path.join(store_dir, "_index.json")
        self._index: Dict[str, dict] = {}
        self._ensure_dirs()
        self._load_index()

    def _ensure_dirs(self):
        os.makedirs(self.entries_dir, exist_ok=True)

    def _load_index(self):
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._index = {}
                self.rebuild_index()
        else:
            self._index = {}

    def _save_index(self):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=1)

    def _entry_path(self, entry_id: str) -> str:
        return os.path.join(self.entries_dir, f"{entry_id}.json")

    def _save_entry(self, entry: KnowledgeEntry):
        with open(self._entry_path(entry.id), "w", encoding="utf-8") as f:
            json.dump(asdict(entry), f, indent=2)

    def _update_index_for(self, entry: KnowledgeEntry):
        self._index[entry.id] = {
            "tier1": entry.tier1_line,
            "tags": entry.tags,
            "category": entry.category,
            "phase": entry.phase,
            "created": entry.created,
            "updated": entry.updated,
            "lifecycle": entry.lifecycle,
            "action_key": entry.action_key,
            "workspace_gen": entry.workspace_gen,
            "stale": entry.stale,
        }

    # --- Public API ---

    def add(self, entry: KnowledgeEntry) -> str:
        entry.updated = datetime.now().isoformat()
        self._save_entry(entry)
        self._update_index_for(entry)
        self._save_index()
        return entry.id

    def get(self, entry_id: str) -> Optional[KnowledgeEntry]:
        path = self._entry_path(entry_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return KnowledgeEntry(**data)
        except (json.JSONDecodeError, TypeError, OSError):
            return None

    def remove(self, entry_id: str) -> bool:
        path = self._entry_path(entry_id)
        if os.path.exists(path):
            os.remove(path)
        if entry_id in self._index:
            del self._index[entry_id]
            self._save_index()
            return True
        return False

    def remove_by_key(self, action_key: str) -> bool:
        """Remove entry matching an action_key. Returns True if found."""
        for eid, meta in list(self._index.items()):
            if meta.get("action_key") == action_key:
                return self.remove(eid)
        return False

    def find_by_key(self, action_key: str) -> Optional[str]:
        """Find entry ID by action_key."""
        for eid, meta in self._index.items():
            if meta.get("action_key") == action_key:
                return eid
        return None

    def mark_volatile_stale(self, current_gen: int):
        """Mark all volatile entries from older generations as stale."""
        changed = False
        for eid, meta in self._index.items():
            if (meta.get("lifecycle") == VOLATILE
                    and meta.get("workspace_gen", 0) < current_gen
                    and not meta.get("stale", False)):
                meta["stale"] = True
                changed = True
        if changed:
            self._save_index()

    def list_entries(self, category: str = None) -> List[dict]:
        results = []
        for eid, meta in self._index.items():
            if category and meta.get("category") != category:
                continue
            results.append({"id": eid, **meta})
        return sorted(results, key=lambda x: x.get("updated", ""), reverse=True)

    def search(self, query: str, category: str = None, phase: str = None,
               max_results: int = 10) -> List[Tuple[float, str, dict]]:
        query_lower = query.lower()
        query_terms = query_lower.split()
        now_ts = time.time()
        results = []

        for eid, meta in self._index.items():
            if category and meta.get("category") != category:
                continue

            score = 0.0
            entry_tags = [t.lower() for t in meta.get("tags", [])]
            for term in query_terms:
                if term in entry_tags:
                    score += 2.0
                elif any(term in tag for tag in entry_tags):
                    score += 1.0

            tier1_lower = meta.get("tier1", "").lower()
            for term in query_terms:
                if term in tier1_lower:
                    score += 1.0

            if score == 0:
                continue

            if phase and meta.get("phase") == phase:
                score += 1.5

            try:
                updated_dt = datetime.fromisoformat(meta.get("updated", ""))
                hours_ago = (now_ts - updated_dt.timestamp()) / 3600
                recency = max(0, 2.0 - (hours_ago * 0.1))
                score += recency
            except (ValueError, TypeError, OSError):
                pass

            # Stale entries get penalized
            if meta.get("stale", False):
                score *= 0.3

            cat_weight = CATEGORY_WEIGHTS.get(meta.get("category", "note"), 0.5)
            score *= cat_weight

            results.append((score, eid, meta))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:max_results]

    def render_for_prompt(self, token_budget: int = 400, phase: str = None) -> str:
        """
        Lifecycle-aware prompt rendering with tiered budget allocation.
        DISCOVERY entries get first priority, then SNAPSHOT, VOLATILE (fresh only), REFERENCE.
        """
        char_budget = token_budget * 4
        now_ts = time.time()

        # Bucket entries by lifecycle
        buckets: Dict[str, List[Tuple[str, dict]]] = {
            DISCOVERY: [], SNAPSHOT: [], VOLATILE: [], REFERENCE: [],
        }
        for eid, meta in self._index.items():
            lc = meta.get("lifecycle", SNAPSHOT)
            # Skip stale volatile entries entirely
            if lc == VOLATILE and meta.get("stale", False):
                continue
            if lc in buckets:
                buckets[lc].append((eid, meta))

        # Sort each bucket: phase match first, then recency
        def _sort_key(item):
            _, meta = item
            phase_bonus = 50 if (phase and meta.get("phase") == phase) else 0
            try:
                age = now_ts - datetime.fromisoformat(meta.get("updated", "")).timestamp()
            except (ValueError, TypeError, OSError):
                age = 999999
            return -(phase_bonus - age / 3600)

        for bucket in buckets.values():
            bucket.sort(key=_sort_key)

        # Render with per-tier budgets
        lines = []
        total_used = 0

        for lifecycle in (DISCOVERY, SNAPSHOT, VOLATILE, REFERENCE):
            tier_budget = int(char_budget * LIFECYCLE_BUDGETS[lifecycle])
            tier_used = 0
            for eid, meta in buckets[lifecycle]:
                tier1 = meta.get("tier1", "").strip()
                if not tier1:
                    continue
                cat_prefix = meta.get("category", "note")[0].upper()
                line = f"[KB:{cat_prefix}] {tier1}"
                line_len = len(line) + 1
                if tier_used + line_len > tier_budget:
                    break
                if total_used + line_len > char_budget:
                    break
                lines.append(line)
                total_used += line_len
                tier_used += line_len

        # Remainder summary
        shown = len(lines)
        total_entries = sum(len(b) for b in buckets.values())
        remaining = total_entries - shown
        if remaining > 0:
            cats = {}
            for bucket in buckets.values():
                for _, meta in bucket[shown:]:
                    c = meta.get("category", "note")
                    cats[c] = cats.get(c, 0) + 1
            summary_parts = [f"{v} {k}s" for k, v in cats.items() if v > 0]
            if summary_parts:
                lines.append(f"[+{remaining} more: {', '.join(summary_parts)}]")

        return "\n".join(lines)

    def rebuild_index(self):
        self._index = {}
        if not os.path.isdir(self.entries_dir):
            self._save_index()
            return
        for fname in os.listdir(self.entries_dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.entries_dir, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                entry = KnowledgeEntry(**data)
                self._update_index_for(entry)
            except (json.JSONDecodeError, TypeError, OSError):
                continue
        self._save_index()

    @property
    def count(self) -> int:
        return len(self._index)


# ===== EXTRACTION PATTERNS =====

_IP_RE = re.compile(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b")
_CVE_RE = re.compile(r"(CVE-\d{4}-\d{4,})", re.IGNORECASE)
_CRED_RE = re.compile(
    r"(?:user(?:name)?|login|account)\s*[:=]\s*(\S+).*?"
    r"(?:pass(?:word)?|pw)\s*[:=]\s*(\S+)",
    re.IGNORECASE,
)
_PORT_SVC_RE = re.compile(r"(\d{1,5})/(?:tcp|udp)\s+open\s+(\S+)\s*(.*)", re.IGNORECASE)
_FILE_PATH_RE = re.compile(r"(/(?:etc|var|usr|opt|home|tmp|root|srv|www|upload)[\w/.-]+)")
_WIN_PATH_RE = re.compile(r"([A-Z]:\\[\w\\.-]+)", re.IGNORECASE)


# ===== CONTEXT ENGINE =====

class ContextEngine:
    """
    Deterministic state machine for processing tool output.

    Handles:
      - Lifecycle classification (volatile/snapshot/discovery/reference)
      - Mutation detection (downloads, writes → bump workspace generation → stale volatiles)
      - Upsert via action keys (same action re-run → REPLACE, not duplicate)
      - Loop detection (same action 3+ times in recent history → suppress)
      - Smart entry building per action type
    """

    # Commands that produce no lasting insight — skip entirely
    _SKIP_CMDS = frozenset({
        "pwd", "cd", "echo", "clear", "cls", "history", "env", "set",
        "export", "alias", "help", "man", "exit", "quit",
    })

    # Commands whose output is volatile — replaced on re-run, staled on mutation
    _VOLATILE_CMDS = frozenset({
        "ls", "dir", "find", "tree", "ps", "top", "df", "du",
        "netstat", "ss", "ip", "ifconfig", "who", "w", "jobs",
    })

    # Patterns in commands that indicate workspace mutation
    _MUTATING_PATTERNS = (
        "download", "wget", "curl -o", "curl -O", "pip install",
        "npm install", "git clone", "git pull", "git checkout",
        "mkdir", "touch", "cp ", "mv ", "rm ", "rmdir",
        "write", "tee ", ">>", "apt install", "yum install",
        "brew install", "cargo build", "make",
    )

    # Action types that are inherently mutating
    _MUTATING_ACTIONS = frozenset({"download", "edit_file"})

    def __init__(self, store: KnowledgeStore):
        self.store = store
        self._gen = 0                       # workspace generation counter
        self._recent = deque(maxlen=10)     # recent action keys for loop detection
        self._key_map: Dict[str, str] = {}  # action_key -> entry_id
        self._rebuild_key_map()

    def _rebuild_key_map(self):
        """Restore key→id map from existing index on startup."""
        for eid, meta in self.store._index.items():
            key = meta.get("action_key", "")
            if key:
                self._key_map[key] = eid

    @property
    def generation(self) -> int:
        return self._gen

    # --- Classification ---

    def _normalize_cmd(self, display: str) -> str:
        """Extract the base command name from a display string."""
        if not display:
            return ""
        first = display.strip().split()[0].lower()
        return first.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

    def _is_mutating(self, action_type: str, display: str) -> bool:
        if action_type in self._MUTATING_ACTIONS:
            return True
        if action_type == "shell":
            display_lower = display.lower()
            return any(p in display_lower for p in self._MUTATING_PATTERNS)
        if action_type == "python":
            # Python that writes files
            display_lower = display.lower()
            return any(p in display_lower for p in ("open(", "write(", "mkdir", "shutil"))
        return False

    def _classify_lifecycle(self, action_type: str, display: str) -> Optional[str]:
        """
        Classify the lifecycle of a tool output.
        Returns None to skip storage entirely.
        """
        if action_type == "shell":
            cmd = self._normalize_cmd(display)
            if cmd in self._SKIP_CMDS:
                return None
            if cmd in self._VOLATILE_CMDS:
                return VOLATILE
            return SNAPSHOT

        if action_type == "read_file":
            return SNAPSHOT
        if action_type == "glob":
            return VOLATILE
        if action_type == "grep":
            return SNAPSHOT
        if action_type in ("search", "web_read"):
            return REFERENCE
        if action_type == "python":
            return SNAPSHOT
        if action_type == "download":
            return SNAPSHOT
        if action_type == "edit_file":
            return SNAPSHOT

        return None  # Unknown action type — skip

    def _make_key(self, action_type: str, display: str) -> str:
        """Generate a deterministic, stable key for upsert tracking."""
        # Normalize: strip whitespace, lowercase for comparison
        normalized = display.strip()
        return f"{action_type}:{normalized}"

    # --- Loop Detection ---

    def _is_loop(self, key: str) -> bool:
        """Returns True if this action has appeared 2+ times in recent history."""
        return self._recent.count(key) >= 2

    # --- Processing ---

    def process(self, action_type: str, display: str, output: str,
                phase: str = "") -> List[str]:
        """
        Main entry point. Process a tool result through the full pipeline.
        Returns list of entry IDs created/updated (empty = filtered out).
        """
        if not output or not self.store:
            return []

        # 1. Classify lifecycle — None means skip entirely
        lifecycle = self._classify_lifecycle(action_type, display)
        if lifecycle is None:
            return []

        # 2. Skip trivial output
        if action_type in ("shell", "python") and len(output.strip()) < 10:
            return []

        # 3. Generate stable key
        key = self._make_key(action_type, display)

        # 4. Loop detection — suppress if same action seen 2+ times recently
        if self._is_loop(key):
            return []
        self._recent.append(key)

        # 5. If this action is mutating, bump workspace generation
        if self._is_mutating(action_type, display):
            self._gen += 1
            self.store.mark_volatile_stale(self._gen)

        # 6. Build the entry
        entry = self._build_entry(action_type, display, output, lifecycle, key, phase)
        if not entry:
            return []

        # 7. Upsert: if key already exists, remove old entry first
        if key in self._key_map:
            old_id = self._key_map[key]
            self.store.remove(old_id)

        # 8. Store
        self.store.add(entry)
        self._key_map[key] = entry.id

        return [entry.id]

    def _build_entry(self, action_type: str, display: str, output: str,
                     lifecycle: str, key: str, phase: str) -> Optional[KnowledgeEntry]:
        """Build a KnowledgeEntry from tool output. Returns None to skip."""

        if action_type == "read_file":
            path = display
            line_count = output.count("\n") + 1
            preview = output[:200].replace("\n", " ").strip()
            return KnowledgeEntry(
                category="reference", lifecycle=lifecycle, action_key=key,
                workspace_gen=self._gen, phase=phase, source="auto:command",
                tags=["file", os.path.basename(path).lower(),
                      os.path.splitext(path)[1].lower().strip(".")],
                tier1_line=f"Read {path} ({line_count} lines)",
                tier2_summary=f"Preview: {preview}",
                tier3_full=output[:2000],
            )

        if action_type == "shell":
            preview = output[:300].replace("\n", " | ").strip()
            return KnowledgeEntry(
                category="finding" if lifecycle != VOLATILE else "topology",
                lifecycle=lifecycle, action_key=key,
                workspace_gen=self._gen, phase=phase, source="auto:command",
                tags=_auto_tags(display + " " + output[:200]),
                tier1_line=f"$ {display}"[:80],
                tier2_summary=f"Output: {preview}"[:300],
                tier3_full=f"$ {display}\n{output[:2000]}",
            )

        if action_type == "glob":
            files = [l.strip() for l in output.strip().split("\n") if l.strip()]
            if not files:
                return None
            return KnowledgeEntry(
                category="topology", lifecycle=lifecycle, action_key=key,
                workspace_gen=self._gen, phase=phase, source="auto:command",
                tags=["files", "glob"] + _auto_tags(display),
                tier1_line=f"glob {display}: {len(files)} files",
                tier2_summary="\n".join(files[:20]),
                tier3_full=output[:3000],
            )

        if action_type == "grep":
            matches = [l.strip() for l in output.strip().split("\n") if l.strip()]
            if not matches:
                return None
            return KnowledgeEntry(
                category="finding", lifecycle=lifecycle, action_key=key,
                workspace_gen=self._gen, phase=phase, source="auto:command",
                tags=["grep", "search"] + _auto_tags(display),
                tier1_line=f"grep {display}: {len(matches)} matches",
                tier2_summary="\n".join(matches[:10]),
                tier3_full=output[:3000],
            )

        if action_type == "search":
            return KnowledgeEntry(
                category="reference", lifecycle=lifecycle, action_key=key,
                workspace_gen=self._gen, phase=phase, source="auto:command",
                tags=["web", "search"] + _auto_tags(display),
                tier1_line=f"Searched: {display}"[:80],
                tier2_summary=output[:400],
                tier3_full=output[:3000],
            )

        if action_type == "web_read":
            preview = output[:300].replace("\n", " ").strip()
            return KnowledgeEntry(
                category="reference", lifecycle=lifecycle, action_key=key,
                workspace_gen=self._gen, phase=phase, source="auto:command",
                tags=["web", "page"] + _auto_tags(display),
                tier1_line=f"Web: {display}"[:80],
                tier2_summary=preview,
                tier3_full=output[:3000],
            )

        if action_type == "python":
            preview = output[:200].replace("\n", " | ").strip()
            code_line = display.split("\n")[0][:60] if display else "python"
            return KnowledgeEntry(
                category="finding", lifecycle=lifecycle, action_key=key,
                workspace_gen=self._gen, phase=phase, source="auto:command",
                tags=["python", "code"] + _auto_tags(output[:200]),
                tier1_line=f"py: {code_line} -> {preview[:40]}",
                tier2_summary=f"Code: {display[:200]}\nOutput: {preview}",
                tier3_full=f"Code:\n{display[:1000]}\nOutput:\n{output[:2000]}",
            )

        if action_type == "download":
            return KnowledgeEntry(
                category="finding", lifecycle=lifecycle, action_key=key,
                workspace_gen=self._gen, phase=phase, source="auto:command",
                tags=["download"] + _auto_tags(display),
                tier1_line=f"Downloaded: {display}"[:80],
                tier2_summary=output[:300],
                tier3_full=output[:2000],
            )

        return None


# ===== MANAGER =====

class KnowledgeManager:
    """
    High-level bridge: session + reference stores, workspace, context engine.
    All tool output flows through process_tool_result() → ContextEngine.
    """

    def __init__(self):
        self.session_store: Optional[KnowledgeStore] = None
        self.reference_store = KnowledgeStore(KNOWLEDGE_REFERENCE_DIR)
        self._engine: Optional[ContextEngine] = None
        self._workspace: Optional[str] = None
        self._workspace_cache: Optional[dict] = None

    def bind_session(self, session_file: str):
        """Create or load a session knowledge store alongside the session JSON."""
        if not session_file:
            return
        kb_dir = session_file.replace(".json", "_kb")
        self.session_store = KnowledgeStore(kb_dir)
        self._engine = ContextEngine(self.session_store)

    def bind_workspace_store(self, workspace_path: str):
        """Bind a workspace-scoped knowledge store (Assistant mode)."""
        if not workspace_path:
            return
        workspace_path = os.path.abspath(workspace_path)
        safe_name = workspace_path.replace("\\", "_").replace("/", "_").replace(":", "")
        safe_name = safe_name.strip("_")[:80]
        ws_kb_dir = os.path.join(KNOWLEDGE_DIR, "workspaces", safe_name)
        self.session_store = KnowledgeStore(ws_kb_dir)
        self._engine = ContextEngine(self.session_store)

    # --- Unified tool output processing ---

    def process_tool_result(self, action_type: str, display: str, output: str,
                            phase: str = "") -> List[str]:
        """
        Process any tool result through the context engine.
        Handles lifecycle, dedup, upsert, loop suppression, mutation tracking.
        Also runs discovery extraction for pentest-relevant content.
        Returns list of entry IDs created/updated.
        """
        ids = []

        # Context engine: stores the tool execution itself
        if self._engine:
            ids.extend(self._engine.process(action_type, display, output, phase))

        # Discovery extraction: CVEs, creds, ports from command output
        if self.session_store and action_type == "shell" and output:
            ids.extend(self._extract_discoveries(display, output, phase))

        return ids

    def _extract_discoveries(self, cmd: str, output: str, phase: str = "") -> List[str]:
        """Extract pentest-specific discoveries (ports, CVEs, creds) as DISCOVERY entries."""
        created = []

        for match in _PORT_SVC_RE.finditer(output):
            port, service, version = match.groups()
            version = version.strip()[:80]
            tier1 = f"Port {port} open: {service}"
            if version:
                tier1 += f" {version}"

            key = f"discovery:port:{port}"
            existing = self.session_store.find_by_key(key)
            if existing:
                continue

            entry = KnowledgeEntry(
                category="finding", lifecycle=DISCOVERY, action_key=key,
                tags=[port, service.lower()],
                phase=phase or "recon", source="auto:command",
                tier1_line=tier1[:80],
                tier2_summary=f"From: {cmd[:100]}\n{match.group(0).strip()}",
                tier3_full=match.group(0).strip(),
            )
            self.session_store.add(entry)
            created.append(entry.id)

        for match in _CVE_RE.finditer(output):
            cve = match.group(1).upper()
            key = f"discovery:cve:{cve}"
            if self.session_store.find_by_key(key):
                continue

            start = max(0, match.start() - 100)
            end = min(len(output), match.end() + 100)
            context = output[start:end].strip().replace("\n", " ")

            entry = KnowledgeEntry(
                category="finding", lifecycle=DISCOVERY, action_key=key,
                tags=[cve.lower(), "cve", "vulnerability"],
                phase=phase or "enum", source="auto:command",
                tier1_line=f"Vuln {cve} detected",
                tier2_summary=context[:200],
                tier3_full=context,
            )
            self.session_store.add(entry)
            created.append(entry.id)

        for match in _CRED_RE.finditer(output):
            user, passwd = match.groups()
            key = f"discovery:cred:{user}"
            if self.session_store.find_by_key(key):
                continue

            entry = KnowledgeEntry(
                category="credential", lifecycle=DISCOVERY, action_key=key,
                tags=[user.lower(), "credential", "password"],
                phase=phase or "enum", source="auto:command",
                tier1_line=f"Cred {user}:{passwd}",
                tier2_summary=f"Extracted from: {cmd[:80]}",
                tier3_full=match.group(0),
            )
            self.session_store.add(entry)
            created.append(entry.id)

        return created

    # --- AI response extraction ---

    def add_from_ai_response(self, response: str, phase: str = "") -> List[str]:
        """Conservative extraction from AI response — CVEs only."""
        if not self.session_store or not response:
            return []

        created = []
        for match in _CVE_RE.finditer(response):
            cve = match.group(1).upper()
            key = f"discovery:cve:{cve}"
            if self.session_store.find_by_key(key):
                continue

            start = max(0, match.start() - 80)
            end = min(len(response), match.end() + 80)
            context = response[start:end].strip().replace("\n", " ")

            entry = KnowledgeEntry(
                category="finding", lifecycle=DISCOVERY, action_key=key,
                tags=[cve.lower(), "cve"], phase=phase, source="auto:ai",
                tier1_line=f"AI mentioned {cve}",
                tier2_summary=context[:200],
            )
            self.session_store.add(entry)
            created.append(entry.id)

        return created

    # --- Manual add ---

    def add_manual(self, text: str, category: str = "note", tags: List[str] = None,
                   phase: str = "") -> str:
        store = self.session_store or self.reference_store
        if not tags:
            tags = _auto_tags(text)
        entry = KnowledgeEntry(
            category=category if category in CATEGORIES else "note",
            lifecycle=SNAPSHOT, tags=tags, phase=phase, source="manual",
            tier1_line=text[:80], tier2_summary=text[:300], tier3_full=text,
        )
        return store.add(entry)

    # --- Search ---

    def search_all(self, query: str, category: str = None, phase: str = None,
                   max_results: int = 10) -> List[Tuple[float, str, dict, str]]:
        results = []
        if self.session_store:
            for score, eid, meta in self.session_store.search(query, category, phase, max_results):
                results.append((score, eid, meta, "session"))
        for score, eid, meta in self.reference_store.search(query, category, phase, max_results):
            results.append((score, eid, meta, "reference"))
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:max_results]

    # --- Prompt rendering ---

    def render_for_prompt(self, token_budget: int = 400, phase: str = "") -> str:
        """Render knowledge for system prompt injection. Lifecycle-aware."""
        parts = []

        if self.session_store and self.session_store.count > 0:
            session_budget = int(token_budget * 0.7)
            session_text = self.session_store.render_for_prompt(session_budget, phase)
            if session_text:
                parts.append(session_text)

        if self.reference_store.count > 0:
            ref_budget = int(token_budget * 0.3)
            ref_text = self.reference_store.render_for_prompt(ref_budget, phase)
            if ref_text:
                parts.append(ref_text)

        return "\n".join(parts) if parts else "(no knowledge entries yet)"

    # --- Promote ---

    def promote(self, entry_id: str) -> bool:
        if not self.session_store:
            return False
        entry = self.session_store.get(entry_id)
        if not entry:
            return False
        self.reference_store.add(entry)
        self.session_store.remove(entry_id)
        return True

    # --- Workspace ---

    def set_workspace(self, path: str) -> bool:
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(path):
            return False
        self._workspace = path
        self._workspace_cache = self._index_workspace(path)
        return True

    def get_workspace(self) -> Optional[str]:
        return self._workspace

    def get_workspace_summary(self, max_tokens: int = 150) -> str:
        if not self._workspace:
            return "(no workspace set)"
        if not self._workspace_cache:
            self._workspace_cache = self._index_workspace(self._workspace)

        cache = self._workspace_cache
        char_budget = max_tokens * 4
        lines = [f"DIR: {self._workspace}"]

        if cache.get("extensions"):
            ext_parts = [f"{ext}:{count}" for ext, count in
                         sorted(cache["extensions"].items(), key=lambda x: -x[1])[:8]]
            lines.append(f"Files: {cache['total_files']} ({', '.join(ext_parts)})")
        if cache.get("dirs"):
            lines.append(f"Dirs: {', '.join(cache['dirs'][:10])}")
        if cache.get("notable"):
            lines.append(f"Notable: {', '.join(cache['notable'][:6])}")

        result = "\n".join(lines)
        if len(result) > char_budget:
            result = result[:char_budget - 3] + "..."
        return result

    def rescan_workspace(self):
        if self._workspace:
            self._workspace_cache = self._index_workspace(self._workspace)
            # Rescan = mutation (files may have changed)
            if self._engine:
                self._engine._gen += 1
                self.session_store.mark_volatile_stale(self._engine._gen)

    @staticmethod
    def _index_workspace(path: str, max_depth: int = 3) -> dict:
        extensions: Dict[str, int] = {}
        dirs = []
        notable = []
        total_files = 0
        notable_names = {
            "readme.md", "readme.txt", "readme", "makefile", "dockerfile",
            "docker-compose.yml", "docker-compose.yaml", ".env", "config.yaml",
            "config.yml", "config.json", "requirements.txt", "setup.py",
            "package.json", "cargo.toml", "go.mod", "pom.xml", "build.gradle",
            ".gitignore", "flag.txt", "user.txt", "root.txt", "proof.txt",
            "wp-config.php", "web.config", ".htaccess", "robots.txt",
        }
        skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", ".idea", ".vs"}

        try:
            for root, subdirs, files in os.walk(path):
                depth = root.replace(path, "").count(os.sep)
                if depth >= max_depth:
                    subdirs.clear()
                    continue
                subdirs[:] = [d for d in subdirs if d not in skip_dirs and not d.startswith(".")]
                if depth == 1:
                    dirs.append(os.path.relpath(root, path))
                for fname in files:
                    total_files += 1
                    ext = os.path.splitext(fname)[1].lower() or "(no ext)"
                    extensions[ext] = extensions.get(ext, 0) + 1
                    if fname.lower() in notable_names:
                        notable.append(os.path.relpath(os.path.join(root, fname), path))
        except PermissionError:
            pass

        return {
            "total_files": total_files,
            "extensions": extensions,
            "dirs": dirs,
            "notable": notable,
        }


# ===== HELPERS =====

def _auto_tags(text: str) -> List[str]:
    tags = []
    for m in _CVE_RE.finditer(text):
        tags.append(m.group(1).lower())
    for m in _IP_RE.finditer(text):
        tags.append(m.group(1))
    stop = {"the", "a", "an", "is", "was", "are", "in", "on", "at", "to", "for",
            "of", "and", "or", "not", "with", "from", "this", "that", "it", "be"}
    words = re.findall(r"[a-zA-Z][\w-]{2,}", text.lower())
    for w in words:
        if w not in stop and w not in tags:
            tags.append(w)
        if len(tags) >= 8:
            break
    return tags
