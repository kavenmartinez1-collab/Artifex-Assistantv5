"""
Artifex Assistant V5 — ASSISTANT agent CLI loop.
General-purpose AI assistant with shell, Python, and web search execution.
"""

import sys
import os
import gc
import platform
import queue
import subprocess
import threading
import time

from colorama import Fore, Style

from core.config import (
    MODES, get_context_profile, get_context_profile_name, set_context_profile,
    CONTEXT_PROFILES, get_active_backend, set_active_backend, get_active_model_name,
    get_ollama_model_config, set_ollama_model_config, get_active_ollama_model,
)
from core.engine_factory import create_engine
from core.inference import (ThinkFilter, compress_history,
                            build_active_messages, auto_compact_if_needed)
from core.prompts import build_assistant_prompt
from core.knowledge import KnowledgeManager
from core import harness
from core.agent_loop import AgentRunner, RunConfig, RunControl, AutonomyLevel, Decision
from core.sandbox.approvals import ApprovalMemory
from tools.agent_tools import (
    extract_agent_actions,
    full_request,
    run_agent_action,
    get_assistant_tools_prompt,
    get_tool_output_limit,
    MAX_AGENT_ROUNDS,
    git_commit_edit,
    git_revert_last,
)
from core.sandbox import check_policy, RiskLevel, install_all_hooks
from tools.tool_cache import maybe_cache_output, clear_cache, SessionMap, update_session_map
from core.resilience import engine_recovery, generate_with_recovery
from core.session import save_session, load_session, list_sessions, find_session, auto_save, cleanup_web_quarantine
from core.health import run_health_check, format_health_report
from core.logging_config import get_logger

_log = get_logger(__name__)

# Expected file types for each pipeline mode
_MODE_FILE_TYPES = {
    "image_edit": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"},
    "vision":     {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"},
    "stt":        {".wav", ".mp3", ".flac", ".ogg", ".m4a"},
}


def _voice_push_to_talk(listener) -> str | None:
    """Hold Space to record, release to transcribe. Returns text or None.

    Falls back to typed input if the user presses Enter instead.
    """
    import threading
    from pynput import keyboard

    stop_event = threading.Event()
    space_held = threading.Event()
    typed_input = [None]
    audio_result = [None]
    record_thread = None
    input_mode = [None]  # "voice" or "typed"

    def record_worker():
        audio_result[0] = listener.record_audio(stop_event)

    def on_press(key):
        nonlocal record_thread
        if key == keyboard.Key.space and not space_held.is_set():
            space_held.set()
            input_mode[0] = "voice"
            print(f"\r{Fore.CYAN}  [Recording...]{Style.RESET_ALL}", end="", flush=True)
            record_thread = threading.Thread(target=record_worker, daemon=True)
            record_thread.start()
        elif key == keyboard.Key.enter:
            input_mode[0] = "typed"
            stop_event.set()
            return False

    def on_release(key):
        if key == keyboard.Key.space and space_held.is_set():
            stop_event.set()
            print(f"\r{Fore.CYAN}  [Processing...]{Style.RESET_ALL}", end="", flush=True)
            return False

    with keyboard.Listener(on_press=on_press, on_release=on_release) as kb:
        kb.join()

    # If user pressed Enter instead, fall back to typed input
    if input_mode[0] == "typed":
        try:
            text = input(f"\r{Fore.GREEN}  > {Fore.WHITE}").strip()
            return text if text else None
        except EOFError:
            return None

    if record_thread:
        record_thread.join(timeout=5)

    if audio_result[0] is None:
        print(f"\r{Fore.YELLOW}  No speech detected.{Style.RESET_ALL}")
        return None

    text = listener.transcribe(audio_result[0])
    print()  # Clear the status line
    return text.strip() if text else None


def _open_file_externally(path: str):
    """Open a file with the system's default application (cross-platform)."""
    system = platform.system()
    if system == "Windows":
        os.startfile(path)
    elif system == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def _handle_kb_command(args, km):
    """Handle /kb slash command for the assistant."""
    import re as _re
    parts = args.strip().split(None, 1)
    if not parts:
        session_count = km.session_store.count if km.session_store else 0
        ref_count = km.reference_store.count
        print(f"{Fore.CYAN}  Knowledge Base: {session_count} workspace entries, {ref_count} reference entries")
        print(f"  Usage: /kb search <query> | add <text> | show <id> | list [category] | remove <id>{Style.RESET_ALL}\n")
        return

    subcmd = parts[0].lower()
    subargs = parts[1] if len(parts) > 1 else ""

    if subcmd == "search":
        if not subargs:
            print(f"{Fore.YELLOW}  Usage: /kb search <query>{Style.RESET_ALL}\n")
            return
        results = km.search_all(subargs, max_results=8)
        if not results:
            print(f"{Fore.YELLOW}  No entries matching '{subargs}'{Style.RESET_ALL}\n")
            return
        print(f"{Fore.CYAN}  Knowledge search: '{subargs}'")
        for score, eid, meta, store in results:
            cat = meta.get("category", "?")
            tier1 = meta.get("tier1", "?")
            print(f"    [{store[0].upper()}:{cat}] {eid[:8]} — {tier1} (score: {score:.1f})")
        print(f"{Style.RESET_ALL}")

    elif subcmd == "add":
        if not subargs:
            print(f"{Fore.YELLOW}  Usage: /kb add <text> [--cat finding|reference|technique|note]{Style.RESET_ALL}\n")
            return
        category = "note"
        cat_match = _re.search(r"--cat\s+(\S+)", subargs)
        if cat_match:
            category = cat_match.group(1).lower()
            subargs = subargs[:cat_match.start()].strip() + subargs[cat_match.end():].strip()
        eid = km.add_manual(subargs.strip(), category=category)
        print(f"{Fore.CYAN}  [+] Knowledge entry added: {eid[:8]} ({category}){Style.RESET_ALL}\n")

    elif subcmd == "show":
        if not subargs:
            print(f"{Fore.YELLOW}  Usage: /kb show <id>{Style.RESET_ALL}\n")
            return
        entry_id = subargs.strip()
        entry = None
        for store in [km.session_store, km.reference_store]:
            if not store:
                continue
            entry = store.get(entry_id)
            if not entry:
                for eid in store._index:
                    if eid.startswith(entry_id):
                        entry = store.get(eid)
                        break
            if entry:
                break
        if not entry:
            print(f"{Fore.YELLOW}  Entry '{entry_id}' not found.{Style.RESET_ALL}\n")
            return
        print(f"{Fore.CYAN}  ID: {entry.id}")
        print(f"  Category: {entry.category}  Source: {entry.source}")
        print(f"  Tags: {', '.join(entry.tags)}")
        print(f"  Tier 1: {entry.tier1_line}")
        print(f"  Tier 2: {entry.tier2_summary}")
        if entry.tier3_full and entry.tier3_full != entry.tier2_summary:
            print(f"  Tier 3: {entry.tier3_full[:500]}")
        print(f"{Style.RESET_ALL}")

    elif subcmd == "list":
        category = subargs.strip().lower() if subargs else None
        entries = []
        if km.session_store:
            for e in km.session_store.list_entries(category):
                e["store"] = "W"
                entries.append(e)
        for e in km.reference_store.list_entries(category):
            e["store"] = "R"
            entries.append(e)
        if not entries:
            print(f"{Fore.YELLOW}  No knowledge entries found.{Style.RESET_ALL}\n")
            return
        print(f"{Fore.CYAN}  Knowledge entries ({len(entries)} total):")
        for e in entries[:20]:
            print(f"    [{e['store']}:{e.get('category', '?')}] {e['id'][:8]} — {e.get('tier1', '?')}")
        if len(entries) > 20:
            print(f"    ...and {len(entries) - 20} more")
        print(f"{Style.RESET_ALL}")

    elif subcmd == "remove":
        if not subargs:
            print(f"{Fore.YELLOW}  Usage: /kb remove <id>{Style.RESET_ALL}\n")
            return
        eid = subargs.strip()
        removed = False
        if km.session_store:
            removed = km.session_store.remove(eid)
        if not removed:
            removed = km.reference_store.remove(eid)
        if removed:
            print(f"{Fore.CYAN}  [+] Entry removed.{Style.RESET_ALL}\n")
        else:
            print(f"{Fore.YELLOW}  Entry '{eid}' not found.{Style.RESET_ALL}\n")

    elif subcmd == "export":
        export_path = subargs.strip()
        if not export_path:
            os.makedirs("output", exist_ok=True)
            export_path = os.path.join("output", "knowledge_export.json")
        fmt = "markdown" if export_path.endswith((".md", ".markdown")) else "json"
        total = 0
        for store in [km.session_store, km.reference_store]:
            if not store:
                continue
            if fmt == "markdown":
                total += store.export_to_markdown(export_path)
            else:
                total += store.export_to_json(export_path)
        print(f"{Fore.CYAN}  Exported {total} entries to {export_path}{Style.RESET_ALL}\n")

    elif subcmd == "import":
        import_path = subargs.strip()
        if not import_path or not os.path.isfile(import_path):
            print(f"{Fore.YELLOW}  Usage: /kb import <path.json>{Style.RESET_ALL}\n")
            return
        added = km.reference_store.import_from_json(import_path)
        print(f"{Fore.CYAN}  Imported {added} entries from {import_path}{Style.RESET_ALL}\n")

    elif subcmd == "prune":
        total = 0
        if km.session_store:
            total += km.session_store.auto_prune()
        total += km.reference_store.auto_prune()
        print(f"{Fore.CYAN}  Pruned {total} stale entries.{Style.RESET_ALL}\n")

    else:
        print(f"{Fore.YELLOW}  Unknown /kb subcommand: {subcmd}{Style.RESET_ALL}\n")


def _glyphs():
    """Marks for the transcript, or plain ASCII when the console can't show them."""
    marks = {"dot": "⏺", "elbow": "⎿", "think": "✻", "stop": "■", "more": "…"}
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        "".join(marks.values()).encode(enc)
    except (UnicodeError, LookupError):
        marks = {"dot": "*", "elbow": "L", "think": "~", "stop": "#", "more": "..."}
    return marks


def _fmt_secs(seconds) -> str:
    s = max(0, int(round(seconds or 0)))
    return f"{s}s" if s < 60 else f"{s // 60}m {s % 60:02d}s"


class _ConsoleHost:
    """Console event sink + approval prompt for the agent loop (CLI).

    Prints a conversation the way a terminal agent does: prose, then one
    marked line per tool call with its (short) result indented under it.
    Everything long is still reachable: /expand prints the last tool output
    in full, /think the model's reasoning for the last turn.

    emit() is called from the turn's worker thread while the main thread
    keeps the terminal (status line, typed messages), so every write goes
    through one lock.
    """

    _RESULT_LINES = 3          # lines of a tool result shown under its call
    _SPIN = "|/-\\"

    def __init__(self):
        self._lock = threading.RLock()
        self._streaming = False
        self._shown = False      # this round's reply is already on screen
        self._held = None        # events buffered while the user types a line
        self._g = _glyphs()
        self._tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._status_on = False  # a status line is on screen (no newline yet)
        self._phase = ""
        self._phase_t0 = 0.0
        self._spin_i = 0
        self._last_tick = 0.0
        self.thoughts = []       # (seconds, text) per round of the current turn
        self.last_output = None  # (tool label, full output) of the last tool
        self.tokens = 0
        self._answered = None    # text typed as an approval answer, not echoed again

    # ── status line (main thread) ───────────────────────────────────────────

    def begin_turn(self):
        with self._lock:
            self.thoughts = []
            self.tokens = 0
            self._phase, self._phase_t0 = "Starting", time.monotonic()

    def end_turn(self):
        with self._lock:
            self._end_stream()
            self._clear_status()
            self._phase = ""

    def status_text(self) -> str:
        """What the agent is doing right now (also the prompt_toolkit toolbar)."""
        if not self._phase:
            return ""
        text = f"{self._phase}{self._g['more']} {_fmt_secs(time.monotonic() - self._phase_t0)}"
        if self.tokens:
            text += f" · {self.tokens} tok"
        return text + " · esc to interrupt"

    def tick(self):
        """Redraw the one-line status while nothing else is being printed."""
        if not self._tty:
            return
        with self._lock:
            now = time.monotonic()
            if (self._streaming or self._held is not None or not self._phase
                    or now - self._last_tick < 0.25):
                return
            self._last_tick = now
            self._spin_i = (self._spin_i + 1) % len(self._SPIN)
            sys.stdout.write(f"\r\x1b[2K{Fore.CYAN}  {self._SPIN[self._spin_i]} "
                             f"{self.status_text()}{Style.RESET_ALL}")
            sys.stdout.flush()
            self._status_on = True

    def _clear_status(self):
        if self._status_on:
            sys.stdout.write("\r\x1b[2K")
            sys.stdout.flush()
            self._status_on = False

    def _set_phase(self, text):
        self._phase, self._phase_t0 = text, time.monotonic()

    # ── typed-while-working support ─────────────────────────────────────────

    def hold(self, on: bool):
        """While the user types a line, output waits (it would scramble the
        line); releasing prints what arrived meanwhile, in order."""
        with self._lock:
            if on:
                if self._held is None:
                    self._end_stream()
                    self._clear_status()
                    self._held = []
                return
            held, self._held = self._held or [], None
            for ev in held:
                self._emit(ev)

    def say(self, text: str, color=Fore.CYAN):
        with self._lock:
            self._end_stream()
            self._clear_status()
            print(f"{color}{text}{Style.RESET_ALL}")

    # ── events (worker thread) ──────────────────────────────────────────────

    def _end_stream(self):
        if self._streaming:
            sys.stdout.write(Style.RESET_ALL + "\n")
            sys.stdout.flush()
            self._streaming = False

    def emit(self, ev):
        with self._lock:
            if self._held is not None:
                self._held.append(ev)
                return
            self._emit(ev)

    def _emit(self, ev):
        k = ev.kind
        g = self._g
        if k == "round_start":
            self._shown = False
            self._set_phase("Thinking")
            return
        if k == "thinking_chunk":
            return
        self._clear_status()
        if k == "assistant_chunk":
            if not self._streaming:
                sys.stdout.write(f"\n{Fore.CYAN}{g['dot']} {Fore.WHITE}")
                self._streaming = True
                self._set_phase("Writing")
            sys.stdout.write(ev.text)
            sys.stdout.flush()
            self._shown = True
        elif k in ("assistant_message", "done"):
            self._end_stream()
            # An answer given through the native finish tool arrives only as
            # the done summary, never as chunks; it used to vanish (saved in
            # the session, never shown). Print whatever wasn't streamed.
            text = ((ev.text if k == "assistant_message" else ev.summary) or "").strip()
            if text and not self._shown:
                print(f"\n{Fore.CYAN}{g['dot']} {Fore.WHITE}{text}{Style.RESET_ALL}")
                self._shown = True
            if k == "assistant_message":
                thinking = getattr(ev, "thinking", "") or ""
                if thinking:
                    secs = getattr(ev, "think_s", 0.0) or 0.0
                    self.thoughts.append((secs, thinking))
                    if secs >= 1:
                        print(f"{Style.DIM}  {g['think']} Thought for {_fmt_secs(secs)}"
                              f" (/think to read){Style.RESET_ALL}")
                stats = getattr(ev, "stats", None) or {}
                self.tokens += int(stats.get("completion_tokens") or 0)
        elif k == "user_message":
            # A message typed while it worked, at the point the loop read it.
            # (An answer typed at an approval prompt is already on screen.)
            self._end_stream()
            if ev.text == self._answered:
                self._answered = None
            else:
                print(f"{Fore.GREEN}> {ev.text}{Style.RESET_ALL}")
        elif k == "action_proposed":
            # The whole request (full command/script, both sides of an edit),
            # never cut: one line when it fits, the full text indented when not.
            self._end_stream()
            full = (full_request(ev.action) or ev.action.display or "").strip("\n")
            if full.startswith(ev.action.type + ": "):
                full = full[len(ev.action.type) + 2:]
            lines = full.splitlines()
            if len(lines) <= 1 and len(full) <= 100:
                print(f"{Fore.CYAN}{g['dot']} {Fore.WHITE}{ev.action.type}"
                      f"({full}){Style.RESET_ALL}")
            else:
                print(f"{Fore.CYAN}{g['dot']} {Fore.WHITE}{ev.action.type}{Style.RESET_ALL}")
                for line in lines:
                    print(f"{Fore.WHITE}    {line}{Style.RESET_ALL}")
        elif k == "action_started":
            self._set_phase(f"Running {ev.action.type}")
        elif k == "action_result":
            out = (ev.output or "").strip()
            self.last_output = (f"{ev.action.type}: {ev.action.display}", out)
            color = Fore.GREEN if ev.success else Fore.RED
            lines = out.splitlines()
            if not lines:
                lines = ["(no output)" if ev.success else "ERROR (no output)"]
            # A failure shows its END (where the error is), a success its start.
            shown = lines[:self._RESULT_LINES] if ev.success else lines[-self._RESULT_LINES:]
            for n, line in enumerate(shown):
                lead = f"  {g['elbow']} " if n == 0 else "    "
                print(f"{color}{lead}{Fore.WHITE}{line[:200]}{Style.RESET_ALL}")
            rest = len(lines) - len(shown)
            if rest > 0:
                print(f"{Style.DIM}    {g['more']} +{rest} lines (/expand){Style.RESET_ALL}")
            self._set_phase("Thinking")
        elif k == "blocked":
            print(f"{Fore.RED}  {g['elbow']} BLOCKED: {ev.reason}{Style.RESET_ALL}")
        elif k == "denied":
            print(f"{Fore.YELLOW}  {g['elbow']} skipped{Style.RESET_ALL}")
        elif k == "breaker_tripped":
            print(f"{Fore.RED}  circuit breaker: {ev.reason}{Style.RESET_ALL}")
        elif k == "gate_pause":
            print(f"{Fore.YELLOW}  checkpoint: {ev.reason}{Style.RESET_ALL}")
        elif k == "compacted":
            print(f"{Style.DIM}  context compacted: {ev.reason}{Style.RESET_ALL}")
        elif k == "git":
            print(f"{Style.DIM}  [git] {ev.text}{Style.RESET_ALL}")
        elif k == "approval_remembered":
            print(f"{Fore.CYAN}  {ev.text}{Style.RESET_ALL}")
        elif k == "auto_approved":
            print(f"{Fore.CYAN}  auto-approved ({ev.reason}){Style.RESET_ALL}")
        elif k == "error":
            print(f"{Fore.RED}  error: {ev.reason}{Style.RESET_ALL}")
        elif k == "stopped":
            self._end_stream()
            print(f"{Fore.YELLOW}  {g['stop']} {ev.reason}{Style.RESET_ALL}")
        elif k == "plan":
            self._end_stream()
            print(f"{Fore.CYAN}  PLAN:{Style.RESET_ALL}")
            for line in (ev.text or "").splitlines():
                print(f"{Fore.CYAN}    {line}{Style.RESET_ALL}")
        elif k == "step":
            self._end_stream()
            print(f"\n{Fore.CYAN}  ▶ {ev.text}{Style.RESET_ALL}")
        elif k == "step_done":
            print(f"{Fore.GREEN}  ✓ step done: {(ev.summary or '')[:200]}{Style.RESET_ALL}")
        elif k == "step_stopped":
            print(f"{Fore.YELLOW}  step stopped: {ev.reason}{Style.RESET_ALL}")
        elif k == "verify":
            first = (ev.text or "").splitlines()[0] if ev.text else ""
            color = Fore.GREEN if ev.success else Fore.YELLOW
            print(f"{color}  [auto-check] {first}{Style.RESET_ALL}")
        elif k == "note":
            print(f"{Fore.CYAN}  note pinned: {ev.text}{Style.RESET_ALL}")
        elif k in ("attempt", "attempts", "judge"):
            print(f"{Fore.CYAN}  ◆ {ev.text}{Style.RESET_ALL}")

    # ── approvals (main thread) ─────────────────────────────────────────────

    @staticmethod
    def read_answer(action, ans: str):
        """What a typed answer to an approval prompt means. Anything that is
        not one of the short answers is taken as a message: "no, and here is
        why" for an action, "carry on, and ..." for a pause."""
        raw = (ans or "").strip()
        low = raw.lower()
        if action is None:   # circuit-breaker / human-gate pause
            if low in ("n", "no", "s", "stop"):
                return Decision.STOP
            if low in ("", "y", "yes"):
                return Decision.APPROVE
            return (Decision.APPROVE, raw)
        if low in ("s", "stop"):
            return Decision.STOP
        if low in ("a", "always"):
            return Decision.APPROVE_ALWAYS
        if low in ("y", "yes"):
            return Decision.APPROVE
        if low in ("", "n", "no"):
            return Decision.DENY
        return (Decision.DENY, raw)

    def approval_prompt(self, action, decision, reason) -> str:
        """Print what is being asked; returns the prompt for the answer."""
        with self._lock:
            self._end_stream()
            self._clear_status()
            self._set_phase("Waiting for your approval")
            if action is None:
                return f"{Fore.YELLOW}  {reason} — continue? [Y/n/stop, or type a message]: {Fore.WHITE}"
            risk = getattr(decision, "risk_level", None)
            risk = risk.name if risk is not None else "?"
            print(f"{Fore.YELLOW}  [{risk}] {action.type}: {Fore.WHITE}{action.display}{Style.RESET_ALL}")
            return (f"{Fore.YELLOW}  Run it? [y / a=always / N / stop, or type why not]: "
                    f"{Fore.WHITE}")

    def answered(self, action, typed: str):
        """Turn what was typed at an approval prompt into the loop's answer."""
        answer = self.read_answer(action, typed)
        if isinstance(answer, tuple):
            self._answered = answer[1]
        return answer

    def approval(self, action, decision, reason):
        prompt = self.approval_prompt(action, decision, reason)
        return self.answered(action, input(prompt))


# ── Running a turn while the terminal stays live ─────────────────────────

_ESC = object()


class _KeyPoller:
    """Typed-while-working input without any extra package.

    Windows: single keys through msvcrt. The first printable key opens a
    "> " line (output is held back meanwhile so it cannot scramble it),
    Enter sends it, Esc on an empty line interrupts the turn. Elsewhere: a
    whole line, once Enter is pressed (the terminal echoes it as it comes).
    """

    def __init__(self, host):
        self.host = host
        self.buf = None
        try:
            import msvcrt
            self._ms = msvcrt
        except ImportError:
            self._ms = None

    @property
    def composing(self) -> bool:
        return self.buf is not None

    def _close(self):
        self.buf = None
        sys.stdout.write(Style.RESET_ALL + "\n")
        sys.stdout.flush()
        self.host.hold(False)

    def poll(self):
        """None, _ESC, or a finished line."""
        if self._ms is None:
            import select
            try:
                if select.select([sys.stdin], [], [], 0)[0]:
                    return sys.stdin.readline().strip() or None
            except (OSError, ValueError):
                pass
            return None
        ms = self._ms
        while ms.kbhit():
            ch = ms.getwch()
            if ch in ("\x00", "\xe0"):       # arrow / function key: 2nd half
                ms.getwch()
                continue
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch == "\x1b":
                if self.buf is None:
                    return _ESC
                self._close()                 # Esc drops the line being typed
                continue
            if self.buf is None:
                if ch in ("\r", "\n", "\x08"):
                    continue
                self.host.hold(True)
                self.buf = ""
                sys.stdout.write(f"{Fore.GREEN}> {Fore.WHITE}")
            if ch in ("\r", "\n"):
                line = self.buf.strip()
                self._close()
                return line or None
            if ch == "\x08":
                if self.buf:
                    self.buf = self.buf[:-1]
                    sys.stdout.write("\b \b")
            elif ch.isprintable():
                self.buf += ch
                sys.stdout.write(ch)
            sys.stdout.flush()
        return None


def _interactive_terminal() -> bool:
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def _prompt_toolkit():
    """The prompt_toolkit module if it is installed (it is optional: nothing
    in requirements pulls it in), else None."""
    try:
        import prompt_toolkit
        import prompt_toolkit.patch_stdout   # noqa: F401
        return prompt_toolkit
    except Exception:
        return None


def _plain_turn_loop(thread, asks, host, control, poller=None):
    """Main-thread side of a turn with no extra package: answer approval
    prompts, read typed messages, keep the status line moving, and turn
    Ctrl+C into an interrupt instead of ending the program."""
    while thread.is_alive():
        try:
            if poller is not None:
                got = poller.poll()
                if got is _ESC:
                    control.request_stop()
                    host.say("  interrupting" + host._g["more"], Fore.YELLOW)
                elif got:
                    control.inject_message(got)
                if poller.composing:
                    time.sleep(0.02)
                    continue
            try:
                action, decision, reason, slot, done = asks.get(timeout=0.05)
            except queue.Empty:
                host.tick()
                continue
            slot["answer"] = host.approval(action, decision, reason)
            done.set()
        except KeyboardInterrupt:
            if not control.stop_requested:
                control.request_stop()
                host.say("\n  interrupting" + host._g["more"]
                         + " (the conversation is kept)", Fore.YELLOW)
        except EOFError:
            control.request_stop()


def _ptk_turn_loop(thread, asks, host, control, ptk):
    """The same, with prompt_toolkit: a real input line stays at the bottom
    while output scrolls above it, with the status in a toolbar. A line typed
    while it works is a message for the agent - or the answer, when it is
    asking for approval. Esc interrupts."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.patch_stdout import patch_stdout

    session = PromptSession()
    pending = {}                    # the approval being asked, if any
    finished = threading.Event()

    kb = KeyBindings()

    @kb.add("escape", eager=True)
    def _interrupt(event):
        control.request_stop()

    def toolbar():
        if pending:
            return " approval: y / a=always / n / stop, or type why not"
        return " " + (host.status_text() or "working")

    def watch():
        # Posts approval questions above the input line, and closes the
        # prompt once the turn is over (it would otherwise wait for Enter).
        while not finished.is_set():
            if not pending:
                try:
                    action, decision, reason, slot, done = asks.get(timeout=0.1)
                    prompt = host.approval_prompt(action, decision, reason)
                    print(prompt + Style.RESET_ALL)
                    pending.update(action=action, slot=slot, done=done)
                except queue.Empty:
                    pass
            else:
                time.sleep(0.1)
            if not thread.is_alive():
                app = session.app
                if app.is_running:
                    try:
                        app.loop.call_soon_threadsafe(
                            lambda: app.is_running and app.exit(result=None))
                    except Exception:
                        pass
                time.sleep(0.05)

    watcher = threading.Thread(target=watch, name="agent-turn-watch", daemon=True)
    with patch_stdout(raw=True):
        watcher.start()
        try:
            while thread.is_alive():
                try:
                    line = session.prompt("> ", bottom_toolbar=toolbar,
                                          key_bindings=kb, refresh_interval=0.5)
                except KeyboardInterrupt:
                    control.request_stop()
                    continue
                except EOFError:
                    control.request_stop()
                    continue
                line = (line or "").strip()
                if pending:
                    if not line and not thread.is_alive():
                        break
                    pending["slot"]["answer"] = host.answered(pending["action"], line)
                    done = pending["done"]
                    pending.clear()
                    done.set()
                elif line:
                    control.inject_message(line)
        finally:
            finished.set()
            watcher.join(timeout=2)


def _drive_turn(runner, text, history, host, control):
    """Run one turn on a worker thread and keep the terminal live meanwhile.

    The agent loop used to run on the main thread, so nothing could be typed
    until it finished and Ctrl+C ended the whole program (losing the turn).
    Now: a line typed while it works is queued for its next step, Esc or
    Ctrl+C interrupts it and keeps the conversation, and approval prompts are
    asked here, on the thread that owns the keyboard.
    """
    box = {}
    asks = queue.Queue()

    def ask(action, decision, reason):          # called on the worker thread
        slot, done = {}, threading.Event()
        asks.put((action, decision, reason, slot, done))
        while not done.wait(0.1):
            if control.stop_requested:
                return Decision.STOP
        return slot.get("answer", Decision.STOP)

    def work():
        try:
            turn = getattr(runner, "run_turn", None) or runner.run
            box["result"] = turn(text, history)
        except BaseException as e:               # re-raised on the main thread
            box["error"] = e

    runner.request_approval = ask
    host.begin_turn()
    thread = threading.Thread(target=work, name="agent-turn", daemon=True)
    thread.start()
    try:
        live = _interactive_terminal()
        ptk = _prompt_toolkit() if live else None
        if ptk is not None:
            try:
                _ptk_turn_loop(thread, asks, host, control, ptk)
            except Exception as e:               # fall back rather than lose the turn
                _log.warning("prompt_toolkit input failed (%s); plain input", e)
                _plain_turn_loop(thread, asks, host, control,
                                 _KeyPoller(host) if live else None)
        else:
            _plain_turn_loop(thread, asks, host, control,
                             _KeyPoller(host) if live else None)
    finally:
        if thread.is_alive():        # left the loop on an error: don't hang on it
            control.request_stop()
        thread.join()
        host.hold(False)
        host.end_turn()
    if "error" in box:
        raise box["error"]
    return box["result"]


_last_inventory: list = []   # the numbered list the last /models printed


def _models_command(args: str):
    """/models: list every reachable model. /models add <#> [name] [ctx]: add one."""
    from core import model_inventory as mi
    global _last_inventory
    parts = args.split()
    if not parts:
        inv = mi.scan()
        order = {"broken": 0, "unconfigured": 1, "ready": 2}
        _last_inventory = sorted(inv["models"], key=lambda m: order.get(m["status"], 3))
        colors = {"ready": Fore.GREEN, "unconfigured": Fore.YELLOW, "broken": Fore.RED}
        c = inv["counts"]
        print(f"{Fore.CYAN}  {c['ready']} ready, {c['unconfigured']} not configured, {c['broken']} broken"
              f"{Style.RESET_ALL}")
        for i, m in enumerate(_last_inventory, 1):
            name = m.get("id") or os.path.basename(m.get("path") or "")
            ctx = m.get("num_ctx") or m.get("native_ctx")
            extra = "  ".join(x for x in (
                m.get("family") or "", f"ctx {ctx // 1024}k" if ctx else "",
                f"{m['size_bytes'] / 1e9:.1f} GB" if m.get("size_bytes") else "",
                "vision" if m.get("vision") else "") if x)
            print(f"  {i:2}. {colors.get(m['status'], '')}{m['status']:12}{Style.RESET_ALL} "
                  f"{name}  [{m.get('backend')}]  {extra}")
            for note in (m.get("problems") or []) + (m.get("warnings") or []):
                print(f"      {Fore.YELLOW}{note}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  Add one: /models add <#> [name] [ctx]{Style.RESET_ALL}\n")
        return
    if parts[0] != "add" or len(parts) < 2 or not parts[1].isdigit():
        print(f"{Fore.YELLOW}  Usage: /models   or   /models add <#> [name] [ctx]{Style.RESET_ALL}\n")
        return
    n = int(parts[1])
    if not (1 <= n <= len(_last_inventory)) or _last_inventory[n - 1]["status"] != "unconfigured":
        print(f"{Fore.YELLOW}  #{n} is not an unconfigured model in the last /models list.{Style.RESET_ALL}\n")
        return
    try:
        prop = mi.propose_entry(_last_inventory[n - 1]["path"])
    except ValueError as e:
        print(f"{Fore.YELLOW}  {e}{Style.RESET_ALL}\n")
        return
    name = parts[2] if len(parts) > 2 else prop["name"]
    if len(parts) > 3 and parts[3].isdigit():
        prop["entry"]["num_ctx"] = int(parts[3])
    print(f"{Fore.CYAN}  New entry '{name}': num_ctx {prop['entry']['num_ctx']} ({prop['ctx_reason']})")
    for note in prop["notes"]:
        print(f"    - {note}")
    for w in prop["warnings"]:
        print(f"    {Fore.YELLOW}! {w}{Fore.CYAN}")
    print(f"    flags: {' '.join(prop['entry']['extra_flags'])}{Style.RESET_ALL}")
    if input(f"{Fore.YELLOW}  Add it to llama_cpp_config.json? [y/N]: {Fore.WHITE}").strip().lower() not in ("y", "yes"):
        print(f"{Fore.CYAN}  Not added.{Style.RESET_ALL}\n")
        return
    try:
        backup = mi.add_entry(name, prop["entry"])
    except ValueError as e:
        print(f"{Fore.YELLOW}  Not added: {e}{Style.RESET_ALL}\n")
        return
    print(f"{Fore.CYAN}  Added '{name}'." + (f" Backup: {backup}" if backup else "") + f"{Style.RESET_ALL}\n")
    _last_inventory = []


_HELP_LINES = (
    "  Commands: /workspace <path>|scan, /harness <detect|adopt|on|off>, /refresh, /clear, /cleanup, /purge",
    "  Knowledge: /kb search|add|list|show|remove|export|import|prune, /index [path]",
    "  Agent:    /run <goal>  (works through it on its own),  /autonomy manual|guided|full,  /plan on|off",
    "            /think (the last turn's reasoning),  /expand (the last tool output in full)",
    "  While it works: type a message + Enter to steer it; Esc or Ctrl+C interrupts and keeps the conversation",
    "  Session:  /save [name], /load [name|#], /sessions, /export [path]",
    "  Pipeline: /mode <mode>, /attach <file>, /output <dir>, /open <file>",
    "  System:   /backend transformers|ollama|llama_cpp, /context STANDARD|HIGH, /ctx <num>, /health, /compile, /turboquant",
    "  Models:   /models (inventory), /models add <#> [name] [ctx]  (add an unconfigured GGUF)",
    "  Type /help to see this list again, 'exit' (or Ctrl+C twice) to quit.",
)

_CLI_BACKENDS = ("transformers", "ollama", "llama_cpp")


def _print_help():
    """The command list: shown in the banner and by /help."""
    print(f"{Fore.WHITE}" + "\n".join(_HELP_LINES) + f"{Style.RESET_ALL}\n")


def _settle_history(history, engine, context_window):
    """History to carry into the next turn.

    When the engine reports its context size, AgentRunner already compacts
    by TOKENS (model synopsis at ~85% of the input budget), so the history
    is left alone: squashing everything past `context_window` MESSAGES after
    every turn threw away context on large-window models and changed the
    prompt prefix each turn, so the server re-read the whole prompt. Only an
    engine with no known context size still gets the message-count fold.
    """
    try:
        ctx = int(engine.get_context_size() or 0)
    except Exception:
        ctx = 0
    if ctx > 0:
        return history
    return compress_history(history, context_window)


def run_assistant():
    """Main ASSISTANT agent CLI loop."""
    install_all_hooks()
    print(f"{Fore.CYAN}")
    print(f"  ▄▀▀▀▀▀▀▀▀▀▀▀▀▀▄")
    print(f"  █   │││ │││   █")
    print(f"  █ ╔═╧╧══╧╧═╗ █")
    print(f"  █═╣ ┌────┐  ╠═█")
    print(f"  █═╣ │ ☩  │  ╠═█")
    print(f"  █═╣ └────┘  ╠═█")
    print(f"  █═╣Ψ  ▲   ☰╠═█")
    print(f"  █ ╚═╤╤══╤╤═╝ █")
    print(f"  █   │││ │││   █")
    print(f"  █      ◎      █")
    print(f"  █    ╬╬╬╬╬    █")
    print(f"  ▀▄▄▄▄▄▄▄▄▄▄▄▄▄▀")
    print(f" Artifex-Assistant-v5{Style.RESET_ALL}")
    print()
    print(f"{Fore.WHITE}  Type your questions. The AI can run shell commands, Python, and web searches.")
    _print_help()

    # Knowledge manager + workspace setup
    km = KnowledgeManager()

    workspace_input = input(
        f"{Fore.CYAN}  Working directory (Enter for cwd): {Fore.WHITE}"
    ).strip()
    workspace = workspace_input if workspace_input else os.getcwd()

    if km.set_workspace(workspace):
        km.bind_workspace_store(workspace)
        print(f"{Fore.CYAN}  Workspace: {workspace}")
        print(f"  {km.get_workspace_summary()}{Style.RESET_ALL}\n")
    else:
        print(f"{Fore.YELLOW}  Directory not found, using cwd.{Style.RESET_ALL}")
        workspace = os.getcwd()
        km.set_workspace(workspace)
        km.bind_workspace_store(workspace)

    # Harness ingestion — absorb any prior agent's context (.claude, AGENTS.md,
    # .cursor/rules, GEMINI.md, …) from the workspace into a normalized .artifex
    # bundle and inject it the way Claude Code absorbs a folder.
    harness_state = {"text": "", "on": True}

    def _adopt_harness(path, announce=True):
        try:
            report = harness.detect(path)
            if report.is_empty:
                harness_state["text"] = ""
                if announce:
                    print(f"{Fore.CYAN}  Harness: no prior-agent config found.{Style.RESET_ALL}")
                return
            res = harness.adopt(path)
            harness_state["text"] = harness.load_injection(path) if harness_state["on"] else ""
            tools = ", ".join(n for _, n, _ in res.tools)
            if announce:
                print(f"{Fore.CYAN}  Harness absorbed → .artifex: {tools} "
                      f"(~{res.injected_token_estimate} tokens, "
                      f"{'injected' if harness_state['on'] else 'OFF'}).{Style.RESET_ALL}")
        except Exception as e:
            _log.warning("CLI harness adopt failed: %s", e)
            harness_state["text"] = ""

    _adopt_harness(workspace, announce=True)

    # Model servers this CLI starts die with it, however it is closed.
    from core.proc_tree import tie_children_to_this_process
    tie_children_to_this_process()

    # Load engine (Transformers or Ollama based on config)
    engine = create_engine()
    engine.load(status_callback=lambda msg: print(f"{Fore.CYAN}  {msg}{Style.RESET_ALL}"))
    mode_cfg = MODES["ASSISTANT"]
    session_map = SessionMap()

    # Pipeline mode state
    _cli_pipeline_mode = "chat"
    _cli_attached_files = []
    _cli_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    _CLI_MODE_MAP = {
        "chat": None, "code": None,
        "image_gen": "text-to-image", "image_edit": "image-to-image",
        "vision": "image-text-to-text", "tts": "text-to-audio",
        "stt": "automatic-speech-recognition", "music": "text-to-music",
        "video": "text-to-video", "3d": "shap-e",
        "voice": "voice-assistant", "artifex": "voice-assistant",
    }

    # Build environment info once
    system_info = get_assistant_tools_prompt()

    def _build_system_prompt():
        profile = get_context_profile()
        prompt = build_assistant_prompt(
            system_info, os.getcwd(),
            workspace_text=km.get_workspace_summary(max_tokens=profile.workspace_token_budget),
            knowledge_text=km.render_for_prompt(token_budget=profile.knowledge_token_budget),
            session_map_text=session_map.render(token_budget=profile.session_map_token_budget),
            agent_context=harness_state["text"] if harness_state["on"] else "",
        )
        # Ollama models need extra emphasis on tool marker format
        if get_active_backend() == "ollama":
            prompt += (
                "\n\nIMPORTANT REMINDER — Tool markers go in PLAIN TEXT, never in code blocks:\n"
                "CORRECT: @search(\"my query\")\n"
                "WRONG:   ```bash\\n@search(\"my query\")\\n```\n"
                "WRONG:   ```bash\\nsearch \"my query\"\\n```\n"
                "Tool markers are NOT shell commands. Write them as plain text on their own line."
            )
        return prompt

    history = [{"role": "system", "content": _build_system_prompt()}]
    _first_message = True

    _voice_listener = None  # Lazy-loaded for voice/artifex mode

    host = _ConsoleHost()
    # One conversation: plain messages and /run goals are turns of the same
    # history, and these last as long as it does. Stop/interrupt state and
    # queued messages; "always" approvals; the notes the model pinned.
    session_control = RunControl()
    session_approvals = ApprovalMemory()
    session_state = {"_notes": [], "_repo_map": "", "_steer_log": []}
    last_interrupt = [0.0]                # Ctrl+C at the prompt: twice to quit

    def _agent_turn(text, cfg, history):
        """Run one turn (a plain message or a /run goal) and return its result."""
        cfg.interactive = True
        session_control.reset()
        # Built once per turn: the knowledge and session-map text in it move
        # as tools run, and a prompt that changed every round made the
        # server re-read the whole conversation each time.
        base_prompt = _build_system_prompt()
        runner = AgentRunner(
            engine, build_system_prompt=lambda: base_prompt,
            emit=host.emit, request_approval=host.approval,
            km=km, session_map=session_map, config=cfg,
            control=session_control, approvals=session_approvals,
        )
        for attr, value in session_state.items():
            if hasattr(runner, attr):
                setattr(runner, attr, value)
        try:
            return _drive_turn(runner, text, history, host, session_control)
        finally:
            for attr in session_state:
                session_state[attr] = getattr(runner, attr, session_state[attr])
            # Typed, but the turn was interrupted before reading it.
            for unread in session_control.drain_messages():
                print(f"{Fore.YELLOW}  not read before the interrupt: {unread}{Style.RESET_ALL}")

    cli_autonomy = AutonomyLevel.GUIDED  # level used by /run
    cli_plan = "off"                      # /plan on|off — plan-first mode for /run
    _recent_failures = []                 # times of recent command failures

    while True:
        try:
            # Voice mode: push-to-talk instead of typed input
            if _cli_pipeline_mode in ("voice", "artifex"):
                if _voice_listener is None:
                    try:
                        from core.pipelines.voice_assistant import _Listener
                        _voice_listener = _Listener()
                        _voice_listener.load(
                            status_callback=lambda msg: print(f"{Fore.CYAN}  {msg}{Style.RESET_ALL}")
                        )
                    except Exception as e:
                        print(f"{Fore.RED}  STT init failed: {e}{Style.RESET_ALL}")
                        print(f"{Fore.YELLOW}  Falling back to text input.{Style.RESET_ALL}\n")
                        _cli_pipeline_mode = "chat"
                        continue

                print(f"{Fore.CYAN}  Hold Space to talk, release to send "
                      f"(or type and press Enter){Style.RESET_ALL}")

                user_input = _voice_push_to_talk(_voice_listener)
                if user_input is None:
                    continue
                print(f"{Fore.GREEN}  You: {user_input}{Style.RESET_ALL}\n")
            else:
                cwd = os.getcwd()
                user_input = input(f"{Fore.GREEN}assistant {Fore.WHITE}{cwd} > ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                break

            if user_input.lower() in ("/help", "/?"):
                _print_help()
                continue

            if user_input.lower() == "/refresh":
                history = compress_history(history, mode_cfg.context_window)
                gc.collect()
                # torch only exists on transformers installs; chat-only runs without it.
                try:
                    import torch
                except ImportError:
                    torch = None
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                n = len(history) - 1
                print(f"{Fore.CYAN}  Refreshed — {n} messages cached as key points")
                print(f"  VRAM freed.{Style.RESET_ALL}\n")
                continue

            if user_input.lower() == "/clear":
                history = [{"role": "system", "content": _build_system_prompt()}]
                _first_message = True
                # A new conversation: pinned notes and "always" approvals go too.
                session_state.update(_notes=[], _repo_map="", _steer_log=[])
                session_approvals = ApprovalMemory()
                session_map.clear()
                clear_cache()
                engine.periodic_cleanup()
                # Clean up web quarantine (downloaded files in tmpfs)
                q_ok, q_msg = cleanup_web_quarantine()
                print(f"{Fore.CYAN}  Cleared — conversation, session map, and tool cache reset.")
                if q_ok and "removed" in q_msg.lower():
                    print(f"  Web quarantine: {q_msg}")
                print(f"  Knowledge preserved ({km.session_store.count if km.session_store else 0} entries).{Style.RESET_ALL}\n")
                continue

            if user_input.lower() == "/cleanup":
                from core.knowledge import cleanup_stale_workspaces
                # Clear everything /clear does
                history = [{"role": "system", "content": _build_system_prompt()}]
                _first_message = True
                # A new conversation: pinned notes and "always" approvals go too.
                session_state.update(_notes=[], _repo_map="", _steer_log=[])
                session_approvals = ApprovalMemory()
                session_map.clear()
                clear_cache()
                engine.periodic_cleanup()
                # Clean up web quarantine
                q_ok, q_msg = cleanup_web_quarantine()
                # Plus deep clean: stale workspace knowledge dirs
                removed = cleanup_stale_workspaces()
                print(f"{Fore.CYAN}  Deep cleanup complete:")
                print(f"    Conversation, session map, tool cache: reset")
                print(f"    VRAM: freed")
                if q_ok and "removed" in q_msg.lower():
                    print(f"    Web quarantine: {q_msg}")
                if removed:
                    print(f"    Stale workspaces removed: {len(removed)}")
                    for r in removed:
                        print(f"      - {r}")
                else:
                    print(f"    No stale workspaces found")
                print(f"  Knowledge preserved ({km.session_store.count if km.session_store else 0} entries).{Style.RESET_ALL}\n")
                continue

            if user_input.lower() == "/purge":
                # Shut Artifex down and purge it (core.purge). Detached: it
                # has to stop this CLI too before the log files can go.
                from core import purge
                print(f"{Fore.YELLOW}  Purge shuts Artifex down (API, GUI, this CLI, llama-server)")
                print(f"  and permanently deletes:")
                for r in purge.preview():
                    print(f"    - {r['label']}: {r['files']} files, {purge.fmt_bytes(r['bytes'])}")
                print(f"  Models and settings stay.{Style.RESET_ALL}")
                n_av = purge.kept_avatars()
                if n_av:
                    print(f"{Fore.YELLOW}  Holo avatars ({n_av}) are kept; type PURGE ALL to erase them too.{Style.RESET_ALL}")
                confirm = input(f"{Fore.RED}  Type PURGE to confirm: {Style.RESET_ALL}").strip()
                if confirm not in ("PURGE", "PURGE ALL"):
                    print(f"{Fore.CYAN}  Purge cancelled.{Style.RESET_ALL}\n")
                    continue
                purge.spawn(delay=1.0, erase_avatars=confirm == "PURGE ALL")
                print(f"{Fore.CYAN}  Purging. Artifex is shutting down.{Style.RESET_ALL}")
                break

            # /backend command
            if user_input.lower().startswith("/backend"):
                arg = user_input[8:].strip().lower()
                if arg in _CLI_BACKENDS:
                    engine.unload()
                    set_active_backend(arg)
                    engine = create_engine()
                    engine.load(status_callback=lambda msg: print(f"{Fore.CYAN}  {msg}{Style.RESET_ALL}"))
                    print(f"{Fore.CYAN}  Backend switched to: {arg}{Style.RESET_ALL}\n")
                elif not arg:
                    backend = get_active_backend()
                    model = get_active_model_name()
                    print(f"{Fore.CYAN}  Backend: {backend}  Model: {model}")
                    print(f"  Usage: /backend transformers|ollama|llama_cpp{Style.RESET_ALL}\n")
                else:
                    print(f"{Fore.YELLOW}  Unknown backend: {arg}. Use 'transformers', 'ollama', or 'llama_cpp'.{Style.RESET_ALL}\n")
                continue

            # /context command
            if user_input.lower().startswith("/context"):
                arg = user_input[8:].strip().upper()
                if arg in CONTEXT_PROFILES:
                    set_context_profile(arg)
                    p = get_context_profile()
                    print(f"{Fore.CYAN}  Context profile: {arg}")
                    print(f"    output={p.max_output_tokens} history={p.max_history_tokens} "
                          f"knowledge={p.knowledge_token_budget} tools={p.tool_output_limit}{Style.RESET_ALL}\n")
                else:
                    name = get_context_profile_name()
                    p = get_context_profile()
                    profiles = ", ".join(CONTEXT_PROFILES.keys())
                    print(f"{Fore.CYAN}  Current profile: {name}")
                    print(f"    output={p.max_output_tokens} history={p.max_history_tokens} "
                          f"knowledge={p.knowledge_token_budget} tools={p.tool_output_limit}")
                    print(f"  Usage: /context [{profiles}]{Style.RESET_ALL}\n")
                continue

            # /models — the model inventory, and adding an unconfigured GGUF
            if user_input.lower().startswith("/models"):
                _models_command(user_input[7:].strip())
                continue

            # /ctx command — per-model Ollama context window
            if user_input.lower().startswith("/ctx"):
                arg = user_input[4:].strip()
                model = get_active_ollama_model() if get_active_backend() == "ollama" else None
                if not model:
                    print(f"{Fore.YELLOW}  /ctx only works with Ollama backend{Style.RESET_ALL}\n")
                    continue

                if arg:
                    try:
                        new_ctx = int(arg)
                        if new_ctx < 512 or new_ctx > 131072:
                            print(f"{Fore.YELLOW}  num_ctx must be between 512 and 131072{Style.RESET_ALL}\n")
                            continue
                        set_ollama_model_config(model, {"num_ctx": new_ctx})
                        print(f"{Fore.CYAN}  [{model}] num_ctx set to {new_ctx}")
                        print(f"  Saved to ollama_config.json{Style.RESET_ALL}\n")
                    except ValueError:
                        print(f"{Fore.YELLOW}  Usage: /ctx <num_ctx>  (e.g., /ctx 8192){Style.RESET_ALL}\n")
                else:
                    cfg = get_ollama_model_config(model)
                    ctx = cfg.get("num_ctx", "auto")
                    print(f"{Fore.CYAN}  [{model}] num_ctx = {ctx}")
                    print(f"  Usage: /ctx <value>  (e.g., /ctx 8192, /ctx 16384){Style.RESET_ALL}\n")
                continue

            # /workspace command
            if user_input.lower().startswith("/workspace"):
                ws_args = user_input[10:].strip()
                if not ws_args:
                    print(f"{Fore.CYAN}  {km.get_workspace_summary()}{Style.RESET_ALL}\n")
                elif ws_args.lower() == "scan":
                    km.rescan_workspace()
                    print(f"{Fore.CYAN}  [+] Workspace re-scanned.")
                    print(f"  {km.get_workspace_summary()}{Style.RESET_ALL}\n")
                    _adopt_harness(km.get_workspace() or workspace)
                else:
                    if km.set_workspace(ws_args):
                        km.bind_workspace_store(ws_args)
                        print(f"{Fore.CYAN}  [+] Workspace: {ws_args}")
                        print(f"  {km.get_workspace_summary()}{Style.RESET_ALL}\n")
                        _adopt_harness(ws_args)
                    else:
                        print(f"{Fore.YELLOW}  Directory not found: {ws_args}{Style.RESET_ALL}\n")
                continue

            # /harness command — absorb a folder's prior-agent context into .artifex
            if user_input.lower().startswith("/harness"):
                sub = user_input[8:].strip().lower()
                ws = km.get_workspace() or workspace
                if sub in ("", "status"):
                    man = harness.read_manifest(ws)
                    if man:
                        tools = ", ".join(t["name"] for t in man.get("tools", []))
                        print(f"{Fore.CYAN}  Harness: adopted [{tools}] · injection "
                              f"{'ON' if harness_state['on'] else 'OFF'}{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.CYAN}  Harness: nothing adopted in {ws}{Style.RESET_ALL}")
                    print(f"  Usage: /harness detect|adopt|resync|on|off{Style.RESET_ALL}\n")
                elif sub == "detect":
                    rep = harness.detect(ws)
                    print(f"{Fore.CYAN}  {rep.summary()}{Style.RESET_ALL}")
                    for h in rep.hits:
                        print(f"    [{h.spec.id}] {', '.join(f.relpath for f in h.files)}")
                    print()
                elif sub in ("adopt", "resync", "sync"):
                    _adopt_harness(ws, announce=True)
                    print()
                elif sub == "on":
                    harness_state["on"] = True
                    _adopt_harness(ws, announce=False)
                    print(f"{Fore.CYAN}  Harness injection ON.{Style.RESET_ALL}\n")
                elif sub == "off":
                    harness_state["on"] = False
                    harness_state["text"] = ""
                    print(f"{Fore.CYAN}  Harness injection OFF.{Style.RESET_ALL}\n")
                else:
                    print(f"{Fore.YELLOW}  Usage: /harness detect|adopt|resync|on|off{Style.RESET_ALL}\n")
                continue

            # /autonomy command — set the level used by /run
            if user_input.lower().startswith("/autonomy"):
                arg = user_input[9:].strip().lower()
                amap = {"manual": AutonomyLevel.MANUAL, "guided": AutonomyLevel.GUIDED,
                        "full": AutonomyLevel.FULL_AUTO, "full-auto": AutonomyLevel.FULL_AUTO,
                        "auto": AutonomyLevel.FULL_AUTO}
                if arg in amap:
                    cli_autonomy = amap[arg]
                    print(f"{Fore.CYAN}  /run autonomy → {cli_autonomy.value}{Style.RESET_ALL}\n")
                else:
                    print(f"{Fore.CYAN}  /run autonomy: {cli_autonomy.value}")
                    print(f"  Usage: /autonomy manual|guided|full{Style.RESET_ALL}\n")
                continue

            # /plan command — plan-first mode for /run (core/agent_plan.py)
            if user_input.lower().startswith("/plan"):
                arg = user_input[5:].strip().lower()
                if arg in ("on", "auto"):
                    cli_plan = "auto"
                elif arg == "off":
                    cli_plan = "off"
                else:
                    print(f"{Fore.CYAN}  /run plan first: {'on' if cli_plan == 'auto' else 'off'}")
                    print(f"  Usage: /plan on|off   (on: split the goal into steps, each in a "
                          f"fresh context; best for multi-part goals){Style.RESET_ALL}\n")
                    continue
                print(f"{Fore.CYAN}  /run plan first → {'on' if cli_plan == 'auto' else 'off'}"
                      f"{Style.RESET_ALL}\n")
                continue

            # /run command — autonomous goal execution via the shared runner
            if user_input.lower().startswith("/run"):
                goal = user_input[4:].strip()
                if not goal:
                    print(f"{Fore.YELLOW}  Usage: /run <goal>   (set mode via /autonomy manual|guided|full){Style.RESET_ALL}\n")
                    continue
                run_cfg = RunConfig.default(cli_autonomy)
                run_cfg.max_rounds = MAX_AGENT_ROUNDS
                # Autonomous runs use the bench-tuned agent preset (see
                # core/sampling.py + agent_bench/TUNING_REPORT.md).
                run_cfg.sampler_preset = "agent"
                # Same default as the API's agent runs (api/agent_api.py): a
                # template default of xhigh can think past the whole
                # completion budget and return nothing.
                run_cfg.reasoning_effort = "medium"
                run_cfg.plan = cli_plan
                if cli_autonomy == AutonomyLevel.FULL_AUTO:
                    print(f"{Fore.YELLOW}  Full-auto — all policy-allowed actions run unattended; "
                          f"CRITICAL + ratchet still stop it.{Style.RESET_ALL}")
                result = _agent_turn(goal, run_cfg, history)
                print(f"\n{Fore.CYAN}  ■ {result.status} "
                      f"({result.rounds} rounds, {result.actions_run} actions){Style.RESET_ALL}\n")
                history = _settle_history(history, engine, mode_cfg.context_window)
                continue

            # /think — the reasoning behind the last turn (kept in memory only)
            if user_input.lower() == "/think":
                if not host.thoughts:
                    print(f"{Fore.CYAN}  No thinking kept for the last turn.{Style.RESET_ALL}\n")
                for n, (secs, thought) in enumerate(host.thoughts, 1):
                    print(f"{Fore.CYAN}  ── round {n}, thought for {_fmt_secs(secs)} ──{Style.RESET_ALL}")
                    print(f"{Style.DIM}{thought}{Style.RESET_ALL}\n")
                continue

            # /expand — the last tool output, in full
            if user_input.lower() == "/expand":
                if not host.last_output:
                    print(f"{Fore.CYAN}  No tool has run yet.{Style.RESET_ALL}\n")
                else:
                    label, output = host.last_output
                    print(f"{Fore.CYAN}  {label}{Style.RESET_ALL}")
                    print(f"{Fore.WHITE}{output or '(no output)'}{Style.RESET_ALL}\n")
                continue

            # /health command
            if user_input.lower() == "/health":
                report = run_health_check()
                print(f"{Fore.CYAN}{format_health_report(report)}{Style.RESET_ALL}\n")
                continue

            # /compile command
            if user_input.lower().startswith("/compile"):
                from core.config import set_torch_compile, get_torch_compile
                arg = user_input[8:].strip().lower()
                if arg in ("on", "true", "yes", "1"):
                    set_torch_compile(True)
                    print(f"{Fore.CYAN}  torch.compile enabled — reload model to apply{Style.RESET_ALL}\n")
                elif arg in ("off", "false", "no", "0"):
                    set_torch_compile(False)
                    print(f"{Fore.CYAN}  torch.compile disabled{Style.RESET_ALL}\n")
                else:
                    state = "ON" if get_torch_compile() else "OFF"
                    print(f"{Fore.CYAN}  torch.compile: {state}")
                    print(f"  Usage: /compile on|off{Style.RESET_ALL}\n")
                continue

            # /turboquant command
            if user_input.lower().startswith("/turboquant"):
                from core.config import set_turboquant_kv, get_turboquant_kv
                arg = user_input[11:].strip().lower()
                if arg in ("on", "true", "yes", "1"):
                    set_turboquant_kv(True)
                    print(f"{Fore.CYAN}  TurboQuant KV cache enabled (1.7x compression){Style.RESET_ALL}\n")
                elif arg in ("off", "false", "no", "0"):
                    set_turboquant_kv(False)
                    print(f"{Fore.CYAN}  TurboQuant KV cache disabled{Style.RESET_ALL}\n")
                else:
                    state = "ON" if get_turboquant_kv() else "OFF"
                    print(f"{Fore.CYAN}  TurboQuant KV cache: {state}")
                    print(f"  Usage: /turboquant on|off{Style.RESET_ALL}\n")
                continue

            # /save command
            if user_input.lower().startswith("/save"):
                name = user_input[5:].strip() or "session"
                metadata = {
                    "model": get_active_model_name(),
                    "backend": get_active_backend(),
                    "workspace": workspace,
                }
                smap_data = session_map.to_dict() if hasattr(session_map, "to_dict") else {}
                path = save_session(name, history, smap_data, metadata)
                print(f"{Fore.CYAN}  Session saved: {os.path.basename(path)}{Style.RESET_ALL}\n")
                continue

            # /load command
            if user_input.lower().startswith("/load"):
                query = user_input[5:].strip()
                if not query:
                    print(f"{Fore.YELLOW}  Usage: /load <name|#index>{Style.RESET_ALL}\n")
                    continue
                path = find_session(query)
                if not path:
                    print(f"{Fore.YELLOW}  Session not found: {query}{Style.RESET_ALL}\n")
                    continue
                state = load_session(path)
                if state:
                    history = [{"role": "system", "content": _build_system_prompt()}] + state.messages
                    _first_message = False
                    print(f"{Fore.CYAN}  Session loaded: {os.path.basename(path)} ({len(state.messages)} messages){Style.RESET_ALL}\n")
                else:
                    print(f"{Fore.YELLOW}  Failed to load session.{Style.RESET_ALL}\n")
                continue

            # /sessions command
            if user_input.lower() == "/sessions":
                sessions = list_sessions()
                if not sessions:
                    print(f"{Fore.YELLOW}  No saved sessions.{Style.RESET_ALL}\n")
                else:
                    print(f"{Fore.CYAN}  Saved sessions:")
                    for i, s in enumerate(sessions[:15], 1):
                        mc = s["message_count"]
                        ts = s["timestamp"]
                        print(f"    {i}. {s['name']} ({mc} msgs, {ts})")
                    print(f"  Use /load <name|#> to restore.{Style.RESET_ALL}\n")
                continue

            # /export command
            if user_input.lower().startswith("/export"):
                export_path = user_input[7:].strip()
                if not export_path:
                    import time as _time
                    os.makedirs("output", exist_ok=True)
                    export_path = os.path.join("output", f"conversation_{_time.strftime('%Y%m%d_%H%M%S')}.md")
                lines = []
                for m in history:
                    role = m.get("role", "")
                    if role == "system":
                        continue
                    content = m.get("content", "")
                    if role == "user":
                        if content.startswith("[TOOL OUTPUT"):
                            lines.append(f"### Tool Output\n\n{content}\n")
                        else:
                            lines.append(f"## User\n\n{content}\n")
                    elif role == "assistant":
                        lines.append(f"## Assistant\n\n{content}\n")
                os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)
                with open(export_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                print(f"{Fore.CYAN}  Exported to: {export_path}{Style.RESET_ALL}\n")
                continue

            # /index command (RAG indexing)
            if user_input.lower().startswith("/index"):
                index_path = user_input[6:].strip() or workspace
                try:
                    from core.pipelines.embedding import EmbeddingPipeline
                    from core.rag import RAGPipeline
                    print(f"{Fore.CYAN}  Indexing {index_path}...{Style.RESET_ALL}")
                    emb = EmbeddingPipeline()
                    emb.load("", status_callback=lambda m: print(f"{Fore.CYAN}  {m}{Style.RESET_ALL}"))
                    rag = RAGPipeline(emb)
                    count = rag.index_directory(
                        index_path,
                        status_callback=lambda m: print(f"{Fore.CYAN}  {m}{Style.RESET_ALL}"),
                    )
                    rag.save(os.path.join(index_path, ".rag_index"))
                    emb.unload()
                    print(f"{Fore.CYAN}  Indexed {count} chunks. Saved to .rag_index{Style.RESET_ALL}\n")
                except ImportError:
                    print(f"{Fore.YELLOW}  sentence-transformers not installed. Run: pip install sentence-transformers{Style.RESET_ALL}\n")
                except Exception as e:
                    print(f"{Fore.RED}  Indexing failed: {e}{Style.RESET_ALL}\n")
                continue

            # /mode command — switch pipeline mode
            if user_input.lower().startswith("/mode"):
                arg = user_input[5:].strip().lower()
                if not arg:
                    modes = ", ".join(_CLI_MODE_MAP.keys())
                    print(f"{Fore.CYAN}  Current mode: {_cli_pipeline_mode}")
                    print(f"  Available: {modes}{Style.RESET_ALL}\n")
                elif arg in _CLI_MODE_MAP:
                    _cli_pipeline_mode = arg
                    print(f"{Fore.CYAN}  Pipeline mode: {_cli_pipeline_mode}{Style.RESET_ALL}")
                    if arg in ("voice", "artifex"):
                        print(f"{Fore.CYAN}  Voice Assistant active — hold Space to talk, "
                              f"release to send{Style.RESET_ALL}")
                    print()
                else:
                    print(f"{Fore.YELLOW}  Unknown mode: {arg}. Available: {', '.join(_CLI_MODE_MAP.keys())}{Style.RESET_ALL}\n")
                continue

            # /attach command — attach file for pipeline input
            if user_input.lower().startswith("/attach"):
                path = user_input[7:].strip()
                if not path:
                    if _cli_attached_files:
                        print(f"{Fore.CYAN}  Attached: {', '.join(os.path.basename(f) for f in _cli_attached_files)}{Style.RESET_ALL}\n")
                    else:
                        print(f"{Fore.YELLOW}  No files attached. Usage: /attach <path>{Style.RESET_ALL}\n")
                elif os.path.isfile(path):
                    # Validate file type matches current pipeline mode
                    expected_exts = _MODE_FILE_TYPES.get(_cli_pipeline_mode)
                    if expected_exts:
                        ext = os.path.splitext(path)[1].lower()
                        if ext not in expected_exts:
                            print(f"{Fore.YELLOW}  Warning: '{os.path.basename(path)}' may not work "
                                  f"with {_cli_pipeline_mode} mode. "
                                  f"Expected: {', '.join(sorted(expected_exts))}{Style.RESET_ALL}")
                    _cli_attached_files.append(os.path.abspath(path))
                    print(f"{Fore.CYAN}  Attached: {os.path.basename(path)}{Style.RESET_ALL}\n")
                else:
                    print(f"{Fore.YELLOW}  File not found: {path}{Style.RESET_ALL}\n")
                continue

            # /output command — set output directory
            if user_input.lower().startswith("/output"):
                path = user_input[7:].strip()
                if not path:
                    print(f"{Fore.CYAN}  Output dir: {_cli_output_dir}{Style.RESET_ALL}\n")
                elif os.path.isdir(path):
                    _cli_output_dir = os.path.abspath(path)
                    print(f"{Fore.CYAN}  Output dir: {_cli_output_dir}{Style.RESET_ALL}\n")
                else:
                    print(f"{Fore.YELLOW}  Directory not found: {path}{Style.RESET_ALL}\n")
                continue

            # /open command — open a file with system viewer
            if user_input.lower().startswith("/open"):
                path = user_input[5:].strip()
                if path and os.path.isfile(path):
                    _open_file_externally(path)
                    print(f"{Fore.CYAN}  Opened: {path}{Style.RESET_ALL}\n")
                else:
                    print(f"{Fore.YELLOW}  File not found: {path}{Style.RESET_ALL}\n")
                continue

            # /kb command (before the pipeline block: in a non-chat /mode it
            # used to be sent to the pipeline as a prompt)
            if user_input.lower().startswith("/kb"):
                _handle_kb_command(user_input[3:].strip(), km)
                continue

            # Pipeline execution for non-chat modes
            if _cli_pipeline_mode not in ("chat", "code") and _CLI_MODE_MAP.get(_cli_pipeline_mode):
                pipeline_type = _CLI_MODE_MAP[_cli_pipeline_mode]
                kwargs = {}

                if _cli_pipeline_mode == "image_gen":
                    kwargs = {"prompt": user_input, "width": 512, "height": 512, "num_steps": 30}
                elif _cli_pipeline_mode == "image_edit":
                    if not _cli_attached_files:
                        print(f"{Fore.YELLOW}  Attach an image first: /attach <path>{Style.RESET_ALL}\n")
                        continue
                    kwargs = {"image_path": _cli_attached_files[0], "prompt": user_input, "strength": 0.75}
                elif _cli_pipeline_mode == "vision":
                    if not _cli_attached_files:
                        print(f"{Fore.YELLOW}  Attach an image first: /attach <path>{Style.RESET_ALL}\n")
                        continue
                    kwargs = {"image_path": _cli_attached_files[0], "prompt": user_input or "Describe this image."}
                elif _cli_pipeline_mode == "tts":
                    kwargs = {"text": user_input}
                elif _cli_pipeline_mode == "stt":
                    if not _cli_attached_files:
                        print(f"{Fore.YELLOW}  Attach an audio file first: /attach <path>{Style.RESET_ALL}\n")
                        continue
                    kwargs = {"audio_path": _cli_attached_files[0]}
                elif _cli_pipeline_mode == "music":
                    kwargs = {"prompt": user_input, "duration_seconds": 10}
                elif _cli_pipeline_mode == "video":
                    kwargs = {"prompt": user_input, "num_frames": 16, "fps": 8}
                elif _cli_pipeline_mode == "3d":
                    kwargs = {"prompt": user_input, "num_steps": 64}
                elif _cli_pipeline_mode in ("voice", "artifex"):
                    kwargs = {"text": user_input, "play_audio": True}

                # Run through service layer with progress
                from core.services import get_service
                svc = get_service()
                print(f"{Fore.CYAN}  Running {_cli_pipeline_mode}...{Style.RESET_ALL}")

                def _cli_progress(cur, tot, msg):
                    print(f"{Fore.CYAN}  {msg}{Style.RESET_ALL}")

                result = svc.run_pipeline(
                    pipeline_type, kwargs=kwargs,
                    progress_callback=_cli_progress, store_output=True,
                )

                if result.success:
                    # Voice assistant: show response text from metadata
                    response_text = result.metadata.get("response_text")
                    if response_text:
                        print(f"\n{Fore.CYAN}  Artifex: {response_text}{Style.RESET_ALL}\n")

                    if result.output_type == "text" and not response_text:
                        print(f"\n{Fore.WHITE}  {result.content}{Style.RESET_ALL}\n")
                    elif result.output_type != "text":
                        path = result.metadata.get("stored_path") or result.metadata.get("saved_to", "")
                        if not path and result.content and isinstance(result.content, str):
                            path = result.content
                        file_id = result.metadata.get("file_id", "")
                        if path and not response_text:
                            print(f"{Fore.GREEN}  Done! Output: {os.path.basename(path)}")
                            if file_id:
                                print(f"  File ID: {file_id}")
                            print(f"  Full path: {path}{Style.RESET_ALL}\n")
                        # Auto-open images
                        if result.output_type == "image" and path and os.path.isfile(path):
                            try:
                                _open_file_externally(path)
                            except Exception:
                                pass
                else:
                    print(f"{Fore.RED}  Error: {result.error}{Style.RESET_ALL}\n")

                _cli_attached_files.clear()
                continue

            # Set task from first user message
            if _first_message:
                session_map.set_task(user_input)
                _first_message = False

            # Conversational turn via the shared autonomous runner (MANUAL + no
            # autonomous framing → confirm each action, exactly as before, but
            # one code path for both chat and /run).
            turn_cfg = RunConfig.default(AutonomyLevel.MANUAL)
            turn_cfg.framing = False
            turn_cfg.max_rounds = MAX_AGENT_ROUNDS
            _agent_turn(user_input, turn_cfg, history)

            # Settle the history (see _settle_history), cleanup, and auto-save
            history = _settle_history(history, engine, mode_cfg.context_window)
            if len(history) % 6 == 0:
                engine.periodic_cleanup()
            try:
                auto_save(history, metadata={
                    "model": get_active_model_name(),
                    "backend": get_active_backend(),
                })
            except Exception:
                pass  # auto-save is best-effort

        except KeyboardInterrupt:
            # Ctrl+C while the agent works interrupts the turn (see
            # _drive_turn). Here, at the prompt, the first one only warns: it
            # is too easy to press it once more than was needed.
            if time.monotonic() - last_interrupt[0] > 2.0:
                last_interrupt[0] = time.monotonic()
                print(f"\n{Fore.YELLOW}  Press Ctrl+C again to quit.{Style.RESET_ALL}")
                continue
            kb_count = km.session_store.count if km.session_store else 0
            print(f"\n{Fore.CYAN}  Knowledge: {kb_count} entries saved to workspace store.")
            print(f"{Fore.YELLOW}  Goodbye.{Style.RESET_ALL}")
            break
        except EOFError:
            kb_count = km.session_store.count if km.session_store else 0
            print(f"\n{Fore.CYAN}  Knowledge: {kb_count} entries saved to workspace store.")
            print(f"{Fore.YELLOW}  Goodbye.{Style.RESET_ALL}")
            break
        except Exception as e:
            # One failed command (a backend that is not running, a bad path,
            # a pipeline error) must not end the session and lose the chat.
            _log.exception("CLI command failed")
            print(f"\n{Fore.RED}  Error: {type(e).__name__}: {e}{Style.RESET_ALL}")
            # A failure that repeats without waiting for input (a broken
            # terminal, say) would spin forever: stop after 5 within 2 s.
            _now = time.monotonic()
            _recent_failures[:] = [t for t in _recent_failures if _now - t < 2.0] + [_now]
            if len(_recent_failures) >= 5:
                print(f"{Fore.RED}  Repeated errors; exiting.{Style.RESET_ALL}")
                break
            print(f"{Fore.YELLOW}  That did not finish; the session is still open "
                  f"(details in logs/artifex.log).{Style.RESET_ALL}\n")
            continue


if __name__ == "__main__":
    run_assistant()
