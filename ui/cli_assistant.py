"""
Artifex Assistant V5 — ASSISTANT agent CLI loop.
General-purpose AI assistant with shell, Python, and web search execution.
"""

import sys
import os
import gc
import platform
import subprocess

import torch
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
from core.agent_loop import AgentRunner, RunConfig, AutonomyLevel, Decision
from tools.agent_tools import (
    extract_agent_actions,
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


class _ConsoleHost:
    """Console event sink + approval callback for the autonomous runner (CLI)."""

    def __init__(self):
        self._streaming = False

    def _end_stream(self):
        if self._streaming:
            sys.stdout.write(Style.RESET_ALL + "\n")
            sys.stdout.flush()
            self._streaming = False

    def emit(self, ev):
        k = ev.kind
        if k == "assistant_chunk":
            if not self._streaming:
                sys.stdout.write(f"\n{Fore.CYAN}  assistant > {Fore.WHITE}")
                self._streaming = True
            sys.stdout.write(ev.text)
            sys.stdout.flush()
        elif k in ("assistant_message", "done"):
            self._end_stream()
        elif k == "action_started":
            print(f"{Fore.CYAN}  running: {Fore.WHITE}{ev.action.display}{Style.RESET_ALL}")
        elif k == "action_result":
            status = "ok" if ev.success else "ERROR"
            color = Fore.GREEN if ev.success else Fore.RED
            out = (ev.output or "").strip()
            print(f"{color}  [{status}]{Style.RESET_ALL} {Fore.WHITE}{out[:1000]}{Style.RESET_ALL}")
        elif k == "blocked":
            print(f"{Fore.RED}  BLOCKED: {ev.action.display} — {ev.reason}{Style.RESET_ALL}")
        elif k == "breaker_tripped":
            print(f"{Fore.RED}  circuit breaker: {ev.reason}{Style.RESET_ALL}")
        elif k == "gate_pause":
            print(f"{Fore.YELLOW}  human gate: {ev.reason}{Style.RESET_ALL}")
        elif k == "git":
            print(f"{Fore.CYAN}  [git] {ev.text}{Style.RESET_ALL}")
        elif k == "error":
            print(f"{Fore.RED}  error: {ev.reason}{Style.RESET_ALL}")

    def approval(self, action, decision, reason):
        self._end_stream()
        if action is None:   # circuit-breaker / human-gate pause
            ans = input(f"{Fore.YELLOW}  {reason} — continue? [Y/n/stop]: {Fore.WHITE}").strip().lower()
            return Decision.STOP if ans in ("n", "no", "s", "stop") else Decision.APPROVE
        risk = getattr(decision, "risk_level", None)
        risk = risk.name if risk is not None else "?"
        print(f"{Fore.YELLOW}  [{risk}] {action.type}: {Fore.WHITE}{action.display}{Style.RESET_ALL}")
        ans = input(f"{Fore.YELLOW}  Execute? [y/N/stop]: {Fore.WHITE}").strip().lower()
        if ans in ("s", "stop"):
            return Decision.STOP
        return Decision.APPROVE if ans in ("y", "yes") else Decision.DENY


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
    print(f"  Commands: /workspace <path>, /harness <detect|adopt|on|off>, /kb search|add|list, /refresh, /clear, /purge")
    print(f"  Agent:    /run <goal>  (autonomous loop),  /autonomy manual|guided|full")
    print(f"  Session:  /save [name], /load [name|#], /sessions, /export [path]")
    print(f"  Pipeline: /mode <mode>, /attach <file>, /output <dir>")
    print(f"  System:   /backend transformers|ollama|llama_cpp, /ctx <num>, /health, /compile, /turboquant")
    print(f"  Type 'exit' to quit.{Style.RESET_ALL}\n")

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
    cli_autonomy = AutonomyLevel.GUIDED  # level used by /run

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

            if user_input.lower() == "/refresh":
                history = compress_history(history, mode_cfg.context_window)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                n = len(history) - 1
                print(f"{Fore.CYAN}  Refreshed — {n} messages cached as key points")
                print(f"  VRAM freed.{Style.RESET_ALL}\n")
                continue

            if user_input.lower() == "/clear":
                history = [{"role": "system", "content": _build_system_prompt()}]
                _first_message = True
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
                from core.config import BASE_DIR, SESSION_DIR, KNOWLEDGE_DIR
                from core.session import purge_sessions
                from core.services import get_service
                from tools.tool_cache import CACHE_DIR

                def _dir_size(path):
                    total = 0
                    if os.path.isdir(path):
                        for root, _, files in os.walk(path):
                            for f in files:
                                try:
                                    total += os.path.getsize(os.path.join(root, f))
                                except OSError:
                                    pass
                    return total

                def _fmt(n):
                    size = float(n)
                    for unit in ("B", "KB", "MB", "GB"):
                        if size < 1024 or unit == "GB":
                            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
                        size /= 1024

                stores = [
                    ("Generated media + uploads", os.path.join(BASE_DIR, "output")),
                    ("Tool cache", CACHE_DIR),
                    ("Saved sessions", SESSION_DIR),
                    ("Knowledge base", KNOWLEDGE_DIR),
                ]
                sizes = [(label, _dir_size(path)) for label, path in stores]
                total = sum(s for _, s in sizes)
                print(f"{Fore.YELLOW}  Purge will permanently delete (configs untouched):")
                for label, size in sizes:
                    print(f"    - {label}: {_fmt(size)}")
                print(f"  Total to reclaim: {_fmt(total)}")
                print(f"  Sessions and knowledge cannot be recovered.{Style.RESET_ALL}")
                confirm = input(f"{Fore.RED}  Type 'yes' to confirm: {Style.RESET_ALL}").strip().lower()
                if confirm != "yes":
                    print(f"{Fore.CYAN}  Purge cancelled.{Style.RESET_ALL}\n")
                    continue
                reclaimed = 0
                try:
                    reclaimed += get_service().file_manager.purge()
                except Exception as e:
                    print(f"{Fore.YELLOW}  media purge error: {e}{Style.RESET_ALL}")
                try:
                    reclaimed += _dir_size(CACHE_DIR)
                    clear_cache()
                except Exception as e:
                    print(f"{Fore.YELLOW}  tool-cache purge error: {e}{Style.RESET_ALL}")
                try:
                    reclaimed += purge_sessions()
                except Exception as e:
                    print(f"{Fore.YELLOW}  session purge error: {e}{Style.RESET_ALL}")
                try:
                    reclaimed += km.purge_all()
                except Exception as e:
                    print(f"{Fore.YELLOW}  knowledge purge error: {e}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}  Purge complete — reclaimed {_fmt(reclaimed)}.{Style.RESET_ALL}\n")
                continue

            # /backend command
            if user_input.lower().startswith("/backend"):
                arg = user_input[8:].strip().lower()
                if arg in ("transformers", "ollama"):
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

            # /run command — autonomous goal execution via the shared runner
            if user_input.lower().startswith("/run"):
                goal = user_input[4:].strip()
                if not goal:
                    print(f"{Fore.YELLOW}  Usage: /run <goal>   (set mode via /autonomy manual|guided|full){Style.RESET_ALL}\n")
                    continue
                run_cfg = RunConfig.default(cli_autonomy)
                run_cfg.max_rounds = MAX_AGENT_ROUNDS
                if cli_autonomy == AutonomyLevel.FULL_AUTO:
                    print(f"{Fore.YELLOW}  Full-auto — all policy-allowed actions run unattended; "
                          f"CRITICAL + ratchet still stop it.{Style.RESET_ALL}")
                result = AgentRunner(
                    engine, build_system_prompt=_build_system_prompt,
                    emit=host.emit, request_approval=host.approval,
                    km=km, session_map=session_map, config=run_cfg,
                ).run(goal, history)
                print(f"\n{Fore.CYAN}  ■ {result.status} "
                      f"({result.rounds} rounds, {result.actions_run} actions){Style.RESET_ALL}\n")
                history = compress_history(history, mode_cfg.context_window)
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

            # /kb command
            if user_input.lower().startswith("/kb"):
                _handle_kb_command(user_input[3:].strip(), km)
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
            AgentRunner(
                engine, build_system_prompt=_build_system_prompt,
                emit=host.emit, request_approval=host.approval,
                km=km, session_map=session_map, config=turn_cfg,
            ).run(user_input, history)

            # Compress, cleanup, and auto-save
            history = compress_history(history, mode_cfg.context_window)
            if len(history) % 6 == 0:
                engine.periodic_cleanup()
            try:
                auto_save(history, metadata={
                    "model": get_active_model_name(),
                    "backend": get_active_backend(),
                })
            except Exception:
                pass  # auto-save is best-effort

        except (KeyboardInterrupt, EOFError):
            kb_count = km.session_store.count if km.session_store else 0
            print(f"\n{Fore.CYAN}  Knowledge: {kb_count} entries saved to workspace store.")
            print(f"{Fore.YELLOW}  Goodbye.{Style.RESET_ALL}")
            break


if __name__ == "__main__":
    run_assistant()
