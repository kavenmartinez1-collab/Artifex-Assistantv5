"""
Artifex Assistant V5 — ASSISTANT agent CLI loop.
General-purpose AI assistant with shell, Python, and web search execution.
"""

import sys
import os
import gc

import torch
from colorama import Fore, Style

from core.config import (
    MODES, get_context_profile, get_context_profile_name, set_context_profile,
    CONTEXT_PROFILES, get_active_backend, set_active_backend, get_active_model_name,
)
from core.engine_factory import create_engine
from core.inference import ThinkFilter, compress_history, build_active_messages
from core.prompts import build_assistant_prompt
from core.knowledge import KnowledgeManager
from tools.agent_tools import (
    extract_agent_actions,
    run_agent_action,
    get_assistant_tools_prompt,
    get_tool_output_limit,
)
from tools.tool_cache import maybe_cache_output, clear_cache, SessionMap, update_session_map


def _truncate(text, limit=None):
    if limit is None:
        limit = get_tool_output_limit()
    if len(text) > limit:
        return text[:limit] + "\n[...truncated...]"
    return text


def _execute_action(action, km=None, smap=None):
    """Execute an agent action with user confirmation."""
    type_label = action.type.upper()
    print(f"\n{Fore.YELLOW}  [{type_label}] {Fore.WHITE}{action.display}")
    confirm = input(f"{Fore.YELLOW}  Execute? [y/N]: {Fore.WHITE}").strip().lower()
    if confirm not in ("y", "yes"):
        print(f"{Fore.YELLOW}  Skipped.{Style.RESET_ALL}")
        return None

    print(f"{Fore.CYAN}  Running...{Style.RESET_ALL}\n")
    success, output = run_agent_action(action)

    if not success:
        print(f"{Fore.RED}  ERROR: {output}{Style.RESET_ALL}")
        return f"ERROR: {output}"

    # Show FULL output to the user (they see everything on screen)
    print(f"{Fore.YELLOW}  --- Output ---{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{output or '(no output)'}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}  --- End ---{Style.RESET_ALL}")

    # Process tool output through context engine
    if km and output:
        kb_ids = km.process_tool_result(action.type, action.display, output)
        if kb_ids:
            print(f"{Fore.CYAN}  [KB] {len(kb_ids)} entries{Style.RESET_ALL}")

    # Update session map BEFORE caching (needs full output for extraction)
    if smap and output:
        update_session_map(smap, action.type, action.display, output)

    # Cache large outputs — return summary for model context, full output on disk.
    # The user already saw the full output above; only the model gets the summary.
    if output:
        cached = maybe_cache_output(action.type, action.display, output)
        if cached != output:
            print(f"{Fore.CYAN}  [CACHED] Large output saved to disk — summary sent to model{Style.RESET_ALL}")
        return cached

    return "(no output — command completed successfully)"


def offer_action_execution(actions, km=None, smap=None):
    """After AI response, offer to execute detected actions."""
    if not actions:
        return None

    print(f"\n{Fore.YELLOW}  Detected actions:{Style.RESET_ALL}")
    for i, action in enumerate(actions):
        label = action.type.upper()
        print(f"    {Fore.GREEN}{i+1}. [{label}] {Fore.WHITE}{action.display}")

    print(
        f"\n{Fore.YELLOW}  Run which? {Fore.WHITE}"
        f"(1-{len(actions)}, 'a' for all, Enter to skip): ",
        end="",
    )
    choice = input().strip().lower()

    if not choice:
        return None

    outputs = []
    indices = []

    if choice == "a":
        indices = list(range(len(actions)))
    else:
        for part in choice.replace(",", " ").split():
            try:
                idx = int(part) - 1
                if 0 <= idx < len(actions):
                    indices.append(idx)
            except ValueError:
                pass

    for idx in indices:
        output = _execute_action(actions[idx], km, smap)
        if output is not None:
            label = actions[idx].type
            outputs.append(f"[{label} output] `{actions[idx].display}`:\n{output}")

    if outputs:
        return "\n\n".join(outputs)
    return None


def _make_think_indicator():
    """Create a simple thinking indicator for CLI."""
    shown = [False]

    def on_thinking(text):
        if not shown[0]:
            sys.stdout.write(f"{Fore.MAGENTA}  (thinking...){Style.RESET_ALL}")
            sys.stdout.flush()
            shown[0] = True

    return on_thinking


def _stream_response(engine, active_messages, mode_cfg):
    """Stream a response with thinking filtered out. Returns clean response."""
    print(f"\n{Fore.CYAN}  assistant > ", end="")

    def on_response(text):
        sys.stdout.write(f"{Fore.WHITE}{text}")
        sys.stdout.flush()

    think_filter = ThinkFilter(
        on_response=on_response,
        on_thinking=_make_think_indicator(),
    )

    response = engine.generate_streaming(
        active_messages,
        max_tokens=mode_cfg.max_tokens,
        temperature=mode_cfg.temperature,
        on_token=think_filter.feed,
    )
    think_filter.flush()

    print(f"{Style.RESET_ALL}\n")
    return response


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

    else:
        print(f"{Fore.YELLOW}  Unknown /kb subcommand: {subcmd}{Style.RESET_ALL}\n")


def run_assistant():
    """Main ASSISTANT agent CLI loop."""
    print(f"{Fore.CYAN}  Artifex Assistant V5 — AI Assistant{Style.RESET_ALL}")
    print(f"{Fore.WHITE}  Type your questions. The AI can run shell commands, Python, and web searches.")
    print(f"  Commands: /workspace <path>, /kb search|add|list|show|remove, /refresh, /clear")
    print(f"  Backend:  /backend transformers|ollama")
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

    # Load engine (Transformers or Ollama based on config)
    engine = create_engine()
    engine.load(status_callback=lambda msg: print(f"{Fore.CYAN}  {msg}{Style.RESET_ALL}"))
    mode_cfg = MODES["ASSISTANT"]
    session_map = SessionMap()

    # Build environment info once
    system_info = get_assistant_tools_prompt()

    def _build_system_prompt():
        profile = get_context_profile()
        return build_assistant_prompt(
            system_info, os.getcwd(),
            workspace_text=km.get_workspace_summary(max_tokens=profile.workspace_token_budget),
            knowledge_text=km.render_for_prompt(token_budget=profile.knowledge_token_budget),
            session_map_text=session_map.render(token_budget=profile.session_map_token_budget),
        )

    history = [{"role": "system", "content": _build_system_prompt()}]
    _first_message = True

    while True:
        try:
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
                print(f"{Fore.CYAN}  Cleared — conversation, session map, and tool cache reset.")
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
                # Plus deep clean: stale workspace knowledge dirs
                removed = cleanup_stale_workspaces()
                print(f"{Fore.CYAN}  Deep cleanup complete:")
                print(f"    Conversation, session map, tool cache: reset")
                print(f"    VRAM: freed")
                if removed:
                    print(f"    Stale workspaces removed: {len(removed)}")
                    for r in removed:
                        print(f"      - {r}")
                else:
                    print(f"    No stale workspaces found")
                print(f"  Knowledge preserved ({km.session_store.count if km.session_store else 0} entries).{Style.RESET_ALL}\n")
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
                    print(f"  Usage: /backend transformers|ollama{Style.RESET_ALL}\n")
                else:
                    print(f"{Fore.YELLOW}  Unknown backend: {arg}. Use 'transformers' or 'ollama'.{Style.RESET_ALL}\n")
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

            # /workspace command
            if user_input.lower().startswith("/workspace"):
                ws_args = user_input[10:].strip()
                if not ws_args:
                    print(f"{Fore.CYAN}  {km.get_workspace_summary()}{Style.RESET_ALL}\n")
                elif ws_args.lower() == "scan":
                    km.rescan_workspace()
                    print(f"{Fore.CYAN}  [+] Workspace re-scanned.")
                    print(f"  {km.get_workspace_summary()}{Style.RESET_ALL}\n")
                else:
                    if km.set_workspace(ws_args):
                        km.bind_workspace_store(ws_args)
                        print(f"{Fore.CYAN}  [+] Workspace: {ws_args}")
                        print(f"  {km.get_workspace_summary()}{Style.RESET_ALL}\n")
                    else:
                        print(f"{Fore.YELLOW}  Directory not found: {ws_args}{Style.RESET_ALL}\n")
                continue

            # /kb command
            if user_input.lower().startswith("/kb"):
                _handle_kb_command(user_input[3:].strip(), km)
                continue

            # Build system prompt with fresh context
            system_prompt = _build_system_prompt()
            history[0] = {"role": "system", "content": system_prompt}

            history.append({"role": "user", "content": user_input})

            # Set task from first user message
            if _first_message:
                session_map.set_task(user_input)
                _first_message = False

            # Token-aware sliding window
            history, active_messages = build_active_messages(history, mode_cfg.context_window)

            # Stream response (thinking filtered out)
            response = _stream_response(engine, active_messages, mode_cfg)

            history.append({"role": "assistant", "content": response})

            # Auto-extract knowledge from AI response
            ai_kb = km.add_from_ai_response(response)
            if ai_kb:
                print(f"{Fore.CYAN}  [KB] {len(ai_kb)} entries from AI response{Style.RESET_ALL}")

            # Extract actions and offer execution
            actions = extract_agent_actions(response)
            tool_output = offer_action_execution(actions, km, session_map)

            # If we ran actions, feed output back for analysis (loop for chained actions)
            while tool_output:
                truncated = _truncate(tool_output)
                feedback_msg = (
                    "[TOOL OUTPUT — this is automated command output, not a human message]\n\n"
                    f"{truncated}\n\n"
                    "Analyze the output above and tell the user what you found."
                )
                history.append({"role": "user", "content": feedback_msg})

                # Rebuild with potentially new cwd + fresh knowledge + updated session map
                system_prompt = _build_system_prompt()
                history[0] = {"role": "system", "content": system_prompt}
                active_messages = [history[0]] + history[1:][-mode_cfg.context_window:]

                response = _stream_response(engine, active_messages, mode_cfg)

                history.append({"role": "assistant", "content": response})

                # Extract knowledge from followup
                ai_kb = km.add_from_ai_response(response)
                if ai_kb:
                    print(f"{Fore.CYAN}  [KB] {len(ai_kb)} entries from AI response{Style.RESET_ALL}")

                # Check for more actions in followup
                more_actions = extract_agent_actions(response)
                tool_output = offer_action_execution(more_actions, km, session_map)

            # Compress and cleanup
            history = compress_history(history, mode_cfg.context_window)
            if len(history) % 6 == 0:
                engine.periodic_cleanup()

        except (KeyboardInterrupt, EOFError):
            kb_count = km.session_store.count if km.session_store else 0
            print(f"\n{Fore.CYAN}  Knowledge: {kb_count} entries saved to workspace store.")
            print(f"{Fore.YELLOW}  Goodbye.{Style.RESET_ALL}")
            break


if __name__ == "__main__":
    run_assistant()
