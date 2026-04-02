"""
Artifex Assistant V5 — Cyberpunk GUI application.
Universal AI hosting with persistent knowledge base.
Organized layout with grouped controls, image preview, resource monitor, and themes.
"""

import gc
import io
import threading
import traceback
import os
import time

import pyperclip
import psutil
import FreeSimpleGUI as sg

from core.config import (
    MODES, GPU_TIER,
    get_context_profile, set_context_profile, get_context_profile_name,
    get_model_names, get_active_model_name, set_active_model,
    get_active_backend, set_active_backend,
)
from core.engine_factory import create_engine
from core.inference import (
    ThinkFilter, compress_history, build_active_messages,
    check_vram_pressure, vram_pressure_relief,
)
from core.prompts import build_assistant_prompt
from core.knowledge import KnowledgeManager
from tools.agent_tools import (
    extract_agent_actions, run_agent_action, get_assistant_tools_prompt,
    get_tool_output_limit,
)
from tools.tool_cache import maybe_cache_output, clear_cache, SessionMap, update_session_map
from ui.gui_theme import (
    apply_theme, get_theme_names, get_active_theme,
    BG_COLOR, OUTPUT_BG,
    FONT_MAIN, FONT_TITLE, FONT_MONO, FONT_MONO_SM,
    FONT_SMALL,
    VRAM_COLOR_OK, VRAM_COLOR_WARN, VRAM_COLOR_CRIT,
)

# ─── Pipeline Configuration ──────────────────────────────────────────────────

_PIPELINE_MODES = [
    "Chat", "Code", "Image Gen", "Image Edit", "Vision",
    "3D (ShapE)", "Audio TTS", "Audio STT", "Music Gen", "Video Gen",
]
_PIPELINE_MAP = {
    "Chat": "text-generation",
    "Code": "text-generation",
    "Image Gen": "text-to-image",
    "Image Edit": "image-to-image",
    "Vision": "image-text-to-text",
    "3D (ShapE)": "shap-e",
    "Audio TTS": "text-to-audio",
    "Audio STT": "automatic-speech-recognition",
    "Music Gen": "text-to-music",
    "Video Gen": "text-to-video",
}
_FILE_INPUT_MODES = {"Vision", "Audio STT", "Image Edit"}
_PROMPT_MODES = {
    "Chat", "Code", "Image Gen", "Image Edit", "Vision",
    "3D (ShapE)", "Audio TTS", "Music Gen", "Video Gen",
}

# Categorize pipeline modes for the dropdown display
_MODE_CATEGORIES = {
    "Text": ["Chat", "Code"],
    "Image": ["Image Gen", "Image Edit", "Vision"],
    "Audio": ["Audio TTS", "Audio STT", "Music Gen"],
    "3D/Video": ["3D (ShapE)", "Video Gen"],
}


def _vram_color(fraction):
    """Return color based on VRAM usage fraction."""
    if fraction < 0.6:
        return VRAM_COLOR_OK
    elif fraction < 0.85:
        return VRAM_COLOR_WARN
    return VRAM_COLOR_CRIT


def _get_resource_text():
    """Get formatted resource usage string."""
    parts = []
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            frac = alloc / total if total > 0 else 0
            parts.append(f"VRAM: {alloc:.1f}/{total:.1f}GB")
    except Exception:
        pass
    ram = psutil.virtual_memory()
    ram_used = ram.used / (1024 ** 3)
    ram_total = ram.total / (1024 ** 3)
    parts.append(f"RAM: {ram_used:.1f}/{ram_total:.0f}GB")
    parts.append(f"CPU: {psutil.cpu_percent():.0f}%")
    return " | ".join(parts)


class ArtifexGUI:
    def __init__(self):
        apply_theme()

        self.engine = None
        self._pipeline = None
        self.busy = False

        # Knowledge manager
        self.km = KnowledgeManager()
        self.km.set_workspace(os.getcwd())
        self.km.bind_workspace_store(os.getcwd())
        self.session_map = SessionMap()

        self.messages = [{"role": "system", "content": self._get_system_prompt()}]

        self._pending_agent_actions = []
        self._pending_commands = []
        self._last_image_path = None

        # Ensure output dir exists
        self._output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        os.makedirs(self._output_dir, exist_ok=True)

        self.window = self._build_layout()
        self._setup_resource_timer()

    # ═════════════════════════════════════════════════════════════════════════
    # LAYOUT
    # ═════════════════════════════════════════════════════════════════════════

    def _build_layout(self):
        accent = "#00f0ff"
        dim = "#666688"
        panel_bg = "#0d0f1a"

        # ─── Title Bar ────────────────────────────────────────────────────
        title_bar = [
            sg.Text("ARTIFEX", font=("Segoe UI", 16, "bold"), text_color=accent),
            sg.Text("ASSISTANT V5", font=("Segoe UI", 16), text_color="#8090cc"),
            sg.Push(),
            sg.Text("Theme:", font=FONT_SMALL, text_color=dim),
            sg.Combo(get_theme_names(), default_value=get_active_theme(),
                     key="-THEME-", size=(14, 1), font=FONT_SMALL, readonly=True,
                     background_color="#161a2b", text_color=accent, enable_events=True),
            sg.Button("Health", key="-HEALTH-", button_color=(panel_bg, "#44ff88"),
                      font=FONT_SMALL, size=(7, 1)),
        ]

        # ─── Left Sidebar: Model & Workspace ─────────────────────────────
        model_frame = sg.Frame("Model & Backend", [
            [
                sg.Text("Backend:", font=FONT_SMALL, text_color=dim, size=(7, 1)),
                sg.Combo(["transformers", "ollama"], default_value=get_active_backend(),
                         key="-BACKEND-SELECT-", size=(13, 1), font=FONT_SMALL,
                         readonly=True, background_color="#161a2b", text_color=accent,
                         enable_events=True),
            ],
            [
                sg.Text("Model:", font=FONT_SMALL, text_color=dim, size=(7, 1)),
                sg.Combo(get_model_names() or [get_active_model_name()],
                         default_value=get_active_model_name(),
                         key="-MODEL-SELECT-", size=(13, 1), font=FONT_SMALL,
                         readonly=True, background_color="#161a2b", text_color=accent,
                         enable_events=True),
            ],
            [
                sg.Text(f"GPU: {GPU_TIER}", font=FONT_SMALL, text_color="#f9f871"),
                sg.Push(),
                sg.Button(f"CTX: {get_context_profile_name()}", key="-CONTEXT-TOGGLE-",
                          button_color=(panel_bg, "#f9f871"), font=FONT_SMALL, size=(12, 1)),
            ],
        ], font=FONT_SMALL, title_color=accent, border_width=1,
           background_color=panel_bg, expand_x=True)

        workspace_frame = sg.Frame("Workspace", [
            [
                sg.Input(os.getcwd(), size=(18, 1), key="-WS-PATH-", font=FONT_MONO_SM,
                         background_color="#161a2b", text_color=accent),
            ],
            [
                sg.Button("Set", key="-WS-SET-", button_color=(panel_bg, accent), size=(5, 1)),
                sg.Button("Scan", key="-WS-SCAN-", button_color=(panel_bg, accent), size=(5, 1)),
            ],
        ], font=FONT_SMALL, title_color=accent, border_width=1,
           background_color=panel_bg, expand_x=True)

        session_frame = sg.Frame("Session", [
            [
                sg.Button("Save", key="-SAVE-", button_color=(panel_bg, "#44ff88"), size=(5, 1)),
                sg.Button("Load", key="-LOAD-", button_color=(panel_bg, accent), size=(5, 1)),
                sg.Button("Export", key="-EXPORT-", button_color=(panel_bg, dim), size=(6, 1)),
            ],
        ], font=FONT_SMALL, title_color=accent, border_width=1,
           background_color=panel_bg, expand_x=True)

        sidebar = sg.Column([
            [model_frame],
            [workspace_frame],
            [session_frame],
        ], background_color=BG_COLOR, vertical_alignment="top", expand_y=True)

        # ─── Pipeline Mode Bar ────────────────────────────────────────────
        pipeline_bar = [
            sg.Text("Pipeline:", font=FONT_SMALL, text_color="#f9f871"),
            sg.Combo(_PIPELINE_MODES, default_value="Chat",
                     key="-PIPELINE-MODE-", size=(14, 1), font=FONT_SMALL,
                     readonly=True, background_color="#161a2b", text_color=accent,
                     enable_events=True),
            sg.VSeparator(),
            sg.Text("Input:", font=FONT_SMALL, text_color=dim, key="-FILE-LABEL-", visible=False),
            sg.Input("", size=(30, 1), key="-FILE-PATH-", font=FONT_MONO_SM,
                     background_color="#161a2b", text_color=accent, visible=False),
            sg.FileBrowse("Browse", key="-FILE-BROWSE-", button_color=(panel_bg, accent),
                          size=(7, 1), target="-FILE-PATH-", visible=False,
                          file_types=(("All Files", "*.*"),
                                      ("Images", "*.png *.jpg *.jpeg *.webp *.bmp"),
                                      ("Audio", "*.wav *.mp3 *.flac"))),
            sg.Push(),
            sg.Text("Output:", font=FONT_SMALL, text_color=dim, key="-OUTPUT-DIR-LABEL-", visible=False),
            sg.FolderBrowse("Set Dir", key="-OUTPUT-BROWSE-", button_color=(panel_bg, dim),
                            size=(7, 1), target="-OUTPUT-DIR-PATH-", visible=False),
            sg.Input(key="-OUTPUT-DIR-PATH-", visible=False, enable_events=True),
        ]

        # ─── Input Area ──────────────────────────────────────────────────
        input_section = [
            [
                sg.Multiline(size=(50, 3), key="-PROMPT-", font=FONT_MONO,
                             background_color="#161a2b", text_color="#9efeff",
                             focus=True, expand_x=True),
            ],
            [
                sg.Text("Tokens:", text_color="#f9f871", font=FONT_SMALL),
                sg.Input(str(MODES["ASSISTANT"].max_tokens), size=(5, 1), key="-TOKENS-",
                         font=FONT_SMALL),
                sg.Text("Temp:", text_color="#f9f871", font=FONT_SMALL),
                sg.Slider((0.1, 1.5), 0.7, 0.1, orientation="h", size=(8, 12),
                          key="-TEMP-", font=FONT_SMALL),
                sg.Push(),
                sg.Checkbox("Compile", default=False, key="-TORCH-COMPILE-",
                            font=FONT_SMALL, text_color="#44ff44",
                            tooltip="torch.compile — 20-40% faster (slow first run)"),
                sg.Checkbox("TQ Cache", default=False, key="-TURBOQUANT-KV-",
                            font=FONT_SMALL, text_color="#44ff44",
                            tooltip="TurboQuant KV cache — 1.7x compression, longer context"),
                sg.Button("EXECUTE", key="-RUN-", button_color=("#ffffff", "#ff2a6d"),
                          font=("Segoe UI", 11, "bold"), size=(10, 1), bind_return_key=True),
                sg.Button("Refresh", key="-REFRESH-", button_color=(panel_bg, accent), size=(7, 1)),
                sg.Button("Clear", key="-RESET-", button_color=(panel_bg, "#666666"), size=(5, 1)),
                sg.Button("VRAM", key="-VRAM-RELIEF-", button_color=(panel_bg, "#ff6b6b"),
                          size=(5, 1), tooltip="Free VRAM by unloading model"),
            ],
        ]

        # ─── Output Panel (tabbed: Chat / Preview) ───────────────────────
        chat_tab = sg.Tab("Chat Output", [
            [sg.Multiline(size=(55, 18), key="-OUTPUT-", font=FONT_MONO,
                          background_color=OUTPUT_BG, text_color="#c7d0ff",
                          expand_x=True, expand_y=True, disabled=True, autoscroll=True)],
        ], background_color=BG_COLOR)

        preview_tab = sg.Tab("Preview", [
            [sg.Image(key="-IMAGE-PREVIEW-", size=(512, 384), background_color="#0a0a14")],
            [
                sg.Text("No preview", key="-PREVIEW-LABEL-", font=FONT_SMALL,
                         text_color=dim, expand_x=True),
                sg.Button("Open File", key="-OPEN-PREVIEW-", button_color=(panel_bg, accent),
                          size=(9, 1), visible=False),
            ],
        ], background_color=BG_COLOR)

        output_tabs = sg.TabGroup(
            [[chat_tab, preview_tab]],
            key="-OUTPUT-TABS-",
            background_color=BG_COLOR,
            title_color=dim,
            selected_title_color=accent,
            selected_background_color="#161a2b",
            expand_x=True, expand_y=True,
        )

        # ─── Thinking Panel ──────────────────────────────────────────────
        thinking_col = sg.Column([
            [sg.Text("THINKING", font=FONT_SMALL, text_color="#b060ff")],
            [sg.Multiline(size=(22, 18), key="-THINKING-", font=FONT_MONO_SM,
                          background_color="#1a1030", text_color="#9070cc",
                          expand_x=True, expand_y=True, disabled=True, autoscroll=True)],
        ], expand_x=True, expand_y=True, background_color=BG_COLOR)

        # ─── Action Panel (suggested commands) ────────────────────────────
        cmd_panel = [
            sg.pin(sg.Column([
                [sg.Text("Suggested actions:", text_color=accent, font=FONT_SMALL)],
                [sg.Listbox(values=[], size=(40, 3), key="-CMD-LIST-",
                            font=FONT_MONO_SM, background_color="#161a2b", text_color="#9efeff",
                            select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, expand_x=True)],
                [
                    sg.Button("Run Selected", key="-CMD-RUN-", button_color=(panel_bg, accent), size=(12, 1)),
                    sg.Button("Run All", key="-CMD-ALL-", button_color=(panel_bg, accent), size=(8, 1)),
                    sg.Button("Skip", key="-CMD-SKIP-", button_color=(panel_bg, "#444444"), size=(5, 1)),
                    sg.Push(),
                    sg.Text("", key="-CMD-STATUS-", text_color=accent, font=FONT_MONO_SM),
                ],
            ], key="-CMD-PANEL-", visible=False, background_color=BG_COLOR, expand_x=True))
        ]

        # ─── Status Bar with Resource Monitor ─────────────────────────────
        status_bar = [
            sg.Text("STANDBY", key="-STATUS-", size=(50, 1),
                    background_color="#161a2b", text_color="#ff2a6d", font=FONT_SMALL),
            sg.Push(),
            sg.Text("", key="-RESOURCES-", font=FONT_SMALL, text_color="#44ff44",
                    background_color="#161a2b"),
            sg.VSeparator(),
            sg.Button("COPY", key="-COPY-", button_color=(panel_bg, dim), size=(5, 1)),
        ]

        # ─── Main Layout Assembly ─────────────────────────────────────────
        main_area = sg.Column([
            pipeline_bar,
            [sg.HSeparator(color="#222244")],
            *input_section,
            [sg.HSeparator(color="#222244")],
            [output_tabs, sg.VSeparator(color="#222244"), thinking_col],
            cmd_panel,
        ], expand_x=True, expand_y=True, background_color=BG_COLOR)

        layout = [
            title_bar,
            [sg.HSeparator(color="#222244")],
            [sidebar, sg.VSeparator(color="#222244"), main_area],
            [sg.HSeparator(color="#222244")],
            status_bar,
        ]

        return sg.Window(
            "Artifex Assistant V5", layout, resizable=True, finalize=True,
            size=(1340, 780), background_color=BG_COLOR,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # RESOURCE MONITORING
    # ═════════════════════════════════════════════════════════════════════════

    def _setup_resource_timer(self):
        """Start a periodic timer to update resource display."""
        self._update_resources()

    def _update_resources(self):
        """Update resource monitor in status bar."""
        try:
            text = _get_resource_text()
            self.window["-RESOURCES-"].update(text)

            # Color-code based on VRAM pressure
            try:
                import torch
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    frac = alloc / total if total > 0 else 0
                    self.window["-RESOURCES-"].update(text_color=_vram_color(frac))
            except Exception:
                pass
        except Exception:
            pass

        # Schedule next update (3 seconds)
        try:
            self.window.TKroot.after(3000, self._update_resources)
        except Exception:
            pass

    # ═════════════════════════════════════════════════════════════════════════
    # IMAGE PREVIEW
    # ═════════════════════════════════════════════════════════════════════════

    def _show_image_preview(self, image_path):
        """Load and display an image in the Preview tab."""
        if not image_path or not os.path.isfile(image_path):
            return
        try:
            from PIL import Image
            img = Image.open(image_path)

            # Resize to fit preview area (max 512x384)
            max_w, max_h = 512, 384
            img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

            # Convert to PNG bytes for sg.Image
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_data = buf.getvalue()

            self.window["-IMAGE-PREVIEW-"].update(data=png_data)
            self.window["-PREVIEW-LABEL-"].update(
                f"{os.path.basename(image_path)} ({img.size[0]}x{img.size[1]})"
            )
            self.window["-OPEN-PREVIEW-"].update(visible=True)
            self._last_image_path = image_path
        except Exception as e:
            self.window["-PREVIEW-LABEL-"].update(f"Preview failed: {e}")

    # ═════════════════════════════════════════════════════════════════════════
    # CORE LOGIC (unchanged from original)
    # ═════════════════════════════════════════════════════════════════════════

    def _get_system_prompt(self):
        profile = get_context_profile()
        system_info = get_assistant_tools_prompt()
        return build_assistant_prompt(
            system_info, os.getcwd(),
            workspace_text=self.km.get_workspace_summary(max_tokens=profile.workspace_token_budget),
            knowledge_text=self.km.render_for_prompt(token_budget=profile.knowledge_token_budget),
            session_map_text=self.session_map.render(token_budget=profile.session_map_token_budget),
        )

    def reset_history(self):
        self.messages = [{"role": "system", "content": self._get_system_prompt()}]
        self.window["-OUTPUT-"].update("")
        self.window["-THINKING-"].update("")
        self._pending_commands = []
        self._pending_agent_actions = []
        self.window["-CMD-PANEL-"].update(visible=False)

    def refresh_context(self):
        mode_cfg = MODES["ASSISTANT"]
        old_count = len(self.messages) - 1
        self.messages = compress_history(self.messages, mode_cfg.context_window)
        new_count = len(self.messages) - 1

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        self._pending_commands = []
        self._pending_agent_actions = []
        self.window["-CMD-PANEL-"].update(visible=False)
        self.window["-STATUS-"].update(
            f"REFRESHED -- {old_count} msgs compressed to {new_count}"
        )

    def _run_assistant_actions(self, actions=None):
        actions = actions or self._pending_agent_actions[:]
        outputs = []
        for action in actions:
            label = action.type.upper()
            self.window["-OUTPUT-"].update(f"\n[{label}] {action.display}\n", append=True)
            self.window["-CMD-STATUS-"].update(f"Running: {action.display[:40]}...")
            self.window.refresh()

            success, output = run_agent_action(action)
            if success:
                display_text = output if output else "(no output)"
                display = display_text[:3000] + "\n[...truncated...]" if len(display_text) > 3000 else display_text
                self.window["-OUTPUT-"].update(display + "\n", append=True)
                if output:
                    kb_ids = self.km.process_tool_result(action.type, action.display, output)
                    if kb_ids:
                        self.window["-OUTPUT-"].update(f"[KB] {len(kb_ids)} entries\n", append=True)
                if output:
                    update_session_map(self.session_map, action.type, action.display, output)
                result = output if output else "(no output)"
                result = maybe_cache_output(action.type, action.display, result)
                outputs.append(f"[{action.type} output] `{action.display}`:\n{result}")
            else:
                self.window["-OUTPUT-"].update(f"ERROR: {output}\n", append=True)
                outputs.append(f"[{action.type} ERROR] `{action.display}`:\nERROR: {output}")

        self.window["-CMD-STATUS-"].update("")
        self.window["-CMD-PANEL-"].update(visible=False)
        self._pending_commands = []
        self._pending_agent_actions = []
        return "\n\n".join(outputs) if outputs else None

    def _generate_thread(self, user_prompt, max_tokens, temp):
        try:
            if self.engine is None:
                self.engine = create_engine()
            self.engine.load(
                status_callback=lambda msg: self.window["-STATUS-"].update(msg)
            )

            self.messages.append({"role": "user", "content": user_prompt})
            self.messages[0] = {"role": "system", "content": self._get_system_prompt()}

            mode_cfg = MODES["ASSISTANT"]
            self.messages, active_messages = build_active_messages(self.messages, mode_cfg.context_window)

            history_text = ""
            for m in self.messages[1:-1]:
                if m["role"] == "assistant":
                    label = "ASSISTANT"
                elif m["content"].startswith("[TOOL OUTPUT"):
                    label = "[TOOL RESULT]"
                else:
                    label = "USER"
                history_text += f"{label}: {m['content']}\n\n"
            history_text += f"USER: {user_prompt}\n\nASSISTANT: "
            self.window["-OUTPUT-"].update(history_text)
            self.window["-THINKING-"].update("")

            def on_response(text):
                self.window["-OUTPUT-"].update(text, append=True, autoscroll=True)

            def on_thinking(text):
                self.window["-THINKING-"].update(text, append=True, autoscroll=True)

            think_filter = ThinkFilter(on_response=on_response, on_thinking=on_thinking)

            response = self.engine.generate_streaming(
                active_messages, max_tokens=max_tokens, temperature=temp,
                on_token=think_filter.feed,
            )
            think_filter.flush()

            self.messages.append({"role": "assistant", "content": response})

            # Show generation speed in output (persists) and status bar
            stats = getattr(self.engine, "_last_gen_stats", None)
            if stats:
                perf_line = (
                    f"\n[{stats['tok_per_sec']} tok/s | "
                    f"{stats['tokens']} tokens | "
                    f"TTFT {stats['ttft']}s | "
                    f"decode {stats['decode_tok_per_sec']} tok/s]\n"
                )
                self.window["-OUTPUT-"].update(perf_line, append=True)
                self.window["-STATUS-"].update(
                    f"READY — {stats['tok_per_sec']} tok/s"
                )
            else:
                self.window["-OUTPUT-"].update("\n\n", append=True)
                self.window["-STATUS-"].update("READY")

            try:
                self.km.add_from_ai_response(response)
            except Exception:
                pass

            actions = extract_agent_actions(response)
            if actions:
                self._pending_agent_actions = actions
                display = [f"[{a.type.upper()}] {a.display}" for a in actions]
                self._pending_commands = [a.content for a in actions]
                self.window["-CMD-LIST-"].update(values=display)
                self.window["-CMD-PANEL-"].update(visible=True)

        except Exception as e:
            traceback.print_exc()
            self.window["-OUTPUT-"].update(f"\nERROR ({type(e).__name__}): {e}", append=True)
        finally:
            self.busy = False

    def _feed_tool_output(self, tool_output):
        if not tool_output or self.busy:
            return
        self.busy = True
        self.window["-STATUS-"].update("ANALYZING OUTPUT...")

        def _analysis_thread():
            try:
                _limit = get_tool_output_limit()
                truncated = tool_output[:_limit]
                if len(tool_output) > _limit:
                    truncated += "\n[...truncated...]"
                feedback_msg = (
                    "[TOOL OUTPUT]\n\n" + truncated + "\n\n"
                    "Analyze the output above and tell the user what you found."
                )
                self.messages.append({"role": "user", "content": feedback_msg})
                self.messages[0] = {"role": "system", "content": self._get_system_prompt()}

                mode_cfg = MODES["ASSISTANT"]
                self.messages, active_messages = build_active_messages(self.messages, mode_cfg.context_window)

                self.window["-OUTPUT-"].update("\nASSISTANT: ", append=True)
                self.window["-THINKING-"].update("")

                def on_response(text):
                    self.window["-OUTPUT-"].update(text, append=True, autoscroll=True)
                def on_thinking(text):
                    self.window["-THINKING-"].update(text, append=True, autoscroll=True)

                think_filter = ThinkFilter(on_response=on_response, on_thinking=on_thinking)
                response = self.engine.generate_streaming(
                    active_messages, max_tokens=mode_cfg.max_tokens,
                    temperature=mode_cfg.temperature, on_token=think_filter.feed,
                )
                think_filter.flush()

                self.messages.append({"role": "assistant", "content": response})
                self.window["-OUTPUT-"].update("\n\n", append=True)
                self.window["-STATUS-"].update("READY")

                self.km.add_from_ai_response(response)

                actions = extract_agent_actions(response)
                if actions:
                    self._pending_agent_actions = actions
                    display = [f"[{a.type.upper()}] {a.display}" for a in actions]
                    self._pending_commands = [a.content for a in actions]
                    self.window["-CMD-LIST-"].update(values=display)
                    self.window["-CMD-PANEL-"].update(visible=True)

            except Exception as e:
                traceback.print_exc()
                self.window["-OUTPUT-"].update(f"\nERROR ({type(e).__name__}): {e}", append=True)
            finally:
                self.busy = False

        threading.Thread(target=_analysis_thread, daemon=True).start()

    def _pipeline_thread(self, mode, prompt, values):
        try:
            from core.pipelines.registry import create_pipeline

            pipeline_type = _PIPELINE_MAP.get(mode, "text-generation")
            self.window["-STATUS-"].update(f"Loading {mode} pipeline...")
            self.window["-OUTPUT-"].update(f"[{mode}] Starting...\n")

            pipeline = create_pipeline(pipeline_type)

            model_defaults = {
                "text-to-image": "runwayml/stable-diffusion-v1-5",
                "image-to-image": "runwayml/stable-diffusion-v1-5",
                "shap-e": "openai/shap-e",
                "image-text-to-text": None,
                "text-to-audio": "suno/bark-small",
                "automatic-speech-recognition": "openai/whisper-small",
                "text-to-music": "",
                "text-to-video": "",
            }
            model_path = model_defaults.get(pipeline_type, "")

            if pipeline_type == "image-text-to-text":
                from core.config import get_active_model_path
                model_path = get_active_model_path()

            self.window["-OUTPUT-"].update(f"Model: {model_path or '(auto-select)'}\n", append=True)
            self.window["-STATUS-"].update("Loading model...")

            try:
                pipeline.load(model_path, status_callback=lambda msg: self.window["-STATUS-"].update(msg))
            except Exception as e:
                self.window["-OUTPUT-"].update(f"\nFailed to load: {e}\n", append=True)
                self.window["-OUTPUT-"].update(
                    f"Download model first:\n  python download_model.py --repo {model_path}\n", append=True)
                self.window["-STATUS-"].update("LOAD FAILED")
                return

            self.window["-STATUS-"].update(f"Running {mode}...")

            kwargs = {}
            timestamp = int(time.time())

            if mode == "Image Gen":
                kwargs = {"prompt": prompt, "width": 512, "height": 512, "num_steps": 30,
                          "output_path": os.path.join(self._output_dir, f"image_{timestamp}.png")}
            elif mode == "Image Edit":
                kwargs = {"image_path": values.get("-FILE-PATH-", ""), "prompt": prompt,
                          "strength": 0.75, "num_steps": 30,
                          "output_path": os.path.join(self._output_dir, f"edit_{timestamp}.png")}
            elif mode == "Vision":
                kwargs = {"image_path": values.get("-FILE-PATH-", ""),
                          "prompt": prompt or "Describe this image in detail.", "max_tokens": 512}
            elif mode == "3D (ShapE)":
                kwargs = {"prompt": prompt, "num_steps": 64,
                          "output_path": os.path.join(self._output_dir, f"mesh_{timestamp}.ply")}
            elif mode == "Audio TTS":
                kwargs = {"text": prompt,
                          "output_path": os.path.join(self._output_dir, f"tts_{timestamp}.wav")}
            elif mode == "Audio STT":
                kwargs = {"audio_path": values.get("-FILE-PATH-", "")}
            elif mode == "Music Gen":
                kwargs = {"prompt": prompt, "duration_seconds": 10,
                          "output_path": os.path.join(self._output_dir, f"music_{timestamp}.wav")}
            elif mode == "Video Gen":
                kwargs = {"prompt": prompt, "num_frames": 16, "fps": 8,
                          "output_path": os.path.join(self._output_dir, f"video_{timestamp}.mp4")}

            self.window["-OUTPUT-"].update("Processing...\n", append=True)
            result = pipeline.run(**kwargs)

            if result.success:
                self.window["-OUTPUT-"].update(f"\nSUCCESS\n", append=True)
                if result.output_type == "text":
                    self.window["-OUTPUT-"].update(f"\n{result.content}\n", append=True)
                elif result.output_type in ("image",):
                    path = result.content or result.metadata.get("saved_to", "")
                    self.window["-OUTPUT-"].update(f"Saved: {path}\n", append=True)
                    self._show_image_preview(path)
                elif result.output_type == "mesh":
                    self.window["-OUTPUT-"].update(f"Mesh: {result.content}\n", append=True)
                elif result.output_type in ("audio", "video"):
                    self.window["-OUTPUT-"].update(f"Saved: {result.content}\n", append=True)

                if result.metadata:
                    self.window["-OUTPUT-"].update(f"Details: {result.metadata}\n", append=True)
            else:
                self.window["-OUTPUT-"].update(f"\nERROR: {result.error}\n", append=True)

            pipeline.unload(status_callback=lambda msg: self.window["-STATUS-"].update(msg))
            self.window["-STATUS-"].update("READY")

        except Exception as e:
            traceback.print_exc()
            self.window["-OUTPUT-"].update(f"\nERROR ({type(e).__name__}): {e}", append=True)
            self.window["-STATUS-"].update("ERROR")
        finally:
            self.busy = False

    # ═════════════════════════════════════════════════════════════════════════
    # EVENT LOOP
    # ═════════════════════════════════════════════════════════════════════════

    def run(self):
        while True:
            event, values = self.window.read(timeout=100)

            if event == sg.WIN_CLOSED:
                break

            if event == "__TIMEOUT__":
                continue

            # ── Action panel events ──────────────────────────────────
            if event == "-CMD-RUN-" and self._pending_agent_actions:
                selected_indices = self.window["-CMD-LIST-"].get_indexes()
                if selected_indices:
                    selected = [self._pending_agent_actions[i] for i in selected_indices]
                    tool_output = self._run_assistant_actions(selected)
                    if tool_output:
                        self._feed_tool_output(tool_output)

            elif event == "-CMD-ALL-" and self._pending_agent_actions:
                tool_output = self._run_assistant_actions()
                if tool_output:
                    self._feed_tool_output(tool_output)

            elif event == "-CMD-SKIP-":
                self.window["-CMD-PANEL-"].update(visible=False)
                self._pending_commands = []
                self._pending_agent_actions = []

            # ── Pipeline mode change ─────────────────────────────────
            elif event == "-PIPELINE-MODE-":
                mode = values["-PIPELINE-MODE-"]
                needs_file = mode in _FILE_INPUT_MODES
                needs_output = mode not in ("Chat", "Code")
                for key in ("-FILE-LABEL-", "-FILE-PATH-", "-FILE-BROWSE-"):
                    self.window[key].update(visible=needs_file)
                for key in ("-OUTPUT-DIR-LABEL-", "-OUTPUT-BROWSE-"):
                    self.window[key].update(visible=needs_output)
                self.window["-STATUS-"].update(f"Mode: {mode}")

            elif event == "-OUTPUT-DIR-PATH-":
                new_dir = values["-OUTPUT-DIR-PATH-"]
                if new_dir and os.path.isdir(new_dir):
                    self._output_dir = new_dir
                    self.window["-OUTPUT-DIR-LABEL-"].update(f"Output: {os.path.basename(new_dir)}/")

            # ── Execute ──────────────────────────────────────────────
            elif event == "-RUN-":
                if self.busy:
                    continue
                prompt = values["-PROMPT-"]
                mode = values["-PIPELINE-MODE-"]

                if mode not in ("Chat", "Code"):
                    if mode in _PROMPT_MODES and not prompt.strip():
                        self.window["-STATUS-"].update("Prompt is required.")
                        continue
                    if mode in _FILE_INPUT_MODES:
                        fp = values["-FILE-PATH-"]
                        if not fp or not os.path.isfile(fp):
                            self.window["-STATUS-"].update("Select an input file first.")
                            continue
                    self.busy = True
                    self.window["-STATUS-"].update(f"RUNNING {mode.upper()}...")
                    self.window["-PROMPT-"].update("")
                    threading.Thread(target=self._pipeline_thread,
                                     args=(mode, prompt.strip(), values), daemon=True).start()
                    continue

                if not prompt.strip():
                    continue

                self.busy = True
                self.window["-STATUS-"].update("GENERATING...")
                self.window["-PROMPT-"].update("")
                max_tokens = int(values["-TOKENS-"])
                temp = values["-TEMP-"]

                # Sync optimization toggles to config
                from core.config import set_torch_compile, set_turboquant_kv
                set_torch_compile(values.get("-TORCH-COMPILE-", False))
                set_turboquant_kv(values.get("-TURBOQUANT-KV-", False))

                threading.Thread(target=self._generate_thread,
                                 args=(prompt, max_tokens, temp), daemon=True).start()

            # ── Toolbar buttons ──────────────────────────────────────
            elif event == "-REFRESH-":
                if not self.busy:
                    self.refresh_context()

            elif event == "-RESET-":
                if self.engine is not None:
                    self.engine.unload(status_callback=lambda msg: self.window["-STATUS-"].update(msg))
                    self.engine = None
                self.session_map.clear()
                clear_cache()
                gc.collect()
                self.reset_history()
                self.window["-STATUS-"].update("CLEARED")

            elif event == "-CONTEXT-TOGGLE-":
                current = get_context_profile_name()
                new_name = "HIGH" if current == "STANDARD" else "STANDARD"
                set_context_profile(new_name)
                profile = get_context_profile()
                self.window["-CONTEXT-TOGGLE-"].update(f"CTX: {new_name}")
                self.window["-TOKENS-"].update(str(profile.max_output_tokens))
                self.window["-STATUS-"].update(f"Context: {new_name}")

            elif event == "-VRAM-RELIEF-":
                mode_cfg = MODES["ASSISTANT"]
                pressured, frac, alloc, total = check_vram_pressure(threshold=0.0)
                self.window["-STATUS-"].update(f"VRAM: {alloc:.1f}/{total:.1f}GB -- unloading...")
                self.window.refresh()
                if self.engine is not None:
                    relieved, msg = vram_pressure_relief(
                        self.engine, self.messages, mode_cfg.context_window)
                    if relieved:
                        self.messages[0] = {"role": "system", "content": self._get_system_prompt()}
                        self.engine = None
                        self.window["-STATUS-"].update(f"VRAM freed. Model reloads on next message.")
                    else:
                        if self.engine.is_loaded():
                            self.engine.unload()
                        self.engine = None
                        gc.collect()
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        self.window["-STATUS-"].update("Model unloaded. Will reload on next message.")
                else:
                    self.window["-STATUS-"].update("No engine loaded.")

            # ── Backend / Model selection ─────────────────────────────
            elif event == "-BACKEND-SELECT-":
                new_backend = values["-BACKEND-SELECT-"]
                if new_backend != get_active_backend():
                    if self.engine is not None:
                        self.engine.unload(status_callback=lambda msg: self.window["-STATUS-"].update(msg))
                        self.engine = None
                    set_active_backend(new_backend)
                    models = get_model_names()
                    if models:
                        self.window["-MODEL-SELECT-"].update(values=models, value=models[0])
                    else:
                        self.window["-MODEL-SELECT-"].update(
                            values=[get_active_model_name()], value=get_active_model_name())
                    self.window["-STATUS-"].update(f"Backend: {new_backend}")

            elif event == "-MODEL-SELECT-":
                new_model = values["-MODEL-SELECT-"]
                if set_active_model(new_model):
                    if self.engine is not None and self.engine.is_loaded():
                        self.engine.unload(status_callback=lambda msg: self.window["-STATUS-"].update(msg))
                        self.engine = None
                    self.window["-STATUS-"].update(f"Model: {new_model}")
                else:
                    self.window["-MODEL-SELECT-"].update(value=get_active_model_name())

            # ── Workspace ────────────────────────────────────────────
            elif event == "-WS-SET-":
                ws_path = values["-WS-PATH-"].strip()
                if ws_path and os.path.isdir(ws_path):
                    os.chdir(ws_path)
                    self.km.set_workspace(ws_path)
                    self.km.bind_workspace_store(ws_path)
                    self.window["-STATUS-"].update(f"Workspace: {ws_path}")
                    self.window["-OUTPUT-"].update(
                        f"[Workspace: {ws_path}]\n{self.km.get_workspace_summary()}\n", append=True)
                else:
                    self.window["-STATUS-"].update(f"Directory not found: {ws_path}")

            elif event == "-WS-SCAN-":
                self.km.rescan_workspace()
                self.window["-OUTPUT-"].update(
                    f"[Re-scanned]\n{self.km.get_workspace_summary()}\n", append=True)

            # ── Session management ───────────────────────────────────
            elif event == "-SAVE-":
                from core.session import save_session
                name = sg.popup_get_text("Session name:", title="Save Session",
                                          default_text="session", background_color=BG_COLOR)
                if name:
                    metadata = {"model": get_active_model_name(), "backend": get_active_backend()}
                    path = save_session(name, self.messages, metadata=metadata)
                    self.window["-STATUS-"].update(f"Saved: {os.path.basename(path)}")

            elif event == "-LOAD-":
                from core.session import list_sessions, load_session
                sessions = list_sessions()
                if not sessions:
                    self.window["-STATUS-"].update("No saved sessions.")
                    continue
                names = [f"{s['name']} ({s['message_count']} msgs, {s['timestamp']})" for s in sessions]
                choice = sg.popup_get_text(
                    "Enter session # or name:\n\n" + "\n".join(f"  {i+1}. {n}" for i, n in enumerate(names[:10])),
                    title="Load Session", background_color=BG_COLOR)
                if choice:
                    from core.session import find_session
                    path = find_session(choice)
                    if path:
                        state = load_session(path)
                        if state:
                            self.messages = [{"role": "system", "content": self._get_system_prompt()}] + state.messages
                            self.window["-STATUS-"].update(f"Loaded: {len(state.messages)} messages")
                            history = ""
                            for m in state.messages:
                                role = "USER" if m["role"] == "user" else "ASSISTANT"
                                history += f"{role}: {m['content']}\n\n"
                            self.window["-OUTPUT-"].update(history)

            elif event == "-EXPORT-":
                export_path = sg.popup_get_file("Save conversation as:", save_as=True,
                                                 default_extension=".md",
                                                 file_types=(("Markdown", "*.md"), ("Text", "*.txt")))
                if export_path:
                    lines = []
                    for m in self.messages:
                        if m.get("role") == "system":
                            continue
                        role = "## User" if m["role"] == "user" else "## Assistant"
                        lines.append(f"{role}\n\n{m.get('content', '')}\n")
                    with open(export_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines))
                    self.window["-STATUS-"].update(f"Exported: {export_path}")

            # ── Theme switching ──────────────────────────────────────
            elif event == "-THEME-":
                new_theme = values["-THEME-"]
                if new_theme != get_active_theme():
                    # Theme change requires window recreation
                    self.window.close()
                    apply_theme(new_theme)
                    self.window = self._build_layout()
                    self._setup_resource_timer()
                    self.window["-STATUS-"].update(f"Theme: {new_theme}")

            # ── Health check ─────────────────────────────────────────
            elif event == "-HEALTH-":
                from core.health import run_health_check, format_health_report
                report = run_health_check()
                formatted = format_health_report(report)
                sg.popup_scrolled(formatted, title="System Health",
                                   size=(60, 20), background_color=BG_COLOR,
                                   text_color="#c7d0ff", font=FONT_MONO_SM)

            # ── Preview ──────────────────────────────────────────────
            elif event == "-OPEN-PREVIEW-":
                if self._last_image_path and os.path.isfile(self._last_image_path):
                    os.startfile(self._last_image_path)

            # ── Copy ─────────────────────────────────────────────────
            elif event == "-COPY-":
                pyperclip.copy(values["-OUTPUT-"])
                self.window["-STATUS-"].update("Copied to clipboard")

        self.window.close()
