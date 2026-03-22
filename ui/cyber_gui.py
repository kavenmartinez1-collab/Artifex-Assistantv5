"""
Artifex Assistant V5 — Cyberpunk GUI application.
Universal AI hosting with persistent knowledge base.
"""

import gc
import threading
import traceback
import os

import pyperclip
import FreeSimpleGUI as sg

from core.config import (
    MODES, get_context_profile, set_context_profile, get_context_profile_name,
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
    apply_theme, BG_COLOR, OUTPUT_BG,
    FONT_MAIN, FONT_TITLE, FONT_MONO, FONT_MONO_SM,
    FONT_SMALL,
)


_PIPELINE_MODES = ["Chat", "Image Gen", "Vision", "3D (ShapE)", "Audio TTS", "Audio STT"]
_PIPELINE_MAP = {
    "Chat": "text-generation",
    "Image Gen": "text-to-image",
    "Vision": "image-text-to-text",
    "3D (ShapE)": "shap-e",
    "Audio TTS": "text-to-audio",
    "Audio STT": "automatic-speech-recognition",
}
# Modes that need a file input
_FILE_INPUT_MODES = {"Vision", "Audio STT"}
# Modes that need a prompt
_PROMPT_MODES = {"Chat", "Image Gen", "Vision", "3D (ShapE)", "Audio TTS"}


class ArtifexGUI:
    def __init__(self):
        apply_theme()

        self.engine = None
        self._pipeline = None  # active non-chat pipeline
        self.busy = False

        # Knowledge manager
        self.km = KnowledgeManager()
        self.km.set_workspace(os.getcwd())
        self.km.bind_workspace_store(os.getcwd())
        self.session_map = SessionMap()

        self.messages = [{"role": "system", "content": self._get_system_prompt()}]

        self._pending_agent_actions = []
        self._pending_commands = []

        # Ensure output dir exists
        self._output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        os.makedirs(self._output_dir, exist_ok=True)

        self.window = self._build_layout()

    def _build_layout(self):
        title_row = [
            sg.Push(),
            sg.Text("ARTIFEX ASSISTANT V5", font=FONT_TITLE, text_color="#00f0ff", key="-TITLE-"),
            sg.Push(),
        ]

        ws_bar = [
            sg.Text("CWD:", text_color="#00f0ff", font=FONT_SMALL),
            sg.Input(os.getcwd(), size=(40, 1), key="-WS-PATH-", font=FONT_MONO_SM,
                     background_color="#161a2b", text_color="#00f0ff"),
            sg.Button("Set", key="-WS-SET-",
                      button_color=("#0f111a", "#00f0ff"), size=(4, 1)),
            sg.Button("Scan", key="-WS-SCAN-",
                      button_color=("#0f111a", "#00f0ff"), size=(5, 1)),
        ]

        cmd_panel = [
            sg.pin(sg.Column([
                [sg.Text("Suggested actions:", text_color="#00f0ff", font=FONT_SMALL)],
                [sg.Listbox(
                    values=[], size=(40, 4), key="-CMD-LIST-",
                    font=FONT_MONO_SM, background_color="#161a2b", text_color="#9efeff",
                    select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE,
                    expand_x=True,
                )],
                [
                    sg.Button("Run Selected", key="-CMD-RUN-",
                              button_color=("#0f111a", "#00f0ff"), size=(12, 1)),
                    sg.Button("Run All", key="-CMD-ALL-",
                              button_color=("#0f111a", "#00f0ff"), size=(8, 1)),
                    sg.Button("Skip", key="-CMD-SKIP-",
                              button_color=("#0f111a", "#444444"), size=(6, 1)),
                    sg.Push(),
                    sg.Text("", key="-CMD-STATUS-", text_color="#00f0ff", font=FONT_MONO_SM),
                ],
            ], key="-CMD-PANEL-", visible=False, background_color=BG_COLOR, expand_x=True))
        ]

        # Output + Thinking side by side
        output_col = sg.Column([
            [
                sg.Text("OUTPUT:", font=FONT_MAIN, text_color="#00f0ff", key="-OUTPUT-LABEL-"),
                sg.Push(),
                sg.Button("COPY", key="-COPY-"),
            ],
            [sg.Multiline(
                size=(55, 15), key="-OUTPUT-", font=FONT_MONO,
                background_color=OUTPUT_BG, text_color="#c7d0ff",
                expand_x=True, expand_y=True, disabled=True, autoscroll=True,
            )],
        ], expand_x=True, expand_y=True)

        thinking_col = sg.Column([
            [sg.Text("THINKING:", font=FONT_SMALL, text_color="#b060ff")],
            [sg.Multiline(
                size=(25, 15), key="-THINKING-", font=FONT_MONO_SM,
                background_color="#1a1030", text_color="#9070cc",
                expand_x=True, expand_y=True, disabled=True, autoscroll=True,
            )],
        ], expand_x=True, expand_y=True)

        # Pipeline mode selector + file browse row
        pipeline_row = [
            sg.Text("Mode:", font=FONT_SMALL, text_color="#f9f871"),
            sg.Combo(
                _PIPELINE_MODES, default_value="Chat",
                key="-PIPELINE-MODE-", size=(14, 1), font=FONT_SMALL,
                readonly=True, background_color="#161a2b", text_color="#00f0ff",
                enable_events=True,
            ),
            sg.VSeparator(),
            sg.Text("Input file:", font=FONT_SMALL, text_color="#888888", key="-FILE-LABEL-",
                     visible=False),
            sg.Input("", size=(35, 1), key="-FILE-PATH-", font=FONT_MONO_SM,
                     background_color="#161a2b", text_color="#00f0ff", visible=False),
            sg.FileBrowse("Browse", key="-FILE-BROWSE-",
                          button_color=("#0f111a", "#00f0ff"), size=(8, 1),
                          target="-FILE-PATH-", visible=False,
                          file_types=(("All Files", "*.*"), ("Images", "*.png *.jpg *.jpeg *.webp *.bmp"),
                                      ("Audio", "*.wav *.mp3 *.flac"))),
            sg.VSeparator(),
            sg.Text(f"Output: output/", font=FONT_SMALL, text_color="#888888", key="-OUTPUT-DIR-LABEL-",
                     visible=False),
            sg.FolderBrowse("Set Output", key="-OUTPUT-BROWSE-",
                            button_color=("#0f111a", "#444444"), size=(10, 1),
                            target="-OUTPUT-DIR-PATH-", visible=False),
            sg.Input(key="-OUTPUT-DIR-PATH-", visible=False, enable_events=True),
        ]

        layout = [
            title_row,
            ws_bar,
            pipeline_row,
            [sg.HSeparator()],
            [sg.Text("INPUT", font=FONT_MAIN, text_color="#00f0ff", key="-INPUT-LABEL-")],
            [sg.Multiline(
                size=(40, 3), key="-PROMPT-", font=FONT_MONO,
                background_color="#161a2b", text_color="#9efeff",
                focus=True, expand_x=True,
            )],
            [
                sg.Text("Tokens:", text_color="#f9f871"),
                sg.Input(str(MODES["ASSISTANT"].max_tokens), size=(6, 1), key="-TOKENS-"),
                sg.VSeparator(),
                sg.Text("Temp:", text_color="#f9f871"),
                sg.Slider((0.1, 1.5), 0.7, 0.1, orientation="h", size=(10, 15), key="-TEMP-"),
                sg.Push(),
                sg.Button("EXECUTE", key="-RUN-",
                          button_color=("#0f111a", "#ff2a6d"), bind_return_key=True),
                sg.Button("REFRESH", key="-REFRESH-",
                          button_color=("#0f111a", "#00f0ff"), size=(8, 1)),
                sg.Button("CLEAR", key="-RESET-", button_color=("#0f111a", "#444444")),
                sg.Button("VRAM RELIEF", key="-VRAM-RELIEF-",
                          button_color=("#0f111a", "#ff6b6b"), size=(12, 1),
                          tooltip="Unload model + compress history to free VRAM. Keeps conversation. Model reloads on next message."),
                sg.VSeparator(),
                sg.Button(f"CTX: {get_context_profile_name()}", key="-CONTEXT-TOGGLE-",
                          button_color=("#0f111a", "#f9f871"), size=(14, 1), font=FONT_SMALL,
                          tooltip="Toggle STANDARD/HIGH context profile"),
                sg.VSeparator(),
                sg.Text("Backend:", font=FONT_SMALL, text_color="#888888"),
                sg.Combo(
                    ["transformers", "ollama"],
                    default_value=get_active_backend(),
                    key="-BACKEND-SELECT-",
                    size=(12, 1),
                    font=FONT_SMALL,
                    readonly=True,
                    background_color="#161a2b",
                    text_color="#00f0ff",
                    enable_events=True,
                ),
                sg.Text("Model:", font=FONT_SMALL, text_color="#888888"),
                sg.Combo(
                    get_model_names() if get_model_names() else [get_active_model_name()],
                    default_value=get_active_model_name(),
                    key="-MODEL-SELECT-",
                    size=(22, 1),
                    font=FONT_SMALL,
                    readonly=True,
                    background_color="#161a2b",
                    text_color="#00f0ff",
                    enable_events=True,
                ),
            ],
            [sg.HSeparator()],
            [output_col, sg.VSeparator(), thinking_col],
            cmd_panel,
            [sg.StatusBar("STANDBY", key="-STATUS-",
                          background_color="#161a2b", text_color="#ff2a6d")],
        ]

        return sg.Window(
            "Artifex Assistant V5", layout, resizable=True, finalize=True,
            size=(1200, 700),
        )

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
        """Compress old messages into key-point summaries and free VRAM.
        Keeps the model loaded — much faster than CLEAR."""
        mode_cfg = MODES["ASSISTANT"]
        old_count = len(self.messages) - 1  # exclude system prompt
        self.messages = compress_history(self.messages, mode_cfg.context_window)
        new_count = len(self.messages) - 1

        # Free VRAM without unloading the model
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
            f"REFRESHED — {old_count} messages compressed to {new_count} (VRAM freed)"
        )

    def _run_assistant_actions(self, actions=None):
        """Execute agent actions."""
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
                # Process through knowledge engine
                if output:
                    kb_ids = self.km.process_tool_result(action.type, action.display, output)
                    if kb_ids:
                        self.window["-OUTPUT-"].update(
                            f"[KB] {len(kb_ids)} entries\n", append=True
                        )

                # Update session map BEFORE caching (needs full output)
                if output:
                    update_session_map(self.session_map, action.type, action.display, output)

                # Cache large outputs — model gets summary, full output on disk
                result = output if output else "(no output — command completed successfully)"
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

            # Build display history
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

            # Clear thinking panel for new generation
            self.window["-THINKING-"].update("")

            def on_response(text):
                self.window["-OUTPUT-"].update(text, append=True, autoscroll=True)

            def on_thinking(text):
                self.window["-THINKING-"].update(text, append=True, autoscroll=True)

            think_filter = ThinkFilter(on_response=on_response, on_thinking=on_thinking)

            response = self.engine.generate_streaming(
                active_messages,
                max_tokens=max_tokens,
                temperature=temp,
                on_token=think_filter.feed,
            )
            think_filter.flush()

            self.messages.append({"role": "assistant", "content": response})
            self.window["-OUTPUT-"].update("\n\n", append=True)
            self.window["-STATUS-"].update("READY")

            # Extract knowledge from AI response (don't let failures block tool detection)
            try:
                self.km.add_from_ai_response(response)
            except Exception:
                pass

            # Check for agent actions in response
            actions = extract_agent_actions(response)
            if actions:
                self._pending_agent_actions = actions
                display = [f"[{a.type.upper()}] {a.display}" for a in actions]
                self._pending_commands = [a.content for a in actions]
                self.window["-CMD-LIST-"].update(values=display)
                self.window["-CMD-PANEL-"].update(visible=True)

        except Exception as e:
            traceback.print_exc()  # full traceback to console
            self.window["-OUTPUT-"].update(
                f"\nERROR ({type(e).__name__}): {e}", append=True
            )
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
                    "[TOOL OUTPUT — this is automated command output, not a human message]\n\n"
                    f"{truncated}\n\n"
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
                    active_messages,
                    max_tokens=mode_cfg.max_tokens,
                    temperature=mode_cfg.temperature,
                    on_token=think_filter.feed,
                )
                think_filter.flush()

                self.messages.append({"role": "assistant", "content": response})
                self.window["-OUTPUT-"].update("\n\n", append=True)
                self.window["-STATUS-"].update("READY")

                # Extract knowledge from followup response
                self.km.add_from_ai_response(response)

                actions = extract_agent_actions(response)
                if actions:
                    self._pending_agent_actions = actions
                    display = [f"[{a.type.upper()}] {a.display}" for a in actions]
                    self._pending_commands = [a.content for a in actions]
                    self.window["-CMD-LIST-"].update(values=display)
                    self.window["-CMD-PANEL-"].update(visible=True)

            except Exception as e:
                traceback.print_exc()  # full traceback to console
                self.window["-OUTPUT-"].update(
                    f"\nERROR ({type(e).__name__}): {e}", append=True
                )
            finally:
                self.busy = False

        threading.Thread(target=_analysis_thread, daemon=True).start()

    def _pipeline_thread(self, mode, prompt, values):
        """Run a non-chat pipeline (image gen, vision, 3D, audio)."""
        try:
            from core.pipelines.registry import create_pipeline
            import time

            pipeline_type = _PIPELINE_MAP.get(mode, "text-generation")
            self.window["-STATUS-"].update(f"Loading {mode} pipeline...")
            self.window["-OUTPUT-"].update(f"[{mode}] Starting...\n")

            pipeline = create_pipeline(pipeline_type)

            # Determine model path — use HF defaults for non-text pipelines
            model_defaults = {
                "text-to-image": "runwayml/stable-diffusion-v1-5",
                "shap-e": "openai/shap-e",
                "image-text-to-text": None,  # uses loaded transformers model
                "text-to-audio": "suno/bark-small",
                "automatic-speech-recognition": "openai/whisper-small",
            }
            model_path = model_defaults.get(pipeline_type, "")

            # For vision, try to use the currently active transformers model
            if pipeline_type == "image-text-to-text":
                from core.config import get_active_model_path
                model_path = get_active_model_path()

            self.window["-OUTPUT-"].update(f"Model: {model_path}\n", append=True)
            self.window["-STATUS-"].update(f"Loading model...")

            try:
                pipeline.load(model_path, status_callback=lambda msg: self.window["-STATUS-"].update(msg))
            except Exception as e:
                self.window["-OUTPUT-"].update(f"\nFailed to load model: {e}\n", append=True)
                self.window["-OUTPUT-"].update(
                    f"\nTo use this pipeline, download the model first:\n"
                    f"  python download_model.py --repo {model_path}\n", append=True
                )
                self.window["-STATUS-"].update(f"LOAD FAILED")
                return

            self.window["-STATUS-"].update(f"Running {mode}...")

            # Build kwargs based on mode
            kwargs = {}
            timestamp = int(time.time())

            if mode == "Image Gen":
                kwargs = {
                    "prompt": prompt,
                    "width": 512,
                    "height": 512,
                    "num_steps": 30,
                    "output_path": os.path.join(self._output_dir, f"image_{timestamp}.png"),
                }

            elif mode == "Vision":
                kwargs = {
                    "image_path": values.get("-FILE-PATH-", ""),
                    "prompt": prompt or "Describe this image in detail.",
                    "max_tokens": 512,
                }

            elif mode == "3D (ShapE)":
                kwargs = {
                    "prompt": prompt,
                    "num_steps": 64,
                    "output_path": os.path.join(self._output_dir, f"mesh_{timestamp}.ply"),
                }

            elif mode == "Audio TTS":
                kwargs = {
                    "text": prompt,
                    "output_path": os.path.join(self._output_dir, f"audio_{timestamp}.wav"),
                }

            elif mode == "Audio STT":
                kwargs = {
                    "audio_path": values.get("-FILE-PATH-", ""),
                }

            self.window["-OUTPUT-"].update(f"Processing...\n", append=True)
            result = pipeline.run(**kwargs)

            if result.success:
                self.window["-OUTPUT-"].update(f"\nSUCCESS\n", append=True)
                if result.output_type == "text":
                    self.window["-OUTPUT-"].update(f"\n{result.content}\n", append=True)
                elif result.output_type == "image":
                    saved = result.metadata.get("saved_to", "")
                    self.window["-OUTPUT-"].update(f"Image saved to: {saved}\n", append=True)
                elif result.output_type == "mesh":
                    self.window["-OUTPUT-"].update(f"Mesh saved to: {result.content}\n", append=True)
                elif result.output_type == "audio":
                    self.window["-OUTPUT-"].update(f"Audio saved to: {result.content}\n", append=True)

                if result.metadata:
                    self.window["-OUTPUT-"].update(f"\nDetails: {result.metadata}\n", append=True)
            else:
                self.window["-OUTPUT-"].update(f"\nERROR: {result.error}\n", append=True)

            # Unload pipeline to free VRAM
            pipeline.unload(status_callback=lambda msg: self.window["-STATUS-"].update(msg))
            self.window["-STATUS-"].update("READY")

        except Exception as e:
            traceback.print_exc()
            self.window["-OUTPUT-"].update(f"\nERROR ({type(e).__name__}): {e}", append=True)
            self.window["-STATUS-"].update("ERROR")
        finally:
            self.busy = False

    def run(self):
        while True:
            event, values = self.window.read()
            if event == sg.WIN_CLOSED:
                break

            if event == "-CMD-RUN-" and self._pending_agent_actions:
                selected_indices = self.window["-CMD-LIST-"].get_indexes()
                if selected_indices:
                    selected_actions = [self._pending_agent_actions[i] for i in selected_indices]
                    tool_output = self._run_assistant_actions(selected_actions)
                    if tool_output:
                        self._feed_tool_output(tool_output)

            if event == "-CMD-ALL-" and self._pending_agent_actions:
                tool_output = self._run_assistant_actions()
                if tool_output:
                    self._feed_tool_output(tool_output)

            if event == "-CMD-SKIP-":
                self.window["-CMD-PANEL-"].update(visible=False)
                self._pending_commands = []
                self._pending_agent_actions = []

            if event == "-PIPELINE-MODE-":
                mode = values["-PIPELINE-MODE-"]
                needs_file = mode in _FILE_INPUT_MODES
                needs_output = mode != "Chat"
                # Show/hide file browse elements
                for key in ("-FILE-LABEL-", "-FILE-PATH-", "-FILE-BROWSE-"):
                    self.window[key].update(visible=needs_file)
                for key in ("-OUTPUT-DIR-LABEL-", "-OUTPUT-BROWSE-"):
                    self.window[key].update(visible=needs_output)
                self.window["-STATUS-"].update(f"Mode: {mode}")

            if event == "-OUTPUT-DIR-PATH-":
                new_dir = values["-OUTPUT-DIR-PATH-"]
                if new_dir and os.path.isdir(new_dir):
                    self._output_dir = new_dir
                    self.window["-OUTPUT-DIR-LABEL-"].update(f"Output: {os.path.basename(new_dir)}/")

            if event == "-RUN-":
                if self.busy:
                    continue
                prompt = values["-PROMPT-"]
                mode = values["-PIPELINE-MODE-"]

                # Non-chat pipeline modes
                if mode != "Chat":
                    if mode in _PROMPT_MODES and not prompt.strip():
                        self.window["-STATUS-"].update("Prompt is required.")
                        continue
                    if mode in _FILE_INPUT_MODES:
                        file_path = values["-FILE-PATH-"]
                        if not file_path or not os.path.isfile(file_path):
                            self.window["-STATUS-"].update("Select an input file first.")
                            continue
                    self.busy = True
                    self.window["-STATUS-"].update(f"RUNNING {mode.upper()}...")
                    self.window["-PROMPT-"].update("")
                    threading.Thread(
                        target=self._pipeline_thread,
                        args=(mode, prompt.strip(), values),
                        daemon=True,
                    ).start()
                    continue

                if not prompt.strip():
                    continue

                self.busy = True
                self.window["-STATUS-"].update("GENERATING...")
                self.window["-PROMPT-"].update("")
                max_tokens = int(values["-TOKENS-"])
                temp = values["-TEMP-"]
                threading.Thread(
                    target=self._generate_thread,
                    args=(prompt, max_tokens, temp),
                    daemon=True,
                ).start()

            elif event == "-REFRESH-":
                if not self.busy:
                    self.refresh_context()

            elif event == "-RESET-":
                if self.engine is not None:
                    self.engine.unload(
                        status_callback=lambda msg: self.window["-STATUS-"].update(msg)
                    )
                    self.engine = None
                self.session_map.clear()
                clear_cache()
                gc.collect()
                self.reset_history()

            elif event == "-CONTEXT-TOGGLE-":
                current = get_context_profile_name()
                new_name = "HIGH" if current == "STANDARD" else "STANDARD"
                set_context_profile(new_name)
                profile = get_context_profile()
                self.window["-CONTEXT-TOGGLE-"].update(f"CTX: {new_name}")
                self.window["-TOKENS-"].update(str(profile.max_output_tokens))
                self.window["-STATUS-"].update(
                    f"Context: {new_name} | output={profile.max_output_tokens} "
                    f"history={profile.max_history_tokens}"
                )

            elif event == "-VRAM-RELIEF-":
                mode_cfg = MODES["ASSISTANT"]
                pressured, frac, alloc, total = check_vram_pressure(threshold=0.0)
                self.window["-STATUS-"].update(
                    f"VRAM: {alloc:.1f}/{total:.1f} GB ({frac:.0%}) — unloading model..."
                )
                self.window.refresh()
                if self.engine is not None:
                    relieved, msg = vram_pressure_relief(
                        self.engine, self.messages, mode_cfg.context_window
                    )
                    if relieved:
                        self.messages[0] = {"role": "system", "content": self._get_system_prompt()}
                        self.engine = None  # force re-create on next message
                        self.window["-STATUS-"].update(f"VRAM RELIEF: {msg}")
                    else:
                        if self.engine.is_loaded():
                            self.engine.unload()
                        self.engine = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        _, post_frac, post_gb, _ = check_vram_pressure(threshold=1.0)
                        self.window["-STATUS-"].update(
                            f"Model unloaded. VRAM: {post_gb:.1f}/{total:.1f} GB ({post_frac:.0%}). "
                            f"Will reload on next message."
                        )
                else:
                    self.window["-STATUS-"].update("No engine loaded.")

            elif event == "-BACKEND-SELECT-":
                new_backend = values["-BACKEND-SELECT-"]
                if new_backend != get_active_backend():
                    if self.engine is not None:
                        self.engine.unload(
                            status_callback=lambda msg: self.window["-STATUS-"].update(msg)
                        )
                        self.engine = None
                    set_active_backend(new_backend)
                    # Refresh model dropdown for new backend
                    models = get_model_names()
                    if models:
                        self.window["-MODEL-SELECT-"].update(
                            values=models, value=models[0]
                        )
                    else:
                        self.window["-MODEL-SELECT-"].update(
                            values=[get_active_model_name()],
                            value=get_active_model_name()
                        )
                    self.window["-STATUS-"].update(
                        f"Backend: {new_backend} (will load on next message)"
                    )

            elif event == "-MODEL-SELECT-":
                new_model = values["-MODEL-SELECT-"]
                if set_active_model(new_model):
                    if self.engine is not None and self.engine.is_loaded():
                        self.engine.unload(
                            status_callback=lambda msg: self.window["-STATUS-"].update(msg)
                        )
                        self.engine = None
                    self.window["-STATUS-"].update(
                        f"Model: {new_model} (will load on next message)"
                    )
                else:
                    self.window["-MODEL-SELECT-"].update(value=get_active_model_name())
                    self.window["-STATUS-"].update(
                        f"Invalid model: {new_model}. Staying on {get_active_model_name()}"
                    )

            elif event == "-WS-SET-":
                ws_path = values["-WS-PATH-"].strip()
                if ws_path and os.path.isdir(ws_path):
                    os.chdir(ws_path)
                    self.km.set_workspace(ws_path)
                    self.km.bind_workspace_store(ws_path)
                    self.window["-STATUS-"].update(f"Workspace: {ws_path}")
                    self.window["-OUTPUT-"].update(
                        f"[CWD changed to {ws_path}]\n" +
                        self.km.get_workspace_summary() + "\n", append=True
                    )
                else:
                    self.window["-STATUS-"].update(f"Directory not found: {ws_path}")

            elif event == "-WS-SCAN-":
                self.km.rescan_workspace()
                self.window["-OUTPUT-"].update(
                    f"[Workspace re-scanned]\n{self.km.get_workspace_summary()}\n",
                    append=True,
                )

            elif event == "-COPY-":
                pyperclip.copy(values["-OUTPUT-"])

        self.window.close()
