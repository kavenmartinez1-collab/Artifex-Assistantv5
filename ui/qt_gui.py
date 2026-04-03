"""
Artifex Assistant V5 — PyQt6 main GUI window.
Replaces the FreeSimpleGUI (Tkinter) GUI with a production-grade Qt interface.
Supports full multimodal I/O: text, images, audio, video, 3D, with
proper threading, token batching, and rich inline media display.
"""

import gc
import os
import sys
import threading

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QComboBox, QLineEdit, QTextEdit, QPlainTextEdit,
    QTabWidget, QGroupBox, QCheckBox, QSlider, QProgressBar,
    QListWidget, QStatusBar, QFrame, QFileDialog, QMessageBox,
    QSizePolicy, QApplication,
)

from ui.qt_theme import (
    THEMES, get_theme_names, get_active_theme, set_active_theme,
    get_theme, generate_stylesheet, vram_color,
)
from ui.qt_widgets import (
    DropZone, ImageViewer, AudioPlayer, VideoPlayer,
    MicrophoneRecorder, ChatView, ChatBubble,
)
from ui.qt_workers import GenerationWorker, PipelineWorker, ActionWorker, TokenBatcher

from core.config import (
    MODES, get_active_backend, set_active_backend, get_model_names,
    get_active_model_name, set_active_model, get_context_profile,
    get_context_profile_name, set_context_profile,
    get_torch_compile, set_torch_compile, get_turboquant_kv, set_turboquant_kv,
    BASE_DIR,
)
from core.engine_factory import create_engine
from core.services import get_service
from core.knowledge import KnowledgeManager
from tools.tool_cache import SessionMap, clear_cache, update_session_map, maybe_cache_output
from tools.agent_tools import get_assistant_tools_prompt, get_tool_output_limit


# Pipeline mode definitions
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
_PROMPT_MODES = {"Chat", "Code", "Image Gen", "Image Edit", "Vision",
                 "3D (ShapE)", "Audio TTS", "Music Gen", "Video Gen"}


class ArtifexMainWindow(QMainWindow):
    """Main application window for Artifex Assistant V5."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Artifex Assistant V5")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # State
        self.engine = None
        self.messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        self._busy = False
        self._cancel_event = threading.Event()
        self._current_worker = None
        self._pending_actions = []
        self._current_bubble = None
        self._output_dir = os.path.join(BASE_DIR, "output")

        # Knowledge & context (matches old GUI)
        self.km = KnowledgeManager()
        self.session_map = SessionMap()
        self._system_info = get_assistant_tools_prompt()

        # Build UI
        self._build_ui()
        self._connect_signals()
        self._setup_resource_timer()
        self._update_mode_visibility()

        # Apply theme
        self._apply_theme()

    # ═══════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Main splitter: sidebar | center | thinking panel
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self._splitter)

        self._build_sidebar()
        self._build_center()
        self._build_thinking_panel()

        self._splitter.setSizes([240, 800, 280])
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)

        # Status bar
        self._build_status_bar()

    def _build_sidebar(self):
        sidebar = QWidget()
        sidebar.setMaximumWidth(280)
        sidebar.setMinimumWidth(200)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Title
        title = QLabel("ARTIFEX V5")
        title.setProperty("class", "title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Theme selector
        theme_layout = QHBoxLayout()
        theme_layout.addWidget(QLabel("Theme:"))
        self._theme_combo = QComboBox()
        self._theme_combo.addItems(get_theme_names())
        self._theme_combo.setCurrentText(get_active_theme())
        theme_layout.addWidget(self._theme_combo)
        layout.addLayout(theme_layout)

        # Backend group
        backend_group = QGroupBox("Backend")
        bg_layout = QVBoxLayout(backend_group)

        be_layout = QHBoxLayout()
        be_layout.addWidget(QLabel("Engine:"))
        self._backend_combo = QComboBox()
        self._backend_combo.addItems(["transformers", "ollama"])
        self._backend_combo.setCurrentText(get_active_backend())
        be_layout.addWidget(self._backend_combo)
        bg_layout.addLayout(be_layout)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.addItems(get_model_names())
        model_layout.addWidget(self._model_combo)
        bg_layout.addLayout(model_layout)

        self._ctx_btn = QPushButton(f"CTX: {get_context_profile_name()}")
        self._ctx_btn.setProperty("class", "secondary")
        bg_layout.addWidget(self._ctx_btn)

        layout.addWidget(backend_group)

        # Optimizations
        opt_group = QGroupBox("Optimizations")
        opt_layout = QVBoxLayout(opt_group)
        self._torch_compile_cb = QCheckBox("torch.compile()")
        self._torch_compile_cb.setChecked(get_torch_compile())
        opt_layout.addWidget(self._torch_compile_cb)
        self._turboquant_cb = QCheckBox("TurboQuant KV")
        self._turboquant_cb.setChecked(get_turboquant_kv())
        opt_layout.addWidget(self._turboquant_cb)
        layout.addWidget(opt_group)

        # Workspace
        ws_group = QGroupBox("Workspace")
        ws_layout = QVBoxLayout(ws_group)
        self._ws_path = QLineEdit()
        self._ws_path.setPlaceholderText("Workspace directory...")
        ws_layout.addWidget(self._ws_path)
        ws_btn_layout = QHBoxLayout()
        self._ws_set_btn = QPushButton("Set")
        self._ws_scan_btn = QPushButton("Scan")
        ws_btn_layout.addWidget(self._ws_set_btn)
        ws_btn_layout.addWidget(self._ws_scan_btn)
        ws_layout.addLayout(ws_btn_layout)
        layout.addWidget(ws_group)

        # Session
        session_group = QGroupBox("Session")
        sess_layout = QHBoxLayout(session_group)
        self._save_btn = QPushButton("Save")
        self._load_btn = QPushButton("Load")
        self._export_btn = QPushButton("Export")
        sess_layout.addWidget(self._save_btn)
        sess_layout.addWidget(self._load_btn)
        sess_layout.addWidget(self._export_btn)
        layout.addWidget(session_group)

        # Health button
        self._health_btn = QPushButton("System Health")
        self._health_btn.setProperty("class", "secondary")
        layout.addWidget(self._health_btn)

        layout.addStretch()
        self._splitter.addWidget(sidebar)

    def _build_center(self):
        center = QWidget()
        layout = QVBoxLayout(center)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Pipeline mode bar
        mode_bar = QHBoxLayout()
        mode_bar.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(_PIPELINE_MODES)
        self._mode_combo.setMinimumWidth(120)
        mode_bar.addWidget(self._mode_combo)
        mode_bar.addStretch()

        # Drop zone (for file input modes)
        self._drop_zone = DropZone()
        mode_bar.addWidget(self._drop_zone, 1)

        # Mic recorder (for audio modes)
        self._mic_recorder = MicrophoneRecorder(
            output_dir=os.path.join(BASE_DIR, "output")
        )
        self._mic_recorder.setMaximumWidth(300)
        mode_bar.addWidget(self._mic_recorder)

        layout.addLayout(mode_bar)

        # Output tabs
        self._tabs = QTabWidget()

        # Chat tab — rich chat view
        self._chat_view = ChatView()
        self._tabs.addTab(self._chat_view, "Chat")

        # Preview tab — image preview
        self._preview_viewer = ImageViewer(max_width=640, max_height=480)
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.addWidget(self._preview_viewer, 1,
                                 Qt.AlignmentFlag.AlignCenter)
        self._preview_label = QLabel("")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self._preview_label)
        self._tabs.addTab(preview_container, "Preview")

        # Audio tab
        self._audio_player = AudioPlayer()
        audio_container = QWidget()
        audio_layout = QVBoxLayout(audio_container)
        audio_layout.addWidget(self._audio_player)
        audio_layout.addStretch()
        self._tabs.addTab(audio_container, "Audio")

        # Video tab
        self._video_player = VideoPlayer()
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.addWidget(self._video_player)
        video_layout.addStretch()
        self._tabs.addTab(video_container, "Video")

        layout.addWidget(self._tabs, 1)

        # Action panel (suggested commands — hidden by default)
        self._action_panel = QFrame()
        self._action_panel.setVisible(False)
        ap_layout = QVBoxLayout(self._action_panel)
        ap_layout.setContentsMargins(4, 4, 4, 4)
        ap_label = QLabel("Suggested Actions")
        ap_label.setProperty("class", "accent")
        ap_layout.addWidget(ap_label)

        self._action_list = QListWidget()
        self._action_list.setMaximumHeight(100)
        ap_layout.addWidget(self._action_list)

        ap_btn_layout = QHBoxLayout()
        self._action_run_btn = QPushButton("Run Selected")
        self._action_all_btn = QPushButton("Run All")
        self._action_skip_btn = QPushButton("Skip")
        self._action_skip_btn.setProperty("class", "secondary")
        ap_btn_layout.addWidget(self._action_run_btn)
        ap_btn_layout.addWidget(self._action_all_btn)
        ap_btn_layout.addWidget(self._action_skip_btn)
        ap_layout.addLayout(ap_btn_layout)

        layout.addWidget(self._action_panel)

        # Input area
        input_frame = QFrame()
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(0, 0, 0, 0)

        self._prompt_input = QPlainTextEdit()
        self._prompt_input.setPlaceholderText("Enter your message...")
        self._prompt_input.setMaximumHeight(100)
        self._prompt_input.setFont(QFont("Consolas", 11))
        input_layout.addWidget(self._prompt_input)

        # Controls row
        ctrl_layout = QHBoxLayout()

        ctrl_layout.addWidget(QLabel("Tokens:"))
        self._tokens_input = QLineEdit(str(MODES["ASSISTANT"].max_tokens))
        self._tokens_input.setMaximumWidth(60)
        ctrl_layout.addWidget(self._tokens_input)

        ctrl_layout.addWidget(QLabel("Temp:"))
        self._temp_slider = QSlider(Qt.Orientation.Horizontal)
        self._temp_slider.setRange(1, 15)  # 0.1 to 1.5
        self._temp_slider.setValue(7)       # 0.7
        self._temp_slider.setMaximumWidth(100)
        ctrl_layout.addWidget(self._temp_slider)
        self._temp_label = QLabel("0.7")
        self._temp_label.setMinimumWidth(30)
        ctrl_layout.addWidget(self._temp_label)

        ctrl_layout.addStretch()

        self._execute_btn = QPushButton("EXECUTE")
        self._execute_btn.setMinimumWidth(120)
        self._execute_btn.setMinimumHeight(32)
        ctrl_layout.addWidget(self._execute_btn)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setProperty("class", "secondary")
        ctrl_layout.addWidget(self._refresh_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setProperty("class", "secondary")
        ctrl_layout.addWidget(self._clear_btn)

        self._vram_btn = QPushButton("VRAM")
        self._vram_btn.setProperty("class", "secondary")
        ctrl_layout.addWidget(self._vram_btn)

        input_layout.addLayout(ctrl_layout)
        layout.addWidget(input_frame)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumHeight(16)
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._splitter.addWidget(center)

    def _build_thinking_panel(self):
        thinking = QWidget()
        thinking.setMaximumWidth(350)
        thinking.setMinimumWidth(200)
        layout = QVBoxLayout(thinking)
        layout.setContentsMargins(4, 4, 4, 4)

        label = QLabel("Thinking")
        label.setProperty("class", "accent")
        layout.addWidget(label)

        self._thinking_output = QPlainTextEdit()
        self._thinking_output.setReadOnly(True)
        self._thinking_output.setFont(QFont("Consolas", 9))
        layout.addWidget(self._thinking_output)

        self._splitter.addWidget(thinking)

    def _build_status_bar(self):
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._status_label = QLabel("STANDBY")
        self._status_label.setProperty("class", "status")
        self._status_bar.addWidget(self._status_label, 1)

        self._copy_btn = QPushButton("COPY")
        self._copy_btn.setMaximumWidth(60)
        self._copy_btn.setProperty("class", "secondary")
        self._status_bar.addPermanentWidget(self._copy_btn)

        self._resource_label = QLabel("")
        self._resource_label.setProperty("class", "status")
        self._status_bar.addPermanentWidget(self._resource_label)

    # ═══════════════════════════════════════════════════════════════════
    # SIGNAL CONNECTIONS
    # ═══════════════════════════════════════════════════════════════════

    def _connect_signals(self):
        # Theme
        self._theme_combo.currentTextChanged.connect(self._on_theme_changed)

        # Backend/Model
        self._backend_combo.currentTextChanged.connect(self._on_backend_changed)
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        self._ctx_btn.clicked.connect(self._on_ctx_toggle)

        # Optimizations
        self._torch_compile_cb.toggled.connect(
            lambda v: set_torch_compile(v))
        self._turboquant_cb.toggled.connect(
            lambda v: set_turboquant_kv(v))

        # Pipeline mode
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)

        # Mic
        self._mic_recorder.recording_finished.connect(self._on_recording_done)

        # Drop zone
        self._drop_zone.files_dropped.connect(self._on_files_dropped)

        # Execute
        self._execute_btn.clicked.connect(self._on_execute)
        self._refresh_btn.clicked.connect(self._on_refresh)
        self._clear_btn.clicked.connect(self._on_clear)
        self._vram_btn.clicked.connect(self._on_vram_relief)

        # Actions
        self._action_run_btn.clicked.connect(self._on_run_selected_actions)
        self._action_all_btn.clicked.connect(self._on_run_all_actions)
        self._action_skip_btn.clicked.connect(
            lambda: self._action_panel.setVisible(False))

        # Temperature slider
        self._temp_slider.valueChanged.connect(
            lambda v: self._temp_label.setText(f"{v / 10:.1f}")
        )

        # Health
        self._health_btn.clicked.connect(self._on_health)

        # Session
        self._save_btn.clicked.connect(self._on_save_session)
        self._load_btn.clicked.connect(self._on_load_session)
        self._export_btn.clicked.connect(self._on_export)

        # Workspace
        self._ws_set_btn.clicked.connect(self._on_ws_set)
        self._ws_scan_btn.clicked.connect(self._on_ws_scan)

        # Copy button
        self._copy_btn.clicked.connect(self._on_copy)

        # Enter key shortcut for execute (Ctrl+Enter to avoid conflict with multiline)
        from PyQt6.QtGui import QShortcut, QKeySequence
        self._enter_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        self._enter_shortcut.activated.connect(self._on_execute)

        # Token batchers for streaming output
        self._response_batcher = TokenBatcher(flush_interval_ms=50)
        self._response_batcher.batch_ready.connect(self._on_response_batch)
        self._thinking_batcher = TokenBatcher(flush_interval_ms=50)
        self._thinking_batcher.batch_ready.connect(self._on_thinking_batch)

    # ═══════════════════════════════════════════════════════════════════
    # THEME
    # ═══════════════════════════════════════════════════════════════════

    def _apply_theme(self, theme_name: str = None):
        if theme_name:
            set_active_theme(theme_name)
        qss = generate_stylesheet()
        QApplication.instance().setStyleSheet(qss)

    def _on_theme_changed(self, name):
        self._apply_theme(name)

    # ═══════════════════════════════════════════════════════════════════
    # BACKEND / MODEL
    # ═══════════════════════════════════════════════════════════════════

    def _on_backend_changed(self, backend):
        if self.engine:
            self.engine.unload()
            self.engine = None
        set_active_backend(backend)
        self._model_combo.clear()
        self._model_combo.addItems(get_model_names())
        self._set_status(f"Backend: {backend}")

    def _on_model_changed(self, model_name):
        if model_name:
            set_active_model(model_name)
            if self.engine and self.engine.is_loaded():
                self.engine.unload()
                self.engine = None

    def _on_ctx_toggle(self):
        name = get_context_profile_name()
        new = "HIGH" if name == "STANDARD" else "STANDARD"
        set_context_profile(new)
        self._ctx_btn.setText(f"CTX: {new}")
        profile = get_context_profile()
        self._tokens_input.setText(str(profile.max_output_tokens))

    # ═══════════════════════════════════════════════════════════════════
    # MODE SWITCHING
    # ═══════════════════════════════════════════════════════════════════

    def _on_mode_changed(self, mode):
        self._update_mode_visibility()

    def _update_mode_visibility(self):
        mode = self._mode_combo.currentText()
        needs_file = mode in _FILE_INPUT_MODES
        self._drop_zone.setVisible(needs_file or mode in {
            "Image Gen", "Vision", "Image Edit"
        })
        self._mic_recorder.setVisible(mode in {"Audio STT"})

    # ═══════════════════════════════════════════════════════════════════
    # EXECUTION
    # ═══════════════════════════════════════════════════════════════════

    def _on_execute(self):
        if self._busy:
            # Cancel
            self._cancel_event.set()
            self._execute_btn.setText("EXECUTE")
            self._set_status("Cancelling...")
            return

        mode = self._mode_combo.currentText()
        prompt = self._prompt_input.toPlainText().strip()

        if mode in _PROMPT_MODES and not prompt:
            self._set_status("Enter a prompt first")
            return

        max_tokens = int(self._tokens_input.text() or "2048")
        temperature = self._temp_slider.value() / 10.0

        self._busy = True
        self._cancel_event.clear()
        self._execute_btn.setText("CANCEL")
        self._thinking_output.clear()
        self._prompt_input.clear()

        if mode in ("Chat", "Code"):
            self._run_text_generation(prompt, max_tokens, temperature, mode)
        else:
            self._run_pipeline(mode, prompt)

    def _run_text_generation(self, prompt: str, max_tokens: int,
                             temperature: float, mode: str):
        """Start text generation in a background thread."""
        # Create engine if needed
        if self.engine is None:
            self.engine = create_engine()

        # Add user message
        self.messages.append({"role": "user", "content": prompt})

        # Update system prompt
        self._update_system_prompt(mode)

        # Build active messages
        from core.inference import build_active_messages
        mode_cfg = MODES.get(mode.upper(), MODES["ASSISTANT"])
        _, active_msgs = build_active_messages(
            self.messages, mode_cfg.context_window
        )

        # Show user message in chat
        user_bubble = self._chat_view.add_bubble("user")
        user_bubble.add_text(prompt)

        # Create assistant bubble for streaming
        self._current_bubble = self._chat_view.add_bubble("assistant")
        self._current_bubble.add_text("")

        # Launch worker
        worker = GenerationWorker(
            self.engine, active_msgs, max_tokens, temperature,
            enable_thinking=mode_cfg.enable_thinking,
            cancel_event=self._cancel_event,
        )
        worker.response_received.connect(self._response_batcher.add)
        worker.thinking_received.connect(self._thinking_batcher.add)
        worker.status_changed.connect(self._set_status)
        worker.finished.connect(self._on_generation_finished)
        worker.error.connect(self._on_generation_error)
        self._current_worker = worker
        worker.start()

    def _run_pipeline(self, mode: str, prompt: str):
        """Start a non-text pipeline in a background thread."""
        pipeline_type = _PIPELINE_MAP.get(mode)
        if not pipeline_type:
            self._set_status(f"Unknown mode: {mode}")
            self._busy = False
            self._execute_btn.setText("EXECUTE")
            return

        svc = get_service()
        kwargs = self._build_pipeline_kwargs(mode, prompt)

        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)  # Indeterminate

        worker = PipelineWorker(
            svc, pipeline_type, kwargs=kwargs,
            cancel_event=self._cancel_event,
        )
        worker.progress.connect(self._on_pipeline_progress)
        worker.result_ready.connect(self._on_pipeline_result)
        worker.error.connect(self._on_generation_error)
        worker.status_changed.connect(self._set_status)
        self._current_worker = worker
        worker.start()

    def _build_pipeline_kwargs(self, mode: str, prompt: str) -> dict:
        """Build kwargs for the pipeline based on mode and UI state."""
        output_dir = os.path.join(BASE_DIR, "output")
        kwargs = {}

        if mode == "Image Gen":
            kwargs = {"prompt": prompt, "width": 512, "height": 512,
                      "num_steps": 30}
        elif mode == "Image Edit":
            files = self._drop_zone.attached_files
            if files:
                kwargs = {"image_path": files[0], "prompt": prompt,
                          "strength": 0.75, "num_steps": 30}
        elif mode == "Vision":
            files = self._drop_zone.attached_files
            if files:
                kwargs = {"image_path": files[0], "prompt": prompt or "Describe this image in detail.",
                          "max_tokens": 512}
        elif mode == "3D (ShapE)":
            kwargs = {"prompt": prompt, "num_steps": 64}
        elif mode == "Audio TTS":
            kwargs = {"text": prompt}
        elif mode == "Audio STT":
            files = self._drop_zone.attached_files
            if files:
                kwargs = {"audio_path": files[0]}
        elif mode == "Music Gen":
            kwargs = {"prompt": prompt, "duration_seconds": 10}
        elif mode == "Video Gen":
            kwargs = {"prompt": prompt, "num_frames": 16, "fps": 8}

        return kwargs

    # ═══════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ═══════════════════════════════════════════════════════════════════

    def _on_response_batch(self, text):
        """Handle batched response tokens."""
        if self._current_bubble:
            self._current_bubble.append_text(text)
            self._chat_view.scroll_to_bottom()

    def _on_thinking_batch(self, text):
        """Handle batched thinking tokens."""
        cursor = self._thinking_output.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self._thinking_output.setTextCursor(cursor)
        self._thinking_output.ensureCursorVisible()

    def _on_generation_finished(self, response: str):
        """Generation complete — store response, extract knowledge, check for actions."""
        self._response_batcher.flush_now()
        self._thinking_batcher.flush_now()

        self.messages.append({"role": "assistant", "content": response})

        # Extract knowledge from AI response (matches old GUI)
        try:
            self.km.add_from_ai_response(response)
        except Exception:
            pass

        self._finish_busy("Generation complete")

        # Check for agent actions
        try:
            from tools.agent_tools import extract_agent_actions
            actions = extract_agent_actions(response)
            if actions:
                self._pending_actions = actions
                self._action_list.clear()
                for a in actions:
                    display = getattr(a, "display", str(a))
                    self._action_list.addItem(display)
                self._action_panel.setVisible(True)
        except Exception:
            pass

    def _on_generation_error(self, error_msg: str):
        """Handle generation error."""
        self._response_batcher.flush_now()
        self._thinking_batcher.flush_now()

        if self._current_bubble:
            self._current_bubble.append_text(f"\n\nERROR: {error_msg}")
        self._finish_busy("ERROR")

    def _on_pipeline_progress(self, current, total, message):
        """Update progress bar."""
        if total > 0:
            self._progress_bar.setRange(0, total)
            self._progress_bar.setValue(current)
        self._set_status(message)

    def _on_pipeline_result(self, result):
        """Handle pipeline result — display media inline."""
        from core.pipelines.base import PipelineResult

        self._progress_bar.setVisible(False)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)

        if not result.success:
            self._set_status(f"Pipeline error: {result.error}")
            bubble = self._chat_view.add_bubble("assistant")
            bubble.add_text(f"Error: {result.error}")
            self._finish_busy("Pipeline failed")
            return

        # Create result bubble
        bubble = self._chat_view.add_bubble("assistant")

        if result.output_type == "text":
            bubble.add_text(result.content or "(empty)")

        elif result.output_type == "image":
            path = result.metadata.get("stored_path") or result.metadata.get("saved_to")
            if path and os.path.isfile(path):
                bubble.add_image(path)
                bubble.add_text(f"Saved: {os.path.basename(path)}")
                self._preview_viewer.set_image(path)
                self._preview_label.setText(os.path.basename(path))
                self._tabs.setCurrentIndex(1)  # Preview tab
            else:
                bubble.add_text("Image generated (no file path)")

        elif result.output_type == "audio":
            path = result.metadata.get("stored_path") or result.metadata.get("saved_to")
            if path and os.path.isfile(path):
                bubble.add_audio(path)
                bubble.add_text(f"Audio: {os.path.basename(path)}")
                self._audio_player.load_file(path)
                self._tabs.setCurrentIndex(2)  # Audio tab
            else:
                bubble.add_text("Audio generated (no file path)")

        elif result.output_type == "video":
            path = result.metadata.get("stored_path") or result.metadata.get("saved_to")
            if path and os.path.isfile(path):
                bubble.add_video(path)
                bubble.add_text(f"Video: {os.path.basename(path)}")
                self._video_player.load_file(path)
                self._tabs.setCurrentIndex(3)  # Video tab
            else:
                bubble.add_text("Video generated (no file path)")

        elif result.output_type == "mesh":
            path = result.metadata.get("stored_path") or result.metadata.get("saved_to")
            bubble.add_text(f"3D mesh saved: {os.path.basename(path or 'unknown')}")

        self._chat_view.scroll_to_bottom()
        self._finish_busy("Pipeline complete")

    def _on_recording_done(self, path):
        """Mic recording finished — auto-attach for STT."""
        self._drop_zone._attached_files = [path]
        self._drop_zone._file_label.setText(os.path.basename(path))
        self._set_status(f"Recording saved: {os.path.basename(path)}")

    def _on_files_dropped(self, paths):
        """Files dropped — show preview if image."""
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                self._preview_viewer.set_image(p)
                self._preview_label.setText(os.path.basename(p))
                self._tabs.setCurrentIndex(1)
                break

    # ═══════════════════════════════════════════════════════════════════
    # ACTIONS
    # ═══════════════════════════════════════════════════════════════════

    def _on_run_selected_actions(self):
        selected = self._action_list.selectedItems()
        if not selected:
            return
        indices = [self._action_list.row(item) for item in selected]
        actions = [self._pending_actions[i] for i in indices
                   if i < len(self._pending_actions)]
        self._run_actions(actions)

    def _on_run_all_actions(self):
        self._run_actions(self._pending_actions[:])

    def _run_actions(self, actions):
        self._action_panel.setVisible(False)
        if not actions:
            return

        worker = ActionWorker(actions)
        worker.output_ready.connect(self._on_action_output)
        worker.error.connect(lambda e: self._set_status(f"Action error: {e}"))
        worker.status_changed.connect(self._set_status)
        worker.start()

    def _on_action_output(self, display, output):
        """Display action output and feed it back to AI for analysis."""
        bubble = self._chat_view.add_bubble("assistant")
        bubble.add_text(f"[Action: {display}]\n{output}")
        self._chat_view.scroll_to_bottom()

        # Process through knowledge manager
        if self.km and output:
            self.km.process_tool_result("shell", display, output)
            update_session_map(self.session_map, "shell", display, output)

        # Feed output back to AI for analysis (matches old GUI behavior)
        if self.engine and output and self.engine.is_loaded():
            truncated = output
            limit = get_tool_output_limit()
            if len(truncated) > limit:
                truncated = truncated[:limit] + "\n[...truncated...]"

            feedback_msg = (
                "[TOOL OUTPUT — this is automated command output, not a human message]\n\n"
                f"{truncated}\n\n"
                "Analyze the output above and tell the user what you found."
            )
            self.messages.append({"role": "user", "content": feedback_msg})
            self._update_system_prompt()

            from core.inference import build_active_messages
            mode_cfg = MODES["ASSISTANT"]
            _, active_msgs = build_active_messages(
                self.messages, mode_cfg.context_window
            )

            self._current_bubble = self._chat_view.add_bubble("assistant")
            self._current_bubble.add_text("")

            worker = GenerationWorker(
                self.engine, active_msgs, mode_cfg.max_tokens,
                mode_cfg.temperature,
                enable_thinking=mode_cfg.enable_thinking,
                cancel_event=self._cancel_event,
            )
            worker.response_received.connect(self._response_batcher.add)
            worker.thinking_received.connect(self._thinking_batcher.add)
            worker.status_changed.connect(self._set_status)
            worker.finished.connect(self._on_generation_finished)
            worker.error.connect(self._on_generation_error)
            self._current_worker = worker
            self._busy = True
            self._execute_btn.setText("CANCEL")
            worker.start()

    # ═══════════════════════════════════════════════════════════════════
    # UTILITY
    # ═══════════════════════════════════════════════════════════════════

    def _finish_busy(self, status_msg: str):
        self._busy = False
        self._execute_btn.setText("EXECUTE")
        self._current_worker = None
        self._current_bubble = None
        self._set_status(status_msg)

    def _set_status(self, msg: str):
        self._status_label.setText(msg)

    def _on_refresh(self):
        from core.inference import compress_history
        if len(self.messages) > 2:
            self.messages = compress_history(
                self.messages, MODES["ASSISTANT"].context_window
            )
            self._set_status("History compressed")
        gc.collect()

    def _on_clear(self):
        if self.engine:
            self.engine.unload()
            self.engine = None
        self.messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        self._chat_view.clear_chat()
        self._thinking_output.clear()
        self._action_panel.setVisible(False)
        self._pending_actions.clear()
        self._drop_zone.clear()
        self.session_map.clear()
        clear_cache()
        self._set_status("STANDBY")

    def _on_vram_relief(self):
        if self.engine:
            self.engine.unload()
            self.engine = None
        svc = get_service()
        svc.unload_all()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
        self._set_status("VRAM released")

    def _on_health(self):
        from core.health import run_health_check, format_health_report
        report = run_health_check()
        text = format_health_report(report)
        QMessageBox.information(self, "System Health", text)

    def _on_save_session(self):
        from core.session import save_session
        name, ok = QFileDialog.getSaveFileName(
            self, "Save Session", os.path.join(BASE_DIR, "sessions"),
            "JSON Files (*.json)"
        )
        if ok and name:
            metadata = {
                "model": get_active_model_name(),
                "backend": get_active_backend(),
            }
            smap_data = self.session_map.to_dict() if hasattr(
                self.session_map, "to_dict") else {}
            save_session(os.path.basename(name).replace(".json", ""),
                         self.messages, smap_data, metadata)
            self._set_status("Session saved")

    def _on_load_session(self):
        from core.session import list_sessions, load_session
        sessions = list_sessions()
        if not sessions:
            self._set_status("No saved sessions")
            return
        name, ok = QFileDialog.getOpenFileName(
            self, "Load Session", os.path.join(BASE_DIR, "sessions"),
            "JSON Files (*.json)"
        )
        if ok and name:
            data = load_session(os.path.basename(name).replace(".json", ""))
            if data:
                self.messages = data.get("messages", self.messages)
                self._chat_view.clear_chat()
                for msg in self.messages[1:]:
                    bubble = self._chat_view.add_bubble(msg["role"])
                    bubble.add_text(msg.get("content", ""))
                self._set_status("Session loaded")

    def _on_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Conversation", "",
            "Markdown (*.md);;Text (*.txt)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                for msg in self.messages[1:]:
                    f.write(f"**{msg['role'].upper()}**\n\n")
                    f.write(f"{msg.get('content', '')}\n\n---\n\n")
            self._set_status(f"Exported to {os.path.basename(path)}")

    def _on_ws_set(self):
        path = self._ws_path.text().strip()
        if not path:
            path = QFileDialog.getExistingDirectory(self, "Select Workspace")
            if path:
                self._ws_path.setText(path)
        if path and os.path.isdir(path):
            if self.km.set_workspace(path):
                self.km.bind_workspace_store(path)
                summary = self.km.get_workspace_summary()
                self._set_status(f"Workspace: {path}")
                bubble = self._chat_view.add_bubble("assistant")
                bubble.add_text(f"Workspace set: {path}\n{summary}")
            else:
                self._set_status(f"Invalid workspace: {path}")

    def _on_ws_scan(self):
        self.km.rescan_workspace()
        summary = self.km.get_workspace_summary()
        self._set_status("Workspace rescanned")
        bubble = self._chat_view.add_bubble("assistant")
        bubble.add_text(f"Workspace rescanned.\n{summary}")

    def _on_copy(self):
        """Copy last assistant response to clipboard."""
        # Find last assistant message
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant":
                QApplication.clipboard().setText(msg.get("content", ""))
                self._set_status("Copied to clipboard")
                return
        self._set_status("Nothing to copy")

    def _update_system_prompt(self, mode: str = "ASSISTANT"):
        """Build system prompt with full context (workspace, knowledge, session map)."""
        try:
            from core.prompts import build_assistant_prompt
            profile = get_context_profile()
            prompt = build_assistant_prompt(
                self._system_info, os.getcwd(),
                workspace_text=self.km.get_workspace_summary(
                    max_tokens=profile.workspace_token_budget),
                knowledge_text=self.km.render_for_prompt(
                    token_budget=profile.knowledge_token_budget),
                session_map_text=self.session_map.render(
                    token_budget=profile.session_map_token_budget),
            )
            self.messages[0]["content"] = prompt
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════════
    # RESOURCE MONITOR
    # ═══════════════════════════════════════════════════════════════════

    def _setup_resource_timer(self):
        self._resource_timer = QTimer()
        self._resource_timer.setInterval(3000)
        self._resource_timer.timeout.connect(self._update_resources)
        self._resource_timer.start()

    def _update_resources(self):
        parts = []
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                frac = alloc / total if total > 0 else 0
                color = vram_color(frac)
                parts.append(f"VRAM: {alloc:.1f}/{total:.1f}GB")
                self._resource_label.setStyleSheet(f"color: {color};")
        except ImportError:
            pass

        try:
            import psutil
            ram = psutil.virtual_memory()
            ram_used = ram.used / (1024**3)
            ram_total = ram.total / (1024**3)
            cpu = psutil.cpu_percent()
            parts.append(f"RAM: {ram_used:.1f}/{ram_total:.0f}GB")
            parts.append(f"CPU: {cpu:.0f}%")
        except ImportError:
            pass

        self._resource_label.setText(" | ".join(parts) if parts else "")
