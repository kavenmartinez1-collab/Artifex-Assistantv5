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

# Use the project's get_logger so setup_logging() runs lazily and the
# rotating file handler in logs/artifex.log gets registered. Bare
# logging.getLogger doesn't trigger setup, which is why GUI-side log
# calls were silently dropping into the void.
from core.logging_config import get_logger
_log = get_logger(__name__)

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
    get_active_model_name, get_active_model_path, set_active_model,
    get_context_profile, get_context_profile_name, set_context_profile,
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
    "Voice Assistant",
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
    "Voice Assistant": "voice-assistant",
}

_FILE_INPUT_MODES = {"Vision", "Audio STT", "Image Edit"}
_PROMPT_MODES = {"Chat", "Code", "Image Gen", "Image Edit", "Vision",
                 "3D (ShapE)", "Audio TTS", "Music Gen", "Video Gen",
                 "Voice Assistant"}


_MODEL_GUIDE_HTML = """
<h2>Choosing a model for each pipeline</h2>
<p>Each pipeline mode expects a specific kind of model. Picking the wrong type
(for example, a text LLM for Audio STT, or a TTS voice for Vision) will fail
at load. Below is what works for each mode.</p>

<h3>Chat / Code</h3>
<p><b>Type:</b> Text LLM &mdash; works on either backend.</p>
<ul>
  <li><b>Ollama:</b> any text LLM installed via <code>ollama pull</code></li>
  <li><b>llama.cpp:</b> any GGUF model configured in llama_cpp_config.json</li>
  <li><b>Transformers:</b> any HF causal LM in the models/ directory</li>
</ul>

<h3>Vision</h3>
<p><b>Type:</b> Multimodal vision-language model. Use the
<b>transformers</b> backend.
<i>Image or video files are accepted</i> &mdash; video requires
<code>qwen-vl-utils</code> (<code>pip install 'qwen-vl-utils[decord]'</code>).</p>
<ul>
  <li><code>Qwen/Qwen3-VL-Instruct</code></li>
  <li><code>Qwen/Qwen2.5-VL-7B-Instruct</code></li>
  <li><code>llava-hf/llava-1.5-7b-hf</code></li>
  <li><code>google/gemma-3-4b-it</code> (vision-capable variants)</li>
</ul>

<h3>Image Gen / Image Edit</h3>
<p><b>Type:</b> Diffusion model (Stable Diffusion / FLUX). Use
<b>transformers</b> backend.</p>
<ul>
  <li><code>stabilityai/stable-diffusion-xl-base-1.0</code></li>
  <li><code>runwayml/stable-diffusion-v1-5</code></li>
  <li><code>black-forest-labs/FLUX.1-schnell</code></li>
</ul>

<h3>3D (ShapE)</h3>
<p><b>Type:</b> Hardcoded to <code>openai/shap-e</code>. No model selection
needed; it downloads on first use.</p>

<h3>Audio TTS (text &rarr; speech)</h3>
<p><b>Type:</b> Text-to-speech model. Examples:</p>
<ul>
  <li><b>Piper</b> ONNX voices (already in your <code>models/piper-voices</code>)</li>
  <li><code>suno/bark-small</code></li>
  <li><code>microsoft/speecht5_tts</code></li>
</ul>

<h3>Audio STT (speech &rarr; text)</h3>
<p><b>Type:</b> Whisper or wav2vec2 ASR model. Use <b>transformers</b> backend.
<i>Audio or video files are accepted</i> &mdash; video audio tracks are auto-extracted.</p>
<ul>
  <li><code>openai/whisper-small</code> (good default, ~500&nbsp;MB)</li>
  <li><code>openai/whisper-base</code> (smaller, faster)</li>
  <li><code>openai/whisper-large-v3</code> (best quality)</li>
  <li><code>Systran/faster-whisper-base.en</code> (CTranslate2-optimized, fast)</li>
</ul>
<p><b>Note:</b> Piper voices and text LLMs <b>cannot</b> transcribe audio.</p>

<h3>Music Gen</h3>
<p><b>Type:</b> MusicGen. Use <b>transformers</b> backend.</p>
<ul>
  <li><code>facebook/musicgen-small</code></li>
  <li><code>facebook/musicgen-medium</code></li>
</ul>

<h3>Video Gen</h3>
<p><b>Type:</b> Text-to-video diffusion. Use <b>transformers</b> backend.</p>
<ul>
  <li><code>THUDM/CogVideoX-2b</code></li>
  <li><code>damo-vilab/text-to-video-ms-1.7b</code></li>
</ul>

<h3>Voice Assistant</h3>
<p>Bundled pipeline: uses <b>faster-whisper</b> for STT and <b>Piper</b> for TTS
internally. The model dropdown here only picks the <i>LLM</i> &mdash; any
text model from the Chat list works.</p>

<hr>
<p style="color:#888"><i>Tip: if a download is required, the model fetches
on first use from Hugging Face. Make sure you have a working internet connection
the first time you run a new pipeline.</i></p>
"""


def _dedupe_actions(actions):
    """Remove duplicate AgentActions by (type, content) identity.

    Preserves first-occurrence order. Needed because the response and
    thinking channels can contain the same tool block when the engine
    folds an empty content field back to thinking, and because some
    models echo the same action twice in a single response.
    """
    seen = set()
    out = []
    for a in actions:
        key = (a.type, a.content)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


class ArtifexMainWindow(QMainWindow):
    """Main application window for Artifex Assistant V5."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Artifex Assistant V5")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 1050)

        # State
        self.engine = None
        self.messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        self._busy = False
        self._cancel_event = threading.Event()
        self._current_worker = None
        # Action workers run independently of the generation worker (an
        # action can complete while a new generation is being kicked off
        # by _on_action_output, so they can't share a slot). Storing on
        # `self` is mandatory: a QThread garbage-collected while still
        # running crashes the process with STATUS_STACK_BUFFER_OVERRUN
        # (Windows exit code 0xC0000409) via PyQt6's __fastfail handler.
        self._current_action_worker = None
        self._pending_actions = []
        self._current_bubble = None
        self._output_dir = os.path.join(BASE_DIR, "output")

        # Knowledge & context (matches old GUI)
        self.km = KnowledgeManager()
        self.km.set_workspace(os.getcwd())  # default workspace = launch dir
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
        self._backend_combo.addItems(["transformers", "ollama", "llama_cpp"])
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

        self._model_help_btn = QPushButton("Help choose model")
        self._model_help_btn.setProperty("class", "secondary")
        self._model_help_btn.setToolTip(
            "Show which kinds of models work for each pipeline (Chat, Vision, STT, etc.)"
        )
        bg_layout.addWidget(self._model_help_btn)

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

        # Logs tab — live log stream
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(2000)
        self._log_view.setStyleSheet(
            "font-family: Consolas; font-size: 9pt; background: #0a0a0a; color: #c8c8c8;"
        )
        self._tabs.addTab(self._log_view, "Logs")
        self._install_log_handler()

        self._tabs.setMinimumHeight(400)
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
        self._prompt_input.setMaximumHeight(70)
        self._prompt_input.setFont(QFont("Consolas", 11))
        input_layout.addWidget(self._prompt_input)

        # Controls row
        ctrl_layout = QHBoxLayout()

        ctrl_layout.addWidget(QLabel("Tokens:"))
        self._tokens_input = QLineEdit("max")
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

        # Vision-mode video sampling controls — blank = auto-plan based on
        # video length + free RAM. Hidden unless mode == "Vision".
        self._vision_ctrl = QWidget()
        vision_layout = QHBoxLayout(self._vision_ctrl)
        vision_layout.setContentsMargins(0, 0, 0, 0)
        vision_layout.addWidget(QLabel("Frames:"))
        self._vision_nframes = QLineEdit()
        self._vision_nframes.setMaximumWidth(60)
        self._vision_nframes.setPlaceholderText("auto")
        self._vision_nframes.setToolTip(
            "Fixed number of frames to sample from the video. "
            "Blank = auto (based on video length + free RAM)."
        )
        vision_layout.addWidget(self._vision_nframes)

        vision_layout.addWidget(QLabel("FPS:"))
        self._vision_fps = QLineEdit()
        self._vision_fps.setMaximumWidth(60)
        self._vision_fps.setPlaceholderText("auto")
        self._vision_fps.setToolTip(
            "Uniform sampling rate. Overrides Frames. Blank = auto."
        )
        vision_layout.addWidget(self._vision_fps)

        vision_layout.addWidget(QLabel("Max px:"))
        self._vision_max_pixels = QLineEdit()
        self._vision_max_pixels.setMaximumWidth(80)
        self._vision_max_pixels.setPlaceholderText(str(360 * 420))
        self._vision_max_pixels.setToolTip(
            "Per-frame pixel cap after resize. Lower = less VRAM, worse OCR."
        )
        vision_layout.addWidget(self._vision_max_pixels)
        vision_layout.addStretch()
        input_layout.addWidget(self._vision_ctrl)

        # Mode hint bar — contextual instructions
        self._hint_label = QLabel("")
        self._hint_label.setStyleSheet(
            "font-family: Consolas; font-size: 9pt; color: #888; padding: 2px 4px;"
        )
        input_layout.addWidget(self._hint_label)

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

    def _install_log_handler(self):
        """Route Python logging into the Logs tab via a signal-safe handler."""
        import logging
        from PyQt6.QtCore import QObject, pyqtSignal

        class _Emitter(QObject):
            log_message = pyqtSignal(str)

        emitter = _Emitter()
        emitter.log_message.connect(self._append_log_line)
        self._log_emitter = emitter

        class _QtHandler(logging.Handler):
            def __init__(self, sig):
                super().__init__()
                self._sig = sig

            def emit(self, record):
                try:
                    self._sig.emit(self.format(record))
                except RuntimeError:
                    pass

        handler = _QtHandler(emitter.log_message)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            fmt="[%(asctime)s %(levelname)s %(name)s] %(message)s",
            datefmt="%H:%M:%S",
        ))
        logging.getLogger().addHandler(handler)

    def _append_log_line(self, text: str):
        self._log_view.appendPlainText(text)

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
        self._model_help_btn.clicked.connect(self._on_help_choose_model)

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

        # Push-to-talk state
        self._ptt_active = False

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
        self._tokens_input.setText("max")

    def _on_help_choose_model(self):
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QDialogButtonBox

        dlg = QDialog(self)
        dlg.setWindowTitle("Help choose a model for each pipeline")
        dlg.resize(640, 560)
        layout = QVBoxLayout(dlg)

        browser = QTextBrowser(dlg)
        browser.setOpenExternalLinks(True)
        browser.setHtml(_MODEL_GUIDE_HTML)
        layout.addWidget(browser)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=dlg)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)

        dlg.exec()

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
        self._mic_recorder.setVisible(mode in {"Audio STT", "Voice Assistant"})
        self._vision_ctrl.setVisible(mode == "Vision")

        # Update hint text
        hints = {
            "Chat": "Type a message and press Ctrl+Enter or click EXECUTE",
            "Code": "Describe what you want to build, then EXECUTE",
            "Image Gen": "Describe the image you want to generate",
            "Image Edit": "Drop an image above, then describe the edit",
            "Vision": "Drop an image or video above, then ask a question about it",
            "3D (ShapE)": "Describe a 3D object to generate",
            "Audio TTS": "Type text to convert to speech",
            "Audio STT": "Record or drop an audio file to transcribe",
            "Music Gen": "Describe the music you want to generate",
            "Video Gen": "Describe the video you want to generate",
            "Voice Assistant": "Hold Space to talk (click outside the text box first) "
                               "| Or type a message and EXECUTE",
        }
        self._hint_label.setText(hints.get(mode, ""))

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

        # Voice Assistant can work with either text OR a recorded audio file
        has_recording = (mode == "Voice Assistant"
                         and self._drop_zone._attached_files)
        if mode in _PROMPT_MODES and not prompt and not has_recording:
            if mode == "Voice Assistant":
                self._set_status("Hold Space to talk, or type a message")
            else:
                self._set_status("Enter a prompt first")
            return

        _tok_text = self._tokens_input.text().strip().lower()
        max_tokens = -1 if _tok_text in ("max", "") else int(_tok_text)
        temperature = self._temp_slider.value() / 10.0

        # Voice Assistant with recording but no text — transcribe first
        if mode == "Voice Assistant" and not prompt and has_recording:
            self._prompt_input.clear()
            self._run_voice_from_recording(self._drop_zone._attached_files[0])
            return

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
        from core.inference import build_active_messages, trim_messages_to_context
        mode_cfg = MODES.get(mode.upper(), MODES["ASSISTANT"])
        _, active_msgs = build_active_messages(
            self.messages, mode_cfg.context_window
        )
        if self.engine:
            ctx = self.engine.get_context_size()
            if ctx > 0:
                active_msgs = trim_messages_to_context(active_msgs, int(ctx * 0.85))

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

        if not kwargs:
            # File-dependent modes return empty kwargs if no files attached
            if mode in _FILE_INPUT_MODES:
                self._set_status(f"Attach a file first for {mode} mode")
            else:
                self._set_status(f"Cannot build parameters for {mode}")
            self._busy = False
            self._execute_btn.setText("EXECUTE")
            return

        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)  # Indeterminate

        model_path = get_active_model_path() if get_active_backend() == "transformers" else ""

        worker = PipelineWorker(
            svc, pipeline_type, model_path=model_path, kwargs=kwargs,
            cancel_event=self._cancel_event,
        )
        worker.progress.connect(self._on_pipeline_progress)
        # Voice Assistant results need the voice handler (shows chat + stays on Chat tab)
        if mode == "Voice Assistant":
            worker.result_ready.connect(self._on_voice_result)
        else:
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
                try:
                    _vt = self._tokens_input.text().strip().lower()
                    max_tokens = -1 if _vt in ("max", "") else int(_vt)
                except ValueError:
                    max_tokens = -1
                kwargs = {"image_path": files[0],
                          "prompt": prompt or "Describe what is happening in detail.",
                          "max_tokens": max_tokens}

                # Video sampling overrides — blank fields leave auto-planning active.
                nframes_text = self._vision_nframes.text().strip()
                if nframes_text:
                    try:
                        kwargs["nframes"] = int(nframes_text)
                    except ValueError:
                        pass
                fps_text = self._vision_fps.text().strip()
                if fps_text:
                    try:
                        kwargs["fps"] = float(fps_text)
                    except ValueError:
                        pass
                pixels_text = self._vision_max_pixels.text().strip()
                if pixels_text:
                    try:
                        kwargs["max_pixels"] = int(pixels_text)
                    except ValueError:
                        pass
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
        elif mode == "Voice Assistant":
            # Text input — skip STT, go straight to LLM → TTS
            if prompt:
                kwargs = {"text": prompt, "play_audio": True}

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

        # Pull the thinking accumulator off the worker BEFORE _finish_busy
        # clears self._current_worker. Some models (notably Gemma 4 in its
        # harmony-format Ollama variant) put tool-call blocks inside the
        # thinking/analysis channel rather than the visible response. We
        # need both halves to find every tool call the model intended.
        thinking_text = ""
        if self._current_worker is not None:
            thinking_text = getattr(self._current_worker, "_thinking_text", "") or ""

        self.messages.append({"role": "assistant", "content": response})

        # Extract knowledge from AI response (matches old GUI)
        try:
            self.km.add_from_ai_response(response)
        except Exception as e:
            _log.warning("Knowledge extraction failed: %s", e)

        self._finish_busy("Generation complete")

        # Check for agent actions. Strategy:
        #
        #   1. Scan the visible response first. If the model committed to
        #      tool calls in its final answer, those are the canonical ones
        #      and we use them as-is.
        #
        #   2. ONLY if the response yielded nothing, fall back to scanning
        #      the thinking/analysis channel. This catches models like
        #      Gemma 4 that emit tool blocks inside their reasoning instead
        #      of the visible response — without producing false positives
        #      from a model that abandoned a tool plan mid-thought.
        #
        #   3. Dedupe by (type, content) so we never show the same action
        #      twice. This matters because engine_ollama.py:245 folds
        #      `thinking_response` into `full_response` when content is
        #      empty, which would otherwise leak duplicates through the
        #      `response` variable on top of the worker's thinking_text.
        try:
            from tools.agent_tools import extract_agent_actions
            actions = extract_agent_actions(response or "")
            extracted_from = "response"
            if not actions and thinking_text:
                actions = extract_agent_actions(thinking_text)
                extracted_from = "thinking"
            actions = _dedupe_actions(actions)

            _log.info(
                "Action extraction: resp_len=%d think_len=%d "
                "edit_fence_resp=%s edit_fence_think=%s extracted=%d from=%s",
                len(response or ""),
                len(thinking_text),
                "```edit" in (response or ""),
                "```edit" in thinking_text,
                len(actions),
                extracted_from if actions else "none",
            )
            if actions:
                self._pending_actions = actions
                self._action_list.clear()
                for a in actions:
                    display = getattr(a, "display", str(a))
                    self._action_list.addItem(display)
                self._action_panel.setVisible(True)
        except Exception as e:
            _log.exception("Action extraction failed: %s", e)

    def _on_generation_error(self, error_msg: str):
        """Handle generation error."""
        self._response_batcher.flush_now()
        self._thinking_batcher.flush_now()

        _log.error("Pipeline/generation error: %s", error_msg)

        if self._current_bubble:
            self._current_bubble.append_text(f"\n\nERROR: {error_msg}")
        else:
            bubble = self._chat_view.add_bubble("assistant")
            bubble.add_text(f"ERROR: {error_msg}")
            self._chat_view.scroll_to_bottom()
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
            # Voice Assistant returns response_text in metadata
            response_text = result.metadata.get("response_text")
            if response_text:
                bubble.add_text(response_text)

            path = (result.metadata.get("stored_path")
                    or result.metadata.get("saved_to")
                    or result.content)
            if path and os.path.isfile(str(path)):
                bubble.add_audio(str(path))
                if not response_text:
                    bubble.add_text(f"Audio: {os.path.basename(str(path))}")
                self._audio_player.load_file(str(path))
                if not response_text:
                    self._tabs.setCurrentIndex(2)  # Audio tab only for plain TTS
            elif not response_text:
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
        """Mic recording finished — auto-attach for STT, auto-send for Voice Assistant."""
        self._drop_zone._attached_files = [path]
        self._drop_zone._file_label.setText(os.path.basename(path))
        self._set_status(f"Recording saved: {os.path.basename(path)}")

        # Voice Assistant mode: auto-transcribe and send to LLM
        mode = self._mode_combo.currentText()
        if mode == "Voice Assistant" and not self._busy:
            self._run_voice_from_recording(path)

    def _run_voice_from_recording(self, wav_path: str):
        """Voice Assistant: load WAV and transcribe."""
        import numpy as np
        from scipy.io import wavfile

        try:
            sr, audio = wavfile.read(wav_path)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        except Exception as e:
            self._set_status(f"Failed to read recording: {e}")
            return

        if np.max(np.abs(audio)) < 0.0001:
            self._set_status("No audio from microphone — check mic settings")
            return

        self._run_voice_from_audio(audio)

    def _run_voice_from_audio(self, audio):
        """Voice Assistant: transcribe audio first, show text, then send to LLM."""
        self._set_status("Transcribing speech...")
        self._pending_voice_audio = audio

        # Step 1: STT only — transcribe and show the text
        svc = get_service()
        worker = PipelineWorker(
            svc, "voice-assistant",
            kwargs={"audio_data": audio, "stt_only": True},
            cancel_event=self._cancel_event,
        )
        worker.result_ready.connect(self._on_stt_result)
        worker.error.connect(self._on_generation_error)
        worker.status_changed.connect(self._set_status)
        self._current_worker = worker
        worker.start()

    def _on_stt_result(self, result):
        """STT finished — show transcription then auto-send to LLM → TTS."""
        if not result.success or not result.content or not result.content.strip():
            self._set_status("No speech detected — try again")
            return

        transcribed = result.content.strip()

        # Show what we heard as a user bubble
        user_bubble = self._chat_view.add_bubble("user")
        user_bubble.add_text(transcribed)
        self.messages.append({"role": "user", "content": transcribed})
        self._chat_view.scroll_to_bottom()

        # Step 2: Send the text through LLM �� TTS
        self._set_status("Thinking...")
        self._busy = True
        self._cancel_event.clear()
        self._execute_btn.setText("CANCEL")
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)

        svc = get_service()
        worker = PipelineWorker(
            svc, "voice-assistant",
            kwargs={"text": transcribed, "play_audio": True},
            cancel_event=self._cancel_event,
        )
        worker.progress.connect(self._on_pipeline_progress)
        worker.result_ready.connect(self._on_voice_result)
        worker.error.connect(self._on_generation_error)
        worker.status_changed.connect(self._set_status)
        self._current_worker = worker
        worker.start()

    def _on_voice_result(self, result):
        """Handle voice assistant LLM → TTS result — show response in chat."""
        self._progress_bar.setVisible(False)

        if not result.success:
            bubble = self._chat_view.add_bubble("assistant")
            bubble.add_text(f"Error: {result.error}")
            self._chat_view.scroll_to_bottom()
            self._finish_busy("Voice pipeline failed")
            return

        response_text = result.metadata.get("response_text", "")

        if response_text:
            bubble = self._chat_view.add_bubble("assistant")
            bubble.add_text(response_text)
            self.messages.append({"role": "assistant", "content": response_text})

            # Show audio player if TTS was generated
            audio_path = result.metadata.get("audio_path") or result.content
            if audio_path and os.path.isfile(str(audio_path)):
                bubble.add_audio(str(audio_path))
                self._audio_player.load_file(str(audio_path))

        self._chat_view.scroll_to_bottom()
        self._finish_busy("Hold Space to talk | Or type and EXECUTE")

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
        # CRITICAL: store the worker on `self`. A QThread that goes out of
        # scope while still running gets garbage-collected mid-execution,
        # which PyQt6 reports as a fatal Qt invariant violation via
        # __fastfail() — visible as Windows exit code 0xC0000409
        # (STATUS_STACK_BUFFER_OVERRUN). Clear the reference and schedule
        # deletion when the thread finishes so we don't leak workers.
        self._current_action_worker = worker
        worker.finished.connect(self._on_action_worker_done)
        worker.start()

    def _on_action_worker_done(self):
        """Release the action worker after its QThread finishes naturally."""
        worker = self._current_action_worker
        if worker is not None:
            worker.deleteLater()
            self._current_action_worker = None

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

            from core.inference import build_active_messages, trim_messages_to_context
            mode_cfg = MODES["ASSISTANT"]
            _, active_msgs = build_active_messages(
                self.messages, mode_cfg.context_window
            )
            ctx = self.engine.get_context_size()
            if ctx > 0:
                active_msgs = trim_messages_to_context(active_msgs, int(ctx * 0.85))

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
        except Exception as e:
            _log.warning("VRAM cleanup error: %s", e)
        self._set_status("VRAM released")

    def _on_health(self):
        from core.health import run_health_check, format_health_report
        report = run_health_check()
        text = format_health_report(report)
        QMessageBox.information(self, "System Health", text)

    def _on_save_session(self):
        from core.session import save_session
        sessions_dir = os.path.join(BASE_DIR, "sessions")
        os.makedirs(sessions_dir, exist_ok=True)
        path, ok = QFileDialog.getSaveFileName(
            self, "Save Session", sessions_dir,
            "JSON Files (*.json)"
        )
        if ok and path:
            metadata = {
                "model": get_active_model_name(),
                "backend": get_active_backend(),
            }
            smap_data = self.session_map.to_dict() if hasattr(
                self.session_map, "to_dict") else {}
            # Use the full chosen path's basename as session name,
            # but save_session will store it in the sessions directory
            session_name = os.path.splitext(os.path.basename(path))[0]
            save_session(session_name, self.messages, smap_data, metadata)
            self._set_status(f"Session saved: {session_name}")

    def _on_load_session(self):
        from core.session import list_sessions, load_session
        sessions = list_sessions()
        if not sessions:
            self._set_status("No saved sessions")
            return
        path, ok = QFileDialog.getOpenFileName(
            self, "Load Session", os.path.join(BASE_DIR, "sessions"),
            "JSON Files (*.json)"
        )
        if ok and path:
            session_name = os.path.splitext(os.path.basename(path))[0]
            data = load_session(session_name)
            if data:
                self.messages = data.get("messages", self.messages)
                self._chat_view.clear_chat()
                for msg in self.messages[1:]:
                    role = msg.get("role", "user")
                    bubble = self._chat_view.add_bubble(role)
                    bubble.add_text(msg.get("content", ""))
                self._set_status(f"Session loaded: {session_name}")
            else:
                self._set_status(f"Failed to load session: {session_name}")

    def _on_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Conversation", "",
            "Markdown (*.md);;Text (*.txt)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                for msg in self.messages[1:]:
                    role = msg.get("role", "unknown").upper()
                    f.write(f"**{role}**\n\n")
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
                # Change the process CWD so that:
                #   1. The system prompt's CWD field reflects the workspace
                #      (build_assistant_prompt reads os.getcwd())
                #   2. All agent tools resolve relative paths against the
                #      workspace (14 os.getcwd() calls in agent_tools.py)
                #   3. Subprocess shells (```bash blocks) execute in the
                #      workspace directory
                # The GUI itself uses absolute paths via BASE_DIR for its
                # own files (logs, output, sessions), so this is safe.
                os.chdir(path)
                summary = self.km.get_workspace_summary()
                self._set_status(f"Workspace: {path}")
                bubble = self._chat_view.add_bubble("assistant")
                bubble.add_text(f"Workspace set: {path}\n{summary}")
                _log.info("Workspace changed: %s (CWD updated)", path)
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
        except Exception as e:
            _log.warning("System prompt update failed: %s", e)

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
        except Exception as e:
            _log.debug("VRAM monitor error: %s", e)

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
        except Exception as e:
            _log.debug("System monitor error: %s", e)

        self._resource_label.setText(" | ".join(parts) if parts else "")

    # ═══════════════════════════════════════════════════════════════════
    # PUSH-TO-TALK (hold Space in Voice Assistant mode)
    # ═══════════════════════════════════════════════════════════════════

    def keyPressEvent(self, event):
        """Hold Space to start recording in Voice Assistant mode."""
        if (event.key() == Qt.Key.Key_Space
                and not event.isAutoRepeat()
                and self._mode_combo.currentText() == "Voice Assistant"
                and not self._prompt_input.hasFocus()
                and not self._ptt_active
                and not self._busy):
            self._ptt_active = True
            self._ptt_audio_data = []
            self._ptt_stream = None
            try:
                import sounddevice as sd
                self._ptt_stream = sd.InputStream(
                    samplerate=16000, channels=1, dtype="float32",
                    callback=self._ptt_audio_callback, blocksize=1024,
                )
                self._ptt_stream.start()
                self._set_status("Listening... (release Space to send)")
                self._mic_recorder._record_btn.setStyleSheet("background-color: #ff4444;")
                self._mic_recorder._status_label.setText("Recording...")
            except Exception as e:
                self._ptt_active = False
                self._set_status(f"Mic error: {e}")
            return
        super().keyPressEvent(event)

    def _ptt_audio_callback(self, indata, frames, time_info, status):
        """Collect audio samples during push-to-talk."""
        self._ptt_audio_data.append(indata.copy())

    def keyReleaseEvent(self, event):
        """Release Space to stop recording and auto-send."""
        if (event.key() == Qt.Key.Key_Space
                and not event.isAutoRepeat()
                and self._ptt_active):
            self._ptt_active = False

            # Stop the stream
            if self._ptt_stream:
                self._ptt_stream.stop()
                self._ptt_stream.close()
                self._ptt_stream = None

            self._mic_recorder._record_btn.setStyleSheet("")
            self._mic_recorder._status_label.setText("Ready")

            if not self._ptt_audio_data:
                self._set_status("No audio captured")
                return

            # Convert to numpy and send directly to voice pipeline
            import numpy as np
            audio = np.concatenate(self._ptt_audio_data, axis=0).flatten()

            if np.max(np.abs(audio)) < 0.01:
                self._set_status("No speech detected — try again")
                return

            self._set_status("Processing speech...")
            self._run_voice_from_audio(audio)
            return
        super().keyReleaseEvent(event)
