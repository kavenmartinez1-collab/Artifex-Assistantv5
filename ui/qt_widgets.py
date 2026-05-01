"""
Artifex Assistant V5 — Custom PyQt6 widgets for multimodal I/O.
Provides drag-drop, image viewer, audio/video player, mic recorder,
and rich chat bubbles with inline media.
"""

import os
import platform
import subprocess
import time
from pathlib import Path


def _open_file_externally(path: str):
    """Open a file with the system's default application (cross-platform)."""
    system = platform.system()
    if system == "Windows":
        os.startfile(path)
    elif system == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

from PyQt6.QtCore import Qt, pyqtSignal, QUrl, QSize, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QFont
from PyQt6.QtWidgets import (
    QFrame, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
    QScrollArea, QPushButton, QSlider, QSizePolicy, QFileDialog,
    QDialog, QPlainTextEdit,
)


# NOTE: ChatBubble used to back text blocks with QTextEdit and manage their
# heights manually. That approach had three independent failure modes:
#
#   1. Re-entrant relayout from contentsChanged → setMaximumHeight crashed
#      Windows with STATUS_ACCESS_VIOLATION on Gemma 4's large flushes.
#   2. Reading doc.size() before the widget had been shown produced a
#      0-width textWidth, which collapsed every bubble to the 30px minimum.
#   3. QTextEdit's own internal viewport fought the outer ChatView scroll
#      area (ensureCursorVisible scrolled the inner view, hiding the top
#      of long responses).
#
# QLabel with word wrap sidesteps all three: it has no internal viewport,
# no QTextDocument signal storms, and auto-sizes to its wrapped content
# via the parent layout. Selectable text gives users copy/paste back.

# Accepted file extensions
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
_VIDEO_EXTS = {".mp4", ".avi", ".webm", ".mov", ".mkv"}
_DOC_EXTS = {".pdf", ".txt", ".md", ".docx", ".csv", ".json"}
_ALL_EXTS = _IMAGE_EXTS | _AUDIO_EXTS | _VIDEO_EXTS | _DOC_EXTS


# ── Drop Zone ────────────────────────────────────────────────────────────

class DropZone(QFrame):
    """Drag-and-drop zone that accepts images, audio, video, documents.

    Emits files_dropped(list[str]) with paths of dropped files.
    Also provides a browse button as fallback.
    """

    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        self.setStyleSheet("""
            DropZone {
                border: 2px dashed #444;
                border-radius: 8px;
                background-color: transparent;
            }
            DropZone[dragOver="true"] {
                border-color: #00f0ff;
                background-color: rgba(0, 240, 255, 0.05);
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        self._label = QLabel("Drop files here or")
        self._label.setStyleSheet("border: none; color: #666;")
        layout.addWidget(self._label)

        self._browse_btn = QPushButton("Browse")
        self._browse_btn.setMaximumWidth(80)
        self._browse_btn.clicked.connect(self._on_browse)
        layout.addWidget(self._browse_btn)

        self._file_label = QLabel("")
        self._file_label.setStyleSheet("border: none; color: #9efeff;")
        self._file_label.setMaximumWidth(300)
        layout.addWidget(self._file_label, 1)

        self._attached_files: list[str] = []

    @property
    def attached_files(self) -> list[str]:
        return list(self._attached_files)

    def clear(self):
        self._attached_files.clear()
        self._file_label.setText("")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("dragOver", True)
            self.style().polish(self)

    def dragLeaveEvent(self, event):
        self.setProperty("dragOver", False)
        self.style().polish(self)

    def dropEvent(self, event):
        self.setProperty("dragOver", False)
        self.style().polish(self)

        paths = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            ext = Path(path).suffix.lower()
            if ext in _ALL_EXTS:
                paths.append(path)

        if paths:
            self._attached_files = paths
            names = ", ".join(os.path.basename(p) for p in paths)
            self._file_label.setText(names)
            self.files_dropped.emit(paths)

    def _on_browse(self):
        filter_str = "All Supported (*.png *.jpg *.jpeg *.webp *.bmp *.wav *.mp3 *.flac *.mp4 *.avi *.webm *.pdf *.txt *.md *.docx);;Images (*.png *.jpg *.jpeg *.webp *.bmp);;Audio (*.wav *.mp3 *.flac);;Video (*.mp4 *.avi *.webm);;Documents (*.pdf *.txt *.md *.docx)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", filter_str)
        if paths:
            self._attached_files = paths
            names = ", ".join(os.path.basename(p) for p in paths)
            self._file_label.setText(names)
            self.files_dropped.emit(paths)


# ── Image Viewer ─────────────────────────────────────────────────────────

class ImageViewer(QLabel):
    """Image display with click-to-fullscreen zoom."""

    def __init__(self, max_width: int = 480, max_height: int = 360, parent=None):
        super().__init__(parent)
        self._max_w = max_width
        self._max_h = max_height
        self._pixmap = None
        self._source_path = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(100, 75)
        self.setStyleSheet("background-color: transparent; border: none;")

    def set_image(self, path_or_pixmap):
        """Set image from file path or QPixmap."""
        if isinstance(path_or_pixmap, str):
            self._source_path = path_or_pixmap
            self._pixmap = QPixmap(path_or_pixmap)
        elif isinstance(path_or_pixmap, QPixmap):
            self._pixmap = path_or_pixmap
        else:
            return

        if not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self._max_w, self._max_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)

    def mouseDoubleClickEvent(self, event):
        """Open fullscreen dialog on double-click."""
        if self._pixmap and not self._pixmap.isNull():
            dialog = _FullscreenImageDialog(self._pixmap, self._source_path,
                                            self.window())
            dialog.exec()


class _FullscreenImageDialog(QDialog):
    """Fullscreen image viewer dialog."""

    def __init__(self, pixmap: QPixmap, source_path: str = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Preview")
        self.setMinimumSize(800, 600)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Scale to dialog size
        screen = self.screen().availableGeometry()
        max_w = int(screen.width() * 0.8)
        max_h = int(screen.height() * 0.8)
        scaled = pixmap.scaled(max_w, max_h,
                               Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(scaled)
        layout.addWidget(label)

        if source_path:
            btn_layout = QHBoxLayout()
            open_btn = QPushButton("Open in Explorer")
            open_btn.clicked.connect(lambda: _open_file_externally(source_path))
            btn_layout.addStretch()
            btn_layout.addWidget(open_btn)
            btn_layout.addStretch()
            layout.addLayout(btn_layout)


# ── Audio Player ─────────────────────────────────────────────────────────

class AudioPlayer(QWidget):
    """Audio playback widget with play/pause, seek, and time display.

    Uses QMediaPlayer + QAudioOutput from PyQt6.QtMultimedia.
    Falls back to a simple "Open in Player" button if multimedia unavailable.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_path = None
        self._use_native = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        try:
            from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

            self._player = QMediaPlayer()
            self._audio_output = QAudioOutput()
            self._player.setAudioOutput(self._audio_output)
            self._use_native = True

            self._play_btn = QPushButton("Play")
            self._play_btn.setMaximumWidth(60)
            self._play_btn.clicked.connect(self._toggle_play)
            layout.addWidget(self._play_btn)

            self._seek = QSlider(Qt.Orientation.Horizontal)
            self._seek.setRange(0, 100)
            self._seek.sliderMoved.connect(self._on_seek)
            layout.addWidget(self._seek, 1)

            self._time_label = QLabel("0:00 / 0:00")
            self._time_label.setMinimumWidth(90)
            self._time_label.setStyleSheet("font-family: Consolas; font-size: 9pt;")
            layout.addWidget(self._time_label)

            self._player.positionChanged.connect(self._on_position)
            self._player.durationChanged.connect(self._on_duration)

        except ImportError:
            self._use_native = False
            self._open_btn = QPushButton("Open in Player")
            self._open_btn.clicked.connect(self._open_external)
            layout.addWidget(self._open_btn)
            self._path_label = QLabel("")
            layout.addWidget(self._path_label, 1)

    def load_file(self, path: str):
        self._file_path = path
        if self._use_native:
            self._player.setSource(QUrl.fromLocalFile(path))
            self._play_btn.setText("Play")
        else:
            self._path_label.setText(os.path.basename(path))

    def _toggle_play(self):
        from PyQt6.QtMultimedia import QMediaPlayer
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
            self._play_btn.setText("Play")
        else:
            self._player.play()
            self._play_btn.setText("Pause")

    def _on_seek(self, position):
        if self._player.duration() > 0:
            ms = int(position / 100 * self._player.duration())
            self._player.setPosition(ms)

    def _on_position(self, pos_ms):
        dur = self._player.duration()
        if dur > 0:
            self._seek.setValue(int(pos_ms / dur * 100))
        self._time_label.setText(
            f"{self._fmt(pos_ms)} / {self._fmt(dur)}"
        )

    def _on_duration(self, dur_ms):
        self._time_label.setText(f"0:00 / {self._fmt(dur_ms)}")

    def _open_external(self):
        if self._file_path and os.path.isfile(self._file_path):
            _open_file_externally(self._file_path)

    @staticmethod
    def _fmt(ms):
        s = max(0, ms // 1000)
        return f"{s // 60}:{s % 60:02d}"


# ── Video Player ─────────────────────────────────────────────────────────

class VideoPlayer(QWidget):
    """Video playback widget using QMediaPlayer + QVideoWidget.

    Falls back to "Open in Player" button if multimedia unavailable.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_path = None
        self._use_native = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        try:
            from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
            from PyQt6.QtMultimediaWidgets import QVideoWidget

            self._video_widget = QVideoWidget()
            self._video_widget.setMinimumSize(320, 240)
            layout.addWidget(self._video_widget)

            self._player = QMediaPlayer()
            self._audio_output = QAudioOutput()
            self._player.setAudioOutput(self._audio_output)
            self._player.setVideoOutput(self._video_widget)
            self._use_native = True

            # Controls
            ctrl_layout = QHBoxLayout()
            self._play_btn = QPushButton("Play")
            self._play_btn.setMaximumWidth(60)
            self._play_btn.clicked.connect(self._toggle_play)
            ctrl_layout.addWidget(self._play_btn)

            self._seek = QSlider(Qt.Orientation.Horizontal)
            self._seek.setRange(0, 100)
            self._seek.sliderMoved.connect(self._on_seek)
            ctrl_layout.addWidget(self._seek, 1)

            self._time_label = QLabel("0:00 / 0:00")
            self._time_label.setStyleSheet("font-family: Consolas; font-size: 9pt;")
            ctrl_layout.addWidget(self._time_label)

            layout.addLayout(ctrl_layout)

            self._player.positionChanged.connect(self._on_position)
            self._player.durationChanged.connect(self._on_duration)

        except ImportError:
            self._use_native = False
            self._open_btn = QPushButton("Open Video in Player")
            self._open_btn.clicked.connect(self._open_external)
            layout.addWidget(self._open_btn)

    def load_file(self, path: str):
        self._file_path = path
        if self._use_native:
            self._player.setSource(QUrl.fromLocalFile(path))

    def _toggle_play(self):
        from PyQt6.QtMultimedia import QMediaPlayer
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
            self._play_btn.setText("Play")
        else:
            self._player.play()
            self._play_btn.setText("Pause")

    def _on_seek(self, position):
        if self._player.duration() > 0:
            ms = int(position / 100 * self._player.duration())
            self._player.setPosition(ms)

    def _on_position(self, pos_ms):
        dur = self._player.duration()
        if dur > 0:
            self._seek.setValue(int(pos_ms / dur * 100))
        self._time_label.setText(
            f"{self._fmt(pos_ms)} / {self._fmt(dur)}"
        )

    def _on_duration(self, dur_ms):
        self._time_label.setText(f"0:00 / {self._fmt(dur_ms)}")

    def _open_external(self):
        if self._file_path and os.path.isfile(self._file_path):
            _open_file_externally(self._file_path)

    @staticmethod
    def _fmt(ms):
        s = max(0, ms // 1000)
        return f"{s // 60}:{s % 60:02d}"


# ── Microphone Recorder ─────────────────────────────────────────────────

class MicrophoneRecorder(QWidget):
    """Microphone recording widget.

    Emits recording_finished(str) with the path to the saved WAV file.
    Uses QAudioSource for capture when available, falls back to
    sounddevice/scipy.
    """

    recording_finished = pyqtSignal(str)

    def __init__(self, output_dir: str = "output", parent=None):
        super().__init__(parent)
        self._output_dir = output_dir
        self._recording = False
        self._audio_data = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        self._record_btn = QPushButton("Record")
        self._record_btn.setMaximumWidth(80)
        self._record_btn.setCheckable(True)
        self._record_btn.clicked.connect(self._toggle_recording)
        layout.addWidget(self._record_btn)

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("font-family: Consolas; font-size: 9pt;")
        layout.addWidget(self._status_label, 1)

    def _toggle_recording(self):
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        try:
            import sounddevice as sd
            self._recording = True
            self._audio_data = []
            self._record_btn.setText("Stop")
            self._record_btn.setStyleSheet("background-color: #ff4444;")
            self._status_label.setText("Recording...")

            self._sample_rate = 16000
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="int16",
                callback=self._audio_callback,
            )
            self._stream.start()

        except ImportError:
            self._status_label.setText("sounddevice not installed")
            self._record_btn.setChecked(False)
        except Exception as e:
            self._status_label.setText(f"Mic error: {e}")
            self._record_btn.setChecked(False)

    def _audio_callback(self, indata, frames, time_info, status):
        self._audio_data.append(indata.copy())

    def _stop_recording(self):
        self._recording = False
        self._record_btn.setText("Record")
        self._record_btn.setStyleSheet("")

        if hasattr(self, "_stream"):
            self._stream.stop()
            self._stream.close()

        if not self._audio_data:
            self._status_label.setText("No audio captured")
            return

        try:
            import numpy as np
            from scipy.io import wavfile

            audio = np.concatenate(self._audio_data, axis=0)
            os.makedirs(self._output_dir, exist_ok=True)
            timestamp = int(time.time())
            path = os.path.join(self._output_dir, f"recording_{timestamp}.wav")
            wavfile.write(path, self._sample_rate, audio)

            duration = len(audio) / self._sample_rate
            self._status_label.setText(
                f"Saved: {os.path.basename(path)} ({duration:.1f}s)"
            )
            self.recording_finished.emit(path)

        except Exception as e:
            self._status_label.setText(f"Save error: {e}")


# ── Chat Bubble ──────────────────────────────────────────────────────────

class ChatBubble(QFrame):
    """A single chat message that can contain text, images, audio players.

    Supports mixed content: text rendered as HTML, images as inline QPixmap,
    audio/video as embedded player widgets.
    """

    _MAX_LABEL_CHARS = 8192

    def __init__(self, role: str, parent=None):
        super().__init__(parent)
        self._role = role
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 6, 8, 6)
        self._layout.setSpacing(4)
        self._stream_widget = None

        # Role label
        role_label = QLabel(role.upper())
        role_label.setStyleSheet(
            "font-weight: bold; font-size: 9pt; "
            f"color: {'#00f0ff' if role == 'assistant' else '#9efeff'}; "
            "background: transparent;"
        )
        self._layout.addWidget(role_label)

        self.setStyleSheet("""
            ChatBubble {
                border-bottom: 1px solid #1e2340;
                background-color: transparent;
            }
        """)

    def add_text(self, text: str):
        """Add a text block to the bubble.

        Uses a word-wrapped, selectable QLabel rather than QTextEdit so the
        bubble grows naturally to fit its wrapped content via the parent
        layout — no manual height calculation, no re-entrant signal storms,
        no internal viewport to fight the outer chat scroll area.

        Very large text (>8KB) is truncated for display to avoid QLabel
        word-wrap layout stalls.
        """
        display = text
        if len(text) > self._MAX_LABEL_CHARS:
            remaining = len(text) - self._MAX_LABEL_CHARS
            display = text[:self._MAX_LABEL_CHARS] + f"\n\n[...{remaining:,} more characters]"
        label = QLabel(display)
        label.setWordWrap(True)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.MinimumExpanding,
        )
        label.setStyleSheet(
            "background: transparent; "
            "font-family: Consolas; font-size: 10pt;"
        )
        self._layout.addWidget(label)
        return label

    def add_image(self, path: str):
        """Add an inline image to the bubble."""
        viewer = ImageViewer(max_width=480, max_height=360)
        viewer.set_image(path)
        self._layout.addWidget(viewer)
        return viewer

    def add_audio(self, path: str):
        """Add an audio player to the bubble."""
        player = AudioPlayer()
        player.load_file(path)
        self._layout.addWidget(player)
        return player

    def add_video(self, path: str):
        """Add a video player to the bubble."""
        player = VideoPlayer()
        player.load_file(path)
        self._layout.addWidget(player)
        return player

    def append_text(self, text: str):
        """Append text to the streaming content area.

        On first call, promotes the placeholder QLabel (from add_text(""))
        to a read-only QPlainTextEdit that supports O(1) cursor-based
        append — avoids the O(n^2) QLabel.setText(text()+new) relayout
        that freezes the GUI on long responses (24K+ tokens).
        """
        if self._stream_widget is not None:
            cursor = self._stream_widget.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertText(text)
            self._update_stream_height()
            return

        # First call: remove placeholder QLabel, create stream widget
        for i in range(self._layout.count() - 1, -1, -1):
            widget = self._layout.itemAt(i).widget()
            if isinstance(widget, QLabel) and widget.wordWrap():
                existing = widget.text()
                self._layout.removeWidget(widget)
                widget.deleteLater()
                break
        else:
            existing = ""

        edit = QPlainTextEdit()
        edit.setReadOnly(True)
        edit.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        edit.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        edit.setStyleSheet(
            "background: transparent; border: none; "
            "font-family: Consolas; font-size: 10pt;"
        )
        edit.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        edit.setMinimumHeight(30)
        self._layout.addWidget(edit)
        self._stream_widget = edit

        if existing:
            edit.setPlainText(existing)

        cursor = edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self._update_stream_height()

    def _update_stream_height(self):
        """Resize stream widget to fit content without signal storms.

        QPlainTextDocumentLayout defers full layout, so document().size()
        often underreports height.  Instead, estimate visual line count
        from text content and the monospace character grid.
        """
        if self._stream_widget is None:
            return
        text = self._stream_widget.toPlainText()
        if not text:
            self._stream_widget.setFixedHeight(30)
            return

        fm = self._stream_widget.fontMetrics()
        vw = self._stream_widget.viewport().width()
        if vw <= 0:
            vw = self._stream_widget.width() - 20
        if vw <= 0:
            vw = 600

        char_width = fm.averageCharWidth() or 8
        chars_per_line = max(1, vw // char_width)

        visual_lines = 0
        for paragraph in text.split('\n'):
            visual_lines += max(1, -(-len(paragraph) // chars_per_line))

        doc = self._stream_widget.document()
        margins = self._stream_widget.contentsMargins()
        height = int(visual_lines * fm.lineSpacing() + 2 * doc.documentMargin())
        height += margins.top() + margins.bottom()
        height += self._stream_widget.frameWidth() * 2
        self._stream_widget.setFixedHeight(max(30, height))


# ── Chat View ────────────────────────────────────────────────────────────

class ChatView(QScrollArea):
    """Scrollable chat view with ChatBubble widgets.

    Replaces the plain QTextEdit for chat output. Each message is a
    ChatBubble that can contain mixed text + images + audio/video players.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(4, 4, 4, 4)
        self._container_layout.setSpacing(2)
        self._container_layout.addStretch()

        self.setWidget(self._container)
        self._bubbles: list[ChatBubble] = []
        self._auto_scroll = True

    def add_bubble(self, role: str) -> ChatBubble:
        """Add a new chat bubble and return it for content population."""
        bubble = ChatBubble(role)
        # Insert before the stretch
        idx = self._container_layout.count() - 1
        self._container_layout.insertWidget(idx, bubble)
        self._bubbles.append(bubble)

        if self._auto_scroll:
            self._scroll_to_bottom()

        return bubble

    def clear_chat(self):
        """Remove all bubbles."""
        for bubble in self._bubbles:
            self._container_layout.removeWidget(bubble)
            bubble.deleteLater()
        self._bubbles.clear()

    def scroll_to_bottom(self):
        """Scroll to the bottom of the chat."""
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(10, lambda: self.verticalScrollBar().setValue(
            self.verticalScrollBar().maximum()
        ))

    def get_last_bubble(self) -> ChatBubble | None:
        return self._bubbles[-1] if self._bubbles else None
