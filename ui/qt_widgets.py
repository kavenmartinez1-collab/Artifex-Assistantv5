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
        # Styled by ui/qt_theme.py (DropZone / DropZone[dragOver="true"]).

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        self._label = QLabel("Drop files here or")
        layout.addWidget(self._label)

        self._browse_btn = QPushButton("Browse")
        self._browse_btn.setMaximumWidth(80)
        self._browse_btn.clicked.connect(self._on_browse)
        layout.addWidget(self._browse_btn)

        self._file_label = QLabel("")
        self._file_label.setProperty("class", "file")
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
        self._queued_label = None
        self._stream_text_len = 0
        self._stream_newlines = 0
        self._stream_last_line_chars = 0
        self._last_height_update_at_chars = 0

        # Header: role tag + timestamp. Colours, the role bar and the tinted
        # background come from ui/qt_theme.py via the "role" property.
        self.setProperty("role", role)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        tag = {"assistant": "ARTIFEX", "user": "YOU"}.get(role, role.upper())
        role_label = QLabel(f"◢ {tag}")
        role_label.setProperty("class", "bubbleRole")
        header.addWidget(role_label)
        header.addStretch()
        stamp = QLabel(time.strftime("%H:%M:%S"))
        stamp.setProperty("class", "bubbleTime")
        header.addWidget(stamp)
        self._layout.addLayout(header)

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
        label.setProperty("class", "bubbleText")
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

    def add_widget(self, widget: QWidget, top: bool = False):
        """Put any widget in the bubble (top=True: right under the header)."""
        if top:
            self._layout.insertWidget(1, widget)
        else:
            self._layout.addWidget(widget)
        return widget

    def set_pending(self, on: bool):
        """Dim a message that was sent but not read yet ("queued")."""
        if on and self._queued_label is None:
            self._queued_label = QLabel("queued — it reads this at its next step")
            self._queued_label.setProperty("class", "bubbleQueued")
            self._layout.addWidget(self._queued_label)
        elif not on and self._queued_label is not None:
            self._layout.removeWidget(self._queued_label)
            self._queued_label.deleteLater()
            self._queued_label = None
        self.setProperty("pending", "true" if on else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def set_markdown(self, text: str):
        """Replace the bubble's text with `text` rendered as markdown (a
        finished reply). Long text stays plain: a word-wrapped rich label
        that size stalls the layout, same as add_text's cap."""
        for i in range(self._layout.count() - 1, 0, -1):
            w = self._layout.itemAt(i).widget()
            if w is None:
                continue
            if w is self._stream_widget or w.property("class") == "bubbleText":
                self._layout.removeWidget(w)
                w.deleteLater()
        self._stream_widget = None
        if not text:
            return None
        label = self.add_text(text)
        if len(text) <= self._MAX_LABEL_CHARS:
            label.setTextFormat(Qt.TextFormat.MarkdownText)
            label.setOpenExternalLinks(False)
        return label

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
            self._track_appended(text)
            self._maybe_update_stream_height()
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
        edit.setStyleSheet("background: transparent; border: none;")
        edit.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        edit.setMinimumHeight(30)
        self._layout.addWidget(edit)
        self._stream_widget = edit

        if existing:
            edit.setPlainText(existing)
            self._track_appended(existing)

        cursor = edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self._track_appended(text)
        self._maybe_update_stream_height()

    def finalize_stream(self):
        """Size the stream widget to its final content.

        Height updates are throttled to every 500 chars while streaming, so
        without this the tail of a reply was never sized (clipped: the widget
        has no scrollbars) and a reply under 500 chars never left the
        default 192 px size hint.
        """
        if self._stream_widget is not None:
            self._update_stream_height()
            self._last_height_update_at_chars = self._stream_text_len

    def _track_appended(self, text: str):
        """Maintain incremental counters so we can size the stream widget
        without rescanning the full document on every append (was O(n²) and
        starved the worker thread of GIL during long generations)."""
        for ch in text:
            if ch == '\n':
                self._stream_newlines += 1
                self._stream_last_line_chars = 0
            else:
                self._stream_last_line_chars += 1
        self._stream_text_len += len(text)

    def _maybe_update_stream_height(self):
        """Throttle height updates so we don't recompute on every batch.
        500-char gating keeps GUI responsive without holding the GIL during
        streaming."""
        if self._stream_text_len - self._last_height_update_at_chars >= 500:
            self._update_stream_height()
            self._last_height_update_at_chars = self._stream_text_len

    def _update_stream_height(self):
        """Resize stream widget from incrementally tracked counters — O(1)
        regardless of total response length."""
        if self._stream_widget is None:
            return
        if self._stream_text_len == 0:
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

        # Approximate visual line count from newlines + wrapped chars on the
        # current paragraph.  Earlier paragraphs are approximated by their
        # average chars-per-line; close enough for sizing, avoids O(n) split.
        avg_chars_per_paragraph = (
            (self._stream_text_len - self._stream_last_line_chars)
            / max(1, self._stream_newlines)
            if self._stream_newlines else 0
        )
        wrapped_earlier = (
            self._stream_newlines
            * max(1, -(-int(avg_chars_per_paragraph) // chars_per_line))
        )
        wrapped_current = max(1, -(-self._stream_last_line_chars // chars_per_line))
        visual_lines = wrapped_earlier + wrapped_current

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

    # Retained rendered bubbles. The model's own history already decays
    # (core.inference.auto_compact_if_needed / compress_history), but the
    # rendered view had no bound at all — every bubble from a session stayed
    # live, holding its text and any embedded image/audio players. A long
    # session therefore grew monotonically with nothing to release it.
    # Oldest bubbles are evicted first, so the view keeps the recent window
    # and lets older turns fall out. Scroll-back is what is traded away;
    # the conversation itself is unaffected (self.messages in the window is
    # the source of truth, and sessions are saved separately).
    MAX_BUBBLES = int(os.environ.get("ARTIFEX_MAX_CHAT_BUBBLES", "400"))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self._container = QWidget()
        # Transparent (ui/qt_theme.py) so paintEvent's HUD grid shows through.
        self._container.setObjectName("chatContainer")
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(4, 4, 4, 4)
        self._container_layout.setSpacing(2)
        # Stretch factor 1 so spare height goes to the spacer, not to the
        # bubbles' MinimumExpanding labels (which padded short messages).
        self._container_layout.addStretch(1)

        self.setWidget(self._container)
        self._bubbles: list[ChatBubble] = []
        # Follow new output only while the view is at the bottom: scrolling
        # up to read turns it off, returning to the bottom turns it back on.
        self._auto_scroll = True
        self.verticalScrollBar().valueChanged.connect(self._on_scroll_value)

    @staticmethod
    def at_bottom(value: int, maximum: int, slack: int = 24) -> bool:
        """True when a scrollbar at `value` of `maximum` is (nearly) at the end."""
        return value >= maximum - slack

    def _on_scroll_value(self, value: int):
        self._auto_scroll = self.at_bottom(value, self.verticalScrollBar().maximum())

    def paintEvent(self, event):
        # QAbstractScrollArea routes the viewport's paint here.
        from ui.qt_cyber import paint_backdrop
        painter = QPainter(self.viewport())
        paint_backdrop(painter, self.viewport().rect())
        painter.end()

    def add_bubble(self, role: str) -> ChatBubble:
        """Add a new chat bubble and return it for content population."""
        bubble = ChatBubble(role)
        # Insert before the stretch
        idx = self._container_layout.count() - 1
        self._container_layout.insertWidget(idx, bubble)
        self._bubbles.append(bubble)
        self._trim_bubbles()

        # The user's own message always brings the view back to the bottom.
        if role == "user":
            self._auto_scroll = True
        if self._auto_scroll:
            self._scroll_to_bottom()

        return bubble

    def _trim_bubbles(self):
        """Evict oldest bubbles once past MAX_BUBBLES.

        Only ever drops from the front, never the bubble just returned by
        add_bubble: callers hold that reference and stream into it, and
        touching a deleteLater'd widget raises RuntimeError inside a Qt
        slot — which PyQt6 escalates to qFatal() and a process abort. A cap
        below 2 would make that reachable, so the floor is enforced here
        rather than trusting the env var.
        """
        cap = max(2, self.MAX_BUBBLES)
        while len(self._bubbles) > cap:
            stale = self._bubbles.pop(0)
            self._container_layout.removeWidget(stale)
            stale.setParent(None)
            stale.deleteLater()

    def clear_chat(self):
        """Remove all bubbles."""
        for bubble in self._bubbles:
            self._container_layout.removeWidget(bubble)
            bubble.deleteLater()
        self._bubbles.clear()

    def scroll_to_bottom(self, force: bool = False):
        """Follow new output to the bottom of the chat.

        A no-op while the user has scrolled up to read (it used to yank the
        view back down on every streamed batch); force=True overrides.
        """
        if force:
            self._auto_scroll = True
        if self._auto_scroll:
            self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        from PyQt6.QtCore import QTimer

        def _go():
            if self._auto_scroll:   # the user may have scrolled up meanwhile
                self.verticalScrollBar().setValue(
                    self.verticalScrollBar().maximum())

        QTimer.singleShot(10, _go)

    def get_last_bubble(self) -> ChatBubble | None:
        return self._bubbles[-1] if self._bubbles else None


    def add_item(self, widget: QWidget, follow: bool = False) -> QWidget:
        """Add any row widget (a tool row, an approval card, a note) to the
        conversation. Rows count toward MAX_BUBBLES like bubbles do."""
        idx = self._container_layout.count() - 1
        self._container_layout.insertWidget(idx, widget)
        self._bubbles.append(widget)
        self._trim_bubbles()
        if follow:
            self._auto_scroll = True
        if self._auto_scroll:
            self._scroll_to_bottom()
        return widget

    def remove_item(self, widget: QWidget):
        """Take one row out again (an empty reply, a retracted message)."""
        if widget in self._bubbles:
            self._bubbles.remove(widget)
            self._container_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()

    def move_to_end(self, widget: QWidget):
        """Re-seat a row at the bottom (a queued message, once it is read)."""
        if widget in self._bubbles:
            self._bubbles.remove(widget)
            self._bubbles.append(widget)
            self._container_layout.removeWidget(widget)
            self._container_layout.insertWidget(
                self._container_layout.count() - 1, widget)
            if self._auto_scroll:
                self._scroll_to_bottom()


# ── Agent conversation rows ──────────────────────────────────────────────
#
# The AGENT tab is a ChatView like the chat tab, plus these rows: a fold for
# the model's thinking, one collapsed line per tool call, the approval card,
# and dim one-line notes. Colours come from ui/qt_theme.py (object classes
# below), so every theme restyles them.

_TOOL_FENCE_RE = None


def strip_tool_syntax(text: str) -> str:
    """A reply's prose without the tool calls it made (@read_file("x"), a
    ```bash block): the tool row under the bubble shows those in full. A reply
    with an edit block is left exactly as it is: the file text it writes can
    hold fences and @markers of its own, and those are content."""
    import re
    global _TOOL_FENCE_RE
    if "```edit" in (text or ""):
        return text
    if _TOOL_FENCE_RE is None:
        _TOOL_FENCE_RE = (
            re.compile(r"```(?:bash|sh|shell|powershell|ps1|pwsh|cmd|python|py)\b"
                       r"[^\n]*\n.*?\n```[ \t]*(?=\n|$)", re.S),
            re.compile(r"^[ \t]*@[a-z_]+\([^\n]*\)[ \t]*$", re.M),
        )
    out = _TOOL_FENCE_RE[0].sub("", text or "")
    out = _TOOL_FENCE_RE[1].sub("", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def fmt_secs(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s}s" if s < 60 else f"{s // 60}m {s % 60:02d}s"


class _FoldRow(QFrame):
    """A one-line header that opens a body built on first use."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(8, 1, 8, 1)
        self._outer.setSpacing(2)
        self._head = QPushButton()
        self._head.setFlat(True)
        self._head.setCursor(Qt.CursorShape.PointingHandCursor)
        self._head.setProperty("class", "foldHead")
        self._head.clicked.connect(self.toggle)
        self._outer.addWidget(self._head)
        self._body = None
        self._open = False

    def _make_body(self) -> QWidget:      # pragma: no cover - overridden
        return QWidget()

    def toggle(self):
        self._open = not self._open
        if self._open and self._body is None:
            self._body = self._make_body()
            self._outer.addWidget(self._body)
        if self._body is not None:
            self._body.setVisible(self._open)
        self._render_head()

    def _render_head(self):               # pragma: no cover - overridden
        pass

    @staticmethod
    def _text_box(text: str, max_height: int = 240) -> QPlainTextEdit:
        box = QPlainTextEdit()
        box.setReadOnly(True)
        box.setProperty("class", "foldBody")
        box.setPlainText(text)
        lines = min(text.count("\n") + 2, 40)
        box.setFixedHeight(min(max_height, max(44, lines * 17 + 10)))
        return box


class ThinkFold(_FoldRow):
    """ "Thought for 12s" - closed; click to read the round's thinking."""

    def __init__(self, text: str, seconds: float = 0.0, parent=None):
        super().__init__(parent)
        self.setProperty("class", "thinkFold")
        self._text = text or ""
        self._label = (f"Thought for {fmt_secs(seconds)}" if seconds >= 1
                       else "Thought briefly")
        self._render_head()

    def _make_body(self):
        return self._text_box(self._text, 220)

    def _render_head(self):
        self._head.setText(("▾ " if self._open else "▸ ") + self._label)


class ToolRow(_FoldRow):
    """One tool call: state, tool, what was asked, how it went. The whole
    request and the output are inside the fold."""

    _OUTPUT_SHOWN = 20000      # chars of output put in the fold

    def __init__(self, tool: str, display: str, full: str = "", risk: str = "",
                 parent=None):
        super().__init__(parent)
        self.setProperty("class", "toolRow")
        self.setProperty("risk", risk or "NONE")
        self._tool = tool or "action"
        disp = (display or "").strip()
        if disp.startswith(self._tool + ": "):
            disp = disp[len(self._tool) + 2:]
        self._display = " ".join(disp.split())
        self._full = full or display or ""
        self._glyph = "○"
        self._result = risk if risk not in ("", "SAFE", "LOW", "NONE") else ""
        self._output = ""
        self._why = ""
        self._preview = None
        self._render_head()

    def set_state(self, glyph: str, state: str = "", result: str = None):
        self._glyph = glyph
        self._head.setProperty("state", state)
        self._head.style().unpolish(self._head)
        self._head.style().polish(self._head)
        if result is not None:
            self._result = result
        self._render_head()

    def set_output(self, text: str, failed: bool = False):
        self._output = text or ""
        if failed and self._output.strip() and self._preview is None:
            # A failure shows its last lines without being opened.
            tail = "\n".join(self._output.strip().splitlines()[-3:])
            self._preview = QLabel(tail[:600])
            self._preview.setWordWrap(True)
            self._preview.setProperty("class", "toolFail")
            self._outer.addWidget(self._preview)
        self._rebuild_body()

    def set_reason(self, why: str):
        self._why = why or ""
        self._rebuild_body()

    def _rebuild_body(self):
        if self._body is not None:
            was_open = self._open
            self._outer.removeWidget(self._body)
            self._body.deleteLater()
            self._body = None
            if was_open:
                self._open = False
                self.toggle()

    def _make_body(self):
        text = self._full
        if self._why:
            text += "\n\n" + self._why
        if self._output:
            out = self._output
            if len(out) > self._OUTPUT_SHOWN:
                out = (out[:self._OUTPUT_SHOWN]
                       + f"\n[... {len(self._output) - self._OUTPUT_SHOWN:,} more characters]")
            text += "\n\n── output ──\n" + out
        return self._text_box(text)

    def _render_head(self):
        if self._preview is not None:
            self._preview.setVisible(not self._open)
        disp = self._display if len(self._display) <= 96 else self._display[:95] + "…"
        tail = f"   — {self._result}" if self._result else ""
        self._head.setText(f"{self._glyph}  {self._tool}  {disp}{tail}")
        self._head.setToolTip(self._full[:1500])


class ApprovalCard(QFrame):
    """The approval prompt, in the conversation under the tool row it is
    about. `decided` carries approve | always | deny; typing a reply in the
    composer answers it too (the window does that). Shrinks to one line once
    answered."""

    decided = pyqtSignal(str)

    def __init__(self, what: str, why: str = "", gate: bool = False, parent=None):
        super().__init__(parent)
        self.setProperty("class", "approvalCard")
        self.is_gate = gate
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 7, 10, 8)
        lay.setSpacing(5)
        self._title = QLabel("PAUSED" if gate else "APPROVAL REQUIRED")
        self._title.setProperty("class", "approvalTitle")
        lay.addWidget(self._title)
        self._parts = []
        # Read-only text box, not a QLabel: it holds the FULL request (a whole
        # script or both sides of an edit) as plain text and scrolls when long.
        what_box = QPlainTextEdit()
        what_box.setReadOnly(True)
        what_box.setProperty("class", "foldBody")
        what_box.setPlainText(what)
        what_box.setFixedHeight(min(200, max(40, (what.count("\n") + 2) * 17 + 8)))
        lay.addWidget(what_box)
        self._parts.append(what_box)
        if why:
            why_label = QLabel(why)
            why_label.setWordWrap(True)
            why_label.setProperty("class", "dim")
            lay.addWidget(why_label)
            self._parts.append(why_label)
        row = QHBoxLayout()
        row.setSpacing(6)
        buttons = [("Continue" if gate else "Approve", "approve", "primary")]
        if not gate:
            # A gate/breaker pause has nothing to remember, and nothing to deny.
            buttons += [("Always", "always", "secondary"), ("Deny", "deny", "secondary")]
        for label, decision, cls in buttons:
            b = QPushButton(label)
            b.setProperty("class", cls)
            if decision == "always":
                b.setToolTip(
                    "Approve, and don't ask again in this conversation for the "
                    "same kind of action (same command prefix, or edits to the "
                    "same file). CRITICAL actions still ask every time.")
            b.clicked.connect(lambda _=False, d=decision: self.decided.emit(d))
            row.addWidget(b)
            self._parts.append(b)
        hint = QLabel("or type a message to continue with it" if gate
                      else "or type a reply to say what to do instead")
        hint.setProperty("class", "dim")
        row.addWidget(hint, 1)
        self._parts.append(hint)
        lay.addLayout(row)
        self.answered = False

    def mark(self, label: str):
        """Answered: keep one line saying how."""
        if self.answered:
            return
        self.answered = True
        for w in self._parts:
            w.setVisible(False)
        self._title.setText(f"APPROVAL · {label.upper()}")
        self.setProperty("answered", "true")
        self.style().unpolish(self)
        self.style().polish(self)


class ComposerEdit(QPlainTextEdit):
    """The message box under a conversation: Enter sends, Shift+Enter makes a
    new line, Esc interrupts."""

    submitted = pyqtSignal()
    escaped = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "composer")
        self.setTabChangesFocus(True)
        self.setFixedHeight(58)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
                event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.submitted.emit()
            return
        if key == Qt.Key.Key_Escape:
            self.escaped.emit()
            return
        super().keyPressEvent(event)
