"""
Artifex Assistant V5 — PyQt6 background workers.
QThread-based workers for text generation, pipeline execution,
and tool actions. Replaces daemon threading with signal-based
communication for thread-safe GUI updates.
"""

import threading
import time
import traceback

from PyQt6.QtCore import QThread, pyqtSignal, QTimer, QObject


class TokenBatcher(QObject):
    """Batches streaming tokens and flushes to the GUI at a capped rate.

    Instead of updating the GUI per-token (which causes Tkinter freezes),
    this collects tokens and emits them in batches every flush_interval_ms.
    At 50ms (20 FPS), streaming looks smooth without overwhelming the GUI.
    """

    batch_ready = pyqtSignal(str)  # Emitted with concatenated tokens

    def __init__(self, flush_interval_ms: int = 50, parent=None):
        super().__init__(parent)
        self._buffer: list[str] = []
        self._timer = QTimer(self)
        self._timer.setInterval(flush_interval_ms)
        self._timer.timeout.connect(self._flush)

    def add(self, token: str):
        """Add a token to the buffer. Starts the flush timer if needed."""
        self._buffer.append(token)
        if not self._timer.isActive():
            self._timer.start()

    def _flush(self):
        if self._buffer:
            text = "".join(self._buffer)
            self._buffer.clear()
            self.batch_ready.emit(text)
        else:
            self._timer.stop()

    def flush_now(self):
        """Force immediate flush of any remaining tokens."""
        self._timer.stop()
        if self._buffer:
            text = "".join(self._buffer)
            self._buffer.clear()
            self.batch_ready.emit(text)


class GenerationWorker(QThread):
    """Background worker for text generation (chat/code mode).

    Signals:
        token_received(str): Raw token from streaming (fed to TokenBatcher)
        thinking_received(str): Thinking block content
        response_received(str): Response content (after ThinkFilter)
        finished(str): Final complete response text
        error(str): Error message if generation fails
        status_changed(str): Status updates during loading
        telemetry_received(dict): Per-token forward-pass telemetry frame
            (per-layer norms + MoE routing); see core/forward_telemetry.py
    """

    token_received = pyqtSignal(str)
    thinking_received = pyqtSignal(str)
    response_received = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    telemetry_received = pyqtSignal(dict)

    def __init__(self, engine, messages: list, max_tokens: int,
                 temperature: float, enable_thinking: bool = True,
                 cancel_event: threading.Event = None):
        super().__init__()
        self.engine = engine
        self.messages = list(messages)  # Snapshot
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.cancel_event = cancel_event or threading.Event()
        # Accumulator for thinking-channel tokens. Read by the GUI's
        # _on_generation_finished so action extraction can scan tool
        # blocks (```edit, ```bash, etc.) regardless of whether the
        # model emitted them in the visible response or its private
        # reasoning channel — Gemma 4 in particular tends to put
        # tool calls inside its analysis channel, which the regular
        # `response` variable from generate_streaming never sees.
        self._thinking_text: str = ""

    def run(self):
        try:
            from core.inference import ThinkFilter

            # Set up ThinkFilter to split thinking from response.
            # The thinking callback both emits the streaming signal
            # AND appends to our local accumulator so the GUI handler
            # can read the full thinking text after generation completes.
            def _on_thinking(t):
                self._thinking_text += t
                self.thinking_received.emit(t)

            from core.engine_ollama import OllamaEngine
            starts_in_think = self.enable_thinking and not isinstance(self.engine, OllamaEngine)

            tf = ThinkFilter(
                on_response=lambda t: self.response_received.emit(t),
                on_thinking=_on_thinking,
                starts_in_think=starts_in_think,
            )

            def on_token(text):
                if self.cancel_event.is_set():
                    raise InterruptedError("Generation cancelled")
                self.token_received.emit(text)
                tf.feed(text)

            def on_telemetry(frame):
                # Small JSON-safe dict; Qt marshals the emit to the GUI thread.
                # Best-effort — visualization only, never gates generation.
                self.telemetry_received.emit(frame)

            # Load engine if needed
            if not self.engine.is_loaded():
                self.status_changed.emit("Loading model...")
                self.engine.load(
                    status_callback=lambda m: self.status_changed.emit(m)
                )

            self.status_changed.emit("Generating...")

            response = self.engine.generate_streaming(
                self.messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                on_token=on_token,
                on_telemetry=on_telemetry,
                enable_thinking=self.enable_thinking,
            )

            tf.flush()
            self.finished.emit(response or "")

        except InterruptedError:
            self.finished.emit("[Generation cancelled]")
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class PipelineWorker(QThread):
    """Background worker for non-text pipelines (image, audio, video, etc.).

    Signals:
        progress(int, int, str): current, total, message
        result_ready(object): PipelineResult when done
        error(str): Error message
        status_changed(str): Status updates
    """

    progress = pyqtSignal(int, int, str)
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)
    status_changed = pyqtSignal(str)

    def __init__(self, service, pipeline_type: str, model_path: str = "",
                 kwargs: dict = None,
                 cancel_event: threading.Event = None):
        super().__init__()
        self.service = service
        self.pipeline_type = pipeline_type
        self.model_path = model_path
        self.kwargs = kwargs or {}
        self.cancel_event = cancel_event or threading.Event()

    def run(self):
        try:
            def on_progress(current, total, message):
                self.progress.emit(current, total, message)
                self.status_changed.emit(message)

            t0 = time.perf_counter()
            result = self.service.run_pipeline(
                self.pipeline_type,
                model_path=self.model_path,
                kwargs=self.kwargs,
                progress_callback=on_progress,
                cancel_event=self.cancel_event,
                store_output=True,
            )
            elapsed = time.perf_counter() - t0
            # Wall-clock time for the GUI's perf summary line. The
            # individual pipelines don't track this themselves; this
            # worker is the one place that runs every pipeline type,
            # so timing here covers vision/image/video/audio/3D/voice/
            # embeddings uniformly. setdefault so a pipeline that
            # ever computes its own better timing isn't overwritten.
            if result is not None:
                if result.metadata is None:
                    result.metadata = {}
                result.metadata.setdefault("elapsed_s", elapsed)
            self.result_ready.emit(result)

        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class ActionWorker(QThread):
    """Background worker for running agent tool actions.

    Signals:
        output_ready(str, str): (action_display, output_text)
        error(str): Error message
        status_changed(str): Status update
    """

    output_ready = pyqtSignal(str, str)
    error = pyqtSignal(str)
    status_changed = pyqtSignal(str)

    def __init__(self, actions: list):
        super().__init__()
        self.actions = actions

    def run(self):
        try:
            from tools.agent_tools import run_agent_action

            for action in self.actions:
                display = getattr(action, "display", str(action))
                self.status_changed.emit(f"Running: {display}")
                try:
                    # The user clicked this action in the panel, so consent
                    # is established — the confirm_cb accepts. The policy
                    # screen (injection validator, blocklist, egress) still
                    # runs inside run_agent_action; a click is consent, not
                    # a safety review.
                    success, output = run_agent_action(
                        action, confirm_cb=lambda _a, _d: True)
                    self.output_ready.emit(display, output or "(no output)")
                except Exception as e:
                    self.output_ready.emit(display, f"ERROR: {e}")

        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")
