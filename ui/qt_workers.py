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

from ui.gui_logic import read_vram_devices, vram_summary


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
        compacted(list): The full message list after a compaction ran at
            the start of this generation (compact_cut > 0); the GUI adopts
            it as its history
    """

    token_received = pyqtSignal(str)
    thinking_received = pyqtSignal(str)
    response_received = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    telemetry_received = pyqtSignal(dict)
    compacted = pyqtSignal(list)

    def __init__(self, engine, messages: list, max_tokens: int,
                 temperature: float, enable_thinking: bool = True,
                 cancel_event: threading.Event = None, compact_cut: int = 0):
        super().__init__()
        self.engine = engine
        self.messages = list(messages)  # Snapshot
        # Conversation messages to fold into a synopsis before generating
        # (core.inference.plan_compaction); 0 = none. Done here, off the GUI
        # thread, because the summary is a model call.
        self.compact_cut = compact_cut
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
        # Visible answer so far (what the bubble shows). On cancel this is
        # what goes into the history, so the model later sees what the user
        # saw instead of a "[Generation cancelled]" placeholder.
        self._response_text: str = ""
        self.cancelled: bool = False
        # Forward-pass telemetry is opt-in: the GUI turns it on only while
        # the Forward Pass tab is showing (it hooks every layer per token).
        self.want_telemetry: bool = False

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

            # Engines that emit explicit <think> tags (llama.cpp, ollama,
            # claude CLI) start the stream OUTSIDE a think block; only
            # transformers/webgpu pre-fill <think>. Same rule as api/server.py
            # and core/agent_loop.py — an isinstance(OllamaEngine) check here
            # sent a llama.cpp model's whole answer to the thinking pane.
            starts_in_think = bool(
                self.enable_thinking
                and getattr(self.engine, "stream_starts_in_think", True))

            def _on_response(t):
                self._response_text += t
                self.response_received.emit(t)

            tf = ThinkFilter(
                on_response=_on_response,
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

            if self.compact_cut:
                from core.inference import compact_with_engine
                self.status_changed.emit("Compacting context...")
                self.messages = compact_with_engine(
                    self.messages, self.compact_cut, self.engine,
                    enable_thinking=self.enable_thinking)
                self.compacted.emit(list(self.messages))
                if self.cancel_event.is_set():
                    raise InterruptedError("Generation cancelled")

            self.status_changed.emit("Generating...")

            response = self.engine.generate_streaming(
                self.messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                on_token=on_token,
                on_telemetry=on_telemetry if self.want_telemetry else None,
                enable_thinking=self.enable_thinking,
            )

            tf.flush()
            self.finished.emit(response or "")

        except InterruptedError:
            # The partial answer, not a placeholder: see _response_text.
            self.cancelled = True
            try:
                tf.flush()   # text the filter was still holding back
            except Exception:
                pass
            self.finished.emit(self._response_text)
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


class BackendSwitchWorker(QThread):
    """Switch the active backend and list its models off the GUI thread.

    core.config.set_active_backend("ollama") probes the Ollama server with
    retries and sleeps, and get_model_names() repeats the probe when it found
    nothing — with Ollama not running that froze the window for seconds.

    Signals:
        switched(str, list): backend name, its model names (possibly empty)
    """

    switched = pyqtSignal(str, list)

    def __init__(self, backend: str):
        super().__init__()
        self.backend = backend

    def run(self):
        names = []
        try:
            from core.config import set_active_backend, get_model_names
            set_active_backend(self.backend)
            names = list(get_model_names() or [])
        except Exception:
            names = []
        self.switched.emit(self.backend, names)


class ResourceMonitor(QThread):
    """Samples VRAM / RAM / CPU off the GUI thread.

    Signals:
        sample(dict): {"vram": vram_summary() tuple or None,
                       "ram": (used_gb, total_gb, fraction) or None,
                       "cpu": percent or None}
    """

    sample = pyqtSignal(dict)

    def __init__(self, interval_s: float = 3.0):
        super().__init__()
        self._interval_s = interval_s
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            data = {"vram": vram_summary(read_vram_devices()), "ram": None, "cpu": None}
            try:
                import psutil
                ram = psutil.virtual_memory()
                data["ram"] = (ram.used / (1024 ** 3), ram.total / (1024 ** 3),
                               ram.percent / 100)
                data["cpu"] = psutil.cpu_percent()
            except Exception:
                pass
            if self._stop.is_set():
                break
            self.sample.emit(data)
            self._stop.wait(self._interval_s)
