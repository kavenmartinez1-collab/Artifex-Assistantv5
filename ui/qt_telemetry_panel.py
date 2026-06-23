"""3D forward-pass telemetry panel (pyqtgraph / OpenGL).

Faithful, not decorative: every animated pulse is ONE generated token's forward
pass. A frame (one token) arrives, a front travels input -> output through the
layer slices, and each layer lights from its REAL telemetry the moment the front
reaches it — in the same order the GPU computes them (layer 0, 1, ..., N-1, head).
When the front reaches the head the next-token distribution reads out; then the
next token's pulse begins. Pulses are paced to the measured token rate, so the
wave you see corresponds to actual generation.

  * layers           -> slices along the X axis (golden-angle disc per slice)
  * MoE              -> the per-token top-k experts glow; the rest stay dark
  * dense            -> per-layer residual norm (log-compressed) glow
  * routing/edges    -> revealed layer-by-layer as the front passes
  * token word       -> rides the front; next-token bars read out at the head

Frames arrive via push_frame(frame) on the GUI thread and queue; a timer drains
them one pulse at a time. Renders only when the tab is visible. Degrades to a
label (never crashes) if pyqtgraph / PyOpenGL are unavailable.
"""
import math
import time
from collections import deque

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor

import os
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

_GL_OK = True
_GL_ERR = ""
try:
    import numpy as np
    import pyqtgraph.opengl as gl
except Exception as e:  # pragma: no cover - depends on optional deps
    _GL_OK = False
    _GL_ERR = f"{type(e).__name__}: {e}"

GOLDEN = math.pi * (3.0 - math.sqrt(5.0))   # golden angle (rad)
SPAN = 50.0                                  # x-extent of the layer stack

MIN_PULSE_DT = 0.06      # never cross the stack faster than this (visibility)
MAX_PULSE_DT = 2.0       # never slower than this (don't stall on a slow model)
PENDING_CAP = 24         # bound the queue if the model outruns the animation
AHEAD = 0.06             # brightness of layers the front hasn't reached
TRAIL = 0.50             # brightness of already-computed layers
LEAD_BAND = 1.5          # how many layers around the front read as "computing"


class TelemetryPanel(QWidget):
    """Live 3D visualization of the transformer forward pass (one pulse / token)."""

    def __init__(self, nodes_per_layer=128, parent=None):
        super().__init__(parent)
        self._M = nodes_per_layer
        self._n_layers = 0
        self._positions = None         # (total, 3) float32

        # pulse state -------------------------------------------------------
        self._pending = deque()        # frames (tokens) awaiting their pulse
        self._cur = None               # frame currently animating
        self._cur_rgba = None          # (total,4) destination colors for _cur
        self._cur_size = None          # (total,)  destination sizes
        self._cur_edges = None         # (S,2,3)  routing/flow segments
        self._cur_edge_layer = None    # (S,)     higher layer index per segment
        self._front = 0.0              # leading edge position (0 .. n_layers)
        self._pulse_done = False

        # display state -----------------------------------------------------
        self._active = 0
        self._mode = "idle"
        self._peak_norm = 0.0
        self._disp_top_logits = None   # revealed when the front reaches the head
        self._disp_token = ""
        self._anim_t = None
        self._last_t = None
        self._tps = 0.0
        self._pulse_dt = 0.2

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not _GL_OK:
            lbl = QLabel(
                "3D forward-pass view needs pyqtgraph + PyOpenGL\n"
                f"({_GL_ERR})\n\n    pip install pyqtgraph PyOpenGL"
            )
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl)
            self._view = None
            return

        self._view = gl.GLViewWidget()
        self._view.setBackgroundColor(8, 10, 16)
        self._view.opts["distance"] = 70
        self._view.opts["fov"] = 60
        layout.addWidget(self._view)

        self._scatter = gl.GLScatterPlotItem()
        self._scatter.setGLOptions("additive")          # glow on overlap
        self._view.addItem(self._scatter)

        self._edges = gl.GLLinePlotItem(width=1.0, antialias=True, mode="lines")
        self._edges.setGLOptions("additive")
        self._view.addItem(self._edges)

        # Current token, riding the front in 3D (optional — needs GLTextItem)
        self._token_item = None
        try:
            self._token_item = gl.GLTextItem(pos=np.zeros(3, dtype=np.float32),
                                             text="", color=QColor(150, 210, 255))
            self._view.addItem(self._token_item)
        except Exception:
            self._token_item = None

        # HUD overlay (transparent, click-through)
        self._hud = QLabel(self._view)
        self._hud.setStyleSheet(
            "color:#7fd3ff; background:transparent; font:9pt 'Consolas';")
        self._hud.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._hud.move(8, 6)
        self._hud.setText("forward-pass · idle — run a prompt on the transformers backend")
        self._hud.adjustSize()

        # Next-token distribution overlay (top-k logits, read out at the head)
        self._logits_hud = QLabel(self._view)
        self._logits_hud.setStyleSheet(
            "color:#bfe6ff; background:transparent; font:9pt 'Consolas';")
        self._logits_hud.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._logits_hud.move(8, 26)
        self._logits_hud.setText("")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(16)                            # ~60 FPS for smooth pulses

    @property
    def n_layers(self):
        return self._n_layers

    # ── geometry ──────────────────────────────────────────────────────
    def _build_geometry(self, n_layers):
        self._n_layers = n_layers
        M = self._M
        total = n_layers * M
        pos = np.zeros((total, 3), dtype=np.float32)
        denom = max(1, n_layers - 1)
        for li in range(n_layers):
            x = (li / denom - 0.5) * SPAN
            bulge = 0.35 + 0.65 * math.sin(math.pi * li / denom)   # fat mid-stack
            R = 11.0 * bulge
            for j in range(M):
                r = math.sqrt((j + 0.5) / M) * R
                th = j * GOLDEN
                idx = li * M + j
                pos[idx, 0] = x
                pos[idx, 1] = r * math.cos(th)
                pos[idx, 2] = r * math.sin(th)
        self._positions = pos

    @staticmethod
    def _idle_rgba(total):
        rgba = np.zeros((total, 4), dtype=np.float32)
        rgba[:, 2] = 0.18      # dim blue
        rgba[:, 3] = 0.08
        return rgba

    # ── frame -> destination node arrays (the fully-computed token state) ──
    def _frame_to_arrays(self, frame):
        n = int(frame.get("n_layers", self._n_layers) or self._n_layers)
        M = self._M
        total = n * M
        rgba = self._idle_rgba(total)
        size = np.full(total, 2.0, dtype=np.float32)
        layers = frame.get("layers", [])
        is_moe = bool(frame.get("moe"))
        seg_pts = []      # each entry: (p_a, p_b)
        seg_layer = []    # the higher layer index that segment reaches
        active = 0
        peak = 0.0

        if is_moe:
            prev_lit = None
            for li in range(min(n, len(layers))):
                experts = layers[li].get("experts")
                lit = []
                if experts:
                    for eid, w in experts:
                        slot = li * M + (int(eid) % M)
                        w = float(w)
                        rgba[slot] = (1.0, 0.45 + 0.55 * w, 0.12 + 0.5 * w,
                                      min(1.0, 0.45 + 0.55 * w))
                        size[slot] = 6.0 + 9.0 * w
                        lit.append(slot)
                        active += 1
                if prev_lit and lit:
                    for a in prev_lit:
                        for b in lit:
                            seg_pts.append((self._positions[a], self._positions[b]))
                            seg_layer.append(li)
                prev_lit = lit
            mode = "moe"
        else:
            norms = [float(layers[li].get("norm") or 0.0)
                     for li in range(min(n, len(layers)))]
            peak = max(norms) if norms else 0.0
            # Log-compress: a massive-activation outlier (one layer ~500+ while
            # the rest sit ~20-50) would otherwise flatten every other layer to
            # black under linear scaling. log1p lifts the whole stack into view.
            lp = math.log1p(peak) or 1.0
            for li, nrm in enumerate(norms):
                inten = math.log1p(max(0.0, nrm)) / lp if lp > 0 else 0.0
                inten = max(0.0, min(1.0, inten))
                sl = slice(li * M, (li + 1) * M)
                rgba[sl] = (0.10, 0.30 + 0.60 * inten,
                            0.55 + 0.45 * inten, 0.12 + 0.55 * inten)
                size[sl] = 2.0 + 5.0 * inten
            step = max(1, M // 6)
            for li in range(min(n, len(norms)) - 1):
                a = math.log1p(max(0.0, norms[li])) / lp
                b = math.log1p(max(0.0, norms[li + 1])) / lp
                if a > 0.35 and b > 0.35:
                    for j in range(0, M, step):
                        seg_pts.append((self._positions[li * M + j],
                                        self._positions[(li + 1) * M + j]))
                        seg_layer.append(li + 1)
            mode = "dense"

        if seg_pts:
            edges = np.asarray(seg_pts, dtype=np.float32)        # (S,2,3)
            elayer = np.asarray(seg_layer, dtype=np.float32)     # (S,)
            if len(edges) > 4000:
                edges, elayer = edges[:4000], elayer[:4000]
        else:
            edges = np.zeros((0, 2, 3), dtype=np.float32)
            elayer = np.zeros((0,), dtype=np.float32)
        return rgba, size, edges, elayer, active, mode, peak

    # ── queue / pulse control ─────────────────────────────────────────
    def push_frame(self, frame):
        """Slot: receive one token's telemetry frame (GUI thread)."""
        if self._view is None:
            return
        n = int(frame.get("n_layers", 0) or 0)
        if n <= 0:
            return
        if n != self._n_layers or self._positions is None:
            self._build_geometry(n)

        self._pending.append(frame)
        while len(self._pending) > PENDING_CAP:
            self._pending.popleft()              # bound memory if we fall behind

        # smoothed tok/s from real arrival cadence -> paces the pulse
        t = time.perf_counter()
        if self._last_t is not None:
            dt = t - self._last_t
            if dt > 0:
                inst = 1.0 / dt
                self._tps = (0.8 * self._tps + 0.2 * inst) if self._tps else inst
        self._last_t = t

        if self._cur is None or self._pulse_done:
            self._promote()

    def _promote(self):
        if not self._pending:
            return
        self._cur = self._pending.popleft()
        self._pulse_done = False
        self._front = 0.0
        (self._cur_rgba, self._cur_size, self._cur_edges, self._cur_edge_layer,
         self._active, self._mode, self._peak_norm) = self._frame_to_arrays(self._cur)

    def _advance(self, dt):
        """Move the front for the active pulse; finalize + chain at the head."""
        if self._cur is None or self._pulse_done:
            return
        if self._tps > 0:
            self._pulse_dt = min(MAX_PULSE_DT, max(MIN_PULSE_DT, 1.0 / self._tps))
        eff = max(MIN_PULSE_DT, self._pulse_dt) / (1.0 + 0.5 * len(self._pending))
        self._front += (self._n_layers / eff) * dt
        if self._front >= self._n_layers:
            # reached the head — the prediction reads out now
            self._disp_top_logits = self._cur.get("top_logits")
            self._disp_token = self._cur.get("token") or ""
            if self._pending:
                self._promote()
            else:
                self._pulse_done = True
                self._front = self._n_layers + 2.0   # hold the completed pattern

    def reset(self):
        """Clear state at the start of a new generation."""
        self._pending.clear()
        self._cur = None
        self._cur_rgba = self._cur_size = self._cur_edges = self._cur_edge_layer = None
        self._pulse_done = False
        self._front = 0.0
        self._active = 0
        self._mode = "idle"
        self._peak_norm = 0.0
        self._disp_top_logits = None
        self._disp_token = ""
        self._tps = 0.0
        self._last_t = None
        self._anim_t = None
        if self._view is not None:
            if self._token_item is not None:
                try:
                    self._token_item.setData(text="")
                except Exception:
                    pass
            self._logits_hud.setText("")

    # ── render loop ───────────────────────────────────────────────────
    def _animate(self):
        if self._view is None:
            return
        if not self.isVisible():
            return          # don't burn CPU/GPU rendering a hidden tab

        self._view.orbit(0.18, 0)            # gentle auto-rotate

        # real elapsed time -> pulses track wall-clock (and thus generation)
        now = time.perf_counter()
        dt = (now - self._anim_t) if self._anim_t else 0.016
        self._anim_t = now
        dt = min(dt, 0.1)                     # clamp after a stall/tab-switch

        if self._cur is None or self._positions is None:
            if self._positions is not None:
                total = len(self._positions)
                self._scatter.setData(pos=self._positions,
                                      color=self._idle_rgba(total),
                                      size=np.full(total, 2.0, dtype=np.float32),
                                      pxMode=True)
                self._edges.setData(pos=np.zeros((0, 3), dtype=np.float32))
            self._hud.setText(
                "forward-pass · idle — run a prompt on the transformers backend")
            self._hud.adjustSize()
            if self._logits_hud.text():
                self._logits_hud.setText("")
            return

        self._advance(dt)

        # reveal layers in computation order: ahead of the front = dark,
        # at the front = computing (bright), behind = computed (trail).
        n, M, f = self._n_layers, self._M, self._front
        d = f - np.arange(n)
        reveal = np.full(n, TRAIL, dtype=np.float32)
        reveal[d < 0.0] = AHEAD
        reveal[(d >= 0.0) & (d < LEAD_BAND)] = 1.0
        node_rev = np.repeat(reveal, M)

        disp_rgba = self._cur_rgba * node_rev[:, None]
        disp_size = self._cur_size * (0.4 + 0.6 * node_rev)
        self._scatter.setData(pos=self._positions, color=disp_rgba,
                              size=disp_size, pxMode=True)

        if self._cur_edges is not None and self._cur_edges.shape[0]:
            vis = self._cur_edges[self._cur_edge_layer <= f].reshape(-1, 3)
        else:
            vis = np.zeros((0, 3), dtype=np.float32)
        self._edges.setData(pos=vis, color=(0.30, 0.62, 1.0, 0.25))

        # token word rides the front
        if self._token_item is not None:
            denom = max(1, n - 1)
            x = (min(f, n - 1) / denom - 0.5) * SPAN
            tok = (self._cur.get("token") or "").replace("\n", "⏎")[:14]
            try:
                self._token_item.setData(
                    pos=np.array([x, 14.0, 0.0], dtype=np.float32), text=tok)
            except Exception:
                pass

        # next-token distribution (revealed once the front hit the head)
        if self._disp_top_logits:
            lines = ["next ▸"]
            for s, p in self._disp_top_logits[:5]:
                disp = s.replace("\n", "⏎").replace("\t", "→")[:12]
                bar = "█" * max(0, int(round(p * 12)))
                lines.append(f"{disp:<12}{bar} {p * 100:4.1f}%")
            self._logits_hud.setText("\n".join(lines))
            self._logits_hud.adjustSize()
        elif self._logits_hud.text():
            self._logits_hud.setText("")

        step = self._cur.get("step", 0)
        prog = min(100, int(100 * f / max(1, n)))
        queued = len(self._pending)
        if self._mode == "moe":
            detail = f"{self._active} experts lit"
        else:
            detail = f"dense · peak‖h‖ {self._peak_norm:5.1f}"
        self._hud.setText(
            f"token {step} · layer {min(int(f), n)}/{n} ({prog}%) · {detail} · "
            f"{self._tps:4.1f} tok/s" + (f" · +{queued} queued" if queued else ""))
        self._hud.adjustSize()
