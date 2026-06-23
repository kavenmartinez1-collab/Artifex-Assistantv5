"""3D forward-pass telemetry panel (pyqtgraph / OpenGL).

Draws TelemetryCapture frames (see core/forward_telemetry.py) live:

  * transformer layers as slices along the X axis,
  * expert / neuron nodes as a golden-angle disc per slice (wide in the
    middle, pinched at the ends — residual-stream-ish),
  * for MoE models, the per-token top-k experts glow, the rest stay dark,
  * sparse edges between the active experts of adjacent layers,
  * a continuously sweeping "forward pass" front.

Frames arrive via push_frame(frame) on the GUI thread. An internal ~30 FPS
timer animates the sweep and auto-orbit independently of token rate, so the
view stays alive between tokens and only lags a token or two — fine at decode
speeds. Dense (non-MoE) models fall back to a per-layer norm glow.

If pyqtgraph / PyOpenGL aren't importable, the panel degrades to a label and
never crashes the GUI.
"""
import math
import os
import time

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer

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


class TelemetryPanel(QWidget):
    """Live 3D visualization of the transformer forward pass."""

    def __init__(self, nodes_per_layer=128, parent=None):
        super().__init__(parent)
        self._M = nodes_per_layer
        self._n_layers = 0
        self._positions = None         # (total, 3) float32
        self._base_rgba = None         # (total, 4) float32 — last frame's colors
        self._base_size = None         # (total,)  float32
        self._edge_arr = None          # (2K, 3)  float32 line-segment endpoints
        self._last_frame = None
        self._sweep = 0.0
        self._active = 0
        self._mode = "idle"
        self._peak_norm = 0.0
        self._last_t = None
        self._tps = 0.0

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

        # HUD overlay (transparent, click-through)
        self._hud = QLabel(self._view)
        self._hud.setStyleSheet(
            "color:#7fd3ff; background:transparent; font:9pt 'Consolas';"
        )
        self._hud.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._hud.move(8, 6)
        self._hud.setText("forward-pass · idle — run a prompt on the transformers backend")
        self._hud.adjustSize()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(33)                            # ~30 FPS

    # ── geometry ──────────────────────────────────────────────────────
    def _build_geometry(self, n_layers):
        self._n_layers = n_layers
        M = self._M
        total = n_layers * M
        pos = np.zeros((total, 3), dtype=np.float32)
        span = 50.0
        denom = max(1, n_layers - 1)
        for li in range(n_layers):
            x = (li / denom - 0.5) * span
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
        self._base_rgba = self._idle_rgba(total)
        self._base_size = np.full(total, 2.0, dtype=np.float32)

    @staticmethod
    def _idle_rgba(total):
        rgba = np.zeros((total, 4), dtype=np.float32)
        rgba[:, 2] = 0.18      # dim blue
        rgba[:, 3] = 0.08
        return rgba

    # ── data in ───────────────────────────────────────────────────────
    def push_frame(self, frame):
        """Slot: receive one telemetry frame (GUI thread)."""
        if self._view is None:
            return
        n = int(frame.get("n_layers", 0) or 0)
        if n <= 0:
            return
        if n != self._n_layers or self._positions is None:
            self._build_geometry(n)

        M = self._M
        total = n * M
        rgba = self._idle_rgba(total)
        size = np.full(total, 2.0, dtype=np.float32)
        layers = frame.get("layers", [])
        is_moe = bool(frame.get("moe"))

        edge_pts = []
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
                        rgba[slot] = (1.0,
                                      0.45 + 0.55 * w,
                                      0.12 + 0.5 * w,
                                      min(1.0, 0.45 + 0.55 * w))
                        size[slot] = 6.0 + 9.0 * w
                        lit.append(slot)
                        active += 1
                if prev_lit and lit:
                    for a in prev_lit:
                        for b in lit:
                            edge_pts.append(self._positions[a])
                            edge_pts.append(self._positions[b])
                prev_lit = lit
            self._mode = "moe"
        else:
            norms = [float(layers[li].get("norm") or 0.0)
                     for li in range(min(n, len(layers)))]
            peak = max(norms) if norms else 0.0
            scale = peak or 1.0
            for li, nrm in enumerate(norms):
                inten = max(0.0, min(1.0, nrm / scale))
                sl = slice(li * M, (li + 1) * M)
                rgba[sl] = (0.10,
                            0.30 + 0.60 * inten,
                            0.55 + 0.45 * inten,
                            0.10 + 0.55 * inten)
                size[sl] = 2.0 + 5.0 * inten
            # a few sweep edges between consecutive bright layers
            step = max(1, M // 6)
            for li in range(min(n, len(norms)) - 1):
                if norms[li] / scale > 0.3 and norms[li + 1] / scale > 0.3:
                    for j in range(0, M, step):
                        edge_pts.append(self._positions[li * M + j])
                        edge_pts.append(self._positions[(li + 1) * M + j])
            self._mode = "dense"

        self._base_rgba = rgba
        self._base_size = size
        self._active = active
        self._peak_norm = peak
        self._last_frame = frame

        if edge_pts:
            arr = np.asarray(edge_pts, dtype=np.float32)
            if len(arr) > 4000:                # cap segment count
                arr = arr[:4000]
            self._edge_arr = arr
        else:
            self._edge_arr = None

        # smoothed tok/s from frame arrival cadence
        t = time.perf_counter()
        if self._last_t is not None:
            dt = t - self._last_t
            if dt > 0:
                inst = 1.0 / dt
                self._tps = (0.8 * self._tps + 0.2 * inst) if self._tps else inst
        self._last_t = t

    def reset(self):
        """Clear state at the start of a new generation."""
        self._last_frame = None
        self._active = 0
        self._mode = "idle"
        self._peak_norm = 0.0
        self._tps = 0.0
        self._last_t = None
        self._edge_arr = None
        if self._positions is not None:
            self._base_rgba = self._idle_rgba(len(self._positions))
            self._base_size = np.full(len(self._positions), 2.0, dtype=np.float32)

    # ── render loop ───────────────────────────────────────────────────
    def _animate(self):
        if self._view is None or self._positions is None or self._base_rgba is None:
            return
        if not self.isVisible():
            return          # don't burn CPU/GPU rendering a hidden tab

        # advance the forward-pass front
        self._sweep += 0.18
        if self._sweep >= self._n_layers:
            self._sweep = 0.0

        rgba = self._base_rgba.copy()
        M = self._M
        front = self._sweep
        for li in range(self._n_layers):
            d = abs(li - front)
            if d < 1.6:
                boost = (1.6 - d) / 1.6 * 0.55
                sl = slice(li * M, (li + 1) * M)
                rgba[sl, :3] = np.clip(rgba[sl, :3] + boost, 0.0, 1.0)
                rgba[sl, 3] = np.clip(rgba[sl, 3] + boost * 0.5, 0.0, 1.0)

        self._scatter.setData(pos=self._positions, color=rgba,
                              size=self._base_size, pxMode=True)
        if self._edge_arr is not None:
            self._edges.setData(pos=self._edge_arr, color=(0.30, 0.62, 1.0, 0.22))
        else:
            self._edges.setData(pos=np.zeros((0, 3), dtype=np.float32))

        self._view.orbit(0.22, 0)            # gentle auto-rotate

        if self._last_frame is None:
            self._hud.setText(
                "forward-pass · idle — run a prompt on the transformers backend")
        elif self._mode == "moe":
            self._hud.setText(
                f"step {self._last_frame.get('step', 0)} · {self._n_layers} layers · "
                f"{self._active} experts lit · {self._tps:4.1f} tok/s")
        else:
            self._hud.setText(
                f"step {self._last_frame.get('step', 0)} · {self._n_layers} layers · "
                f"dense · peak‖h‖ {self._peak_norm:5.1f} · {self._tps:4.1f} tok/s")
        self._hud.adjustSize()
