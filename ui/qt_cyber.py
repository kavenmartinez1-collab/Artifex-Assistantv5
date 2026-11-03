"""
Artifex Assistant V5 — custom-painted cyberpunk HUD pieces for the Qt GUI.

QSS can't do glows, chromatic offsets, segmented meters or a grid backdrop,
so these few widgets paint themselves. Every colour is read from
ui.qt_theme.get_theme() at paint time, so a theme switch only needs an
update(); the "neon" flag gates the louder effects (glitch, glow, grid) so
the plain themes stay plain.
"""

import math
import random
import sys

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QIcon, QLinearGradient, QPainter, QPainterPath,
    QPen, QPixmap, QPolygonF, QRadialGradient,
)
from PyQt6.QtWidgets import QGraphicsDropShadowEffect, QSizePolicy, QWidget

from ui.qt_theme import get_theme


def _c(hex_color: str, alpha: int = 255) -> QColor:
    c = QColor(hex_color)
    c.setAlpha(alpha)
    return c


def _family(css_list: str) -> str:
    """First family of a QSS font list ('"Bahnschrift", "Segoe UI"')."""
    return css_list.split(",")[0].strip().strip('"')


def ui_font(pt: float, weight=QFont.Weight.Normal) -> QFont:
    f = QFont(_family(get_theme()["font_ui"]))
    f.setPointSizeF(pt)
    f.setWeight(weight)
    return f


def mono_font(pt: float) -> QFont:
    f = QFont(_family(get_theme()["font_mono"]))
    f.setPointSizeF(pt)
    f.setStyleHint(QFont.StyleHint.Monospace)
    return f


# ── Glow ─────────────────────────────────────────────────────────────────

def add_glow(widget: QWidget, key: str = "accent", blur: float = 22,
             alpha: int = 170) -> None:
    """Neon halo around a widget (neon themes only). Re-call on theme change."""
    t = get_theme()
    if not t["neon"]:
        widget.setGraphicsEffect(None)
        return
    fx = QGraphicsDropShadowEffect(widget)
    fx.setBlurRadius(blur)
    fx.setOffset(0, 0)
    fx.setColor(_c(t[key], alpha))
    widget.setGraphicsEffect(fx)


# ── Backdrop ─────────────────────────────────────────────────────────────

def paint_backdrop(painter: QPainter, rect, grid: int = 28) -> None:
    """HUD backdrop for a scroll viewport: faint grid, two corner glows and
    corner brackets. Drawn over the QSS background, under the content."""
    t = get_theme()
    r = QRectF(rect)
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    if t["neon"]:
        for (cx, cy, key, a) in ((0.0, 0.0, "accent", 26),
                                 (1.0, 1.0, "accent2", 22)):
            g = QRadialGradient(QPointF(r.left() + r.width() * cx,
                                        r.top() + r.height() * cy),
                                max(r.width(), r.height()) * 0.7)
            g.setColorAt(0.0, _c(t[key], a))
            g.setColorAt(1.0, _c(t[key], 0))
            painter.fillRect(r, QBrush(g))
        painter.setPen(QPen(_c(t["accent"], 11), 1))
        x = r.left() + grid
        while x < r.right():
            painter.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            x += grid
        y = r.top() + grid
        while y < r.bottom():
            painter.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            y += grid
    paint_brackets(painter, r.adjusted(3, 3, -4, -4), _c(t["accent"], 150))
    painter.restore()


def paint_brackets(painter: QPainter, r: QRectF, color: QColor,
                   arm: float = 12) -> None:
    painter.setPen(QPen(color, 1.5))
    for (x, y, dx, dy) in ((r.left(), r.top(), 1, 1), (r.right(), r.top(), -1, 1),
                           (r.left(), r.bottom(), 1, -1), (r.right(), r.bottom(), -1, -1)):
        painter.drawLine(QPointF(x, y), QPointF(x + arm * dx, y))
        painter.drawLine(QPointF(x, y), QPointF(x, y + arm * dy))


# ── Title ────────────────────────────────────────────────────────────────

class NeonTitle(QWidget):
    """'ARTIFEX' wordmark with chromatic-aberration layers, a short glitch
    burst every few seconds (neon themes), a subtitle and a gradient rule."""

    def __init__(self, text="ARTIFEX", subtitle="V5 // LOCAL NEURAL CONSOLE",
                 parent=None):
        super().__init__(parent)
        self._text = text
        self._sub = subtitle
        self._glitch = 0          # frames left in the current burst
        self._slices = []
        self.setFixedHeight(62)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._schedule()

    def _schedule(self):
        self._timer.start(random.randint(4500, 9000))

    def _tick(self):
        if not get_theme()["neon"] or not self.isVisible():
            self._glitch = 0
            self._schedule()
            return
        if self._glitch == 0:
            self._glitch = random.randint(3, 6)
        self._glitch -= 1
        h = self.height()
        self._slices = [(random.uniform(4, 34), random.uniform(3, 9),
                         random.uniform(-7, 7)) for _ in range(random.randint(1, 3))
                        ] if self._glitch else []
        self.update()
        if self._glitch:
            self._timer.start(55)
        else:
            self._schedule()

    def paintEvent(self, e):
        t = get_theme()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        w = self.width()
        f = ui_font(21, QFont.Weight.Bold)
        f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 118)
        p.setFont(f)
        box = QRectF(0, 2, w, 36)
        align = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        if t["neon"]:
            jitter = 1.0 + (2.5 if self._glitch else 0)
            p.setPen(_c(t["accent2"], 200))
            p.drawText(box.translated(-jitter, 0), align, self._text)
            p.setPen(_c(t["accent"], 200))
            p.drawText(box.translated(jitter, 0), align, self._text)
            p.setPen(_c(t["hover"]))
        else:
            p.setPen(_c(t["accent"]))
        p.drawText(box, align, self._text)
        # glitch slices: copy a horizontal band of the wordmark sideways
        for (y, hgt, dx) in self._slices:
            band = QRectF(0, y, w, hgt)
            p.save()
            p.setClipRect(band)
            p.fillRect(band, _c(t["bg"]))
            p.setPen(_c(t["accent3"] if dx > 0 else t["accent"], 230))
            p.drawText(box.translated(dx, 0), align, self._text)
            p.restore()
        sub = mono_font(6.5)
        sub.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.6)
        p.setFont(sub)
        p.setPen(_c(t["dim"]))
        p.drawText(QRectF(0, 38, w, 12), align, self._sub)
        g = QLinearGradient(0, 0, w, 0)
        g.setColorAt(0.0, _c(t["accent"], 0))
        g.setColorAt(0.2, _c(t["accent"], 220))
        g.setColorAt(0.75, _c(t["accent2"], 220))
        g.setColorAt(1.0, _c(t["accent2"], 0))
        p.fillRect(QRectF(0, 55, w, 1.5), QBrush(g))
        p.fillRect(QRectF(w * 0.2 - 2, 53.5, 18, 4.5), _c(t["accent3"]))
        p.end()


# ── HUD meters / LED ─────────────────────────────────────────────────────

class HudMeter(QWidget):
    """'VRAM ▮▮▮▮▯▯▯▯▯▯ 3.1/8.0G' — segmented LED bar for the status bar."""

    SEGS = 10

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label = label
        self._frac = None
        self._value = "--"
        self.setFixedHeight(18)
        self.setMinimumWidth(170)

    def set_value(self, frac, text: str):
        self._frac = None if frac is None else max(0.0, min(1.0, frac))
        self._value = text
        self.update()

    def paintEvent(self, e):
        t = get_theme()
        p = QPainter(self)
        f = mono_font(7.5)
        p.setFont(f)
        fm = p.fontMetrics()
        h = self.height()
        x = 4
        p.setPen(_c(t["dim"]))
        p.drawText(QRectF(x, 0, 40, h), Qt.AlignmentFlag.AlignVCenter, self._label)
        x += fm.horizontalAdvance("VRAM") + 6
        frac = self._frac or 0.0
        lit = math.ceil(frac * self.SEGS) if self._frac else 0
        col = t["success"] if frac < 0.6 else t["warning"] if frac < 0.85 else t["error"]
        for i in range(self.SEGS):
            seg = QRectF(x + i * 6, 4, 4, h - 8)
            p.fillRect(seg, _c(col, 235) if i < lit else _c(t["border_hi"], 160))
        x += self.SEGS * 6 + 4
        p.setPen(_c(col if self._frac is not None else t["dim"]))
        p.drawText(QRectF(x, 0, self.width() - x, h), Qt.AlignmentFlag.AlignVCenter,
                   self._value)
        p.end()


class StatusLed(QWidget):
    """Status-bar LED: steady when idle, pulsing while is_busy() is true."""

    def __init__(self, is_busy, parent=None):
        super().__init__(parent)
        self._is_busy = is_busy
        self._phase = 0.0
        self._was_busy = False
        self.setFixedSize(16, 16)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(70)

    def _tick(self):
        busy = bool(self._is_busy())
        if busy:
            self._phase = (self._phase + 0.35) % (2 * math.pi)
        if busy or busy != self._was_busy:
            self._was_busy = busy
            self.update()

    def paintEvent(self, e):
        t = get_theme()
        busy = self._was_busy
        key = "accent3" if busy else "success"
        level = 0.55 + 0.45 * math.sin(self._phase) if busy else 1.0
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QPointF(8, 8)
        halo = QRadialGradient(c, 8)
        halo.setColorAt(0.0, _c(t[key], int(150 * level)))
        halo.setColorAt(1.0, _c(t[key], 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(halo))
        p.drawEllipse(c, 8, 8)
        p.setBrush(_c(t[key], int(120 + 135 * level)))
        p.drawEllipse(c, 3, 3)
        p.end()


# ── Window chrome / icon ─────────────────────────────────────────────────

def style_native_titlebar(window: QWidget) -> None:
    """Windows 11: dark caption in the theme bg with a neon window border.
    A no-op elsewhere and on older Windows (the DWM call just fails)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        t = get_theme()
        hwnd = int(window.winId())
        dwm = ctypes.windll.dwmapi

        def setattr_(attr, value):
            v = ctypes.c_int(value)
            dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v))

        def colorref(hex_color):
            r, g, b = QColor(hex_color).getRgb()[:3]
            return r | (g << 8) | (b << 16)

        light = QColor(t["bg"]).lightness() > 128
        setattr_(20, 0 if light else 1)                 # immersive dark mode
        setattr_(35, colorref(t["status_bg"]))          # caption colour
        setattr_(34, colorref(t["accent"]))             # border colour
        setattr_(36, colorref(t["accent"] if t["neon"] else t["text"]))  # title text
    except Exception:
        pass


def make_app_icon() -> QIcon:
    """Painted fallback icon: neon hexagon with an 'A' (no asset needed)."""
    t = get_theme()
    icon = QIcon()
    for size in (16, 32, 48, 64, 128, 256):
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = size / 64.0
        hexagon = QPolygonF([QPointF(32 * s + 29 * s * math.cos(math.radians(a)),
                                     32 * s + 29 * s * math.sin(math.radians(a)))
                             for a in range(-90, 270, 60)])
        path = QPainterPath()
        path.addPolygon(hexagon)
        path.closeSubpath()
        p.fillPath(path, _c(t["bg"] if t["neon"] else "#0d1024"))
        p.setPen(QPen(_c(t["accent2"]), max(1.0, 3 * s)))
        p.drawPath(path.translated(-1.2 * s, 0))
        p.setPen(QPen(_c(t["accent"]), max(1.0, 3 * s)))
        p.drawPath(path)
        f = QFont("Bahnschrift")
        f.setBold(True)
        f.setPixelSize(max(8, int(34 * s)))
        p.setFont(f)
        p.setPen(_c(t["accent2"]))
        p.drawText(QRectF(-1.5 * s, 1 * s, size, size), Qt.AlignmentFlag.AlignCenter, "A")
        p.setPen(_c(t["accent"]))
        p.drawText(QRectF(1.5 * s, 1 * s, size, size), Qt.AlignmentFlag.AlignCenter, "A")
        p.end()
        icon.addPixmap(pm)
    return icon
