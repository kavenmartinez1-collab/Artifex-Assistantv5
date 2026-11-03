"""
Artifex Assistant V5 — PyQt6 theme system.
Ports the GUI themes to Qt stylesheets (QSS).

Every colour the window uses comes from here, including the custom-painted
widgets in ui/qt_cyber.py (they read get_theme() at paint time), so switching
themes restyles the whole window. Keys a theme leaves out are filled by
_with_defaults(): "neon" themes get glow effects and the grid backdrop.
"""

# Theme definitions — same colors as gui_theme.py, plus the neon extras
THEMES = {
    "CYBERPUNK": {
        "bg": "#070812",
        "text": "#c8d3f5",
        "dim": "#6b7399",
        "input_bg": "#0d1024",
        "input_text": "#7ff9ff",
        "button_text": "#04050c",
        "button_bg": "#00f0ff",
        "output_bg": "#05060e",
        "status_bg": "#0a0c1c",
        "accent": "#00f0ff",
        "accent2": "#ff2a6d",
        "accent3": "#fcee0a",
        "warning": "#fcee0a",
        "error": "#ff3b5c",
        "success": "#3dffa0",
        "border": "#1c2348",
        "border_hi": "#2f3d80",
        "hover": "#7ffbff",
        "selection_bg": "#3a1033",
        "scrollbar_bg": "#070812",
        "scrollbar_handle": "#00f0ff",
        "font_ui": '"Bahnschrift", "Segoe UI"',
        "font_mono": '"Cascadia Mono", "Consolas"',
        "neon": True,
    },
    "NIGHT_CITY": {
        "bg": "#0a0908",
        "text": "#ece4c8",
        "dim": "#8a8366",
        "input_bg": "#15130b",
        "input_text": "#fcee0a",
        "button_text": "#0a0908",
        "button_bg": "#fcee0a",
        "output_bg": "#070605",
        "status_bg": "#12100a",
        "accent": "#fcee0a",
        "accent2": "#00f0ff",
        "accent3": "#ff003c",
        "warning": "#ff9f1c",
        "error": "#ff003c",
        "success": "#00ff9f",
        "border": "#2b2715",
        "border_hi": "#5a5222",
        "hover": "#fff59a",
        "selection_bg": "#3d3708",
        "scrollbar_bg": "#0a0908",
        "scrollbar_handle": "#fcee0a",
        "font_ui": '"Bahnschrift", "Segoe UI"',
        "font_mono": '"Cascadia Mono", "Consolas"',
        "neon": True,
    },
    "DARK_BLUE": {
        "bg": "#0d1117",
        "text": "#c9d1d9",
        "input_bg": "#161b22",
        "input_text": "#58a6ff",
        "button_text": "#0d1117",
        "button_bg": "#58a6ff",
        "output_bg": "#010409",
        "status_bg": "#161b22",
        "accent": "#58a6ff",
        "accent2": "#bc8cff",
        "warning": "#d29922",
        "error": "#f85149",
        "success": "#3fb950",
        "border": "#21262d",
        "hover": "#388bfd",
        "selection_bg": "#1f3050",
        "scrollbar_bg": "#161b22",
        "scrollbar_handle": "#58a6ff",
    },
    "BLOOD_DRAGON": {
        "bg": "#120015",
        "text": "#e0c0ff",
        "input_bg": "#1a0025",
        "input_text": "#ff2a6d",
        "button_text": "#120015",
        "button_bg": "#ff2a6d",
        "output_bg": "#0a0010",
        "status_bg": "#1a0025",
        "accent": "#ff2a6d",
        "accent2": "#ff6b00",
        "accent3": "#00e5ff",
        "warning": "#ff6b00",
        "error": "#ff0040",
        "success": "#00ff9f",
        "border": "#2a0040",
        "border_hi": "#5a1070",
        "hover": "#ff6f9a",
        "selection_bg": "#3a0050",
        "scrollbar_bg": "#1a0025",
        "scrollbar_handle": "#ff2a6d",
        "font_ui": '"Bahnschrift", "Segoe UI"',
        "font_mono": '"Cascadia Mono", "Consolas"',
        "neon": True,
    },
    "FOREST": {
        "bg": "#0a1410",
        "text": "#b0d0c0",
        "input_bg": "#0f1f18",
        "input_text": "#50e090",
        "button_text": "#0a1410",
        "button_bg": "#50e090",
        "output_bg": "#060f0a",
        "status_bg": "#0f1f18",
        "accent": "#50e090",
        "accent2": "#a0d050",
        "warning": "#e0c040",
        "error": "#e04040",
        "success": "#40ff80",
        "border": "#1a3020",
        "hover": "#40c070",
        "selection_bg": "#1a3020",
        "scrollbar_bg": "#0f1f18",
        "scrollbar_handle": "#50e090",
    },
    "LIGHT": {
        "bg": "#f5f5f5",
        "text": "#1a1a2e",
        "dim": "#6a6a7a",
        "input_bg": "#ffffff",
        "input_text": "#1a1a2e",
        "button_text": "#ffffff",
        "button_bg": "#0066cc",
        "output_bg": "#ffffff",
        "status_bg": "#e8e8e8",
        "accent": "#0066cc",
        "accent2": "#6633cc",
        "warning": "#cc8800",
        "error": "#cc0000",
        "success": "#008800",
        "border": "#d0d0d0",
        "border_hi": "#a8a8b8",
        "hover": "#0052a3",
        "selection_bg": "#cce0ff",
        "scrollbar_bg": "#e8e8e8",
        "scrollbar_handle": "#0066cc",
    },
}

_active_theme = "CYBERPUNK"


def get_theme_names() -> list[str]:
    return list(THEMES.keys())


def get_active_theme() -> str:
    return _active_theme


def set_active_theme(name: str) -> bool:
    global _active_theme
    if name in THEMES:
        _active_theme = name
        return True
    return False


def _with_defaults(t: dict) -> dict:
    """Fill the keys older themes don't define."""
    out = {
        "dim": _mix(t["text"], t["bg"], 0.55),
        "accent3": t["accent2"],
        "border_hi": _mix(t["border"], t["accent"], 0.35),
        "font_ui": '"Segoe UI"',
        "font_mono": '"Consolas"',
        "neon": False,
    }
    out.update(t)
    return out


def get_theme() -> dict:
    """Get the active theme's color dict (with defaults filled in)."""
    return _with_defaults(THEMES[_active_theme])


def vram_color(fraction: float) -> str:
    t = get_theme()
    if fraction < 0.6:
        return t["success"]
    elif fraction < 0.85:
        return t["warning"]
    return t["error"]


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _mix(a: str, b: str, k: float) -> str:
    """Blend hex colour a toward b by k (0 = a, 1 = b)."""
    ra, ga, ba = _rgb(a)
    rb, gb, bb = _rgb(b)
    return "#%02x%02x%02x" % (round(ra + (rb - ra) * k),
                              round(ga + (gb - ga) * k),
                              round(ba + (bb - ba) * k))


def rgba(hex_color: str, alpha: float) -> str:
    """QSS rgba() from a hex colour and a 0..1 alpha (QSS wants 0..255)."""
    r, g, b = _rgb(hex_color)
    return f"rgba({r}, {g}, {b}, {round(alpha * 255)})"


def _chevron(color: str) -> str:
    """Path to a tiny chevron SVG in `color` for QComboBox::down-arrow
    (QSS can only take arrow images from files). Cached in the temp dir."""
    import os
    import tempfile
    d = os.path.join(tempfile.gettempdir(), "artifex_qss")
    path = os.path.join(d, f"chevron_{color.lstrip('#')}.svg")
    if not os.path.isfile(path):
        os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
                    f'<path d="M1 3 L5 7 L9 3" fill="none" stroke="{color}" '
                    'stroke-width="1.6" stroke-linecap="square"/></svg>')
    return path.replace("\\", "/")


def generate_stylesheet(theme: dict = None) -> str:
    """Generate a Qt stylesheet (QSS) from a theme dict."""
    t = _with_defaults(theme or THEMES[_active_theme])
    a, a2 = t["accent"], t["accent2"]
    ui, mono = t["font_ui"], t["font_mono"]

    return f"""
    /* === Global === */
    QMainWindow, QDialog {{
        background-color: {t['bg']};
        color: {t['text']};
    }}
    QWidget {{
        background-color: {t['bg']};
        color: {t['text']};
        font-family: {ui}, sans-serif;
        font-size: 10pt;
        selection-background-color: {t['selection_bg']};
        selection-color: {t['text']};
    }}

    /* === Labels === */
    QLabel {{
        background-color: transparent;
        color: {t['text']};
    }}
    QLabel[class="accent"] {{
        color: {a};
        font-weight: bold;
    }}
    QLabel[class="title"] {{
        color: {a};
        font-size: 14pt;
        font-weight: bold;
    }}
    QLabel[class="hud"] {{
        color: {a};
        font-family: {ui};
        font-size: 9pt;
        font-weight: 600;
        padding: 2px 0 4px 0;
        border-bottom: 1px solid {rgba(a, 0.35)};
    }}
    QLabel[class="status"] {{
        color: {a};
        font-family: {mono}, monospace;
        font-size: 8pt;
    }}
    QLabel[class="dim"], QLabel[class="hint"] {{
        color: {t['dim']};
        font-size: 9pt;
    }}
    QLabel[class="hint"] {{
        font-family: {mono}, monospace;
        padding: 2px 4px;
    }}

    /* === Input fields === */
    QLineEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {t['input_bg']};
        color: {t['input_text']};
        border: 1px solid {t['border']};
        border-bottom: 1px solid {t['border_hi']};
        border-radius: 2px;
        padding: 4px 6px;
        font-family: {mono}, monospace;
    }}
    QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: {t['border_hi']};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid {a};
        background-color: {_mix(t['input_bg'], a, 0.04)};
    }}

    /* === Text editors === */
    QTextEdit, QPlainTextEdit, QTextBrowser {{
        background-color: {t['output_bg']};
        color: {t['text']};
        border: 1px solid {t['border']};
        border-radius: 2px;
        padding: 6px;
        font-family: {mono}, monospace;
        font-size: 10pt;
    }}
    QTextEdit:focus, QPlainTextEdit:focus {{
        border: 1px solid {a};
    }}
    QPlainTextEdit[class="log"] {{
        background-color: {_mix(t['output_bg'], '#000000', 0.4)};
        color: {_mix(t['text'], t['dim'], 0.4)};
        font-size: 9pt;
    }}
    QPlainTextEdit#thinkingOutput {{
        color: {_mix(t['text'], t['accent3'], 0.25)};
        border-left: 2px solid {rgba(t['accent3'], 0.35)};
        font-size: 9pt;
    }}

    /* === Buttons: outlined neon by default, filled for primary === */
    QPushButton {{
        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {rgba(a, 0.16)}, stop:1 {rgba(a, 0.04)});
        color: {a};
        border: 1px solid {rgba(a, 0.55)};
        border-radius: 2px;
        padding: 6px 14px;
        font-family: {ui};
        font-weight: 600;
        font-size: 10pt;
    }}
    QPushButton:hover {{
        background-color: {rgba(a, 0.26)};
        border-color: {a};
        color: {t['hover']};
    }}
    QPushButton:pressed {{
        background-color: {rgba(a2, 0.35)};
        border-color: {a2};
    }}
    QPushButton:disabled {{
        background-color: transparent;
        border-color: {t['border']};
        color: {t['dim']};
    }}
    QPushButton[class="primary"] {{
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {a}, stop:1 {_mix(a, t['hover'], 0.5)});
        color: {t['button_text']};
        border: 1px solid {t['hover']};
        font-weight: bold;
    }}
    QPushButton[class="primary"]:hover {{
        background-color: {t['hover']};
    }}
    QPushButton[class="primary"]:pressed {{
        background-color: {a2};
        border-color: {a2};
    }}
    QPushButton[class="primary"]:disabled {{
        background-color: {t['border']};
        color: {t['dim']};
        border-color: {t['border']};
    }}
    QPushButton[class="secondary"] {{
        background-color: {rgba(t['input_bg'], 0.85)};
        color: {_mix(a, t['text'], 0.25)};
        border: 1px solid {t['border_hi']};
    }}
    QPushButton[class="secondary"]:hover {{
        background-color: {rgba(a, 0.14)};
        border-color: {a};
        color: {t['hover']};
    }}
    QPushButton[class="danger"], QPushButton[recording="true"] {{
        background-color: {rgba(t['error'], 0.18)};
        color: {t['error']};
        border: 1px solid {t['error']};
    }}

    /* === ComboBox === */
    QComboBox {{
        background-color: {t['input_bg']};
        color: {t['input_text']};
        border: 1px solid {t['border']};
        border-bottom: 1px solid {t['border_hi']};
        border-radius: 2px;
        padding: 4px 8px;
        min-height: 22px;
    }}
    QComboBox:hover, QComboBox:focus {{
        border-color: {a};
    }}
    QComboBox::drop-down {{
        border: none;
        border-left: 1px solid {t['border']};
        width: 18px;
    }}
    QComboBox::down-arrow {{
        image: url({_chevron(a)});
        width: 9px;
        height: 9px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {t['input_bg']};
        color: {t['input_text']};
        border: 1px solid {a};
        outline: none;
        selection-background-color: {rgba(a, 0.22)};
        selection-color: {t['hover']};
    }}

    /* === Tabs === */
    QTabWidget::pane {{
        border: 1px solid {t['border']};
        border-top: 1px solid {rgba(a, 0.45)};
        border-radius: 0;
        background-color: {t['output_bg']};
        top: -1px;
    }}
    QTabBar::tab {{
        background-color: transparent;
        color: {t['dim']};
        padding: 7px 16px;
        border: none;
        border-bottom: 2px solid transparent;
        margin-right: 2px;
        font-family: {ui};
        font-weight: 600;
        font-size: 9pt;
    }}
    QTabBar::tab:selected {{
        color: {a};
        border-bottom: 2px solid {a};
        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 transparent, stop:1 {rgba(a, 0.16)});
    }}
    QTabBar::tab:hover:!selected {{
        color: {t['text']};
        background-color: {rgba(a, 0.07)};
        border-bottom: 2px solid {rgba(a2, 0.6)};
    }}

    /* === Splitter === */
    QSplitter::handle {{
        background-color: {t['border']};
        width: 1px;
        height: 1px;
    }}
    QSplitter::handle:hover {{
        background-color: {a};
    }}

    /* === Scrollbars === */
    QScrollBar:vertical {{
        background-color: transparent;
        width: 8px;
        margin: 2px 1px;
        border: none;
    }}
    QScrollBar::handle:vertical {{
        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {rgba(t['scrollbar_handle'], 0.75)}, stop:1 {rgba(a2, 0.6)});
        min-height: 30px;
        border-radius: 2px;
    }}
    QScrollBar::handle:vertical:hover {{
        background-color: {t['scrollbar_handle']};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar::add-page, QScrollBar::sub-page {{
        background: transparent;
    }}
    QScrollBar:horizontal {{
        background-color: transparent;
        height: 8px;
        margin: 1px 2px;
        border: none;
    }}
    QScrollBar::handle:horizontal {{
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {rgba(t['scrollbar_handle'], 0.75)}, stop:1 {rgba(a2, 0.6)});
        min-width: 30px;
        border-radius: 2px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}

    /* === Progress bar === */
    QProgressBar {{
        background-color: {t['input_bg']};
        border: 1px solid {t['border_hi']};
        border-radius: 0;
        text-align: center;
        color: {t['text']};
        font-family: {mono}, monospace;
        font-size: 8pt;
    }}
    QProgressBar::chunk {{
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {a}, stop:1 {a2});
    }}

    /* === Slider === */
    QSlider::groove:horizontal {{
        background-color: {t['input_bg']};
        border: 1px solid {t['border']};
        height: 4px;
    }}
    QSlider::sub-page:horizontal {{
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {rgba(a, 0.4)}, stop:1 {a});
    }}
    QSlider::handle:horizontal {{
        background-color: {t['bg']};
        border: 2px solid {a};
        width: 10px;
        height: 10px;
        margin: -6px 0;
        border-radius: 1px;
    }}
    QSlider::handle:horizontal:hover {{
        background-color: {a};
    }}

    /* === List / table / tree === */
    QListWidget, QTableWidget, QTableView, QTreeView, QTreeWidget {{
        background-color: {t['input_bg']};
        color: {t['text']};
        border: 1px solid {t['border']};
        border-radius: 2px;
        gridline-color: {t['border']};
        alternate-background-color: {_mix(t['input_bg'], t['bg'], 0.5)};
    }}
    QListWidget::item:selected, QTableView::item:selected, QTreeView::item:selected {{
        background-color: {rgba(a, 0.18)};
        color: {t['hover']};
    }}
    QHeaderView::section {{
        background-color: {t['status_bg']};
        color: {a};
        border: none;
        border-bottom: 1px solid {rgba(a, 0.45)};
        border-right: 1px solid {t['border']};
        padding: 4px 6px;
        font-family: {ui};
        font-weight: 600;
    }}

    /* === Group box === */
    QGroupBox {{
        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {rgba(a, 0.06)}, stop:0.35 transparent);
        border: 1px solid {t['border']};
        border-top: 1px solid {rgba(a, 0.5)};
        border-radius: 0;
        margin-top: 10px;
        padding-top: 12px;
        font-family: {ui};
        font-weight: 600;
        font-size: 9pt;
        color: {a};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 5px;
        color: {a};
        background-color: {t['bg']};
    }}

    /* === CheckBox === */
    QCheckBox {{
        color: {t['text']};
        spacing: 7px;
        background: transparent;
    }}
    QCheckBox::indicator {{
        width: 12px;
        height: 12px;
        border: 1px solid {t['border_hi']};
        border-radius: 1px;
        background-color: {t['input_bg']};
    }}
    QCheckBox::indicator:hover {{
        border-color: {a};
    }}
    QCheckBox::indicator:checked {{
        background-color: {a};
        border: 1px solid {t['hover']};
    }}

    /* === Status bar === */
    QStatusBar {{
        background-color: {t['status_bg']};
        color: {a};
        border-top: 1px solid {rgba(a, 0.35)};
        font-family: {mono}, monospace;
        font-size: 9pt;
    }}
    QStatusBar::item {{
        border: none;
    }}

    /* === Menu bar === */
    QMenuBar {{
        background-color: {t['bg']};
        color: {t['text']};
    }}
    QMenuBar::item:selected {{
        background-color: {rgba(a, 0.18)};
    }}
    QMenu {{
        background-color: {t['input_bg']};
        color: {t['text']};
        border: 1px solid {rgba(a, 0.6)};
    }}
    QMenu::item {{
        padding: 5px 18px;
    }}
    QMenu::item:selected {{
        background-color: {rgba(a, 0.18)};
        color: {t['hover']};
    }}

    /* === Tooltip === */
    QToolTip {{
        background-color: {t['input_bg']};
        color: {t['text']};
        border: 1px solid {a};
        padding: 5px;
        font-size: 9pt;
    }}

    /* === Chat view (ui/qt_widgets.py) === */
    QWidget#chatContainer {{
        background: transparent;
    }}
    ChatBubble {{
        background: transparent;
        border: none;
        margin: 3px 2px;
    }}
    ChatBubble[role="assistant"] {{
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {rgba(a, 0.07)}, stop:0.6 {rgba(t['output_bg'], 0.55)});
        border-left: 2px solid {a};
    }}
    ChatBubble[role="user"] {{
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {rgba(a2, 0.09)}, stop:0.6 {rgba(t['output_bg'], 0.55)});
        border-left: 2px solid {a2};
    }}
    QLabel[class="bubbleRole"] {{
        font-family: {ui};
        font-weight: bold;
        font-size: 8pt;
        color: {a};
    }}
    ChatBubble[role="user"] QLabel[class="bubbleRole"] {{
        color: {a2};
    }}
    QLabel[class="bubbleTime"] {{
        font-family: {mono}, monospace;
        font-size: 7pt;
        color: {t['dim']};
    }}
    QLabel[class="bubbleText"], ChatBubble QPlainTextEdit {{
        font-family: {mono}, monospace;
        font-size: 10pt;
        color: {t['text']};
    }}

    /* === Drop zone === */
    DropZone {{
        border: 1px dashed {t['border_hi']};
        border-radius: 2px;
        background-color: transparent;
    }}
    DropZone[dragOver="true"] {{
        border: 1px solid {a};
        background-color: {rgba(a, 0.08)};
    }}
    DropZone QLabel {{
        border: none;
        color: {t['dim']};
    }}
    DropZone QLabel[class="file"] {{
        color: {t['input_text']};
    }}
    """
