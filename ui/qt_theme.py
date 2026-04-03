"""
Artifex Assistant V5 — PyQt6 theme system.
Ports the 5 GUI themes to Qt stylesheets (QSS).
"""

# Theme definitions — same colors as gui_theme.py
THEMES = {
    "CYBERPUNK": {
        "bg": "#0f111a",
        "text": "#c7d0ff",
        "input_bg": "#161a2b",
        "input_text": "#9efeff",
        "button_text": "#0f111a",
        "button_bg": "#00f0ff",
        "output_bg": "#0b0d16",
        "status_bg": "#161a2b",
        "accent": "#00f0ff",
        "accent2": "#b060ff",
        "warning": "#ffcc00",
        "error": "#ff4444",
        "success": "#44ff44",
        "border": "#1e2340",
        "hover": "#00c8d8",
        "selection_bg": "#1a3050",
        "scrollbar_bg": "#161a2b",
        "scrollbar_handle": "#00f0ff",
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
        "warning": "#ff6b00",
        "error": "#ff0040",
        "success": "#00ff9f",
        "border": "#2a0040",
        "hover": "#cc2258",
        "selection_bg": "#3a0050",
        "scrollbar_bg": "#1a0025",
        "scrollbar_handle": "#ff2a6d",
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
        "hover": "#0052a3",
        "selection_bg": "#cce0ff",
        "scrollbar_bg": "#e8e8e8",
        "scrollbar_handle": "#0066cc",
    },
}

# VRAM usage color thresholds
VRAM_COLOR_OK = "#44ff44"
VRAM_COLOR_WARN = "#ffcc00"
VRAM_COLOR_CRIT = "#ff4444"

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


def get_theme() -> dict:
    """Get the active theme's color dict."""
    return THEMES[_active_theme]


def vram_color(fraction: float) -> str:
    if fraction < 0.6:
        return VRAM_COLOR_OK
    elif fraction < 0.85:
        return VRAM_COLOR_WARN
    return VRAM_COLOR_CRIT


def generate_stylesheet(theme: dict = None) -> str:
    """Generate a Qt stylesheet (QSS) from a theme dict."""
    t = theme or THEMES[_active_theme]

    return f"""
    /* === Global === */
    QMainWindow, QDialog {{
        background-color: {t['bg']};
        color: {t['text']};
    }}
    QWidget {{
        background-color: {t['bg']};
        color: {t['text']};
        font-family: "Segoe UI", sans-serif;
        font-size: 10pt;
    }}

    /* === Labels === */
    QLabel {{
        background-color: transparent;
        color: {t['text']};
    }}
    QLabel[class="accent"] {{
        color: {t['accent']};
        font-weight: bold;
    }}
    QLabel[class="title"] {{
        color: {t['accent']};
        font-size: 14pt;
        font-weight: bold;
    }}
    QLabel[class="status"] {{
        color: {t['accent']};
        font-family: "Consolas", monospace;
        font-size: 8pt;
    }}

    /* === Input fields === */
    QLineEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {t['input_bg']};
        color: {t['input_text']};
        border: 1px solid {t['border']};
        border-radius: 4px;
        padding: 4px 6px;
        font-family: "Consolas", monospace;
        selection-background-color: {t['selection_bg']};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid {t['accent']};
    }}

    /* === Text editors === */
    QTextEdit, QPlainTextEdit {{
        background-color: {t['output_bg']};
        color: {t['text']};
        border: 1px solid {t['border']};
        border-radius: 4px;
        padding: 6px;
        font-family: "Consolas", monospace;
        font-size: 10pt;
        selection-background-color: {t['selection_bg']};
    }}
    QTextEdit:focus, QPlainTextEdit:focus {{
        border: 1px solid {t['accent']};
    }}

    /* === Buttons === */
    QPushButton {{
        background-color: {t['button_bg']};
        color: {t['button_text']};
        border: none;
        border-radius: 4px;
        padding: 6px 14px;
        font-weight: bold;
        font-size: 10pt;
    }}
    QPushButton:hover {{
        background-color: {t['hover']};
    }}
    QPushButton:pressed {{
        background-color: {t['accent2']};
    }}
    QPushButton:disabled {{
        background-color: {t['border']};
        color: {t['status_bg']};
    }}
    QPushButton[class="danger"] {{
        background-color: {t['error']};
    }}
    QPushButton[class="secondary"] {{
        background-color: {t['input_bg']};
        color: {t['accent']};
        border: 1px solid {t['border']};
    }}

    /* === ComboBox === */
    QComboBox {{
        background-color: {t['input_bg']};
        color: {t['input_text']};
        border: 1px solid {t['border']};
        border-radius: 4px;
        padding: 4px 8px;
        min-height: 24px;
    }}
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {t['input_bg']};
        color: {t['input_text']};
        border: 1px solid {t['border']};
        selection-background-color: {t['selection_bg']};
    }}

    /* === Tabs === */
    QTabWidget::pane {{
        border: 1px solid {t['border']};
        border-radius: 4px;
        background-color: {t['output_bg']};
    }}
    QTabBar::tab {{
        background-color: {t['input_bg']};
        color: {t['text']};
        padding: 6px 14px;
        border: 1px solid {t['border']};
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background-color: {t['output_bg']};
        color: {t['accent']};
        border-bottom: 2px solid {t['accent']};
    }}
    QTabBar::tab:hover {{
        background-color: {t['selection_bg']};
    }}

    /* === Splitter === */
    QSplitter::handle {{
        background-color: {t['border']};
        width: 2px;
        height: 2px;
    }}
    QSplitter::handle:hover {{
        background-color: {t['accent']};
    }}

    /* === Scrollbars === */
    QScrollBar:vertical {{
        background-color: {t['scrollbar_bg']};
        width: 10px;
        border: none;
    }}
    QScrollBar::handle:vertical {{
        background-color: {t['scrollbar_handle']};
        min-height: 30px;
        border-radius: 5px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar:horizontal {{
        background-color: {t['scrollbar_bg']};
        height: 10px;
        border: none;
    }}
    QScrollBar::handle:horizontal {{
        background-color: {t['scrollbar_handle']};
        min-width: 30px;
        border-radius: 5px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}

    /* === Progress bar === */
    QProgressBar {{
        background-color: {t['input_bg']};
        border: 1px solid {t['border']};
        border-radius: 4px;
        text-align: center;
        color: {t['text']};
        font-size: 8pt;
    }}
    QProgressBar::chunk {{
        background-color: {t['accent']};
        border-radius: 3px;
    }}

    /* === Slider === */
    QSlider::groove:horizontal {{
        background-color: {t['input_bg']};
        height: 6px;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background-color: {t['accent']};
        width: 14px;
        height: 14px;
        margin: -4px 0;
        border-radius: 7px;
    }}

    /* === List widget === */
    QListWidget {{
        background-color: {t['input_bg']};
        color: {t['text']};
        border: 1px solid {t['border']};
        border-radius: 4px;
    }}
    QListWidget::item:selected {{
        background-color: {t['selection_bg']};
        color: {t['accent']};
    }}

    /* === Group box === */
    QGroupBox {{
        border: 1px solid {t['border']};
        border-radius: 4px;
        margin-top: 8px;
        padding-top: 12px;
        font-weight: bold;
        color: {t['accent']};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
    }}

    /* === CheckBox === */
    QCheckBox {{
        color: {t['text']};
        spacing: 6px;
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {t['border']};
        border-radius: 3px;
        background-color: {t['input_bg']};
    }}
    QCheckBox::indicator:checked {{
        background-color: {t['accent']};
        border-color: {t['accent']};
    }}

    /* === Status bar === */
    QStatusBar {{
        background-color: {t['status_bg']};
        color: {t['accent']};
        font-family: "Consolas", monospace;
        font-size: 9pt;
    }}

    /* === Menu bar === */
    QMenuBar {{
        background-color: {t['bg']};
        color: {t['text']};
    }}
    QMenuBar::item:selected {{
        background-color: {t['selection_bg']};
    }}
    QMenu {{
        background-color: {t['input_bg']};
        color: {t['text']};
        border: 1px solid {t['border']};
    }}
    QMenu::item:selected {{
        background-color: {t['selection_bg']};
    }}

    /* === Tooltip === */
    QToolTip {{
        background-color: {t['input_bg']};
        color: {t['text']};
        border: 1px solid {t['accent']};
        padding: 4px;
        font-size: 9pt;
    }}
    """
