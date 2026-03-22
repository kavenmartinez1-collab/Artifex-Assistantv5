"""
Artifex Assistant V5 — GUI theme constants and color schemes.
Supports multiple themes with runtime switching.
"""

import FreeSimpleGUI as sg

# ===== THEME DEFINITIONS =====
THEMES = {
    "CYBERPUNK": {
        "bg": "#0f111a",
        "text": "#c7d0ff",
        "input_bg": "#161a2b",
        "input_text": "#9efeff",
        "button": ("#0f111a", "#00f0ff"),
        "output_bg": "#0b0d16",
        "status_bg": "#161a2b",
        "accent": "#00f0ff",
        "accent2": "#b060ff",
        "warning": "#ffcc00",
        "error": "#ff4444",
        "success": "#44ff44",
    },
    "DARK_BLUE": {
        "bg": "#0d1117",
        "text": "#c9d1d9",
        "input_bg": "#161b22",
        "input_text": "#58a6ff",
        "button": ("#0d1117", "#58a6ff"),
        "output_bg": "#010409",
        "status_bg": "#161b22",
        "accent": "#58a6ff",
        "accent2": "#bc8cff",
        "warning": "#d29922",
        "error": "#f85149",
        "success": "#3fb950",
    },
    "BLOOD_DRAGON": {
        "bg": "#120015",
        "text": "#e0c0ff",
        "input_bg": "#1a0025",
        "input_text": "#ff2a6d",
        "button": ("#120015", "#ff2a6d"),
        "output_bg": "#0a0010",
        "status_bg": "#1a0025",
        "accent": "#ff2a6d",
        "accent2": "#ff6b00",
        "warning": "#ff6b00",
        "error": "#ff0040",
        "success": "#00ff9f",
    },
    "FOREST": {
        "bg": "#0a1410",
        "text": "#b0d0c0",
        "input_bg": "#0f1f18",
        "input_text": "#50e090",
        "button": ("#0a1410", "#50e090"),
        "output_bg": "#060f0a",
        "status_bg": "#0f1f18",
        "accent": "#50e090",
        "accent2": "#a0d050",
        "warning": "#e0c040",
        "error": "#e04040",
        "success": "#40ff80",
    },
    "LIGHT": {
        "bg": "#f5f5f5",
        "text": "#1a1a2e",
        "input_bg": "#ffffff",
        "input_text": "#1a1a2e",
        "button": ("#ffffff", "#0066cc"),
        "output_bg": "#ffffff",
        "status_bg": "#e8e8e8",
        "accent": "#0066cc",
        "accent2": "#6633cc",
        "warning": "#cc8800",
        "error": "#cc0000",
        "success": "#008800",
    },
}

# Active theme name
_active_theme = "CYBERPUNK"

# Convenience accessors (point to active theme)
def _get(key):
    return THEMES[_active_theme][key]

BG_COLOR = property(lambda self: _get("bg"))
TEXT_COLOR = property(lambda self: _get("text"))
INPUT_BG = property(lambda self: _get("input_bg"))
INPUT_TEXT = property(lambda self: _get("input_text"))
OUTPUT_BG = property(lambda self: _get("output_bg"))
STATUS_BG = property(lambda self: _get("status_bg"))

# Static defaults for module-level access (initial theme)
BG_COLOR = THEMES["CYBERPUNK"]["bg"]
TEXT_COLOR = THEMES["CYBERPUNK"]["text"]
INPUT_BG = THEMES["CYBERPUNK"]["input_bg"]
INPUT_TEXT = THEMES["CYBERPUNK"]["input_text"]
BUTTON_COLOR = THEMES["CYBERPUNK"]["button"]
OUTPUT_BG = THEMES["CYBERPUNK"]["output_bg"]
STATUS_BG = THEMES["CYBERPUNK"]["status_bg"]

# ===== MODE COLOR SCHEMES (legacy compat) =====
MODE_COLORS = {
    "ASSISTANT": {"accent": "#00f0ff", "button": "#00f0ff", "status": "#00f0ff"},
    "BLOOD_DRAGON": {"accent": "#ff2a6d", "button": "#ff2a6d", "status": "#ff2a6d"},
    "RED_TEAM": {"accent": "#ff6600", "button": "#ff6600", "status": "#ff6600"},
}

# ===== FONTS =====
FONT_MAIN = ("Segoe UI", 10, "bold")
FONT_TITLE = ("Segoe UI", 14, "bold")
FONT_MONO = ("Consolas", 11)
FONT_MONO_SM = ("Consolas", 10)
FONT_SMALL = ("Segoe UI", 9, "bold")
FONT_TINY = ("Segoe UI", 8)
FONT_TINY_MONO = ("Consolas", 8)

# VRAM usage color thresholds
VRAM_COLOR_OK = "#44ff44"
VRAM_COLOR_WARN = "#ffcc00"
VRAM_COLOR_CRIT = "#ff4444"


def get_theme_names():
    """Get list of available theme names."""
    return list(THEMES.keys())


def get_active_theme():
    """Get the currently active theme name."""
    return _active_theme


def apply_theme(theme_name=None):
    """Apply a theme to FreeSimpleGUI.

    Args:
        theme_name: Theme name from THEMES. If None, uses active theme.
    """
    global _active_theme, BG_COLOR, TEXT_COLOR, INPUT_BG, INPUT_TEXT
    global BUTTON_COLOR, OUTPUT_BG, STATUS_BG

    if theme_name and theme_name in THEMES:
        _active_theme = theme_name

    t = THEMES[_active_theme]
    BG_COLOR = t["bg"]
    TEXT_COLOR = t["text"]
    INPUT_BG = t["input_bg"]
    INPUT_TEXT = t["input_text"]
    BUTTON_COLOR = t["button"]
    OUTPUT_BG = t["output_bg"]
    STATUS_BG = t["status_bg"]

    sg.theme_background_color(BG_COLOR)
    sg.theme_text_color(TEXT_COLOR)
    sg.theme_input_background_color(INPUT_BG)
    sg.theme_input_text_color(INPUT_TEXT)
    sg.theme_button_color(BUTTON_COLOR)
    sg.theme_border_width(0)
