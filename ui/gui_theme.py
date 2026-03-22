"""
Artifex-Lite v2 — GUI theme constants and color schemes.
"""

import FreeSimpleGUI as sg

# ===== BASE THEME =====
BG_COLOR = "#0f111a"
TEXT_COLOR = "#c7d0ff"
INPUT_BG = "#161a2b"
INPUT_TEXT = "#9efeff"
BUTTON_COLOR = ("#0f111a", "#00f0ff")
OUTPUT_BG = "#0b0d16"
STATUS_BG = "#161a2b"

# ===== MODE COLOR SCHEMES =====
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


def apply_theme():
    """Apply the Artifex cyberpunk theme to FreeSimpleGUI."""
    sg.theme_background_color(BG_COLOR)
    sg.theme_text_color(TEXT_COLOR)
    sg.theme_input_background_color(INPUT_BG)
    sg.theme_input_text_color(INPUT_TEXT)
    sg.theme_button_color(BUTTON_COLOR)
    sg.theme_border_width(0)
