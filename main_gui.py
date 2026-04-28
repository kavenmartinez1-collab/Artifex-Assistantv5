"""
Artifex Assistant V5 — Legacy GUI launcher redirect.
FreeSimpleGUI GUI was removed. This redirects to the PyQt6 GUI.
"""
import sys
import os

print("main_gui.py → Redirecting to main_gui_qt.py (FreeSimpleGUI removed)", file=sys.stderr)

os.execv(sys.executable, [sys.executable, os.path.join(os.path.dirname(__file__), "main_gui_qt.py")] + sys.argv[1:])
