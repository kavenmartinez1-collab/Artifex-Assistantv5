"""
Artifex Assistant V5 — Legacy GUI entry point.
The FreeSimpleGUI GUI was removed; this starts the PyQt6 GUI, exactly as
main_gui_qt.py does.
"""
import os
import runpy

if __name__ == "__main__":
    # Run main_gui_qt.py in this process. The old os.execv() redirect did not
    # quote its arguments on Windows, so an install path containing a space
    # was split in two and Python could not find the file.
    runpy.run_path(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_gui_qt.py"),
        run_name="__main__",
    )
