"""
Artifex Assistant V5 — PyQt6 GUI entry point.
Run: python main_gui_qt.py
"""

# Enable faulthandler BEFORE any heavy imports so a C-level segfault in
# torch / bitsandbytes / Qt produces a stack trace instead of just a
# cryptic Windows exit code (e.g. 0xC0000005). Dump file survives the
# process dying without flushing stdout. Read after a crash with:
# tail logs/faultdump.log
import os
import faulthandler
os.makedirs("logs", exist_ok=True)
_fault_log = open(os.path.join("logs", "faultdump.log"), "a", buffering=1)
faulthandler.enable(file=_fault_log, all_threads=True)

from ui.qt_launcher import main

if __name__ == "__main__":
    main()
