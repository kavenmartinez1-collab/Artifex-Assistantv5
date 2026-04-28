"""
Artifex Assistant V5 — Legacy GUI launcher (DEPRECATED).
Use main_gui_qt.py instead.
"""
import warnings
import sys

warnings.warn(
    "main_gui.py is deprecated — use main_gui_qt.py for the current GUI",
    DeprecationWarning, stacklevel=1,
)
print("WARNING: main_gui.py is deprecated. Use main_gui_qt.py instead.", file=sys.stderr)

# Enable faulthandler BEFORE any heavy imports so a C-level segfault in
# torch / bitsandbytes / tcl / Qt produces a stack trace instead of just
# a cryptic Windows exit code (e.g. 0xC0000005). The dump goes to a
# dedicated file so it survives the GUI process dying without flushing
# stdout. Read it after a crash with: tail logs/faultdump.log
import os
import faulthandler
os.makedirs("logs", exist_ok=True)
_fault_log = open(os.path.join("logs", "faultdump.log"), "a", buffering=1)
faulthandler.enable(file=_fault_log, all_threads=True)

from core.logging_config import setup_logging
setup_logging()

from ui.cyber_gui import ArtifexGUI


def main():
    app = ArtifexGUI()
    app.run()


if __name__ == "__main__":
    main()
