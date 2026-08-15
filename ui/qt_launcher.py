"""
Artifex Assistant V5 — PyQt6 GUI launcher.
Entry point for the Qt-based GUI application.
"""

import sys
import os

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def install_crash_logging():
    """Log unhandled exceptions before PyQt6 aborts the process.

    PyQt6 escalates any exception that escapes a slot or a reimplemented
    virtual (QThread.run, paintEvent, ...) to qFatal().  On Windows that
    calls __fastfail(FAST_FAIL_FATAL_APP_EXIT), killing the process via
    interrupt 0x29 without raising a catchable signal — so the faulthandler
    armed in main_gui_qt.py never sees it, logs/faultdump.log stays empty,
    and the crash surfaces as a bare exit code 3221226505 (0xC0000409) with
    no traceback anywhere.

    sys.excepthook still runs before the abort, so this is the last point
    at which the traceback can be captured.  Handlers are flushed
    explicitly because the abort gives Python no chance to unwind.

    Scope note: this covers the *Python-exception* route to that exit code
    only.  Qt also calls qFatal on its own for allocation failure, which
    raises no Python exception and so never reaches this hook — that
    variant is diagnosed from the System event log (Event 26, "Virtual
    Memory Minimum Too Low") rather than from anything logged here.
    """
    import logging
    import threading
    import traceback

    from core.logging_config import get_logger
    log = get_logger(__name__)

    def _log_exc(exc_type, exc_value, exc_tb, source):
        try:
            text = "".join(
                traceback.format_exception(exc_type, exc_value, exc_tb)
            )
            log.critical(
                "UNHANDLED EXCEPTION in %s — PyQt6 will abort the process "
                "(exit 3221226505) immediately after this line:\n%s",
                source, text,
            )
            for handler in logging.getLogger().handlers:
                try:
                    handler.flush()
                except Exception:
                    pass
        except Exception:
            # Never let the crash handler raise; the real traceback below
            # is worth more than anything this could report.
            pass

    def _excepthook(exc_type, exc_value, exc_tb):
        _log_exc(exc_type, exc_value, exc_tb, "main thread / Qt slot")
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    def _thread_excepthook(args):
        if args.exc_type is SystemExit:
            return
        _log_exc(args.exc_type, args.exc_value, args.exc_traceback,
                 "thread %r" % (getattr(args.thread, "name", "?"),))

    threading.excepthook = _thread_excepthook


def main():
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QIcon

    install_crash_logging()

    app = QApplication(sys.argv)
    app.setApplicationName("Artifex Assistant V5")
    app.setOrganizationName("Artifex")

    # Set app icon if available
    icon_path = os.path.join(PROJECT_ROOT, "assets", "icon.png")
    if os.path.isfile(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    from ui.qt_gui import ArtifexMainWindow

    window = ArtifexMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
