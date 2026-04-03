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


def main():
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QIcon

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
