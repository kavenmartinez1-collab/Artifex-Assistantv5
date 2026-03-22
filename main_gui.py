"""
Artifex Assistant V5 — GUI launcher.
Usage:
    python main_gui.py
"""

from core.logging_config import setup_logging
setup_logging()

from ui.cyber_gui import ArtifexGUI


def main():
    app = ArtifexGUI()
    app.run()


if __name__ == "__main__":
    main()
