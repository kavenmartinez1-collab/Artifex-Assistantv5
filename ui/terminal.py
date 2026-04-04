"""
Artifex-Assistant v5 — Terminal rendering effects.
Boot sequence, glitch text, and BLOOD_DRAGON visual styling.
"""

import sys
import time
import random

from colorama import init, Fore, Style, Back

init(autoreset=True)


def boot_sequence(manifest):
    """Blood Dragon style boot diagnostic."""
    print(f"{Fore.MAGENTA}{'=' * 75}")
    print(f"{Fore.RED}  [BLACKWALL BREACH] {Fore.YELLOW}| {Fore.CYAN}GATEWAY: {manifest['OS']}")
    print(f"{Fore.MAGENTA}  HARDWARE_ID: {Fore.WHITE}{manifest['GPU']}")
    print(f"{Fore.MAGENTA}  VRAM_LOAD:   {Fore.WHITE}{manifest['VRAM']}")
    print(f"{Fore.MAGENTA}  NEURAL_CORE: {Fore.WHITE}{manifest['CPU']}")
    print(f"{Fore.MAGENTA}{'=' * 75}\n")

    for _ in range(3):
        sys.stdout.write(
            f"\r{Fore.RED}[SCANNED_ICE: {random.randint(100, 999)}ms] "
            + Fore.YELLOW
            + "." * random.randint(1, 5)
        )
        sys.stdout.flush()
        time.sleep(0.4)
    print("\n")


def glitch_print(text, color=Fore.CYAN, delay=0.005):
    """Flickering text output with Blood Dragon palette."""
    for char in text:
        char_color = (
            random.choice([Fore.RED, Fore.MAGENTA, Fore.WHITE])
            if random.random() < 0.02
            else color
        )
        sys.stdout.write(char_color + char)
        sys.stdout.flush()
        time.sleep(delay)


