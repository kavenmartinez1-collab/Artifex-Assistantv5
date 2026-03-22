"""
Artifex Assistant V5 — Progress tracking utilities.
Provides progress feedback for long operations (model loading, downloads, indexing).
Works in both CLI (tqdm) and GUI (callback) modes.
"""

import sys
import time


class ProgressTracker:
    """Unified progress tracker for CLI and GUI.

    Usage:
        # CLI mode (uses tqdm if available)
        tracker = ProgressTracker(total=100, desc="Loading model")
        for i in range(100):
            tracker.update(1)
        tracker.close()

        # GUI mode (uses a callback)
        tracker = ProgressTracker(total=100, callback=update_status_bar)
        tracker.update(10, message="Loading weights...")
        tracker.close()
    """

    def __init__(self, total=100, desc="", callback=None, unit="it"):
        """Initialize the progress tracker.

        Args:
            total: Total number of steps
            desc: Description prefix
            callback: Optional callback(current, total, message) for GUI mode
            unit: Unit label (default "it")
        """
        self.total = total
        self.desc = desc
        self.callback = callback
        self.current = 0
        self._start_time = time.time()
        self._tqdm = None

        if callback is None:
            # CLI mode — try tqdm, fallback to simple print
            try:
                from tqdm import tqdm
                self._tqdm = tqdm(total=total, desc=desc, unit=unit, file=sys.stderr)
            except ImportError:
                pass

    def update(self, n=1, message=""):
        """Update progress by n steps.

        Args:
            n: Number of steps completed
            message: Optional status message
        """
        self.current = min(self.current + n, self.total)

        if self.callback:
            self.callback(self.current, self.total, message or self.desc)
        elif self._tqdm:
            self._tqdm.update(n)
            if message:
                self._tqdm.set_postfix_str(message)
        else:
            # Simple fallback
            pct = (self.current / self.total * 100) if self.total > 0 else 0
            msg = f"  {self.desc}: {pct:.0f}%"
            if message:
                msg += f" — {message}"
            sys.stderr.write(f"\r{msg}")
            sys.stderr.flush()

    def set_total(self, total):
        """Update the total (useful when total is discovered mid-operation)."""
        self.total = total
        if self._tqdm:
            self._tqdm.total = total
            self._tqdm.refresh()

    def close(self):
        """Finalize the progress bar."""
        if self._tqdm:
            self._tqdm.close()
        elif self.callback is None and self.total > 0:
            elapsed = time.time() - self._start_time
            sys.stderr.write(f"\r  {self.desc}: done ({elapsed:.1f}s)\n")
            sys.stderr.flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def download_with_progress(url, dest_path, callback=None):
    """Download a file with progress tracking.

    Args:
        url: URL to download
        dest_path: Destination file path
        callback: Optional progress callback(current, total, message)

    Returns:
        dest_path on success
    """
    import urllib.request
    import os

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    # Get file size
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
    except Exception:
        total_size = 0

    tracker = ProgressTracker(
        total=total_size or 100,
        desc=f"Downloading {os.path.basename(dest_path)}",
        callback=callback,
        unit="B",
    )

    def reporthook(block_num, block_size, total):
        if total > 0 and tracker.total != total:
            tracker.set_total(total)
        tracker.update(block_size)

    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    tracker.close()

    return dest_path
