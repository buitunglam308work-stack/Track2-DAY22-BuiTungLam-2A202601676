"""Small cross-platform console setup used by every standalone entrypoint."""

import sys


def configure_utf8_console() -> None:
    """Prefer UTF-8 output on Windows while remaining safe on older streams."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
