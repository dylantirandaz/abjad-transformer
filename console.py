"""Console setup shared by the command-line entry points."""

import io
import sys


def configure_console() -> None:
    """Print any character without crashing.

    Windows consoles default to a legacy code page that cannot encode every character
    a dataset or a half-trained model can produce.
    """
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
