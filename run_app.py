"""Launcher for the FourSight AI Streamlit application."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Run the Streamlit app using the current Python environment."""
    app_path = Path(__file__).resolve().parent / "app" / "main.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)])


if __name__ == "__main__":
    raise SystemExit(main())

