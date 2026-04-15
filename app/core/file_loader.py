"""File loading helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from utils.validators import ensure_uploaded_csv
except ModuleNotFoundError:
    from app.utils.validators import ensure_uploaded_csv


def load_file(path: str | Path) -> Path:
    """Return the normalized input path for future loading workflows."""
    return Path(path)


def load_csv_file(uploaded_file: Any) -> pd.DataFrame:
    """Safely load an uploaded CSV file into a dataframe."""
    ensure_uploaded_csv(uploaded_file)
    file_bytes = uploaded_file.getvalue()
    return pd.read_csv(BytesIO(file_bytes))


def get_uploaded_file_metadata(uploaded_file: Any) -> dict[str, object]:
    """Return lightweight metadata for an uploaded file."""
    ensure_uploaded_csv(uploaded_file)
    file_name = getattr(uploaded_file, "name", "uploaded.csv")
    file_size = len(uploaded_file.getvalue())
    return {
        "file_name": file_name,
        "file_size_bytes": file_size,
        "file_size_display": _format_file_size(file_size),
    }


def _format_file_size(file_size_bytes: int) -> str:
    """Format file size for lightweight UI display."""
    if file_size_bytes < 1024:
        return f"{file_size_bytes} B"
    if file_size_bytes < 1024 * 1024:
        return f"{file_size_bytes / 1024:.1f} KB"
    return f"{file_size_bytes / (1024 * 1024):.1f} MB"
