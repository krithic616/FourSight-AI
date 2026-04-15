"""Validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def is_supported_file(path: str | Path) -> bool:
    """Check whether a file extension is currently supported."""
    return Path(path).suffix.lower() in {".csv", ".xlsx", ".json"}


def is_csv_file(path: str | Path) -> bool:
    """Check whether a file path or file name points to a CSV file."""
    return Path(path).suffix.lower() == ".csv"


def ensure_uploaded_csv(uploaded_file: Any) -> None:
    """Validate that the uploaded object represents a CSV file."""
    if uploaded_file is None:
        raise ValueError("No file was uploaded.")

    file_name = getattr(uploaded_file, "name", "")
    if not file_name or not is_csv_file(file_name):
        raise ValueError("Please upload a CSV file.")

