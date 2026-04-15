"""General helper utilities."""

from __future__ import annotations

from collections.abc import Iterable


def identity(value: object) -> object:
    """Return the input value unchanged."""
    return value


def format_percentage(value: int, total: int, precision: int = 2) -> float:
    """Convert a count into a percentage rounded for display."""
    if total <= 0:
        return 0.0
    return round((value / total) * 100, precision)


def summarize_column_list(columns: Iterable[str]) -> str:
    """Convert a column collection into a readable inline string."""
    items = list(columns)
    if not items:
        return "None"
    return ", ".join(items)

