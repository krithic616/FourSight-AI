"""Phase 1 data quality helpers."""

from __future__ import annotations

import pandas as pd

try:
    from utils.helpers import format_percentage
except ModuleNotFoundError:
    from app.utils.helpers import format_percentage


def clean_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the input dataframe without modification."""
    return dataframe.copy()


def count_duplicate_rows(dataframe: pd.DataFrame) -> int:
    """Count duplicate rows without altering the source dataframe."""
    return int(dataframe.duplicated().sum())


def build_missing_value_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a missing value summary for columns with at least one null value."""
    total_rows = int(len(dataframe))
    missing_counts = dataframe.isna().sum()
    summary = pd.DataFrame(
        {
            "column_name": missing_counts.index,
            "missing_count": missing_counts.values.astype(int),
        }
    )
    summary["missing_percentage"] = summary["missing_count"].apply(
        lambda value: format_percentage(int(value), total_rows)
    )
    summary = summary.loc[summary["missing_count"] > 0]
    summary = summary.sort_values(
        by=["missing_percentage", "missing_count", "column_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return summary


def build_preprocessing_summary(
    dataframe: pd.DataFrame,
    duplicate_rows: int,
    missing_value_summary: pd.DataFrame,
) -> dict[str, object]:
    """Summarize Phase 1 preprocessing observations without mutating data."""
    return {
        "input_rows": int(dataframe.shape[0]),
        "input_columns": int(dataframe.shape[1]),
        "duplicate_rows_detected": duplicate_rows,
        "columns_with_missing_values": int(len(missing_value_summary)),
        "profiling_actions": [
            "CSV loaded into pandas for in-memory profiling.",
            "Duplicate rows counted without removing records.",
            "Missing values measured by column.",
            "Datetime detection attempted conservatively on text-like columns.",
            "No permanent cleaning transformations were applied to the uploaded dataset.",
        ],
    }
