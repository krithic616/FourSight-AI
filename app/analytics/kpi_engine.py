"""KPI calculation helpers."""

from __future__ import annotations

import pandas as pd


def list_kpis() -> list[str]:
    """Return the currently supported Phase 1 KPI keys."""
    return [
        "total_rows",
        "total_columns",
        "duplicate_rows",
        "columns_with_missing_values",
    ]


def compute_phase_one_kpis(
    dataframe: pd.DataFrame,
    duplicate_rows: int,
    missing_value_summary: pd.DataFrame,
) -> dict[str, int]:
    """Compute the Phase 1 KPI overview values."""
    return {
        "total_rows": int(dataframe.shape[0]),
        "total_columns": int(dataframe.shape[1]),
        "duplicate_rows": int(duplicate_rows),
        "columns_with_missing_values": int(len(missing_value_summary)),
    }
