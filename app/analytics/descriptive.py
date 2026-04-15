"""Descriptive analytics placeholders."""

from __future__ import annotations

import pandas as pd


def summarize_dataframe(dataframe: pd.DataFrame) -> dict[str, int]:
    """Return a minimal descriptive summary."""
    return {"rows": int(dataframe.shape[0]), "columns": int(dataframe.shape[1])}

