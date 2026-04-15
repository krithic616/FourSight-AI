"""Preprocessing placeholders."""

from __future__ import annotations

import pandas as pd


def prepare_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe for future preprocessing steps."""
    return dataframe.copy()

