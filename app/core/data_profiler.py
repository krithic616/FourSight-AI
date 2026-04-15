"""Data profiling helpers."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


def profile_dataframe(dataframe: pd.DataFrame) -> dict[str, int]:
    """Return a minimal profile summary."""
    return {"rows": int(dataframe.shape[0]), "columns": int(dataframe.shape[1])}


def build_dataset_summary(dataframe: pd.DataFrame) -> dict[str, int]:
    """Return the top-level dataset dimensions."""
    return {
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
    }


def profile_column_types(dataframe: pd.DataFrame) -> dict[str, list[str]]:
    """Classify columns into numeric, categorical, datetime, and other buckets."""
    profiles = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "other": [],
    }

    for column_name in dataframe.columns:
        series = dataframe[column_name]

        if is_datetime64_any_dtype(series):
            profiles["datetime"].append(column_name)
            continue

        if is_numeric_dtype(series) and not is_bool_dtype(series):
            profiles["numeric"].append(column_name)
            continue

        if _looks_like_datetime(series):
            profiles["datetime"].append(column_name)
            continue

        if is_object_dtype(series) or is_string_dtype(series) or is_bool_dtype(series):
            profiles["categorical"].append(column_name)
            continue

        profiles["other"].append(column_name)

    return profiles


def _looks_like_datetime(series: pd.Series) -> bool:
    """Conservatively infer whether a text-like series contains datetimes."""
    if not (is_object_dtype(series) or is_string_dtype(series)):
        return False

    non_null = series.dropna()
    if non_null.empty:
        return False

    sample = non_null.astype(str).str.strip()
    sample = sample[sample.ne("")]
    if sample.empty or len(sample) < 3:
        return False

    sampled_values = sample.head(50)
    parsed = pd.to_datetime(sampled_values, errors="coerce", format="mixed")
    parsed_ratio = parsed.notna().mean()
    unique_ratio = sampled_values.nunique(dropna=True) / len(sampled_values)
    return bool(parsed_ratio >= 0.9 and unique_ratio >= 0.2)
