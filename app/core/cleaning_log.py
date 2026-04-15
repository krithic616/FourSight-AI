"""Cleaning log helpers."""

from __future__ import annotations

import pandas as pd


def create_cleaning_log(
    duplicate_rows: int,
    missing_value_summary: pd.DataFrame,
    column_profiles: dict[str, list[str]],
) -> list[str]:
    """Generate readable Phase 1 cleaning observations."""
    log_messages = [
        "Loaded the uploaded CSV successfully for profiling.",
        f"Detected {duplicate_rows} duplicate row(s) without removing them.",
    ]

    if missing_value_summary.empty:
        log_messages.append("No missing values were detected across the dataset.")
    else:
        log_messages.append(
            f"Detected missing values in {len(missing_value_summary)} column(s)."
        )

    log_messages.append(
        "Profiled columns into numeric, categorical, datetime, and other groups."
    )

    datetime_columns = column_profiles.get("datetime", [])
    if datetime_columns:
        log_messages.append(
            f"Conservative datetime detection flagged {len(datetime_columns)} column(s)."
        )
    else:
        log_messages.append("No columns met the conservative datetime detection threshold.")

    log_messages.append("No permanent cleaning transformations were applied in Phase 1.")
    return log_messages
