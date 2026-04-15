"""Cautious time-series forecasting helpers."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


MIN_PARSE_SUCCESS_RATE = 0.8
MIN_FORECAST_PERIODS = 6
MAX_FORECAST_HORIZON = 12

ID_KEYWORDS = {
    "id",
    "key",
    "code",
    "number",
    "num",
    "customer_id",
    "order_id",
    "invoice_id",
    "transaction_id",
    "account_id",
    "user_id",
}
BUSINESS_METRIC_KEYWORDS = {
    "quantity",
    "qty",
    "price",
    "amount",
    "revenue",
    "sales",
    "cost",
    "profit",
    "margin",
    "discount",
    "score",
    "rating",
    "value",
    "units",
    "total",
    "demand",
    "volume",
}
FORECAST_METRIC_PRIORITY = [
    "net_revenue",
    "profit",
    "quantity",
    "cost",
    "discount_pct",
    "rating",
    "unit_price",
]
GRAIN_MAP = {
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "M",
}
GRAIN_LABELS = {
    "D": "Daily",
    "W": "Weekly",
    "M": "Monthly",
}
FORECAST_METHOD_LABEL = "Trend-based lightweight projection"


def predictive_readiness(dataframe: pd.DataFrame) -> bool:
    """Legacy compatibility helper for non-empty forecasting input."""
    return not dataframe.empty


def get_predictive_options(
    dataframe: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    readiness: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return forecasting selector options and tab availability."""
    metric_columns = _get_forecast_metric_columns(dataframe, column_profiles.get("numeric", []))
    datetime_columns = _get_forecast_datetime_columns(dataframe, column_profiles.get("datetime", []))
    readiness_status = (readiness or {}).get("status", "Not Ready")
    readiness_reason = (readiness or {}).get("reason", "")
    time_series_support = bool(metric_columns) and bool(datetime_columns)
    enabled = time_series_support and readiness_status in {"Ready", "Conditional"}

    return {
        "enabled": enabled,
        "time_series_support": time_series_support,
        "metric_columns": metric_columns,
        "default_metric_column": metric_columns[0] if metric_columns else None,
        "datetime_columns": datetime_columns,
        "default_datetime_column": datetime_columns[0] if len(datetime_columns) == 1 else None,
        "grain_options": ["Auto", "Daily", "Weekly", "Monthly"],
        "horizon_options": [3, 6, 9, 12],
        "readiness_status": readiness_status,
        "readiness_reason": readiness_reason,
    }


def run_predictive_forecast(
    dataframe: pd.DataFrame,
    metric_column: str,
    datetime_column: str,
    aggregation_grain: str,
    forecast_horizon: int,
    readiness_status: str,
    readiness_reason: str,
) -> dict[str, Any]:
    """Run a cautious deterministic forecast when time-series validation passes."""
    working_frame = dataframe[[metric_column, datetime_column]].copy()
    working_frame[metric_column] = pd.to_numeric(working_frame[metric_column], errors="coerce")
    parsed_datetime = pd.to_datetime(working_frame[datetime_column], errors="coerce", format="mixed")

    parse_success_rate = float(parsed_datetime.notna().mean()) if len(parsed_datetime) else 0.0
    metric_valid_ratio = float(working_frame[metric_column].notna().mean()) if len(working_frame) else 0.0
    grain_code = _resolve_grain(aggregation_grain, parsed_datetime)
    grain_label = _format_grain_label(grain_code)

    usable_frame = pd.DataFrame(
        {
            "timestamp": parsed_datetime,
            "metric_value": working_frame[metric_column],
        }
    ).dropna(subset=["timestamp", "metric_value"])

    if usable_frame.empty:
        return _build_unavailable_forecast(
            readiness_status=readiness_status,
            readiness_reason=readiness_reason,
            parse_success_rate=parse_success_rate,
            metric_valid_ratio=metric_valid_ratio,
            aggregation_grain=grain_code,
            forecast_horizon=forecast_horizon,
            validation_messages=[
                "No rows contain both a valid datetime value and a numeric forecast metric."
            ],
        )

    usable_frame["period"] = usable_frame["timestamp"].dt.to_period(grain_code).dt.to_timestamp()
    historical = (
        usable_frame.groupby("period")["metric_value"]
        .sum()
        .reset_index()
        .sort_values(by="period")
        .reset_index(drop=True)
    )

    unique_periods = int(historical["period"].nunique())
    validation_messages = _validate_forecast_inputs(
        parse_success_rate=parse_success_rate,
        metric_valid_ratio=metric_valid_ratio,
        unique_periods=unique_periods,
        observation_count=int(len(historical)),
        forecast_horizon=forecast_horizon,
    )
    readiness_reasons = _build_readiness_reasons(
        readiness_status=readiness_status,
        readiness_reason=readiness_reason,
        parse_success_rate=parse_success_rate,
        unique_periods=unique_periods,
        grain_label=grain_label,
        validation_messages=validation_messages,
    )

    if validation_messages:
        return _build_unavailable_forecast(
            readiness_status=readiness_status,
            readiness_reason=readiness_reason,
            parse_success_rate=parse_success_rate,
            metric_valid_ratio=metric_valid_ratio,
            aggregation_grain=grain_code,
            forecast_horizon=forecast_horizon,
            validation_messages=validation_messages,
            readiness_reasons=readiness_reasons,
            historical=historical,
        )

    forecast_table = _forecast_next_periods(historical, grain_code, forecast_horizon)
    combined = pd.concat(
        [
            historical.assign(series_type="Actual"),
            forecast_table.assign(series_type="Forecast"),
        ],
        ignore_index=True,
    )

    summary_cards = _build_forecast_summary_cards(historical, forecast_table)
    findings = _build_predictive_findings(
        historical=historical,
        forecast_table=forecast_table,
        readiness_status=readiness_status,
        parse_success_rate=parse_success_rate,
    )

    return {
        "status": "Ready" if readiness_status == "Ready" else "Conditional",
        "status_reason": _build_status_reason(
            readiness_status=readiness_status,
            grain_label=grain_label,
            unique_periods=unique_periods,
        ),
        "readiness_reasons": readiness_reasons,
        "input_summary": {
            "metric_column": metric_column,
            "datetime_column": datetime_column,
            "aggregation_grain": grain_code,
            "aggregation_grain_label": grain_label,
            "forecast_horizon": forecast_horizon,
            "datetime_parse_success_rate": parse_success_rate,
            "metric_valid_ratio": metric_valid_ratio,
            "historical_periods": unique_periods,
            "non_null_aggregated_observations": int(len(historical)),
            "forecast_method_label": FORECAST_METHOD_LABEL,
        },
        "historical_trend": historical,
        "forecast_table": _format_forecast_table(forecast_table),
        "forecast_chart": combined,
        "summary_cards": summary_cards,
        "findings": findings,
        "validation_messages": [],
    }


def _get_forecast_metric_columns(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
) -> list[str]:
    """Return numeric columns suitable for time aggregation."""
    metric_columns: list[str] = []
    row_count = max(len(dataframe), 1)
    for column_name in numeric_columns:
        normalized_name = _normalize_name(column_name)
        series = pd.to_numeric(dataframe[column_name], errors="coerce")
        valid_ratio = float(series.notna().mean()) if len(series) else 0.0
        unique_ratio = series.nunique(dropna=True) / max(series.notna().sum(), 1)
        if valid_ratio < 0.6 or series.nunique(dropna=True) < 2:
            continue
        if _looks_like_identifier(normalized_name, unique_ratio, row_count):
            continue
        if _contains_keyword(normalized_name, BUSINESS_METRIC_KEYWORDS) or unique_ratio < 0.95:
            metric_columns.append(column_name)
    return sorted(metric_columns, key=_metric_priority_key)


def _get_forecast_datetime_columns(
    dataframe: pd.DataFrame,
    datetime_columns: list[str],
) -> list[str]:
    """Return datetime columns with enough support for cautious forecasting."""
    usable_columns: list[str] = []
    for column_name in datetime_columns:
        parsed = pd.to_datetime(dataframe[column_name], errors="coerce", format="mixed")
        parse_success_rate = float(parsed.notna().mean()) if len(parsed) else 0.0
        if parse_success_rate < MIN_PARSE_SUCCESS_RATE:
            continue
        if parsed.dropna().nunique() < MIN_FORECAST_PERIODS:
            continue
        usable_columns.append(column_name)
    return usable_columns


def _resolve_grain(aggregation_grain: str, parsed_datetime: pd.Series) -> str:
    """Resolve the selected grain or infer one conservatively."""
    if aggregation_grain in GRAIN_MAP:
        return GRAIN_MAP[aggregation_grain]

    valid_dates = parsed_datetime.dropna().sort_values()
    if valid_dates.empty:
        return "D"
    time_depth_days = int((valid_dates.max() - valid_dates.min()).days)
    if time_depth_days >= 365:
        return "M"
    if time_depth_days >= 90:
        return "W"
    return "D"


def _validate_forecast_inputs(
    parse_success_rate: float,
    metric_valid_ratio: float,
    unique_periods: int,
    observation_count: int,
    forecast_horizon: int,
) -> list[str]:
    """Return validation messages that block forecasting."""
    messages: list[str] = []
    if parse_success_rate < MIN_PARSE_SUCCESS_RATE:
        messages.append(
            f"Datetime parse success is {parse_success_rate:.0%}, which is below the {MIN_PARSE_SUCCESS_RATE:.0%} threshold for trustworthy forecasting."
        )
    if metric_valid_ratio < 0.6:
        messages.append(
            f"The selected metric only has {metric_valid_ratio:.0%} usable numeric values, so aggregation is not reliable enough."
        )
    if unique_periods < MIN_FORECAST_PERIODS:
        messages.append(
            f"Only {unique_periods} unique time periods are available after aggregation, which is below the minimum of {MIN_FORECAST_PERIODS}."
        )
    if observation_count < MIN_FORECAST_PERIODS:
        messages.append(
            f"Only {observation_count} aggregated observations are available, so the series is too short for a cautious forecast."
        )
    if forecast_horizon > max(3, observation_count // 2):
        messages.append(
            "The requested forecast horizon is too long relative to the available history, so the projection would not be trustworthy."
        )
    return messages


def _forecast_next_periods(
    historical: pd.DataFrame,
    grain_code: str,
    forecast_horizon: int,
) -> pd.DataFrame:
    """Project the next periods using a cautious blend of trend and moving average."""
    history_values = historical["metric_value"].astype(float).to_numpy()
    recent_window = max(3, min(5, len(history_values)))
    moving_average = float(np.mean(history_values[-recent_window:]))
    slope = _estimate_recent_slope(history_values[-recent_window:])
    future_periods = _build_future_periods(historical["period"], grain_code, forecast_horizon)

    forecasts: list[float] = []
    last_actual = float(history_values[-1])
    for step in range(1, forecast_horizon + 1):
        trend_projection = last_actual + (slope * step)
        blended_value = (moving_average * 0.65) + (trend_projection * 0.35)
        if history_values.min() >= 0:
            blended_value = max(blended_value, 0.0)
        forecasts.append(round(float(blended_value), 2))

    return pd.DataFrame({"period": future_periods, "metric_value": forecasts})


def _estimate_recent_slope(values: np.ndarray) -> float:
    """Estimate a simple recent slope for short-term projection."""
    if len(values) < 2:
        return 0.0
    x_values = np.arange(len(values))
    slope, _ = np.polyfit(x_values, values, 1)
    return float(slope)


def _build_future_periods(
    historical_periods: pd.Series,
    grain_code: str,
    forecast_horizon: int,
) -> list[pd.Timestamp]:
    """Build the future period index for the selected grain."""
    last_period = pd.Timestamp(historical_periods.iloc[-1])
    if grain_code == "D":
        offset = pd.offsets.Day()
    elif grain_code == "W":
        offset = pd.offsets.Week()
    else:
        offset = pd.offsets.MonthBegin()
    future_periods: list[pd.Timestamp] = []
    current_period = last_period
    for _ in range(forecast_horizon):
        current_period = current_period + offset
        future_periods.append(current_period)
    return future_periods


def _build_forecast_summary_cards(
    historical: pd.DataFrame,
    forecast_table: pd.DataFrame,
) -> dict[str, Any]:
    """Build summary card values for the predictive UI."""
    latest_actual = float(historical["metric_value"].iloc[-1])
    first_forecast = float(forecast_table["metric_value"].iloc[0])
    projected_average = float(forecast_table["metric_value"].mean())
    direction = _classify_direction(
        reference_value=latest_actual,
        comparison_value=first_forecast,
    )

    return {
        "historical_periods_used": int(len(historical)),
        "forecast_horizon": int(len(forecast_table)),
        "latest_actual_value": latest_actual,
        "first_forecast_value": first_forecast,
        "projected_average_forecast_value": projected_average,
        "projected_direction": direction,
    }


def _build_predictive_findings(
    historical: pd.DataFrame,
    forecast_table: pd.DataFrame,
    readiness_status: str,
    parse_success_rate: float,
) -> list[str]:
    """Create deterministic findings for the cautious forecast."""
    findings: list[str] = []
    latest_actual = float(historical["metric_value"].iloc[-1])
    recent_window = min(4, len(historical))
    recent_average = float(historical["metric_value"].tail(recent_window).mean())
    first_forecast = float(forecast_table["metric_value"].iloc[0])
    projected_average = float(forecast_table["metric_value"].mean())
    direction = _classify_direction(
        reference_value=latest_actual,
        comparison_value=first_forecast,
    ).lower()

    findings.append(
        f"Short-term direction is {direction}: the first forecasted period is {first_forecast:.2f}, the latest actual value is {latest_actual:.2f}, and the recent actual average is {recent_average:.2f}."
    )
    findings.append(
        f"The projected average across the forecast horizon is {projected_average:.2f}, compared with the latest actual value of {latest_actual:.2f}."
    )

    volatility_label, volatility_reason = _classify_volatility(historical["metric_value"])
    findings.append(
        f"Recent series behavior appears {volatility_label} because {volatility_reason}."
    )

    if readiness_status == "Conditional":
        findings.append(
            "Forecasting is being shown cautiously because predictive readiness is Conditional, so the projection should be treated as directional rather than definitive."
        )
    if len(historical) < 9:
        findings.append(
            f"Caution: only {len(historical)} historical periods were available, so the forecast is based on limited time depth."
        )
    if parse_success_rate < 0.95:
        findings.append(
            f"Caution: datetime parsing succeeded on {parse_success_rate:.0%} of rows, so a small portion of the source data was excluded from the forecast."
        )
    return findings


def _classify_volatility(series: pd.Series) -> tuple[str, str]:
    """Classify recent series volatility from period-over-period changes."""
    values = series.astype(float)
    recent_values = values.tail(min(6, len(values)))
    pct_changes = recent_values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if pct_changes.empty:
        return "stable", "there are too few period-over-period changes to show meaningful instability"
    volatility_score = float(pct_changes.std())
    relative_std = float(recent_values.std() / abs(recent_values.mean())) if recent_values.mean() else 0.0
    if volatility_score >= 0.35:
        return (
            "volatile",
            f"the last {len(recent_values)} historical periods show high variance relative to the series mean, with a volatility ratio of {relative_std:.2f}",
        )
    if volatility_score >= 0.15:
        return (
            "moderately variable",
            f"the last {len(recent_values)} historical periods show moderate variance relative to the series mean, with a volatility ratio of {relative_std:.2f}",
        )
    return (
        "stable",
        f"the last {len(recent_values)} historical periods remain relatively close to the series mean, with a volatility ratio of {relative_std:.2f}",
    )


def _build_status_reason(
    readiness_status: str,
    grain_label: str,
    unique_periods: int,
) -> str:
    """Build a concise readiness explanation for the forecast workflow."""
    prefix = "Forecasting is enabled cautiously." if readiness_status == "Conditional" else "Forecasting is enabled."
    return f"{prefix} Using {grain_label.lower()} aggregation across {unique_periods} historical periods."


def _build_unavailable_forecast(
    readiness_status: str,
    readiness_reason: str,
    parse_success_rate: float,
    metric_valid_ratio: float,
    aggregation_grain: str,
    forecast_horizon: int,
    validation_messages: list[str],
    readiness_reasons: list[str],
    historical: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return a blocked forecast result with a professional explanation."""
    if historical is None:
        historical = pd.DataFrame(columns=["period", "metric_value"])

    return {
        "status": "Not Ready",
        "status_reason": "Forecasting is blocked because the selected time-series inputs do not meet the minimum validation rules.",
        "readiness_reasons": readiness_reasons,
        "input_summary": {
            "aggregation_grain": aggregation_grain,
            "aggregation_grain_label": _format_grain_label(aggregation_grain),
            "forecast_horizon": forecast_horizon,
            "datetime_parse_success_rate": parse_success_rate,
            "metric_valid_ratio": metric_valid_ratio,
            "historical_periods": int(len(historical)),
            "non_null_aggregated_observations": int(len(historical)),
            "predictive_readiness_status": readiness_status,
            "predictive_readiness_reason": readiness_reason,
            "forecast_method_label": FORECAST_METHOD_LABEL,
        },
        "historical_trend": historical,
        "forecast_table": pd.DataFrame(columns=["Period", "metric_value"]),
        "forecast_chart": historical.assign(series_type="Actual"),
        "summary_cards": {
            "historical_periods_used": int(len(historical)),
            "forecast_horizon": forecast_horizon,
            "latest_actual_value": float(historical["metric_value"].iloc[-1]) if not historical.empty else 0.0,
            "first_forecast_value": 0.0,
            "projected_average_forecast_value": 0.0,
            "projected_direction": "Blocked",
        },
        "findings": [
            "Forecasting was not forced because the selected data does not provide a sufficiently strong time-series basis."
        ],
        "validation_messages": validation_messages,
    }


def _metric_priority_key(column_name: str) -> tuple[int, str]:
    """Rank forecast metrics so stronger business metrics are selected by default."""
    normalized_name = _normalize_name(column_name)
    for index, preferred_metric in enumerate(FORECAST_METRIC_PRIORITY):
        if preferred_metric in normalized_name:
            return (index, normalized_name)
    return (len(FORECAST_METRIC_PRIORITY), normalized_name)


def _format_grain_label(grain_code: str) -> str:
    """Convert an internal grain code into a user-friendly label."""
    return GRAIN_LABELS.get(grain_code, grain_code)


def _format_forecast_table(forecast_table: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready forecast table with clean period labels."""
    display_table = forecast_table.copy()
    display_table["Period"] = pd.to_datetime(display_table["period"]).dt.strftime("%Y-%m-%d")
    return display_table[["Period", "metric_value"]]


def _classify_direction(reference_value: float, comparison_value: float) -> str:
    """Classify forecast direction using one shared threshold rule."""
    difference = comparison_value - reference_value
    if abs(difference) <= max(abs(reference_value) * 0.02, 1.0):
        return "Flat"
    if difference > 0:
        return "Up"
    return "Down"


def _build_readiness_reasons(
    readiness_status: str,
    readiness_reason: str,
    parse_success_rate: float,
    unique_periods: int,
    grain_label: str,
    validation_messages: list[str],
) -> list[str]:
    """Build a compact, professional readiness reason list."""
    reasons = [
        f"{parse_success_rate:.0%} datetime parse success",
        f"{unique_periods} historical periods available",
        f"{grain_label} aggregation selected",
        f"Predictive readiness status is {readiness_status.lower()}",
    ]
    if readiness_reason:
        reasons.append(readiness_reason.rstrip("."))
    reasons.extend(validation_messages)
    return reasons


def _normalize_name(name: str) -> str:
    """Normalize column names for rule checks."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _contains_keyword(normalized_name: str, keywords: set[str]) -> bool:
    """Check whether a normalized name includes any target keyword."""
    return any(keyword in normalized_name for keyword in keywords)


def _looks_like_identifier(normalized_name: str, unique_ratio: float, row_count: int) -> bool:
    """Check whether a numeric field is more likely to be an identifier than a metric."""
    if normalized_name in ID_KEYWORDS or normalized_name.endswith("_id"):
        return True
    if (
        _contains_keyword(normalized_name, {"id", "code", "key"})
        and not _contains_keyword(normalized_name, BUSINESS_METRIC_KEYWORDS)
    ):
        return True
    return (
        row_count >= 20
        and unique_ratio >= 0.98
        and not _contains_keyword(normalized_name, BUSINESS_METRIC_KEYWORDS)
    )
