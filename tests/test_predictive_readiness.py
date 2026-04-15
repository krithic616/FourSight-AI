"""Tests for cautious predictive forecasting."""

from __future__ import annotations

import pandas as pd

from app.analytics.predictive import get_predictive_options, predictive_readiness, run_predictive_forecast
from app.core.data_profiler import profile_column_types


def test_predictive_readiness_for_non_empty_dataframe():
    dataframe = pd.DataFrame({"value": [1]})
    assert predictive_readiness(dataframe) is True


def test_get_predictive_options_enables_time_series_forecasting_for_usable_data():
    dataframe = pd.DataFrame(
        {
            "order_date": pd.date_range("2024-01-01", periods=180, freq="D"),
            "net_revenue": [100 + (index % 14) * 3 for index in range(180)],
            "discount_pct": [5 + (index % 4) for index in range(180)],
            "order_id": range(1, 181),
        }
    )

    options = get_predictive_options(
        dataframe=dataframe,
        column_profiles=profile_column_types(dataframe),
        readiness={"status": "Ready", "reason": "Strong datetime support."},
    )

    assert options["enabled"] is True
    assert "net_revenue" in options["metric_columns"]
    assert "order_id" not in options["metric_columns"]
    assert options["default_metric_column"] == "net_revenue"
    assert options["default_datetime_column"] == "order_date"


def test_run_predictive_forecast_returns_deterministic_projection_for_valid_series():
    dataframe = pd.DataFrame(
        {
            "order_date": pd.date_range("2024-01-01", periods=180, freq="D"),
            "quantity": [50 + (index % 10) for index in range(180)],
        }
    )

    forecast = run_predictive_forecast(
        dataframe=dataframe,
        metric_column="quantity",
        datetime_column="order_date",
        aggregation_grain="Auto",
        forecast_horizon=6,
        readiness_status="Ready",
        readiness_reason="Time-series structure is strong.",
    )

    assert forecast["status"] == "Ready"
    assert not forecast["historical_trend"].empty
    assert len(forecast["forecast_table"]) == 6
    assert forecast["input_summary"]["aggregation_grain_label"] in {"Daily", "Weekly", "Monthly"}
    assert "Period" in forecast["forecast_table"].columns
    assert forecast["summary_cards"]["projected_direction"] in {"Up", "Down", "Flat"}
    assert any(
        forecast["summary_cards"]["projected_direction"].lower() in finding
        for finding in forecast["findings"]
        if "Short-term direction" in finding
    )
    assert any("volatility ratio" in finding for finding in forecast["findings"])


def test_run_predictive_forecast_blocks_weak_history_without_forcing_projection():
    dataframe = pd.DataFrame(
        {
            "order_date": pd.date_range("2025-01-01", periods=5, freq="D"),
            "quantity": [10, 12, 11, 13, 14],
        }
    )

    forecast = run_predictive_forecast(
        dataframe=dataframe,
        metric_column="quantity",
        datetime_column="order_date",
        aggregation_grain="Daily",
        forecast_horizon=3,
        readiness_status="Conditional",
        readiness_reason="Time-series evidence is limited.",
    )

    assert forecast["status"] == "Not Ready"
    assert forecast["forecast_table"].empty
    assert forecast["validation_messages"]
    assert forecast["input_summary"]["aggregation_grain_label"] == "Daily"
