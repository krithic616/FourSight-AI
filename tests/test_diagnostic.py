"""Tests for deterministic diagnostic analytics."""

from __future__ import annotations

import pandas as pd

from app.analytics.diagnostic import get_diagnostic_options, run_diagnostic_analysis
from app.core.data_profiler import profile_column_types


def test_get_diagnostic_options_excludes_numeric_identifiers():
    dataframe = pd.DataFrame(
        {
            "order_id": range(1, 11),
            "net_revenue": [100, 120, 90, 130, 110, 140, 95, 150, 160, 105],
            "region": ["North", "South"] * 5,
            "segment": ["Retail", "Wholesale"] * 5,
            "order_date": pd.date_range("2025-01-01", periods=10, freq="D"),
        }
    )

    options = get_diagnostic_options(
        dataframe=dataframe,
        column_profiles=profile_column_types(dataframe),
        readiness={"status": "Ready"},
    )

    assert options["enabled"] is True
    assert "order_id" not in options["metric_columns"]
    assert "net_revenue" in options["metric_columns"]
    assert "region" in options["dimension_columns"]
    assert "order_date" in options["datetime_columns"]
    assert options["default_date_column"] == "order_date"


def test_run_diagnostic_analysis_returns_group_summary_and_findings():
    dataframe = pd.DataFrame(
        {
            "net_revenue": [100, 120, 70, 50, 40, 30],
            "region": ["North", "North", "South", "South", "West", "West"],
            "order_date": pd.date_range("2025-01-01", periods=6, freq="D"),
            "discount_pct": [5, 7, 10, 12, 25, 30],
            "profit": [30, 35, 18, 12, 3, 2],
        }
    )

    analysis = run_diagnostic_analysis(
        dataframe=dataframe,
        metric_column="net_revenue",
        dimension_column="region",
        date_column="order_date",
        breakdown_dimension=True,
    )

    assert analysis["summary"]["total_metric_value"] == 410.0
    assert analysis["summary"]["group_count"] == 3
    assert analysis["summary"]["top_group"] == "North"
    assert analysis["summary"]["bottom_group"] == "West"
    assert analysis["summary"]["lowest_group_label"] == "Lowest-Contributing Group"
    assert analysis["grouped_comparison"].iloc[0]["group"] == "North"
    assert analysis["show_split_top_bottom"] is False
    assert not analysis["ranked_groups"].empty
    assert not analysis["top_groups"].empty
    assert not analysis["bottom_groups"].empty
    assert analysis["trend_analysis"] is not None
    assert "metric_value" in analysis["trend_analysis"]["summary"].columns
    assert "metric_value" in analysis["trend_analysis"]["by_dimension"].columns
    assert any("Performance spread is" in finding for finding in analysis["findings"])
    assert any("Contribution concentration is" in finding for finding in analysis["findings"])
    assert any("Average contribution per group" in finding for finding in analysis["findings"])


def test_run_diagnostic_analysis_splits_top_and_bottom_when_group_count_is_larger():
    dataframe = pd.DataFrame(
        {
            "net_revenue": [200, 180, 160, 140, 120, 100, 80, 60, 40, 20, 10, 5],
            "region": ["A", "B", "C", "D", "E", "F"] * 2,
            "order_date": pd.date_range("2025-01-01", periods=12, freq="MS"),
        }
    )

    analysis = run_diagnostic_analysis(
        dataframe=dataframe,
        metric_column="net_revenue",
        dimension_column="region",
        date_column="order_date",
        breakdown_dimension=False,
    )

    assert analysis["summary"]["group_count"] == 6
    assert analysis["show_split_top_bottom"] is True
    assert len(analysis["top_groups"]) == 5
    assert len(analysis["bottom_groups"]) == 5
