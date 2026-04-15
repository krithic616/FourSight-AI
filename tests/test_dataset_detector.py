"""Tests for dataset intelligence readiness classification."""

from __future__ import annotations

import pandas as pd

from app.core.data_profiler import profile_column_types
from app.core.dataset_detector import build_dataset_intelligence


def test_transactional_dataset_stays_conservative_for_predictive_and_prescriptive():
    dataframe = pd.DataFrame(
        {
            "order_date": pd.date_range("2025-01-01", periods=90, freq="D").strftime("%Y-%m-%d"),
            "sales_amount": [100 + (index % 7) * 10 for index in range(90)],
            "region": ["North", "South", "East"] * 30,
            "product_category": ["A", "B", "C", "D", "E"] * 18,
        }
    )

    intelligence = build_dataset_intelligence(dataframe, profile_column_types(dataframe))
    readiness = intelligence["readiness"]

    assert readiness["descriptive"]["status"] == "Ready"
    assert readiness["diagnostic"]["status"] == "Ready"
    assert readiness["predictive"]["status"] == "Conditional"
    assert readiness["prescriptive"]["status"] == "Conditional"
    assert "Conditional" in readiness["predictive"]["reason"]
    assert "should be confirmed before anything is forced" in readiness["prescriptive"]["reason"]


def test_predictive_becomes_ready_when_time_structure_is_strong():
    dataframe = pd.DataFrame(
        {
            "event_date": pd.date_range("2024-01-01", periods=180, freq="D"),
            "units_sold": [50 + (index % 14) for index in range(180)],
            "promo_flag": [index % 2 for index in range(180)],
            "store_region": ["North", "South", "West"] * 60,
        }
    )

    intelligence = build_dataset_intelligence(dataframe, profile_column_types(dataframe))
    readiness = intelligence["readiness"]

    assert readiness["predictive"]["status"] == "Ready"
    assert "time periods" in readiness["predictive"]["reason"]


def test_diagnostic_and_predictive_drop_when_dataset_lacks_dimensions():
    dataframe = pd.DataFrame(
        {
            "sales_amount": [10, 12, 15, 13, 16],
            "profit": [3, 4, 5, 3, 6],
        }
    )

    intelligence = build_dataset_intelligence(dataframe, profile_column_types(dataframe))
    readiness = intelligence["readiness"]

    assert readiness["descriptive"]["status"] == "Ready"
    assert readiness["diagnostic"]["status"] == "Conditional"
    assert readiness["predictive"]["status"] == "Not Ready"
    assert readiness["prescriptive"]["status"] == "Not Ready"
