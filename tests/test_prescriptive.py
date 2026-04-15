"""Tests for deterministic prescriptive analytics."""

from __future__ import annotations

import pandas as pd

from app.analytics.prescriptive import build_prescriptive_analysis, get_prescriptive_options
from app.core.data_profiler import profile_column_types
from app.core.dataset_detector import build_dataset_intelligence


def test_get_prescriptive_options_enables_when_supporting_evidence_exists():
    dataframe = pd.DataFrame(
        {
            "order_date": pd.date_range("2024-01-01", periods=180, freq="D"),
            "net_revenue": [100 + (index % 14) * 3 for index in range(180)],
            "profit": [25 + (index % 8) for index in range(180)],
            "region": ["North", "South", "West"] * 60,
        }
    )
    column_profiles = profile_column_types(dataframe)
    intelligence = build_dataset_intelligence(dataframe, column_profiles)

    options = get_prescriptive_options(dataframe, column_profiles, intelligence)

    assert options["enabled"] is True
    assert options["prescriptive_readiness"]["status"] in {"Ready", "Conditional"}


def test_build_prescriptive_analysis_returns_structured_recommendations():
    dataframe = pd.DataFrame(
        {
            "order_date": pd.date_range("2024-01-01", periods=180, freq="D"),
            "net_revenue": [240] * 60 + [120] * 60 + [40] * 60,
            "profit": [60] * 60 + [20] * 60 + [2] * 60,
            "discount_pct": [5] * 60 + [12] * 60 + [30] * 60,
            "region": ["North"] * 60 + ["South"] * 60 + ["West"] * 60,
        }
    )
    column_profiles = profile_column_types(dataframe)
    intelligence = build_dataset_intelligence(dataframe, column_profiles)

    analysis = build_prescriptive_analysis(dataframe, column_profiles, intelligence)

    assert analysis["summary"]["total_recommendations"] >= 1
    assert analysis["recommendations"]
    assert any(
        recommendation["category"] in {
            "Growth Opportunities",
            "Risk Controls",
            "Efficiency Improvements",
            "Monitoring Priorities",
        }
        for recommendation in analysis["recommendations"]
    )
    assert any(
        recommendation["recommendation_basis"] in {
            "diagnostic-based",
            "predictive-based",
            "combined",
        }
        for recommendation in analysis["recommendations"]
    )
    assert analysis["findings"]


def test_build_prescriptive_analysis_can_generate_combined_and_diverse_categories():
    dataframe = pd.DataFrame(
        {
            "order_date": pd.date_range("2024-01-01", periods=180, freq="D"),
            "net_revenue": ([300] * 60) + ([110] * 60) + ([20] * 60),
            "profit": ([70] * 60) + ([18] * 60) + ([1] * 60),
            "discount_pct": ([4] * 60) + ([12] * 60) + ([28] * 60),
            "region": (["North"] * 60) + (["South"] * 60) + (["West"] * 60),
        }
    )
    column_profiles = profile_column_types(dataframe)
    intelligence = build_dataset_intelligence(dataframe, column_profiles)

    analysis = build_prescriptive_analysis(dataframe, column_profiles, intelligence)
    categories = {recommendation["category"] for recommendation in analysis["recommendations"]}
    bases = {recommendation["recommendation_basis"] for recommendation in analysis["recommendations"]}

    assert "Risk Controls" in categories
    assert "Growth Opportunities" in categories
    assert "Efficiency Improvements" in categories
    assert "combined" in bases or "predictive-based" in bases
