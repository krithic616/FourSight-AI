"""Tests for deterministic report generation."""

from __future__ import annotations

import pandas as pd

from app.core.data_cleaner import build_missing_value_summary, build_preprocessing_summary, count_duplicate_rows
from app.core.data_profiler import build_dataset_summary, profile_column_types
from app.core.dataset_detector import build_dataset_intelligence
from app.reporting.export_html import export_html
from app.reporting.report_builder import build_report, export_txt


def test_build_report_generates_without_ai_output():
    dataframe = pd.DataFrame(
        {
            "order_date": pd.date_range("2024-01-01", periods=90, freq="D"),
            "net_revenue": [100 + (index % 7) * 5 for index in range(90)],
            "profit": [20 + (index % 5) for index in range(90)],
            "region": ["North", "South", "West"] * 30,
        }
    )
    dataset_summary = build_dataset_summary(dataframe)
    dataset_summary.update({"file_name": "sample.csv", "file_size_display": "1.0 KB"})
    duplicate_rows = count_duplicate_rows(dataframe)
    missing_value_summary = build_missing_value_summary(dataframe)
    preprocessing_summary = build_preprocessing_summary(
        dataframe=dataframe,
        duplicate_rows=duplicate_rows,
        missing_value_summary=missing_value_summary,
    )
    column_profiles = profile_column_types(dataframe)
    dataset_intelligence = build_dataset_intelligence(dataframe, column_profiles)

    report = build_report(
        dataset_summary=dataset_summary,
        kpis={"total_rows": 90, "total_columns": 4, "duplicate_rows": 0, "columns_with_missing_values": 0},
        dataframe=dataframe,
        duplicate_rows=duplicate_rows,
        missing_value_summary=missing_value_summary,
        column_profiles=column_profiles,
        preprocessing_summary=preprocessing_summary,
        dataset_intelligence=dataset_intelligence,
    )

    assert report["title"] == "FourSight AI Business Insight Report"
    assert all(section["title"] != "AI Insight Summary" for section in report["sections"])


def test_report_exports_include_expected_content():
    report = {
        "title": "FourSight AI Business Insight Report",
        "dataset_name": "sample.csv",
        "generated_at": "2026-04-14 12:00:00",
        "sections": [
            {"title": "Descriptive Analytics Summary", "items": ["Total rows: 90"]},
            {"title": "AI Insight Summary", "items": ["Leadership should focus on margin discipline."]},
        ],
    }

    txt_report = export_txt(report)
    html_report = export_html(report)

    assert "FourSight AI Business Insight Report" in txt_report
    assert "AI Insight Summary" in txt_report
    assert "<html" in html_report.lower()
    assert "Leadership should focus on margin discipline." in html_report


def test_build_report_uses_cleaned_ai_summary_when_available():
    dataframe = pd.DataFrame(
        {
            "order_date": pd.date_range("2024-01-01", periods=90, freq="D"),
            "net_revenue": [100 + (index % 7) * 5 for index in range(90)],
            "profit": [20 + (index % 5) for index in range(90)],
            "region": ["North", "South", "West"] * 30,
        }
    )
    dataset_summary = build_dataset_summary(dataframe)
    dataset_summary.update({"file_name": "sample.csv", "file_size_display": "1.0 KB"})
    duplicate_rows = count_duplicate_rows(dataframe)
    missing_value_summary = build_missing_value_summary(dataframe)
    preprocessing_summary = build_preprocessing_summary(
        dataframe=dataframe,
        duplicate_rows=duplicate_rows,
        missing_value_summary=missing_value_summary,
    )
    column_profiles = profile_column_types(dataframe)
    dataset_intelligence = build_dataset_intelligence(dataframe, column_profiles)

    report = build_report(
        dataset_summary=dataset_summary,
        kpis={"total_rows": 90, "total_columns": 4, "duplicate_rows": 0, "columns_with_missing_values": 0},
        dataframe=dataframe,
        duplicate_rows=duplicate_rows,
        missing_value_summary=missing_value_summary,
        column_profiles=column_profiles,
        preprocessing_summary=preprocessing_summary,
        dataset_intelligence=dataset_intelligence,
        ai_response=(
            "Executive Summary:\n"
            "- revenue remains stable.\n"
            "Key Insights:\n"
            "- top segment concentration remains meaningful.\n"
            "Risks:\n"
            "- margin pressure persists.\n"
            "Recommended Actions:\n"
            "- protect pricing discipline.\n"
        ),
    )

    ai_section = next(section for section in report["sections"] if section["title"] == "AI Insight Summary")
    assert ai_section["items"][0] == "Executive Summary"
    assert "Revenue remains stable." in ai_section["items"]


def test_build_report_falls_back_to_short_ai_summary_when_quality_is_weak():
    dataframe = pd.DataFrame(
        {
            "order_date": pd.date_range("2024-01-01", periods=90, freq="D"),
            "net_revenue": [100 + (index % 7) * 5 for index in range(90)],
            "profit": [20 + (index % 5) for index in range(90)],
            "region": ["North", "South", "West"] * 30,
        }
    )
    dataset_summary = build_dataset_summary(dataframe)
    dataset_summary.update({"file_name": "sample.csv", "file_size_display": "1.0 KB"})
    duplicate_rows = count_duplicate_rows(dataframe)
    missing_value_summary = build_missing_value_summary(dataframe)
    preprocessing_summary = build_preprocessing_summary(
        dataframe=dataframe,
        duplicate_rows=duplicate_rows,
        missing_value_summary=missing_value_summary,
    )
    column_profiles = profile_column_types(dataframe)
    dataset_intelligence = build_dataset_intelligence(dataframe, column_profiles)

    report = build_report(
        dataset_summary=dataset_summary,
        kpis={"total_rows": 90, "total_columns": 4, "duplicate_rows": 0, "columns_with_missing_values": 0},
        dataframe=dataframe,
        duplicate_rows=duplicate_rows,
        missing_value_summary=missing_value_summary,
        column_profiles=column_profiles,
        preprocessing_summary=preprocessing_summary,
        dataset_intelligence=dataset_intelligence,
        ai_response="###\n- ok\n11111\nrevenue is stable",
    )

    ai_section = next(section for section in report["sections"] if section["title"] == "AI Insight Summary")
    assert "Revenue is stable" in ai_section["items"] or "Revenue is stable." in ai_section["items"]
