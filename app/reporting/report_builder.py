"""Deterministic report building helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

try:
    from ai.report_writer import build_short_ai_summary, prepare_ai_summary
    from analytics.diagnostic import get_diagnostic_options, run_diagnostic_analysis
    from analytics.prescriptive import build_prescriptive_analysis, get_prescriptive_options
    from analytics.predictive import get_predictive_options, run_predictive_forecast
except ModuleNotFoundError:
    from app.ai.report_writer import build_short_ai_summary, prepare_ai_summary
    from app.analytics.diagnostic import get_diagnostic_options, run_diagnostic_analysis
    from app.analytics.prescriptive import build_prescriptive_analysis, get_prescriptive_options
    from app.analytics.predictive import get_predictive_options, run_predictive_forecast


REPORT_TITLE = "FourSight AI Business Insight Report"


def build_report(
    dataset_summary: dict[str, object],
    kpis: dict[str, int],
    dataframe: pd.DataFrame,
    duplicate_rows: int,
    missing_value_summary: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    preprocessing_summary: dict[str, object],
    dataset_intelligence: dict[str, object],
    ai_response: str = "",
) -> dict[str, Any]:
    """Build a deterministic report payload from computed analytics outputs."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    diagnostic_section = _build_diagnostic_section(
        dataframe=dataframe,
        column_profiles=column_profiles,
        dataset_intelligence=dataset_intelligence,
    )
    predictive_section = _build_predictive_section(
        dataframe=dataframe,
        column_profiles=column_profiles,
        dataset_intelligence=dataset_intelligence,
    )
    prescriptive_section = _build_prescriptive_section(
        dataframe=dataframe,
        column_profiles=column_profiles,
        dataset_intelligence=dataset_intelligence,
    )

    sections: list[dict[str, object]] = [
        {
            "title": "Data Quality Summary",
            "items": [
                f"Duplicate rows detected: {duplicate_rows}",
                f"Columns with missing values: {int(len(missing_value_summary))}",
                *preprocessing_summary.get("profiling_actions", [])[:3],
            ],
        },
        {
            "title": "Column Profiling Summary",
            "items": [
                f"Numeric columns: {len(column_profiles.get('numeric', []))}",
                f"Categorical columns: {len(column_profiles.get('categorical', []))}",
                f"Datetime columns: {len(column_profiles.get('datetime', []))}",
                f"Example numeric fields: {', '.join(column_profiles.get('numeric', [])[:5]) or 'None'}",
            ],
        },
        {
            "title": "Dataset Intelligence Summary",
            "items": [
                f"Detected dataset type: {str(dataset_intelligence.get('dataset_type', 'unknown')).replace('_', ' ').title()}",
                "Readiness states: "
                + ", ".join(
                    f"{key.title()}={value['status']}"
                    for key, value in dataset_intelligence.get("readiness", {}).items()
                ),
                "Business signals: "
                + ", ".join(dataset_intelligence.get("business_signals", [])[:5]),
            ],
        },
        {
            "title": "Descriptive Analytics Summary",
            "items": [
                f"Total rows: {kpis.get('total_rows', 0)}",
                f"Total columns: {kpis.get('total_columns', 0)}",
                f"Dataset name: {dataset_summary.get('file_name', 'uploaded.csv')}",
                f"File size: {dataset_summary.get('file_size_display', 'Unknown')}",
            ],
        },
        {
            "title": "Diagnostic Analytics Summary",
            "items": diagnostic_section["items"],
        },
        {
            "title": "Predictive Analytics Summary",
            "items": predictive_section["items"],
        },
        {
            "title": "Prescriptive Analytics Summary",
            "items": prescriptive_section["items"],
        },
        {
            "title": "Key Recommendations",
            "items": prescriptive_section["recommendations"],
        },
    ]

    if ai_response.strip():
        prepared_ai_summary = prepare_ai_summary(ai_response)
        ai_items = prepared_ai_summary["items"]
        if not prepared_ai_summary["include_in_report"]:
            ai_items = build_short_ai_summary(ai_items)
        if ai_items:
            sections.append(
                {
                    "title": "AI Insight Summary",
                    "items": ai_items,
                }
            )

    return {
        "title": REPORT_TITLE,
        "dataset_name": dataset_summary.get("file_name", "uploaded.csv"),
        "generated_at": generated_at,
        "sections": sections,
    }


def export_txt(report: dict[str, Any]) -> str:
    """Convert the report payload into a clean plain-text report."""
    lines = [
        report["title"],
        f"Dataset: {report['dataset_name']}",
        f"Generated: {report['generated_at']}",
        "",
    ]

    for section in report["sections"]:
        lines.append(section["title"])
        lines.append("-" * len(section["title"]))
        for item in section["items"]:
            lines.append(f"- {item}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_report_preview(report: dict[str, Any], max_sections: int = 4) -> str:
    """Return a compact preview string for the report tab."""
    preview_sections = report["sections"][:max_sections]
    preview_report = {
        **report,
        "sections": preview_sections,
    }
    return export_txt(preview_report)


def _build_diagnostic_section(
    dataframe: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    dataset_intelligence: dict[str, object],
) -> dict[str, list[str]]:
    """Build a deterministic diagnostic report section."""
    options = get_diagnostic_options(
        dataframe=dataframe,
        column_profiles=column_profiles,
        readiness=dataset_intelligence["readiness"]["diagnostic"],
    )
    if not options["enabled"]:
        return {"items": [dataset_intelligence["readiness"]["diagnostic"]["reason"]]}

    analysis = run_diagnostic_analysis(
        dataframe=dataframe,
        metric_column=options["metric_columns"][0],
        dimension_column=options["dimension_columns"][0],
        date_column=options["default_date_column"],
        breakdown_dimension=False,
    )
    summary = analysis["summary"]
    return {
        "items": [
            f"Primary metric reviewed: {analysis['metric_column']}",
            f"Primary dimension reviewed: {analysis['dimension_column']}",
            f"Leading group: {summary['top_group']}",
            f"Lowest-contributing group: {summary['bottom_group']}",
            *analysis["findings"][:3],
        ]
    }


def _build_predictive_section(
    dataframe: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    dataset_intelligence: dict[str, object],
) -> dict[str, list[str]]:
    """Build a deterministic predictive report section."""
    options = get_predictive_options(
        dataframe=dataframe,
        column_profiles=column_profiles,
        readiness=dataset_intelligence["readiness"]["predictive"],
    )
    if not options["enabled"]:
        return {"items": [dataset_intelligence["readiness"]["predictive"]["reason"]]}

    forecast = run_predictive_forecast(
        dataframe=dataframe,
        metric_column=options["default_metric_column"],
        datetime_column=options["default_datetime_column"] or options["datetime_columns"][0],
        aggregation_grain="Auto",
        forecast_horizon=6,
        readiness_status=options["readiness_status"],
        readiness_reason=options["readiness_reason"],
    )
    summary = forecast["summary_cards"]
    items = [
        f"Forecast status: {forecast['status']}",
        f"Forecast metric: {forecast['input_summary']['metric_column']}",
        f"Projected direction: {summary['projected_direction']} (directional only when readiness is conditional)",
        f"Historical periods used: {summary['historical_periods_used']}",
        *forecast["findings"][:3],
    ]
    if forecast["validation_messages"]:
        items.extend(forecast["validation_messages"][:2])
    return {"items": items}


def _build_prescriptive_section(
    dataframe: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    dataset_intelligence: dict[str, object],
) -> dict[str, list[str]]:
    """Build a deterministic prescriptive report section."""
    options = get_prescriptive_options(
        dataframe=dataframe,
        column_profiles=column_profiles,
        dataset_intelligence=dataset_intelligence,
    )
    if not options["enabled"]:
        return {
            "items": [dataset_intelligence["readiness"]["prescriptive"]["reason"]],
            "recommendations": ["Prescriptive recommendations are not available yet for this dataset."],
        }

    analysis = build_prescriptive_analysis(
        dataframe=dataframe,
        column_profiles=column_profiles,
        dataset_intelligence=dataset_intelligence,
    )
    recommendation_lines = [
        (
            f"Category: {recommendation['category']} | "
            f"Priority: {recommendation['priority']} | "
            f"Issue: {recommendation['issue_detected']} | "
            f"Action: {recommendation['suggested_action']}"
        )
        for recommendation in analysis["recommendations"][:5]
    ]
    return {
        "items": [
            f"Prescriptive status: {analysis['status']}",
            f"Recommendations triggered: {analysis['summary']['total_recommendations']}",
            f"Highest-priority issue: {analysis['summary']['highest_priority_issue']}",
            *analysis["findings"][:3],
        ],
        "recommendations": recommendation_lines or ["No recommendations were triggered."],
    }
