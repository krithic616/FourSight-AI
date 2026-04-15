"""Grounded AI insight generation helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

try:
    from ai.instruction_handler import normalize_instruction, should_use_compact_context
    from ai.ollama_client import generate_ollama_response
    from ai.prompt_builder import build_prompt
    from analytics.diagnostic import get_diagnostic_options, run_diagnostic_analysis
    from analytics.prescriptive import build_prescriptive_analysis, get_prescriptive_options
    from analytics.predictive import get_predictive_options, run_predictive_forecast
except ModuleNotFoundError:
    from app.ai.instruction_handler import normalize_instruction, should_use_compact_context
    from app.ai.ollama_client import generate_ollama_response
    from app.ai.prompt_builder import build_prompt
    from app.analytics.diagnostic import get_diagnostic_options, run_diagnostic_analysis
    from app.analytics.prescriptive import build_prescriptive_analysis, get_prescriptive_options
    from app.analytics.predictive import get_predictive_options, run_predictive_forecast


def build_ai_context(
    dataset_summary: dict[str, object],
    kpis: dict[str, int],
    duplicate_rows: int,
    missing_value_summary: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    preprocessing_summary: dict[str, object],
    dataset_intelligence: dict[str, object],
    dataframe: pd.DataFrame,
) -> dict[str, object]:
    """Build a grounded AI context object from computed app outputs."""
    diagnostic_context = _build_diagnostic_context(
        dataframe=dataframe,
        column_profiles=column_profiles,
        dataset_intelligence=dataset_intelligence,
    )
    predictive_context = _build_predictive_context(
        dataframe=dataframe,
        column_profiles=column_profiles,
        dataset_intelligence=dataset_intelligence,
    )
    prescriptive_context = _build_prescriptive_context(
        dataframe=dataframe,
        column_profiles=column_profiles,
        dataset_intelligence=dataset_intelligence,
    )

    return {
        "data_quality_summary": {
            "duplicate_rows": duplicate_rows,
            "columns_with_missing_values": int(len(missing_value_summary)),
            "top_missing_columns": _compact_missing_columns(missing_value_summary),
            "top_quality_notes": _compact_quality_notes(
                duplicate_rows=duplicate_rows,
                missing_value_summary=missing_value_summary,
                preprocessing_summary=preprocessing_summary,
            ),
        },
        "column_profiling_summary": {
            "numeric_column_count": len(column_profiles.get("numeric", [])),
            "categorical_column_count": len(column_profiles.get("categorical", [])),
            "datetime_column_count": len(column_profiles.get("datetime", [])),
        },
        "dataset_intelligence_summary": {
            "dataset_type": dataset_intelligence.get("dataset_type"),
            "readiness_states": {
                key: value.get("status")
                for key, value in dataset_intelligence.get("readiness", {}).items()
            },
        },
        "descriptive_summary": {
            "kpis": kpis,
            "dataset_metadata": {
                "row_count": dataset_summary.get("row_count"),
                "column_count": dataset_summary.get("column_count"),
            },
        },
        "diagnostic_summary": diagnostic_context,
        "predictive_summary": predictive_context,
        "prescriptive_summary": prescriptive_context,
    }


def generate_insight(
    model_name: str,
    response_mode: str,
    instruction: str,
    context: dict[str, object],
) -> dict[str, object]:
    """Generate a grounded response from the local Ollama model."""
    normalized_instruction = normalize_instruction(instruction)
    prompt = build_prompt(
        context=context,
        instruction=normalized_instruction,
        response_mode=response_mode,
    )
    result = generate_ollama_response(model=model_name, prompt=prompt)
    if not result["success"]:
        return {
            "success": False,
            "response": "",
            "error": result["error"],
            "error_type": result.get("error_type", "generation_failed"),
            "instruction": normalized_instruction,
        }
    return {
        "success": True,
        "response": result["response"],
        "error": "",
        "error_type": "",
        "instruction": normalized_instruction,
    }


def build_ai_generation_bundle(
    dataset_summary: dict[str, object],
    kpis: dict[str, int],
    duplicate_rows: int,
    missing_value_summary: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    preprocessing_summary: dict[str, object],
    dataset_intelligence: dict[str, object],
    dataframe: pd.DataFrame,
    instruction: str,
) -> dict[str, object]:
    """Build both standard and ultra-compact AI contexts for generation and fallback."""
    standard_context = build_ai_context(
        dataset_summary=dataset_summary,
        kpis=kpis,
        duplicate_rows=duplicate_rows,
        missing_value_summary=missing_value_summary,
        column_profiles=column_profiles,
        preprocessing_summary=preprocessing_summary,
        dataset_intelligence=dataset_intelligence,
        dataframe=dataframe,
    )
    compact_context = build_compact_ai_context(
        dataset_summary=dataset_summary,
        kpis=kpis,
        duplicate_rows=duplicate_rows,
        missing_value_summary=missing_value_summary,
        column_profiles=column_profiles,
        preprocessing_summary=preprocessing_summary,
        dataset_intelligence=dataset_intelligence,
        dataframe=dataframe,
    )
    normalized_instruction = normalize_instruction(instruction)
    use_compact_context = should_use_compact_context(normalized_instruction)
    active_context = compact_context if use_compact_context else standard_context
    fallback_summary = build_deterministic_fallback_summary(compact_context)
    return {
        "context": active_context,
        "compact_context": compact_context,
        "standard_context": standard_context,
        "fallback_summary": fallback_summary,
        "normalized_instruction": normalized_instruction,
        "used_compact_context": use_compact_context,
    }


def build_compact_ai_context(
    dataset_summary: dict[str, object],
    kpis: dict[str, int],
    duplicate_rows: int,
    missing_value_summary: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    preprocessing_summary: dict[str, object],
    dataset_intelligence: dict[str, object],
    dataframe: pd.DataFrame,
) -> dict[str, object]:
    """Build an ultra-compact context specifically for low-resource AI generation."""
    full_context = build_ai_context(
        dataset_summary=dataset_summary,
        kpis=kpis,
        duplicate_rows=duplicate_rows,
        missing_value_summary=missing_value_summary,
        column_profiles=column_profiles,
        preprocessing_summary=preprocessing_summary,
        dataset_intelligence=dataset_intelligence,
        dataframe=dataframe,
    )
    return {
        "dataset_type": full_context["dataset_intelligence_summary"].get("dataset_type"),
        "top_kpi_facts": _build_top_kpi_facts(full_context),
        "top_diagnostic_findings": _limit_text_items(
            full_context["diagnostic_summary"].get("findings", []),
            limit=3,
        ),
        "predictive_summary": _build_predictive_summary(full_context["predictive_summary"]),
        "top_recommendations": _build_top_recommendations(full_context["prescriptive_summary"]),
    }


def build_deterministic_fallback_summary(compact_context: dict[str, object]) -> str:
    """Build a compact deterministic fallback summary for the AI output panel."""
    lines = [
        "Compact fallback summary",
        "",
        f"- Dataset type: {compact_context.get('dataset_type') or 'Unknown'}",
    ]
    for fact in compact_context.get("top_kpi_facts", [])[:3]:
        lines.append(f"- KPI: {fact}")
    for finding in compact_context.get("top_diagnostic_findings", [])[:3]:
        lines.append(f"- Diagnostic: {finding}")
    predictive_summary = str(compact_context.get("predictive_summary", "")).strip()
    if predictive_summary:
        lines.append(f"- Predictive direction: {predictive_summary}")
    for recommendation in compact_context.get("top_recommendations", [])[:3]:
        lines.append(f"- Recommendation: {recommendation}")
    return "\n".join(lines)


def _build_diagnostic_context(
    dataframe: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    dataset_intelligence: dict[str, object],
) -> dict[str, object]:
    """Build a compact diagnostic context block."""
    options = get_diagnostic_options(
        dataframe=dataframe,
        column_profiles=column_profiles,
        readiness=dataset_intelligence["readiness"]["diagnostic"],
    )
    if not options["enabled"]:
        return {
            "available": False,
            "reason": dataset_intelligence["readiness"]["diagnostic"]["reason"],
        }

    analysis = run_diagnostic_analysis(
        dataframe=dataframe,
        metric_column=options["metric_columns"][0],
        dimension_column=options["dimension_columns"][0],
        date_column=options["default_date_column"],
        breakdown_dimension=False,
    )
    summary = analysis["summary"]
    return {
        "available": True,
        "metric": analysis["metric_column"],
        "dimension": analysis["dimension_column"],
        "top_group": summary["top_group"],
        "lowest_group": summary["bottom_group"],
        "concentration": summary["concentration_classification"],
        "spread": summary["spread_classification"],
        "findings": analysis["findings"][:3],
    }


def _build_predictive_context(
    dataframe: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    dataset_intelligence: dict[str, object],
) -> dict[str, object]:
    """Build a compact predictive context block."""
    options = get_predictive_options(
        dataframe=dataframe,
        column_profiles=column_profiles,
        readiness=dataset_intelligence["readiness"]["predictive"],
    )
    if not options["enabled"]:
        return {
            "available": False,
            "reason": dataset_intelligence["readiness"]["predictive"]["reason"],
        }

    forecast = run_predictive_forecast(
        dataframe=dataframe,
        metric_column=options["default_metric_column"],
        datetime_column=options["default_datetime_column"] or options["datetime_columns"][0],
        aggregation_grain="Auto",
        forecast_horizon=6,
        readiness_status=options["readiness_status"],
        readiness_reason=options["readiness_reason"],
    )
    return {
        "available": True,
        "status": forecast["status"],
        "forecast_summary": forecast["summary_cards"],
        "readiness_reasons": forecast["readiness_reasons"][:4],
        "findings": forecast["findings"][:3],
    }


def _build_prescriptive_context(
    dataframe: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    dataset_intelligence: dict[str, object],
) -> dict[str, object]:
    """Build a compact prescriptive context block."""
    options = get_prescriptive_options(
        dataframe=dataframe,
        column_profiles=column_profiles,
        dataset_intelligence=dataset_intelligence,
    )
    if not options["enabled"]:
        return {
            "available": False,
            "reason": dataset_intelligence["readiness"]["prescriptive"]["reason"],
        }

    analysis = build_prescriptive_analysis(
        dataframe=dataframe,
        column_profiles=column_profiles,
        dataset_intelligence=dataset_intelligence,
    )
    return {
        "available": True,
        "status": analysis["status"],
        "summary": analysis["summary"],
        "findings": analysis["findings"][:3],
        "top_recommendations": [
            {
                "category": recommendation["category"],
                "priority": recommendation["priority"],
                "issue_detected": recommendation["issue_detected"],
                "suggested_action": recommendation["suggested_action"],
                "recommendation_basis": recommendation["recommendation_basis"],
            }
            for recommendation in analysis["recommendations"][:2]
        ],
    }


def _compact_missing_columns(missing_value_summary: pd.DataFrame) -> list[dict[str, object]]:
    """Return a compact missing-value summary for prompt grounding."""
    if missing_value_summary.empty:
        return []
    records = missing_value_summary.head(3).to_dict(orient="records")
    compact_records: list[dict[str, object]] = []
    for record in records:
        compact_records.append(
            {
                "column_name": record.get("column_name"),
                "missing_count": record.get("missing_count"),
                "missing_percentage": record.get("missing_percentage"),
            }
        )
    return compact_records


def _compact_quality_notes(
    duplicate_rows: int,
    missing_value_summary: pd.DataFrame,
    preprocessing_summary: dict[str, object],
) -> list[str]:
    """Return the smallest useful quality notes for AI grounding."""
    notes = [
        f"Duplicate rows detected: {duplicate_rows}",
        f"Columns with missing values: {int(len(missing_value_summary))}",
    ]
    notes.extend(preprocessing_summary.get("profiling_actions", [])[:2])
    return notes[:4]


def _build_top_kpi_facts(context: dict[str, object]) -> list[str]:
    """Build the smallest useful KPI fact list for low-resource prompting."""
    descriptive_summary = context.get("descriptive_summary", {})
    data_quality_summary = context.get("data_quality_summary", {})
    kpis = descriptive_summary.get("kpis", {})
    dataset_metadata = descriptive_summary.get("dataset_metadata", {})

    candidate_facts = [
        f"Rows: {dataset_metadata.get('row_count', kpis.get('total_rows', 0))}",
        f"Columns: {dataset_metadata.get('column_count', kpis.get('total_columns', 0))}",
        f"Duplicate rows: {data_quality_summary.get('duplicate_rows', kpis.get('duplicate_rows', 0))}",
        f"Columns with missing values: {data_quality_summary.get('columns_with_missing_values', kpis.get('columns_with_missing_values', 0))}",
    ]
    prioritized_facts = [fact for fact in candidate_facts if not fact.endswith(": None")]
    return prioritized_facts[:3]


def _build_predictive_summary(predictive_summary: dict[str, object]) -> str:
    """Return one short predictive summary line."""
    if not predictive_summary.get("available"):
        reason = str(predictive_summary.get("reason", "")).strip()
        return reason or "Predictive summary is unavailable."

    forecast_summary = predictive_summary.get("forecast_summary", {})
    direction = forecast_summary.get("projected_direction", "Unknown")
    latest_actual = forecast_summary.get("latest_actual_value")
    first_forecast = forecast_summary.get("first_forecast_value")
    if latest_actual is None or first_forecast is None:
        return f"Projected direction is {direction}."
    return (
        f"Projected direction is {direction}, with the first forecast at {first_forecast:.2f} "
        f"versus the latest actual value of {latest_actual:.2f}."
    )


def _build_top_recommendations(prescriptive_summary: dict[str, object]) -> list[str]:
    """Return at most three concise recommendation lines."""
    if not prescriptive_summary.get("available"):
        reason = str(prescriptive_summary.get("reason", "")).strip()
        return [reason or "No prescriptive recommendation is available."]

    recommendations: list[str] = []
    for recommendation in prescriptive_summary.get("top_recommendations", [])[:3]:
        priority = recommendation.get("priority", "Unknown")
        issue = recommendation.get("issue_detected", "Issue not specified")
        action = recommendation.get("suggested_action", "Action not specified")
        recommendations.append(f"{priority} priority: {issue}. Action: {action}")
    return recommendations[:3]


def _limit_text_items(items: list[object], limit: int) -> list[str]:
    """Normalize a list of short text items."""
    normalized_items = [str(item).strip() for item in items if str(item).strip()]
    return normalized_items[:limit]
