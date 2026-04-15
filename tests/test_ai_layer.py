"""Tests for the grounded local AI analyst layer."""

from __future__ import annotations

import pandas as pd

from app.ai.insight_generator import (
    build_ai_context,
    build_ai_generation_bundle,
    build_compact_ai_context,
    build_deterministic_fallback_summary,
)
from app.ai.instruction_handler import (
    DEFAULT_AI_INSTRUCTION,
    normalize_instruction,
    should_use_compact_context,
)
from app.ai.ollama_client import (
    _extract_ollama_error_message,
    _is_memory_error_message,
    _select_default_model,
)
from app.ai.prompt_builder import build_prompt
from app.ai.report_writer import prepare_ai_summary
from app.core.data_cleaner import build_missing_value_summary, build_preprocessing_summary, count_duplicate_rows
from app.core.data_profiler import build_dataset_summary, profile_column_types
from app.core.dataset_detector import build_dataset_intelligence


def test_normalize_instruction_uses_safe_default():
    assert normalize_instruction("   ") == DEFAULT_AI_INSTRUCTION


def test_build_ai_context_includes_computed_analytics_layers():
    dataframe = pd.DataFrame(
        {
            "order_date": pd.date_range("2024-01-01", periods=90, freq="D"),
            "net_revenue": [100 + (index % 7) * 5 for index in range(90)],
            "profit": [20 + (index % 5) for index in range(90)],
            "region": ["North", "South", "West"] * 30,
        }
    )
    dataset_summary = build_dataset_summary(dataframe)
    dataset_summary.update(
        {
            "file_name": "sample.csv",
            "file_size_display": "1.0 KB",
        }
    )
    duplicate_rows = count_duplicate_rows(dataframe)
    missing_value_summary = build_missing_value_summary(dataframe)
    preprocessing_summary = build_preprocessing_summary(
        dataframe=dataframe,
        duplicate_rows=duplicate_rows,
        missing_value_summary=missing_value_summary,
    )
    column_profiles = profile_column_types(dataframe)
    dataset_intelligence = build_dataset_intelligence(dataframe, column_profiles)
    context = build_ai_context(
        dataset_summary=dataset_summary,
        kpis={"total_rows": 90, "total_columns": 4, "duplicate_rows": 0, "columns_with_missing_values": 0},
        duplicate_rows=duplicate_rows,
        missing_value_summary=missing_value_summary,
        column_profiles=column_profiles,
        preprocessing_summary=preprocessing_summary,
        dataset_intelligence=dataset_intelligence,
        dataframe=dataframe,
    )

    assert "data_quality_summary" in context
    assert "column_profiling_summary" in context
    assert "dataset_intelligence_summary" in context
    assert "descriptive_summary" in context
    assert "diagnostic_summary" in context
    assert "predictive_summary" in context
    assert "prescriptive_summary" in context


def test_build_prompt_includes_grounding_instructions():
    prompt = build_prompt(
        context={"descriptive_summary": {"kpis": {"total_rows": 10}}},
        instruction="Summarize the key business insights",
        response_mode="Executive",
    )

    assert "Do not invent unsupported facts" in prompt
    assert "Use only the structured analytics context" in prompt
    assert "Use short business-focused bullet points" in prompt
    assert "Do not convert raw values into percentages unless a percentage is explicitly provided" in prompt
    assert "Executive Summary:" in prompt
    assert "Summarize the key business insights" in prompt


def test_select_default_model_prefers_installed_model():
    assert _select_default_model(["llama3:latest", "gemma3n:e2b"]) == "gemma3n:e2b"


def test_select_default_model_prefers_lighter_installed_model_when_available():
    assert _select_default_model(["phi3:mini", "tinyllama", "llama3.2:3b"]) == "tinyllama"


def test_memory_error_detection_handles_ollama_error_messages():
    ollama_body = '{"error":"model requires more system memory (6.5 GiB) than is available (4.2 GiB)"}'

    extracted_error = _extract_ollama_error_message(ollama_body)

    assert "requires more system memory" in extracted_error
    assert _is_memory_error_message(extracted_error) is True


def test_build_ai_context_stays_compact():
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
    context = build_ai_context(
        dataset_summary=dataset_summary,
        kpis={"total_rows": 90, "total_columns": 4, "duplicate_rows": 0, "columns_with_missing_values": 0},
        duplicate_rows=duplicate_rows,
        missing_value_summary=missing_value_summary,
        column_profiles=column_profiles,
        preprocessing_summary=preprocessing_summary,
        dataset_intelligence=dataset_intelligence,
        dataframe=dataframe,
    )

    assert "top_quality_notes" in context["data_quality_summary"]
    assert "readiness_states" in context["dataset_intelligence_summary"]
    assert "top_recommendations" in context["prescriptive_summary"]


def test_quick_prompts_force_compact_context():
    assert should_use_compact_context("Key Business Insights")
    assert should_use_compact_context("Give 5 business insights.")
    assert not should_use_compact_context("Explain the full readiness rationale in detail")


def test_build_compact_context_contains_only_low_resource_sections():
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

    compact_context = build_compact_ai_context(
        dataset_summary=dataset_summary,
        kpis={"total_rows": 90, "total_columns": 4, "duplicate_rows": 0, "columns_with_missing_values": 0},
        duplicate_rows=duplicate_rows,
        missing_value_summary=missing_value_summary,
        column_profiles=column_profiles,
        preprocessing_summary=preprocessing_summary,
        dataset_intelligence=dataset_intelligence,
        dataframe=dataframe,
    )

    assert set(compact_context.keys()) == {
        "dataset_type",
        "top_kpi_facts",
        "top_diagnostic_findings",
        "predictive_summary",
        "top_recommendations",
    }
    assert len(compact_context["top_kpi_facts"]) <= 3
    assert len(compact_context["top_diagnostic_findings"]) <= 3
    assert len(compact_context["top_recommendations"]) <= 3


def test_generation_bundle_uses_compact_context_for_quick_prompt():
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

    bundle = build_ai_generation_bundle(
        dataset_summary=dataset_summary,
        kpis={"total_rows": 90, "total_columns": 4, "duplicate_rows": 0, "columns_with_missing_values": 0},
        duplicate_rows=duplicate_rows,
        missing_value_summary=missing_value_summary,
        column_profiles=column_profiles,
        preprocessing_summary=preprocessing_summary,
        dataset_intelligence=dataset_intelligence,
        dataframe=dataframe,
        instruction="Key Business Insights",
    )

    assert bundle["used_compact_context"] is True
    assert bundle["context"] == bundle["compact_context"]
    fallback_summary = build_deterministic_fallback_summary(bundle["compact_context"])
    assert "Compact fallback summary" in fallback_summary


def test_prepare_ai_summary_cleans_and_structures_output():
    prepared = prepare_ai_summary(
        "ExecutiveSummary:\n"
        "- revenue remains stable.\n"
        "- revenue remains stable.\n\n"
        "Risks:\n"
        "* margin pressure may persist..\n"
        "RecommendedActions:\n"
        "- protect pricing discipline\n"
    )

    assert prepared["quality"]["passes"] is True
    assert "Executive Summary" in prepared["items"]
    assert "Revenue remains stable." in prepared["items"]
    assert "Margin pressure may persist." in prepared["items"]
    assert "Protect pricing discipline" in prepared["items"]
