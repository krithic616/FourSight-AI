"""Tab helpers for the Streamlit interface."""

from __future__ import annotations

import pandas as pd
import streamlit as st

try:
    from analytics.diagnostic import get_diagnostic_options
    from analytics.prescriptive import get_prescriptive_options
    from analytics.predictive import get_predictive_options
    from ui.components import (
        render_ai_analyst,
        render_cleaning_log,
        render_download_report,
        render_dataset_intelligence,
        render_diagnostic_analytics,
        render_prescriptive_analytics,
        render_predictive_analytics,
        render_column_profile_groups,
        render_data_preview,
        render_dataset_overview,
        render_duplicate_summary,
        render_kpi_overview,
        render_missing_values_table,
        render_preprocessing_summary,
    )
except ModuleNotFoundError:
    from app.analytics.diagnostic import get_diagnostic_options
    from app.analytics.prescriptive import get_prescriptive_options
    from app.analytics.predictive import get_predictive_options
    from app.ui.components import (
        render_ai_analyst,
        render_cleaning_log,
        render_download_report,
        render_dataset_intelligence,
        render_diagnostic_analytics,
        render_prescriptive_analytics,
        render_predictive_analytics,
        render_column_profile_groups,
        render_data_preview,
        render_dataset_overview,
        render_duplicate_summary,
        render_kpi_overview,
        render_missing_values_table,
        render_preprocessing_summary,
    )


def get_default_tabs(
    include_diagnostic: bool = False,
    include_predictive: bool = False,
    include_prescriptive: bool = False,
) -> list[str]:
    """Return the current set of app tabs."""
    tabs = [
        "Data Quality",
        "Column Profiling",
        "Dataset Intelligence",
        "Descriptive Analytics",
    ]
    if include_diagnostic:
        tabs.append("Diagnostic Analytics")
    if include_predictive:
        tabs.append("Predictive Analytics")
    if include_prescriptive:
        tabs.append("Prescriptive Analytics")
    tabs.append("AI Analyst")
    tabs.append("Download Report")
    return tabs


def render_phase_one_tabs(
    dataset_summary: dict[str, object],
    kpis: dict[str, int],
    dataframe: pd.DataFrame,
    duplicate_rows: int,
    missing_value_summary: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    preprocessing_summary: dict[str, object],
    cleaning_log: list[str],
    dataset_intelligence: dict[str, object],
) -> None:
    """Render the Phase 1 analytics workspace."""
    diagnostic_options = get_diagnostic_options(
        dataframe=dataframe,
        column_profiles=column_profiles,
        readiness=dataset_intelligence["readiness"]["diagnostic"],
    )
    predictive_options = get_predictive_options(
        dataframe=dataframe,
        column_profiles=column_profiles,
        readiness=dataset_intelligence["readiness"]["predictive"],
    )
    prescriptive_options = get_prescriptive_options(
        dataframe=dataframe,
        column_profiles=column_profiles,
        dataset_intelligence=dataset_intelligence,
    )
    tabs = st.tabs(
        get_default_tabs(
            include_diagnostic=diagnostic_options["enabled"],
            include_predictive=predictive_options["enabled"],
            include_prescriptive=prescriptive_options["enabled"],
        )
    )
    quality_tab, profiling_tab, intelligence_tab, descriptive_tab = tabs[:4]
    next_tab_index = 4
    diagnostic_tab = tabs[next_tab_index] if diagnostic_options["enabled"] else None
    if diagnostic_tab is not None:
        next_tab_index += 1
    predictive_tab = tabs[next_tab_index] if predictive_options["enabled"] else None
    if predictive_tab is not None:
        next_tab_index += 1
    prescriptive_tab = tabs[next_tab_index] if prescriptive_options["enabled"] else None
    if prescriptive_tab is not None:
        next_tab_index += 1
    ai_analyst_tab = tabs[next_tab_index]
    next_tab_index += 1
    download_report_tab = tabs[next_tab_index]

    with descriptive_tab:
        st.subheader("Descriptive Analytics")
        render_kpi_overview(kpis)
        st.subheader("Dataset Overview")
        render_dataset_overview(dataset_summary)

        st.subheader("Data Preview")
        render_data_preview(dataframe)

    with quality_tab:
        st.subheader("Data Quality")
        render_duplicate_summary(duplicate_rows)
        render_missing_values_table(missing_value_summary)

        st.subheader("Cleaning Log")
        render_cleaning_log(cleaning_log)

    with profiling_tab:
        st.subheader("Column Profiling")
        render_column_profile_groups(column_profiles)

        st.subheader("Preprocessing Summary")
        render_preprocessing_summary(preprocessing_summary)

    with intelligence_tab:
        st.subheader("Dataset Intelligence")
        render_dataset_intelligence(dataset_intelligence)

    if diagnostic_tab is not None:
        with diagnostic_tab:
            st.subheader("Diagnostic Analytics")
            render_diagnostic_analytics(
                dataframe=dataframe,
                column_profiles=column_profiles,
                dataset_intelligence=dataset_intelligence,
            )

    if predictive_tab is not None:
        with predictive_tab:
            st.subheader("Predictive Analytics")
            render_predictive_analytics(
                dataframe=dataframe,
                column_profiles=column_profiles,
                dataset_intelligence=dataset_intelligence,
            )

    if prescriptive_tab is not None:
        with prescriptive_tab:
            st.subheader("Prescriptive Analytics")
            render_prescriptive_analytics(
                dataframe=dataframe,
                column_profiles=column_profiles,
                dataset_intelligence=dataset_intelligence,
            )

    with ai_analyst_tab:
        st.subheader("AI Analyst")
        render_ai_analyst(
            dataset_summary=dataset_summary,
            kpis=kpis,
            dataframe=dataframe,
            duplicate_rows=duplicate_rows,
            missing_value_summary=missing_value_summary,
            column_profiles=column_profiles,
            preprocessing_summary=preprocessing_summary,
            dataset_intelligence=dataset_intelligence,
        )

    with download_report_tab:
        st.subheader("Download Report")
        render_download_report(
            dataset_summary=dataset_summary,
            kpis=kpis,
            dataframe=dataframe,
            duplicate_rows=duplicate_rows,
            missing_value_summary=missing_value_summary,
            column_profiles=column_profiles,
            preprocessing_summary=preprocessing_summary,
            dataset_intelligence=dataset_intelligence,
        )
