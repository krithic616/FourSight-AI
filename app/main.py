"""Streamlit entrypoint for FourSight AI."""

from __future__ import annotations

import streamlit as st

try:
    from config import APP_NAME, APP_PAGE_TITLE
    from analytics.kpi_engine import compute_phase_one_kpis
    from core.cleaning_log import create_cleaning_log
    from core.data_cleaner import (
        build_missing_value_summary,
        build_preprocessing_summary,
        count_duplicate_rows,
    )
    from core.dataset_detector import build_dataset_intelligence
    from core.data_profiler import build_dataset_summary, profile_column_types
    from core.file_loader import get_uploaded_file_metadata, load_csv_file
    from ui.layout import get_layout_name, render_header
    from ui.tabs import render_phase_one_tabs
except ModuleNotFoundError:
    from app.config import APP_NAME, APP_PAGE_TITLE
    from app.analytics.kpi_engine import compute_phase_one_kpis
    from app.core.cleaning_log import create_cleaning_log
    from app.core.data_cleaner import (
        build_missing_value_summary,
        build_preprocessing_summary,
        count_duplicate_rows,
    )
    from app.core.dataset_detector import build_dataset_intelligence
    from app.core.data_profiler import build_dataset_summary, profile_column_types
    from app.core.file_loader import get_uploaded_file_metadata, load_csv_file
    from app.ui.layout import get_layout_name, render_header
    from app.ui.tabs import render_phase_one_tabs


PROJECT_DESCRIPTION = (
    "Phase 1 initializes the analytics workspace with safe CSV ingestion, "
    "dataset profiling, and data quality visibility."
)


def main() -> None:
    """Render the initial application shell."""
    st.set_page_config(page_title=APP_PAGE_TITLE, layout=get_layout_name())
    render_header(APP_NAME, PROJECT_DESCRIPTION)

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file to begin the Phase 1 analytics workflow.")
        return

    try:
        dataframe = load_csv_file(uploaded_file)
    except Exception as exc:
        st.error(f"Unable to load the uploaded CSV: {exc}")
        return

    dataset_summary = build_dataset_summary(dataframe)
    dataset_summary.update(get_uploaded_file_metadata(uploaded_file))
    duplicate_rows = count_duplicate_rows(dataframe)
    missing_value_summary = build_missing_value_summary(dataframe)
    column_profiles = profile_column_types(dataframe)
    dataset_intelligence = build_dataset_intelligence(dataframe, column_profiles)
    preprocessing_summary = build_preprocessing_summary(
        dataframe=dataframe,
        duplicate_rows=duplicate_rows,
        missing_value_summary=missing_value_summary,
    )
    cleaning_log = create_cleaning_log(
        duplicate_rows=duplicate_rows,
        missing_value_summary=missing_value_summary,
        column_profiles=column_profiles,
    )
    kpis = compute_phase_one_kpis(
        dataframe=dataframe,
        duplicate_rows=duplicate_rows,
        missing_value_summary=missing_value_summary,
    )

    render_phase_one_tabs(
        dataset_summary=dataset_summary,
        kpis=kpis,
        dataframe=dataframe,
        duplicate_rows=duplicate_rows,
        missing_value_summary=missing_value_summary,
        column_profiles=column_profiles,
        preprocessing_summary=preprocessing_summary,
        cleaning_log=cleaning_log,
        dataset_intelligence=dataset_intelligence,
    )


if __name__ == "__main__":
    main()
