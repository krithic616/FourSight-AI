"""Reusable UI component helpers."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from ai.insight_generator import build_ai_generation_bundle, generate_insight
    from ai.instruction_handler import DEFAULT_AI_INSTRUCTION
    from ai.ollama_client import get_ollama_status
    from ai.report_writer import build_short_ai_summary, prepare_ai_summary
    from analytics.diagnostic import get_diagnostic_options, run_diagnostic_analysis
    from analytics.prescriptive import build_prescriptive_analysis, get_prescriptive_options
    from analytics.predictive import get_predictive_options, run_predictive_forecast
    from reporting.export_html import export_html
    from reporting.report_builder import build_report, build_report_preview, export_txt
    from utils.helpers import summarize_column_list
except ModuleNotFoundError:
    from app.ai.insight_generator import build_ai_generation_bundle, generate_insight
    from app.ai.instruction_handler import DEFAULT_AI_INSTRUCTION
    from app.ai.ollama_client import get_ollama_status
    from app.ai.report_writer import build_short_ai_summary, prepare_ai_summary
    from app.analytics.diagnostic import get_diagnostic_options, run_diagnostic_analysis
    from app.analytics.prescriptive import build_prescriptive_analysis, get_prescriptive_options
    from app.analytics.predictive import get_predictive_options, run_predictive_forecast
    from app.reporting.export_html import export_html
    from app.reporting.report_builder import build_report, build_report_preview, export_txt
    from app.utils.helpers import summarize_column_list


def get_status_message() -> str:
    """Return a default status message."""
    return "Components ready"


def render_kpi_overview(kpis: dict[str, int]) -> None:
    """Render the top-level KPI cards."""
    labels = [
        ("Total Rows", "total_rows"),
        ("Total Columns", "total_columns"),
        ("Duplicate Rows", "duplicate_rows"),
        ("Columns With Missing Values", "columns_with_missing_values"),
    ]
    columns = st.columns(len(labels))
    for column, (label, key) in zip(columns, labels):
        column.metric(label, kpis.get(key, 0))


def render_dataset_overview(summary: dict[str, object]) -> None:
    """Render dataset metadata details."""
    metadata_columns = st.columns(4)
    metadata_columns[0].write(f"File name: `{summary['file_name']}`")
    metadata_columns[1].write(f"File size: `{summary['file_size_display']}`")
    metadata_columns[2].write(f"Rows: `{summary['row_count']}`")
    metadata_columns[3].write(f"Columns: `{summary['column_count']}`")


def render_data_preview(dataframe: pd.DataFrame) -> None:
    """Render a preview of the uploaded dataset."""
    st.dataframe(dataframe.head(10), use_container_width=True)


def render_missing_values_table(summary: pd.DataFrame) -> None:
    """Render the missing value summary table."""
    if summary.empty:
        st.success("No missing values detected.")
        return
    st.dataframe(summary, use_container_width=True, hide_index=True)


def render_column_profile_groups(column_profiles: dict[str, list[str]]) -> None:
    """Render grouped column profiling results."""
    groups = [
        ("Numeric Columns", column_profiles["numeric"]),
        ("Categorical Columns", column_profiles["categorical"]),
        ("Datetime Columns", column_profiles["datetime"]),
        ("Other Columns", column_profiles["other"]),
    ]
    left_column, right_column = st.columns(2)
    layout_columns = [left_column, right_column, left_column, right_column]

    for container_column, (title, values) in zip(layout_columns, groups):
        with container_column:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(summarize_column_list(values))


def render_preprocessing_summary(summary: dict[str, object]) -> None:
    """Render preprocessing observations collected during profiling."""
    st.write(f"Input rows profiled: `{summary['input_rows']}`")
    st.write(f"Input columns profiled: `{summary['input_columns']}`")
    st.write(f"Duplicate rows detected: `{summary['duplicate_rows_detected']}`")
    st.write(
        f"Columns with missing values: `{summary['columns_with_missing_values']}`"
    )
    st.write("Profiling actions performed:")
    for action in summary["profiling_actions"]:
        st.write(f"- {action}")


def render_cleaning_log(log_messages: list[str]) -> None:
    """Render the Phase 1 cleaning log."""
    for message in log_messages:
        st.write(f"- {message}")


def render_duplicate_summary(duplicate_rows: int) -> None:
    """Render duplicate row findings."""
    if duplicate_rows == 0:
        st.success("No duplicate rows detected.")
        return
    st.warning(f"Detected `{duplicate_rows}` duplicate row(s).")


def render_dataset_intelligence(intelligence: dict[str, object]) -> None:
    """Render detected dataset intelligence and analytics readiness."""
    with st.container(border=True):
        st.markdown("**Detected Dataset Type**")
        st.write(str(intelligence["dataset_type"]).replace("_", " ").title())

    with st.container(border=True):
        st.markdown("**Key Detected Business Signals**")
        for signal in intelligence["business_signals"]:
            st.write(f"- {signal}")

    st.markdown("**Analytics Readiness**")
    readiness = intelligence["readiness"]
    for key in ["descriptive", "diagnostic", "predictive", "prescriptive"]:
        details = readiness[key]
        with st.container(border=True):
            st.write(f"{key.replace('_', ' ').title()}: `{details['status']}`")
            st.write(details["reason"])


def render_diagnostic_analytics(
    dataframe: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    dataset_intelligence: dict[str, object],
) -> None:
    """Render the Phase 2 deterministic diagnostic analytics workspace."""
    diagnostic_readiness = dataset_intelligence["readiness"]["diagnostic"]
    options = get_diagnostic_options(
        dataframe=dataframe,
        column_profiles=column_profiles,
        readiness=diagnostic_readiness,
    )

    if not options["enabled"]:
        st.info(
            "Diagnostic Analytics becomes available when there is at least one usable metric and one usable dimension for grouped comparison."
        )
        return

    st.caption(diagnostic_readiness["reason"])

    selector_columns = st.columns(3)
    metric_column = selector_columns[0].selectbox(
        "Metric",
        options["metric_columns"],
        help="Choose a numeric business field for grouped analysis.",
    )
    dimension_column = selector_columns[1].selectbox(
        "Dimension",
        options["dimension_columns"],
        help="Choose a categorical field for segment comparison.",
    )
    date_options = ["None"] + options["datetime_columns"]
    default_date_index = 0
    if options["default_date_column"] in date_options:
        default_date_index = date_options.index(options["default_date_column"])
    selected_date_option = selector_columns[2].selectbox(
        "Date Column",
        date_options,
        index=default_date_index,
        help="Optionally add a time trend if a usable datetime field exists.",
    )
    date_column = None if selected_date_option == "None" else selected_date_option

    breakdown_dimension = False
    if date_column:
        breakdown_dimension = st.toggle(
            "Break down the trend by the selected dimension when feasible",
            value=False,
        )

    analysis = run_diagnostic_analysis(
        dataframe=dataframe,
        metric_column=metric_column,
        dimension_column=dimension_column,
        date_column=date_column,
        breakdown_dimension=breakdown_dimension,
    )

    summary = analysis["summary"]
    summary_columns = st.columns(5)
    summary_columns[0].metric("Total Metric", f"{summary['total_metric_value']:,.2f}")
    summary_columns[1].metric("Groups", summary["group_count"])
    summary_columns[2].metric("Top Group", summary["top_group"], f"{summary['top_group_value']:,.2f}")
    summary_columns[3].metric(
        summary["lowest_group_label"],
        summary["bottom_group"],
        f"{summary['bottom_group_value']:,.2f}",
    )
    summary_columns[4].metric(
        "Top-Bottom Gap",
        f"{summary['difference_top_to_bottom']:,.2f}",
    )
    st.caption(
        f"Spread: `{summary['spread_classification'].title()}` | Concentration: `{summary['concentration_classification'].title()}`"
    )

    grouped_comparison = analysis["grouped_comparison"]
    st.subheader("Grouped Comparison")
    if grouped_comparison.empty:
        st.warning("No grouped comparison is available for the selected metric and dimension.")
    else:
        comparison_chart = px.bar(
            grouped_comparison,
            x="group",
            y="metric_value",
            color="metric_value",
            color_continuous_scale="Blues",
            labels={"group": dimension_column, "metric_value": metric_column},
        )
        comparison_chart.update_traces(
            hovertemplate=(
                f"{dimension_column}: %{{x}}<br>"
                f"{metric_column}: %{{y:,.2f}}<br>"
                "Rows: %{customdata[0]}<br>"
                "Share: %{customdata[1]:.2f}%<extra></extra>"
            ),
            customdata=grouped_comparison[["row_count", "share_pct"]].to_numpy(),
        )
        comparison_chart.update_layout(
            xaxis_title=dimension_column,
            yaxis_title=metric_column,
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis={"categoryorder": "array", "categoryarray": grouped_comparison["group"].tolist()},
        )
        st.plotly_chart(comparison_chart, use_container_width=True)
        st.dataframe(grouped_comparison, use_container_width=True, hide_index=True)

    if analysis["show_split_top_bottom"]:
        top_bottom_columns = st.columns(2)
        with top_bottom_columns[0]:
            st.subheader("Top 5 Groups")
            st.dataframe(analysis["top_groups"], use_container_width=True, hide_index=True)
        with top_bottom_columns[1]:
            st.subheader("Bottom 5 Groups")
            st.dataframe(analysis["bottom_groups"], use_container_width=True, hide_index=True)
    else:
        st.subheader("Ranked Group Summary")
        st.dataframe(analysis["ranked_groups"], use_container_width=True, hide_index=True)

    if analysis["trend_analysis"]:
        st.subheader("Trend Analysis")
        trend_analysis = analysis["trend_analysis"]
        trend_summary = trend_analysis["summary"]
        if trend_analysis["by_dimension"] is not None:
            trend_chart = px.line(
                trend_analysis["by_dimension"],
                x="period",
                y="metric_value",
                color="group",
                labels={"period": "Period", "metric_value": metric_column, "group": dimension_column},
            )
        else:
            trend_chart = px.line(
                trend_summary,
                x="period",
                y="metric_value",
                labels={"period": "Period", "metric_value": metric_column},
            )
        trend_chart.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Period",
            yaxis_title=metric_column,
        )
        st.plotly_chart(trend_chart, use_container_width=True)
        st.caption(
            f"Trend aggregation grain: `{trend_analysis['frequency']}`"
        )
        if date_column and breakdown_dimension and trend_analysis["by_dimension"] is None:
            st.caption(
                "Dimension breakdown was skipped because the selected dimension has too many groups or too few usable trend points."
            )

    st.subheader("Diagnostic Findings")
    for finding in analysis["findings"]:
        st.write(f"- {finding}")


def render_predictive_analytics(
    dataframe: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    dataset_intelligence: dict[str, object],
) -> None:
    """Render the cautious Phase 3 predictive forecasting workspace."""
    predictive_readiness = dataset_intelligence["readiness"]["predictive"]
    options = get_predictive_options(
        dataframe=dataframe,
        column_profiles=column_profiles,
        readiness=predictive_readiness,
    )

    if not options["enabled"]:
        st.info(
            "Predictive Analytics becomes available only when the dataset shows a usable time-series structure for cautious forecasting."
        )
        return

    st.caption(predictive_readiness["reason"])

    selector_columns = st.columns(4)
    metric_index = 0
    if options["default_metric_column"] in options["metric_columns"]:
        metric_index = options["metric_columns"].index(options["default_metric_column"])
    metric_column = selector_columns[0].selectbox(
        "Forecast Metric",
        options["metric_columns"],
        index=metric_index,
        help="Choose a numeric business metric to aggregate over time.",
    )
    datetime_index = 0
    if options["default_datetime_column"] in options["datetime_columns"]:
        datetime_index = options["datetime_columns"].index(options["default_datetime_column"])
    datetime_column = selector_columns[1].selectbox(
        "Datetime Column",
        options["datetime_columns"],
        index=datetime_index,
        help="Choose the datetime column used to build the historical time series.",
    )
    aggregation_grain = selector_columns[2].selectbox(
        "Aggregation Grain",
        options["grain_options"],
        help="Use Auto to let the app choose a conservative grain from the data depth.",
    )
    forecast_horizon = selector_columns[3].selectbox(
        "Forecast Horizon",
        options["horizon_options"],
        index=1,
        help="Choose how many future periods to project.",
    )

    forecast = run_predictive_forecast(
        dataframe=dataframe,
        metric_column=metric_column,
        datetime_column=datetime_column,
        aggregation_grain=aggregation_grain,
        forecast_horizon=int(forecast_horizon),
        readiness_status=options["readiness_status"],
        readiness_reason=options["readiness_reason"],
    )

    st.subheader("Forecast Readiness")
    status_text = forecast["status"]
    with st.container(border=True):
        if status_text == "Ready":
            st.success(f"Predictive Readiness: {status_text}")
        elif status_text == "Conditional":
            st.warning(f"Predictive Readiness: {status_text}")
        else:
            st.error(f"Predictive Readiness: {status_text}")
        st.write(forecast["status_reason"])
        st.caption("Reasons")
        for reason in forecast["readiness_reasons"]:
            st.write(f"- {reason}")
        if status_text == "Not Ready":
            st.caption("Validation limitations")
            for message in forecast["validation_messages"]:
                st.write(f"- {message}")

    st.subheader("Forecast Input Summary")
    input_summary = forecast["input_summary"]
    input_columns = st.columns(4)
    input_columns[0].write(f"Metric: `{metric_column}`")
    input_columns[1].write(f"Datetime: `{datetime_column}`")
    input_columns[2].write(f"Aggregation: `{input_summary['aggregation_grain_label']}`")
    input_columns[3].write(f"Horizon: `{input_summary['forecast_horizon']}`")
    st.caption(f"Forecast method: `{input_summary['forecast_method_label']}`")
    st.write(
        f"Datetime parse success: `{input_summary['datetime_parse_success_rate']:.0%}` | Metric valid ratio: `{input_summary['metric_valid_ratio']:.0%}` | Historical periods: `{input_summary['historical_periods']}`"
    )

    summary_cards = forecast["summary_cards"]
    card_columns = st.columns(6)
    card_columns[0].metric("Historical Periods", summary_cards["historical_periods_used"])
    card_columns[1].metric("Forecast Horizon", summary_cards["forecast_horizon"])
    card_columns[2].metric("Latest Actual", f"{summary_cards['latest_actual_value']:,.2f}")
    card_columns[3].metric("First Forecast", f"{summary_cards['first_forecast_value']:,.2f}")
    card_columns[4].metric(
        "Projected Avg",
        f"{summary_cards['projected_average_forecast_value']:,.2f}",
    )
    card_columns[5].metric("Direction", summary_cards["projected_direction"])

    historical_trend = forecast["historical_trend"]
    st.subheader("Historical Trend")
    if historical_trend.empty:
        st.info("No historical trend is available for the selected forecast inputs.")
    else:
        historical_chart = px.line(
            historical_trend,
            x="period",
            y="metric_value",
            labels={"period": "Period", "metric_value": metric_column},
        )
        historical_chart.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Period",
            yaxis_title=metric_column,
        )
        st.plotly_chart(historical_chart, use_container_width=True)

    st.subheader("Forecast Chart")
    forecast_chart = forecast["forecast_chart"]
    if forecast_chart.empty:
        st.info("No forecast chart is available because forecasting did not pass validation.")
    else:
        combined_chart = px.line(
            forecast_chart,
            x="period",
            y="metric_value",
            color="series_type",
            labels={"period": "Period", "metric_value": metric_column, "series_type": "Series"},
        )
        actual_period_end = historical_trend["period"].max() if not historical_trend.empty else None
        combined_chart.update_traces(
            line={"dash": "solid"},
            selector={"name": "Actual"},
        )
        combined_chart.update_traces(
            line={"dash": "dash"},
            selector={"name": "Forecast"},
        )
        if actual_period_end is not None:
            combined_chart.add_vline(
                x=actual_period_end,
                line_dash="dot",
                line_color="#B0B0B0",
                opacity=0.8,
            )
            combined_chart.add_vrect(
                x0=actual_period_end,
                x1=forecast_chart["period"].max(),
                fillcolor="rgba(120, 170, 255, 0.08)",
                line_width=0,
                layer="below",
            )
        combined_chart.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Period",
            yaxis_title=metric_column,
        )
        st.plotly_chart(combined_chart, use_container_width=True)

    st.subheader("Predicted Next Periods")
    st.dataframe(forecast["forecast_table"], use_container_width=True, hide_index=True)

    st.subheader("Predictive Findings")
    for finding in forecast["findings"]:
        st.write(f"- {finding}")


def render_prescriptive_analytics(
    dataframe: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    dataset_intelligence: dict[str, object],
) -> None:
    """Render the deterministic Phase 4 prescriptive analytics workspace."""
    options = get_prescriptive_options(
        dataframe=dataframe,
        column_profiles=column_profiles,
        dataset_intelligence=dataset_intelligence,
    )
    prescriptive_readiness = options["prescriptive_readiness"]

    if not options["enabled"]:
        st.info(
            "Prescriptive Analytics becomes available when the dataset has enough diagnostic or predictive support to justify evidence-based actions."
        )
        st.caption(prescriptive_readiness["reason"])
        return

    analysis = build_prescriptive_analysis(
        dataframe=dataframe,
        column_profiles=column_profiles,
        dataset_intelligence=dataset_intelligence,
    )

    st.subheader("Prescriptive Readiness")
    with st.container(border=True):
        status = analysis["status"]
        if status == "Ready":
            st.success(f"Prescriptive Readiness: {status}")
        elif status == "Conditional":
            st.warning(f"Prescriptive Readiness: {status}")
        else:
            st.error(f"Prescriptive Readiness: {status}")
        st.write(analysis["status_reason"])
        for limitation in analysis["limitation_messages"]:
            st.write(f"- {limitation}")

    st.subheader("Prescriptive Summary")
    summary = analysis["summary"]
    summary_columns = st.columns(3)
    summary_columns[0].metric("Recommendations", summary["total_recommendations"])
    summary_columns[1].metric(
        "Diagnostic-Based",
        summary["basis_mix"].get("diagnostic-based", 0),
    )
    summary_columns[2].metric(
        "Predictive-Based",
        summary["basis_mix"].get("predictive-based", 0),
    )
    with st.container(border=True):
        st.markdown("**Highest-Priority Issue**")
        st.write(summary["highest_priority_issue"])
        st.write(f"Combined basis recommendations: `{summary['basis_mix'].get('combined', 0)}`")

    category_columns = st.columns(4)
    category_columns[0].write(f"Growth Opportunities: `{summary['category_counts'].get('Growth Opportunities', 0)}`")
    category_columns[1].write(f"Risk Controls: `{summary['category_counts'].get('Risk Controls', 0)}`")
    category_columns[2].write(f"Efficiency Improvements: `{summary['category_counts'].get('Efficiency Improvements', 0)}`")
    category_columns[3].write(f"Monitoring Priorities: `{summary['category_counts'].get('Monitoring Priorities', 0)}`")

    st.subheader("Recommendations")
    if not analysis["recommendations"]:
        st.info("No prescriptive recommendation was triggered from the currently available evidence.")
    else:
        for recommendation in analysis["recommendations"]:
            with st.container(border=True):
                st.markdown(
                    f"**{recommendation['category']}** | Priority: `{recommendation['priority']}` | Basis: `{recommendation['recommendation_basis']}`"
                )
                st.markdown("**Issue**")
                st.write(recommendation["issue_detected"])
                st.markdown("**Affected Group**")
                st.write(recommendation["affected_group"])
                st.markdown("**Supporting Evidence**")
                st.write(recommendation["supporting_evidence"])
                st.markdown("**Suggested Action**")
                st.write(recommendation["suggested_action"])

    st.subheader("Prescriptive Findings")
    for finding in analysis["findings"]:
        st.write(f"- {finding}")


def render_ai_analyst(
    dataset_summary: dict[str, object],
    kpis: dict[str, int],
    dataframe: pd.DataFrame,
    duplicate_rows: int,
    missing_value_summary: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    preprocessing_summary: dict[str, object],
    dataset_intelligence: dict[str, object],
) -> None:
    """Render the grounded local AI analyst workspace."""
    if "ai_instruction_pending" in st.session_state:
        st.session_state["ai_instruction"] = st.session_state.pop("ai_instruction_pending")

    ollama_status = get_ollama_status()
    default_model = str(ollama_status.get("default_model", "phi3:mini"))
    model_options = [default_model] + [
        model_name
        for model_name in ollama_status.get("models", [])
        if model_name != default_model
    ]

    if "ai_model_name" not in st.session_state:
        st.session_state["ai_model_name"] = default_model
    if "ai_instruction" not in st.session_state:
        st.session_state["ai_instruction"] = DEFAULT_AI_INSTRUCTION
    if "ai_blocked_model" not in st.session_state:
        st.session_state["ai_blocked_model"] = ""
    if "ai_blocked_reason" not in st.session_state:
        st.session_state["ai_blocked_reason"] = ""

    st.subheader("Ollama Status")
    with st.container(border=True):
        if ollama_status["available"]:
            st.success(ollama_status["message"])
        else:
            st.warning(ollama_status["message"])
            st.write("Start the local Ollama service to enable grounded AI answers. The rest of the analytics app remains fully available.")
        st.caption(
            "Low-resource note: smaller local models and concise prompts usually respond more reliably on lower-memory systems."
        )

    st.subheader("Model Selector")
    model_columns = st.columns(2)
    suggested_model = model_columns[0].selectbox(
        "Available Models",
        model_options,
        index=model_options.index(st.session_state["ai_model_name"])
        if st.session_state["ai_model_name"] in model_options
        else 0,
        help="Choose a detected local Ollama model or use the custom model field.",
    )
    if suggested_model != st.session_state["ai_model_name"]:
        st.session_state["ai_model_name"] = suggested_model
    selected_model = model_columns[1].text_input(
        "Model Name",
        key="ai_model_name",
        help="Type a local model name if you want to override the detected list.",
    )
    if selected_model != st.session_state.get("ai_blocked_model", ""):
        st.session_state["ai_blocked_model"] = ""
        st.session_state["ai_blocked_reason"] = ""
    st.caption(f"Current model: `{selected_model}`")
    if st.session_state.get("ai_blocked_model") == selected_model and st.session_state.get("ai_blocked_reason"):
        st.warning(st.session_state["ai_blocked_reason"])

    st.subheader("Response Mode Selector")
    response_mode = st.selectbox(
        "Response Mode",
        ["Executive", "Analyst", "Action Focus"],
        help="Control the tone and structure of the grounded AI response.",
    )

    st.subheader("Instruction Bar")
    instruction = st.text_area(
        "Ask the AI Analyst",
        key="ai_instruction",
        height=120,
        help="Examples: Summarize the key business insights. What are the biggest risks in this dataset? What actions should leadership take next?",
    )
    st.caption(
        "Tip: concise prompts work better on low-resource machines. Try: `Give 5 key business insights`, `What are the top risks?`, or `What should leadership focus on next?`"
    )
    quick_prompt_columns = st.columns(3)
    if quick_prompt_columns[0].button("Key Business Insights", use_container_width=True):
        st.session_state["ai_instruction_pending"] = "Give 5 key business insights."
        st.rerun()
    if quick_prompt_columns[1].button("Top Risks", use_container_width=True):
        st.session_state["ai_instruction_pending"] = "What are the top risks?"
        st.rerun()
    if quick_prompt_columns[2].button("Recommended Actions", use_container_width=True):
        st.session_state["ai_instruction_pending"] = "What should leadership focus on next?"
        st.rerun()

    generate_clicked = st.button("Generate Insight", use_container_width=True)

    st.subheader("AI Response Output Panel")
    if generate_clicked:
        generation_bundle = build_ai_generation_bundle(
            dataset_summary=dataset_summary,
            kpis=kpis,
            duplicate_rows=duplicate_rows,
            missing_value_summary=missing_value_summary,
            column_profiles=column_profiles,
            preprocessing_summary=preprocessing_summary,
            dataset_intelligence=dataset_intelligence,
            dataframe=dataframe,
            instruction=instruction,
        )

        if not ollama_status["available"]:
            st.info("AI Analyst is currently unavailable because Ollama is not reachable locally. Start Ollama and try again.")
            return
        if st.session_state.get("ai_blocked_model") == selected_model and st.session_state.get("ai_blocked_reason"):
            st.warning(st.session_state["ai_blocked_reason"])
            with st.container(border=True):
                st.markdown(generation_bundle["fallback_summary"])
            return

        result = generate_insight(
            model_name=selected_model,
            response_mode=response_mode,
            instruction=instruction,
            context=generation_bundle["context"],
        )
        if not result["success"]:
            if result.get("error_type") == "insufficient_memory":
                st.session_state["ai_blocked_model"] = selected_model
                st.session_state["ai_blocked_reason"] = result["error"]
            st.warning(result["error"])
            with st.container(border=True):
                st.markdown(generation_bundle["fallback_summary"])
            return
        st.session_state["ai_blocked_model"] = ""
        st.session_state["ai_blocked_reason"] = ""
        prepared_summary = prepare_ai_summary(result["response"])
        display_items = prepared_summary["items"]
        if not prepared_summary["quality"]["passes"]:
            st.warning(
                "The AI output was generated, but it did not meet the report-quality threshold. A shorter cleaned version is shown instead."
            )
            display_items = build_short_ai_summary(display_items)
        cleaned_display = "\n".join(f"- {item}" for item in display_items)
        st.session_state["ai_last_response"] = prepared_summary["cleaned_text"]
        with st.container(border=True):
            st.markdown(cleaned_display or "No response was returned by the local model.")
        return

    st.info("Choose a model, set the response mode, enter an instruction, and click Generate Insight.")


def render_download_report(
    dataset_summary: dict[str, object],
    kpis: dict[str, int],
    dataframe: pd.DataFrame,
    duplicate_rows: int,
    missing_value_summary: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    preprocessing_summary: dict[str, object],
    dataset_intelligence: dict[str, object],
) -> None:
    """Render the deterministic report preview and download workspace."""
    ai_response = str(st.session_state.get("ai_last_response", "")).strip()
    report = build_report(
        dataset_summary=dataset_summary,
        kpis=kpis,
        dataframe=dataframe,
        duplicate_rows=duplicate_rows,
        missing_value_summary=missing_value_summary,
        column_profiles=column_profiles,
        preprocessing_summary=preprocessing_summary,
        dataset_intelligence=dataset_intelligence,
        ai_response=ai_response,
    )
    txt_report = export_txt(report)
    html_report = export_html(report)

    st.subheader("Report Preview")
    st.caption(
        "This report is generated deterministically from the app's computed analytics outputs and will still work even if AI insight generation is unavailable."
    )
    st.text_area(
        "Preview",
        value=build_report_preview(report),
        height=360,
        disabled=True,
    )

    st.subheader("Download Options")
    download_columns = st.columns(2)
    download_columns[0].download_button(
        "Download TXT Report",
        data=txt_report,
        file_name="foursight_business_insight_report.txt",
        mime="text/plain",
        use_container_width=True,
    )
    download_columns[1].download_button(
        "Download HTML Report",
        data=html_report,
        file_name="foursight_business_insight_report.html",
        mime="text/html",
        use_container_width=True,
    )
