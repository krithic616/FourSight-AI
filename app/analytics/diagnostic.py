"""Deterministic diagnostic analytics helpers."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


MAX_DIMENSION_GROUPS = 24
MAX_TREND_BREAKDOWN_GROUPS = 8

ID_KEYWORDS = {
    "id",
    "key",
    "code",
    "number",
    "num",
    "customer_id",
    "order_id",
    "invoice_id",
    "transaction_id",
    "account_id",
    "user_id",
}
BUSINESS_METRIC_KEYWORDS = {
    "quantity",
    "qty",
    "price",
    "amount",
    "revenue",
    "sales",
    "cost",
    "profit",
    "margin",
    "discount",
    "score",
    "rating",
    "value",
    "units",
    "total",
}
DISCOUNT_KEYWORDS = {"discount", "discount_pct", "discount_rate", "markdown"}
PROFIT_KEYWORDS = {"profit", "margin", "contribution"}


def get_diagnostic_options(
    dataframe: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    readiness: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return usable diagnostic selector options and tab availability."""
    metric_columns = _get_metric_columns(dataframe, column_profiles.get("numeric", []))
    dimension_columns = _get_dimension_columns(
        dataframe=dataframe,
        categorical_columns=column_profiles.get("categorical", []),
    )
    datetime_columns = _get_datetime_columns(dataframe, column_profiles.get("datetime", []))
    readiness_status = (readiness or {}).get("status", "Not Ready")
    enabled = readiness_status in {"Ready", "Conditional"} and bool(metric_columns) and bool(
        dimension_columns
    )

    return {
        "enabled": enabled,
        "metric_columns": metric_columns,
        "dimension_columns": dimension_columns,
        "datetime_columns": datetime_columns,
        "default_date_column": datetime_columns[0] if len(datetime_columns) == 1 else None,
        "readiness_status": readiness_status,
    }


def run_diagnostic_analysis(
    dataframe: pd.DataFrame,
    metric_column: str,
    dimension_column: str,
    date_column: str | None = None,
    breakdown_dimension: bool = False,
) -> dict[str, Any]:
    """Compute deterministic grouped diagnostic analysis."""
    working_frame = dataframe.copy()
    working_frame[metric_column] = pd.to_numeric(working_frame[metric_column], errors="coerce")
    working_frame[dimension_column] = _clean_dimension_values(working_frame[dimension_column])
    analysis_frame = working_frame.loc[
        working_frame[metric_column].notna() & working_frame[dimension_column].notna(),
        [metric_column, dimension_column],
    ].copy()

    if analysis_frame.empty:
        return _empty_analysis(metric_column, dimension_column, date_column)

    grouped = (
        analysis_frame.groupby(dimension_column, dropna=False)[metric_column]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .rename(
            columns={
                dimension_column: "group",
                "sum": "metric_value",
                "count": "row_count",
                "mean": "average_value",
            }
        )
        .sort_values(by="metric_value", ascending=False)
        .reset_index(drop=True)
    )

    total_metric_value = float(grouped["metric_value"].sum())
    grouped["share_pct"] = grouped["metric_value"].apply(
        lambda value: round((value / total_metric_value) * 100, 2) if total_metric_value else 0.0
    )
    grouped["efficiency_index"] = grouped["average_value"].apply(lambda value: round(float(value), 2))

    top_group = grouped.iloc[0]
    bottom_group = grouped.iloc[-1]
    group_count = int(len(grouped))
    top_share = float(top_group["share_pct"])
    summary = {
        "total_metric_value": total_metric_value,
        "group_count": group_count,
        "top_group": str(top_group["group"]),
        "top_group_value": float(top_group["metric_value"]),
        "bottom_group": str(bottom_group["group"]),
        "bottom_group_value": float(bottom_group["metric_value"]),
        "difference_top_to_bottom": float(
            top_group["metric_value"] - bottom_group["metric_value"]
        ),
        "lowest_group_label": "Lowest-Contributing Group",
        "spread_classification": _classify_spread(top_group["metric_value"], bottom_group["metric_value"]),
        "concentration_classification": _classify_concentration(top_share),
    }

    trend_analysis = _build_trend_analysis(
        dataframe=working_frame,
        metric_column=metric_column,
        dimension_column=dimension_column,
        date_column=date_column,
        breakdown_dimension=breakdown_dimension,
    )
    findings = build_diagnostic_findings(
        dataframe=working_frame,
        metric_column=metric_column,
        dimension_column=dimension_column,
        grouped=grouped,
    )
    ranked_groups = grouped.reset_index(drop=True).copy()
    ranked_groups.insert(0, "rank", range(1, len(ranked_groups) + 1))

    return {
        "summary": summary,
        "grouped_comparison": grouped,
        "ranked_groups": ranked_groups,
        "show_split_top_bottom": group_count > 5,
        "top_groups": grouped.head(5).reset_index(drop=True),
        "bottom_groups": grouped.tail(5).sort_values(by="metric_value").reset_index(drop=True),
        "trend_analysis": trend_analysis,
        "findings": findings,
        "metric_column": metric_column,
        "dimension_column": dimension_column,
        "date_column": date_column,
    }


def build_diagnostic_findings(
    dataframe: pd.DataFrame,
    metric_column: str,
    dimension_column: str,
    grouped: pd.DataFrame,
) -> list[str]:
    """Create deterministic, evidence-based diagnostic findings."""
    if grouped.empty:
        return ["No diagnostic findings were generated because the selected metric has no usable rows."]

    findings: list[str] = []
    total_metric = float(grouped["metric_value"].sum())
    top_group = grouped.iloc[0]
    bottom_group = grouped.iloc[-1]
    average_group_total = float(grouped["metric_value"].mean())
    overall_row_average = float(dataframe[metric_column].dropna().mean()) if dataframe[metric_column].notna().any() else 0.0
    spread_classification = _classify_spread(top_group["metric_value"], bottom_group["metric_value"])
    concentration_classification = _classify_concentration(float(top_group["share_pct"]))
    findings.append(
        f"Top contributor: {top_group['group']} drives {top_group['share_pct']:.1f}% of total {metric_column} ({top_group['metric_value']:.2f})."
    )
    findings.append(
        f"Lowest-contributing group: {bottom_group['group']} contributes {bottom_group['metric_value']:.2f}, which is {top_group['metric_value'] - bottom_group['metric_value']:.2f} below the top group."
    )
    findings.append(
        f"Performance spread is {spread_classification}: the top-to-bottom contribution gap is {top_group['metric_value'] - bottom_group['metric_value']:.2f}."
    )
    findings.append(
        f"Contribution concentration is {concentration_classification}: the leading group accounts for {top_group['share_pct']:.1f}% of total {metric_column}."
    )

    underperformers = grouped.loc[grouped["metric_value"] < (average_group_total * 0.6), "group"].tolist()
    if underperformers:
        findings.append(
            "Underperforming groups: "
            + ", ".join(map(str, underperformers[:5]))
            + f" are materially below the group average for {metric_column}."
        )

    findings.append(
        f"Average contribution per group is {average_group_total:.2f}, while the overall per-row average for {metric_column} is {overall_row_average:.2f}."
    )

    efficiency_finding = _build_efficiency_finding(grouped, metric_column)
    if efficiency_finding:
        findings.append(efficiency_finding)

    margin_finding = _build_margin_concern_finding(
        dataframe=dataframe,
        dimension_column=dimension_column,
    )
    if margin_finding:
        findings.append(margin_finding)

    if total_metric <= 0:
        findings.append(
            f"The selected metric totals {total_metric:.2f}, so overall performance is weak before any deeper causal interpretation."
        )

    return findings


def diagnose_dataframe(dataframe: pd.DataFrame) -> dict[str, str]:
    """Return a simple compatibility diagnostic summary."""
    _ = dataframe
    return {"status": "implemented"}


def _get_metric_columns(dataframe: pd.DataFrame, numeric_columns: list[str]) -> list[str]:
    """Return usable business metric columns while excluding obvious identifiers."""
    metric_columns: list[str] = []
    row_count = max(len(dataframe), 1)

    for column_name in numeric_columns:
        normalized_name = _normalize_name(column_name)
        series = pd.to_numeric(dataframe[column_name], errors="coerce")
        valid_ratio = _valid_ratio(series)
        unique_ratio = series.nunique(dropna=True) / max(series.notna().sum(), 1)
        if valid_ratio < 0.6 or series.nunique(dropna=True) < 2:
            continue
        if _looks_like_identifier(normalized_name, unique_ratio, row_count):
            continue
        if _contains_keyword(normalized_name, BUSINESS_METRIC_KEYWORDS) or unique_ratio < 0.95:
            metric_columns.append(column_name)

    return metric_columns


def _get_dimension_columns(
    dataframe: pd.DataFrame,
    categorical_columns: list[str],
) -> list[str]:
    """Return categorical dimensions that are usable for grouped analysis."""
    dimension_columns: list[str] = []

    for column_name in categorical_columns:
        series = _clean_dimension_values(dataframe[column_name])
        non_null = series.dropna()
        if non_null.empty:
            continue
        unique_count = int(non_null.nunique(dropna=True))
        valid_ratio = len(non_null) / max(len(series), 1)
        unique_ratio = unique_count / max(len(non_null), 1)
        if valid_ratio < 0.5:
            continue
        if unique_count < 2 or unique_count > min(MAX_DIMENSION_GROUPS, max(len(series) // 2, 2)):
            continue
        if unique_ratio >= 0.95:
            continue
        dimension_columns.append(column_name)

    return dimension_columns


def _get_datetime_columns(
    dataframe: pd.DataFrame,
    datetime_columns: list[str],
) -> list[str]:
    """Return datetime columns with enough parse success for trend analysis."""
    usable_columns: list[str] = []
    for column_name in datetime_columns:
        series = pd.to_datetime(dataframe[column_name], errors="coerce", format="mixed")
        if _valid_ratio(series) >= 0.75 and series.dropna().nunique() >= 2:
            usable_columns.append(column_name)
    return usable_columns


def _build_trend_analysis(
    dataframe: pd.DataFrame,
    metric_column: str,
    dimension_column: str,
    date_column: str | None,
    breakdown_dimension: bool,
) -> dict[str, Any] | None:
    """Build time-based aggregation if a usable datetime column is selected."""
    if not date_column:
        return None

    trend_frame = dataframe[[metric_column, dimension_column, date_column]].copy()
    trend_frame[metric_column] = pd.to_numeric(trend_frame[metric_column], errors="coerce")
    trend_frame[dimension_column] = _clean_dimension_values(trend_frame[dimension_column])
    trend_frame[date_column] = pd.to_datetime(trend_frame[date_column], errors="coerce", format="mixed")
    trend_frame = trend_frame.dropna(subset=[metric_column, date_column])
    if trend_frame.empty:
        return None

    frequency = _infer_time_frequency(trend_frame[date_column])
    trend_frame["period"] = trend_frame[date_column].dt.to_period(frequency).dt.to_timestamp()

    summary = (
        trend_frame.groupby("period")[metric_column]
        .sum()
        .reset_index()
        .sort_values(by="period")
        .reset_index(drop=True)
    )

    by_dimension = None
    if breakdown_dimension:
        usable_dimension_values = trend_frame[dimension_column].dropna()
        group_count = int(usable_dimension_values.nunique(dropna=True))
        if 1 < group_count <= MAX_TREND_BREAKDOWN_GROUPS:
            by_dimension = (
                trend_frame.dropna(subset=[dimension_column])
                .groupby(["period", dimension_column])[metric_column]
                .sum()
                .reset_index()
                .rename(columns={dimension_column: "group", metric_column: "metric_value"})
                .sort_values(by=["period", "group"])
                .reset_index(drop=True)
            )

    return {
        "frequency": frequency,
        "summary": summary.rename(columns={metric_column: "metric_value"}),
        "by_dimension": by_dimension,
    }


def _build_margin_concern_finding(
    dataframe: pd.DataFrame,
    dimension_column: str,
) -> str | None:
    """Flag discount and profit tension when the data supports it."""
    discount_column = _find_matching_column(dataframe.columns, DISCOUNT_KEYWORDS)
    profit_column = _find_matching_column(dataframe.columns, PROFIT_KEYWORDS)
    if not discount_column or not profit_column:
        return None

    margin_frame = dataframe[[dimension_column, discount_column, profit_column]].copy()
    margin_frame[dimension_column] = _clean_dimension_values(margin_frame[dimension_column])
    margin_frame[discount_column] = pd.to_numeric(margin_frame[discount_column], errors="coerce")
    margin_frame[profit_column] = pd.to_numeric(margin_frame[profit_column], errors="coerce")
    margin_frame = margin_frame.dropna(subset=[dimension_column, discount_column, profit_column])
    if margin_frame.empty:
        return None

    grouped = (
        margin_frame.groupby(dimension_column)
        .agg(
            average_discount=(discount_column, "mean"),
            total_profit=(profit_column, "sum"),
        )
        .reset_index()
    )
    if len(grouped) < 2:
        return None

    discount_cutoff = float(grouped["average_discount"].quantile(0.75))
    profit_cutoff = float(grouped["total_profit"].median())
    flagged = grouped.loc[
        (grouped["average_discount"] >= discount_cutoff)
        & (grouped["total_profit"] <= profit_cutoff)
    ]
    if flagged.empty:
        return None

    flagged_groups = ", ".join(flagged[dimension_column].astype(str).head(3).tolist())
    return (
        f"Margin concern: {flagged_groups} show relatively high discount levels while profit remains weak, so discounting may be eroding value in those segments."
    )


def _empty_analysis(
    metric_column: str,
    dimension_column: str,
    date_column: str | None,
) -> dict[str, Any]:
    """Return an empty analysis structure when the selection has no usable rows."""
    return {
        "summary": {
            "total_metric_value": 0.0,
            "group_count": 0,
            "top_group": "N/A",
            "top_group_value": 0.0,
            "bottom_group": "N/A",
            "bottom_group_value": 0.0,
            "difference_top_to_bottom": 0.0,
            "lowest_group_label": "Lowest-Contributing Group",
            "spread_classification": "N/A",
            "concentration_classification": "N/A",
        },
        "grouped_comparison": pd.DataFrame(
            columns=["group", "metric_value", "row_count", "average_value", "share_pct", "efficiency_index"]
        ),
        "ranked_groups": pd.DataFrame(
            columns=["rank", "group", "metric_value", "row_count", "average_value", "share_pct", "efficiency_index"]
        ),
        "show_split_top_bottom": False,
        "top_groups": pd.DataFrame(
            columns=["group", "metric_value", "row_count", "average_value", "share_pct", "efficiency_index"]
        ),
        "bottom_groups": pd.DataFrame(
            columns=["group", "metric_value", "row_count", "average_value", "share_pct", "efficiency_index"]
        ),
        "trend_analysis": None,
        "findings": [
            f"No diagnostic findings were generated because `{metric_column}` and `{dimension_column}` do not overlap on usable rows."
        ],
        "metric_column": metric_column,
        "dimension_column": dimension_column,
        "date_column": date_column,
    }


def _clean_dimension_values(series: pd.Series) -> pd.Series:
    """Normalize dimension values for consistent grouping."""
    cleaned = series.copy()
    cleaned = cleaned.where(cleaned.notna(), other=pd.NA)
    text_values = cleaned.astype("string").str.strip()
    text_values = text_values.mask(text_values.eq(""), other=pd.NA)
    return text_values


def _infer_time_frequency(series: pd.Series) -> str:
    """Choose a readable aggregation grain based on time depth."""
    parsed = pd.to_datetime(series, errors="coerce")
    parsed = parsed.dropna().sort_values()
    if parsed.empty:
        return "D"
    time_depth_days = int((parsed.max() - parsed.min()).days)
    if time_depth_days >= 365:
        return "M"
    if time_depth_days >= 180:
        return "M"
    if time_depth_days >= 60:
        return "W"
    return "D"


def _classify_spread(top_value: float, bottom_value: float) -> str:
    """Classify the contribution spread between top and bottom groups."""
    if bottom_value <= 0:
        return "wide"
    ratio = float(top_value / bottom_value)
    if ratio >= 2.5:
        return "wide"
    if ratio >= 1.5:
        return "moderate"
    return "narrow"


def _classify_concentration(top_share: float) -> str:
    """Classify concentration risk from the leading group's share."""
    if top_share >= 50:
        return "high"
    if top_share >= 30:
        return "moderate"
    return "low"


def _build_efficiency_finding(grouped: pd.DataFrame, metric_column: str) -> str | None:
    """Highlight when volume leadership differs from per-row efficiency leadership."""
    if grouped.empty or len(grouped) < 2:
        return None
    top_volume_group = grouped.sort_values(by="metric_value", ascending=False).iloc[0]
    top_efficiency_group = grouped.sort_values(by="average_value", ascending=False).iloc[0]
    if top_volume_group["group"] == top_efficiency_group["group"]:
        return (
            f"Volume and per-row efficiency align: {top_volume_group['group']} leads both total {metric_column} and average value per row."
        )
    return (
        f"Volume versus efficiency split: {top_volume_group['group']} leads total {metric_column}, but {top_efficiency_group['group']} has the strongest average value per row."
    )


def _looks_like_identifier(normalized_name: str, unique_ratio: float, row_count: int) -> bool:
    """Check whether a numeric field is more likely to be an identifier than a metric."""
    if normalized_name in ID_KEYWORDS or normalized_name.endswith("_id"):
        return True
    if (
        _contains_keyword(normalized_name, {"id", "code", "key"})
        and not _contains_keyword(normalized_name, BUSINESS_METRIC_KEYWORDS)
    ):
        return True
    return (
        row_count >= 20
        and unique_ratio >= 0.98
        and not _contains_keyword(normalized_name, BUSINESS_METRIC_KEYWORDS)
    )


def _find_matching_column(columns: pd.Index, keywords: set[str]) -> str | None:
    """Return the first column whose normalized name matches the supplied keywords."""
    for column_name in columns:
        normalized_name = _normalize_name(column_name)
        if _contains_keyword(normalized_name, keywords):
            return str(column_name)
    return None


def _contains_keyword(normalized_name: str, keywords: set[str]) -> bool:
    """Check whether a normalized name includes any target keyword."""
    return any(keyword in normalized_name for keyword in keywords)


def _normalize_name(name: str) -> str:
    """Normalize column names for rule checks."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _valid_ratio(series: pd.Series) -> float:
    """Return the proportion of non-null values in a series."""
    if len(series) == 0:
        return 0.0
    return float(series.notna().mean())
