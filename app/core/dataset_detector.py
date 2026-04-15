"""Dataset intelligence helpers."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


MIN_MODELING_ROWS = 60
STRONG_MODELING_ROWS = 120
MIN_TARGET_VALID_ROWS = 40
STRONG_TARGET_VALID_ROWS = 80
MIN_TIME_PERIODS = 4
STRONG_TIME_PERIODS = 6
MIN_TIME_DEPTH_DAYS = 45
STRONG_TIME_DEPTH_DAYS = 120
MIN_DIMENSION_COMPLETENESS = 0.6
MIN_NUMERIC_COMPLETENESS = 0.6


def detect_dataset_type(
    dataframe: pd.DataFrame,
    column_profiles: dict[str, list[str]],
) -> str:
    """Infer the most likely dataset type using lightweight rules."""
    signal_flags = _extract_signal_flags(dataframe, column_profiles)
    scores = {
        "transactional": 0,
        "time_series": 0,
        "customer": 0,
        "service_support": 0,
        "marketing_funnel": 0,
        "generic_tabular": 1,
    }

    if signal_flags["date_column_present"]:
        scores["transactional"] += 2
        scores["time_series"] += 3
        scores["marketing_funnel"] += 1

    if signal_flags["revenue_like_column_detected"]:
        scores["transactional"] += 3
        scores["marketing_funnel"] += 2
        scores["time_series"] += 1

    if signal_flags["id_column_detected"]:
        scores["transactional"] += 2
        scores["customer"] += 2
        scores["service_support"] += 1

    if signal_flags["category_dimension_detected"]:
        scores["transactional"] += 1
        scores["customer"] += 1
        scores["service_support"] += 1
        scores["marketing_funnel"] += 2

    if signal_flags["target_like_numeric_outcome_detected"]:
        scores["transactional"] += 2
        scores["time_series"] += 2
        scores["marketing_funnel"] += 1

    column_names = _normalized_columns(dataframe.columns)

    if _matches_keywords(column_names, {"customer", "client", "user", "segment", "churn"}):
        scores["customer"] += 4

    if _matches_keywords(
        column_names,
        {"ticket", "case", "issue", "agent", "resolution", "support", "sla"},
    ):
        scores["service_support"] += 5

    if _matches_keywords(
        column_names,
        {"campaign", "lead", "conversion", "impression", "click", "funnel", "channel"},
    ):
        scores["marketing_funnel"] += 5

    if len(column_profiles.get("datetime", [])) == 1 and len(column_profiles.get("numeric", [])) <= 2:
        scores["time_series"] += 2

    if len(column_profiles.get("categorical", [])) >= 2 and signal_flags["revenue_like_column_detected"]:
        scores["transactional"] += 2

    best_match = max(scores, key=scores.get)
    if scores[best_match] <= 2:
        return "generic_tabular"
    return best_match


def build_dataset_intelligence(
    dataframe: pd.DataFrame,
    column_profiles: dict[str, list[str]],
) -> dict[str, object]:
    """Build Phase 1.5 dataset intelligence and analytics readiness results."""
    signal_flags = _extract_signal_flags(dataframe, column_profiles)
    detected_dataset_type = detect_dataset_type(dataframe, column_profiles)
    business_signals = _build_business_signals(signal_flags)
    readiness = _build_analytics_readiness(
        dataframe=dataframe,
        column_profiles=column_profiles,
        signal_flags=signal_flags,
    )

    return {
        "dataset_type": detected_dataset_type,
        "business_signals": business_signals,
        "readiness": readiness,
    }


def _build_analytics_readiness(
    dataframe: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    signal_flags: dict[str, bool],
) -> dict[str, dict[str, str]]:
    """Estimate readiness for each analytics layer using conservative rules."""
    row_count = int(len(dataframe))
    column_count = int(len(dataframe.columns))
    dataset_valid = row_count > 0 and column_count > 0
    numeric_columns = column_profiles.get("numeric", [])
    categorical_columns = column_profiles.get("categorical", [])
    datetime_columns = column_profiles.get("datetime", [])
    usable_numeric_columns = _get_usable_numeric_columns(dataframe, numeric_columns)
    usable_dimension_columns = _get_usable_dimension_columns(
        dataframe=dataframe,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
    )
    datetime_support = _evaluate_datetime_support(dataframe, datetime_columns)
    target_support = _evaluate_target_support(
        dataframe=dataframe,
        numeric_columns=numeric_columns,
        usable_numeric_columns=usable_numeric_columns,
        usable_dimension_columns=usable_dimension_columns,
    )
    decision_support = _evaluate_decision_support(
        dataframe=dataframe,
        usable_dimension_columns=usable_dimension_columns,
        target_support=target_support,
    )

    descriptive_status = "Ready" if dataset_valid else "Not Ready"
    descriptive_reason = (
        f"The dataset is valid tabular data with {row_count} rows and {column_count} columns, so descriptive summaries and KPI reporting can be trusted."
        if dataset_valid
        else "The dataset is not yet a valid tabular structure, so descriptive analysis should stay blocked."
    )

    diagnostic_status = _classify_diagnostic_status(
        dataset_valid=dataset_valid,
        usable_numeric_columns=usable_numeric_columns,
        usable_dimension_columns=usable_dimension_columns,
    )
    diagnostic_reason = _build_diagnostic_reason(
        dataset_valid=dataset_valid,
        usable_numeric_columns=usable_numeric_columns,
        usable_dimension_columns=usable_dimension_columns,
    )

    predictive_status = _classify_predictive_status(
        dataset_valid=dataset_valid,
        diagnostic_status=diagnostic_status,
        row_count=row_count,
        datetime_support=datetime_support,
        target_support=target_support,
    )
    predictive_reason = _build_predictive_reason(
        row_count=row_count,
        predictive_status=predictive_status,
        diagnostic_status=diagnostic_status,
        datetime_support=datetime_support,
        target_support=target_support,
    )

    prescriptive_status = _classify_prescriptive_status(
        diagnostic_status=diagnostic_status,
        predictive_status=predictive_status,
        decision_support=decision_support,
    )
    prescriptive_reason = _build_prescriptive_reason(
        prescriptive_status=prescriptive_status,
        predictive_status=predictive_status,
        decision_support=decision_support,
        signal_flags=signal_flags,
    )

    return {
        "descriptive": {
            "status": descriptive_status,
            "reason": descriptive_reason,
        },
        "diagnostic": {
            "status": diagnostic_status,
            "reason": diagnostic_reason,
        },
        "predictive": {
            "status": predictive_status,
            "reason": predictive_reason,
        },
        "prescriptive": {
            "status": prescriptive_status,
            "reason": prescriptive_reason,
        },
    }


def _classify_diagnostic_status(
    dataset_valid: bool,
    usable_numeric_columns: list[str],
    usable_dimension_columns: list[str],
) -> str:
    """Classify diagnostic readiness from measurable fields and dimensions."""
    if not dataset_valid:
        return "Not Ready"
    if usable_numeric_columns and usable_dimension_columns:
        return "Ready"
    if usable_numeric_columns or usable_dimension_columns:
        return "Conditional"
    return "Not Ready"


def _build_diagnostic_reason(
    dataset_valid: bool,
    usable_numeric_columns: list[str],
    usable_dimension_columns: list[str],
) -> str:
    """Explain the diagnostic readiness result."""
    if not dataset_valid:
        return "Diagnostic analysis is blocked because the dataset is not yet a valid tabular input."
    if usable_numeric_columns and usable_dimension_columns:
        return (
            f"The dataset has {len(usable_numeric_columns)} measurable field(s) and {len(usable_dimension_columns)} usable dimension(s), which supports slice-and-compare diagnostics."
        )
    if usable_numeric_columns:
        return (
            f"The dataset has {len(usable_numeric_columns)} measurable field(s), but it still needs a reliable dimension for meaningful drill-down analysis."
        )
    if usable_dimension_columns:
        return (
            f"The dataset has {len(usable_dimension_columns)} usable dimension(s), but it still needs at least one measurable field for diagnostic comparisons."
        )
    return "Diagnostic readiness is not established because measurable fields and explanatory dimensions were not confirmed."


def _classify_predictive_status(
    dataset_valid: bool,
    diagnostic_status: str,
    row_count: int,
    datetime_support: dict[str, Any],
    target_support: dict[str, Any],
) -> str:
    """Classify predictive readiness conservatively from pre-modeling evidence."""
    if not dataset_valid or diagnostic_status == "Not Ready":
        return "Not Ready"
    if datetime_support["strong"] or target_support["strong"]:
        return "Ready"
    if (
        row_count >= MIN_MODELING_ROWS
        and diagnostic_status == "Ready"
        and (
            datetime_support["candidate_present"]
            or target_support["candidate_count"] > 0
            or target_support["feature_support_count"] >= 2
        )
    ):
        return "Conditional"
    if (
        row_count >= MIN_MODELING_ROWS
        and diagnostic_status == "Conditional"
        and (
        datetime_support["candidate_present"] or target_support["candidate_count"] > 0
        )
    ):
        return "Conditional"
    return "Not Ready"


def _build_predictive_reason(
    row_count: int,
    predictive_status: str,
    diagnostic_status: str,
    datetime_support: dict[str, Any],
    target_support: dict[str, Any],
) -> str:
    """Explain predictive readiness using conservative pre-modeling checks."""
    if predictive_status == "Ready":
        if datetime_support["strong"]:
            return (
                f"Predictive readiness is Ready because `{datetime_support['best_column']}` parsed at {datetime_support['parse_success_rate']:.0%}, covers {datetime_support['unique_periods']} time periods, and spans {datetime_support['time_depth_days']} days."
            )
        return (
            f"Predictive readiness is Ready because `{target_support['best_target_column']}` looks like a plausible outcome with {target_support['best_target_valid_rows']} valid rows and {target_support['feature_support_count']} supporting feature(s)."
        )

    gaps: list[str] = []
    signals: list[str] = []

    if row_count < MIN_MODELING_ROWS:
        gaps.append(f"only {row_count} rows are available for pre-modeling checks")
    if diagnostic_status != "Ready":
        gaps.append("measurable fields and slice dimensions are not both established yet")

    if datetime_support["candidate_present"]:
        signals.append(
            f"datetime evidence is partial ({datetime_support['parse_success_rate']:.0%} parse success across {datetime_support['unique_periods']} time periods)"
        )
        if not datetime_support["strong"]:
            gaps.append("time depth is not strong enough yet for trustworthy forecasting-style readiness")
    else:
        gaps.append("no reliable datetime structure was confirmed")

    if target_support["candidate_count"] > 0:
        signals.append(
            f"{target_support['candidate_count']} plausible target-like column(s) were detected"
        )
        if not target_support["strong"]:
            gaps.append(
                f"the best outcome candidate has {target_support['best_target_valid_rows']} valid rows with {target_support['feature_support_count']} supporting feature(s)"
            )
    else:
        gaps.append("no plausible target-like outcome column was detected")

    if predictive_status == "Conditional":
        signal_text = "; ".join(signals) if signals else "some early modeling signals exist"
        return (
            "Predictive readiness is Conditional because "
            + signal_text
            + ", but "
            + ", ".join(gaps)
            + "."
        )

    return "Predictive readiness is Not Ready because " + ", ".join(gaps) + "."


def _classify_prescriptive_status(
    diagnostic_status: str,
    predictive_status: str,
    decision_support: dict[str, Any],
) -> str:
    """Classify prescriptive readiness without implementing prescriptive logic."""
    if diagnostic_status == "Not Ready":
        return "Not Ready"
    if predictive_status == "Ready" and decision_support["strong"]:
        return "Ready"
    if diagnostic_status == "Ready" and decision_support["candidate"]:
        return "Conditional"
    return "Not Ready"


def _build_prescriptive_reason(
    prescriptive_status: str,
    predictive_status: str,
    decision_support: dict[str, Any],
    signal_flags: dict[str, bool],
) -> str:
    """Explain prescriptive readiness from decision-support structure only."""
    actionable_count = len(decision_support["actionable_dimensions"])
    if prescriptive_status == "Ready":
        return (
            f"Prescriptive readiness is Ready because the dataset already shows strong outcome evidence plus {actionable_count} actionable decision dimension(s) for future rule logic."
        )
    if prescriptive_status == "Conditional":
        reasons = [
            f"{actionable_count} actionable dimension(s) were detected",
            f"{decision_support['valid_outcome_rows']} valid outcome row(s) are available",
        ]
        if predictive_status != "Ready":
            reasons.append("predictive evidence is still being treated conservatively")
        return (
            "Prescriptive readiness is Conditional because "
            + ", ".join(reasons)
            + ", but stronger decision logic signals should be confirmed before anything is forced."
        )

    if signal_flags["category_dimension_detected"]:
        return (
            "Prescriptive readiness is Not Ready because category-style dimensions exist, but there is not yet enough trustworthy outcome evidence or actionable structure."
        )
    return (
        "Prescriptive readiness is Not Ready because the dataset does not yet show enough actionable dimensions or dependable outcome support."
    )


def _get_usable_numeric_columns(dataframe: pd.DataFrame, numeric_columns: list[str]) -> list[str]:
    """Return numeric columns with enough valid values to support analysis."""
    usable_columns: list[str] = []
    for column_name in numeric_columns:
        series = pd.to_numeric(dataframe[column_name], errors="coerce")
        valid_ratio = _valid_ratio(series)
        if valid_ratio >= MIN_NUMERIC_COMPLETENESS and series.nunique(dropna=True) >= 2:
            usable_columns.append(column_name)
    return usable_columns


def _get_usable_dimension_columns(
    dataframe: pd.DataFrame,
    categorical_columns: list[str],
    datetime_columns: list[str],
) -> list[str]:
    """Return dimension columns that are suitable for slice analysis."""
    usable_columns: list[str] = []

    for column_name in categorical_columns:
        series = dataframe[column_name]
        non_null = _non_empty_series(series)
        if non_null.empty:
            continue
        unique_count = int(non_null.nunique(dropna=True))
        unique_ratio = unique_count / max(len(non_null), 1)
        valid_ratio = len(non_null) / max(len(series), 1)
        if valid_ratio >= MIN_DIMENSION_COMPLETENESS and 2 <= unique_count <= max(25, int(len(series) * 0.5)):
            if unique_ratio <= 0.9:
                usable_columns.append(column_name)

    for column_name in datetime_columns:
        metrics = _get_datetime_column_metrics(dataframe[column_name])
        if metrics["parse_success_rate"] >= 0.75 and metrics["unique_periods"] >= 2:
            usable_columns.append(column_name)

    return list(dict.fromkeys(usable_columns))


def _evaluate_datetime_support(
    dataframe: pd.DataFrame,
    datetime_columns: list[str],
) -> dict[str, Any]:
    """Evaluate whether the dataset has strong time-based modeling structure."""
    candidate_columns = list(datetime_columns)
    for column_name in dataframe.columns:
        normalized_name = _normalize_name(column_name)
        if any(token in normalized_name for token in {"date", "time", "month", "year", "timestamp"}):
            if column_name not in candidate_columns:
                candidate_columns.append(column_name)

    best_metrics: dict[str, Any] = {
        "candidate_present": bool(candidate_columns),
        "best_column": None,
        "parse_success_rate": 0.0,
        "unique_periods": 0,
        "time_depth_days": 0,
        "strong": False,
    }

    for column_name in candidate_columns:
        metrics = _get_datetime_column_metrics(dataframe[column_name])
        if (
            metrics["parse_success_rate"],
            metrics["unique_periods"],
            metrics["time_depth_days"],
        ) > (
            best_metrics["parse_success_rate"],
            best_metrics["unique_periods"],
            best_metrics["time_depth_days"],
        ):
            best_metrics = {
                "candidate_present": True,
                "best_column": column_name,
                **metrics,
            }

    best_metrics["strong"] = bool(
        best_metrics["candidate_present"]
        and best_metrics["parse_success_rate"] >= 0.9
        and best_metrics["unique_periods"] >= STRONG_TIME_PERIODS
        and best_metrics["time_depth_days"] >= STRONG_TIME_DEPTH_DAYS
        and len(dataframe) >= STRONG_MODELING_ROWS
    )
    return best_metrics


def _evaluate_target_support(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    usable_numeric_columns: list[str],
    usable_dimension_columns: list[str],
) -> dict[str, Any]:
    """Evaluate plausible supervised-modeling support from outcome-like columns."""
    target_keywords = {
        "sales",
        "revenue",
        "amount",
        "profit",
        "margin",
        "cost",
        "quantity",
        "count",
        "units",
        "demand",
        "score",
        "target",
        "label",
        "y",
    }
    candidate_columns = [
        column_name
        for column_name in numeric_columns
        if any(keyword in _normalize_name(column_name) for keyword in target_keywords)
    ]

    best_target_column: str | None = None
    best_target_valid_rows = 0
    best_target_valid_ratio = 0.0

    for column_name in candidate_columns:
        series = pd.to_numeric(dataframe[column_name], errors="coerce")
        valid_rows = int(series.notna().sum())
        valid_ratio = _valid_ratio(series)
        if (valid_rows, valid_ratio) > (best_target_valid_rows, best_target_valid_ratio):
            best_target_column = column_name
            best_target_valid_rows = valid_rows
            best_target_valid_ratio = valid_ratio

    feature_support_count = max(
        len(usable_numeric_columns) - (1 if best_target_column in usable_numeric_columns else 0),
        0,
    ) + len(usable_dimension_columns)

    strong = bool(
        best_target_column
        and len(dataframe) >= STRONG_MODELING_ROWS
        and best_target_valid_rows >= STRONG_TARGET_VALID_ROWS
        and best_target_valid_ratio >= 0.75
        and feature_support_count >= 3
    )

    candidate = bool(
        candidate_columns
        and best_target_valid_rows >= MIN_TARGET_VALID_ROWS
        and feature_support_count >= 1
    )

    return {
        "candidate_count": len(candidate_columns),
        "best_target_column": best_target_column,
        "best_target_valid_rows": best_target_valid_rows,
        "best_target_valid_ratio": best_target_valid_ratio,
        "feature_support_count": feature_support_count,
        "candidate": candidate,
        "strong": strong,
    }


def _evaluate_decision_support(
    dataframe: pd.DataFrame,
    usable_dimension_columns: list[str],
    target_support: dict[str, Any],
) -> dict[str, Any]:
    """Estimate whether decision-support structure exists for future phases."""
    actionable_keywords = {
        "category",
        "segment",
        "region",
        "channel",
        "product",
        "department",
        "store",
        "plan",
        "status",
        "tier",
        "priority",
        "discount",
        "price",
        "promotion",
    }
    actionable_dimensions = [
        column_name
        for column_name in usable_dimension_columns
        if any(keyword in _normalize_name(column_name) for keyword in actionable_keywords)
    ]

    return {
        "actionable_dimensions": actionable_dimensions,
        "valid_outcome_rows": target_support["best_target_valid_rows"],
        "candidate": bool(actionable_dimensions and target_support["candidate"]),
        "strong": bool(
            len(actionable_dimensions) >= 2
            and target_support["strong"]
            and target_support["best_target_valid_rows"] >= STRONG_TARGET_VALID_ROWS
            and len(dataframe) >= STRONG_MODELING_ROWS
        ),
    }


def _get_datetime_column_metrics(series: pd.Series) -> dict[str, Any]:
    """Measure parse success and time depth for a datetime-like column."""
    non_empty = _non_empty_series(series)
    if non_empty.empty:
        return {
            "parse_success_rate": 0.0,
            "unique_periods": 0,
            "time_depth_days": 0,
        }

    parsed = pd.to_datetime(non_empty, errors="coerce", format="mixed")
    parse_success_rate = _valid_ratio(parsed)
    parsed_non_null = parsed.dropna()
    if parsed_non_null.empty:
        return {
            "parse_success_rate": parse_success_rate,
            "unique_periods": 0,
            "time_depth_days": 0,
        }

    time_depth_days = int((parsed_non_null.max() - parsed_non_null.min()).days)
    if time_depth_days >= MIN_TIME_DEPTH_DAYS:
        unique_periods = int(parsed_non_null.dt.to_period("M").nunique())
    else:
        unique_periods = int(parsed_non_null.dt.normalize().nunique())

    return {
        "parse_success_rate": parse_success_rate,
        "unique_periods": unique_periods,
        "time_depth_days": time_depth_days,
    }


def _non_empty_series(series: pd.Series) -> pd.Series:
    """Return non-null, non-blank values from a series."""
    non_null = series.dropna()
    if non_null.empty:
        return non_null
    text_values = non_null.astype(str).str.strip()
    return non_null.loc[text_values.ne("")]


def _valid_ratio(series: pd.Series) -> float:
    """Return the proportion of non-null values in a series."""
    if len(series) == 0:
        return 0.0
    return float(series.notna().mean())


def _build_business_signals(signal_flags: dict[str, bool]) -> list[str]:
    """Convert signal flags into readable business signal messages."""
    label_map = {
        "date_column_present": "Date column present",
        "revenue_like_column_detected": "Revenue-like column detected",
        "id_column_detected": "ID column detected",
        "category_dimension_detected": "Category or segment dimension detected",
        "target_like_numeric_outcome_detected": "Target-like numeric outcome detected",
    }
    signals = [label for key, label in label_map.items() if signal_flags.get(key)]
    if not signals:
        return ["No strong business signals detected beyond generic tabular structure"]
    return signals


def _extract_signal_flags(
    dataframe: pd.DataFrame,
    column_profiles: dict[str, list[str]],
) -> dict[str, bool]:
    """Derive reusable dataset signals from names and profile groups."""
    normalized_columns = _normalized_columns(dataframe.columns)
    numeric_columns = {
        _normalize_name(column_name) for column_name in column_profiles.get("numeric", [])
    }
    categorical_columns = {
        _normalize_name(column_name)
        for column_name in column_profiles.get("categorical", [])
    }
    datetime_columns = {
        _normalize_name(column_name) for column_name in column_profiles.get("datetime", [])
    }

    revenue_keywords = {"sales", "revenue", "amount", "price", "profit", "total", "value"}
    id_keywords = {"id", "order", "invoice", "transaction", "customer", "user", "account"}
    category_keywords = {
        "category",
        "segment",
        "type",
        "region",
        "channel",
        "product",
        "department",
        "group",
        "status",
    }
    target_keywords = {
        "sales",
        "revenue",
        "amount",
        "profit",
        "total",
        "score",
        "cost",
        "quantity",
        "count",
    }

    return {
        "date_column_present": bool(datetime_columns)
        or _matches_keywords(normalized_columns, {"date", "time", "month", "year", "timestamp"}),
        "revenue_like_column_detected": _matches_keywords(numeric_columns, revenue_keywords),
        "id_column_detected": _matches_keywords(normalized_columns, id_keywords),
        "category_dimension_detected": _matches_keywords(
            categorical_columns.union(normalized_columns), category_keywords
        ),
        "target_like_numeric_outcome_detected": _matches_keywords(
            numeric_columns, target_keywords
        ),
    }


def _normalized_columns(columns: pd.Index) -> set[str]:
    """Normalize dataframe column names for rule checks."""
    return {_normalize_name(column_name) for column_name in columns}


def _normalize_name(name: str) -> str:
    """Normalize a column name for keyword matching."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _matches_keywords(names: set[str], keywords: set[str]) -> bool:
    """Check whether any normalized name contains a keyword token."""
    for name in names:
        for keyword in keywords:
            if keyword in name:
                return True
    return False
