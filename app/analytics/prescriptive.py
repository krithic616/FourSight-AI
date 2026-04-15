"""Rule-based prescriptive analytics helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

try:
    from analytics.diagnostic import get_diagnostic_options, run_diagnostic_analysis
    from analytics.predictive import get_predictive_options, run_predictive_forecast
except ModuleNotFoundError:
    from app.analytics.diagnostic import get_diagnostic_options, run_diagnostic_analysis
    from app.analytics.predictive import get_predictive_options, run_predictive_forecast


PRIORITY_RANK = {"High": 3, "Medium": 2, "Low": 1}


def get_prescriptive_options(
    dataframe: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    dataset_intelligence: dict[str, object],
) -> dict[str, Any]:
    """Return prescriptive availability and supporting analytic defaults."""
    prescriptive_readiness = dataset_intelligence["readiness"]["prescriptive"]
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

    enough_support = diagnostic_options["enabled"] or predictive_options["enabled"]
    enabled = prescriptive_readiness["status"] in {"Ready", "Conditional"} and enough_support

    return {
        "enabled": enabled,
        "prescriptive_readiness": prescriptive_readiness,
        "diagnostic_options": diagnostic_options,
        "predictive_options": predictive_options,
    }


def build_prescriptive_analysis(
    dataframe: pd.DataFrame,
    column_profiles: dict[str, list[str]],
    dataset_intelligence: dict[str, object],
) -> dict[str, Any]:
    """Build rule-based prescriptive recommendations from supporting analytics."""
    options = get_prescriptive_options(
        dataframe=dataframe,
        column_profiles=column_profiles,
        dataset_intelligence=dataset_intelligence,
    )
    prescriptive_readiness = options["prescriptive_readiness"]
    diagnostic_options = options["diagnostic_options"]
    predictive_options = options["predictive_options"]

    if not options["enabled"]:
        return _build_limited_prescriptive_result(
            readiness=prescriptive_readiness,
            limitation_messages=[
                "Prescriptive recommendations require at least a usable diagnostic or predictive evidence path."
            ],
        )

    diagnostic_analysis = None
    if diagnostic_options["enabled"]:
        diagnostic_analysis = run_diagnostic_analysis(
            dataframe=dataframe,
            metric_column=diagnostic_options["metric_columns"][0],
            dimension_column=diagnostic_options["dimension_columns"][0],
            date_column=diagnostic_options["default_date_column"],
            breakdown_dimension=False,
        )

    predictive_analysis = None
    predictive_limitations: list[str] = []
    if predictive_options["enabled"]:
        predictive_analysis = run_predictive_forecast(
            dataframe=dataframe,
            metric_column=predictive_options["default_metric_column"],
            datetime_column=(
                predictive_options["default_datetime_column"]
                or predictive_options["datetime_columns"][0]
            ),
            aggregation_grain="Auto",
            forecast_horizon=6,
            readiness_status=predictive_options["readiness_status"],
            readiness_reason=predictive_options["readiness_reason"],
        )
        if predictive_analysis["status"] == "Not Ready":
            predictive_limitations = predictive_analysis["validation_messages"]

    recommendations = _build_recommendations(
        diagnostic_analysis=diagnostic_analysis,
        predictive_analysis=predictive_analysis,
        prescriptive_readiness=prescriptive_readiness,
    )
    findings = _build_prescriptive_findings(
        recommendations=recommendations,
        prescriptive_readiness=prescriptive_readiness,
        predictive_analysis=predictive_analysis,
        predictive_limitations=predictive_limitations,
    )
    summary = _build_prescriptive_summary(recommendations)
    limitation_messages = _build_limitation_messages(
        prescriptive_readiness=prescriptive_readiness,
        predictive_analysis=predictive_analysis,
        predictive_limitations=predictive_limitations,
    )

    return {
        "status": prescriptive_readiness["status"],
        "status_reason": prescriptive_readiness["reason"],
        "summary": summary,
        "recommendations": recommendations,
        "findings": findings,
        "diagnostic_analysis": diagnostic_analysis,
        "predictive_analysis": predictive_analysis,
        "limitation_messages": limitation_messages,
    }


def generate_recommendations(dataframe: pd.DataFrame) -> list[str]:
    """Legacy compatibility helper for placeholder callers."""
    _ = dataframe
    return []


def _build_recommendations(
    diagnostic_analysis: dict[str, Any] | None,
    predictive_analysis: dict[str, Any] | None,
    prescriptive_readiness: dict[str, str],
) -> list[dict[str, str]]:
    """Build conservative rule-based recommendations."""
    recommendations: list[dict[str, str]] = []

    if diagnostic_analysis is not None:
        grouped = diagnostic_analysis["grouped_comparison"]
        summary = diagnostic_analysis["summary"]
        metric_name = diagnostic_analysis["metric_column"]
        dimension_name = diagnostic_analysis["dimension_column"]

        if summary["concentration_classification"] == "high":
            recommendations.append(
                _recommendation(
                    category="Risk Controls",
                    priority="High",
                    issue_detected="Top group concentration risk",
                    affected_group=summary["top_group"],
                    supporting_evidence=(
                        f"{summary['top_group']} contributes {float(grouped.iloc[0]['share_pct']):.1f}% of total {metric_name}."
                    ),
                    suggested_action=(
                        f"Reduce reliance on {summary['top_group']} by protecting secondary {dimension_name} contributors and rebalancing growth efforts."
                    ),
                    recommendation_basis="diagnostic-based",
                )
            )
        elif summary["concentration_classification"] == "moderate":
            recommendations.append(
                _recommendation(
                    category="Risk Controls",
                    priority="Medium",
                    issue_detected="Moderate group concentration",
                    affected_group=summary["top_group"],
                    supporting_evidence=(
                        f"{summary['top_group']} is the leading {dimension_name} and contributes {float(grouped.iloc[0]['share_pct']):.1f}% of total {metric_name}."
                    ),
                    suggested_action=(
                        f"Protect the leading {dimension_name} while developing secondary groups so overall performance is less dependent on one segment."
                    ),
                    recommendation_basis="diagnostic-based",
                )
            )

        if summary["spread_classification"] == "wide":
            recommendations.append(
                _recommendation(
                    category="Growth Opportunities",
                    priority="High",
                    issue_detected="Wide performance spread across groups",
                    affected_group=summary["bottom_group"],
                    supporting_evidence=(
                        f"{summary['lowest_group_label']} is {summary['difference_top_to_bottom']:.2f} below the top group on total {metric_name}."
                    ),
                    suggested_action=(
                        f"Investigate weak {summary['bottom_group']} performance and test targeted actions to lift contribution before scaling them more broadly."
                    ),
                    recommendation_basis="diagnostic-based",
                )
            )
        elif summary["spread_classification"] == "moderate":
            recommendations.append(
                _recommendation(
                    category="Growth Opportunities",
                    priority="Medium",
                    issue_detected="Moderate performance gap across groups",
                    affected_group=summary["bottom_group"],
                    supporting_evidence=(
                        f"{summary['bottom_group']} trails the top group by {summary['difference_top_to_bottom']:.2f} on total {metric_name}."
                    ),
                    suggested_action=(
                        f"Review what the stronger {dimension_name} segments are doing well and adapt those practices for {summary['bottom_group']} where feasible."
                    ),
                    recommendation_basis="diagnostic-based",
                )
            )

        weak_efficiency = grouped.loc[
            grouped["average_value"] < grouped["average_value"].mean() * 0.75
        ]
        if not weak_efficiency.empty:
            weakest = weak_efficiency.sort_values(by="average_value").iloc[0]
            recommendations.append(
                _recommendation(
                    category="Efficiency Improvements",
                    priority="Medium",
                    issue_detected="Weak per-row efficiency",
                    affected_group=str(weakest["group"]),
                    supporting_evidence=(
                        f"{weakest['group']} averages {float(weakest['average_value']):.2f} per row versus a group average of {float(grouped['average_value'].mean()):.2f}."
                    ),
                    suggested_action=(
                        f"Improve efficiency in {weakest['group']} by reviewing pricing, mix, and execution rather than relying on volume alone."
                    ),
                    recommendation_basis="diagnostic-based",
                )
            )

        top_efficiency_group = grouped.sort_values(by="average_value", ascending=False).iloc[0]
        if top_efficiency_group["group"] != summary["top_group"]:
            recommendations.append(
                _recommendation(
                    category="Growth Opportunities",
                    priority="Medium",
                    issue_detected="High-efficiency segment is not yet the top contributor",
                    affected_group=str(top_efficiency_group["group"]),
                    supporting_evidence=(
                        f"{top_efficiency_group['group']} has the strongest per-row average at {float(top_efficiency_group['average_value']):.2f}, but it is not the top total contributor."
                    ),
                    suggested_action=(
                        f"Consider scaling what works in {top_efficiency_group['group']} to unlock growth without depending only on the current leading segment."
                    ),
                    recommendation_basis="diagnostic-based",
                )
            )

        margin_recommendation = _build_margin_recommendation(diagnostic_analysis)
        if margin_recommendation is not None:
            recommendations.append(margin_recommendation)

    if predictive_analysis is not None and predictive_analysis["status"] != "Not Ready":
        direction = predictive_analysis["summary_cards"]["projected_direction"]
        metric_name = predictive_analysis["input_summary"]["metric_column"]
        first_forecast = predictive_analysis["summary_cards"]["first_forecast_value"]
        latest_actual = predictive_analysis["summary_cards"]["latest_actual_value"]
        predictive_priority = "High" if direction == "Down" else "Medium"
        if direction in {"Down", "Flat"}:
            recommendations.append(
                _recommendation(
                    category="Risk Controls",
                    priority=predictive_priority,
                    issue_detected="Forecast indicates weak near-term trajectory",
                    affected_group="Overall series",
                    supporting_evidence=(
                        f"The forecast direction is {direction.lower()}, with a first projected value of {first_forecast:.2f} versus the latest actual of {latest_actual:.2f}."
                    ),
                    suggested_action=(
                        f"Prepare a short-term response plan if {metric_name} does not recover, and protect core performance drivers over the next forecast periods."
                    ),
                    recommendation_basis="predictive-based",
                )
            )
        elif direction == "Up":
            recommendations.append(
                _recommendation(
                    category="Growth Opportunities",
                    priority="Medium",
                    issue_detected="Forecast points to near-term momentum",
                    affected_group="Overall series",
                    supporting_evidence=(
                        f"The forecast direction is up, with a first projected value of {first_forecast:.2f} above the latest actual of {latest_actual:.2f}."
                    ),
                    suggested_action=(
                        f"Protect the conditions supporting recent momentum in {metric_name} and look for segments that can absorb incremental growth."
                    ),
                    recommendation_basis="predictive-based",
                )
            )

        if any("volatile" in finding for finding in predictive_analysis["findings"]):
            recommendations.append(
                _recommendation(
                    category="Monitoring Priorities",
                    priority="Medium",
                    issue_detected="Recent trend instability",
                    affected_group="Overall series",
                    supporting_evidence=(
                        next(
                            finding
                            for finding in predictive_analysis["findings"]
                            if "Recent series behavior appears" in finding
                        )
                    ),
                    suggested_action=(
                        f"Monitor {metric_name} more frequently and phase actions carefully while recent volatility remains elevated."
                    ),
                    recommendation_basis="predictive-based",
                )
            )

        if predictive_analysis["status"] == "Conditional":
            recommendations.append(
                _recommendation(
                    category="Monitoring Priorities",
                    priority="Low",
                    issue_detected="Forecast confidence is limited",
                    affected_group="Overall series",
                    supporting_evidence=(
                        "Predictive readiness is conditional, so the forecast should be treated as directional rather than high-confidence."
                    ),
                    suggested_action=(
                        "Use the forecast as a monitoring aid and re-evaluate once more historical depth is available."
                    ),
                    recommendation_basis="predictive-based",
                )
            )

    if diagnostic_analysis is not None and predictive_analysis is not None:
        if (
            diagnostic_analysis["summary"]["concentration_classification"] == "high"
            and predictive_analysis["status"] != "Not Ready"
            and predictive_analysis["summary_cards"]["projected_direction"] == "Down"
        ):
            recommendations.append(
                _recommendation(
                    category="Risk Controls",
                    priority="High",
                    issue_detected="Dominant segment risk with downward outlook",
                    affected_group=diagnostic_analysis["summary"]["top_group"],
                    supporting_evidence=(
                        f"{diagnostic_analysis['summary']['top_group']} dominates current performance and the overall forecast direction is down."
                    ),
                    suggested_action=(
                        "Prioritize contingency planning for the leading segment and accelerate diversification into secondary contributors."
                    ),
                    recommendation_basis="combined",
                )
            )
        if (
            diagnostic_analysis["summary"]["bottom_group"] != "N/A"
            and predictive_analysis["status"] == "Conditional"
            and predictive_analysis["summary_cards"]["projected_direction"] in {"Down", "Flat"}
        ):
            recommendations.append(
                _recommendation(
                    category="Efficiency Improvements",
                    priority="High",
                    issue_detected="Weak segment performance with limited forecast confidence",
                    affected_group=diagnostic_analysis["summary"]["bottom_group"],
                    supporting_evidence=(
                        f"{diagnostic_analysis['summary']['bottom_group']} is the lowest contributor, and the forecast outlook is {predictive_analysis['summary_cards']['projected_direction'].lower()} with conditional confidence."
                    ),
                    suggested_action=(
                        f"Improve efficiency in {diagnostic_analysis['summary']['bottom_group']} first, because weaker segment performance is combining with an uncertain near-term outlook."
                    ),
                    recommendation_basis="combined",
                )
            )
        if (
            diagnostic_analysis["summary"]["concentration_classification"] in {"high", "moderate"}
            and any("volatile" in finding for finding in predictive_analysis["findings"])
        ):
            recommendations.append(
                _recommendation(
                    category="Risk Controls",
                    priority="High" if diagnostic_analysis["summary"]["concentration_classification"] == "high" else "Medium",
                    issue_detected="Concentrated performance with unstable trend",
                    affected_group=diagnostic_analysis["summary"]["top_group"],
                    supporting_evidence=(
                        f"{diagnostic_analysis['summary']['top_group']} leads current contribution, while the predictive layer flags recent instability."
                    ),
                    suggested_action=(
                        "Protect high-contribution groups while reducing operational dependence on a single unstable performance driver."
                    ),
                    recommendation_basis="combined",
                )
            )
        if (
            diagnostic_analysis["summary"]["spread_classification"] == "wide"
            and predictive_analysis["summary_cards"]["projected_direction"] == "Up"
        ):
            recommendations.append(
                _recommendation(
                    category="Growth Opportunities",
                    priority="Medium",
                    issue_detected="Forecasted growth with uneven segment performance",
                    affected_group=diagnostic_analysis["summary"]["bottom_group"],
                    supporting_evidence=(
                        f"The overall forecast points up, but {diagnostic_analysis['summary']['bottom_group']} remains materially behind the leading group."
                    ),
                    suggested_action=(
                        f"Use the improving outlook to selectively invest in lifting {diagnostic_analysis['summary']['bottom_group']} rather than concentrating all growth on already-strong segments."
                    ),
                    recommendation_basis="combined",
                )
            )

    if not recommendations:
        recommendations.append(
            _recommendation(
                category="Monitoring Priorities",
                priority="Low",
                issue_detected="Evidence is limited for stronger prescriptive actions",
                affected_group="Overall dataset",
                supporting_evidence=(
                    f"Prescriptive readiness is {prescriptive_readiness['status'].lower()}, and the current evidence does not justify stronger interventions."
                ),
                suggested_action=(
                    "Continue tracking the existing diagnostic and predictive signals before making more targeted decisions."
                ),
                recommendation_basis="diagnostic-based",
            )
        )

    return sorted(recommendations, key=lambda item: PRIORITY_RANK[item["priority"]], reverse=True)


def _build_margin_recommendation(diagnostic_analysis: dict[str, Any]) -> dict[str, str] | None:
    """Build a recommendation when discounting appears to weaken profit."""
    margin_finding = next(
        (finding for finding in diagnostic_analysis["findings"] if finding.startswith("Margin concern:")),
        None,
    )
    if margin_finding is None:
        return None
    affected_group = margin_finding.replace("Margin concern:", "").split(" show")[0].strip()
    return _recommendation(
        category="Efficiency Improvements",
        priority="High",
        issue_detected="Discount pressure is weakening profit",
        affected_group=affected_group,
        supporting_evidence=margin_finding,
        suggested_action=(
            "Review discounting rules, pricing discipline, and segment-level margin guardrails before expanding promotions."
        ),
        recommendation_basis="diagnostic-based",
    )


def _build_prescriptive_summary(recommendations: list[dict[str, str]]) -> dict[str, Any]:
    """Build summary KPIs for the prescriptive view."""
    category_counts = Counter(item["category"] for item in recommendations)
    basis_mix = Counter(item["recommendation_basis"] for item in recommendations)
    highest_priority = max(recommendations, key=lambda item: PRIORITY_RANK[item["priority"]])
    return {
        "total_recommendations": len(recommendations),
        "highest_priority_issue": highest_priority["issue_detected"],
        "category_counts": dict(category_counts),
        "basis_mix": dict(basis_mix),
    }


def _build_prescriptive_findings(
    recommendations: list[dict[str, str]],
    prescriptive_readiness: dict[str, str],
    predictive_analysis: dict[str, Any] | None,
    predictive_limitations: list[str],
) -> list[str]:
    """Build concise deterministic findings for the prescriptive tab."""
    findings = [f"Prescriptive readiness is {prescriptive_readiness['status'].lower()}, so recommendation strength stays conservative."]
    top_priority = max(recommendations, key=lambda item: PRIORITY_RANK[item["priority"]])
    findings.append(f"The highest-priority issue is {top_priority['issue_detected'].lower()}.")
    category_counts = Counter(item["category"] for item in recommendations)
    active_categories = [category for category, count in category_counts.items() if count > 0]
    findings.append("Recommendation coverage spans " + ", ".join(sorted(active_categories)) + ".")
    basis_counts = Counter(item["recommendation_basis"] for item in recommendations)
    if basis_counts.get("combined", 0) > 0:
        findings.append(
            f"{basis_counts['combined']} recommendation(s) are backed by both current-state and forecast evidence."
        )
    if predictive_analysis is not None and predictive_analysis["status"] != "Not Ready":
        findings.append(f"Predictive support is available and currently points {predictive_analysis['summary_cards']['projected_direction'].lower()} in the short term.")
    elif predictive_limitations:
        findings.append("Predictive support is limited, so recommendations rely more heavily on diagnostic evidence.")
    return findings


def _build_limitation_messages(
    prescriptive_readiness: dict[str, str],
    predictive_analysis: dict[str, Any] | None,
    predictive_limitations: list[str],
) -> list[str]:
    """Build professional limitation messages for the prescriptive layer."""
    messages: list[str] = []
    if prescriptive_readiness["status"] == "Conditional":
        messages.append(
            "Recommendations are conditional because the evidence base is still maturing."
        )
    if predictive_analysis is None or predictive_analysis["status"] == "Not Ready":
        messages.append(
            "Predictive support is limited or unavailable, so recommendations lean more heavily on current-state diagnostic evidence."
        )
    if predictive_limitations:
        messages.extend(predictive_limitations[:2])
    return messages


def _build_limited_prescriptive_result(
    readiness: dict[str, str],
    limitation_messages: list[str],
) -> dict[str, Any]:
    """Return a professional limited prescriptive result when support is weak."""
    return {
        "status": readiness["status"],
        "status_reason": readiness["reason"],
        "summary": {
            "total_recommendations": 0,
            "highest_priority_issue": "No recommendation triggered",
            "category_counts": {},
            "basis_mix": {},
        },
        "recommendations": [],
        "findings": [
            "Prescriptive recommendations are currently limited because the supporting evidence path is not strong enough yet."
        ],
        "diagnostic_analysis": None,
        "predictive_analysis": None,
        "limitation_messages": limitation_messages,
    }


def _recommendation(
    category: str,
    priority: str,
    issue_detected: str,
    affected_group: str,
    supporting_evidence: str,
    suggested_action: str,
    recommendation_basis: str,
) -> dict[str, str]:
    """Return a normalized recommendation payload."""
    return {
        "category": category,
        "priority": priority,
        "issue_detected": issue_detected,
        "affected_group": affected_group,
        "supporting_evidence": supporting_evidence,
        "suggested_action": suggested_action,
        "recommendation_basis": recommendation_basis,
    }
