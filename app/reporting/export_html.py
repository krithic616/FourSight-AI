"""HTML export helpers for deterministic reports."""

from __future__ import annotations

from html import escape
from typing import Any


def export_html(report: dict[str, Any]) -> str:
    """Convert the report payload into a lightweight HTML report."""
    sections_markup: list[str] = []
    for section in report["sections"]:
        is_recommendation_section = str(section["title"]) == "Key Recommendations"
        item_class = "recommendation-item" if is_recommendation_section else "report-item"
        items_markup = "".join(
            f"<li class='{item_class}'>{escape(str(item))}</li>"
            for item in section["items"]
        )
        sections_markup.append(
            "<section class='report-section'>"
            f"<h2>{escape(str(section['title']))}</h2>"
            f"<ul>{items_markup}</ul>"
            "</section>"
        )

    return (
        "<!DOCTYPE html>"
        "<html lang='en'>"
        "<head>"
        "<meta charset='utf-8' />"
        "<meta name='viewport' content='width=device-width, initial-scale=1' />"
        f"<title>{escape(str(report['title']))}</title>"
        "<style>"
        "body { background: #0f1116; color: #e8eaed; font-family: Arial, sans-serif; margin: 0; padding: 32px; }"
        ".report-shell { max-width: 900px; margin: 0 auto; }"
        "h1 { margin-bottom: 8px; }"
        ".meta-card { background: #151922; border: 1px solid #273142; border-radius: 12px; padding: 16px 18px; color: #c7d0db; margin-bottom: 20px; }"
        ".meta { color: #b8c0cc; line-height: 1.6; }"
        ".report-section { background: #171b22; border: 1px solid #273142; border-radius: 12px; padding: 18px 20px; margin-bottom: 16px; }"
        "h2 { margin-top: 0; color: #f3f6fa; }"
        "ul { margin: 0; padding-left: 18px; }"
        "li { margin-bottom: 8px; line-height: 1.5; }"
        ".recommendation-item { border-left: 3px solid #6ea8fe; padding-left: 10px; margin-bottom: 12px; }"
        ".report-item { color: #e8eaed; }"
        "</style>"
        "</head>"
        "<body>"
        "<div class='report-shell'>"
        f"<h1>{escape(str(report['title']))}</h1>"
        "<div class='meta-card'>"
        f"<div class='meta'><strong>Dataset:</strong> {escape(str(report['dataset_name']))}<br/>"
        f"<strong>Generated:</strong> {escape(str(report['generated_at']))}</div>"
        "</div>"
        + "".join(sections_markup)
        + "</div></body></html>"
    )
