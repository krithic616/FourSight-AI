"""AI output cleanup and quality control helpers."""

from __future__ import annotations

import re
from typing import Any


SECTION_ORDER = [
    "Executive Summary",
    "Key Insights",
    "Risks",
    "Recommended Actions",
]
SECTION_LOOKUP = {section.lower(): section for section in SECTION_ORDER}
DEFAULT_SECTION_FOR_BULLETS = "Key Insights"


def draft_report_title() -> str:
    """Return a placeholder report title."""
    return "FourSight AI Report"


def prepare_ai_summary(ai_response: str) -> dict[str, Any]:
    """Clean, structure, and quality-check AI output before display or reporting."""
    cleaned_text = cleanup_ai_text(ai_response)
    parsed_sections = _parse_sections(cleaned_text)
    structured_sections = _normalize_sections(parsed_sections)
    flat_items = flatten_ai_sections(structured_sections)
    quality = assess_ai_summary_quality(flat_items)

    return {
        "cleaned_text": cleaned_text,
        "sections": structured_sections,
        "items": flat_items,
        "include_in_report": quality["passes"],
        "quality": quality,
    }


def cleanup_ai_text(ai_response: str) -> str:
    """Apply safe cleanup to raw AI output without changing its meaning."""
    text = str(ai_response or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^[\-\*\u2022]\s*", "- ", text)
    text = re.sub(r"(?m)^([A-Z])([a-z]+)([A-Z][a-z]+):\s*$", r"\1\2 \3:", text)
    cleaned_lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue
        line = re.sub(r"\s+([,.;:])", r"\1", line)
        line = re.sub(r"([:;,.!?]){2,}", r"\1", line)
        line = _fix_broken_capitalization(line)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def flatten_ai_sections(sections: dict[str, list[str]]) -> list[str]:
    """Convert structured AI sections into report-friendly flat lines."""
    items: list[str] = []
    for section_name in SECTION_ORDER:
        section_items = sections.get(section_name, [])
        if not section_items:
            continue
        items.append(section_name)
        items.extend(section_items)
    return items


def assess_ai_summary_quality(items: list[str]) -> dict[str, Any]:
    """Apply lightweight quality gates before including AI text in reports."""
    content_items = [item for item in items if item not in SECTION_ORDER]
    if not content_items:
        return {"passes": False, "reason": "No usable AI content was produced."}

    unique_items = []
    seen: set[str] = set()
    for item in content_items:
        normalized = _normalize_for_comparison(item)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_items.append(item)

    if len(unique_items) < 2:
        return {"passes": False, "reason": "AI output is too thin to include in the report."}

    malformed_count = sum(1 for item in unique_items if _looks_malformed(item))
    if malformed_count >= max(2, len(unique_items) // 2):
        return {"passes": False, "reason": "AI output looks malformed after cleanup."}

    return {"passes": True, "reason": ""}


def build_short_ai_summary(items: list[str], max_items: int = 4) -> list[str]:
    """Return a shorter cleaned AI summary when the full output should not be used."""
    content_items = [item for item in items if item not in SECTION_ORDER]
    return content_items[:max_items]


def _parse_sections(cleaned_text: str) -> dict[str, list[str]]:
    """Parse sectioned AI output and recover bullets when headings are imperfect."""
    sections: dict[str, list[str]] = {section: [] for section in SECTION_ORDER}
    current_section = DEFAULT_SECTION_FOR_BULLETS

    for raw_line in cleaned_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        maybe_section = _match_section_heading(line)
        if maybe_section:
            current_section = maybe_section
            continue

        content = _normalize_content_line(line)
        if content:
            sections[current_section].append(content)

    return sections


def _normalize_sections(sections: dict[str, list[str]]) -> dict[str, list[str]]:
    """Deduplicate, trim, and cap items inside each report section."""
    section_limits = {
        "Executive Summary": 1,
        "Key Insights": 3,
        "Risks": 2,
        "Recommended Actions": 3,
    }
    normalized_sections: dict[str, list[str]] = {}
    for section_name in SECTION_ORDER:
        deduped_items: list[str] = []
        seen: set[str] = set()
        for item in sections.get(section_name, []):
            normalized_item = _normalize_for_comparison(item)
            if not normalized_item or normalized_item in seen:
                continue
            seen.add(normalized_item)
            deduped_items.append(item)
        normalized_sections[section_name] = deduped_items[: section_limits[section_name]]
    return normalized_sections


def _match_section_heading(line: str) -> str | None:
    """Map loose heading text to the supported executive report sections."""
    heading = line.strip().strip("#").strip("-").strip().rstrip(":").lower()
    return SECTION_LOOKUP.get(heading)


def _normalize_content_line(line: str) -> str:
    """Normalize one bullet or sentence into a short clean report line."""
    content = line.lstrip("-* ").strip()
    content = re.sub(r"^\d+\.\s*", "", content)
    content = re.sub(r"^(executive summary|key insights|risks|recommended actions)\s*:\s*", "", content, flags=re.I)
    content = re.sub(r"\s+", " ", content).strip(" -")
    if not content:
        return ""
    return _ensure_sentence_case(content)


def _ensure_sentence_case(text: str) -> str:
    """Keep text readable without aggressively rewriting the model output."""
    if not text:
        return text
    first_char = text[0]
    if first_char.isalpha():
        text = first_char.upper() + text[1:]
    return text


def _fix_broken_capitalization(text: str) -> str:
    """Repair simple broken heading capitalization patterns."""
    for section_name in SECTION_ORDER:
        squashed = section_name.replace(" ", "")
        if text.lower().rstrip(":") == squashed.lower():
            return f"{section_name}:"
    return text


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for duplicate detection."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _looks_malformed(text: str) -> bool:
    """Flag obviously poor AI lines while staying conservative."""
    if len(text) < 8:
        return True
    if re.search(r"(.)\1{4,}", text):
        return True
    alpha_ratio = sum(char.isalpha() for char in text) / max(len(text), 1)
    if alpha_ratio < 0.45:
        return True
    return False
