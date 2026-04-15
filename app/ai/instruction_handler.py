"""Instruction handling helpers for the AI analyst layer."""

from __future__ import annotations


DEFAULT_AI_INSTRUCTION = "Summarize descriptive, diagnostic, predictive, and prescriptive insights."
COMPACT_CONTEXT_PROMPTS = {
    "key business insights",
    "top risks",
    "recommended actions",
    "give 3 business insights",
    "give 5 business insights",
}


def normalize_instruction(instruction: str) -> str:
    """Return a normalized instruction string with a safe fallback."""
    cleaned_instruction = instruction.strip()
    if not cleaned_instruction:
        return DEFAULT_AI_INSTRUCTION
    return cleaned_instruction


def should_use_compact_context(instruction: str) -> bool:
    """Return whether the instruction should force the ultra-compact context path."""
    normalized = normalize_instruction(instruction).strip().lower().rstrip(".?!")
    return normalized in COMPACT_CONTEXT_PROMPTS
