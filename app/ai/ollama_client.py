"""Local Ollama client helpers."""

from __future__ import annotations

import json
import os
import re
from urllib import error, request


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


def get_ollama_status(timeout_seconds: int = 5) -> dict[str, object]:
    """Return local Ollama availability and detected models."""
    try:
        payload = _http_request("/api/tags", method="GET", timeout_seconds=timeout_seconds)
    except RuntimeError as exc:
        return {
            "available": False,
            "message": str(exc),
            "models": [],
            "default_model": "phi3:mini",
        }

    models = [model.get("name", "") for model in payload.get("models", []) if model.get("name")]
    default_model = _select_default_model(models)
    return {
        "available": True,
        "message": "Ollama is reachable locally.",
        "models": models,
        "default_model": default_model,
    }


def generate_ollama_response(
    model: str,
    prompt: str,
    timeout_seconds: int = 180,
) -> dict[str, object]:
    """Generate a non-streaming response from a local Ollama model."""
    if not model.strip():
        return {
            "success": False,
            "response": "",
            "error": "A model name is required before generating a response.",
        }

    payload = {
        "model": model.strip(),
        "prompt": prompt,
        "stream": False,
    }

    try:
        result = _http_request(
            "/api/generate",
            method="POST",
            body=payload,
            timeout_seconds=timeout_seconds,
        )
    except RuntimeError as exc:
        error_message = str(exc)
        return {
            "success": False,
            "response": "",
            "error": error_message,
            "error_type": "insufficient_memory" if _is_memory_error_message(error_message) else "generation_failed",
        }

    return {
        "success": True,
        "response": str(result.get("response", "")).strip(),
        "error": "",
        "error_type": "",
    }


def _http_request(
    path: str,
    method: str,
    timeout_seconds: int,
    body: dict[str, object] | None = None,
) -> dict[str, object]:
    """Execute a JSON request against the local Ollama server."""
    target_url = f"{OLLAMA_BASE_URL.rstrip('/')}{path}"
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")

    http_request = request.Request(
        url=target_url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="ignore")
        ollama_error = _extract_ollama_error_message(response_body)
        if _is_memory_error_message(ollama_error):
            raise RuntimeError(
                "The selected local model requires more memory than is currently available on this system. FourSight AI is showing a deterministic fallback summary instead. Try a smaller model if available."
            ) from exc
        raise RuntimeError(
            f"Ollama returned HTTP {exc.code}. {ollama_error or 'No additional details were provided.'}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(
            "Ollama is not reachable at http://127.0.0.1:11434. Start the local Ollama service and try again."
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            "The selected local model timed out after 180 seconds. This can happen on lower-memory systems or when the prompt context is still too large. The AI Analyst will fall back to a compact deterministic summary so you still have a usable result."
        ) from exc

    if not response_body.strip():
        return {}

    try:
        return json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned a response that could not be parsed as JSON.") from exc


def _select_default_model(models: list[str]) -> str:
    """Choose a sensible default model, preferring installed options."""
    if not models:
        return "phi3:mini"

    ordered_models = sorted(models, key=_model_lightness_key)
    preferred_order = [
        "tinyllama",
        "llama3.2:1b",
        "phi3:mini",
        "gemma2:2b",
        "gemma3n:e2b",
        "llama3.2:3b",
        "phi3",
        "llama3:latest",
    ]
    normalized_lookup = {model.lower(): model for model in ordered_models}
    for preferred in preferred_order:
        if preferred.lower() in normalized_lookup:
            return normalized_lookup[preferred.lower()]
    return ordered_models[0]


def _extract_ollama_error_message(response_body: str) -> str:
    """Return the most useful error string from an Ollama error response."""
    cleaned_body = response_body.strip()
    if not cleaned_body:
        return ""
    try:
        parsed_body = json.loads(cleaned_body)
    except json.JSONDecodeError:
        return cleaned_body
    if isinstance(parsed_body, dict):
        return str(parsed_body.get("error", "")).strip() or cleaned_body
    return cleaned_body


def _is_memory_error_message(message: str) -> bool:
    """Return whether an Ollama error message indicates insufficient memory."""
    normalized_message = str(message).lower()
    memory_patterns = [
        "requires more system memory",
        "requires more memory",
        "not enough memory",
        "insufficient memory",
        "out of memory",
        "model requires",
        "available memory",
        "system memory",
    ]
    if not any(pattern in normalized_message for pattern in memory_patterns):
        return False
    return "memory" in normalized_message


def _model_lightness_key(model_name: str) -> tuple[int, float, str]:
    """Rank models so lighter local options are preferred when practical."""
    normalized_name = model_name.strip().lower()
    family_priority = 5
    if "tinyllama" in normalized_name:
        family_priority = 0
    elif "1b" in normalized_name:
        family_priority = 1
    elif "mini" in normalized_name:
        family_priority = 2
    elif "2b" in normalized_name or "e2b" in normalized_name:
        family_priority = 3
    elif "3b" in normalized_name:
        family_priority = 4

    size_match = re.search(r"(\d+(?:\.\d+)?)\s*b", normalized_name)
    if size_match:
        size_score = float(size_match.group(1))
    elif "mini" in normalized_name or "tiny" in normalized_name:
        size_score = 0.5
    else:
        size_score = 99.0

    return (family_priority, size_score, normalized_name)
