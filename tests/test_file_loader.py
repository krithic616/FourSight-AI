"""Placeholder tests for file loading."""

from app.core.file_loader import load_file


def test_load_file_returns_path_object():
    result = load_file("sample.csv")
    assert result.name == "sample.csv"

