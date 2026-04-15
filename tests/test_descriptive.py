"""Placeholder tests for descriptive analytics."""

import pandas as pd

from app.analytics.descriptive import summarize_dataframe


def test_summarize_dataframe_returns_shape_summary():
    dataframe = pd.DataFrame({"value": [1, 2, 3]})
    result = summarize_dataframe(dataframe)
    assert result["rows"] == 3
    assert result["columns"] == 1

