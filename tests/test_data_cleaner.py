"""Placeholder tests for data cleaning."""

import pandas as pd

from app.core.data_cleaner import clean_dataframe


def test_clean_dataframe_returns_copy():
    dataframe = pd.DataFrame({"value": [1, 2]})
    result = clean_dataframe(dataframe)
    assert result.equals(dataframe)
    assert result is not dataframe

