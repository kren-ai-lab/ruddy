from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ruddy import TabularDataset
from ruddy.profiling import (
    pairwise_completeness,
    profile_columns,
    summarize_missingness,
    summarize_missingness_patterns,
)


def _dataset() -> TabularDataset:
    return TabularDataset(
        pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0, 4.0],
                "b": [1.0, 2.0, np.nan, 4.0],
                "c": [np.inf, 2.0, 3.0, 4.0],
            }
        )
    )


def test_missingness_keeps_nonfinite_separate_from_missing() -> None:
    table = summarize_missingness(profile_columns(_dataset())).set_index("column")
    assert table.loc["a", "n_missing"] == 1
    assert table.loc["a", "finite_count"] == 3
    assert table.loc["c", "n_missing"] == 0
    assert table.loc["c", "non_finite_count"] == 1


def test_pairwise_completeness_uses_missing_values_not_infinity() -> None:
    table = pairwise_completeness(_dataset())
    ab = table.loc[(table["column_x"] == "a") & (table["column_y"] == "b")].iloc[0]
    cc = table.loc[(table["column_x"] == "c") & (table["column_y"] == "c")].iloc[0]
    assert ab["n_complete"] == 2
    assert ab["fraction_complete"] == pytest.approx(0.5)
    assert cc["n_complete"] == 4


def test_missingness_patterns_are_deterministic_and_bounded() -> None:
    table = summarize_missingness_patterns(_dataset(), max_patterns=2)
    assert len(table) == 2
    assert bool(table.iloc[0]["patterns_truncated"]) is True
    patterns = [json.loads(value) for value in table["missing_columns"]]
    assert [] in patterns
    assert table.iloc[0]["count"] == 2
    assert table.iloc[0]["unreported_count"] == 1
