from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ruddy import TabularDataset
from ruddy.profiling import (
    pairwise_completeness,
    profile_columns,
    summarize_missingness,
    summarize_missingness_patterns,
)


def _row(table: pl.DataFrame, name: str) -> dict[str, Any]:
    return table.filter(pl.col("column") == name).row(0, named=True)


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
    table = summarize_missingness(profile_columns(_dataset()))
    row_a = _row(table, "a")
    row_c = _row(table, "c")
    assert row_a["n_missing"] == 1
    assert row_a["finite_count"] == 3
    assert row_c["n_missing"] == 0
    assert row_c["non_finite_count"] == 1


def test_pairwise_completeness_uses_missing_values_not_infinity() -> None:
    table = pairwise_completeness(_dataset())
    ab = table.filter((pl.col("column_x") == "a") & (pl.col("column_y") == "b")).row(0, named=True)
    cc = table.filter((pl.col("column_x") == "c") & (pl.col("column_y") == "c")).row(0, named=True)
    assert ab["n_complete"] == 2
    assert ab["fraction_complete"] == pytest.approx(0.5)
    assert cc["n_complete"] == 4


def test_missingness_patterns_are_deterministic_and_bounded() -> None:
    table = summarize_missingness_patterns(_dataset(), max_patterns=2)
    assert table.height == 2
    row0 = table.row(0, named=True)
    assert bool(row0["patterns_truncated"]) is True
    patterns = [json.loads(value) for value in table["missing_columns"].to_list()]
    assert [] in patterns
    assert row0["count"] == 2
    assert row0["unreported_count"] == 1
