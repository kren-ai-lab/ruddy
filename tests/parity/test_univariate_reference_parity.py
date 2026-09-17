"""Frozen numerical parity against Ruddy's reference corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ruddy import TabularDataset
from ruddy.profiling import profile_columns
from ruddy.univariate import summarize_categorical_statistics, summarize_numeric_statistics

HERE = Path(__file__).parent


def _expected() -> dict[str, Any]:
    return json.loads((HERE / "reference" / "univariate_reference.json").read_text())


def _as_float(value) -> float:
    return np.nan if value is None or value == "NaN" else float(value)


def _compare_column(observed_list: list[Any], reference_list: list[Any], col_name: str) -> None:
    if col_name in {"column", "mode", "level", "status", "reason"}:
        norm_obs = [None if v is None or v == "NaN" else str(v) for v in observed_list]
        norm_ref = [None if v is None or v == "NaN" else str(v) for v in reference_list]
        assert norm_obs == norm_ref, f"Mismatch in string column {col_name}"
    elif col_name == "frequencies_truncated":
        assert [bool(v) for v in observed_list] == [bool(v) for v in reference_list], col_name
    else:
        obs_arr = np.array([_as_float(v) for v in observed_list], dtype=float)
        ref_arr = np.array([_as_float(v) for v in reference_list], dtype=float)
        assert np.allclose(obs_arr, ref_arr, rtol=1e-12, atol=1e-12, equal_nan=True), col_name


def test_shared_univariate_metrics_match_frozen_reference() -> None:
    frame = pd.read_csv(HERE / "fixtures" / "mixed_tabular.csv")
    dataset = TabularDataset(frame, id_column="observation_id")
    columns = profile_columns(dataset)
    numeric = summarize_numeric_statistics(dataset, columns)
    categorical, frequencies = summarize_categorical_statistics(
        dataset,
        columns,
        max_category_levels=5,
    )
    expected = _expected()
    expected_numeric_rows = expected["numeric_statistics"]
    expected_categorical_rows = expected["categorical_statistics"]
    expected_frequencies_rows = expected["categorical_frequencies"]

    expected_numeric_cols = [r["column"] for r in expected_numeric_rows]
    numeric = numeric.filter(pl.col("column").is_in(expected_numeric_cols))
    for column in sorted(c for c in expected_numeric_rows[0] if c in numeric.columns and c != "role"):
        _compare_column(
            numeric.get_column(column).to_list(),
            [r[column] for r in expected_numeric_rows],
            column,
        )

    expected_categorical_cols = [r["column"] for r in expected_categorical_rows]
    categorical = categorical.filter(pl.col("column").is_in(expected_categorical_cols))
    for column in sorted(c for c in expected_categorical_rows[0] if c in categorical.columns and c != "role"):
        _compare_column(
            categorical.get_column(column).to_list(),
            [r[column] for r in expected_categorical_rows],
            column,
        )

    assert frequencies.height == len(expected_frequencies_rows)
    for column in sorted(c for c in expected_frequencies_rows[0] if c in frequencies.columns and c != "role"):
        _compare_column(
            frequencies.get_column(column).to_list(),
            [r[column] for r in expected_frequencies_rows],
            column,
        )
