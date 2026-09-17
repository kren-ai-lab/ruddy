"""Frozen numerical parity against the Phase 0 PUMA reference corpus."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ruddy import TabularDataset
from ruddy.profiling import profile_columns
from ruddy.univariate import summarize_categorical_statistics, summarize_numeric_statistics

HERE = Path(__file__).parent


def _expected() -> dict:
    return json.loads((HERE / "reference" / "puma_univariate.json").read_text())


def _numeric_equal(left: pd.Series, right: pd.Series) -> bool:
    a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    normalized = right.map(lambda value: np.nan if value == "NaN" else value)
    b = pd.to_numeric(normalized, errors="coerce").to_numpy(dtype=float)
    return bool(np.allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True))


def test_shared_univariate_metrics_match_frozen_puma_reference() -> None:
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
    expected_numeric = pd.DataFrame(expected["numeric_statistics"])
    expected_categorical = pd.DataFrame(expected["categorical_statistics"])
    expected_frequencies = pd.DataFrame(expected["categorical_frequencies"])

    numeric = numeric[numeric["column"].isin(expected_numeric["column"])].reset_index(drop=True)
    for column in [c for c in expected_numeric.columns if c in numeric.columns and c != "role"]:
        if column in {"column", "status", "reason"}:
            assert (
                numeric[column].fillna("<NA>").astype(str).tolist()
                == expected_numeric[column].fillna("<NA>").astype(str).tolist()
            )
        else:
            assert _numeric_equal(numeric[column], expected_numeric[column]), column

    categorical = categorical[categorical["column"].isin(expected_categorical["column"])].reset_index(
        drop=True
    )
    for column in [c for c in expected_categorical.columns if c in categorical.columns and c != "role"]:
        if column in {"column", "mode", "status", "reason"}:
            assert (
                categorical[column].fillna("<NA>").astype(str).tolist()
                == expected_categorical[column].fillna("<NA>").astype(str).tolist()
            )
        elif column == "frequencies_truncated":
            assert (
                categorical[column].astype(bool).tolist()
                == expected_categorical[column].astype(bool).tolist()
            )
        else:
            assert _numeric_equal(categorical[column], expected_categorical[column]), column

    assert len(frequencies) == len(expected_frequencies)
    for column in [c for c in expected_frequencies.columns if c in frequencies.columns and c != "role"]:
        if column in {"column", "level"}:
            assert (
                frequencies[column].astype(str).tolist() == expected_frequencies[column].astype(str).tolist()
            )
        else:
            assert _numeric_equal(frequencies[column], expected_frequencies[column]), column
