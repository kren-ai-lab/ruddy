from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import polars.testing as pl_testing
import pytest

from ruddy import ColumnRole, TabularDataset
from ruddy.profiling import (
    COLUMN_PROFILE_COLUMNS,
    COLUMN_PROFILE_SCHEMA,
    profile_columns,
    profile_dataset,
    summarize_overview,
)


def _row(table: pl.DataFrame, name: str) -> dict[str, Any]:
    return table.filter(pl.col("column") == name).row(0, named=True)


def _dataset() -> TabularDataset:
    frame = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "response": [1.0, 2.0, 3.0, 4.0],
            "factor_code": [1, 1, 2, 2],
            "measurement": [1.0, np.nan, np.inf, 4.0],
            "flag": [True, False, True, False],
            "when": pd.to_datetime(["2026-01-01", "2026-01-02", None, "2026-01-04"]),  # pyrefly: ignore[no-matching-overload]
            "constant": [7, 7, 7, 7],
            "all_missing": [np.nan, np.nan, np.nan, np.nan],
            "excluded": ["x", "y", "z", "q"],
        }
    )
    return TabularDataset(
        frame,
        id_column="id",
        role_overrides={
            "response": ColumnRole.RESPONSE,
            "factor_code": ColumnRole.FACTOR,
            "excluded": ColumnRole.EXCLUDED,
        },
    )


def test_profile_columns_preserves_order_and_separates_role_from_kind() -> None:
    dataset = _dataset()
    columns = profile_columns(dataset)

    assert tuple(columns.columns) == COLUMN_PROFILE_COLUMNS
    assert columns["column"].to_list() == list(dataset.columns)
    assert _row(columns, "id")["role"] == "identifier"
    assert _row(columns, "id")["data_kind"] == "categorical"
    assert _row(columns, "response")["role"] == "response"
    assert _row(columns, "response")["data_kind"] == "numeric"
    assert _row(columns, "factor_code")["role"] == "factor"
    assert _row(columns, "factor_code")["data_kind"] == "numeric"
    assert _row(columns, "flag")["data_kind"] == "boolean"
    assert _row(columns, "when")["data_kind"] == "datetime"


def test_profile_counts_missing_nonfinite_constant_and_all_missing() -> None:
    table = profile_columns(_dataset())
    row = _row(table, "measurement")
    assert row["n_total"] == 4
    assert row["n_missing"] == 1
    assert row["finite_count"] == 2
    assert row["non_finite_count"] == 1
    assert bool(_row(table, "constant")["is_constant"]) is True
    assert bool(_row(table, "all_missing")["is_all_missing"]) is True
    assert bool(_row(table, "excluded")["analysis_eligible"]) is False


def test_overview_reports_missingness_duplicates_and_identifier() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 1.0, 2.0],
            "group": ["A", "A", None],
        }
    )
    dataset = TabularDataset(frame)
    columns = profile_columns(dataset)
    overview = summarize_overview(dataset, columns)

    assert overview["n_records"] == 3
    assert overview["n_columns"] == 2
    assert overview["duplicates"]["n_duplicate_rows"] == 1
    assert overview["duplicates"]["n_records_in_duplicate_groups"] == 2
    assert overview["missingness"]["n_records_with_missing"] == 1
    assert overview["identifier"]["source"] == "generated"
    assert overview["identifier"]["n_duplicate_records"] == 0


def test_profile_dataset_is_non_destructive_and_has_provenance() -> None:
    dataset = _dataset()
    before = dataset.frame.clone()
    result = profile_dataset(dataset)

    pl_testing.assert_frame_equal(dataset.frame, before)
    assert result.provenance.analysis == "profiling"
    assert result.provenance.parameters["missingness_policy"] == "null_or_nan_missing_values"
    assert result.provenance.parameters["numeric_non_finite_policy"] == "count_separately_from_missing"
    assert result.pairwise_completeness.height == len(dataset.columns) * (len(dataset.columns) + 1) // 2
    assert result.missingness_patterns.height > 0


def test_timedelta_columns_are_not_silently_treated_as_numeric() -> None:
    frame = pd.DataFrame({"delta": pd.to_timedelta(["1 day", "2 days"])})
    dataset = TabularDataset(frame)
    table = profile_columns(dataset)
    assert _row(table, "delta")["data_kind"] == "unknown"
    assert bool(_row(table, "delta")["analysis_eligible"]) is False


def test_complex_columns_are_rejected_explicitly() -> None:
    frame = pd.DataFrame({"complex": np.asarray([1 + 2j, 3 + 4j])})
    with pytest.raises(TypeError, match="cannot represent"):
        TabularDataset(frame)


def test_pairwise_column_guard_is_explicit() -> None:
    with pytest.raises(ValueError, match="exceeding"):
        profile_dataset(_dataset(), max_pairwise_columns=2)


def test_profile_columns_float_with_none_nan_and_inf() -> None:
    dataset = TabularDataset(
        pl.DataFrame({"x": pl.Series([None, float("nan"), float("inf")], dtype=pl.Float64)})
    )
    table = profile_columns(dataset)
    row = _row(table, "x")
    assert row["n_missing"] == 2
    assert row["n_present"] == 1
    assert row["finite_count"] == 0
    assert row["non_finite_count"] == 1


def test_profile_columns_schema_matches_declared_schema() -> None:
    dataset = _dataset()
    table = profile_columns(dataset)
    assert table.schema == pl.Schema(COLUMN_PROFILE_SCHEMA)

    empty_dataset = TabularDataset(pl.DataFrame())
    empty_table = profile_columns(empty_dataset)
    assert empty_table.height == 0
    assert empty_table.schema == pl.Schema(COLUMN_PROFILE_SCHEMA)


def test_overview_identifier_source_generated_without_id_column() -> None:
    dataset = TabularDataset(pl.DataFrame({"val": [1, 2, 3]}))
    columns = profile_columns(dataset)
    overview = summarize_overview(dataset, columns)
    assert overview["identifier"]["source"] == "generated"
    assert overview["identifier"]["column"] is None
    assert overview["identifier"]["n_present"] == 3
    assert overview["identifier"]["n_unique"] == 3
    assert overview["identifier"]["n_duplicate_records"] == 0
