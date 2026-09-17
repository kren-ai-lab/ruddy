from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ruddy import ColumnRole, TabularDataset
from ruddy.profiling import (
    COLUMN_PROFILE_COLUMNS,
    profile_columns,
    profile_dataset,
    summarize_overview,
)


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
    assert columns["column"].tolist() == list(dataset.columns)
    table = columns.set_index("column")
    assert table.loc["id", "role"] == "identifier"
    assert table.loc["id", "data_kind"] == "categorical"
    assert table.loc["response", "role"] == "response"
    assert table.loc["response", "data_kind"] == "numeric"
    assert table.loc["factor_code", "role"] == "factor"
    assert table.loc["factor_code", "data_kind"] == "numeric"
    assert table.loc["flag", "data_kind"] == "boolean"
    assert table.loc["when", "data_kind"] == "datetime"


def test_profile_counts_missing_nonfinite_constant_and_all_missing() -> None:
    table = profile_columns(_dataset()).set_index("column")
    row = table.loc["measurement"]
    assert row["n_total"] == 4
    assert row["n_missing"] == 1
    assert row["finite_count"] == 2
    assert row["non_finite_count"] == 1
    assert bool(table.loc["constant", "is_constant"]) is True
    assert bool(table.loc["all_missing", "is_all_missing"]) is True
    assert bool(table.loc["excluded", "analysis_eligible"]) is False


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
    assert overview["identifier"]["source"] == "index"
    assert overview["identifier"]["n_duplicate_records"] == 0


def test_profile_dataset_is_non_destructive_and_has_provenance() -> None:
    dataset = _dataset()
    before = dataset.to_frame()
    result = profile_dataset(dataset)

    pd.testing.assert_frame_equal(dataset.to_frame(), before)
    assert result.provenance.analysis == "profiling"
    assert result.provenance.parameters["numeric_non_finite_policy"] == "count_separately_from_missing"
    assert len(result.pairwise_completeness) == len(dataset.columns) * (len(dataset.columns) + 1) // 2
    assert not result.missingness_patterns.empty


def test_timedelta_columns_are_not_silently_treated_as_numeric() -> None:
    frame = pd.DataFrame({"delta": pd.to_timedelta(["1 day", "2 days"])})
    dataset = TabularDataset(frame)
    table = profile_columns(dataset).set_index("column")
    assert table.loc["delta", "data_kind"] == "unknown"
    assert bool(table.loc["delta", "analysis_eligible"]) is False


def test_complex_columns_are_rejected_explicitly() -> None:
    frame = pd.DataFrame({"complex": np.asarray([1 + 2j, 3 + 4j])})
    with pytest.raises(TypeError, match="cannot represent"):
        TabularDataset(frame)


def test_pairwise_column_guard_is_explicit() -> None:
    with pytest.raises(ValueError, match="exceeding"):
        profile_dataset(_dataset(), max_pairwise_columns=2)
