from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ruddy.statistics import adjust_pvalues, apply_multiple_testing, family_sizes


def test_bh_reference_values_and_order() -> None:
    observed = adjust_pvalues(np.array([0.01, 0.04, 0.03, 0.20]), "fdr_bh")
    expected = np.array([0.04, 0.05333333333333334, 0.05333333333333334, 0.20])
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_none_returns_raw_pvalues() -> None:
    values = np.array([0.1, 0.02])
    np.testing.assert_array_equal(adjust_pvalues(values, "none"), values)


def test_family_adjustment_excludes_non_ok_rows_and_records_size() -> None:
    table = pd.DataFrame(
        {
            "family_id": ["a", "a", "a", "b"],
            "status": ["ok", "degenerate", "ok", "ok"],
            "p_value": [0.01, np.nan, 0.04, 0.20],
            "q_value": [np.nan] * 4,
            "correction": ["fdr_bh"] * 4,
        }
    )
    result = apply_multiple_testing(table, "fdr_bh")
    np.testing.assert_allclose(result.loc[[0, 2], "q_value"], [0.02, 0.04])
    assert np.isnan(result.loc[1, "q_value"])  # pyrefly: ignore[no-matching-overload]
    assert result["family_size"].tolist() == [2, 2, 2, 1]


def test_family_adjustment_polars_excludes_non_ok_rows_and_records_size() -> None:
    table = pl.DataFrame(
        {
            "family_id": ["a", "a", "a", "b"],
            "status": ["ok", "degenerate", "ok", "ok"],
            "p_value": [0.01, None, 0.04, 0.20],
            "correction": ["fdr_bh"] * 4,
        },
        schema={
            "family_id": pl.String,
            "status": pl.String,
            "p_value": pl.Float64,
            "correction": pl.String,
        },
    )
    result = apply_multiple_testing(table, "fdr_bh")
    q_vals = result["q_value"].to_list()
    assert q_vals[0] == pytest.approx(0.02)
    assert q_vals[1] is None
    assert q_vals[2] == pytest.approx(0.04)
    assert q_vals[3] == pytest.approx(0.20)
    assert result["family_size"].to_list() == [2, 2, 2, 1]


def test_family_sizes_polars_and_pandas() -> None:
    pl_table = pl.DataFrame(
        {
            "family_id": ["b", "a", "b", "c"],
            "status": ["ok", "ok", "skipped", "ok"],
        }
    )
    assert family_sizes(pl_table) == {"b": 1, "a": 1, "c": 1}
    assert family_sizes(pl.DataFrame(schema={"family_id": pl.String, "status": pl.String})) == {}


def test_polars_apply_multiple_testing_rejects_blank_family_id() -> None:
    table_blank = pl.DataFrame(
        {
            "family_id": ["a", "   "],
            "status": ["ok", "ok"],
            "p_value": [0.01, 0.04],
            "correction": ["fdr_bh", "fdr_bh"],
        }
    )
    with pytest.raises(ValueError, match="family_id values must be present and non-empty"):
        apply_multiple_testing(table_blank, "fdr_bh")

    table_null = pl.DataFrame(
        {
            "family_id": ["a", None],
            "status": ["ok", "ok"],
            "p_value": [0.01, 0.04],
            "correction": ["fdr_bh", "fdr_bh"],
        }
    )
    with pytest.raises(ValueError, match="family_id values must be present and non-empty"):
        apply_multiple_testing(table_null, "fdr_bh")


def test_polars_apply_multiple_testing_rejects_inconsistent_correction() -> None:
    table = pl.DataFrame(
        {
            "family_id": ["a", "b"],
            "status": ["ok", "ok"],
            "p_value": [0.01, 0.04],
            "correction": ["fdr_bh", "bonferroni"],
        }
    )
    with pytest.raises(ValueError, match="correction column is inconsistent"):
        apply_multiple_testing(table, "fdr_bh")


def test_polars_apply_multiple_testing_empty_table() -> None:
    table = pl.DataFrame(
        schema={
            "family_id": pl.String,
            "status": pl.String,
            "p_value": pl.Float64,
            "correction": pl.String,
        }
    )
    result = apply_multiple_testing(table, "fdr_bh")
    assert result.height == 0
    assert "q_value" in result.columns
    assert "family_size" in result.columns
    assert result["q_value"].dtype == pl.Float64
    assert result["family_size"].dtype == pl.Int64


def test_adjustment_rejects_nonfinite_pvalues() -> None:
    with pytest.raises(ValueError, match="finite"):
        adjust_pvalues(np.array([0.1, np.nan]), "fdr_bh")
