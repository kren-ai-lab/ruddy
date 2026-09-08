from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ruddy import TabularDataset, analyze_outliers


def _summary(result, column: str, method: str) -> pd.Series:
    return result.summaries.loc[
        result.summaries["column"].eq(column)
        & result.summaries["method"].eq(method)
    ].iloc[0]


def test_missing_nonfinite_and_finite_counts_are_transparent() -> None:
    frame = pd.DataFrame(
        {
            "id": list("abcdefg"),
            "x": [0.0, 1.0, np.nan, np.inf, -np.inf, 2.0, 100.0],
        }
    )
    result = analyze_outliers(TabularDataset(frame, id_column="id"), include_flags=True)
    quality = result.quality.set_index("column").loc["x"]
    assert quality["n_total"] == 7
    assert quality["n_finite"] == 4
    assert quality["n_missing"] == 1
    assert quality["n_non_finite"] == 2
    assert bool(quality["has_missing"])
    assert bool(quality["has_non_finite"])
    for method in ("iqr", "robust_z"):
        row = _summary(result, "x", method)
        assert row["n_finite"] == 4
        flags = result.flags.loc[(result.flags["column"] == "x") & (result.flags["method"] == method)]
        assert np.isfinite(flags["value"].to_numpy(dtype=float)).all()
        assert not set(flags["source_row_index"].astype(int)) & {2, 3, 4}


def test_constant_all_missing_no_finite_and_insufficient_are_explicit() -> None:
    frame = pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(6)],
            "constant": [4.0] * 6,
            "all_missing": pd.Series([np.nan] * 6, dtype=float),
            "no_finite": [np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf],
            "tiny": [1.0, 2.0, np.nan, np.nan, np.nan, np.nan],
        }
    )
    result = analyze_outliers(TabularDataset(frame, id_column="id"), min_numeric_n=3)
    for method in ("iqr", "robust_z"):
        assert _summary(result, "constant", method)["reason"] == "constant"
        assert _summary(result, "constant", method)["status"] == "degenerate"
        assert _summary(result, "all_missing", method)["reason"] == "all_missing"
        assert _summary(result, "all_missing", method)["status"] == "skipped"
        assert _summary(result, "no_finite", method)["reason"] == "no_finite_values"
        assert _summary(result, "tiny", method)["reason"] == "insufficient_finite_observations"


def test_zero_mad_is_degenerate_but_zero_iqr_nonconstant_remains_valid_iqr_rule() -> None:
    frame = pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(6)],
            "x": [0.0, 0.0, 0.0, 0.0, 0.0, 10.0],
        }
    )
    result = analyze_outliers(TabularDataset(frame, id_column="id"), include_flags=True)
    robust = _summary(result, "x", "robust_z")
    assert robust["status"] == "degenerate"
    assert robust["reason"] == "zero_mad"
    assert pd.isna(robust["n_flagged"])

    iqr = _summary(result, "x", "iqr")
    assert iqr["status"] == "ok"
    assert iqr["scale"] == pytest.approx(0.0)
    assert iqr["lower_bound"] == pytest.approx(0.0)
    assert iqr["upper_bound"] == pytest.approx(0.0)
    assert iqr["n_flagged"] == 1

    quality = result.quality.set_index("column").loc["x"]
    assert bool(quality["zero_iqr_nonconstant"])
    assert bool(quality["zero_mad_nonconstant"])
