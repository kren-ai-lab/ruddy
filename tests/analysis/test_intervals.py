from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import polars.testing

from ruddy import TabularDataset, analyze_confidence_intervals


def _dataset(*, zero_cell: bool = False) -> TabularDataset:
    rng = np.random.default_rng(10)
    n = 60
    group = np.repeat(["A", "B"], 30)
    x = rng.normal(loc=np.where(group == "A", 0.0, 1.0), scale=1.0)
    y = 0.6 * x + rng.normal(scale=0.7, size=n)
    if zero_cell:
        c = np.where(group == "A", "X", "Y")
    else:
        c = np.array(["X"] * 20 + ["Y"] * 10 + ["X"] * 8 + ["Y"] * 22)
    frame = pl.DataFrame({"id": list(range(n)), "x": x, "y": y, "group": group, "c": c})
    return TabularDataset(frame, id_column="id", role_overrides={"group": "factor"})


def test_interval_analysis_contains_core_estimands() -> None:
    result = analyze_confidence_intervals(
        _dataset(), bootstrap_resamples=150, bootstrap_method="percentile", random_state=2
    )
    assert set(result.means.get_column("column")) == {"x", "y"}
    assert result.correlations.height == 1
    assert not result.mean_differences.is_empty()
    assert not result.effect_sizes.is_empty()
    assert not result.odds_ratios.is_empty()


def test_pearson_interval_contains_estimate() -> None:
    result = analyze_confidence_intervals(_dataset(), bootstrap_resamples=100, bootstrap_method="percentile")
    row = result.correlations.row(0, named=True)
    assert row["ci_method"] == "fisher_z"
    assert row["confidence_low"] < row["estimate"] < row["confidence_high"]


def test_spearman_interval_bootstrap_is_deterministic() -> None:
    kwargs: dict[str, Any] = {
        "correlation_methods": ("spearman",),
        "bootstrap_resamples": 120,
        "bootstrap_method": "percentile",
        "random_state": 8,
    }
    a = analyze_confidence_intervals(_dataset(), **kwargs)
    b = analyze_confidence_intervals(_dataset(), **kwargs)
    polars.testing.assert_frame_equal(a.correlations, b.correlations)


def test_zero_cell_odds_ratio_is_explicitly_degenerate() -> None:
    result = analyze_confidence_intervals(
        _dataset(zero_cell=True), bootstrap_resamples=100, bootstrap_method="percentile"
    )
    rows = result.odds_ratios.filter((pl.col("column_x") == "group") | (pl.col("column_y") == "group"))
    assert (rows.get_column("status") == "degenerate").any()
    assert "zero_cell_or_invalid_table" in set(rows.get_column("reason").drop_nulls())


def test_skipped_mean_has_null_estimate_and_retains_n() -> None:
    frame = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "x": [1.0, 2.0, 3.0],
            "sparse": [42.0, None, None],
        }
    )
    ds = TabularDataset(frame, id_column="id")
    result = analyze_confidence_intervals(ds)
    sparse_row = result.means.filter(pl.col("column") == "sparse").row(0, named=True)
    assert sparse_row["status"] == "skipped"
    assert sparse_row["reason"] == "insufficient_finite_observations"
    assert sparse_row["estimate"] is None
    assert sparse_row["n"] == 1


def test_odds_ratio_table_from_polars_dataset_has_float64_estimate() -> None:
    frame = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "cat1": ["A", "A", "B", "B"],
            "cat2": ["X", "Y", "X", "Y"],
        }
    )
    ds = TabularDataset(
        frame,
        id_column="id",
        role_overrides={"cat1": "factor", "cat2": "factor"},
    )
    result = analyze_confidence_intervals(ds)
    assert result.odds_ratios.schema["estimate"] == pl.Float64
