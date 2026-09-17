from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ruddy import TabularDataset, analyze_dependence, distance_correlation
from ruddy.bivariate.dependence import summarize_general_dependence, summarize_partial_correlations


def test_distance_correlation_invariants() -> None:
    x = np.linspace(-2, 2, 51)
    assert distance_correlation(x, x) == pytest.approx(1.0)
    assert 0.0 <= distance_correlation(x, x * x) <= 1.0
    assert distance_correlation(x, x * x) > 0.4


def test_partial_correlation_removes_shared_covariate() -> None:
    rng = np.random.default_rng(4)
    n = 500
    z = rng.normal(size=n)
    x = z + rng.normal(size=n)
    y = z + rng.normal(size=n)
    ds = TabularDataset(
        pd.DataFrame({"id": range(n), "x": x, "y": y, "z": z}),
        id_column="id",
        role_overrides={"z": "covariate"},
    )
    table = summarize_partial_correlations(ds, covariates=("z",), methods=("pearson",), p_adjust="none")
    row = table.iloc[0]
    assert abs(row.coefficient) < 0.1
    assert row.df == n - 3


def test_rank_deficient_partial_covariates_are_degenerate() -> None:
    x = np.arange(30.0)
    y = x + np.sin(x)
    z = np.linspace(0, 1, 30)
    ds = TabularDataset(pd.DataFrame({"id": range(30), "x": x, "y": y, "z1": z, "z2": 2 * z}), id_column="id")
    table = summarize_partial_correlations(ds, covariates=("z1", "z2"), methods=("pearson",), p_adjust="none")
    assert table.iloc[0].status == "degenerate"
    assert table.iloc[0].reason == "rank_deficient_covariates"


def test_distance_correlation_finds_nonlinear_dependence() -> None:
    x = np.linspace(-2, 2, 80)
    y = x * x
    ds = TabularDataset(pd.DataFrame({"id": range(80), "x": x, "y": y}), id_column="id")
    dcor, mi = summarize_general_dependence(ds, n_permutations=49, p_adjust="none", random_state=3)
    assert dcor.iloc[0].statistic > 0.4
    assert dcor.iloc[0].p_value <= 0.1
    assert mi.iloc[0].statistic > 0.0


def test_dependence_permutation_is_deterministic() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=40)
    y = x + rng.normal(scale=0.3, size=40)
    ds = TabularDataset(pd.DataFrame({"id": range(40), "x": x, "y": y}), id_column="id")
    a = analyze_dependence(ds, n_permutations=29, random_state=9, p_adjust="none")
    b = analyze_dependence(ds, n_permutations=29, random_state=9, p_adjust="none")
    pd.testing.assert_frame_equal(a.distance_correlations, b.distance_correlations)
    pd.testing.assert_frame_equal(a.mutual_information, b.mutual_information)


def test_mutual_information_supports_numeric_categorical() -> None:
    x = np.r_[np.zeros(30), np.ones(30)]
    g = np.array(["A"] * 30 + ["B"] * 30)
    ds = TabularDataset(
        pd.DataFrame({"id": range(60), "x": x, "g": g}), id_column="id", role_overrides={"g": "factor"}
    )
    _, mi = summarize_general_dependence(
        ds, methods=("mutual_information",), n_permutations=19, p_adjust="none"
    )
    row = mi.iloc[0]
    assert row.statistic > 0.5
    assert row.column_y_kind == "categorical"


def test_distance_correlation_skips_mixed_pair() -> None:
    ds = TabularDataset(
        pd.DataFrame({"id": range(10), "x": np.arange(10.0), "g": ["A", "B"] * 5}), id_column="id"
    )
    dcor, _ = summarize_general_dependence(
        ds, methods=("distance_correlation",), n_permutations=0, p_adjust="none"
    )
    assert dcor.iloc[0].status == "skipped"
    assert dcor.iloc[0].reason == "distance_correlation_requires_numeric_pair"
