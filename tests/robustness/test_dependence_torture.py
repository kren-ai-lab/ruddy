from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from ruddy import TabularDataset, analyze_dependence


def _dataset(x, y, z=None):
    frame = pd.DataFrame({"id": [f"o{i}" for i in range(len(x))], "x": x, "y": y})
    if z is not None:
        frame["z"] = z
    roles = {"x": "response", "y": "response"}
    if z is not None:
        roles["z"] = "covariate"
    return TabularDataset(frame, id_column="id", role_overrides=roles)


def test_linear_dependence_detected_by_distance_and_mi():
    rng = np.random.default_rng(31)
    x = rng.normal(size=160)
    y = 2 * x + rng.normal(0, 0.1, 160)
    r = analyze_dependence(_dataset(x, y), n_permutations=49, random_state=2)
    assert r.distance_correlations.row(0, named=True)["statistic"] > 0.95
    assert r.distance_correlations.row(0, named=True)["p_value"] <= 0.05
    assert r.mutual_information.row(0, named=True)["statistic"] > 1.0


def test_quadratic_dependence_detected_when_linear_correlation_is_small():
    rng = np.random.default_rng(32)
    x = rng.uniform(-1, 1, 300)
    y = x * x + rng.normal(0, 0.02, 300)
    pearson = abs(np.corrcoef(x, y)[0, 1])
    r = analyze_dependence(_dataset(x, y), n_permutations=49, random_state=3)
    assert pearson < 0.15
    assert r.distance_correlations.row(0, named=True)["statistic"] > 0.4
    assert r.distance_correlations.row(0, named=True)["p_value"] <= 0.05


def test_partial_correlation_removes_common_driver():
    rng = np.random.default_rng(33)
    z = rng.normal(size=240)
    x = z + rng.normal(0, 0.25, 240)
    y = z + rng.normal(0, 0.25, 240)
    r = analyze_dependence(_dataset(x, y, z), partial_covariates=("z",), n_permutations=0)
    partial = r.partial_correlations.filter(pl.col("method") == "pearson").row(0, named=True)
    assert abs(partial["coefficient"]) < 0.2


def test_independent_variables_have_low_general_dependence():
    rng = np.random.default_rng(34)
    x = rng.normal(size=300)
    y = rng.normal(size=300)
    r = analyze_dependence(_dataset(x, y), n_permutations=49, random_state=4)
    assert r.distance_correlations.row(0, named=True)["statistic"] < 0.2
    assert r.mutual_information.row(0, named=True)["statistic"] < 0.2


def test_constant_variable_is_degenerate_not_exception():
    x = np.ones(50)
    y = np.arange(50, dtype=float)
    r = analyze_dependence(_dataset(x, y), n_permutations=9)
    assert r.distance_correlations.row(0, named=True)["status"] == "degenerate"
    assert r.mutual_information.row(0, named=True)["status"] == "degenerate"
