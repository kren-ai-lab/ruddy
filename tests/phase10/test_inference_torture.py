from __future__ import annotations

import numpy as np

from ruddy import analyze_factorial, analyze_marginal_means, analyze_posthoc


def test_posthoc_methods_do_not_auto_switch(robust_tabular):
    tukey = analyze_posthoc(robust_tabular, response="y", factor="group", methods=("tukey_hsd",))
    games = analyze_posthoc(robust_tabular, response="y", factor="group", methods=("games_howell",))
    assert set(tukey.comparisons.method) == {"tukey_hsd"}
    assert set(games.comparisons.method) == {"games_howell"}


def test_marginal_means_covariate_adjustment_is_finite(robust_tabular):
    result = analyze_marginal_means(robust_tabular, response="y", factors=("group",), covariates=("x",))
    assert np.isfinite(result.means["estimate"]).all()
    assert np.isfinite(result.contrasts["estimate_difference"]).all()


def test_factorial_perfect_covariate_collinearity_is_degenerate(robust_tabular):
    frame = robust_tabular.to_frame()
    frame["xcopy"] = frame["x"]
    from ruddy import TabularDataset

    ds = TabularDataset(
        frame,
        id_column="id",
        role_overrides={"y": "response", "group": "factor", "x": "covariate", "xcopy": "covariate"},
    )
    result = analyze_factorial(ds, response="y", factors=("group",), covariates=("x", "xcopy"))
    assert result.status.value == "degenerate"
    assert result.reason == "rank_deficient_design"
