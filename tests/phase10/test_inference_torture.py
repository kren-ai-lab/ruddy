from __future__ import annotations

import numpy as np
import pandas as pd

from ruddy import analyze_factorial, analyze_manova, analyze_marginal_means, analyze_posthoc


def test_type3_effects_are_category_order_invariant(robust_tabular):
    base = robust_tabular.to_frame()
    a = robust_tabular
    reversed_frame = base.copy()
    reversed_frame["group"] = pd.Categorical(reversed_frame["group"], categories=["C", "B", "A"])
    from ruddy import TabularDataset

    b = TabularDataset(
        reversed_frame,
        id_column="id",
        role_overrides={
            "y": "response",
            "y2": "response",
            "group": "factor",
            "batch": "factor",
            "class": "factor",
            "x": "covariate",
            "z": "covariate",
        },
    )
    r1 = analyze_factorial(
        a, response="y", factors=("group", "batch"), interactions=(("group", "batch"),), ss_type=3
    )
    r2 = analyze_factorial(
        b, response="y", factors=("group", "batch"), interactions=(("group", "batch"),), ss_type=3
    )
    e1 = r1.effects.set_index("term").sort_index()
    e2 = r2.effects.set_index("term").sort_index()
    np.testing.assert_allclose(e1["f_value"], e2["f_value"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(e1["p_value"], e2["p_value"], rtol=1e-10, atol=1e-12)


def test_posthoc_methods_do_not_auto_switch(robust_tabular):
    tukey = analyze_posthoc(robust_tabular, response="y", factor="group", methods=("tukey_hsd",))
    games = analyze_posthoc(robust_tabular, response="y", factor="group", methods=("games_howell",))
    assert set(tukey.comparisons.method) == {"tukey_hsd"}
    assert set(games.comparisons.method) == {"games_howell"}


def test_marginal_means_covariate_adjustment_is_finite(robust_tabular):
    result = analyze_marginal_means(robust_tabular, response="y", factors=("group",), covariates=("x",))
    assert np.isfinite(result.means["estimate"]).all()
    assert np.isfinite(result.contrasts["estimate_difference"]).all()


def test_manova_rank_deficient_responses_are_not_fitted(robust_tabular):
    frame = robust_tabular.to_frame()
    frame["ycopy"] = 2.0 * frame["y"]
    from ruddy import TabularDataset

    ds = TabularDataset(
        frame, id_column="id", role_overrides={"y": "response", "ycopy": "response", "group": "factor"}
    )
    result = analyze_manova(ds, responses=("y", "ycopy"), factors=("group",))
    assert result.status.value == "degenerate"
    assert result.reason == "rank_deficient_response_matrix"


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
