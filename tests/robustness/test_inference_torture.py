from __future__ import annotations

from ruddy import analyze_marginal_means, analyze_posthoc


def test_posthoc_methods_do_not_auto_switch(robust_tabular):
    tukey = analyze_posthoc(robust_tabular, response="y", factor="group", methods=("tukey_hsd",))
    games = analyze_posthoc(robust_tabular, response="y", factor="group", methods=("games_howell",))
    assert set(tukey.comparisons.get_column("method").to_list()) == {"tukey_hsd"}
    assert set(games.comparisons.get_column("method").to_list()) == {"games_howell"}


def test_marginal_means_covariate_adjustment_is_finite(robust_tabular):
    result = analyze_marginal_means(robust_tabular, response="y", factors=("group",), covariates=("x",))
    assert result.means.get_column("estimate").is_finite().all()
    assert result.contrasts.get_column("estimate_difference").is_finite().all()
