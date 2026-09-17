from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.oneway import anova_oneway

from ruddy.bivariate import summarize_numeric_categorical_comparisons
from ruddy.statistics import cliffs_delta_from_u, eta_squared, hedges_g


def test_two_group_tests_and_effect_sizes_match_references(mixed_dataset) -> None:
    table = summarize_numeric_categorical_comparisons(
        mixed_dataset,
        tests=("welch_t", "mann_whitney"),
        p_adjust="none",
    )
    frame = mixed_dataset.to_frame()
    a = frame.loc[frame["binary"].eq("A"), "z"].to_numpy(dtype=float)
    b = frame.loc[frame["binary"].eq("B"), "z"].to_numpy(dtype=float)

    welch = table.query("group_column == 'binary' and feature == 'z' and test == 'welch_t'").iloc[0]
    ref_t = stats.ttest_ind(a, b, equal_var=False)
    np.testing.assert_allclose(welch["statistic"], ref_t.statistic, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(welch["p_value"], ref_t.pvalue, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(welch["effect_size"], hedges_g(a, b), rtol=1e-12, atol=1e-12)  # pyrefly: ignore[no-matching-overload]
    assert welch["effect_size_name"] == "hedges_g"

    mann = table.query("group_column == 'binary' and feature == 'z' and test == 'mann_whitney'").iloc[0]
    ref_u = stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
    np.testing.assert_allclose(mann["statistic"], ref_u.statistic, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mann["p_value"], ref_u.pvalue, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
        mann["effect_size"],
        cliffs_delta_from_u(ref_u.statistic, len(a), len(b)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_multigroup_welch_and_kruskal_match_independent_backends(mixed_dataset) -> None:
    table = summarize_numeric_categorical_comparisons(
        mixed_dataset,
        tests=("welch_anova", "kruskal_wallis"),
        p_adjust="none",
    )
    frame = mixed_dataset.to_frame()
    groups = tuple(
        frame.loc[frame["three"].eq(level), "z"].to_numpy(dtype=float) for level in ("H", "L", "M")
    )

    welch = table.query("group_column == 'three' and feature == 'z' and test == 'welch_anova'").iloc[0]
    ref_welch = anova_oneway(groups, use_var="unequal", welch_correction=True)
    np.testing.assert_allclose(welch["statistic"], ref_welch.statistic, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(welch["p_value"], ref_welch.pvalue, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(welch["effect_size"], eta_squared(groups), rtol=1e-12, atol=1e-12)  # pyrefly: ignore[no-matching-overload]

    kruskal = table.query("group_column == 'three' and feature == 'z' and test == 'kruskal_wallis'").iloc[0]
    ref_kw = stats.kruskal(*groups)
    np.testing.assert_allclose(kruskal["statistic"], ref_kw.statistic, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(kruskal["p_value"], ref_kw.pvalue, rtol=1e-12, atol=1e-12)
    assert kruskal["effect_size_name"] == "epsilon_squared"


def test_pairwise_posthoc_is_opt_in(mixed_dataset) -> None:
    default = summarize_numeric_categorical_comparisons(
        mixed_dataset,
        tests=("welch_t", "welch_anova"),
        pairwise=False,
    )
    enabled = summarize_numeric_categorical_comparisons(
        mixed_dataset,
        tests=("welch_t", "welch_anova"),
        pairwise=True,
    )
    assert not default["scope"].eq("pairwise").any()
    assert enabled["scope"].eq("pairwise").any()
    pairs = enabled.query(
        "group_column == 'three' and feature == 'z' and scope == 'pairwise' and test == 'welch_t'"
    )
    assert len(pairs) == 3
    assert len(set(pairs["family_id"])) == 1
    assert set(pairs["family_size"]) == {3}


def test_numeric_factor_is_a_grouping_variable(mixed_dataset) -> None:
    table = summarize_numeric_categorical_comparisons(
        mixed_dataset,
        tests=("kruskal_wallis",),
    )
    rows = table.query("group_column == 'numeric_factor'")
    assert not rows.empty
    assert not table["feature"].eq("numeric_factor").any()
