import numpy as np
from scipy.stats import studentized_range
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from ruddy.bivariate import analyze_posthoc


def test_tukey_matches_statsmodels(group_dataset):
    result = analyze_posthoc(group_dataset, response="y", factor="group", methods=("tukey_hsd",), min_group_n=2)
    frame = group_dataset.select(["y", "group"])
    reference = pairwise_tukeyhsd(frame["y"], frame["group"])
    table = result.comparisons.sort_values(["group_a", "group_b"]).reset_index(drop=True)
    assert np.allclose(table["mean_difference"], reference.meandiffs, atol=1e-12)
    assert np.allclose(table["p_value"], reference.pvalues, atol=1e-10)
    assert np.allclose(table[["ci_lower", "ci_upper"]], reference.confint, atol=1e-10)


def test_games_howell_matches_manual_pair(group_dataset):
    result = analyze_posthoc(group_dataset, response="y", factor="group", methods=("games_howell",), min_group_n=2)
    row = result.comparisons.iloc[0]
    frame = group_dataset.select(["y", "group"])
    xa = frame.loc[frame.group == row.group_a, "y"].to_numpy()
    xb = frame.loc[frame.group == row.group_b, "y"].to_numpy()
    aa = np.var(xa, ddof=1) / len(xa)
    bb = np.var(xb, ddof=1) / len(xb)
    df = (aa + bb) ** 2 / (aa**2 / (len(xa) - 1) + bb**2 / (len(xb) - 1))
    q = abs(xb.mean() - xa.mean()) / np.sqrt(0.5 * (aa + bb))
    p = studentized_range.sf(q, 3, df)
    assert np.isclose(row.df, df)
    assert np.isclose(row.statistic, q)
    assert np.isclose(row.p_value, p)


def test_both_posthoc_methods_are_explicit(group_dataset):
    result = analyze_posthoc(group_dataset, response="y", factor="group")
    assert set(result.comparisons.method) == {"tukey_hsd", "games_howell"}
    assert len(result.comparisons) == 6


def test_posthoc_rejects_numeric_nonfactor(group_dataset):
    try:
        analyze_posthoc(group_dataset, response="y", factor="x")
    except ValueError as exc:
        assert "factor" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError")
