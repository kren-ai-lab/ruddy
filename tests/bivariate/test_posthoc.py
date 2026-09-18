import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import studentized_range
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from ruddy import TabularDataset
from ruddy.bivariate import analyze_posthoc


def test_tukey_matches_statsmodels(group_dataset):
    result = analyze_posthoc(
        group_dataset, response="y", factor="group", methods=("tukey_hsd",), min_group_n=2
    )
    frame = group_dataset.select(["y", "group"])
    reference = pairwise_tukeyhsd(frame["y"], frame["group"])
    table = result.comparisons.sort(["group_a", "group_b"])
    assert np.allclose(table["mean_difference"].to_numpy(), reference.meandiffs, atol=1e-12)  # pyrefly: ignore[bad-argument-type]
    assert np.allclose(table["p_value"].to_numpy(), reference.pvalues, atol=1e-10)  # pyrefly: ignore[bad-argument-type]
    assert np.allclose(table.select(["ci_lower", "ci_upper"]).to_numpy(), reference.confint, atol=1e-10)  # pyrefly: ignore[bad-argument-type]


def test_games_howell_matches_manual_pair(group_dataset):
    result = analyze_posthoc(
        group_dataset, response="y", factor="group", methods=("games_howell",), min_group_n=2
    )
    row = result.comparisons.row(0, named=True)
    frame = group_dataset.frame
    xa = frame.filter(pl.col("group") == row["group_a"]).get_column("y").to_numpy()
    xb = frame.filter(pl.col("group") == row["group_b"]).get_column("y").to_numpy()
    aa = np.var(xa, ddof=1) / len(xa)
    bb = np.var(xb, ddof=1) / len(xb)
    df = (aa + bb) ** 2 / (aa**2 / (len(xa) - 1) + bb**2 / (len(xb) - 1))
    q = abs(xb.mean() - xa.mean()) / np.sqrt(0.5 * (aa + bb))
    p = studentized_range.sf(q, 3, df)
    assert np.isclose(row["df"], df)
    assert np.isclose(row["statistic"], q)
    assert np.isclose(row["p_value"], p)


def test_both_posthoc_methods_are_explicit(group_dataset):
    result = analyze_posthoc(group_dataset, response="y", factor="group")
    assert set(result.comparisons.get_column("method").to_list()) == {"tukey_hsd", "games_howell"}
    assert result.comparisons.height == 6


def test_posthoc_rejects_numeric_nonfactor(group_dataset):
    try:
        analyze_posthoc(group_dataset, response="y", factor="x")
    except ValueError as exc:
        assert "factor" in str(exc).lower()
    else:
        msg = "Expected ValueError"
        raise AssertionError(msg)


def test_posthoc_factor_dtypes_preserved():
    df_int = pd.DataFrame(
        {
            "id": range(30),
            "y": np.arange(30.0),
            "factor": [1] * 15 + [2] * 15,
        }
    )
    ds_int = TabularDataset(df_int, id_column="id", role_overrides={"factor": "factor", "y": "response"})
    res_int = analyze_posthoc(ds_int, response="y", factor="factor")
    assert res_int.comparisons.schema["group_a"] == pl.Int64
    assert res_int.comparisons.schema["group_b"] == pl.Int64

    df_str = pd.DataFrame(
        {
            "id": range(30),
            "y": np.arange(30.0),
            "factor": ["A"] * 15 + ["B"] * 15,
        }
    )
    ds_str = TabularDataset(df_str, id_column="id", role_overrides={"factor": "factor", "y": "response"})
    res_str = analyze_posthoc(ds_str, response="y", factor="factor")
    assert res_str.comparisons.schema["group_a"] == pl.String
    assert res_str.comparisons.schema["group_b"] == pl.String
