from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ruddy import CorrelationMethod, TabularDataset
from ruddy.bivariate import summarize_correlations


def test_correlations_match_scipy_with_pairwise_finite_values(mixed_dataset) -> None:
    table = summarize_correlations(mixed_dataset, p_adjust="none")
    frame = mixed_dataset.to_frame()
    mask = np.isfinite(frame["x"].to_numpy(dtype=float)) & np.isfinite(frame["y"].to_numpy(dtype=float))
    x = frame.loc[mask, "x"].to_numpy(dtype=float)
    y = frame.loc[mask, "y"].to_numpy(dtype=float)
    refs = {
        "pearson": stats.pearsonr(x, y),
        "spearman": stats.spearmanr(x, y),
        "kendall": stats.kendalltau(x, y),
    }
    for method, ref in refs.items():  # noqa: B007, PERF102 - `method` is read by the query below
        row = table.query("method == @method and column_x == 'x' and column_y == 'y'").iloc[0]
        assert row["status"] == "ok"
        assert row["n_complete"] == 10
        assert row["n_missing_pair"] == 2
        np.testing.assert_allclose(row["coefficient"], ref.statistic, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(row["p_value"], ref.pvalue, rtol=1e-12, atol=1e-12)
        assert row["q_value"] == row["p_value"]


def test_numeric_factor_is_not_used_as_continuous_correlation(mixed_dataset) -> None:
    table = summarize_correlations(mixed_dataset)
    assert "numeric_factor" not in set(table["column_x"]) | set(table["column_y"])


def test_constant_and_insufficient_pairs_are_explicit() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 1.0, 1.0, 1.0],
            "b": [1.0, 2.0, 3.0, 4.0],
            "c": [np.nan, np.nan, 1.0, np.nan],
        }
    )
    dataset = TabularDataset(frame)
    table = summarize_correlations(
        dataset,
        methods=(CorrelationMethod.PEARSON,),
        min_complete_pairs=3,
    )
    ab = table.query("column_x == 'a' and column_y == 'b'").iloc[0]
    ac = table.query("column_x == 'a' and column_y == 'c'").iloc[0]
    assert (ab["status"], ab["reason"]) == ("degenerate", "column_x_constant")
    assert (ac["status"], ac["reason"]) == ("skipped", "insufficient_complete_pairs")


def test_fdr_family_size_is_per_correlation_method(mixed_dataset) -> None:
    table = summarize_correlations(mixed_dataset, methods=("pearson", "spearman"))
    for method in ("pearson", "spearman"):
        subset = table.query("method == @method")
        expected = int(subset["status"].eq("ok").sum())
        assert set(subset["family_size"]) == {expected}
        assert set(subset["family_id"]) == {f"correlations:{method}"}
