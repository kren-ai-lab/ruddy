from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from ruddy import CorrelationMethod, TabularDataset
from ruddy.bivariate import summarize_correlations


def test_correlations_match_scipy_with_pairwise_finite_values(mixed_dataset) -> None:
    table = summarize_correlations(mixed_dataset, p_adjust="none")
    frame = mixed_dataset.frame
    x_col = frame.get_column("x").cast(pl.Float64).fill_null(float("nan")).to_numpy()
    y_col = frame.get_column("y").cast(pl.Float64).fill_null(float("nan")).to_numpy()
    mask = np.isfinite(x_col) & np.isfinite(y_col)
    x = x_col[mask]
    y = y_col[mask]
    refs = {
        "pearson": stats.pearsonr(x, y),
        "spearman": stats.spearmanr(x, y),
        "kendall": stats.kendalltau(x, y),
    }
    for method, ref in refs.items():
        row = table.filter(
            (pl.col("method") == method) & (pl.col("column_x") == "x") & (pl.col("column_y") == "y")
        ).row(0, named=True)
        assert row["status"] == "ok"
        assert row["n_complete"] == 10
        assert row["n_missing_pair"] == 2
        np.testing.assert_allclose(row["coefficient"], ref.statistic, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(row["p_value"], ref.pvalue, rtol=1e-12, atol=1e-12)
        assert row["q_value"] == row["p_value"]


def test_numeric_factor_is_not_used_as_continuous_correlation(mixed_dataset) -> None:
    table = summarize_correlations(mixed_dataset)
    assert "numeric_factor" not in set(table.get_column("column_x")) | set(table.get_column("column_y"))


def test_constant_and_insufficient_pairs_are_explicit() -> None:
    frame = pl.DataFrame(
        {
            "a": [1.0, 1.0, 1.0, 1.0],
            "b": [1.0, 2.0, 3.0, 4.0],
            "c": [None, None, 1.0, None],
        }
    )
    dataset = TabularDataset(frame)
    table = summarize_correlations(
        dataset,
        methods=(CorrelationMethod.PEARSON,),
        min_complete_pairs=3,
    )
    ab = table.filter((pl.col("column_x") == "a") & (pl.col("column_y") == "b")).row(0, named=True)
    ac = table.filter((pl.col("column_x") == "a") & (pl.col("column_y") == "c")).row(0, named=True)
    assert (ab["status"], ab["reason"]) == ("degenerate", "column_x_constant")
    assert (ac["status"], ac["reason"]) == ("skipped", "insufficient_complete_pairs")


def test_fdr_family_size_is_per_correlation_method(mixed_dataset) -> None:
    table = summarize_correlations(mixed_dataset, methods=("pearson", "spearman"))
    for method in ("pearson", "spearman"):
        subset = table.filter(pl.col("method") == method)
        expected = int((subset.get_column("status") == "ok").sum())
        assert set(subset.get_column("family_size").to_list()) == {expected}
        assert set(subset.get_column("family_id").to_list()) == {f"correlations:{method}"}


def test_all_null_float_column_reports_insufficient_complete_pairs() -> None:
    frame = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": pl.Series([None, None, None, None], dtype=pl.Float64),
        }
    )
    dataset = TabularDataset(frame)
    table = summarize_correlations(
        dataset,
        methods=(CorrelationMethod.PEARSON,),
        min_complete_pairs=3,
    )
    row = table.filter((pl.col("column_x") == "a") & (pl.col("column_y") == "b")).row(0, named=True)
    assert row["status"] == "skipped"
    assert row["reason"] == "insufficient_complete_pairs"
    assert row["q_value"] is None


def test_empty_correlations_schema_preserves_declared_types() -> None:
    frame = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    dataset = TabularDataset(frame)
    table = summarize_correlations(dataset)
    assert table.height == 0
    assert table.schema["q_value"] == pl.Float64
    assert table.schema["family_size"] == pl.Int64
