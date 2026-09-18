from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy import stats

from ruddy import TabularDataset
from ruddy.bivariate import summarize_categorical_associations
from ruddy.statistics import bias_corrected_cramers_v


def test_chi_square_and_cramers_v_match_scipy(mixed_dataset) -> None:
    table = summarize_categorical_associations(mixed_dataset, p_adjust="none")
    row = table.filter(
        (pl.col("test") == "chi_square") & (pl.col("column_x") == "binary") & (pl.col("column_y") == "flag")
    ).row(0, named=True)
    payload = json.loads(row["contingency_json"])
    counts = np.asarray(payload["counts"], dtype=int)
    chi2, p_value, dof, _ = stats.chi2_contingency(counts, correction=False)
    np.testing.assert_allclose(row["statistic"], chi2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(row["p_value"], p_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
        row["effect_size"],
        bias_corrected_cramers_v(chi2, int(counts.sum()), *counts.shape),
        rtol=1e-12,
        atol=1e-12,
    )
    assert row["df"] == dof


def test_fisher_exact_runs_only_for_2x2(mixed_dataset) -> None:
    table = summarize_categorical_associations(mixed_dataset, p_adjust="none")
    fisher_2x2 = table.filter(
        (pl.col("test") == "fisher_exact") & (pl.col("column_x") == "binary") & (pl.col("column_y") == "flag")
    ).row(0, named=True)
    counts = np.asarray(json.loads(fisher_2x2["contingency_json"])["counts"], dtype=int)
    ref = stats.fisher_exact(counts, alternative="two-sided")
    assert fisher_2x2["status"] == "ok"
    np.testing.assert_allclose(fisher_2x2["p_value"], ref.pvalue, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fisher_2x2["effect_size"], ref.statistic, rtol=1e-12, atol=1e-12)

    fisher_non2x2 = table.filter(
        (pl.col("test") == "fisher_exact")
        & (pl.col("column_x") == "binary")
        & (pl.col("column_y") == "three")
    ).row(0, named=True)
    assert (fisher_non2x2["status"], fisher_non2x2["reason"]) == (
        "skipped",
        "fisher_exact_requires_2x2",
    )


def test_low_expected_counts_are_advisory_without_test_switch() -> None:
    frame = pl.DataFrame(
        {
            "a": ["A"] * 5 + ["B"] * 5,
            "b": ["X", "X", "X", "X", "Y", "X", "Y", "Y", "Y", "Y"],
        }
    )
    table = summarize_categorical_associations(TabularDataset(frame), p_adjust="none")
    chi = table.filter(pl.col("test") == "chi_square").row(0, named=True)
    fisher = table.filter(pl.col("test") == "fisher_exact").row(0, named=True)
    assert chi["status"] == "ok"
    assert fisher["status"] == "ok"
    assert chi["advisory_codes"] == "low_expected_counts"
    assert fisher["advisory_codes"] == "low_expected_counts"
    assert chi["n_expected_lt5"] > 0


def test_association_fdr_family_is_per_test(mixed_dataset) -> None:
    table = summarize_categorical_associations(mixed_dataset)
    for test in ("chi_square", "fisher_exact"):
        subset = table.filter(pl.col("test") == test)
        expected = int((subset.get_column("status") == "ok").sum())
        assert set(subset.get_column("family_id").to_list()) == {f"associations:{test}"}
        assert set(subset.get_column("family_size").to_list()) == {expected}
