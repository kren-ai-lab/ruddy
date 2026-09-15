from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from ruddy import TabularDataset
from ruddy.bivariate import summarize_categorical_associations
from ruddy.statistics import bias_corrected_cramers_v


def test_chi_square_and_cramers_v_match_scipy(mixed_dataset) -> None:
    table = summarize_categorical_associations(mixed_dataset, p_adjust="none")
    row = table.query("test == 'chi_square' and column_x == 'binary' and column_y == 'flag'").iloc[0]
    payload = json.loads(row["contingency_json"])
    counts = np.asarray(payload["counts"], dtype=int)
    chi2, p_value, dof, _ = stats.chi2_contingency(counts, correction=False)
    np.testing.assert_allclose(row["statistic"], chi2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(row["p_value"], p_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(  # ty: ignore[no-matching-overload]
        row["effect_size"],
        bias_corrected_cramers_v(chi2, int(counts.sum()), *counts.shape),
        rtol=1e-12,
        atol=1e-12,
    )
    assert row["df"] == dof


def test_fisher_exact_runs_only_for_2x2(mixed_dataset) -> None:
    table = summarize_categorical_associations(mixed_dataset, p_adjust="none")
    fisher_2x2 = table.query("test == 'fisher_exact' and column_x == 'binary' and column_y == 'flag'").iloc[0]
    counts = np.asarray(json.loads(fisher_2x2["contingency_json"])["counts"], dtype=int)
    ref = stats.fisher_exact(counts, alternative="two-sided")
    assert fisher_2x2["status"] == "ok"
    np.testing.assert_allclose(fisher_2x2["p_value"], ref.pvalue, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fisher_2x2["effect_size"], ref.statistic, rtol=1e-12, atol=1e-12)

    fisher_non2x2 = table.query(
        "test == 'fisher_exact' and column_x == 'binary' and column_y == 'three'"
    ).iloc[0]
    assert (fisher_non2x2["status"], fisher_non2x2["reason"]) == (
        "skipped",
        "fisher_exact_requires_2x2",
    )


def test_low_expected_counts_are_advisory_without_test_switch() -> None:
    frame = pd.DataFrame(
        {
            "a": ["A"] * 5 + ["B"] * 5,
            "b": ["X", "X", "X", "X", "Y", "X", "Y", "Y", "Y", "Y"],
        }
    )
    table = summarize_categorical_associations(TabularDataset(frame), p_adjust="none")
    chi = table.query("test == 'chi_square'").iloc[0]
    fisher = table.query("test == 'fisher_exact'").iloc[0]
    assert chi["status"] == "ok"
    assert fisher["status"] == "ok"
    assert chi["advisory_codes"] == "low_expected_counts"
    assert fisher["advisory_codes"] == "low_expected_counts"
    assert chi["n_expected_lt5"] > 0


def test_association_fdr_family_is_per_test(mixed_dataset) -> None:
    table = summarize_categorical_associations(mixed_dataset)
    for test in ("chi_square", "fisher_exact"):
        subset = table.query("test == @test")
        expected = int(subset["status"].eq("ok").sum())
        assert set(subset["family_id"]) == {f"associations:{test}"}
        assert set(subset["family_size"]) == {expected}
