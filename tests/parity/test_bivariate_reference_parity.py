from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats  # noqa: F401 - parity references

from ruddy import TabularDataset
from ruddy.bivariate import (
    summarize_correlations,
    summarize_numeric_categorical_comparisons,
)

ROOT = Path(__file__).parent


def _assert_value(observed, expected) -> None:
    if expected == "NaN":
        assert observed is None or (isinstance(observed, float) and np.isnan(observed))
    else:
        assert observed is not None
        np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-12)


def test_bivariate_metrics_match_frozen_reference() -> None:
    frame = pd.read_csv(ROOT / "fixtures" / "mixed_tabular.csv")
    dataset = TabularDataset(frame, id_column="observation_id")
    expected = json.loads((ROOT / "reference" / "bivariate_reference.json").read_text())

    correlations = summarize_correlations(
        dataset,
        methods=("pearson", "spearman"),
        p_adjust="fdr_bh",
    )
    observed_correlations = {
        (row["method"], row["column_x"], row["column_y"]): row for row in correlations.iter_rows(named=True)
    }
    for reference in expected["correlations"]:
        key = (reference["method"], reference["column_x"], reference["column_y"])
        row = observed_correlations[key]
        assert row["n_complete"] == reference["n_complete"]
        assert row["status"] == reference["status"]
        assert row["reason"] == reference["reason"]
        for field in ("coefficient", "p_value", "q_value"):
            _assert_value(row[field], reference[field])

    comparisons = summarize_numeric_categorical_comparisons(
        dataset,
        tests=("welch_t", "mann_whitney", "welch_anova", "kruskal_wallis"),
        pairwise=True,
        p_adjust="fdr_bh",
    )
    observed_comparisons = {
        (
            row["group_column"],
            row["scope"],
            row["test"],
            row["feature"],
            row["group_a"],
            row["group_b"],
        ): row
        for row in comparisons.iter_rows(named=True)
    }
    for block in ("comparisons_binary", "comparisons_multigroup"):
        for reference in expected[block]:
            key = (
                reference["group_column"],
                reference["scope"],
                reference["test"],
                reference["feature"],
                reference["group_a"],
                reference["group_b"],
            )
            row = observed_comparisons[key]
            assert row["status"] == reference["status"]
            assert row["reason"] == reference["reason"]
            for field in ("statistic", "p_value", "effect_size"):
                _assert_value(row[field], reference[field])
