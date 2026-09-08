from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ruddy.statistics import adjust_pvalues, apply_multiple_testing


def test_bh_reference_values_and_order() -> None:
    observed = adjust_pvalues(np.array([0.01, 0.04, 0.03, 0.20]), "fdr_bh")
    expected = np.array([0.04, 0.05333333333333334, 0.05333333333333334, 0.20])
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_none_returns_raw_pvalues() -> None:
    values = np.array([0.1, 0.02])
    np.testing.assert_array_equal(adjust_pvalues(values, "none"), values)


def test_family_adjustment_excludes_non_ok_rows_and_records_size() -> None:
    table = pd.DataFrame(
        {
            "family_id": ["a", "a", "a", "b"],
            "status": ["ok", "degenerate", "ok", "ok"],
            "p_value": [0.01, np.nan, 0.04, 0.20],
            "q_value": [np.nan] * 4,
            "correction": ["fdr_bh"] * 4,
        }
    )
    result = apply_multiple_testing(table, "fdr_bh")
    np.testing.assert_allclose(result.loc[[0, 2], "q_value"], [0.02, 0.04])
    assert np.isnan(result.loc[1, "q_value"])
    assert result["family_size"].tolist() == [2, 2, 2, 1]


def test_adjustment_rejects_nonfinite_pvalues() -> None:
    with pytest.raises(ValueError, match="finite"):
        adjust_pvalues(np.array([0.1, np.nan]), "fdr_bh")
