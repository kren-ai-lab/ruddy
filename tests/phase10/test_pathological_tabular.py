from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ruddy import (
    TabularDataset,
    analyze_bivariate,
    analyze_distribution_diagnostics,
    analyze_factorial,
    analyze_groups,
    analyze_outliers,
    analyze_univariate,
)
from ruddy.core.exceptions import DataValidationError


def test_duplicate_observation_ids_remain_hard_error():
    frame = pd.DataFrame({"id": ["a", "a"], "x": [1.0, 2.0]})
    with pytest.raises(DataValidationError):
        TabularDataset(frame, id_column="id")


def test_numeric_strings_are_never_silently_coerced():
    ds = TabularDataset(pd.DataFrame({"id": ["a", "b", "c"], "x": ["1", "2", "3"]}), id_column="id")
    result = analyze_univariate(ds)
    assert result.numeric_statistics.empty
    assert "x" in set(result.categorical_statistics["column"])


def test_single_observation_numeric_analysis_is_observable_not_crash():
    ds = TabularDataset(pd.DataFrame({"id": ["a"], "x": [1.0]}), id_column="id")
    uni = analyze_univariate(ds)
    out = analyze_outliers(ds)
    assert uni.numeric_statistics.loc[uni.numeric_statistics.column == "x", "status"].iloc[0] != "ok"
    assert set(out.summaries.loc[out.summaries.column == "x", "status"]) <= {"skipped", "degenerate"}


def test_all_missing_and_nonfinite_are_distinguished():
    ds = TabularDataset(
        pd.DataFrame({"id": ["a", "b", "c", "d"], "missing": [np.nan]*4, "nonfinite": [np.inf, -np.inf, np.inf, -np.inf]}),
        id_column="id",
        kind_overrides={"missing": "numeric"},
    )
    out = analyze_outliers(ds)
    quality = out.quality.set_index("column")
    assert quality.loc["missing", "n_missing"] == 4
    assert quality.loc["missing", "n_non_finite"] == 0
    assert quality.loc["nonfinite", "n_missing"] == 0
    assert quality.loc["nonfinite", "n_non_finite"] == 4


def test_one_level_factor_group_analysis_is_degenerate():
    ds = TabularDataset(
        pd.DataFrame({"id": [f"o{i}" for i in range(8)], "y": np.arange(8.0), "g": ["A"]*8}),
        id_column="id",
        role_overrides={"y": "response", "g": "factor"},
    )
    result = analyze_groups(ds, responses=("y",), groups=("g",))
    row = result.group_coverage.set_index("group_column").loc["g"]
    assert row["status"] == "degenerate"
    assert row["reason"] == "insufficient_group_levels"
    assert result.numeric_comparisons.empty


def test_high_cardinality_factor_hits_explicit_guard():
    n = 25
    ds = TabularDataset(
        pd.DataFrame({"id": [f"o{i}" for i in range(n)], "y": np.arange(n, dtype=float), "g": [f"g{i}" for i in range(n)]}),
        id_column="id",
        role_overrides={"y": "response", "g": "factor"},
    )
    result = analyze_groups(ds, responses=("y",), groups=("g",), max_group_levels=20)
    coverage = result.group_coverage.set_index("group_column").loc["g"]
    assert coverage["status"] == "skipped"
    assert coverage["reason"] == "group_levels_exceed_max_group_levels"
    assert set(result.numeric_comparisons["status"]) == {"skipped"}


def test_factorial_empty_cell_never_returns_silent_clean_fit():
    frame = pd.DataFrame({
        "id": [f"o{i}" for i in range(18)],
        "y": np.linspace(0, 1, 18),
        "a": ["A"]*9 + ["B"]*9,
        "b": ["X"]*6 + ["Y"]*3 + ["X"]*9,
    })
    ds = TabularDataset(frame, id_column="id", role_overrides={"y": "response", "a": "factor", "b": "factor"})
    result = analyze_factorial(ds, response="y", factors=("a", "b"), interactions=(("a", "b"),), ss_type=3)
    assert (result.cells["is_empty"] == True).any()  # noqa: E712
    assert result.status.value != "ok" or any(a.code for a in result.advisories)


def test_complete_case_collapse_is_explicit_in_factorial():
    frame = pd.DataFrame({
        "id": [f"o{i}" for i in range(8)],
        "y": [1.0, np.nan, np.nan, np.nan, 2.0, np.nan, np.nan, np.nan],
        "g": ["A"]*4 + ["B"]*4,
    })
    ds = TabularDataset(frame, id_column="id", role_overrides={"y": "response", "g": "factor"})
    result = analyze_factorial(ds, response="y", factors=("g",))
    assert result.status.value in {"skipped", "degenerate"}
    assert not result.exclusions.empty


def test_analyses_do_not_mutate_input_dataframe(robust_tabular):
    before = robust_tabular.to_frame()
    analyze_univariate(robust_tabular)
    analyze_bivariate(robust_tabular)
    analyze_distribution_diagnostics(robust_tabular, responses=("y",), groups=("group",))
    analyze_outliers(robust_tabular, include_flags=True)
    after = robust_tabular.to_frame()
    pd.testing.assert_frame_equal(before, after)
