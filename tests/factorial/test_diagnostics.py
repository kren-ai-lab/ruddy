from __future__ import annotations

import pandas as pd
import pytest

from ruddy.factorial.diagnostics import build_factorial_cells
from ruddy import analyze_factorial


def test_balanced_cell_table_is_complete(balanced_factorial_dataset):
    frame = balanced_factorial_dataset.to_frame()
    cells, summary = build_factorial_cells(frame, ("factor_a", "factor_b"))
    assert len(cells) == 6
    assert summary["balanced"] is True
    assert summary["n_empty_cells"] == 0
    assert cells["n"].nunique() == 1


def test_empty_cell_is_explicitly_enumerated(balanced_factorial_dataset):
    frame = balanced_factorial_dataset.to_frame()
    frame = frame.loc[~((frame["factor_a"] == "A1") & (frame["factor_b"] == "B2"))]
    cells, summary = build_factorial_cells(frame, ("factor_a", "factor_b"))
    empty = cells[cells["is_empty"]]
    assert len(empty) == 1
    assert empty.iloc[0]["reason"] == "empty_factorial_cell"
    assert summary["n_empty_cells"] == 1


def test_max_design_cells_guard(balanced_factorial_dataset):
    with pytest.raises(ValueError, match="max_design_cells"):
        build_factorial_cells(
            balanced_factorial_dataset.to_frame(),
            ("factor_a", "factor_b"),
            max_design_cells=4,
        )


def test_model_diagnostics_do_not_change_requested_ss_type(balanced_factorial_dataset):
    result = analyze_factorial(
        balanced_factorial_dataset,
        formula="response ~ factor_a * factor_b + covariate",
        ss_type=3,
        diagnostic_alpha=0.99,
    )
    assert result.design.ss_type == 3
    assert result.model_summary["ss_type"] == 3
    assert len(result.diagnostics) >= 5


def test_diagnostic_table_contains_core_assumption_checks(balanced_factorial_dataset):
    result = analyze_factorial(balanced_factorial_dataset, formula="response ~ factor_a * factor_b + covariate")
    observed = set(result.diagnostics["diagnostic"])
    assert {
        "shapiro_wilk_residual_normality",
        "jarque_bera_residual_normality",
        "breusch_pagan_heteroskedasticity",
        "levene_median_residual_homoscedasticity",
        "design_condition_number",
    } <= observed
