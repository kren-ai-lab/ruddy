from __future__ import annotations

import pytest

from ruddy import AnalysisConfig, PAdjustMethod


def test_factorial_config_kwargs_are_explicit():
    config = AnalysisConfig(
        factorial_ss_type=3,
        factorial_p_adjust="fdr_bh",
        factorial_robust_covariance="HC3",
        factorial_min_cell_n=4,
        factorial_max_factor_levels=12,
        factorial_max_design_cells=1000,
        factorial_max_design_columns=250,
        factorial_max_interaction_order=3,
        factorial_diagnostic_alpha=0.01,
        factorial_condition_number_threshold=50.0,
    )
    kwargs = config.factorial_kwargs()
    assert kwargs["ss_type"] == 3
    assert kwargs["p_adjust"] is PAdjustMethod.FDR_BH
    assert kwargs["robust_covariance"] == "hc3"
    assert kwargs["min_cell_n"] == 4
    assert kwargs["diagnostic_alpha"] == 0.01


@pytest.mark.parametrize(
    "kwargs",
    [
        {"factorial_ss_type": 1},
        {"factorial_robust_covariance": "sandwich"},
        {"factorial_min_cell_n": 0},
        {"factorial_max_factor_levels": 1},
        {"factorial_max_design_cells": 0},
        {"factorial_max_design_columns": 1},
        {"factorial_max_interaction_order": 1},
        {"factorial_diagnostic_alpha": 1.0},
        {"factorial_condition_number_threshold": 0.0},
    ],
)
def test_factorial_config_guards(kwargs):
    with pytest.raises(ValueError):
        AnalysisConfig(**kwargs)
