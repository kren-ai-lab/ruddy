"""Reusable statistical primitives."""

from ruddy.statistics.assumptions import expected_count_diagnostics
from ruddy.statistics.bootstrap import BootstrapResult, bootstrap_confidence_interval
from ruddy.statistics.confidence_intervals import (
    mean_confidence_interval,
    odds_ratio_confidence_interval,
    pearson_confidence_interval,
    welch_mean_difference_confidence_interval,
)
from ruddy.statistics.effect_sizes import (
    bias_corrected_cramers_v,
    cliffs_delta_from_u,
    epsilon_squared,
    eta_squared,
    hedges_g,
)
from ruddy.statistics.multiple_testing import (
    adjust_pvalues,
    apply_multiple_testing,
    family_sizes,
)
from ruddy.statistics.robust import (
    MODIFIED_Z_CONSISTENCY,
    median_absolute_deviation,
    modified_z_scores,
    tukey_fences,
)

__all__ = [
    "MODIFIED_Z_CONSISTENCY",
    "BootstrapResult",
    "adjust_pvalues",
    "apply_multiple_testing",
    "bias_corrected_cramers_v",
    "bootstrap_confidence_interval",
    "cliffs_delta_from_u",
    "epsilon_squared",
    "eta_squared",
    "expected_count_diagnostics",
    "family_sizes",
    "hedges_g",
    "mean_confidence_interval",
    "median_absolute_deviation",
    "modified_z_scores",
    "odds_ratio_confidence_interval",
    "pearson_confidence_interval",
    "tukey_fences",
    "welch_mean_difference_confidence_interval",
]
