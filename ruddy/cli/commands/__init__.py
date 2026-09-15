"""Ruddy CLI command implementations."""

from ruddy.cli.commands.advanced_groups import (
    run_marginal_means,
    run_mixed_effects,
    run_permanova,
    run_posthoc,
    write_marginal_means_result,
    write_mixed_effects_result,
    write_permanova_result,
    write_posthoc_result,
)
from ruddy.cli.commands.analyze import run_analyze, write_unified_result
from ruddy.cli.commands.bivariate import run_bivariate, write_bivariate_result
from ruddy.cli.commands.descriptive import (
    build_config,
    load_dataset,
    run_profile,
    run_univariate,
    write_profiling_result,
    write_univariate_result,
)
from ruddy.cli.commands.extensions import (
    run_contingency,
    run_dependence,
    run_diagnostics,
    run_intervals,
    write_confidence_interval_result,
    write_contingency_diagnostics_result,
    write_dependence_result,
    write_distribution_diagnostics_result,
)
from ruddy.cli.commands.factorial import run_factorial, write_factorial_result
from ruddy.cli.commands.groups import run_groups, write_group_result
from ruddy.cli.commands.multivariate import (
    run_manova,
    run_multivariate,
    write_manova_result,
    write_multivariate_result,
)
from ruddy.cli.commands.outliers import run_outliers, write_outlier_result
from ruddy.cli.commands.projections import (
    load_feature_matrix,
    run_project,
    write_pca_result,
    write_projection_result,
)
from ruddy.cli.commands.specialized import (
    run_anomaly,
    run_bayesian,
    run_compositional,
    run_representation,
    write_anomaly_result,
    write_bayesian_result,
    write_compositional_result,
    write_representation_result,
)

__all__ = [
    "build_config",
    "run_analyze",
    "run_mixed_effects",
    "run_marginal_means",
    "run_posthoc",
    "run_permanova",
    "run_contingency",
    "run_dependence",
    "run_diagnostics",
    "run_intervals",
    "load_dataset",
    "load_feature_matrix",
    "run_bivariate",
    "run_anomaly",
    "run_bayesian",
    "run_compositional",
    "run_representation",
    "run_groups",
    "run_factorial",
    "run_manova",
    "run_multivariate",
    "run_outliers",
    "run_profile",
    "run_project",
    "run_univariate",
    "write_bivariate_result",
    "write_anomaly_result",
    "write_bayesian_result",
    "write_compositional_result",
    "write_representation_result",
    "write_mixed_effects_result",
    "write_marginal_means_result",
    "write_posthoc_result",
    "write_permanova_result",
    "write_confidence_interval_result",
    "write_contingency_diagnostics_result",
    "write_dependence_result",
    "write_distribution_diagnostics_result",
    "write_unified_result",
    "write_group_result",
    "write_factorial_result",
    "write_manova_result",
    "write_multivariate_result",
    "write_outlier_result",
    "write_pca_result",
    "write_projection_result",
    "write_profiling_result",
    "write_univariate_result",
]
