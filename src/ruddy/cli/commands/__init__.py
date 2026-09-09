"""Ruddy CLI command implementations."""

from ruddy.cli.commands.bivariate import run_bivariate, write_bivariate_result
from ruddy.cli.commands.groups import run_groups, write_group_result
from ruddy.cli.commands.factorial import run_factorial, write_factorial_result
from ruddy.cli.commands.outliers import run_outliers, write_outlier_result
from ruddy.cli.commands.multivariate import (
    run_manova,
    run_multivariate,
    write_manova_result,
    write_multivariate_result,
)
from ruddy.cli.commands.projections import (
    load_feature_matrix,
    run_project,
    write_pca_result,
    write_projection_result,
)
from ruddy.cli.commands.descriptive import (
    build_config,
    load_dataset,
    run_profile,
    run_univariate,
    write_profiling_result,
    write_univariate_result,
)

__all__ = [
    "build_config",
    "load_dataset",
    "load_feature_matrix",
    "run_bivariate",
    "run_groups",
    "run_factorial",
    "run_manova",
    "run_multivariate",
    "run_outliers",
    "run_profile",
    "run_project",
    "run_univariate",
    "write_bivariate_result",
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
