"""Ruddy command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ruddy._version import __version__
from ruddy.cli.commands import (
    run_analyze,
    run_bivariate,
    run_anomaly,
    run_bayesian,
    run_compositional,
    run_representation,
    run_contingency,
    run_dependence,
    run_diagnostics,
    run_intervals,
    run_permanova,
    run_posthoc,
    run_marginal_means,
    run_mixed_effects,
    run_groups,
    run_factorial,
    run_manova,
    run_multivariate,
    run_outliers,
    run_profile,
    run_project,
    run_univariate,
)


def _add_descriptive_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Input .csv, .tsv, or .txt table.")
    parser.add_argument("--id-column", default=None, help="Observation identifier column.")
    parser.add_argument("--response", action="append", default=[], help="Response column; repeatable.")
    parser.add_argument("--factor", action="append", default=[], help="Factor column; repeatable.")
    parser.add_argument("--covariate", action="append", default=[], help="Covariate column; repeatable.")
    parser.add_argument("--annotation", action="append", default=[], help="Annotation column; repeatable.")
    parser.add_argument("--exclude", action="append", default=[], help="Excluded column; repeatable.")
    parser.add_argument("--numeric", action="append", default=[], help="Force numeric kind without coercion; repeatable.")
    parser.add_argument("--categorical", action="append", default=[], help="Interpret as categorical; repeatable.")
    parser.add_argument("--min-numeric-n", type=int, default=3)
    parser.add_argument("--max-category-levels", type=int, default=50)
    parser.add_argument("--max-missingness-patterns", type=int, default=20)
    parser.add_argument("--max-pairwise-columns", type=int, default=200)
    parser.add_argument("--output-dir", default=None, help="Write structured CSV/JSON outputs to this directory.")


def _add_phase9a_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--max-shapiro-n", type=int, default=5000)
    parser.add_argument("--partial-covariate", action="append", default=[], help="Numeric covariate for partial correlation; repeatable.")
    parser.add_argument("--dependence-min-pairs", type=int, default=5)
    parser.add_argument("--dependence-max-columns", type=int, default=50)
    parser.add_argument("--dependence-permutations", type=int, default=199)
    parser.add_argument("--mi-neighbors", type=int, default=3)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--interval-correlation", action="append", choices=("pearson", "spearman", "kendall"), default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-method", choices=("percentile", "basic", "bca"), default="bca")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level Ruddy CLI parser."""

    parser = argparse.ArgumentParser(
        prog="ruddy",
        description=(
            "Domain-agnostic statistical exploratory data analysis for tabular "
            "datasets and feature representations."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run an explicitly configured unified Ruddy analysis pipeline.",
    )
    _add_descriptive_options(analyze_parser)
    _add_phase9a_options(analyze_parser)
    analyze_parser.add_argument(
        "--enable",
        action="append",
        choices=(
            "profiling", "univariate", "bivariate", "groups", "outliers",
            "pca", "tsne", "umap", "multivariate", "manova", "factorial",
            "diagnostics", "dependence", "contingency", "intervals",
            "permanova", "posthoc", "marginal_means", "mixed_effects",
            "representation", "compositional", "bayesian", "anomaly",
        ),
        default=None,
        help="Top-level analysis block to execute; repeatable. Defaults to profiling + univariate.",
    )
    analyze_parser.add_argument("--group", action="append", default=[])
    analyze_parser.add_argument("--correlation", action="append", choices=("pearson", "spearman", "kendall"), default=None)
    analyze_parser.add_argument(
        "--comparison-test", action="append",
        choices=("welch_t", "mann_whitney", "welch_anova", "kruskal_wallis", "chi_square", "fisher_exact"),
        default=None,
    )
    analyze_parser.add_argument("--p-adjust", choices=("none", "fdr_bh"), default="fdr_bh")
    analyze_parser.add_argument("--min-group-n", type=int, default=3)
    analyze_parser.add_argument("--min-correlation-pairs", type=int, default=3)
    analyze_parser.add_argument("--max-group-levels", type=int, default=20)
    analyze_parser.add_argument("--max-correlation-columns", type=int, default=100)
    analyze_parser.add_argument("--pairwise", action="store_true")
    analyze_parser.add_argument("--outlier-method", action="append", choices=("iqr", "robust_z"), default=None)
    analyze_parser.add_argument("--iqr-multiplier", type=float, default=1.5)
    analyze_parser.add_argument("--robust-z-threshold", type=float, default=3.5)
    analyze_parser.add_argument("--include-flags", action="store_true")
    analyze_parser.add_argument("--feature-input", default=None, help="Optional separate feature matrix for feature-space blocks.")
    analyze_parser.add_argument("--feature-id-column", default=None)
    analyze_parser.add_argument("--feature-ids-file", default=None)
    analyze_parser.add_argument("--feature-column", action="append", default=[])
    analyze_parser.add_argument("--feature-alignment", choices=("strict", "partial"), default="strict")
    analyze_parser.add_argument("--projection-n-components", type=int, default=2)
    analyze_parser.add_argument("--projection-scaling", choices=("none", "standard", "robust", "minmax"), default="none")
    analyze_parser.add_argument("--projection-metric", default="euclidean")
    analyze_parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    analyze_parser.add_argument("--tsne-max-iter", type=int, default=1000)
    analyze_parser.add_argument("--umap-n-neighbors", type=int, default=15)
    analyze_parser.add_argument("--umap-min-dist", type=float, default=0.1)
    analyze_parser.add_argument("--multivariate-scaling", choices=("none", "standard", "robust", "minmax"), default="none")
    analyze_parser.add_argument("--no-spearman", action="store_true")
    analyze_parser.add_argument("--no-robust-mahalanobis", action="store_true")
    analyze_parser.add_argument("--mahalanobis-quantile", type=float, default=0.975)
    analyze_parser.add_argument("--robust-support-fraction", type=float, default=None)
    analyze_parser.add_argument("--max-covariance-features", type=int, default=200)
    analyze_parser.add_argument("--max-collinearity-features", type=int, default=100)
    analyze_parser.add_argument("--max-mahalanobis-features", type=int, default=100)
    analyze_parser.add_argument("--manova-max-responses", type=int, default=20)
    analyze_parser.add_argument("--manova-max-factor-levels", type=int, default=20)
    analyze_parser.add_argument("--manova-min-level-n", type=int, default=3)
    analyze_parser.add_argument("--factorial-response", default=None)
    analyze_parser.add_argument("--factorial-formula", default=None)
    analyze_parser.add_argument("--interaction", action="append", default=[])
    analyze_parser.add_argument("--ss-type", choices=("2", "3"), default="2")
    analyze_parser.add_argument("--factorial-p-adjust", choices=("none", "fdr_bh"), default="none")
    analyze_parser.add_argument("--robust-covariance", choices=("none", "hc0", "hc1", "hc2", "hc3"), default="none")
    analyze_parser.add_argument("--min-cell-n", type=int, default=2)
    analyze_parser.add_argument("--max-factor-levels", type=int, default=20)
    analyze_parser.add_argument("--max-design-cells", type=int, default=5000)
    analyze_parser.add_argument("--max-design-columns", type=int, default=500)
    analyze_parser.add_argument("--max-interaction-order", type=int, default=3)
    analyze_parser.add_argument("--diagnostic-alpha", type=float, default=0.05)
    analyze_parser.add_argument("--condition-number-threshold", type=float, default=30.0)
    analyze_parser.add_argument("--permanova-factor", default=None)
    analyze_parser.add_argument("--permanova-metric", default="euclidean")
    analyze_parser.add_argument("--permanova-permutations", type=int, default=999)
    analyze_parser.add_argument("--permanova-min-group-n", type=int, default=3)
    analyze_parser.add_argument("--permanova-max-group-levels", type=int, default=20)
    analyze_parser.add_argument("--posthoc-response", default=None)
    analyze_parser.add_argument("--posthoc-factor", default=None)
    analyze_parser.add_argument("--posthoc-method", action="append", choices=("tukey_hsd", "games_howell"), default=None)
    analyze_parser.add_argument("--marginal-response", default=None)
    analyze_parser.add_argument("--marginal-factor", action="append", default=[])
    analyze_parser.add_argument("--marginal-covariate", action="append", default=[])
    analyze_parser.add_argument("--marginal-interaction", action="append", default=[])
    analyze_parser.add_argument("--marginal-formula", default=None)
    analyze_parser.add_argument("--marginal-term", action="append", default=[])
    analyze_parser.add_argument("--marginal-p-adjust", choices=("none", "fdr_bh"), default="fdr_bh")
    analyze_parser.add_argument("--mixed-group", default=None)
    analyze_parser.add_argument("--mixed-response", default=None)
    analyze_parser.add_argument("--mixed-factor", action="append", default=[])
    analyze_parser.add_argument("--mixed-covariate", action="append", default=[])
    analyze_parser.add_argument("--mixed-interaction", action="append", default=[])
    analyze_parser.add_argument("--mixed-formula", default=None)
    analyze_parser.add_argument("--mixed-random-slope", action="append", default=[])
    analyze_parser.add_argument("--mixed-ml", action="store_true", help="Use ML instead of REML.")
    analyze_parser.add_argument("--mixed-optimizer", default="lbfgs")
    analyze_parser.add_argument("--mixed-max-iter", type=int, default=1000)
    analyze_parser.add_argument("--mixed-min-groups", type=int, default=3)
    analyze_parser.add_argument("--mixed-min-group-n", type=int, default=2)
    analyze_parser.add_argument("--comparison-feature-input", default=None)
    analyze_parser.add_argument("--comparison-feature-id-column", default=None)
    analyze_parser.add_argument("--comparison-feature-ids-file", default=None)
    analyze_parser.add_argument("--comparison-feature-column", action="append", default=[])
    analyze_parser.add_argument("--representation-alignment", choices=("strict", "partial"), default="strict")
    analyze_parser.add_argument("--cca-components", type=int, default=2)
    analyze_parser.add_argument("--cca-scaling", choices=("none", "standard", "robust", "minmax"), default="standard")
    analyze_parser.add_argument("--cca-max-iter", type=int, default=1000)
    analyze_parser.add_argument("--cca-tol", type=float, default=1e-6)
    analyze_parser.add_argument("--distance-metric", default="euclidean")
    analyze_parser.add_argument("--distance-similarity-method", choices=("pearson", "spearman"), default="spearman")
    analyze_parser.add_argument("--mantel-permutations", type=int, default=999)
    analyze_parser.add_argument("--compositional-transform", choices=("clr", "alr", "ilr"), default="clr")
    analyze_parser.add_argument("--replace-zeros", action="store_true")
    analyze_parser.add_argument("--zero-replacement-fraction", type=float, default=0.65)
    analyze_parser.add_argument("--alr-denominator", type=int, default=-1)
    analyze_parser.add_argument("--bayesian-variable", action="append", default=[])
    analyze_parser.add_argument("--bayesian-group", action="append", default=[])
    analyze_parser.add_argument("--bayesian-credible-level", type=float, default=0.95)
    analyze_parser.add_argument("--bayesian-rope-low", type=float, default=-0.1)
    analyze_parser.add_argument("--bayesian-rope-high", type=float, default=0.1)
    analyze_parser.add_argument("--bayesian-draws", type=int, default=5000)
    analyze_parser.add_argument("--bayesian-min-n", type=int, default=3)
    analyze_parser.add_argument("--anomaly-method", action="append", choices=("isolation_forest", "lof"), default=None)
    analyze_parser.add_argument("--anomaly-scaling", choices=("none", "standard", "robust", "minmax"), default="none")
    analyze_parser.add_argument("--contamination", default="auto")
    analyze_parser.add_argument("--isolation-estimators", type=int, default=200)
    analyze_parser.add_argument("--lof-neighbors", type=int, default=20)
    analyze_parser.set_defaults(handler=run_analyze)

    diagnostics_parser = subparsers.add_parser(
        "diagnostics", help="Run standalone normality and grouped dispersion diagnostics."
    )
    _add_descriptive_options(diagnostics_parser)
    diagnostics_parser.add_argument("--group", action="append", default=[])
    diagnostics_parser.add_argument("--p-adjust", choices=("none", "fdr_bh"), default="fdr_bh")
    diagnostics_parser.add_argument("--min-group-n", type=int, default=3)
    diagnostics_parser.add_argument("--max-group-levels", type=int, default=20)
    _add_phase9a_options(diagnostics_parser)
    diagnostics_parser.set_defaults(handler=run_diagnostics)

    dependence_parser = subparsers.add_parser(
        "dependence", help="Run partial, distance-correlation, and mutual-information analysis."
    )
    _add_descriptive_options(dependence_parser)
    dependence_parser.add_argument("--p-adjust", choices=("none", "fdr_bh"), default="fdr_bh")
    _add_phase9a_options(dependence_parser)
    dependence_parser.set_defaults(handler=run_dependence)

    contingency_parser = subparsers.add_parser(
        "contingency", help="Run cell-level contingency diagnostics."
    )
    _add_descriptive_options(contingency_parser)
    contingency_parser.add_argument("--p-adjust", choices=("none", "fdr_bh"), default="fdr_bh")
    _add_phase9a_options(contingency_parser)
    contingency_parser.set_defaults(handler=run_contingency)

    intervals_parser = subparsers.add_parser(
        "intervals", help="Estimate confidence intervals for common EDA estimands."
    )
    _add_descriptive_options(intervals_parser)
    intervals_parser.add_argument("--min-group-n", type=int, default=3)
    _add_phase9a_options(intervals_parser)
    intervals_parser.set_defaults(handler=run_intervals)

    permanova_parser = subparsers.add_parser(
        "permanova", help="Run PERMANOVA and PERMDISP on a feature space."
    )
    _add_descriptive_options(permanova_parser)
    permanova_parser.add_argument("--group", action="append", default=[])
    permanova_parser.add_argument("--feature-input", default=None)
    permanova_parser.add_argument("--feature-id-column", default=None)
    permanova_parser.add_argument("--feature-ids-file", default=None)
    permanova_parser.add_argument("--feature-column", action="append", default=[])
    permanova_parser.add_argument("--feature-alignment", choices=("strict", "partial"), default="strict")
    permanova_parser.add_argument("--permanova-factor", default=None)
    permanova_parser.add_argument("--permanova-metric", default="euclidean")
    permanova_parser.add_argument("--permanova-permutations", type=int, default=999)
    permanova_parser.add_argument("--permanova-min-group-n", type=int, default=3)
    permanova_parser.add_argument("--permanova-max-group-levels", type=int, default=20)
    permanova_parser.add_argument("--random-state", type=int, default=0)
    permanova_parser.set_defaults(handler=run_permanova)

    posthoc_parser = subparsers.add_parser(
        "posthoc", help="Run explicitly requested Tukey HSD and/or Games-Howell comparisons."
    )
    _add_descriptive_options(posthoc_parser)
    posthoc_parser.add_argument("--posthoc-response", required=True)
    posthoc_parser.add_argument("--posthoc-factor", required=True)
    posthoc_parser.add_argument("--posthoc-method", action="append", choices=("tukey_hsd", "games_howell"), default=None)
    posthoc_parser.add_argument("--min-group-n", type=int, default=2)
    posthoc_parser.add_argument("--max-group-levels", type=int, default=20)
    _add_phase9a_options(posthoc_parser)
    posthoc_parser.set_defaults(handler=run_posthoc)

    marginal_parser = subparsers.add_parser(
        "marginal-means", help="Estimate marginal means and pairwise model contrasts."
    )
    _add_descriptive_options(marginal_parser)
    marginal_parser.add_argument("--marginal-response", default=None)
    marginal_parser.add_argument("--marginal-factor", action="append", default=[])
    marginal_parser.add_argument("--marginal-covariate", action="append", default=[])
    marginal_parser.add_argument("--marginal-interaction", action="append", default=[])
    marginal_parser.add_argument("--marginal-formula", default=None)
    marginal_parser.add_argument("--marginal-term", action="append", default=[])
    marginal_parser.add_argument("--marginal-p-adjust", choices=("none", "fdr_bh"), default="fdr_bh")
    marginal_parser.add_argument("--max-interaction-order", type=int, default=3)
    _add_phase9a_options(marginal_parser)
    marginal_parser.set_defaults(handler=run_marginal_means)

    mixed_parser = subparsers.add_parser(
        "mixed-effects", help="Fit a random-intercept model with optional numeric random slopes."
    )
    _add_descriptive_options(mixed_parser)
    mixed_parser.add_argument("--mixed-group", required=True)
    mixed_parser.add_argument("--mixed-response", default=None)
    mixed_parser.add_argument("--mixed-factor", action="append", default=[])
    mixed_parser.add_argument("--mixed-covariate", action="append", default=[])
    mixed_parser.add_argument("--mixed-interaction", action="append", default=[])
    mixed_parser.add_argument("--mixed-formula", default=None)
    mixed_parser.add_argument("--mixed-random-slope", action="append", default=[])
    mixed_parser.add_argument("--mixed-ml", action="store_true")
    mixed_parser.add_argument("--mixed-optimizer", default="lbfgs")
    mixed_parser.add_argument("--mixed-max-iter", type=int, default=1000)
    mixed_parser.add_argument("--mixed-min-groups", type=int, default=3)
    mixed_parser.add_argument("--mixed-min-group-n", type=int, default=2)
    mixed_parser.add_argument("--max-interaction-order", type=int, default=3)
    _add_phase9a_options(mixed_parser)
    mixed_parser.set_defaults(handler=run_mixed_effects)

    profile_parser = subparsers.add_parser("profile", help="Profile a tabular dataset.")
    _add_descriptive_options(profile_parser)
    profile_parser.set_defaults(handler=run_profile)

    univariate_parser = subparsers.add_parser(
        "univariate",
        help="Run profiling plus univariate descriptive statistics.",
    )
    _add_descriptive_options(univariate_parser)
    univariate_parser.set_defaults(handler=run_univariate)

    outliers_parser = subparsers.add_parser(
        "outliers",
        help="Run transparent univariate outlier and numeric quality diagnostics.",
    )
    _add_descriptive_options(outliers_parser)
    outliers_parser.add_argument(
        "--outlier-method",
        action="append",
        choices=("iqr", "robust_z"),
        default=None,
        help="Outlier rule; repeatable. Defaults to IQR and robust Z.",
    )
    outliers_parser.add_argument("--iqr-multiplier", type=float, default=1.5)
    outliers_parser.add_argument("--robust-z-threshold", type=float, default=3.5)
    outliers_parser.add_argument(
        "--include-flags",
        action="store_true",
        help="Persist observation-level flags in addition to variable summaries.",
    )
    outliers_parser.set_defaults(handler=run_outliers)

    bivariate_parser = subparsers.add_parser(
        "bivariate",
        help="Run mixed-type bivariate associations and inference.",
    )
    _add_descriptive_options(bivariate_parser)
    bivariate_parser.add_argument(
        "--correlation",
        action="append",
        choices=("pearson", "spearman", "kendall"),
        default=None,
        help="Correlation method; repeatable. Defaults to all supported methods.",
    )
    bivariate_parser.add_argument(
        "--comparison-test",
        action="append",
        choices=(
            "welch_t",
            "mann_whitney",
            "welch_anova",
            "kruskal_wallis",
            "chi_square",
            "fisher_exact",
        ),
        default=None,
        help="Inferential test; repeatable. Defaults to all supported tests.",
    )
    bivariate_parser.add_argument("--min-group-n", type=int, default=3)
    bivariate_parser.add_argument("--min-correlation-pairs", type=int, default=3)
    bivariate_parser.add_argument("--max-group-levels", type=int, default=20)
    bivariate_parser.add_argument("--max-correlation-columns", type=int, default=100)
    bivariate_parser.add_argument(
        "--pairwise", action="store_true", help="Enable pairwise post-hoc tests for multi-group comparisons."
    )
    bivariate_parser.add_argument(
        "--p-adjust", choices=("none", "fdr_bh"), default="fdr_bh"
    )
    bivariate_parser.set_defaults(handler=run_bivariate)

    groups_parser = subparsers.add_parser(
        "groups",
        help="Analyze one or more statistical responses across grouping variables.",
    )
    _add_descriptive_options(groups_parser)
    groups_parser.add_argument(
        "--group", action="append", default=[],
        help="Grouping column; repeatable. Defaults to columns declared as factors.",
    )
    groups_parser.add_argument("--min-group-n", type=int, default=3)
    groups_parser.add_argument("--max-group-levels", type=int, default=20)
    groups_parser.add_argument(
        "--comparison-test",
        action="append",
        choices=(
            "welch_t", "mann_whitney", "welch_anova", "kruskal_wallis",
            "chi_square", "fisher_exact",
        ),
        default=None,
    )
    groups_parser.add_argument(
        "--p-adjust", choices=("none", "fdr_bh"), default="fdr_bh"
    )
    groups_parser.add_argument("--pairwise", action="store_true")
    groups_parser.add_argument("--annotation-file", default=None)
    groups_parser.add_argument("--annotation-name", default="annotations")
    groups_parser.add_argument("--annotation-id-column", default=None)
    groups_parser.add_argument(
        "--annotation-alignment", choices=("strict", "partial"), default="strict"
    )
    groups_parser.add_argument("--annotation-factor", action="append", default=[])
    groups_parser.add_argument("--annotation-response", action="append", default=[])
    groups_parser.add_argument("--annotation-covariate", action="append", default=[])
    groups_parser.add_argument("--annotation-numeric", action="append", default=[])
    groups_parser.add_argument("--annotation-categorical", action="append", default=[])
    groups_parser.set_defaults(handler=run_groups)

    multivariate_parser = subparsers.add_parser(
        "multivariate",
        help="Run covariance, collinearity, and Mahalanobis diagnostics on a feature matrix.",
    )
    multivariate_parser.add_argument("input", help="Feature matrix: CSV/TSV/TXT, NPY, or sparse NPZ.")
    multivariate_parser.add_argument("--id-column", default=None)
    multivariate_parser.add_argument("--ids-file", default=None)
    multivariate_parser.add_argument("--feature-column", action="append", default=[])
    multivariate_parser.add_argument("--scaling", choices=("none", "standard", "robust", "minmax"), default="none")
    multivariate_parser.add_argument("--no-spearman", action="store_true")
    multivariate_parser.add_argument("--no-robust-mahalanobis", action="store_true")
    multivariate_parser.add_argument("--mahalanobis-quantile", type=float, default=0.975)
    multivariate_parser.add_argument("--robust-support-fraction", type=float, default=None)
    multivariate_parser.add_argument("--random-state", type=int, default=0)
    multivariate_parser.add_argument("--max-covariance-features", type=int, default=200)
    multivariate_parser.add_argument("--max-collinearity-features", type=int, default=100)
    multivariate_parser.add_argument("--max-mahalanobis-features", type=int, default=100)
    multivariate_parser.add_argument("--output-dir", default=None)
    multivariate_parser.set_defaults(handler=run_multivariate)

    manova_parser = subparsers.add_parser(
        "manova",
        help="Run low/moderate-dimensional main-effects MANOVA on a tabular dataset.",
    )
    _add_descriptive_options(manova_parser)
    manova_parser.add_argument("--max-responses", type=int, default=20)
    manova_parser.add_argument("--max-factor-levels", type=int, default=20)
    manova_parser.add_argument("--min-level-n", type=int, default=3)
    manova_parser.set_defaults(handler=run_manova)

    factorial_parser = subparsers.add_parser(
        "factorial",
        help="Run explicit factorial ANOVA/ANCOVA with interactions and model diagnostics.",
    )
    _add_descriptive_options(factorial_parser)
    factorial_parser.add_argument(
        "--formula",
        default=None,
        help="Compact formula using direct column names and +, :, *, e.g. y ~ A * B + x.",
    )
    factorial_parser.add_argument(
        "--interaction",
        action="append",
        default=[],
        help="Explicit interaction as colon-separated predictors; repeatable.",
    )
    factorial_parser.add_argument("--ss-type", choices=("2", "3"), default="2")
    factorial_parser.add_argument(
        "--p-adjust", choices=("none", "fdr_bh"), default="none"
    )
    factorial_parser.add_argument(
        "--robust-covariance",
        choices=("none", "hc0", "hc1", "hc2", "hc3"),
        default="none",
    )
    factorial_parser.add_argument("--min-cell-n", type=int, default=2)
    factorial_parser.add_argument("--max-factor-levels", type=int, default=20)
    factorial_parser.add_argument("--max-design-cells", type=int, default=5000)
    factorial_parser.add_argument("--max-design-columns", type=int, default=500)
    factorial_parser.add_argument("--max-interaction-order", type=int, default=3)
    factorial_parser.add_argument("--diagnostic-alpha", type=float, default=0.05)
    factorial_parser.add_argument("--condition-number-threshold", type=float, default=30.0)
    factorial_parser.set_defaults(handler=run_factorial)

    representation_parser = subparsers.add_parser("representation", help="Compare two aligned numerical representation spaces.")
    representation_parser.add_argument("input_x")
    representation_parser.add_argument("input_y")
    representation_parser.add_argument("--x-id-column", default=None)
    representation_parser.add_argument("--y-id-column", default=None)
    representation_parser.add_argument("--x-ids-file", default=None)
    representation_parser.add_argument("--y-ids-file", default=None)
    representation_parser.add_argument("--x-feature-column", action="append", default=[])
    representation_parser.add_argument("--y-feature-column", action="append", default=[])
    representation_parser.add_argument("--representation-alignment", choices=("strict", "partial"), default="strict")
    representation_parser.add_argument("--cca-components", type=int, default=2)
    representation_parser.add_argument("--cca-scaling", choices=("none", "standard", "robust", "minmax"), default="standard")
    representation_parser.add_argument("--cca-max-iter", type=int, default=1000)
    representation_parser.add_argument("--cca-tol", type=float, default=1e-6)
    representation_parser.add_argument("--distance-metric", default="euclidean")
    representation_parser.add_argument("--distance-similarity-method", choices=("pearson", "spearman"), default="spearman")
    representation_parser.add_argument("--mantel-permutations", type=int, default=999)
    representation_parser.add_argument("--random-state", type=int, default=0)
    representation_parser.add_argument("--output-dir", default=None)
    representation_parser.set_defaults(handler=run_representation)

    compositional_parser = subparsers.add_parser("compositional", help="Run explicit compositional log-ratio analysis.")
    compositional_parser.add_argument("input")
    compositional_parser.add_argument("--id-column", default=None)
    compositional_parser.add_argument("--ids-file", default=None)
    compositional_parser.add_argument("--feature-column", action="append", default=[])
    compositional_parser.add_argument("--compositional-transform", choices=("clr", "alr", "ilr"), default="clr")
    compositional_parser.add_argument("--replace-zeros", action="store_true")
    compositional_parser.add_argument("--zero-replacement-fraction", type=float, default=0.65)
    compositional_parser.add_argument("--alr-denominator", type=int, default=-1)
    compositional_parser.add_argument("--output-dir", default=None)
    compositional_parser.set_defaults(handler=run_compositional)

    bayesian_parser = subparsers.add_parser("bayesian", help="Run scoped Bayesian exploratory estimation.")
    _add_descriptive_options(bayesian_parser)
    bayesian_parser.add_argument("--group", action="append", default=[])
    bayesian_parser.add_argument("--bayesian-variable", action="append", default=[])
    bayesian_parser.add_argument("--bayesian-group", action="append", default=[])
    bayesian_parser.add_argument("--bayesian-credible-level", type=float, default=0.95)
    bayesian_parser.add_argument("--bayesian-rope-low", type=float, default=-0.1)
    bayesian_parser.add_argument("--bayesian-rope-high", type=float, default=0.1)
    bayesian_parser.add_argument("--bayesian-draws", type=int, default=5000)
    bayesian_parser.add_argument("--bayesian-min-n", type=int, default=3)
    bayesian_parser.add_argument("--random-state", type=int, default=0)
    bayesian_parser.set_defaults(handler=run_bayesian)

    anomaly_parser = subparsers.add_parser("anomaly", help="Score multivariate anomalies without modifying observations.")
    anomaly_parser.add_argument("input")
    anomaly_parser.add_argument("--id-column", default=None)
    anomaly_parser.add_argument("--ids-file", default=None)
    anomaly_parser.add_argument("--feature-column", action="append", default=[])
    anomaly_parser.add_argument("--anomaly-method", action="append", choices=("isolation_forest", "lof"), default=None)
    anomaly_parser.add_argument("--anomaly-scaling", choices=("none", "standard", "robust", "minmax"), default="none")
    anomaly_parser.add_argument("--contamination", default="auto")
    anomaly_parser.add_argument("--isolation-estimators", type=int, default=200)
    anomaly_parser.add_argument("--lof-neighbors", type=int, default=20)
    anomaly_parser.add_argument("--random-state", type=int, default=0)
    anomaly_parser.add_argument("--output-dir", default=None)
    anomaly_parser.set_defaults(handler=run_anomaly)

    project_parser = subparsers.add_parser(
        "project",
        help="Run PCA or an exploratory nonlinear projection on a feature matrix.",
    )
    project_parser.add_argument("input", help="Feature matrix: CSV/TSV/TXT, NPY, or sparse NPZ.")
    project_parser.add_argument("--method", choices=("pca", "tsne", "umap"), default="pca")
    project_parser.add_argument("--id-column", default=None)
    project_parser.add_argument("--ids-file", default=None, help="One observation ID per line for NPY/NPZ inputs.")
    project_parser.add_argument("--feature-column", action="append", default=[], help="Feature column for tabular inputs; repeatable.")
    project_parser.add_argument("--n-components", type=int, default=2)
    project_parser.add_argument("--scaling", choices=("none", "standard", "robust", "minmax"), default="none")
    project_parser.add_argument("--metric", default="euclidean")
    project_parser.add_argument("--random-state", type=int, default=0)
    project_parser.add_argument("--perplexity", type=float, default=30.0)
    project_parser.add_argument("--max-iter", type=int, default=1000)
    project_parser.add_argument("--n-neighbors", type=int, default=15)
    project_parser.add_argument("--min-dist", type=float, default=0.1)
    project_parser.add_argument("--output-dir", default=None)
    project_parser.set_defaults(handler=run_project)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Ruddy CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        if argv is None or len(argv) == 0:
            parser.print_help()
        return 0
    try:
        return int(args.handler(args))
    except (TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
