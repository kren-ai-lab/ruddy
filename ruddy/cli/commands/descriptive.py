"""CLI helpers for descriptive profiling and univariate analysis."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from ruddy.analysis import AnalysisConfig
from ruddy.cli._console import record_dataset
from ruddy.core import ColumnKind, ColumnRole, UnknownColumnError
from ruddy.core.io import read_table, write_json, write_table
from ruddy.data import TabularDataset
from ruddy.profiling import ProfilingResult, profile_dataset
from ruddy.univariate import UnivariateResult, analyze_univariate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ruddy.cli._options import CliArgs


def _build_role_overrides(args: CliArgs) -> dict[str, ColumnRole]:
    overrides: dict[str, ColumnRole] = {}
    for attr, role in (
        ("response", ColumnRole.RESPONSE),
        ("factor", ColumnRole.FACTOR),
        ("covariate", ColumnRole.COVARIATE),
        ("annotation", ColumnRole.ANNOTATION),
        ("exclude", ColumnRole.EXCLUDED),
    ):
        values = getattr(args, attr, None)
        if values is None:
            continue
        if isinstance(values, str):
            values = [values]
        for column in values:
            if column in overrides and overrides[column] is not role:
                msg = (
                    f"Column {column!r} was assigned multiple CLI roles: "
                    f"{overrides[column].value!r} and {role.value!r}."
                )
                raise ValueError(msg)
            overrides[column] = role
    return overrides


def _build_kind_overrides(args: CliArgs) -> dict[str, ColumnKind]:
    overrides: dict[str, ColumnKind] = {}
    for attr, kind in (
        ("numeric", ColumnKind.NUMERIC),
        ("categorical", ColumnKind.CATEGORICAL),
    ):
        for column in getattr(args, attr, None) or []:
            if column in overrides and overrides[column] is not kind:
                msg = f"Column {column!r} was assigned multiple CLI kinds."
                raise ValueError(msg)
            overrides[column] = kind
    return overrides


def build_config(args: CliArgs) -> AnalysisConfig:
    """Translate CLI options into an AnalysisConfig."""
    return AnalysisConfig(
        id_column=args.id_column,
        random_state=getattr(args, "random_state", 0),
        responses=tuple(getattr(args, "response", None) or ()),
        groups=tuple(getattr(args, "group", None) or ()),
        enabled_blocks=tuple(getattr(args, "enable", None) or ("profiling", "univariate")),
        feature_alignment=getattr(args, "feature_alignment", "strict"),
        factorial_response=getattr(args, "factorial_response", None),
        factorial_interactions=tuple(
            tuple(part.strip() for part in value.split(":"))
            for value in (getattr(args, "interaction", None) or ())
        ),
        factorial_formula=getattr(args, "factorial_formula", None),
        role_overrides=_build_role_overrides(args),
        kind_overrides=_build_kind_overrides(args),
        min_numeric_n=args.min_numeric_n,
        max_category_levels=args.max_category_levels,
        max_missingness_patterns=args.max_missingness_patterns,
        max_pairwise_columns=args.max_pairwise_columns,
        correlations=(
            tuple(args.correlation)
            if getattr(args, "correlation", None)
            else ("pearson", "spearman", "kendall")
        ),
        comparison_tests=(
            tuple(args.comparison_test)
            if getattr(args, "comparison_test", None)
            else (
                "welch_t",
                "mann_whitney",
                "welch_anova",
                "kruskal_wallis",
                "chi_square",
                "fisher_exact",
            )
        ),
        p_adjust=getattr(args, "p_adjust", "fdr_bh"),
        min_group_n=getattr(args, "min_group_n", 3),
        min_correlation_pairs=getattr(args, "min_correlation_pairs", 3),
        max_group_levels=getattr(args, "max_group_levels", 20),
        max_correlation_columns=getattr(args, "max_correlation_columns", 100),
        pairwise=getattr(args, "pairwise", False),
        outlier_methods=(
            tuple(args.outlier_method) if getattr(args, "outlier_method", None) else ("iqr", "robust_z")
        ),
        outlier_iqr_multiplier=getattr(args, "iqr_multiplier", 1.5),
        outlier_robust_z_threshold=getattr(args, "robust_z_threshold", 3.5),
        include_outlier_flags=getattr(args, "include_flags", False),
        projection_scaling=getattr(args, "projection_scaling", "none"),
        projection_n_components=getattr(args, "projection_n_components", 2),
        projection_metric=getattr(args, "projection_metric", "euclidean"),
        tsne_perplexity=getattr(args, "tsne_perplexity", 30.0),
        tsne_max_iter=getattr(args, "tsne_max_iter", 1000),
        umap_n_neighbors=getattr(args, "umap_n_neighbors", 15),
        umap_min_dist=getattr(args, "umap_min_dist", 0.1),
        multivariate_scaling=getattr(args, "multivariate_scaling", "none"),
        multivariate_include_spearman=not getattr(args, "no_spearman", False),
        multivariate_max_covariance_features=getattr(args, "max_covariance_features", 200),
        multivariate_max_collinearity_features=getattr(args, "max_collinearity_features", 100),
        multivariate_max_mahalanobis_features=getattr(args, "max_mahalanobis_features", 100),
        mahalanobis_threshold_quantile=getattr(args, "mahalanobis_quantile", 0.975),
        mahalanobis_include_robust=not getattr(args, "no_robust_mahalanobis", False),
        mahalanobis_robust_support_fraction=getattr(args, "robust_support_fraction", None),
        manova_max_responses=getattr(args, "manova_max_responses", 20),
        manova_max_factor_levels=getattr(args, "manova_max_factor_levels", 20),
        manova_min_level_n=getattr(args, "manova_min_level_n", 3),
        factorial_ss_type=int(getattr(args, "ss_type", 2)),
        factorial_p_adjust=(
            getattr(args, "factorial_p_adjust", "none")
            if getattr(args, "command", None) == "analyze"
            else (
                getattr(args, "p_adjust", "none") if getattr(args, "command", None) == "factorial" else "none"
            )
        ),
        factorial_robust_covariance=(
            None if getattr(args, "robust_covariance", None) in {None, "none"} else args.robust_covariance
        ),
        factorial_min_cell_n=getattr(args, "min_cell_n", 2),
        factorial_max_factor_levels=getattr(args, "max_factor_levels", 20),
        factorial_max_design_cells=getattr(args, "max_design_cells", 5000),
        factorial_max_design_columns=getattr(args, "max_design_columns", 500),
        factorial_max_interaction_order=getattr(args, "max_interaction_order", 3),
        factorial_diagnostic_alpha=getattr(args, "diagnostic_alpha", 0.05),
        factorial_condition_number_threshold=getattr(args, "condition_number_threshold", 30.0),
        distribution_max_shapiro_n=getattr(args, "max_shapiro_n", 5000),
        dependence_partial_covariates=tuple(getattr(args, "partial_covariate", None) or ()),
        dependence_min_complete_pairs=getattr(args, "dependence_min_pairs", 5),
        dependence_max_columns=getattr(args, "dependence_max_columns", 50),
        dependence_n_permutations=getattr(args, "dependence_permutations", 199),
        dependence_mi_neighbors=getattr(args, "mi_neighbors", 3),
        confidence_level=getattr(args, "confidence_level", 0.95),
        interval_correlations=(tuple(getattr(args, "interval_correlation", None) or ()) or ("pearson",)),
        bootstrap_resamples=getattr(args, "bootstrap_resamples", 1000),
        bootstrap_method=getattr(args, "bootstrap_method", "bca"),
        permanova_factor=getattr(args, "permanova_factor", None),
        permanova_metric=getattr(args, "permanova_metric", "euclidean"),
        permanova_permutations=getattr(args, "permanova_permutations", 999),
        permanova_min_group_n=getattr(args, "permanova_min_group_n", getattr(args, "min_group_n", 3)),
        permanova_max_group_levels=getattr(
            args, "permanova_max_group_levels", getattr(args, "max_group_levels", 20)
        ),
        posthoc_response=getattr(args, "posthoc_response", None),
        posthoc_factor=getattr(args, "posthoc_factor", None),
        posthoc_methods=tuple(getattr(args, "posthoc_method", None) or ("tukey_hsd", "games_howell")),
        marginal_response=getattr(args, "marginal_response", None),
        marginal_factors=tuple(getattr(args, "marginal_factor", None) or ()),
        marginal_covariates=tuple(getattr(args, "marginal_covariate", None) or ()),
        marginal_interactions=tuple(
            tuple(part.strip() for part in value.split(":"))
            for value in (getattr(args, "marginal_interaction", None) or ())
        ),
        marginal_formula=getattr(args, "marginal_formula", None),
        marginal_terms=tuple(
            tuple(part.strip() for part in value.split(":"))
            for value in (getattr(args, "marginal_term", None) or ())
        ),
        marginal_p_adjust=getattr(args, "marginal_p_adjust", "fdr_bh"),
        mixed_group=getattr(args, "mixed_group", None),
        mixed_response=getattr(args, "mixed_response", None),
        mixed_factors=tuple(getattr(args, "mixed_factor", None) or ()),
        mixed_covariates=tuple(getattr(args, "mixed_covariate", None) or ()),
        mixed_interactions=tuple(
            tuple(part.strip() for part in value.split(":"))
            for value in (getattr(args, "mixed_interaction", None) or ())
        ),
        mixed_formula=getattr(args, "mixed_formula", None),
        mixed_random_slopes=tuple(getattr(args, "mixed_random_slope", None) or ()),
        mixed_reml=not getattr(args, "mixed_ml", False),
        mixed_optimizer=getattr(args, "mixed_optimizer", "lbfgs"),
        mixed_max_iter=getattr(args, "mixed_max_iter", 1000),
        mixed_min_groups=getattr(args, "mixed_min_groups", 3),
        mixed_min_group_n=getattr(args, "mixed_min_group_n", 2),
        representation_alignment=getattr(args, "representation_alignment", "strict"),
        representation_cca_components=getattr(args, "cca_components", 2),
        representation_cca_scaling=getattr(args, "cca_scaling", "standard"),
        representation_cca_max_iter=getattr(args, "cca_max_iter", 1000),
        representation_cca_tol=getattr(args, "cca_tol", 1e-6),
        representation_distance_metric=getattr(args, "distance_metric", "euclidean"),
        representation_distance_similarity_method=getattr(args, "distance_similarity_method", "spearman"),
        representation_mantel_permutations=getattr(args, "mantel_permutations", 999),
        compositional_transform=getattr(args, "compositional_transform", "clr"),
        compositional_replace_zeros=getattr(args, "replace_zeros", False),
        compositional_zero_replacement_fraction=getattr(args, "zero_replacement_fraction", 0.65),
        compositional_alr_denominator=getattr(args, "alr_denominator", -1),
        bayesian_variables=tuple(getattr(args, "bayesian_variable", None) or ()),
        bayesian_groups=tuple(getattr(args, "bayesian_group", None) or ()),
        bayesian_credible_level=getattr(args, "bayesian_credible_level", 0.95),
        bayesian_rope=(
            getattr(args, "bayesian_rope_low", -0.1),
            getattr(args, "bayesian_rope_high", 0.1),
        ),
        bayesian_draws=getattr(args, "bayesian_draws", 5000),
        bayesian_min_n=getattr(args, "bayesian_min_n", 3),
        anomaly_methods=tuple(getattr(args, "anomaly_method", None) or ("isolation_forest", "lof")),
        anomaly_scaling=getattr(args, "anomaly_scaling", "none"),
        anomaly_contamination=(
            getattr(args, "contamination", "auto")
            if getattr(args, "contamination", "auto") == "auto"
            else float(args.contamination)
        ),
        anomaly_isolation_estimators=getattr(args, "isolation_estimators", 200),
        anomaly_lof_neighbors=getattr(args, "lof_neighbors", 20),
    )


def load_dataset(path: str | Path, config: AnalysisConfig) -> TabularDataset:
    """Load a tabular file into Ruddy without scientific transformation."""
    frame = read_table(path)
    try:
        dataset = TabularDataset(frame, **config.dataset_kwargs())
    except UnknownColumnError as exc:
        message = f"{exc} Available columns: {list(frame.columns)}."
        raise UnknownColumnError(message) from exc
    record_dataset(dataset)
    return dataset


def _write_tables(output_dir: Path, tables: Iterable[tuple[str, pl.DataFrame | pd.DataFrame]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables:
        write_table(table, output_dir / f"{name}.csv")


def write_profiling_result(result: ProfilingResult, output_dir: str | Path) -> Path:
    """Persist structured profiling outputs as JSON/CSV artifacts."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_json(result.overview, target / "overview.json")
    write_json(result.provenance.to_dict(), target / "provenance.json")
    _write_tables(
        target,
        (
            ("columns", result.columns),
            ("missingness", result.missingness),
            ("pairwise_completeness", result.pairwise_completeness),
            ("missingness_patterns", result.missingness_patterns),
        ),
    )
    return target


def write_univariate_result(result: UnivariateResult, output_dir: str | Path) -> Path:
    """Persist structured univariate outputs as JSON/CSV artifacts."""
    target = write_profiling_result(result.profiling, output_dir)
    write_json(result.provenance.to_dict(), target / "univariate_provenance.json")
    _write_tables(
        target,
        (
            ("numeric_statistics", result.numeric_statistics),
            ("categorical_statistics", result.categorical_statistics),
            ("categorical_frequencies", result.categorical_frequencies),
            ("datetime_statistics", result.datetime_statistics),
        ),
    )
    return target


def run_profile(args: CliArgs) -> int:
    """Run the ``ruddy inspect profile`` command."""
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    result = profile_dataset(dataset, **config.profiling_kwargs())
    if args.output_dir:
        path = write_profiling_result(result, args.output_dir)
        print(path)
    else:
        print(json.dumps(result.overview, indent=2, sort_keys=True))
    return 0


def run_univariate(args: CliArgs) -> int:
    """Run the ``ruddy inspect univariate`` command."""
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    result = analyze_univariate(dataset, **config.univariate_kwargs())
    if args.output_dir:
        path = write_univariate_result(result, args.output_dir)
        print(path)
    else:
        payload = {
            "overview": result.profiling.overview,
            "numeric_variables": len(result.numeric_statistics),
            "categorical_variables": len(result.categorical_statistics),
            "datetime_variables": len(result.datetime_statistics),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
