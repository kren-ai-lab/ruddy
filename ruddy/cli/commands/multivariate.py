"""CLI commands for multivariate feature diagnostics and MANOVA."""

from __future__ import annotations

import json
from pathlib import Path

from ruddy.cli.commands.descriptive import build_config, load_dataset
from ruddy.cli.commands.projections import load_feature_matrix
from ruddy.core.io import write_json, write_table
from ruddy.multivariate import (
    MANOVAResult,
    MultivariateResult,
    analyze_manova,
    analyze_multivariate,
)


def write_multivariate_result(
    result: MultivariateResult, output_dir: str | Path
) -> Path:
    """Persist structured multivariate diagnostics."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    cov = result.covariance
    write_table(
        cov.covariance,
        target / "covariance_matrix.csv",
        index=True,
        index_label="feature",
    )
    write_table(
        cov.pearson, target / "pearson_matrix.csv", index=True, index_label="feature"
    )
    write_table(
        cov.spearman, target / "spearman_matrix.csv", index=True, index_label="feature"
    )
    write_table(
        cov.pairwise_counts,
        target / "pairwise_counts.csv",
        index=True,
        index_label="feature",
    )
    write_table(cov.feature_diagnostics, target / "covariance_feature_diagnostics.csv")
    write_table(cov.condition_spectrum, target / "covariance_condition_spectrum.csv")

    col = result.collinearity
    write_table(col.features, target / "collinearity.csv")
    write_table(col.condition_spectrum, target / "collinearity_condition_spectrum.csv")

    mah = result.mahalanobis
    write_table(mah.methods, target / "mahalanobis_methods.csv")
    write_table(mah.distances, target / "mahalanobis_distances.csv")
    if not cov.exclusions.empty:
        write_table(cov.exclusions, target / "multivariate_exclusions.csv")

    write_json(cov.summary, target / "covariance_summary.json")
    write_json(col.summary, target / "collinearity_summary.json")
    write_json(
        {
            "status": result.status.value,
            "reason": result.reason,
            "provenance": result.provenance.to_dict(),
            "components": {
                "covariance": {
                    "status": cov.status.value,
                    "reason": cov.reason,
                    "provenance": cov.provenance.to_dict(),
                },
                "collinearity": {
                    "status": col.status.value,
                    "reason": col.reason,
                    "provenance": col.provenance.to_dict(),
                },
                "mahalanobis": {
                    "status": mah.status.value,
                    "reason": mah.reason,
                    "provenance": mah.provenance.to_dict(),
                },
            },
        },
        target / "multivariate_provenance.json",
    )
    return target


def write_manova_result(result: MANOVAResult, output_dir: str | Path) -> Path:
    """Persist MANOVA statistics, complete-case diagnostics, and provenance."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_table(result.tests, target / "manova_tests.csv")
    write_table(result.factor_levels, target / "manova_factor_levels.csv")
    if not result.exclusions.empty:
        write_table(result.exclusions, target / "manova_exclusions.csv")
    write_json(result.model_summary, target / "manova_model_summary.json")
    write_json(
        {
            "status": result.status.value,
            "reason": result.reason,
            "provenance": result.provenance.to_dict(),
        },
        target / "manova_provenance.json",
    )
    return target


def run_multivariate(args) -> int:
    features = load_feature_matrix(
        args.input,
        id_column=args.id_column,
        feature_columns=tuple(args.feature_column or ()),
        ids_file=args.ids_file,
    )
    result = analyze_multivariate(
        features,
        scaling=args.scaling,
        include_spearman=not args.no_spearman,
        include_robust_mahalanobis=not args.no_robust_mahalanobis,
        mahalanobis_threshold_quantile=args.mahalanobis_quantile,
        robust_support_fraction=args.robust_support_fraction,
        random_state=args.random_state,
        max_covariance_features=args.max_covariance_features,
        max_collinearity_features=args.max_collinearity_features,
        max_mahalanobis_features=args.max_mahalanobis_features,
    )
    if args.output_dir:
        print(write_multivariate_result(result, args.output_dir))
    else:
        print(
            json.dumps(
                {
                    "status": result.status.value,
                    "reason": result.reason,
                    "covariance_status": result.covariance.status.value,
                    "collinearity_status": result.collinearity.status.value,
                    "mahalanobis_status": result.mahalanobis.status.value,
                    "n_features": features.n_features,
                    "n_observations": features.n_observations,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


def run_manova(args) -> int:
    if len(args.response or []) < 2:
        raise ValueError("ruddy manova requires at least two --response columns.")
    if not args.factor:
        raise ValueError("ruddy manova requires at least one --factor column.")
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    result = analyze_manova(
        dataset,
        responses=tuple(args.response),
        factors=tuple(args.factor),
        covariates=tuple(args.covariate or ()),
        max_responses=args.max_responses,
        max_factor_levels=args.max_factor_levels,
        min_level_n=args.min_level_n,
    )
    if args.output_dir:
        print(write_manova_result(result, args.output_dir))
    else:
        print(
            json.dumps(
                {
                    "status": result.status.value,
                    "reason": result.reason,
                    "n_tests": int(result.tests.shape[0]),
                    "n_complete_case": result.model_summary.get("n_complete_case", 0),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0
