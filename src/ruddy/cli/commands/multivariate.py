"""CLI commands for multivariate feature diagnostics and MANOVA."""

from __future__ import annotations

import json
from pathlib import Path

from ruddy.cli.commands.descriptive import build_config, load_dataset
from ruddy.cli.commands.projections import load_feature_matrix
from ruddy.multivariate import MANOVAResult, MultivariateResult, analyze_manova, analyze_multivariate


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_multivariate_result(result: MultivariateResult, output_dir: str | Path) -> Path:
    """Persist structured multivariate diagnostics."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    cov = result.covariance
    cov.covariance.to_csv(target / "covariance_matrix.csv", index=True, index_label="feature")
    cov.pearson.to_csv(target / "pearson_matrix.csv", index=True, index_label="feature")
    cov.spearman.to_csv(target / "spearman_matrix.csv", index=True, index_label="feature")
    cov.pairwise_counts.to_csv(target / "pairwise_counts.csv", index=True, index_label="feature")
    cov.feature_diagnostics.to_csv(target / "covariance_feature_diagnostics.csv", index=False)
    cov.condition_spectrum.to_csv(target / "covariance_condition_spectrum.csv", index=False)

    col = result.collinearity
    col.features.to_csv(target / "collinearity.csv", index=False)
    col.condition_spectrum.to_csv(target / "collinearity_condition_spectrum.csv", index=False)

    mah = result.mahalanobis
    mah.methods.to_csv(target / "mahalanobis_methods.csv", index=False)
    mah.distances.to_csv(target / "mahalanobis_distances.csv", index=False)
    if not cov.exclusions.empty:
        cov.exclusions.to_csv(target / "multivariate_exclusions.csv", index=False)

    _write_json(target / "covariance_summary.json", cov.summary)
    _write_json(target / "collinearity_summary.json", col.summary)
    _write_json(
        target / "multivariate_provenance.json",
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
    )
    return target


def write_manova_result(result: MANOVAResult, output_dir: str | Path) -> Path:
    """Persist MANOVA statistics, complete-case diagnostics, and provenance."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    result.tests.to_csv(target / "manova_tests.csv", index=False)
    result.factor_levels.to_csv(target / "manova_factor_levels.csv", index=False)
    if not result.exclusions.empty:
        result.exclusions.to_csv(target / "manova_exclusions.csv", index=False)
    _write_json(target / "manova_model_summary.json", result.model_summary)
    _write_json(
        target / "manova_provenance.json",
        {
            "status": result.status.value,
            "reason": result.reason,
            "provenance": result.provenance.to_dict(),
        },
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
