"""CLI commands for representation-space and specialized Phase 9C analyses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from ruddy.anomaly import analyze_anomalies
from ruddy.bayesian import analyze_bayesian_eda
from ruddy.cli.commands.descriptive import build_config, load_dataset
from ruddy.cli.commands.projections import load_feature_matrix
from ruddy.compositional import analyze_composition
from ruddy.core.io import write_json, write_table
from ruddy.representation import analyze_representation_similarity

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ruddy.anomaly import AnomalyResult
    from ruddy.bayesian import BayesianEDAResult
    from ruddy.cli._options import CliArgs
    from ruddy.compositional import CompositionalResult
    from ruddy.representation import RepresentationComparisonResult


def write_representation_result(result: RepresentationComparisonResult, output_dir: str | Path) -> Path:
    """Persist structured representation-comparison artifacts under ``output_dir``."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_table(result.cca.correlations, target / "cca_correlations.csv")
    write_table(result.cca.x_weights, target / "cca_x_weights.csv")
    write_table(result.cca.y_weights, target / "cca_y_weights.csv")
    write_table(result.cca.x_loadings, target / "cca_x_loadings.csv")
    write_table(result.cca.y_loadings, target / "cca_y_loadings.csv")
    write_table(result.cca.x_scores, target / "cca_x_scores.csv")
    write_table(result.cca.y_scores, target / "cca_y_scores.csv")
    write_table(result.cka, target / "cka.csv")
    write_table(result.procrustes, target / "procrustes.csv")
    write_table(result.distance_similarity, target / "distance_similarity.csv")
    write_table(result.mantel, target / "mantel.csv")
    if not result.exclusions.empty:
        write_table(result.exclusions, target / "representation_exclusions.csv")
    write_json(result.alignment.to_dict(), target / "representation_alignment.json")
    write_json(result.provenance.to_dict(), target / "representation_provenance.json")
    return target


def write_compositional_result(result: CompositionalResult, output_dir: str | Path) -> Path:
    """Persist structured compositional artifacts under ``output_dir``."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    transformed = pd.DataFrame(result.transformed.to_array(), columns=result.transformed.feature_names)
    transformed.insert(0, "observation_id", result.transformed.observation_ids.to_list())
    write_table(transformed, target / "compositional_transformed.csv")
    write_table(result.variation_matrix, target / "variation_matrix.csv", index=True)
    write_table(result.aitchison_distances, target / "aitchison_distances.csv", index=True)
    write_table(result.zero_replacement, target / "zero_replacement.csv")
    write_json(result.provenance.to_dict(), target / "compositional_provenance.json")
    return target


def write_bayesian_result(result: BayesianEDAResult, output_dir: str | Path) -> Path:
    """Persist structured Bayesian EDA artifacts under ``output_dir``."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_table(result.means, target / "bayesian_means.csv")
    write_table(result.mean_differences, target / "bayesian_mean_differences.csv")
    write_json(result.provenance.to_dict(), target / "bayesian_provenance.json")
    return target


def write_anomaly_result(result: AnomalyResult, output_dir: str | Path) -> Path:
    """Persist structured anomaly-scoring artifacts under ``output_dir``."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_table(result.scores, target / "anomaly_scores.csv")
    write_table(result.methods, target / "anomaly_methods.csv")
    if not result.exclusions.empty:
        write_table(result.exclusions, target / "anomaly_exclusions.csv")
    write_json(result.provenance.to_dict(), target / "anomaly_provenance.json")
    return target


def run_representation(args: CliArgs) -> int:
    """Run the ``ruddy analyze representation`` command."""
    x = load_feature_matrix(
        args.input_x,
        id_column=args.x_id_column,
        feature_columns=tuple(args.x_feature_column or ()),
        ids_file=args.x_ids_file,
    )
    y = load_feature_matrix(
        args.input_y,
        id_column=args.y_id_column,
        feature_columns=tuple(args.y_feature_column or ()),
        ids_file=args.y_ids_file,
    )
    result = analyze_representation_similarity(
        x,
        y,
        alignment=args.representation_alignment,
        cca_components=args.cca_components,
        cca_scaling=args.cca_scaling,
        cca_max_iter=args.cca_max_iter,
        cca_tol=args.cca_tol,
        distance_metric=args.distance_metric,
        distance_similarity_method=args.distance_similarity_method,
        mantel_permutations=args.mantel_permutations,
        random_state=args.random_state,
    )
    if args.output_dir:
        print(write_representation_result(result, args.output_dir))
    else:
        print(
            json.dumps(
                {
                    "cka": result.cka.iloc[0].to_dict(),
                    "mantel": result.mantel.iloc[0].to_dict(),
                    "cca_status": result.cca.status.value,
                },
                indent=2,
                default=str,
            )
        )
    return 0


def run_compositional(args: CliArgs) -> int:
    """Run the ``ruddy analyze compositional`` command."""
    features = load_feature_matrix(
        args.input,
        id_column=args.id_column,
        feature_columns=tuple(args.feature_column or ()),
        ids_file=args.ids_file,
    )
    result = analyze_composition(
        features,
        transform=args.compositional_transform,
        replace_zeros=args.replace_zeros,
        zero_replacement_fraction=args.zero_replacement_fraction,
        alr_denominator=args.alr_denominator,
    )
    if args.output_dir:
        print(write_compositional_result(result, args.output_dir))
    else:
        print(
            json.dumps(
                {
                    "transform": args.compositional_transform,
                    "shape": result.transformed.shape,
                },
                indent=2,
            )
        )
    return 0


def run_bayesian(args: CliArgs) -> int:
    """Run the ``ruddy analyze bayesian`` command."""
    config = build_config(args)
    dataset = load_dataset(args.input, config)
    result = analyze_bayesian_eda(dataset, **config.bayesian_kwargs())
    if args.output_dir:
        print(write_bayesian_result(result, args.output_dir))
    else:
        print(
            json.dumps(
                {
                    "mean_rows": len(result.means),
                    "difference_rows": len(result.mean_differences),
                },
                indent=2,
            )
        )
    return 0


def run_anomaly(args: CliArgs) -> int:
    """Run the ``ruddy analyze anomaly`` command."""
    features = load_feature_matrix(
        args.input,
        id_column=args.id_column,
        feature_columns=tuple(args.feature_column or ()),
        ids_file=args.ids_file,
    )
    result = analyze_anomalies(
        features,
        methods=tuple(args.anomaly_method or ("isolation_forest", "lof")),
        scaling=args.anomaly_scaling,
        contamination=(args.contamination if args.contamination == "auto" else float(args.contamination)),
        isolation_estimators=args.isolation_estimators,
        lof_neighbors=args.lof_neighbors,
        random_state=args.random_state,
    )
    if args.output_dir:
        print(write_anomaly_result(result, args.output_dir))
    else:
        print(
            json.dumps(
                {"score_rows": len(result.scores), "methods": len(result.methods)},
                indent=2,
            )
        )
    return 0


__all__ = [
    "run_anomaly",
    "run_bayesian",
    "run_compositional",
    "run_representation",
    "write_anomaly_result",
    "write_bayesian_result",
    "write_compositional_result",
    "write_representation_result",
]
