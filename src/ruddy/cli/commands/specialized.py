"""CLI commands for representation-space and specialized Phase 9C analyses."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ruddy.anomaly import analyze_anomalies
from ruddy.bayesian import analyze_bayesian_eda
from ruddy.compositional import analyze_composition
from ruddy.representation import analyze_representation_similarity
from ruddy.cli.commands.descriptive import build_config, load_dataset
from ruddy.cli.commands.projections import load_feature_matrix


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_representation_result(result, output_dir: str | Path) -> Path:
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    result.cca.correlations.to_csv(target / "cca_correlations.csv", index=False)
    result.cca.x_weights.to_csv(target / "cca_x_weights.csv", index=False)
    result.cca.y_weights.to_csv(target / "cca_y_weights.csv", index=False)
    result.cca.x_loadings.to_csv(target / "cca_x_loadings.csv", index=False)
    result.cca.y_loadings.to_csv(target / "cca_y_loadings.csv", index=False)
    result.cca.x_scores.to_csv(target / "cca_x_scores.csv", index=False)
    result.cca.y_scores.to_csv(target / "cca_y_scores.csv", index=False)
    result.cka.to_csv(target / "cka.csv", index=False)
    result.procrustes.to_csv(target / "procrustes.csv", index=False)
    result.distance_similarity.to_csv(target / "distance_similarity.csv", index=False)
    result.mantel.to_csv(target / "mantel.csv", index=False)
    if not result.exclusions.empty:
        result.exclusions.to_csv(target / "representation_exclusions.csv", index=False)
    _write_json(target / "representation_alignment.json", result.alignment.to_dict())
    _write_json(target / "representation_provenance.json", result.provenance.to_dict())
    return target


def write_compositional_result(result, output_dir: str | Path) -> Path:
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    transformed = pd.DataFrame(result.transformed.to_array(), columns=result.transformed.feature_names)
    transformed.insert(0, "observation_id", result.transformed.observation_ids.to_list())
    transformed.to_csv(target / "compositional_transformed.csv", index=False)
    result.variation_matrix.to_csv(target / "variation_matrix.csv", index=True)
    result.aitchison_distances.to_csv(target / "aitchison_distances.csv", index=True)
    result.zero_replacement.to_csv(target / "zero_replacement.csv", index=False)
    _write_json(target / "compositional_provenance.json", result.provenance.to_dict())
    return target


def write_bayesian_result(result, output_dir: str | Path) -> Path:
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    result.means.to_csv(target / "bayesian_means.csv", index=False)
    result.mean_differences.to_csv(target / "bayesian_mean_differences.csv", index=False)
    _write_json(target / "bayesian_provenance.json", result.provenance.to_dict())
    return target


def write_anomaly_result(result, output_dir: str | Path) -> Path:
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    result.scores.to_csv(target / "anomaly_scores.csv", index=False)
    result.methods.to_csv(target / "anomaly_methods.csv", index=False)
    if not result.exclusions.empty:
        result.exclusions.to_csv(target / "anomaly_exclusions.csv", index=False)
    _write_json(target / "anomaly_provenance.json", result.provenance.to_dict())
    return target


def run_representation(args) -> int:
    x = load_feature_matrix(args.input_x, id_column=args.x_id_column, feature_columns=tuple(args.x_feature_column or ()), ids_file=args.x_ids_file)
    y = load_feature_matrix(args.input_y, id_column=args.y_id_column, feature_columns=tuple(args.y_feature_column or ()), ids_file=args.y_ids_file)
    result = analyze_representation_similarity(
        x, y, alignment=args.representation_alignment, cca_components=args.cca_components,
        cca_scaling=args.cca_scaling, cca_max_iter=args.cca_max_iter, cca_tol=args.cca_tol,
        distance_metric=args.distance_metric, distance_similarity_method=args.distance_similarity_method,
        mantel_permutations=args.mantel_permutations, random_state=args.random_state,
    )
    if args.output_dir:
        print(write_representation_result(result, args.output_dir))
    else:
        print(json.dumps({"cka": result.cka.iloc[0].to_dict(), "mantel": result.mantel.iloc[0].to_dict(), "cca_status": result.cca.status.value}, indent=2, default=str))
    return 0


def run_compositional(args) -> int:
    features = load_feature_matrix(args.input, id_column=args.id_column, feature_columns=tuple(args.feature_column or ()), ids_file=args.ids_file)
    result = analyze_composition(features, transform=args.compositional_transform, replace_zeros=args.replace_zeros, zero_replacement_fraction=args.zero_replacement_fraction, alr_denominator=args.alr_denominator)
    if args.output_dir:
        print(write_compositional_result(result, args.output_dir))
    else:
        print(json.dumps({"transform": args.compositional_transform, "shape": result.transformed.shape}, indent=2))
    return 0


def run_bayesian(args) -> int:
    config = build_config(args); dataset = load_dataset(args.input, config)
    result = analyze_bayesian_eda(dataset, **config.bayesian_kwargs())
    if args.output_dir:
        print(write_bayesian_result(result, args.output_dir))
    else:
        print(json.dumps({"mean_rows": len(result.means), "difference_rows": len(result.mean_differences)}, indent=2))
    return 0


def run_anomaly(args) -> int:
    features = load_feature_matrix(args.input, id_column=args.id_column, feature_columns=tuple(args.feature_column or ()), ids_file=args.ids_file)
    result = analyze_anomalies(features, methods=tuple(args.anomaly_method or ("isolation_forest", "lof")), scaling=args.anomaly_scaling, contamination=(args.contamination if args.contamination == "auto" else float(args.contamination)), isolation_estimators=args.isolation_estimators, lof_neighbors=args.lof_neighbors, random_state=args.random_state)
    if args.output_dir:
        print(write_anomaly_result(result, args.output_dir))
    else:
        print(json.dumps({"score_rows": len(result.scores), "methods": len(result.methods)}, indent=2))
    return 0


__all__ = [
    "run_anomaly", "run_bayesian", "run_compositional", "run_representation",
    "write_anomaly_result", "write_bayesian_result", "write_compositional_result", "write_representation_result",
]
