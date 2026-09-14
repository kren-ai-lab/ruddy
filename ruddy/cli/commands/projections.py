"""CLI commands for PCA and exploratory feature-space projections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from ruddy.data import FeatureMatrix
from ruddy.projections import PCAResult, ProjectionResult, analyze_pca, analyze_tsne, analyze_umap


def _read_ids(path: str | Path) -> list[str]:
    values = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return [value for value in values if value]


def load_feature_matrix(
    path: str | Path,
    *,
    id_column: str | None = None,
    feature_columns: Sequence[str] = (),
    ids_file: str | Path | None = None,
) -> FeatureMatrix:
    """Load a feature matrix without implicit scientific transformations."""

    source = Path(path)
    suffix = source.suffix.lower()
    ids = None if ids_file is None else _read_ids(ids_file)

    if suffix in {".csv", ".tsv", ".txt"}:
        frame = pd.read_csv(source, sep="\t" if suffix in {".tsv", ".txt"} else ",")
        if id_column is not None:
            if id_column not in frame.columns:
                raise ValueError(f"Unknown feature-matrix ID column: {id_column!r}.")
            if ids is not None:
                raise ValueError("Use either --id-column or --ids-file, not both.")
            ids = frame[id_column].tolist()
        selected = list(feature_columns)
        if not selected:
            selected = [column for column in frame.columns if column != id_column]
        missing = [column for column in selected if column not in frame.columns]
        if missing:
            raise ValueError(f"Unknown feature columns: {missing!r}.")
        return FeatureMatrix(
            frame[selected],
            observation_ids=ids,
            feature_names=selected,
        )

    if suffix == ".npy":
        return FeatureMatrix(np.load(source), observation_ids=ids)
    if suffix == ".npz":
        try:
            matrix = sparse.load_npz(source)
        except Exception as exc:
            raise ValueError(
                "Ruddy CLI expects .npz feature inputs to be SciPy sparse matrices."
            ) from exc
        return FeatureMatrix(matrix, observation_ids=ids)
    raise ValueError("Feature projection accepts .csv, .tsv, .txt, .npy, or sparse .npz inputs.")


def write_pca_result(result: PCAResult, output_dir: str | Path) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    result.scores.to_csv(target / "pca_scores.csv", index=False)
    result.loadings.to_csv(target / "pca_loadings.csv", index=False)
    result.variance.to_csv(target / "pca_variance.csv", index=False)
    if not result.exclusions.empty:
        result.exclusions.to_csv(target / "projection_exclusions.csv", index=False)
    payload = {
        "status": result.status.value,
        "reason": result.reason,
        "inferential_allowed": result.inferential_allowed,
        "preprocessing": result.preprocessing,
        "provenance": result.provenance.to_dict(),
    }
    (target / "pca_provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def write_projection_result(result: ProjectionResult, output_dir: str | Path) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    result.coordinates.to_csv(target / "projection_coordinates.csv", index=False)
    if not result.exclusions.empty:
        result.exclusions.to_csv(target / "projection_exclusions.csv", index=False)
    payload = {
        "status": result.status.value,
        "reason": result.reason,
        "method": result.method.value,
        "exploratory": result.exploratory,
        "inferential_allowed": result.inferential_allowed,
        "parameters": result.parameters,
        "preprocessing": result.preprocessing,
        "warnings": list(result.warnings),
        "provenance": result.provenance.to_dict(),
    }
    (target / "projection_provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def run_project(args) -> int:
    features = load_feature_matrix(
        args.input,
        id_column=args.id_column,
        feature_columns=tuple(args.feature_column or ()),
        ids_file=args.ids_file,
    )
    common = {
        "n_components": args.n_components,
        "scaling": args.scaling,
        "random_state": args.random_state,
    }
    if args.method == "pca":
        result = analyze_pca(features, **common)
        if args.output_dir:
            print(write_pca_result(result, args.output_dir))
        else:
            print(json.dumps({
                "method": "pca",
                "status": result.status.value,
                "n_scores": int(result.scores.shape[0]),
                "n_components": int(result.variance.shape[0]),
                "n_excluded": int(result.exclusions.shape[0]),
                "inferential_allowed": result.inferential_allowed,
            }, indent=2, sort_keys=True))
        return 0

    if args.method == "tsne":
        result = analyze_tsne(
            features,
            **common,
            metric=args.metric,
            perplexity=args.perplexity,
            max_iter=args.max_iter,
        )
    else:
        result = analyze_umap(
            features,
            **common,
            metric=args.metric,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
        )
    if args.output_dir:
        print(write_projection_result(result, args.output_dir))
    else:
        print(json.dumps({
            "method": result.method.value,
            "status": result.status.value,
            "n_coordinates": int(result.coordinates.shape[0]),
            "n_components": args.n_components,
            "n_excluded": int(result.exclusions.shape[0]),
            "exploratory": result.exploratory,
            "inferential_allowed": result.inferential_allowed,
        }, indent=2, sort_keys=True))
    return 0
