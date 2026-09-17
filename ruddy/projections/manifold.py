"""Exploratory nonlinear projections for generic feature matrices."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from inspect import signature
from typing import Any

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from ruddy.core.enums import ProjectionMethod, ResultStatus, ScalingMethod
from ruddy.core.exceptions import OptionalDependencyError
from ruddy.data import FeatureMatrix
from ruddy.projections.preprocessing import prepare_features
from ruddy.results import AnalysisProvenance


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """Structured exploratory coordinates with explicit non-inferential semantics."""

    status: ResultStatus
    reason: str | None
    method: ProjectionMethod
    coordinates: pd.DataFrame
    exclusions: pd.DataFrame
    preprocessing: dict[str, Any]
    parameters: dict[str, Any]
    warnings: tuple[str, ...]
    provenance: AnalysisProvenance
    exploratory: bool = True
    inferential_allowed: bool = False


def _coordinate_table(
    coordinates: np.ndarray,
    *,
    observation_ids: pd.Index,
    source_row_indices: np.ndarray,
) -> pd.DataFrame:
    columns = [f"component_{index + 1}" for index in range(coordinates.shape[1])]
    table = pd.DataFrame(np.asarray(coordinates, dtype=np.float64), columns=pd.Index(columns))
    table.insert(0, "observation_id", observation_ids.to_list())
    table.insert(0, "source_row_index", source_row_indices)
    return table


def analyze_tsne(
    features: FeatureMatrix,
    *,
    n_components: int = 2,
    scaling: ScalingMethod | str = ScalingMethod.NONE,
    metric: str = "euclidean",
    perplexity: float = 30.0,
    random_state: int = 0,
    max_iter: int = 1000,
) -> ProjectionResult:
    """Run deterministic t-SNE under a fixed random seed."""
    if n_components not in (2, 3):
        raise ValueError("t-SNE supports n_components of 2 or 3 in Ruddy.")
    if perplexity <= 0:
        raise ValueError("perplexity must be greater than zero.")
    if max_iter < 250:
        raise ValueError("max_iter must be at least 250.")
    prepared = prepare_features(features, scaling=scaling, minimum_observations=3)
    if perplexity >= prepared.n_observations:
        raise ValueError("t-SNE perplexity must be smaller than the number of finite input observations.")
    scale = scaling if isinstance(scaling, ScalingMethod) else ScalingMethod(scaling)
    parameters = {
        "n_components": int(n_components),
        "scaling": scale.value,
        "metric": str(metric),
        "perplexity": float(perplexity),
        "max_iter": int(max_iter),
        "random_state": int(random_state),
        "exploratory": True,
        "inferential_allowed": False,
    }
    tsne_kwargs = {
        "n_components": int(n_components),
        "metric": str(metric),
        "perplexity": float(perplexity),
        "random_state": int(random_state),
        "init": "pca" if not prepared.is_sparse else "random",
        "learning_rate": "auto",
    }
    iteration_parameter = "max_iter" if "max_iter" in signature(TSNE).parameters else "n_iter"
    tsne_kwargs[iteration_parameter] = int(max_iter)
    estimator = TSNE(**tsne_kwargs)  # pyrefly: ignore[bad-argument-type]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = estimator.fit_transform(prepared.matrix)
    coordinates = _coordinate_table(
        np.asarray(values, dtype=np.float64),
        observation_ids=prepared.observation_ids,
        source_row_indices=prepared.source_row_indices,
    )
    provenance = AnalysisProvenance(
        analysis="tsne",
        parameters=parameters,
        input_summary={
            "n_observations": features.n_observations,
            "n_features": features.n_features,
            "n_input_observations": prepared.n_observations,
            "n_excluded_observations": int(prepared.exclusions.shape[0]),
            "source_storage": "sparse" if features.is_sparse else "dense",
        },
        random_state=int(random_state),
    )
    return ProjectionResult(
        status=ResultStatus.OK,
        reason=None,
        method=ProjectionMethod.TSNE,
        coordinates=coordinates,
        exclusions=prepared.exclusions,
        preprocessing=prepared.metadata,
        parameters=parameters,
        warnings=tuple(str(item.message) for item in caught),
        provenance=provenance,
    )


def analyze_umap(
    features: FeatureMatrix,
    *,
    n_components: int = 2,
    scaling: ScalingMethod | str = ScalingMethod.NONE,
    metric: str = "euclidean",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
) -> ProjectionResult:
    """Run UMAP when the optional ``umap-learn`` dependency is installed."""
    try:
        import umap
    except Exception as exc:  # pragma: no cover - environment dependent
        raise OptionalDependencyError("UMAP requires the optional 'umap-learn' dependency.") from exc

    if n_components < 2:
        raise ValueError("UMAP n_components must be at least 2.")
    if n_neighbors < 2:
        raise ValueError("UMAP n_neighbors must be at least 2.")
    if not 0.0 <= min_dist <= 1.0:
        raise ValueError("UMAP min_dist must be between 0 and 1.")
    prepared = prepare_features(features, scaling=scaling, minimum_observations=4)
    if n_neighbors >= prepared.n_observations:
        raise ValueError("UMAP n_neighbors must be smaller than the number of finite input observations.")
    scale = scaling if isinstance(scaling, ScalingMethod) else ScalingMethod(scaling)
    parameters = {
        "n_components": int(n_components),
        "scaling": scale.value,
        "metric": str(metric),
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "random_state": int(random_state),
        "exploratory": True,
        "inferential_allowed": False,
    }
    estimator = umap.UMAP(
        n_components=int(n_components),
        metric=str(metric),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        random_state=int(random_state),
        transform_seed=int(random_state),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = np.asarray(estimator.fit_transform(prepared.matrix), dtype=np.float64)
    if values.shape != (prepared.n_observations, int(n_components)):
        raise ValueError("UMAP returned coordinates with an unexpected shape.")
    if not np.isfinite(values).all():
        raise ValueError("UMAP returned non-finite coordinates.")
    coordinates = _coordinate_table(
        values,
        observation_ids=prepared.observation_ids,
        source_row_indices=prepared.source_row_indices,
    )
    provenance = AnalysisProvenance(
        analysis="umap",
        parameters=parameters,
        input_summary={
            "n_observations": features.n_observations,
            "n_features": features.n_features,
            "n_input_observations": prepared.n_observations,
            "n_excluded_observations": int(prepared.exclusions.shape[0]),
            "source_storage": "sparse" if features.is_sparse else "dense",
        },
        random_state=int(random_state),
    )
    return ProjectionResult(
        status=ResultStatus.OK,
        reason=None,
        method=ProjectionMethod.UMAP,
        coordinates=coordinates,
        exclusions=prepared.exclusions,
        preprocessing=prepared.metadata,
        parameters=parameters,
        warnings=tuple(str(item.message) for item in caught),
        provenance=provenance,
    )


def analyze_projection(
    features: FeatureMatrix,
    *,
    method: ProjectionMethod | str,
    **kwargs: Any,
) -> ProjectionResult:
    """Dispatch one explicitly exploratory nonlinear projection."""
    normalized = method if isinstance(method, ProjectionMethod) else ProjectionMethod(method)
    if normalized is ProjectionMethod.UMAP:
        return analyze_umap(features, **kwargs)
    if normalized is ProjectionMethod.TSNE:
        return analyze_tsne(features, **kwargs)
    raise ValueError("Use analyze_pca() for PCA so its linear/inferential semantics remain explicit.")
