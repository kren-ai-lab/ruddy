"""Exploratory nonlinear projections for generic feature matrices."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from inspect import signature
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from sklearn.manifold import TSNE

from ruddy.core.enums import ProjectionMethod, ResultStatus, ScalingMethod
from ruddy.core.exceptions import OptionalDependencyError
from ruddy.core.frames import id_dtype
from ruddy.projections.preprocessing import prepare_features
from ruddy.results import AnalysisProvenance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ruddy.data import FeatureMatrix


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """Structured exploratory coordinates with explicit non-inferential semantics."""

    status: ResultStatus
    reason: str | None
    method: ProjectionMethod
    coordinates: pl.DataFrame
    exclusions: pl.DataFrame
    preprocessing: dict[str, Any]
    parameters: dict[str, Any]
    warnings: tuple[str, ...]
    provenance: AnalysisProvenance
    exploratory: bool = True
    inferential_allowed: bool = False


def _coordinate_table(
    coordinates: np.ndarray,
    *,
    observation_ids: Sequence[Any],
    source_row_indices: np.ndarray,
) -> pl.DataFrame:
    id_dt = id_dtype(observation_ids)
    component_names = [f"component_{index + 1}" for index in range(coordinates.shape[1])]
    columns: dict[str, pl.Series] = {
        "observation_id": pl.Series("observation_id", list(observation_ids), dtype=id_dt),
        "source_row_index": pl.Series("source_row_index", source_row_indices, dtype=pl.Int64),
    }
    values = np.asarray(coordinates, dtype=np.float64)
    for index, name in enumerate(component_names):
        columns[name] = pl.Series(name, values[:, index], dtype=pl.Float64)
    return pl.DataFrame(columns)


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
        msg = "t-SNE supports n_components of 2 or 3 in Ruddy."
        raise ValueError(msg)
    if perplexity <= 0:
        msg = "perplexity must be greater than zero."
        raise ValueError(msg)
    if max_iter < 250:
        msg = "max_iter must be at least 250."
        raise ValueError(msg)
    prepared = prepare_features(features, scaling=scaling, minimum_observations=3)
    if perplexity >= prepared.n_observations:
        msg = "t-SNE perplexity must be smaller than the number of finite input observations."
        raise ValueError(msg)
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
            "n_excluded_observations": prepared.exclusions.height,
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
        # Local import: umap-learn is an optional extra, so importing it at module
        # level would make ruddy.projections unimportable without it.
        import umap  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - environment dependent
        msg = "UMAP requires the optional 'umap-learn' dependency."
        raise OptionalDependencyError(msg) from exc

    if n_components < 2:
        msg = "UMAP n_components must be at least 2."
        raise ValueError(msg)
    if n_neighbors < 2:
        msg = "UMAP n_neighbors must be at least 2."
        raise ValueError(msg)
    if not 0.0 <= min_dist <= 1.0:
        msg = "UMAP min_dist must be between 0 and 1."
        raise ValueError(msg)
    prepared = prepare_features(features, scaling=scaling, minimum_observations=4)
    if n_neighbors >= prepared.n_observations:
        msg = "UMAP n_neighbors must be smaller than the number of finite input observations."
        raise ValueError(msg)
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
        msg = "UMAP returned coordinates with an unexpected shape."
        raise ValueError(msg)
    if not np.isfinite(values).all():
        msg = "UMAP returned non-finite coordinates."
        raise ValueError(msg)
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
            "n_excluded_observations": prepared.exclusions.height,
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
    msg = "Use analyze_pca() for PCA so its linear/inferential semantics remain explicit."
    raise ValueError(msg)
