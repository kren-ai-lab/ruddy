"""Principal component analysis for generic feature matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.decomposition import PCA

from ruddy.core.enums import ResultStatus, ScalingMethod
from ruddy.data import FeatureMatrix
from ruddy.projections.preprocessing import prepare_features
from ruddy.results import AnalysisProvenance

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

VARIANCE_SCHEMA: dict[str, PolarsDataType] = {
    "component": pl.String,
    "explained_variance": pl.Float64,
    "explained_variance_ratio": pl.Float64,
    "cumulative_explained_variance_ratio": pl.Float64,
    "singular_value": pl.Float64,
}


@dataclass(frozen=True, slots=True)
class PCAResult:
    """Structured PCA output with scores, loadings, variance, and provenance."""

    status: ResultStatus
    reason: str | None
    scores: pl.DataFrame
    loadings: pl.DataFrame
    variance: pl.DataFrame
    exclusions: pl.DataFrame
    preprocessing: dict[str, Any]
    provenance: AnalysisProvenance

    @property
    def inferential_allowed(self) -> bool:
        """PCA scores are explicit linear derived variables."""
        return self.status is ResultStatus.OK

    def to_feature_matrix(self) -> FeatureMatrix:
        """Return PCA scores as an explicitly derived feature matrix."""
        if self.status is not ResultStatus.OK:
            msg = "Only a successful PCA result can be converted to FeatureMatrix."
            raise ValueError(msg)
        component_columns = [column for column in self.scores.columns if column.startswith("PC")]
        return FeatureMatrix(
            self.scores.select(component_columns),
            observation_ids=self.scores.get_column("observation_id").to_list(),
            feature_names=component_columns,
            provenance={
                "derived_from": "pca",
                "inferential_allowed": True,
                "analysis_provenance": self.provenance.to_dict(),
            },
        )


def _empty_result(
    *,
    reason: str,
    exclusions: pl.DataFrame,
    preprocessing: dict[str, Any],
    provenance: AnalysisProvenance,
    id_dtype: PolarsDataType = pl.String,
) -> PCAResult:
    scores = pl.DataFrame(
        schema={
            "observation_id": id_dtype,
            "source_row_index": pl.Int64,
        }
    )
    loadings = pl.DataFrame(schema={"feature": pl.String})
    variance = pl.DataFrame(schema=VARIANCE_SCHEMA)
    return PCAResult(
        status=ResultStatus.SKIPPED,
        reason=reason,
        scores=scores,
        loadings=loadings,
        variance=variance,
        exclusions=exclusions,
        preprocessing=preprocessing,
        provenance=provenance,
    )


def analyze_pca(
    features: FeatureMatrix,
    *,
    n_components: int = 2,
    scaling: ScalingMethod | str = ScalingMethod.NONE,
    random_state: int = 0,
) -> PCAResult:
    """Run PCA with explicit scaling, row exclusion, and component provenance."""
    if n_components < 1:
        msg = "n_components must be at least 1."
        raise ValueError(msg)
    prepared = prepare_features(features, scaling=scaling, minimum_observations=2)
    scale = scaling if isinstance(scaling, ScalingMethod) else ScalingMethod(scaling)
    provenance = AnalysisProvenance(
        analysis="pca",
        parameters={
            "n_components_requested": int(n_components),
            "scaling": scale.value,
            "solver": "auto",
            "row_exclusion_policy": "exclude_rows_with_any_non_finite_feature",
            "inferential_allowed": True,
        },
        input_summary={
            "n_observations": features.n_observations,
            "n_features": features.n_features,
            "n_input_observations": prepared.n_observations,
            "n_excluded_observations": prepared.exclusions.height,
            "source_storage": "sparse" if features.is_sparse else "dense",
        },
        random_state=int(random_state),
    )

    if sparse.issparse(prepared.matrix):
        msg = (
            "PCA currently requires dense input because centered PCA semantics are preserved; "
            "Ruddy will not silently densify sparse matrices."
        )
        raise ValueError(msg)

    matrix = np.asarray(prepared.matrix, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    rank = int(np.linalg.matrix_rank(centered))
    maximum = min(matrix.shape[0], matrix.shape[1], rank)
    id_dtype = pl.Series(prepared.observation_ids).dtype if len(prepared.observation_ids) > 0 else pl.String
    if maximum < 1:
        return _empty_result(
            reason="zero_rank_matrix",
            exclusions=prepared.exclusions,
            preprocessing=prepared.metadata,
            provenance=provenance,
            id_dtype=id_dtype,
        )
    if n_components > maximum:
        msg = (
            f"PCA requested {n_components} components but at most {maximum} non-zero "
            f"components are available after centering for input shape {matrix.shape}."
        )
        raise ValueError(msg)

    estimator = PCA(n_components=int(n_components), random_state=int(random_state))
    score_matrix = np.asarray(estimator.fit_transform(matrix), dtype=np.float64)
    component_names = [f"PC{index + 1}" for index in range(score_matrix.shape[1])]
    scores_cols: dict[str, pl.Series] = {
        "observation_id": pl.Series("observation_id", list(prepared.observation_ids), dtype=id_dtype),
        "source_row_index": pl.Series("source_row_index", prepared.source_row_indices, dtype=pl.Int64),
    }
    for index, name in enumerate(component_names):
        scores_cols[name] = pl.Series(name, score_matrix[:, index], dtype=pl.Float64)
    scores = pl.DataFrame(scores_cols)

    loadings_cols: dict[str, pl.Series] = {
        "feature": pl.Series("feature", list(prepared.feature_names), dtype=pl.String),
    }
    for index, name in enumerate(component_names):
        loadings_cols[name] = pl.Series(name, estimator.components_[index, :], dtype=pl.Float64)
    loadings = pl.DataFrame(loadings_cols)

    ratios = np.asarray(estimator.explained_variance_ratio_, dtype=np.float64)
    variance = pl.DataFrame(
        {
            "component": pl.Series("component", component_names, dtype=pl.String),
            "explained_variance": pl.Series(
                "explained_variance", estimator.explained_variance_.astype(float), dtype=pl.Float64
            ),
            "explained_variance_ratio": pl.Series("explained_variance_ratio", ratios, dtype=pl.Float64),
            "cumulative_explained_variance_ratio": pl.Series(
                "cumulative_explained_variance_ratio", np.cumsum(ratios), dtype=pl.Float64
            ),
            "singular_value": pl.Series(
                "singular_value", estimator.singular_values_.astype(float), dtype=pl.Float64
            ),
        },
        schema=VARIANCE_SCHEMA,
    )
    return PCAResult(
        status=ResultStatus.OK,
        reason=None,
        scores=scores,
        loadings=loadings,
        variance=variance,
        exclusions=prepared.exclusions,
        preprocessing=prepared.metadata,
        provenance=provenance,
    )
