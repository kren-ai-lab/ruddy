"""Comparison of aligned numerical representation spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from scipy import stats
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist, squareform
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ruddy.core.enums import AlignmentMode, ResultStatus, ScalingMethod
from ruddy.core.exceptions import AlignmentError
from ruddy.data.validation import AlignmentReport
from ruddy.results import Advisory, AnalysisProvenance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polars._typing import PolarsDataType

    from ruddy.core.types import ObservationID
    from ruddy.data import FeatureMatrix

REPRESENTATION_EXCLUSIONS_SCHEMA_BASE: dict[str, PolarsDataType] = {
    "source_row_x": pl.Int64,
    "source_row_y": pl.Int64,
    "stage": pl.String,
    "reason": pl.String,
}

REPRESENTATION_EXCLUSION_COLUMNS: tuple[str, ...] = (
    "observation_id",
    "source_row_x",
    "source_row_y",
    "stage",
    "reason",
)

CCA_CORRELATION_SCHEMA: dict[str, PolarsDataType] = {
    "component": pl.Int64,
    "canonical_correlation": pl.Float64,
    "shared_variance": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}

CKA_SCHEMA: dict[str, PolarsDataType] = {
    "kernel": pl.String,
    "cka": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}

PROCRUSTES_SCHEMA: dict[str, PolarsDataType] = {
    "disparity": pl.Float64,
    "similarity": pl.Float64,
    "n_dimensions": pl.Int64,
    "status": pl.String,
    "reason": pl.String,
}

DISTANCE_SIMILARITY_SCHEMA: dict[str, PolarsDataType] = {
    "metric": pl.String,
    "method": pl.String,
    "coefficient": pl.Float64,
    "p_value": pl.Float64,
    "n_pairs": pl.Int64,
    "status": pl.String,
    "reason": pl.String,
}

MANTEL_SCHEMA: dict[str, PolarsDataType] = {
    "metric": pl.String,
    "correlation": pl.Float64,
    "p_value": pl.Float64,
    "permutations": pl.Int64,
    "alternative": pl.String,
    "status": pl.String,
    "reason": pl.String,
}


@dataclass(frozen=True, slots=True)
class AlignedRepresentationPair:
    x: np.ndarray
    y: np.ndarray
    observation_ids: tuple[ObservationID, ...]
    source_rows_x: np.ndarray
    source_rows_y: np.ndarray
    exclusions: pl.DataFrame
    alignment: AlignmentReport


@dataclass(frozen=True, slots=True)
class CCAResult:
    status: ResultStatus
    reason: str | None
    correlations: pl.DataFrame
    x_weights: pl.DataFrame
    y_weights: pl.DataFrame
    x_loadings: pl.DataFrame
    y_loadings: pl.DataFrame
    x_scores: pl.DataFrame
    y_scores: pl.DataFrame
    advisories: tuple[Advisory, ...]
    provenance: AnalysisProvenance


@dataclass(frozen=True, slots=True)
class RepresentationComparisonResult:
    cca: CCAResult
    cka: pl.DataFrame
    procrustes: pl.DataFrame
    distance_similarity: pl.DataFrame
    mantel: pl.DataFrame
    exclusions: pl.DataFrame
    alignment: AlignmentReport
    provenance: AnalysisProvenance


def _scale(array: np.ndarray, method: ScalingMethod | str) -> np.ndarray:
    method = method if isinstance(method, ScalingMethod) else ScalingMethod(method)
    if method is ScalingMethod.NONE:
        return array.copy()
    if method is ScalingMethod.STANDARD:
        return np.asarray(StandardScaler().fit_transform(array), dtype=float)
    if method is ScalingMethod.ROBUST:
        return np.asarray(RobustScaler().fit_transform(array), dtype=float)
    return np.asarray(MinMaxScaler().fit_transform(array), dtype=float)


def _exclusions_table(
    ids: Sequence[Any],
    excluded_indices: Sequence[int] | np.ndarray,
    source_rows_x: Sequence[int] | np.ndarray,
    source_rows_y: Sequence[int] | np.ndarray,
    *,
    stage: str = "representation_complete_case",
    reason: str = "non_finite_feature_row",
) -> pl.DataFrame:
    id_dtype = pl.Series(ids).dtype if len(ids) > 0 else pl.String
    if len(excluded_indices) == 0:
        schema = {"observation_id": id_dtype, **REPRESENTATION_EXCLUSIONS_SCHEMA_BASE}
        return pl.DataFrame(schema={col: schema[col] for col in REPRESENTATION_EXCLUSION_COLUMNS})

    indices = [int(i) for i in excluded_indices]
    obs_ids = [ids[i] for i in indices]
    sx = [int(source_rows_x[i]) for i in indices]
    sy = [int(source_rows_y[i]) for i in indices]
    n = len(indices)
    rows = [
        {
            "observation_id": obs_ids[j],
            "source_row_x": sx[j],
            "source_row_y": sy[j],
            "stage": stage,
            "reason": reason,
        }
        for j in range(n)
    ]
    frame = pl.DataFrame(
        rows,
        schema_overrides={"observation_id": id_dtype, **REPRESENTATION_EXCLUSIONS_SCHEMA_BASE},
    )
    return frame.select(list(REPRESENTATION_EXCLUSION_COLUMNS))


def align_feature_matrices(
    x: FeatureMatrix,
    y: FeatureMatrix,
    *,
    mode: AlignmentMode | str = AlignmentMode.STRICT,
) -> AlignedRepresentationPair:
    """Align two dense feature matrices by observation identity and finite rows."""
    if x.is_sparse or y.is_sparse:
        raise ValueError(
            "Representation comparison currently requires dense inputs; Ruddy will not silently densify sparse matrices."
        )
    mode = mode if isinstance(mode, AlignmentMode) else AlignmentMode(mode)
    x_ids, y_ids = x.observation_ids, y.observation_ids
    x_set, y_set = set(x_ids), set(y_ids)
    missing = tuple(value for value in x_ids if value not in y_set)
    unmatched = tuple(value for value in y_ids if value not in x_set)
    common = [value for value in x_ids if value in y_set]
    report = AlignmentReport(
        mode=mode,
        base_count=len(x_ids),
        annotation_count=len(y_ids),
        covered_count=len(common),
        missing_ids=missing,
        unmatched_ids=unmatched,
    )
    if mode is AlignmentMode.STRICT and not report.complete:
        raise AlignmentError(
            "Strict representation alignment requires exact one-to-one ID coverage; "
            f"missing={list(missing)!r}, unmatched={list(unmatched)!r}."
        )
    if len(common) < 3:
        raise ValueError("Representation comparison requires at least 3 aligned observations.")

    x_pos = {value: i for i, value in enumerate(x_ids)}
    y_pos = {value: i for i, value in enumerate(y_ids)}
    x_rows = np.asarray([x_pos[v] for v in common], dtype=int)
    y_rows = np.asarray([y_pos[v] for v in common], dtype=int)
    xa, ya = x.to_array()[x_rows], y.to_array()[y_rows]
    finite = np.isfinite(xa).all(axis=1) & np.isfinite(ya).all(axis=1)
    excluded_indices = np.flatnonzero(~finite)
    exclusions = _exclusions_table(common, excluded_indices, x_rows, y_rows)
    if int(finite.sum()) < 3:
        raise ValueError("Representation comparison requires at least 3 jointly finite aligned observations.")
    finite_indices = np.flatnonzero(finite)
    final_ids = tuple(common[i] for i in finite_indices)
    return AlignedRepresentationPair(
        x=np.asarray(xa[finite], dtype=float),
        y=np.asarray(ya[finite], dtype=float),
        observation_ids=final_ids,
        source_rows_x=x_rows[finite],
        source_rows_y=y_rows[finite],
        exclusions=exclusions,
        alignment=report,
    )


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear centered-kernel alignment for two observation-aligned matrices."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("CKA requires two 2D matrices with the same number of observations.")
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(xc.T @ yc, ord="fro") ** 2
    xx = np.linalg.norm(xc.T @ xc, ord="fro")
    yy = np.linalg.norm(yc.T @ yc, ord="fro")
    denom = xx * yy
    if denom <= np.finfo(float).eps:
        raise ValueError("CKA is undefined for a zero-variance representation.")
    return float(np.clip(cross / denom, 0.0, 1.0))


def _empty_cca_result(
    status: ResultStatus,
    reason: str,
    provenance: AnalysisProvenance,
    id_dtype: PolarsDataType = pl.String,
    advisories: tuple[Advisory, ...] = (),
) -> CCAResult:
    empty_feature_df = pl.DataFrame(schema={"feature": pl.String})
    empty_score_df = pl.DataFrame(schema={"observation_id": id_dtype})
    return CCAResult(
        status=status,
        reason=reason,
        correlations=pl.DataFrame(schema=CCA_CORRELATION_SCHEMA),
        x_weights=empty_feature_df,
        y_weights=empty_feature_df,
        x_loadings=empty_feature_df,
        y_loadings=empty_feature_df,
        x_scores=empty_score_df,
        y_scores=empty_score_df,
        advisories=advisories,
        provenance=provenance,
    )


def _cca_result(
    pair: AlignedRepresentationPair,
    *,
    x_names: tuple[str, ...],
    y_names: tuple[str, ...],
    n_components: int,
    scaling: ScalingMethod | str,
    max_iter: int,
    tol: float,
) -> CCAResult:
    provenance = AnalysisProvenance(
        analysis="cca",
        parameters={"n_components": n_components, "scaling": str(scaling), "max_iter": max_iter, "tol": tol},
        input_summary={
            "n_observations": len(pair.observation_ids),
            "x_features": pair.x.shape[1],
            "y_features": pair.y.shape[1],
        },
    )
    id_dtype = pl.Series(pair.observation_ids).dtype if len(pair.observation_ids) > 0 else pl.String
    x = _scale(pair.x, scaling)
    y = _scale(pair.y, scaling)
    rank_x, rank_y = np.linalg.matrix_rank(x), np.linalg.matrix_rank(y)
    max_components = min(rank_x, rank_y, x.shape[0] - 1, x.shape[1], y.shape[1])
    if max_components < 1:
        return _empty_cca_result(
            ResultStatus.DEGENERATE,
            "zero_rank_representation",
            provenance,
            id_dtype=id_dtype,
        )
    if n_components < 1 or n_components > max_components:
        return _empty_cca_result(
            ResultStatus.SKIPPED,
            "cca_components_exceed_effective_rank",
            provenance,
            id_dtype=id_dtype,
        )
    model = CCA(n_components=n_components, scale=False, max_iter=max_iter, tol=tol)
    try:
        xs, ys = model.fit_transform(x, y)
    except Exception as exc:
        advisory = Advisory(code="cca_fit_failed", message=str(exc))
        return _empty_cca_result(
            ResultStatus.DEGENERATE,
            "cca_fit_failed",
            provenance,
            id_dtype=id_dtype,
            advisories=(advisory,),
        )
    rows = []
    for i in range(n_components):
        r = float(np.corrcoef(xs[:, i], ys[:, i])[0, 1])
        rows.append(
            {
                "component": i + 1,
                "canonical_correlation": r,
                "shared_variance": r * r,
                "status": "ok",
                "reason": None,
            }
        )
    correlations = pl.DataFrame(rows, schema=CCA_CORRELATION_SCHEMA)
    cols = [f"CC{i + 1}" for i in range(n_components)]
    x_weights = pl.DataFrame(
        {
            "feature": pl.Series("feature", list(x_names), dtype=pl.String),
            **{c: pl.Series(c, model.x_weights_[:, i], dtype=pl.Float64) for i, c in enumerate(cols)},
        }
    )
    y_weights = pl.DataFrame(
        {
            "feature": pl.Series("feature", list(y_names), dtype=pl.String),
            **{c: pl.Series(c, model.y_weights_[:, i], dtype=pl.Float64) for i, c in enumerate(cols)},
        }
    )
    x_loadings = pl.DataFrame(
        {
            "feature": pl.Series("feature", list(x_names), dtype=pl.String),
            **{c: pl.Series(c, model.x_loadings_[:, i], dtype=pl.Float64) for i, c in enumerate(cols)},
        }
    )
    y_loadings = pl.DataFrame(
        {
            "feature": pl.Series("feature", list(y_names), dtype=pl.String),
            **{c: pl.Series(c, model.y_loadings_[:, i], dtype=pl.Float64) for i, c in enumerate(cols)},
        }
    )
    x_scores = pl.DataFrame(
        {
            "observation_id": pl.Series("observation_id", list(pair.observation_ids), dtype=id_dtype),
            **{c: pl.Series(c, xs[:, i], dtype=pl.Float64) for i, c in enumerate(cols)},
        }
    )
    y_scores = pl.DataFrame(
        {
            "observation_id": pl.Series("observation_id", list(pair.observation_ids), dtype=id_dtype),
            **{c: pl.Series(c, ys[:, i], dtype=pl.Float64) for i, c in enumerate(cols)},
        }
    )
    return CCAResult(
        ResultStatus.OK,
        None,
        correlations,
        x_weights,
        y_weights,
        x_loadings,
        y_loadings,
        x_scores,
        y_scores,
        (),
        provenance,
    )


def _procrustes(pair: AlignedRepresentationPair) -> pl.DataFrame:
    if pair.x.shape[1] != pair.y.shape[1]:
        return pl.DataFrame(
            [
                {
                    "disparity": None,
                    "similarity": None,
                    "n_dimensions": None,
                    "status": "skipped",
                    "reason": "procrustes_requires_equal_dimensions",
                }
            ],
            schema=PROCRUSTES_SCHEMA,
        )
    x = pair.x - pair.x.mean(axis=0, keepdims=True)
    y = pair.y - pair.y.mean(axis=0, keepdims=True)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx <= np.finfo(float).eps or ny <= np.finfo(float).eps:
        return pl.DataFrame(
            [
                {
                    "disparity": None,
                    "similarity": None,
                    "n_dimensions": int(pair.x.shape[1]),
                    "status": "degenerate",
                    "reason": "zero_variance_representation",
                }
            ],
            schema=PROCRUSTES_SCHEMA,
        )
    x, y = x / nx, y / ny
    rotation, scale = orthogonal_procrustes(y, x)
    aligned = y @ rotation * scale
    disparity = float(np.sum((x - aligned) ** 2))
    return pl.DataFrame(
        [
            {
                "disparity": disparity,
                "similarity": float(max(0.0, 1.0 - disparity)),
                "n_dimensions": int(pair.x.shape[1]),
                "status": "ok",
                "reason": None,
            }
        ],
        schema=PROCRUSTES_SCHEMA,
    )


def _distance_vectors(pair: AlignedRepresentationPair, metric: str) -> tuple[np.ndarray, np.ndarray]:
    return pdist(pair.x, metric=metric), pdist(pair.y, metric=metric)  # pyrefly: ignore[no-matching-overload]


def _distance_similarity(pair: AlignedRepresentationPair, metric: str, method: str) -> pl.DataFrame:
    dx, dy = _distance_vectors(pair, metric)
    if method == "pearson":
        stat, p = stats.pearsonr(dx, dy)
    elif method == "spearman":
        stat, p = stats.spearmanr(dx, dy)
    else:
        raise ValueError("distance_similarity_method must be 'pearson' or 'spearman'.")
    return pl.DataFrame(
        [
            {
                "metric": metric,
                "method": method,
                "coefficient": float(stat) if np.isfinite(stat) else None,
                "p_value": float(p) if np.isfinite(p) else None,
                "n_pairs": len(dx),
                "status": "ok",
                "reason": None,
            }
        ],
        schema=DISTANCE_SIMILARITY_SCHEMA,
    )


def _mantel(
    pair: AlignedRepresentationPair, metric: str, permutations: int, random_state: int
) -> pl.DataFrame:
    if permutations < 0:
        raise ValueError("mantel_permutations must be non-negative.")
    dx = squareform(pdist(pair.x, metric=metric))  # pyrefly: ignore[no-matching-overload]
    dy = squareform(pdist(pair.y, metric=metric))  # pyrefly: ignore[no-matching-overload]
    tri = np.triu_indices(dx.shape[0], k=1)
    observed = float(stats.pearsonr(dx[tri], dy[tri]).statistic)
    if permutations == 0:
        p: float | None = None
    else:
        rng = np.random.default_rng(random_state)
        exceed = 0
        for _ in range(permutations):
            perm = rng.permutation(dx.shape[0])
            value = float(stats.pearsonr(dx[tri], dy[np.ix_(perm, perm)][tri]).statistic)
            exceed += abs(value) >= abs(observed)
        p = float((exceed + 1.0) / (permutations + 1.0))
    return pl.DataFrame(
        [
            {
                "metric": metric,
                "correlation": observed,
                "p_value": p,
                "permutations": permutations,
                "alternative": "two-sided",
                "status": "ok",
                "reason": None,
            }
        ],
        schema=MANTEL_SCHEMA,
    )


def analyze_representation_similarity(
    x: FeatureMatrix,
    y: FeatureMatrix,
    *,
    alignment: AlignmentMode | str = AlignmentMode.STRICT,
    cca_components: int = 2,
    cca_scaling: ScalingMethod | str = ScalingMethod.STANDARD,
    cca_max_iter: int = 1000,
    cca_tol: float = 1e-6,
    distance_metric: str = "euclidean",
    distance_similarity_method: str = "spearman",
    mantel_permutations: int = 999,
    random_state: int = 0,
) -> RepresentationComparisonResult:
    """Compare two numerical representation spaces for the same observations."""
    pair = align_feature_matrices(x, y, mode=alignment)
    cca = _cca_result(
        pair,
        x_names=x.feature_names,
        y_names=y.feature_names,
        n_components=cca_components,
        scaling=cca_scaling,
        max_iter=cca_max_iter,
        tol=cca_tol,
    )
    try:
        cka_value = linear_cka(pair.x, pair.y)
        cka = pl.DataFrame(
            [{"kernel": "linear", "cka": cka_value, "status": "ok", "reason": None}],
            schema=CKA_SCHEMA,
        )
    except ValueError as exc:
        cka = pl.DataFrame(
            [{"kernel": "linear", "cka": None, "status": "degenerate", "reason": str(exc)}],
            schema=CKA_SCHEMA,
        )
    procrustes = _procrustes(pair)
    distance_similarity = _distance_similarity(pair, distance_metric, distance_similarity_method)
    mantel = _mantel(pair, distance_metric, mantel_permutations, random_state)
    provenance = AnalysisProvenance(
        analysis="representation_similarity",
        parameters={
            "alignment": str(alignment),
            "cca_components": cca_components,
            "cca_scaling": str(cca_scaling),
            "distance_metric": distance_metric,
            "distance_similarity_method": distance_similarity_method,
            "mantel_permutations": mantel_permutations,
        },
        input_summary={
            "aligned_observations": len(pair.observation_ids),
            "x_features": pair.x.shape[1],
            "y_features": pair.y.shape[1],
            "excluded_observations": pair.exclusions.height,
        },
        random_state=random_state,
    )
    return RepresentationComparisonResult(
        cca, cka, procrustes, distance_similarity, mantel, pair.exclusions, pair.alignment, provenance
    )


__all__ = [
    "AlignedRepresentationPair",
    "CCAResult",
    "RepresentationComparisonResult",
    "align_feature_matrices",
    "analyze_representation_similarity",
    "linear_cka",
]
