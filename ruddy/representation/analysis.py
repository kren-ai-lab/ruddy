"""Comparison of aligned numerical representation spaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist, squareform
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ruddy.core.enums import AlignmentMode, ResultStatus, ScalingMethod
from ruddy.core.exceptions import AlignmentError
from ruddy.data import FeatureMatrix
from ruddy.data.validation import AlignmentReport
from ruddy.results import Advisory, AnalysisProvenance


@dataclass(frozen=True, slots=True)
class AlignedRepresentationPair:
    x: np.ndarray
    y: np.ndarray
    observation_ids: pd.Index
    source_rows_x: np.ndarray
    source_rows_y: np.ndarray
    exclusions: pd.DataFrame
    alignment: AlignmentReport


@dataclass(frozen=True, slots=True)
class CCAResult:
    status: ResultStatus
    reason: str | None
    correlations: pd.DataFrame
    x_weights: pd.DataFrame
    y_weights: pd.DataFrame
    x_loadings: pd.DataFrame
    y_loadings: pd.DataFrame
    x_scores: pd.DataFrame
    y_scores: pd.DataFrame
    advisories: tuple[Advisory, ...]
    provenance: AnalysisProvenance


@dataclass(frozen=True, slots=True)
class RepresentationComparisonResult:
    cca: CCAResult
    cka: pd.DataFrame
    procrustes: pd.DataFrame
    distance_similarity: pd.DataFrame
    mantel: pd.DataFrame
    exclusions: pd.DataFrame
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
    x_set, y_set = set(x_ids.tolist()), set(y_ids.tolist())
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
    excluded_ids = np.asarray(common, dtype=object)[~finite]
    exclusions = pd.DataFrame(
        {
            "observation_id": excluded_ids.tolist(),
            "source_row_x": x_rows[~finite],
            "source_row_y": y_rows[~finite],
            "stage": "representation_complete_case",
            "reason": "non_finite_feature_row",
        }
    )
    if int(finite.sum()) < 3:
        raise ValueError("Representation comparison requires at least 3 jointly finite aligned observations.")
    return AlignedRepresentationPair(
        x=np.asarray(xa[finite], dtype=float),
        y=np.asarray(ya[finite], dtype=float),
        observation_ids=pd.Index(np.asarray(common, dtype=object)[finite]),
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
    x = _scale(pair.x, scaling)
    y = _scale(pair.y, scaling)
    rank_x, rank_y = np.linalg.matrix_rank(x), np.linalg.matrix_rank(y)
    max_components = min(rank_x, rank_y, x.shape[0] - 1, x.shape[1], y.shape[1])
    if max_components < 1:
        return CCAResult(
            ResultStatus.DEGENERATE,
            "zero_rank_representation",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            (),
            provenance,
        )
    if n_components < 1 or n_components > max_components:
        return CCAResult(
            ResultStatus.SKIPPED,
            "cca_components_exceed_effective_rank",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            (),
            provenance,
        )
    model = CCA(n_components=n_components, scale=False, max_iter=max_iter, tol=tol)
    try:
        xs, ys = model.fit_transform(x, y)
    except Exception as exc:
        advisory = Advisory(code="cca_fit_failed", message=str(exc))
        return CCAResult(
            ResultStatus.DEGENERATE,
            "cca_fit_failed",
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            (advisory,),
            provenance,
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
    cols = [f"CC{i + 1}" for i in range(n_components)]
    x_weights = pd.DataFrame(model.x_weights_, index=x_names, columns=cols).reset_index(names="feature")
    y_weights = pd.DataFrame(model.y_weights_, index=y_names, columns=cols).reset_index(names="feature")
    x_loadings = pd.DataFrame(model.x_loadings_, index=x_names, columns=cols).reset_index(names="feature")
    y_loadings = pd.DataFrame(model.y_loadings_, index=y_names, columns=cols).reset_index(names="feature")
    x_scores = pd.DataFrame(xs, columns=cols)
    x_scores.insert(0, "observation_id", pair.observation_ids.to_list())
    y_scores = pd.DataFrame(ys, columns=cols)
    y_scores.insert(0, "observation_id", pair.observation_ids.to_list())
    return CCAResult(
        ResultStatus.OK,
        None,
        pd.DataFrame(rows),
        x_weights,
        y_weights,
        x_loadings,
        y_loadings,
        x_scores,
        y_scores,
        (),
        provenance,
    )


def _procrustes(pair: AlignedRepresentationPair) -> pd.DataFrame:
    if pair.x.shape[1] != pair.y.shape[1]:
        return pd.DataFrame(
            [
                {
                    "disparity": np.nan,
                    "similarity": np.nan,
                    "n_dimensions": np.nan,
                    "status": "skipped",
                    "reason": "procrustes_requires_equal_dimensions",
                }
            ]
        )
    x = pair.x - pair.x.mean(axis=0, keepdims=True)
    y = pair.y - pair.y.mean(axis=0, keepdims=True)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx <= np.finfo(float).eps or ny <= np.finfo(float).eps:
        return pd.DataFrame(
            [
                {
                    "disparity": np.nan,
                    "similarity": np.nan,
                    "n_dimensions": pair.x.shape[1],
                    "status": "degenerate",
                    "reason": "zero_variance_representation",
                }
            ]
        )
    x, y = x / nx, y / ny
    rotation, scale = orthogonal_procrustes(y, x)
    aligned = y @ rotation * scale
    disparity = float(np.sum((x - aligned) ** 2))
    return pd.DataFrame(
        [
            {
                "disparity": disparity,
                "similarity": float(max(0.0, 1.0 - disparity)),
                "n_dimensions": pair.x.shape[1],
                "status": "ok",
                "reason": None,
            }
        ]
    )


def _distance_vectors(pair: AlignedRepresentationPair, metric: str) -> tuple[np.ndarray, np.ndarray]:
    return pdist(pair.x, metric=metric), pdist(pair.y, metric=metric)


def _distance_similarity(pair: AlignedRepresentationPair, metric: str, method: str) -> pd.DataFrame:
    dx, dy = _distance_vectors(pair, metric)
    if method == "pearson":
        stat, p = stats.pearsonr(dx, dy)
    elif method == "spearman":
        stat, p = stats.spearmanr(dx, dy)
    else:
        raise ValueError("distance_similarity_method must be 'pearson' or 'spearman'.")
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "method": method,
                "coefficient": float(stat),
                "p_value": float(p),
                "n_pairs": len(dx),
                "status": "ok",
                "reason": None,
            }
        ]
    )


def _mantel(
    pair: AlignedRepresentationPair, metric: str, permutations: int, random_state: int
) -> pd.DataFrame:
    if permutations < 0:
        raise ValueError("mantel_permutations must be non-negative.")
    dx = squareform(pdist(pair.x, metric=metric))
    dy = squareform(pdist(pair.y, metric=metric))
    tri = np.triu_indices(dx.shape[0], k=1)
    observed = float(stats.pearsonr(dx[tri], dy[tri]).statistic)
    if permutations == 0:
        p = np.nan
    else:
        rng = np.random.default_rng(random_state)
        exceed = 0
        for _ in range(permutations):
            perm = rng.permutation(dx.shape[0])
            value = float(stats.pearsonr(dx[tri], dy[np.ix_(perm, perm)][tri]).statistic)
            exceed += abs(value) >= abs(observed)
        p = (exceed + 1.0) / (permutations + 1.0)
    return pd.DataFrame(
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
        ]
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
        cka = pd.DataFrame([{"kernel": "linear", "cka": cka_value, "status": "ok", "reason": None}])
    except ValueError as exc:
        cka = pd.DataFrame([{"kernel": "linear", "cka": np.nan, "status": "degenerate", "reason": str(exc)}])
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
            "excluded_observations": len(pair.exclusions),
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
