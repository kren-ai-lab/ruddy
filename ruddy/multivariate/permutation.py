"""Permutation-based group structure analysis for feature spaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from polars._typing import PolarsDataType
from scipy.stats import f_oneway
from sklearn.metrics import pairwise_distances

from ruddy.core.enums import AlignmentMode, ResultStatus
from ruddy.data import FeatureMatrix, TabularDataset
from ruddy.data.validation import AlignmentReport, align_annotations
from ruddy.projections.preprocessing import _exclusions_table
from ruddy.results import Advisory, AnalysisProvenance

SUMMARY_SCHEMA: dict[str, PolarsDataType] = {
    "analysis": pl.String,
    "factor": pl.String,
    "statistic": pl.String,
    "value": pl.Float64,
    "df_between": pl.Int64,
    "df_within": pl.Int64,
    "r_squared": pl.Float64,
    "p_value": pl.Float64,
    "n_permutations": pl.Int64,
    "status": pl.String,
    "reason": pl.String,
}

GROUP_SCHEMA: dict[str, PolarsDataType] = {
    "factor": pl.String,
    "level": pl.String,
    "n": pl.Int64,
    "mean_distance_to_centroid": pl.Float64,
}

CENTROID_SCHEMA_BASE: dict[str, PolarsDataType] = {
    "source_row_index": pl.Int64,
    "factor": pl.String,
    "level": pl.String,
    "distance_to_centroid": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}

CENTROID_COLUMNS: tuple[str, ...] = (
    "source_row_index",
    "observation_id",
    "factor",
    "level",
    "distance_to_centroid",
    "status",
    "reason",
)


@dataclass(frozen=True, slots=True)
class PermutationGroupResult:
    """PERMANOVA + PERMDISP result for one grouping factor."""

    status: ResultStatus
    reason: str | None
    summary: pl.DataFrame
    groups: pl.DataFrame
    distances_to_centroid: pl.DataFrame
    exclusions: pl.DataFrame
    alignment: AlignmentReport
    advisories: tuple[Advisory, ...]
    provenance: AnalysisProvenance


def _feature_valid_rows(features: FeatureMatrix) -> np.ndarray:
    if features.is_sparse:
        matrix = features.to_sparse().tocsr()  # pyrefly: ignore[missing-attribute]
        valid = np.ones(matrix.shape[0], dtype=bool)
        for row in range(matrix.shape[0]):
            values = matrix.data[matrix.indptr[row] : matrix.indptr[row + 1]]
            if values.size and not np.isfinite(values).all():
                valid[row] = False
        return valid
    return np.isfinite(features.to_array()).all(axis=1)


def _aligned_factor(
    features: FeatureMatrix,
    dataset: TabularDataset,
    factor: str,
    alignment: AlignmentMode | str,
) -> tuple[pl.Series, AlignmentReport]:
    if factor not in dataset.frame.columns:
        raise ValueError(f"Unknown grouping factor: {factor!r}.")
    annotations = dataset.frame.select(factor).with_columns(pl.Series("__id", dataset.observation_id_tuple))
    aligned, report = align_annotations(
        features.observation_id_tuple,
        annotations,
        id_column="__id",
        mode=alignment,
    )
    return aligned.get_column(factor), report


def _permanova_statistic(distance_matrix: np.ndarray, labels: np.ndarray) -> tuple[float, float, int, int]:
    n = len(labels)
    levels, inverse = np.unique(labels, return_inverse=True)
    g = len(levels)
    if g < 2 or n <= g:
        return np.nan, np.nan, g - 1, n - g
    sq = np.asarray(distance_matrix, dtype=float) ** 2
    ss_total = float(np.triu(sq, 1).sum() / n)
    ss_within = 0.0
    for idx in range(g):
        rows = np.flatnonzero(inverse == idx)
        ng = len(rows)
        if ng:
            block = sq[np.ix_(rows, rows)]
            ss_within += float(np.triu(block, 1).sum() / ng)
    ss_between = ss_total - ss_within
    df_between = g - 1
    df_within = n - g
    if ss_total <= 0.0 or df_between <= 0 or df_within <= 0 or ss_within <= 0.0:
        return np.nan, (ss_between / ss_total if ss_total > 0 else np.nan), df_between, df_within
    pseudo_f = (ss_between / df_between) / (ss_within / df_within)
    return float(pseudo_f), float(ss_between / ss_total), df_between, df_within


def _pcoa(distance_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d**2) @ j
    eigenvalues, eigenvectors = np.linalg.eigh(b)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > max(np.finfo(float).eps * max(1.0, abs(eigenvalues[0])), 1e-14)
    coords = eigenvectors[:, positive] * np.sqrt(eigenvalues[positive])
    return coords, eigenvalues, positive


def _distances_to_group_centroids(coords: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = np.empty(len(labels), dtype=float)
    for level in np.unique(labels):
        mask = labels == level
        centroid = coords[mask].mean(axis=0)
        out[mask] = np.linalg.norm(coords[mask] - centroid, axis=1)
    return out


def _dispersion_f(distances: np.ndarray, labels: np.ndarray) -> float:
    groups = [distances[labels == level] for level in np.unique(labels)]
    if len(groups) < 2 or any(len(group) < 2 for group in groups):
        return np.nan
    with np.errstate(all="ignore"):
        result = f_oneway(*groups)
    return float(result.statistic) if np.isfinite(result.statistic) else np.nan


def analyze_permutation_group_structure(
    features: FeatureMatrix,
    dataset: TabularDataset,
    *,
    factor: str,
    metric: str = "euclidean",
    n_permutations: int = 999,
    random_state: int = 0,
    min_group_n: int = 3,
    max_group_levels: int = 20,
    alignment: AlignmentMode | str = AlignmentMode.STRICT,
) -> PermutationGroupResult:
    """Run PERMANOVA and PERMDISP together on one feature-space grouping factor."""
    if n_permutations < 0:
        raise ValueError("n_permutations cannot be negative.")
    if min_group_n < 2:
        raise ValueError("min_group_n must be at least 2.")
    if max_group_levels < 2:
        raise ValueError("max_group_levels must be at least 2.")
    if not str(metric).strip():
        raise ValueError("metric cannot be empty.")

    group_series, report = _aligned_factor(features, dataset, factor, alignment)
    valid_features = _feature_valid_rows(features)
    valid_group = group_series.is_not_null().to_numpy()
    keep = valid_features & valid_group
    ids = features.observation_id_tuple
    excluded_rows = np.flatnonzero(~keep).astype(np.int64)
    reasons = [
        "non_finite_feature_row" if not valid_features[row] else "missing_group_value"
        for row in excluded_rows
    ]
    exclusions = _exclusions_table(
        ids,
        excluded_rows,
        stage="permutation_group_complete_case",
        reason=reasons,
    )

    labels = np.array(group_series.to_list())[keep]
    levels, counts = np.unique(labels, return_counts=True)
    provenance = AnalysisProvenance(
        analysis="permutation_group_structure",
        parameters={
            "factor": factor,
            "metric": metric,
            "n_permutations": int(n_permutations),
            "min_group_n": int(min_group_n),
            "max_group_levels": int(max_group_levels),
            "alignment": AlignmentMode(alignment).value,
        },
        input_summary={
            "n_source_observations": features.n_observations,
            "n_complete_case": int(keep.sum()),
            "n_excluded": exclusions.height,
            "n_features": features.n_features,
        },
        random_state=random_state,
    )
    id_dtype = pl.Series(ids).dtype if len(ids) > 0 else pl.String
    empty_summary = pl.DataFrame(schema=SUMMARY_SCHEMA)
    empty_groups = pl.DataFrame(schema=GROUP_SCHEMA)
    centroid_schema = {**CENTROID_SCHEMA_BASE, "observation_id": id_dtype}
    empty_dist = pl.DataFrame(schema={col: centroid_schema[col] for col in CENTROID_COLUMNS})

    if len(levels) < 2:
        return PermutationGroupResult(
            ResultStatus.DEGENERATE,
            "factor_has_fewer_than_two_levels",
            empty_summary,
            empty_groups,
            empty_dist,
            exclusions,
            report,
            (),
            provenance,
        )
    if len(levels) > max_group_levels:
        raise ValueError(f"Factor {factor!r} has {len(levels)} levels; maximum is {max_group_levels}.")
    if np.any(counts < min_group_n):
        groups = pl.DataFrame(
            {
                "factor": pl.Series("factor", [factor] * len(levels), dtype=pl.String),
                "level": pl.Series("level", [str(lvl) for lvl in levels], dtype=pl.String),
                "n": pl.Series("n", counts, dtype=pl.Int64),
                "mean_distance_to_centroid": pl.Series(
                    "mean_distance_to_centroid", [np.nan] * len(levels), dtype=pl.Float64
                ),
            },
            schema=GROUP_SCHEMA,
        )
        return PermutationGroupResult(
            ResultStatus.DEGENERATE,
            "group_too_small",
            empty_summary,
            groups,
            empty_dist,
            exclusions,
            report,
            (),
            provenance,
        )
    if keep.sum() <= len(levels):
        return PermutationGroupResult(
            ResultStatus.SKIPPED,
            "insufficient_residual_degrees_of_freedom",
            empty_summary,
            empty_groups,
            empty_dist,
            exclusions,
            report,
            (),
            provenance,
        )

    matrix = features.to_sparse()[keep] if features.is_sparse else features.to_array()[keep]  # pyrefly: ignore[bad-index]
    distance_matrix = pairwise_distances(matrix, metric=metric)
    if not np.isfinite(distance_matrix).all():
        return PermutationGroupResult(
            ResultStatus.DEGENERATE,
            "non_finite_distance_matrix",
            empty_summary,
            empty_groups,
            empty_dist,
            exclusions,
            report,
            (),
            provenance,
        )
    if np.allclose(distance_matrix, 0.0):
        return PermutationGroupResult(
            ResultStatus.DEGENERATE,
            "zero_distance_space",
            empty_summary,
            empty_groups,
            empty_dist,
            exclusions,
            report,
            (),
            provenance,
        )

    permanova_f, r_squared, df_between, df_within = _permanova_statistic(distance_matrix, labels)
    coords, eigenvalues, positive = _pcoa(distance_matrix)
    if coords.shape[1] == 0:
        return PermutationGroupResult(
            ResultStatus.DEGENERATE,
            "zero_rank_distance_space",
            empty_summary,
            empty_groups,
            empty_dist,
            exclusions,
            report,
            (),
            provenance,
        )
    centroid_distances = _distances_to_group_centroids(coords, labels)
    permdisp_f = _dispersion_f(centroid_distances, labels)

    rng = np.random.default_rng(random_state)
    perm_f = np.empty(n_permutations, dtype=float)
    perm_disp = np.empty(n_permutations, dtype=float)
    for idx in range(n_permutations):
        permuted = rng.permutation(labels)
        perm_f[idx] = _permanova_statistic(distance_matrix, permuted)[0]
        perm_dist = _distances_to_group_centroids(coords, permuted)
        perm_disp[idx] = _dispersion_f(perm_dist, permuted)

    def perm_p(observed: float, values: np.ndarray) -> float | None:
        if n_permutations == 0 or not np.isfinite(observed):
            return None
        finite = values[np.isfinite(values)]
        if not len(finite):
            return None
        return float((1 + np.sum(finite >= observed)) / (1 + len(finite)))

    summary = pl.DataFrame(
        [
            {
                "analysis": "permanova",
                "factor": factor,
                "statistic": "pseudo_f",
                "value": permanova_f,
                "df_between": df_between,
                "df_within": df_within,
                "r_squared": r_squared,
                "p_value": perm_p(permanova_f, perm_f),
                "n_permutations": n_permutations,
                "status": ResultStatus.OK.value
                if np.isfinite(permanova_f)
                else ResultStatus.DEGENERATE.value,
                "reason": None if np.isfinite(permanova_f) else "non_estimable_permanova",
            },
            {
                "analysis": "permdisp",
                "factor": factor,
                "statistic": "f_value",
                "value": permdisp_f,
                "df_between": df_between,
                "df_within": df_within,
                "r_squared": np.nan,
                "p_value": perm_p(permdisp_f, perm_disp),
                "n_permutations": n_permutations,
                "status": ResultStatus.OK.value if np.isfinite(permdisp_f) else ResultStatus.DEGENERATE.value,
                "reason": None if np.isfinite(permdisp_f) else "non_estimable_permdisp",
            },
        ],
        schema=SUMMARY_SCHEMA,
    )

    kept_indices = np.flatnonzero(keep)
    kept_ids = tuple(ids[i] for i in kept_indices)
    dist_rows = []
    group_rows = []
    for level in levels:
        mask = labels == level
        group_rows.append(
            {
                "factor": factor,
                "level": str(level),
                "n": int(mask.sum()),
                "mean_distance_to_centroid": float(centroid_distances[mask].mean()),
            }
        )
    for row, (obs_id, level, distance) in enumerate(zip(kept_ids, labels, centroid_distances, strict=True)):
        source_row = int(kept_indices[row])
        dist_rows.append(
            {
                "source_row_index": source_row,
                "observation_id": obs_id,
                "factor": factor,
                "level": str(level),
                "distance_to_centroid": float(distance),
                "status": ResultStatus.OK.value,
                "reason": None,
            }
        )
    groups = pl.DataFrame(group_rows, schema=GROUP_SCHEMA)
    if dist_rows:
        dist_frame = pl.DataFrame(dist_rows, schema_overrides=CENTROID_SCHEMA_BASE)
        distances_df = dist_frame.select(list(CENTROID_COLUMNS))
    else:
        distances_df = empty_dist

    advisories: list[Advisory] = []
    negative = eigenvalues[eigenvalues < 0]
    total_abs = float(np.abs(eigenvalues).sum())
    negative_fraction = float(np.abs(negative).sum() / total_abs) if total_abs else 0.0
    if negative_fraction > 1e-8:
        advisories.append(
            Advisory(
                code="negative_pcoa_eigenvalues",
                message="PERMDISP PCoA contained negative eigenvalues; distances use the positive-coordinate subspace.",
                context={"negative_eigenvalue_fraction": negative_fraction, "metric": metric},
            )
        )
    return PermutationGroupResult(
        status=ResultStatus.OK,
        reason=None,
        summary=summary,
        groups=groups,
        distances_to_centroid=distances_df,
        exclusions=exclusions,
        alignment=report,
        advisories=tuple(advisories),
        provenance=provenance,
    )


__all__ = ["PermutationGroupResult", "analyze_permutation_group_structure"]
