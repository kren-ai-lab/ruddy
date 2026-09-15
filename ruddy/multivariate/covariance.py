"""Covariance, correlation, and matrix-condition diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from ruddy.core.enums import ResultStatus, ScalingMethod
from ruddy.data import FeatureMatrix
from ruddy.projections.preprocessing import prepare_features
from ruddy.results import AnalysisProvenance


@dataclass(frozen=True, slots=True)
class CovarianceResult:
    """Structured covariance/correlation output for a feature matrix."""

    status: ResultStatus
    reason: str | None
    covariance: pd.DataFrame
    pearson: pd.DataFrame
    spearman: pd.DataFrame
    pairwise_counts: pd.DataFrame
    feature_diagnostics: pd.DataFrame
    condition_spectrum: pd.DataFrame
    summary: dict[str, Any]
    exclusions: pd.DataFrame
    preprocessing: dict[str, Any]
    provenance: AnalysisProvenance


def _square_frame(values: np.ndarray, names: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(values, dtype=float), index=pd.Index(names), columns=pd.Index(names))


def _pairwise_finite_counts(array: np.ndarray, names: tuple[str, ...]) -> pd.DataFrame:
    finite = np.isfinite(array)
    counts = finite.T.astype(np.int64) @ finite.astype(np.int64)
    return pd.DataFrame(counts, index=pd.Index(names), columns=pd.Index(names))


def _standardized_condition_diagnostics(
    matrix: np.ndarray,
    names: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    means = np.mean(matrix, axis=0)
    centered = matrix - means
    std = np.std(matrix, axis=0, ddof=1)
    constant = (~np.isfinite(std)) | (std == 0)

    feature_rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        feature_rows.append(
            {
                "feature": name,
                "mean": float(means[index]),
                "std": float(std[index]) if np.isfinite(std[index]) else np.nan,
                "is_constant": bool(constant[index]),
                "status": (ResultStatus.DEGENERATE.value if constant[index] else ResultStatus.OK.value),
                "reason": "constant_feature" if constant[index] else None,
            }
        )
    diagnostics = pd.DataFrame(feature_rows)

    usable = ~constant
    if int(usable.sum()) == 0:
        spectrum = pd.DataFrame(columns=pd.Index(("component", "singular_value", "condition_index")))
        summary = {
            "n_observations": int(matrix.shape[0]),
            "n_features": int(matrix.shape[1]),
            "n_nonconstant_features": 0,
            "rank": 0,
            "full_rank": False,
            "condition_number": np.inf,
            "condition_basis": "centered_sample_standardized_features_without_intercept",
        }
        return diagnostics, spectrum, summary

    z = centered[:, usable] / std[usable]
    singular_values = np.linalg.svd(z, compute_uv=False)
    tolerance = np.finfo(float).eps * max(z.shape) * (singular_values[0] if singular_values.size else 0.0)
    rank = int(np.sum(singular_values > tolerance))
    if singular_values.size and singular_values[-1] > tolerance:
        condition_number = float(singular_values[0] / singular_values[-1])
    else:
        condition_number = np.inf
    condition_indices = np.full(singular_values.shape, np.inf, dtype=float)
    nonzero = singular_values > tolerance
    if singular_values.size:
        condition_indices[nonzero] = singular_values[0] / singular_values[nonzero]
    spectrum = pd.DataFrame(
        {
            "component": np.arange(1, singular_values.size + 1, dtype=int),
            "singular_value": singular_values.astype(float),
            "condition_index": condition_indices.astype(float),
        }
    )
    p = int(usable.sum())
    summary = {
        "n_observations": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "n_nonconstant_features": p,
        "rank": rank,
        "full_rank": bool(rank == p),
        "condition_number": condition_number,
        "condition_basis": "centered_sample_standardized_features_without_intercept",
    }
    return diagnostics, spectrum, summary


def analyze_covariance_structure(
    features: FeatureMatrix,
    *,
    scaling: ScalingMethod | str = ScalingMethod.NONE,
    include_spearman: bool = True,
    max_features: int = 200,
) -> CovarianceResult:
    """Compute complete-case covariance/correlation structure and condition diagnostics.

    Pairwise finite counts are reported from the source matrix. The covariance and
    correlation matrices themselves use one common complete-case set so they share a
    coherent sample basis.
    """
    if max_features < 2:
        raise ValueError("max_features must be at least 2.")
    if features.n_features < 2:
        raise ValueError("Covariance structure requires at least two features.")
    if features.n_features > max_features:
        raise ValueError(
            f"Covariance structure is limited to {max_features} features per run; "
            "reduce dimensionality explicitly before multivariate covariance analysis."
        )
    if features.is_sparse:
        raise ValueError(
            "Covariance structure currently requires dense input; Ruddy will not "
            "silently densify a sparse feature matrix."
        )

    raw = features.to_array().astype(float, copy=False)
    pairwise_counts = _pairwise_finite_counts(raw, features.feature_names)
    prepared = prepare_features(features, scaling=scaling, minimum_observations=2)
    if sparse.issparse(prepared.matrix):  # defensive
        raise ValueError("Dense input is required for covariance analysis.")
    matrix = np.asarray(prepared.matrix, dtype=float)
    names = prepared.feature_names

    covariance = np.cov(matrix, rowvar=False, ddof=1)
    covariance = np.atleast_2d(np.asarray(covariance, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson = np.corrcoef(matrix, rowvar=False)
    pearson = np.atleast_2d(np.asarray(pearson, dtype=float))

    if include_spearman:
        spearman = pd.DataFrame(matrix, columns=pd.Index(names)).corr(method="spearman").to_numpy(dtype=float)
    else:
        spearman = np.full((len(names), len(names)), np.nan, dtype=float)

    feature_diagnostics, spectrum, condition_summary = _standardized_condition_diagnostics(matrix, names)
    n_nonconstant = int((~feature_diagnostics["is_constant"]).sum())
    if n_nonconstant < 2:
        status = ResultStatus.DEGENERATE
        reason = "insufficient_nonconstant_features"
    else:
        status = ResultStatus.OK
        reason = None

    complete_n = int(matrix.shape[0])
    summary = {
        **condition_summary,
        "n_source_observations": int(features.n_observations),
        "n_complete_case_observations": complete_n,
        "n_excluded_observations": int(prepared.exclusions.shape[0]),
        "complete_case_fraction": float(complete_n / features.n_observations),
        "scaling": prepared.metadata["scaling"]["method"],
        "spearman_included": bool(include_spearman),
        "matrix_sample_policy": "common_complete_case",
        "pairwise_count_policy": "source_finite_pairs",
    }
    provenance = AnalysisProvenance(
        analysis="multivariate_covariance_structure",
        parameters={
            "scaling": prepared.metadata["scaling"]["method"],
            "include_spearman": bool(include_spearman),
            "max_features": int(max_features),
        },
        input_summary={
            "n_observations": features.n_observations,
            "n_features": features.n_features,
            "n_complete_case_observations": complete_n,
            "n_excluded_observations": int(prepared.exclusions.shape[0]),
        },
    )
    return CovarianceResult(
        status=status,
        reason=reason,
        covariance=_square_frame(covariance, names),
        pearson=_square_frame(pearson, names),
        spearman=_square_frame(spearman, names),
        pairwise_counts=pairwise_counts,
        feature_diagnostics=feature_diagnostics,
        condition_spectrum=spectrum,
        summary=summary,
        exclusions=prepared.exclusions,
        preprocessing=prepared.metadata,
        provenance=provenance,
    )
