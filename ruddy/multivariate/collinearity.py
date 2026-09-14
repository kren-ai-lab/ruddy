"""Multicollinearity diagnostics for explicitly selected numeric features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ruddy.core.enums import ResultStatus, ScalingMethod
from ruddy.data import FeatureMatrix
from ruddy.projections.preprocessing import prepare_features
from ruddy.results import AnalysisProvenance


@dataclass(frozen=True, slots=True)
class CollinearityResult:
    """VIF/tolerance plus standardized design-rank diagnostics."""

    status: ResultStatus
    reason: str | None
    features: pd.DataFrame
    condition_spectrum: pd.DataFrame
    summary: dict[str, Any]
    exclusions: pd.DataFrame
    preprocessing: dict[str, Any]
    provenance: AnalysisProvenance


def analyze_collinearity(
    features: FeatureMatrix,
    *,
    scaling: ScalingMethod | str = ScalingMethod.NONE,
    max_features: int = 100,
) -> CollinearityResult:
    """Compute VIF/tolerance with an intercept and standardized condition indices.

    VIF regressions include an intercept. Condition diagnostics are computed on
    centered, sample-standardized features without an intercept so scale does not
    dominate the condition number.
    """

    if max_features < 2:
        raise ValueError("max_features must be at least 2.")
    if features.n_features < 2:
        raise ValueError("Collinearity analysis requires at least two features.")
    if features.n_features > max_features:
        raise ValueError(
            f"Collinearity analysis is limited to {max_features} features per run; "
            "reduce dimensionality or select predictors explicitly."
        )
    if features.is_sparse:
        raise ValueError(
            "Collinearity diagnostics currently require dense input; Ruddy will not "
            "silently densify a sparse feature matrix."
        )

    prepared = prepare_features(features, scaling=scaling, minimum_observations=3)
    if sparse.issparse(prepared.matrix):
        raise ValueError("Dense input is required for collinearity diagnostics.")
    matrix = np.asarray(prepared.matrix, dtype=float)
    names = prepared.feature_names
    n, p = matrix.shape
    means = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
    constant = (~np.isfinite(std)) | (std == 0)
    usable = ~constant

    feature_rows: list[dict[str, Any]] = []
    if int(usable.sum()) > 0:
        z = (matrix[:, usable] - means[usable]) / std[usable]
        singular_values = np.linalg.svd(z, compute_uv=False)
        tol = np.finfo(float).eps * max(z.shape) * (
            singular_values[0] if singular_values.size else 0.0
        )
        rank = int(np.sum(singular_values > tol))
        full_rank = bool(rank == int(usable.sum()))
        if singular_values.size and singular_values[-1] > tol:
            condition_number = float(singular_values[0] / singular_values[-1])
        else:
            condition_number = np.inf
        indices = np.full_like(singular_values, np.inf, dtype=float)
        mask = singular_values > tol
        if singular_values.size:
            indices[mask] = singular_values[0] / singular_values[mask]
        spectrum = pd.DataFrame(
            {
                "component": np.arange(1, singular_values.size + 1, dtype=int),
                "singular_value": singular_values.astype(float),
                "condition_index": indices.astype(float),
            }
        )
    else:
        z = np.empty((n, 0), dtype=float)
        rank = 0
        full_rank = False
        condition_number = np.inf
        spectrum = pd.DataFrame(columns=("component", "singular_value", "condition_index"))

    usable_count = int(usable.sum())
    residual_df_ok = bool(n > usable_count + 1)
    can_compute_vif = bool(usable_count >= 2 and full_rank and residual_df_ok)

    vif_by_index: dict[int, float] = {}
    if can_compute_vif:
        design = np.column_stack([np.ones(n, dtype=float), matrix[:, usable]])
        usable_indices = np.flatnonzero(usable)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for local_index, source_index in enumerate(usable_indices, start=1):
                vif_by_index[int(source_index)] = float(
                    variance_inflation_factor(design, local_index)
                )

    for index, name in enumerate(names):
        if constant[index]:
            status = ResultStatus.DEGENERATE
            reason = "constant_feature"
            vif = np.nan
            tolerance = np.nan
        elif not full_rank:
            status = ResultStatus.DEGENERATE
            reason = "rank_deficient_design"
            vif = np.nan
            tolerance = np.nan
        elif not residual_df_ok:
            status = ResultStatus.SKIPPED
            reason = "insufficient_residual_degrees_of_freedom"
            vif = np.nan
            tolerance = np.nan
        else:
            vif = vif_by_index.get(index, np.nan)
            tolerance = 0.0 if np.isinf(vif) else (1.0 / vif if vif > 0 else np.nan)
            status = ResultStatus.OK
            reason = None
        feature_rows.append(
            {
                "feature": name,
                "vif": vif,
                "tolerance": tolerance,
                "is_constant": bool(constant[index]),
                "status": status.value,
                "reason": reason,
            }
        )

    if usable_count < 2:
        overall_status = ResultStatus.DEGENERATE
        overall_reason = "insufficient_nonconstant_features"
    elif not full_rank:
        overall_status = ResultStatus.DEGENERATE
        overall_reason = "rank_deficient_design"
    elif not residual_df_ok:
        overall_status = ResultStatus.SKIPPED
        overall_reason = "insufficient_residual_degrees_of_freedom"
    else:
        overall_status = ResultStatus.OK
        overall_reason = None

    summary = {
        "n_source_observations": int(features.n_observations),
        "n_complete_case_observations": int(n),
        "n_excluded_observations": int(prepared.exclusions.shape[0]),
        "n_features": int(p),
        "n_nonconstant_features": usable_count,
        "rank": int(rank),
        "full_rank": bool(full_rank),
        "condition_number": float(condition_number),
        "vif_intercept_policy": "included_in_auxiliary_regressions_not_reported",
        "condition_basis": "centered_sample_standardized_features_without_intercept",
        "scaling": prepared.metadata["scaling"]["method"],
    }
    provenance = AnalysisProvenance(
        analysis="multivariate_collinearity",
        parameters={
            "scaling": prepared.metadata["scaling"]["method"],
            "max_features": int(max_features),
            "vif_intercept": True,
        },
        input_summary={
            "n_observations": features.n_observations,
            "n_features": features.n_features,
            "n_complete_case_observations": int(n),
        },
    )
    return CollinearityResult(
        status=overall_status,
        reason=overall_reason,
        features=pd.DataFrame(feature_rows),
        condition_spectrum=spectrum,
        summary=summary,
        exclusions=prepared.exclusions,
        preprocessing=prepared.metadata,
        provenance=provenance,
    )
