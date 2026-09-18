"""Classical and robust Mahalanobis distance diagnostics."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from scipy import sparse, stats
from sklearn.covariance import EmpiricalCovariance, MinCovDet

from ruddy.core.enums import ResultStatus, ScalingMethod
from ruddy.core.frames import id_dtype
from ruddy.projections.preprocessing import prepare_features
from ruddy.results import AnalysisProvenance

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polars._typing import PolarsDataType

    from ruddy.data import FeatureMatrix

MAHALANOBIS_DISTANCE_SCHEMA_BASE: dict[str, PolarsDataType] = {
    "source_row_index": pl.Int64,
    "method": pl.String,
    "squared_distance": pl.Float64,
    "distance": pl.Float64,
    "degrees_of_freedom": pl.Int64,
    "threshold_quantile": pl.Float64,
    "threshold_squared": pl.Float64,
    "is_flagged": pl.Boolean,
    "status": pl.String,
    "reason": pl.String,
}

MAHALANOBIS_DISTANCE_COLUMNS: tuple[str, ...] = (
    "source_row_index",
    "observation_id",
    "method",
    "squared_distance",
    "distance",
    "degrees_of_freedom",
    "threshold_quantile",
    "threshold_squared",
    "is_flagged",
    "status",
    "reason",
)

MAHALANOBIS_METHOD_SCHEMA: dict[str, PolarsDataType] = {
    "method": pl.String,
    "n_observations": pl.Int64,
    "n_features": pl.Int64,
    "covariance_rank": pl.Int64,
    "threshold_quantile": pl.Float64,
    "degrees_of_freedom": pl.Int64,
    "status": pl.String,
    "reason": pl.String,
}


@dataclass(frozen=True, slots=True)
class MahalanobisResult:
    """Observation-level classical/robust Mahalanobis diagnostics."""

    status: ResultStatus
    reason: str | None
    distances: pl.DataFrame
    methods: pl.DataFrame
    exclusions: pl.DataFrame
    preprocessing: dict[str, Any]
    provenance: AnalysisProvenance


def _method_rows(
    squared: np.ndarray,
    *,
    method: str,
    ids: Sequence[Any],
    source_rows: np.ndarray,
    df: int,
    threshold_quantile: float,
) -> pl.DataFrame:
    threshold_squared = float(stats.chi2.ppf(threshold_quantile, df=df))
    values = np.asarray(squared, dtype=float)
    id_list = list(ids)
    id_dt = id_dtype(id_list)
    df_var = pl.DataFrame(
        {
            "source_row_index": pl.Series("source_row_index", source_rows.astype(np.int64), dtype=pl.Int64),
            "observation_id": pl.Series("observation_id", id_list, dtype=id_dt),
            "squared_distance": pl.Series("squared_distance", values, dtype=pl.Float64),
            "distance": pl.Series("distance", np.sqrt(np.maximum(values, 0.0)), dtype=pl.Float64),
            "is_flagged": pl.Series("is_flagged", values > threshold_squared, dtype=pl.Boolean),
        }
    )
    return df_var.with_columns(
        pl.lit(method, dtype=pl.String).alias("method"),
        pl.lit(int(df), dtype=pl.Int64).alias("degrees_of_freedom"),
        pl.lit(float(threshold_quantile), dtype=pl.Float64).alias("threshold_quantile"),
        pl.lit(threshold_squared, dtype=pl.Float64).alias("threshold_squared"),
        pl.lit(ResultStatus.OK.value, dtype=pl.String).alias("status"),
        pl.lit(None, dtype=pl.String).alias("reason"),
    ).select(list(MAHALANOBIS_DISTANCE_COLUMNS))


def analyze_mahalanobis(
    features: FeatureMatrix,
    *,
    scaling: ScalingMethod | str = ScalingMethod.NONE,
    threshold_quantile: float = 0.975,
    include_robust: bool = True,
    robust_support_fraction: float | None = None,
    random_state: int = 0,
    max_features: int = 100,
) -> MahalanobisResult:
    """Compute classical and optional robust Mahalanobis distances.

    Threshold flags use a chi-square reference with ``df = n_features``. Flags are
    diagnostic only; observations are never removed or altered.
    """
    if not 0.5 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must be between 0.5 and 1.0.")
    if max_features < 1:
        raise ValueError("max_features must be at least 1.")
    if features.n_features > max_features:
        raise ValueError(
            f"Mahalanobis analysis is limited to {max_features} features per run; "
            "for high-dimensional representations apply explicit dimensionality reduction first."
        )
    if robust_support_fraction is not None and not 0.5 <= robust_support_fraction <= 1.0:
        raise ValueError("robust_support_fraction must be between 0.5 and 1.0.")
    if features.is_sparse:
        raise ValueError(
            "Mahalanobis diagnostics require dense input; Ruddy will not silently "
            "densify a sparse feature matrix."
        )

    prepared = prepare_features(features, scaling=scaling, minimum_observations=3)
    if sparse.issparse(prepared.matrix):
        raise ValueError("Dense input is required for Mahalanobis diagnostics.")
    matrix = np.asarray(prepared.matrix, dtype=float)
    n, p = matrix.shape
    centered_rank = int(np.linalg.matrix_rank(matrix - matrix.mean(axis=0)))

    method_records: list[dict[str, Any]] = []
    distance_frames: list[pl.DataFrame] = []
    classical_ok = bool(n > p and centered_rank == p)
    if classical_ok:
        classical = EmpiricalCovariance(assume_centered=False).fit(matrix)
        squared = classical.mahalanobis(matrix)
        distance_frames.append(
            _method_rows(
                squared,
                method="classical",
                ids=prepared.observation_ids,
                source_rows=prepared.source_row_indices,
                df=p,
                threshold_quantile=threshold_quantile,
            )
        )
        method_records.append(
            {
                "method": "classical",
                "n_observations": int(n),
                "n_features": int(p),
                "covariance_rank": int(np.linalg.matrix_rank(classical.covariance_)),
                "threshold_quantile": float(threshold_quantile),
                "degrees_of_freedom": int(p),
                "status": ResultStatus.OK.value,
                "reason": None,
            }
        )
    else:
        reason = (
            "singular_covariance" if centered_rank < p else "insufficient_observations_for_covariance_inverse"
        )
        method_records.append(
            {
                "method": "classical",
                "n_observations": int(n),
                "n_features": int(p),
                "covariance_rank": int(centered_rank),
                "threshold_quantile": float(threshold_quantile),
                "degrees_of_freedom": int(p),
                "status": ResultStatus.DEGENERATE.value,
                "reason": reason,
            }
        )

    robust_ok = False
    if include_robust:
        if n <= p:
            robust_reason = "insufficient_observations_for_robust_covariance"
        elif centered_rank < p:
            robust_reason = "singular_covariance"
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    robust = MinCovDet(
                        support_fraction=robust_support_fraction,
                        random_state=int(random_state),
                    ).fit(matrix)
                robust_rank = int(np.linalg.matrix_rank(robust.covariance_))
                if robust_rank < p:
                    robust_reason = "singular_robust_covariance"
                else:
                    squared = robust.mahalanobis(matrix)
                    distance_frames.append(
                        _method_rows(
                            squared,
                            method="robust",
                            ids=prepared.observation_ids,
                            source_rows=prepared.source_row_indices,
                            df=p,
                            threshold_quantile=threshold_quantile,
                        )
                    )
                    method_records.append(
                        {
                            "method": "robust",
                            "n_observations": int(n),
                            "n_features": int(p),
                            "covariance_rank": robust_rank,
                            "threshold_quantile": float(threshold_quantile),
                            "degrees_of_freedom": int(p),
                            "status": ResultStatus.OK.value,
                            "reason": None,
                        }
                    )
                    robust_ok = True
                    robust_reason = None
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                robust_reason = "robust_covariance_failed"
        if not robust_ok:
            method_records.append(
                {
                    "method": "robust",
                    "n_observations": int(n),
                    "n_features": int(p),
                    "covariance_rank": int(centered_rank),
                    "threshold_quantile": float(threshold_quantile),
                    "degrees_of_freedom": int(p),
                    "status": ResultStatus.DEGENERATE.value,
                    "reason": robust_reason,
                }
            )

    methods = pl.DataFrame(method_records, schema=MAHALANOBIS_METHOD_SCHEMA)
    if distance_frames:
        distances = pl.concat(distance_frames, how="vertical")
    else:
        id_dt = pl.Series(prepared.observation_ids).dtype if len(prepared.observation_ids) > 0 else pl.String
        schema = {**MAHALANOBIS_DISTANCE_SCHEMA_BASE, "observation_id": id_dt}
        distances = pl.DataFrame(schema={col: schema[col] for col in MAHALANOBIS_DISTANCE_COLUMNS})

    any_ok = bool((methods["status"] == ResultStatus.OK.value).any())
    status = ResultStatus.OK if any_ok else ResultStatus.DEGENERATE
    reason = None if any_ok else "mahalanobis_not_feasible"
    scale_value = prepared.metadata["scaling"]["method"]
    provenance = AnalysisProvenance(
        analysis="multivariate_mahalanobis",
        parameters={
            "scaling": scale_value,
            "threshold_quantile": float(threshold_quantile),
            "include_robust": bool(include_robust),
            "robust_support_fraction": robust_support_fraction,
            "max_features": int(max_features),
            "threshold_reference": "chi_square",
            "degrees_of_freedom_policy": "n_features",
        },
        input_summary={
            "n_observations": features.n_observations,
            "n_features": features.n_features,
            "n_complete_case_observations": int(n),
            "n_excluded_observations": prepared.exclusions.height,
            "centered_rank": centered_rank,
        },
        random_state=int(random_state),
    )
    return MahalanobisResult(
        status=status,
        reason=reason,
        distances=distances,
        methods=methods,
        exclusions=prepared.exclusions,
        preprocessing=prepared.metadata,
        provenance=provenance,
    )
