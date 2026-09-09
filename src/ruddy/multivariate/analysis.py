"""Unified multivariate diagnostics for generic feature matrices."""

from __future__ import annotations

from dataclasses import dataclass

from ruddy.core.enums import ResultStatus, ScalingMethod
from ruddy.data import FeatureMatrix
from ruddy.multivariate.collinearity import CollinearityResult, analyze_collinearity
from ruddy.multivariate.covariance import CovarianceResult, analyze_covariance_structure
from ruddy.multivariate.distances import MahalanobisResult, analyze_mahalanobis
from ruddy.results import AnalysisProvenance


@dataclass(frozen=True, slots=True)
class MultivariateResult:
    """Combined covariance, collinearity, and multivariate-distance diagnostics."""

    status: ResultStatus
    reason: str | None
    covariance: CovarianceResult
    collinearity: CollinearityResult
    mahalanobis: MahalanobisResult
    provenance: AnalysisProvenance


def analyze_multivariate(
    features: FeatureMatrix,
    *,
    scaling: ScalingMethod | str = ScalingMethod.NONE,
    include_spearman: bool = True,
    include_robust_mahalanobis: bool = True,
    mahalanobis_threshold_quantile: float = 0.975,
    robust_support_fraction: float | None = None,
    random_state: int = 0,
    max_covariance_features: int = 200,
    max_collinearity_features: int = 100,
    max_mahalanobis_features: int = 100,
) -> MultivariateResult:
    """Run the core Phase-7 multivariate EDA diagnostics on one feature matrix."""

    covariance = analyze_covariance_structure(
        features,
        scaling=scaling,
        include_spearman=include_spearman,
        max_features=max_covariance_features,
    )
    collinearity = analyze_collinearity(
        features,
        scaling=scaling,
        max_features=max_collinearity_features,
    )
    mahalanobis = analyze_mahalanobis(
        features,
        scaling=scaling,
        threshold_quantile=mahalanobis_threshold_quantile,
        include_robust=include_robust_mahalanobis,
        robust_support_fraction=robust_support_fraction,
        random_state=random_state,
        max_features=max_mahalanobis_features,
    )
    component_statuses = (
        covariance.status,
        collinearity.status,
        mahalanobis.status,
    )
    if all(status is ResultStatus.SKIPPED for status in component_statuses):
        status = ResultStatus.SKIPPED
        reason = "all_multivariate_components_skipped"
    elif all(status is not ResultStatus.OK for status in component_statuses):
        status = ResultStatus.DEGENERATE
        reason = "no_multivariate_component_fully_feasible"
    else:
        status = ResultStatus.OK
        reason = None
    scale = scaling if isinstance(scaling, ScalingMethod) else ScalingMethod(scaling)
    provenance = AnalysisProvenance(
        analysis="multivariate",
        parameters={
            "scaling": scale.value,
            "include_spearman": bool(include_spearman),
            "include_robust_mahalanobis": bool(include_robust_mahalanobis),
            "mahalanobis_threshold_quantile": float(mahalanobis_threshold_quantile),
            "robust_support_fraction": robust_support_fraction,
            "max_covariance_features": int(max_covariance_features),
            "max_collinearity_features": int(max_collinearity_features),
            "max_mahalanobis_features": int(max_mahalanobis_features),
        },
        input_summary={
            "n_observations": features.n_observations,
            "n_features": features.n_features,
            "source_storage": "sparse" if features.is_sparse else "dense",
        },
        random_state=int(random_state),
    )
    return MultivariateResult(
        status=status,
        reason=reason,
        covariance=covariance,
        collinearity=collinearity,
        mahalanobis=mahalanobis,
        provenance=provenance,
    )
