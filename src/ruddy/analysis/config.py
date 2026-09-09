"""Unified analysis configuration contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from ruddy.core.enums import (
    AlignmentMode,
    ColumnKind,
    ColumnRole,
    ComparisonTest,
    CorrelationMethod,
    OutlierMethod,
    PAdjustMethod,
    ScalingMethod,
)
from ruddy.core.types import KindOverrides, RoleOverrides
from ruddy.univariate.numeric import validate_quantiles


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Cross-cutting configuration shared by Ruddy analyses."""

    id_column: str | None = None
    role_overrides: RoleOverrides = field(default_factory=dict)
    kind_overrides: KindOverrides = field(default_factory=dict)
    annotation_alignment: AlignmentMode = AlignmentMode.STRICT
    random_state: int = 0

    # Phase 4 statistical semantics.
    responses: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()

    # Phase 2 descriptive controls.
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    min_numeric_n: int = 3
    max_category_levels: int = 50
    max_missingness_patterns: int = 20
    max_pairwise_columns: int = 200

    # Phase 3 bivariate controls.
    correlations: tuple[CorrelationMethod, ...] = (
        CorrelationMethod.PEARSON,
        CorrelationMethod.SPEARMAN,
        CorrelationMethod.KENDALL,
    )
    comparison_tests: tuple[ComparisonTest, ...] = (
        ComparisonTest.WELCH_T,
        ComparisonTest.MANN_WHITNEY,
        ComparisonTest.WELCH_ANOVA,
        ComparisonTest.KRUSKAL_WALLIS,
        ComparisonTest.CHI_SQUARE,
        ComparisonTest.FISHER_EXACT,
    )
    p_adjust: PAdjustMethod = PAdjustMethod.FDR_BH
    min_group_n: int = 3
    min_correlation_pairs: int = 3
    max_group_levels: int = 20
    max_correlation_columns: int = 100
    pairwise: bool = False

    # Phase 5 univariate outlier controls.
    outlier_methods: tuple[OutlierMethod, ...] = (
        OutlierMethod.IQR,
        OutlierMethod.ROBUST_Z,
    )
    outlier_iqr_multiplier: float = 1.5
    outlier_robust_z_threshold: float = 3.5
    include_outlier_flags: bool = False

    # Phase 6 feature-space controls.
    projection_scaling: ScalingMethod = ScalingMethod.NONE
    projection_n_components: int = 2
    projection_metric: str = "euclidean"
    tsne_perplexity: float = 30.0
    tsne_max_iter: int = 1000
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1

    # Phase 7 multivariate statistical controls.
    multivariate_scaling: ScalingMethod = ScalingMethod.NONE
    multivariate_include_spearman: bool = True
    multivariate_max_covariance_features: int = 200
    multivariate_max_collinearity_features: int = 100
    multivariate_max_mahalanobis_features: int = 100
    mahalanobis_threshold_quantile: float = 0.975
    mahalanobis_include_robust: bool = True
    mahalanobis_robust_support_fraction: float | None = None
    manova_max_responses: int = 20
    manova_max_factor_levels: int = 20
    manova_min_level_n: int = 3

    # Phase 8 factorial ANOVA/ANCOVA controls.
    factorial_ss_type: int = 2
    factorial_p_adjust: PAdjustMethod = PAdjustMethod.NONE
    factorial_robust_covariance: str | None = None
    factorial_min_cell_n: int = 2
    factorial_max_factor_levels: int = 20
    factorial_max_design_cells: int = 5000
    factorial_max_design_columns: int = 500
    factorial_max_interaction_order: int = 3
    factorial_diagnostic_alpha: float = 0.05
    factorial_condition_number_threshold: float = 30.0

    def __post_init__(self) -> None:
        roles = {
            column: role if isinstance(role, ColumnRole) else ColumnRole(role)
            for column, role in self.role_overrides.items()
        }
        kinds = {
            column: kind if isinstance(kind, ColumnKind) else ColumnKind(kind)
            for column, kind in self.kind_overrides.items()
        }
        mode = (
            self.annotation_alignment
            if isinstance(self.annotation_alignment, AlignmentMode)
            else AlignmentMode(self.annotation_alignment)
        )
        responses = tuple(str(column) for column in self.responses)
        groups = tuple(str(column) for column in self.groups)
        if len(set(responses)) != len(responses):
            raise ValueError("responses cannot contain duplicates.")
        if len(set(groups)) != len(groups):
            raise ValueError("groups cannot contain duplicates.")
        overlap = sorted(set(responses) & set(groups))
        if overlap:
            raise ValueError(
                f"Columns cannot be configured as both response and group: {overlap}."
            )
        quantiles = validate_quantiles(tuple(self.quantiles))
        correlations = tuple(
            value if isinstance(value, CorrelationMethod) else CorrelationMethod(value)
            for value in self.correlations
        )
        comparison_tests = tuple(
            value if isinstance(value, ComparisonTest) else ComparisonTest(value)
            for value in self.comparison_tests
        )
        p_adjust = (
            self.p_adjust
            if isinstance(self.p_adjust, PAdjustMethod)
            else PAdjustMethod(self.p_adjust)
        )
        factorial_p_adjust = (
            self.factorial_p_adjust
            if isinstance(self.factorial_p_adjust, PAdjustMethod)
            else PAdjustMethod(self.factorial_p_adjust)
        )
        projection_scaling = (
            self.projection_scaling
            if isinstance(self.projection_scaling, ScalingMethod)
            else ScalingMethod(self.projection_scaling)
        )
        multivariate_scaling = (
            self.multivariate_scaling
            if isinstance(self.multivariate_scaling, ScalingMethod)
            else ScalingMethod(self.multivariate_scaling)
        )
        outlier_methods = tuple(
            value if isinstance(value, OutlierMethod) else OutlierMethod(value)
            for value in self.outlier_methods
        )
        if len(set(correlations)) != len(correlations):
            raise ValueError("correlations cannot contain duplicates.")
        if len(set(comparison_tests)) != len(comparison_tests):
            raise ValueError("comparison_tests cannot contain duplicates.")
        if not outlier_methods:
            raise ValueError("outlier_methods cannot be empty.")
        if len(set(outlier_methods)) != len(outlier_methods):
            raise ValueError("outlier_methods cannot contain duplicates.")
        if self.outlier_iqr_multiplier <= 0:
            raise ValueError("outlier_iqr_multiplier must be greater than zero.")
        if self.outlier_robust_z_threshold <= 0:
            raise ValueError("outlier_robust_z_threshold must be greater than zero.")
        if self.projection_n_components < 1:
            raise ValueError("projection_n_components must be at least 1.")
        if self.tsne_perplexity <= 0:
            raise ValueError("tsne_perplexity must be greater than zero.")
        if self.tsne_max_iter < 250:
            raise ValueError("tsne_max_iter must be at least 250.")
        if self.umap_n_neighbors < 2:
            raise ValueError("umap_n_neighbors must be at least 2.")
        if not 0.0 <= self.umap_min_dist <= 1.0:
            raise ValueError("umap_min_dist must be between 0 and 1.")
        if not str(self.projection_metric).strip():
            raise ValueError("projection_metric cannot be empty.")
        if self.multivariate_max_covariance_features < 2:
            raise ValueError("multivariate_max_covariance_features must be at least 2.")
        if self.multivariate_max_collinearity_features < 2:
            raise ValueError("multivariate_max_collinearity_features must be at least 2.")
        if self.multivariate_max_mahalanobis_features < 1:
            raise ValueError("multivariate_max_mahalanobis_features must be at least 1.")
        if not 0.5 < self.mahalanobis_threshold_quantile < 1.0:
            raise ValueError("mahalanobis_threshold_quantile must be between 0.5 and 1.0.")
        if (
            self.mahalanobis_robust_support_fraction is not None
            and not 0.5 <= self.mahalanobis_robust_support_fraction <= 1.0
        ):
            raise ValueError("mahalanobis_robust_support_fraction must be between 0.5 and 1.0.")
        if self.manova_max_responses < 2:
            raise ValueError("manova_max_responses must be at least 2.")
        if self.manova_max_factor_levels < 2:
            raise ValueError("manova_max_factor_levels must be at least 2.")
        if self.manova_min_level_n < 2:
            raise ValueError("manova_min_level_n must be at least 2.")
        if self.factorial_ss_type not in {2, 3}:
            raise ValueError("factorial_ss_type must be 2 or 3.")
        if self.factorial_robust_covariance is not None:
            robust = str(self.factorial_robust_covariance).strip().lower()
            if robust in {"", "none", "nonrobust"}:
                robust = None
            elif robust not in {"hc0", "hc1", "hc2", "hc3"}:
                raise ValueError("factorial_robust_covariance must be one of hc0, hc1, hc2, hc3, or None.")
        else:
            robust = None
        if self.factorial_min_cell_n < 1:
            raise ValueError("factorial_min_cell_n must be at least 1.")
        if self.factorial_max_factor_levels < 2:
            raise ValueError("factorial_max_factor_levels must be at least 2.")
        if self.factorial_max_design_cells < 1:
            raise ValueError("factorial_max_design_cells must be at least 1.")
        if self.factorial_max_design_columns < 2:
            raise ValueError("factorial_max_design_columns must be at least 2.")
        if self.factorial_max_interaction_order < 2:
            raise ValueError("factorial_max_interaction_order must be at least 2.")
        if not 0.0 < self.factorial_diagnostic_alpha < 1.0:
            raise ValueError("factorial_diagnostic_alpha must lie in (0, 1).")
        if self.factorial_condition_number_threshold <= 0.0:
            raise ValueError("factorial_condition_number_threshold must be greater than zero.")
        if self.min_numeric_n < 2:
            raise ValueError("min_numeric_n must be at least 2.")
        if self.max_category_levels < 2:
            raise ValueError("max_category_levels must be at least 2.")
        if self.max_missingness_patterns < 1:
            raise ValueError("max_missingness_patterns must be at least 1.")
        if self.max_pairwise_columns < 1:
            raise ValueError("max_pairwise_columns must be at least 1.")
        if self.min_group_n < 2:
            raise ValueError("min_group_n must be at least 2.")
        if self.min_correlation_pairs < 2:
            raise ValueError("min_correlation_pairs must be at least 2.")
        if self.max_group_levels < 2:
            raise ValueError("max_group_levels must be at least 2.")
        if self.max_correlation_columns < 2:
            raise ValueError("max_correlation_columns must be at least 2.")

        object.__setattr__(self, "role_overrides", MappingProxyType(roles))
        object.__setattr__(self, "responses", responses)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "kind_overrides", MappingProxyType(kinds))
        object.__setattr__(self, "annotation_alignment", mode)
        object.__setattr__(self, "quantiles", quantiles)
        object.__setattr__(self, "correlations", correlations)
        object.__setattr__(self, "comparison_tests", comparison_tests)
        object.__setattr__(self, "p_adjust", p_adjust)
        object.__setattr__(self, "factorial_p_adjust", factorial_p_adjust)
        object.__setattr__(self, "factorial_robust_covariance", robust)
        object.__setattr__(self, "outlier_methods", outlier_methods)
        object.__setattr__(self, "projection_scaling", projection_scaling)
        object.__setattr__(self, "multivariate_scaling", multivariate_scaling)

    def dataset_kwargs(self) -> dict[str, object]:
        """Return arguments accepted directly by :class:`TabularDataset`."""

        return {
            "id_column": self.id_column,
            "role_overrides": dict(self.role_overrides),
            "kind_overrides": dict(self.kind_overrides),
        }

    def profiling_kwargs(self) -> dict[str, object]:
        """Return controls used by descriptive profiling."""

        return {
            "max_missingness_patterns": self.max_missingness_patterns,
            "max_pairwise_columns": self.max_pairwise_columns,
        }

    def univariate_kwargs(self) -> dict[str, object]:
        """Return controls used by univariate descriptive analysis."""

        return {
            "quantiles": self.quantiles,
            "min_numeric_n": self.min_numeric_n,
            "max_category_levels": self.max_category_levels,
            **self.profiling_kwargs(),
        }
    def outlier_kwargs(self) -> dict[str, object]:
        """Return controls used by univariate outlier diagnostics."""

        return {
            "methods": self.outlier_methods,
            "min_numeric_n": self.min_numeric_n,
            "iqr_multiplier": self.outlier_iqr_multiplier,
            "robust_z_threshold": self.outlier_robust_z_threshold,
            "include_flags": self.include_outlier_flags,
        }

    def pca_kwargs(self) -> dict[str, object]:
        """Return controls used by PCA."""

        return {
            "n_components": self.projection_n_components,
            "scaling": self.projection_scaling,
            "random_state": self.random_state,
        }

    def tsne_kwargs(self) -> dict[str, object]:
        """Return controls used by t-SNE."""

        return {
            "n_components": self.projection_n_components,
            "scaling": self.projection_scaling,
            "metric": self.projection_metric,
            "perplexity": self.tsne_perplexity,
            "random_state": self.random_state,
            "max_iter": self.tsne_max_iter,
        }

    def umap_kwargs(self) -> dict[str, object]:
        """Return controls used by UMAP."""

        return {
            "n_components": self.projection_n_components,
            "scaling": self.projection_scaling,
            "metric": self.projection_metric,
            "n_neighbors": self.umap_n_neighbors,
            "min_dist": self.umap_min_dist,
            "random_state": self.random_state,
        }

    def multivariate_kwargs(self) -> dict[str, object]:
        """Return controls used by core multivariate feature diagnostics."""

        return {
            "scaling": self.multivariate_scaling,
            "include_spearman": self.multivariate_include_spearman,
            "include_robust_mahalanobis": self.mahalanobis_include_robust,
            "mahalanobis_threshold_quantile": self.mahalanobis_threshold_quantile,
            "robust_support_fraction": self.mahalanobis_robust_support_fraction,
            "random_state": self.random_state,
            "max_covariance_features": self.multivariate_max_covariance_features,
            "max_collinearity_features": self.multivariate_max_collinearity_features,
            "max_mahalanobis_features": self.multivariate_max_mahalanobis_features,
        }

    def manova_kwargs(self) -> dict[str, object]:
        """Return generic dimensionality/sample guards used by MANOVA."""

        return {
            "max_responses": self.manova_max_responses,
            "max_factor_levels": self.manova_max_factor_levels,
            "min_level_n": self.manova_min_level_n,
        }

    def factorial_kwargs(self) -> dict[str, object]:
        """Return controls used by factorial ANOVA/ANCOVA."""

        return {
            "ss_type": self.factorial_ss_type,
            "p_adjust": self.factorial_p_adjust,
            "robust_covariance": self.factorial_robust_covariance,
            "min_cell_n": self.factorial_min_cell_n,
            "max_factor_levels": self.factorial_max_factor_levels,
            "max_design_cells": self.factorial_max_design_cells,
            "max_design_columns": self.factorial_max_design_columns,
            "max_interaction_order": self.factorial_max_interaction_order,
            "diagnostic_alpha": self.factorial_diagnostic_alpha,
            "condition_number_threshold": self.factorial_condition_number_threshold,
        }

    def grouped_kwargs(self) -> dict[str, object]:
        """Return controls used by response-centric grouped analysis."""

        return {
            "responses": self.responses or None,
            "groups": self.groups or None,
            "comparison_tests": self.comparison_tests,
            "p_adjust": self.p_adjust,
            "min_group_n": self.min_group_n,
            "max_group_levels": self.max_group_levels,
            "max_category_levels": self.max_category_levels,
            "pairwise": self.pairwise,
        }

    def bivariate_kwargs(self) -> dict[str, object]:
        """Return controls used by Phase 3 mixed-type bivariate analysis."""

        return {
            "correlations": self.correlations,
            "comparison_tests": self.comparison_tests,
            "p_adjust": self.p_adjust,
            "min_group_n": self.min_group_n,
            "min_correlation_pairs": self.min_correlation_pairs,
            "max_group_levels": self.max_group_levels,
            "max_correlation_columns": self.max_correlation_columns,
            "max_category_levels": self.max_category_levels,
            "pairwise": self.pairwise,
        }
