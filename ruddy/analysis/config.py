"""Unified analysis configuration contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ruddy.core.enums import (
    AlignmentMode,
    AnalysisBlock,
    ColumnKind,
    ColumnRole,
    ComparisonTest,
    CorrelationMethod,
    OutlierMethod,
    PAdjustMethod,
    ScalingMethod,
)
from ruddy.univariate.numeric import validate_quantiles

if TYPE_CHECKING:
    from ruddy.core.types import KindOverrides, RoleOverrides


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Cross-cutting configuration shared by Ruddy analyses."""

    id_column: str | None = None
    role_overrides: RoleOverrides = field(default_factory=dict)
    kind_overrides: KindOverrides = field(default_factory=dict)
    annotation_alignment: AlignmentMode | str = AlignmentMode.STRICT
    feature_alignment: AlignmentMode | str = AlignmentMode.STRICT
    random_state: int = 0

    # Response and group selection.
    responses: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()

    # Unified orchestration. Descriptive blocks run by default; every
    # inferential/feature-space block must be explicitly enabled.
    enabled_blocks: tuple[AnalysisBlock | str, ...] = (
        AnalysisBlock.PROFILING,
        AnalysisBlock.UNIVARIATE,
    )
    manova_responses: tuple[str, ...] = ()
    manova_factors: tuple[str, ...] = ()
    manova_covariates: tuple[str, ...] = ()
    factorial_response: str | None = None
    factorial_factors: tuple[str, ...] = ()
    factorial_covariates: tuple[str, ...] = ()
    factorial_interactions: tuple[tuple[str, ...], ...] = ()
    factorial_formula: str | None = None

    # Descriptive controls.
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    min_numeric_n: int = 3
    max_category_levels: int = 50
    max_missingness_patterns: int = 20
    max_pairwise_columns: int = 200

    # Bivariate controls.
    correlations: tuple[CorrelationMethod | str, ...] = (
        CorrelationMethod.PEARSON,
        CorrelationMethod.SPEARMAN,
        CorrelationMethod.KENDALL,
    )
    comparison_tests: tuple[ComparisonTest | str, ...] = (
        ComparisonTest.WELCH_T,
        ComparisonTest.MANN_WHITNEY,
        ComparisonTest.WELCH_ANOVA,
        ComparisonTest.KRUSKAL_WALLIS,
        ComparisonTest.CHI_SQUARE,
        ComparisonTest.FISHER_EXACT,
    )
    p_adjust: PAdjustMethod | str = PAdjustMethod.FDR_BH
    min_group_n: int = 3
    min_correlation_pairs: int = 3
    max_group_levels: int = 20
    max_correlation_columns: int = 100
    pairwise: bool = False

    # Univariate outlier controls.
    outlier_methods: tuple[OutlierMethod | str, ...] = (
        OutlierMethod.IQR,
        OutlierMethod.ROBUST_Z,
    )
    outlier_iqr_multiplier: float = 1.5
    outlier_robust_z_threshold: float = 3.5
    include_outlier_flags: bool = False

    # Feature-space controls.
    projection_scaling: ScalingMethod | str = ScalingMethod.NONE
    projection_n_components: int = 2
    projection_metric: str = "euclidean"
    tsne_perplexity: float = 30.0
    tsne_max_iter: int = 1000
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1

    # Multivariate statistical controls.
    multivariate_scaling: ScalingMethod | str = ScalingMethod.NONE
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

    # Factorial ANOVA/ANCOVA controls.
    factorial_ss_type: int = 2
    factorial_p_adjust: PAdjustMethod | str = PAdjustMethod.NONE
    factorial_robust_covariance: str | None = None
    factorial_min_cell_n: int = 2
    factorial_max_factor_levels: int = 20
    factorial_max_design_cells: int = 5000
    factorial_max_design_columns: int = 500
    factorial_max_interaction_order: int = 3
    factorial_diagnostic_alpha: float = 0.05
    factorial_condition_number_threshold: float = 30.0

    # Diagnostics, dependence, contingency and interval controls.
    distribution_max_shapiro_n: int = 5000
    dependence_partial_covariates: tuple[str, ...] = ()
    dependence_min_complete_pairs: int = 5
    dependence_max_columns: int = 50
    dependence_n_permutations: int = 199
    dependence_mi_neighbors: int = 3
    confidence_level: float = 0.95
    interval_correlations: tuple[CorrelationMethod | str, ...] = (CorrelationMethod.PEARSON,)
    bootstrap_resamples: int = 1000
    bootstrap_method: str = "bca"

    # PERMANOVA, post-hoc, marginal-means and mixed-effects controls.
    permanova_factor: str | None = None
    permanova_metric: str = "euclidean"
    permanova_permutations: int = 999
    permanova_min_group_n: int = 3
    permanova_max_group_levels: int = 20

    posthoc_response: str | None = None
    posthoc_factor: str | None = None
    posthoc_methods: tuple[str, ...] = ("tukey_hsd", "games_howell")

    marginal_response: str | None = None
    marginal_factors: tuple[str, ...] = ()
    marginal_covariates: tuple[str, ...] = ()
    marginal_interactions: tuple[tuple[str, ...], ...] = ()
    marginal_formula: str | None = None
    marginal_terms: tuple[tuple[str, ...], ...] = ()
    marginal_p_adjust: PAdjustMethod | str = PAdjustMethod.FDR_BH

    mixed_group: str | None = None
    mixed_response: str | None = None
    mixed_factors: tuple[str, ...] = ()
    mixed_covariates: tuple[str, ...] = ()
    mixed_interactions: tuple[tuple[str, ...], ...] = ()
    mixed_formula: str | None = None
    mixed_random_slopes: tuple[str, ...] = ()
    mixed_reml: bool = True
    mixed_optimizer: str = "lbfgs"
    mixed_max_iter: int = 1000
    mixed_min_groups: int = 3
    mixed_min_group_n: int = 2

    # Representation, compositional, Bayesian and anomaly controls.
    representation_alignment: AlignmentMode | str = AlignmentMode.STRICT
    representation_cca_components: int = 2
    representation_cca_scaling: ScalingMethod | str = ScalingMethod.STANDARD
    representation_cca_max_iter: int = 1000
    representation_cca_tol: float = 1e-6
    representation_distance_metric: str = "euclidean"
    representation_distance_similarity_method: str = "spearman"
    representation_mantel_permutations: int = 999

    compositional_transform: str = "clr"
    compositional_replace_zeros: bool = False
    compositional_zero_replacement_fraction: float = 0.65
    compositional_alr_denominator: int = -1

    bayesian_variables: tuple[str, ...] = ()
    bayesian_groups: tuple[str, ...] = ()
    bayesian_credible_level: float = 0.95
    bayesian_rope: tuple[float, float] = (-0.1, 0.1)
    bayesian_draws: int = 5000
    bayesian_min_n: int = 3

    anomaly_methods: tuple[str, ...] = ("isolation_forest", "lof")
    anomaly_scaling: ScalingMethod | str = ScalingMethod.NONE
    anomaly_contamination: str | float = "auto"
    anomaly_isolation_estimators: int = 200
    anomaly_lof_neighbors: int = 20

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
        feature_mode = (
            self.feature_alignment
            if isinstance(self.feature_alignment, AlignmentMode)
            else AlignmentMode(self.feature_alignment)
        )
        representation_mode = (
            self.representation_alignment
            if isinstance(self.representation_alignment, AlignmentMode)
            else AlignmentMode(self.representation_alignment)
        )
        responses = tuple(str(column) for column in self.responses)
        groups = tuple(str(column) for column in self.groups)
        dependence_partial_covariates = tuple(str(column) for column in self.dependence_partial_covariates)
        posthoc_methods = tuple(str(value).strip().lower() for value in self.posthoc_methods)
        marginal_factors = tuple(str(column) for column in self.marginal_factors)
        marginal_covariates = tuple(str(column) for column in self.marginal_covariates)
        marginal_interactions = tuple(
            tuple(str(column) for column in term) for term in self.marginal_interactions
        )
        marginal_terms = tuple(tuple(str(column) for column in term) for term in self.marginal_terms)
        mixed_factors = tuple(str(column) for column in self.mixed_factors)
        mixed_covariates = tuple(str(column) for column in self.mixed_covariates)
        mixed_interactions = tuple(tuple(str(column) for column in term) for term in self.mixed_interactions)
        mixed_random_slopes = tuple(str(column) for column in self.mixed_random_slopes)
        bayesian_variables = tuple(str(column) for column in self.bayesian_variables)
        bayesian_groups = tuple(str(column) for column in self.bayesian_groups)
        anomaly_methods = tuple(str(value).strip().lower() for value in self.anomaly_methods)
        enabled_blocks = tuple(
            value if isinstance(value, AnalysisBlock) else AnalysisBlock(value)
            for value in self.enabled_blocks
        )
        manova_responses = tuple(str(column) for column in self.manova_responses)
        manova_factors = tuple(str(column) for column in self.manova_factors)
        manova_covariates = tuple(str(column) for column in self.manova_covariates)
        factorial_factors = tuple(str(column) for column in self.factorial_factors)
        factorial_covariates = tuple(str(column) for column in self.factorial_covariates)
        factorial_interactions = tuple(
            tuple(str(column) for column in interaction) for interaction in self.factorial_interactions
        )
        if not enabled_blocks:
            msg = "enabled_blocks cannot be empty."
            raise ValueError(msg)
        if len(set(enabled_blocks)) != len(enabled_blocks):
            msg = "enabled_blocks cannot contain duplicates."
            raise ValueError(msg)
        for label, values in (
            ("manova_responses", manova_responses),
            ("manova_factors", manova_factors),
            ("manova_covariates", manova_covariates),
            ("factorial_factors", factorial_factors),
            ("factorial_covariates", factorial_covariates),
        ):
            if len(set(values)) != len(values):
                msg = f"{label} cannot contain duplicates."
                raise ValueError(msg)
        if any(len(term) < 2 for term in factorial_interactions):
            msg = "factorial_interactions must contain at least two predictors per interaction."
            raise ValueError(msg)
        if len(set(factorial_interactions)) != len(factorial_interactions):
            msg = "factorial_interactions cannot contain duplicates."
            raise ValueError(msg)
        if self.factorial_formula is not None and not str(self.factorial_formula).strip():
            msg = "factorial_formula cannot be empty."
            raise ValueError(msg)
        if self.factorial_response is not None and not str(self.factorial_response).strip():
            msg = "factorial_response cannot be empty."
            raise ValueError(msg)
        if len(set(responses)) != len(responses):
            msg = "responses cannot contain duplicates."
            raise ValueError(msg)
        if len(set(groups)) != len(groups):
            msg = "groups cannot contain duplicates."
            raise ValueError(msg)
        if len(set(dependence_partial_covariates)) != len(dependence_partial_covariates):
            msg = "dependence_partial_covariates cannot contain duplicates."
            raise ValueError(msg)
        overlap = sorted(set(responses) & set(groups))
        if overlap:
            msg = f"Columns cannot be configured as both response and group: {overlap}."
            raise ValueError(msg)
        quantiles = validate_quantiles(tuple(self.quantiles))
        correlations = tuple(
            value if isinstance(value, CorrelationMethod) else CorrelationMethod(value)
            for value in self.correlations
        )
        interval_correlations = tuple(
            value if isinstance(value, CorrelationMethod) else CorrelationMethod(value)
            for value in self.interval_correlations
        )
        comparison_tests = tuple(
            value if isinstance(value, ComparisonTest) else ComparisonTest(value)
            for value in self.comparison_tests
        )
        p_adjust = self.p_adjust if isinstance(self.p_adjust, PAdjustMethod) else PAdjustMethod(self.p_adjust)
        factorial_p_adjust = (
            self.factorial_p_adjust
            if isinstance(self.factorial_p_adjust, PAdjustMethod)
            else PAdjustMethod(self.factorial_p_adjust)
        )
        marginal_p_adjust = (
            self.marginal_p_adjust
            if isinstance(self.marginal_p_adjust, PAdjustMethod)
            else PAdjustMethod(self.marginal_p_adjust)
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
        representation_cca_scaling = (
            self.representation_cca_scaling
            if isinstance(self.representation_cca_scaling, ScalingMethod)
            else ScalingMethod(self.representation_cca_scaling)
        )
        anomaly_scaling = (
            self.anomaly_scaling
            if isinstance(self.anomaly_scaling, ScalingMethod)
            else ScalingMethod(self.anomaly_scaling)
        )
        outlier_methods = tuple(
            value if isinstance(value, OutlierMethod) else OutlierMethod(value)
            for value in self.outlier_methods
        )
        if len(set(correlations)) != len(correlations):
            msg = "correlations cannot contain duplicates."
            raise ValueError(msg)
        if len(set(interval_correlations)) != len(interval_correlations):
            msg = "interval_correlations cannot contain duplicates."
            raise ValueError(msg)
        if len(set(comparison_tests)) != len(comparison_tests):
            msg = "comparison_tests cannot contain duplicates."
            raise ValueError(msg)
        if not outlier_methods:
            msg = "outlier_methods cannot be empty."
            raise ValueError(msg)
        if len(set(outlier_methods)) != len(outlier_methods):
            msg = "outlier_methods cannot contain duplicates."
            raise ValueError(msg)
        if self.outlier_iqr_multiplier <= 0:
            msg = "outlier_iqr_multiplier must be greater than zero."
            raise ValueError(msg)
        if self.outlier_robust_z_threshold <= 0:
            msg = "outlier_robust_z_threshold must be greater than zero."
            raise ValueError(msg)
        if self.projection_n_components < 1:
            msg = "projection_n_components must be at least 1."
            raise ValueError(msg)
        if self.tsne_perplexity <= 0:
            msg = "tsne_perplexity must be greater than zero."
            raise ValueError(msg)
        if self.tsne_max_iter < 250:
            msg = "tsne_max_iter must be at least 250."
            raise ValueError(msg)
        if self.umap_n_neighbors < 2:
            msg = "umap_n_neighbors must be at least 2."
            raise ValueError(msg)
        if not 0.0 <= self.umap_min_dist <= 1.0:
            msg = "umap_min_dist must be between 0 and 1."
            raise ValueError(msg)
        if not str(self.projection_metric).strip():
            msg = "projection_metric cannot be empty."
            raise ValueError(msg)
        if self.multivariate_max_covariance_features < 2:
            msg = "multivariate_max_covariance_features must be at least 2."
            raise ValueError(msg)
        if self.multivariate_max_collinearity_features < 2:
            msg = "multivariate_max_collinearity_features must be at least 2."
            raise ValueError(msg)
        if self.multivariate_max_mahalanobis_features < 1:
            msg = "multivariate_max_mahalanobis_features must be at least 1."
            raise ValueError(msg)
        if not 0.5 < self.mahalanobis_threshold_quantile < 1.0:
            msg = "mahalanobis_threshold_quantile must be between 0.5 and 1.0."
            raise ValueError(msg)
        if (
            self.mahalanobis_robust_support_fraction is not None
            and not 0.5 <= self.mahalanobis_robust_support_fraction <= 1.0
        ):
            msg = "mahalanobis_robust_support_fraction must be between 0.5 and 1.0."
            raise ValueError(msg)
        if self.manova_max_responses < 2:
            msg = "manova_max_responses must be at least 2."
            raise ValueError(msg)
        if self.manova_max_factor_levels < 2:
            msg = "manova_max_factor_levels must be at least 2."
            raise ValueError(msg)
        if self.manova_min_level_n < 2:
            msg = "manova_min_level_n must be at least 2."
            raise ValueError(msg)
        if self.factorial_ss_type not in {2, 3}:
            msg = "factorial_ss_type must be 2 or 3."
            raise ValueError(msg)
        if self.factorial_robust_covariance is not None:
            robust = str(self.factorial_robust_covariance).strip().lower()
            if robust in {"", "none", "nonrobust"}:
                robust = None
            elif robust not in {"hc0", "hc1", "hc2", "hc3"}:
                msg = "factorial_robust_covariance must be one of hc0, hc1, hc2, hc3, or None."
                raise ValueError(msg)
        else:
            robust = None
        if self.factorial_min_cell_n < 1:
            msg = "factorial_min_cell_n must be at least 1."
            raise ValueError(msg)
        if self.factorial_max_factor_levels < 2:
            msg = "factorial_max_factor_levels must be at least 2."
            raise ValueError(msg)
        if self.factorial_max_design_cells < 1:
            msg = "factorial_max_design_cells must be at least 1."
            raise ValueError(msg)
        if self.factorial_max_design_columns < 2:
            msg = "factorial_max_design_columns must be at least 2."
            raise ValueError(msg)
        if self.factorial_max_interaction_order < 2:
            msg = "factorial_max_interaction_order must be at least 2."
            raise ValueError(msg)
        if not 0.0 < self.factorial_diagnostic_alpha < 1.0:
            msg = "factorial_diagnostic_alpha must lie in (0, 1)."
            raise ValueError(msg)
        if self.factorial_condition_number_threshold <= 0.0:
            msg = "factorial_condition_number_threshold must be greater than zero."
            raise ValueError(msg)
        if self.distribution_max_shapiro_n < 3:
            msg = "distribution_max_shapiro_n must be at least 3."
            raise ValueError(msg)
        if self.dependence_min_complete_pairs < 3:
            msg = "dependence_min_complete_pairs must be at least 3."
            raise ValueError(msg)
        if self.dependence_max_columns < 2:
            msg = "dependence_max_columns must be at least 2."
            raise ValueError(msg)
        if self.dependence_n_permutations < 0:
            msg = "dependence_n_permutations cannot be negative."
            raise ValueError(msg)
        if self.dependence_mi_neighbors < 1:
            msg = "dependence_mi_neighbors must be at least 1."
            raise ValueError(msg)
        if not 0.0 < self.confidence_level < 1.0:
            msg = "confidence_level must lie in (0, 1)."
            raise ValueError(msg)
        if self.bootstrap_resamples < 100:
            msg = "bootstrap_resamples must be at least 100."
            raise ValueError(msg)
        bootstrap_method = str(self.bootstrap_method).strip().lower()
        if bootstrap_method not in {"percentile", "basic", "bca"}:
            msg = "bootstrap_method must be percentile, basic, or bca."
            raise ValueError(msg)
        if self.permanova_factor is not None and not str(self.permanova_factor).strip():
            msg = "permanova_factor cannot be empty."
            raise ValueError(msg)
        if not str(self.permanova_metric).strip():
            msg = "permanova_metric cannot be empty."
            raise ValueError(msg)
        if self.permanova_permutations < 0:
            msg = "permanova_permutations cannot be negative."
            raise ValueError(msg)
        if self.permanova_min_group_n < 2:
            msg = "permanova_min_group_n must be at least 2."
            raise ValueError(msg)
        if self.permanova_max_group_levels < 2:
            msg = "permanova_max_group_levels must be at least 2."
            raise ValueError(msg)
        if not posthoc_methods or any(
            value not in {"tukey_hsd", "games_howell"} for value in posthoc_methods
        ):
            msg = "posthoc_methods must contain tukey_hsd and/or games_howell."
            raise ValueError(msg)
        if len(set(posthoc_methods)) != len(posthoc_methods):
            msg = "posthoc_methods cannot contain duplicates."
            raise ValueError(msg)
        for label, values in (
            ("marginal_factors", marginal_factors),
            ("marginal_covariates", marginal_covariates),
            ("mixed_factors", mixed_factors),
            ("mixed_covariates", mixed_covariates),
            ("mixed_random_slopes", mixed_random_slopes),
        ):
            if len(set(values)) != len(values):
                msg = f"{label} cannot contain duplicates."
                raise ValueError(msg)
        for label, values in (
            ("marginal_interactions", marginal_interactions),
            ("marginal_terms", marginal_terms),
            ("mixed_interactions", mixed_interactions),
        ):
            if any(len(term) < 1 for term in values) or len(set(values)) != len(values):
                msg = f"{label} contains invalid or duplicate terms."
                raise ValueError(msg)
        if self.marginal_formula is not None and not str(self.marginal_formula).strip():
            msg = "marginal_formula cannot be empty."
            raise ValueError(msg)
        if self.mixed_formula is not None and not str(self.mixed_formula).strip():
            msg = "mixed_formula cannot be empty."
            raise ValueError(msg)
        if self.mixed_group is not None and not str(self.mixed_group).strip():
            msg = "mixed_group cannot be empty."
            raise ValueError(msg)
        if self.mixed_optimizer.strip() == "":
            msg = "mixed_optimizer cannot be empty."
            raise ValueError(msg)
        if self.mixed_max_iter < 1 or self.mixed_min_groups < 2 or self.mixed_min_group_n < 1:
            msg = "Invalid mixed-effects iteration/group controls."
            raise ValueError(msg)
        if (
            self.representation_cca_components < 1
            or self.representation_cca_max_iter < 1
            or self.representation_cca_tol <= 0
        ):
            msg = "Invalid CCA controls."
            raise ValueError(msg)
        if not str(self.representation_distance_metric).strip():
            msg = "representation_distance_metric cannot be empty."
            raise ValueError(msg)
        if self.representation_distance_similarity_method not in {"pearson", "spearman"}:
            msg = "representation_distance_similarity_method must be pearson or spearman."
            raise ValueError(msg)
        if self.representation_mantel_permutations < 0:
            msg = "representation_mantel_permutations cannot be negative."
            raise ValueError(msg)
        if str(self.compositional_transform).lower() not in {"clr", "alr", "ilr"}:
            msg = "compositional_transform must be clr, alr, or ilr."
            raise ValueError(msg)
        if not 0 < self.compositional_zero_replacement_fraction < 1:
            msg = "compositional_zero_replacement_fraction must lie in (0, 1)."
            raise ValueError(msg)
        if len(set(bayesian_variables)) != len(bayesian_variables) or len(set(bayesian_groups)) != len(
            bayesian_groups
        ):
            msg = "Bayesian variables/groups cannot contain duplicates."
            raise ValueError(msg)
        if not 0 < self.bayesian_credible_level < 1 or self.bayesian_draws < 100 or self.bayesian_min_n < 2:
            msg = "Invalid Bayesian EDA controls."
            raise ValueError(msg)
        if len(self.bayesian_rope) != 2 or self.bayesian_rope[0] > self.bayesian_rope[1]:
            msg = "bayesian_rope must be an ordered pair."
            raise ValueError(msg)
        if not anomaly_methods or any(v not in {"isolation_forest", "lof"} for v in anomaly_methods):
            msg = "anomaly_methods must contain isolation_forest and/or lof."
            raise ValueError(msg)
        if len(set(anomaly_methods)) != len(anomaly_methods):
            msg = "anomaly_methods cannot contain duplicates."
            raise ValueError(msg)
        if self.anomaly_isolation_estimators < 1 or self.anomaly_lof_neighbors < 2:
            msg = "Invalid anomaly controls."
            raise ValueError(msg)
        if isinstance(self.anomaly_contamination, float) and not 0 < self.anomaly_contamination <= 0.5:
            msg = "numeric anomaly_contamination must lie in (0, 0.5]."
            raise ValueError(msg)
        if not (self.anomaly_contamination == "auto" or isinstance(self.anomaly_contamination, float)):
            msg = "anomaly_contamination must be 'auto' or a float."
            raise ValueError(msg)

        if self.min_numeric_n < 2:
            msg = "min_numeric_n must be at least 2."
            raise ValueError(msg)
        if self.max_category_levels < 2:
            msg = "max_category_levels must be at least 2."
            raise ValueError(msg)
        if self.max_missingness_patterns < 1:
            msg = "max_missingness_patterns must be at least 1."
            raise ValueError(msg)
        if self.max_pairwise_columns < 1:
            msg = "max_pairwise_columns must be at least 1."
            raise ValueError(msg)
        if self.min_group_n < 2:
            msg = "min_group_n must be at least 2."
            raise ValueError(msg)
        if self.min_correlation_pairs < 2:
            msg = "min_correlation_pairs must be at least 2."
            raise ValueError(msg)
        if self.max_group_levels < 2:
            msg = "max_group_levels must be at least 2."
            raise ValueError(msg)
        if self.max_correlation_columns < 2:
            msg = "max_correlation_columns must be at least 2."
            raise ValueError(msg)

        object.__setattr__(self, "role_overrides", MappingProxyType(roles))
        object.__setattr__(self, "responses", responses)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "dependence_partial_covariates", dependence_partial_covariates)
        object.__setattr__(self, "bootstrap_method", bootstrap_method)
        object.__setattr__(self, "posthoc_methods", posthoc_methods)
        object.__setattr__(self, "marginal_factors", marginal_factors)
        object.__setattr__(self, "marginal_covariates", marginal_covariates)
        object.__setattr__(self, "marginal_interactions", marginal_interactions)
        object.__setattr__(self, "marginal_terms", marginal_terms)
        object.__setattr__(self, "marginal_p_adjust", marginal_p_adjust)
        object.__setattr__(self, "mixed_factors", mixed_factors)
        object.__setattr__(self, "mixed_covariates", mixed_covariates)
        object.__setattr__(self, "mixed_interactions", mixed_interactions)
        object.__setattr__(self, "mixed_random_slopes", mixed_random_slopes)
        object.__setattr__(
            self,
            "permanova_factor",
            None if self.permanova_factor is None else str(self.permanova_factor).strip(),
        )
        object.__setattr__(
            self,
            "posthoc_response",
            None if self.posthoc_response is None else str(self.posthoc_response).strip(),
        )
        object.__setattr__(
            self, "posthoc_factor", None if self.posthoc_factor is None else str(self.posthoc_factor).strip()
        )
        object.__setattr__(
            self,
            "marginal_response",
            None if self.marginal_response is None else str(self.marginal_response).strip(),
        )
        object.__setattr__(
            self,
            "marginal_formula",
            None if self.marginal_formula is None else str(self.marginal_formula).strip(),
        )
        object.__setattr__(
            self, "mixed_group", None if self.mixed_group is None else str(self.mixed_group).strip()
        )
        object.__setattr__(
            self, "mixed_response", None if self.mixed_response is None else str(self.mixed_response).strip()
        )
        object.__setattr__(
            self, "mixed_formula", None if self.mixed_formula is None else str(self.mixed_formula).strip()
        )
        object.__setattr__(self, "bayesian_variables", bayesian_variables)
        object.__setattr__(self, "bayesian_groups", bayesian_groups)
        object.__setattr__(self, "anomaly_methods", anomaly_methods)
        object.__setattr__(self, "representation_alignment", representation_mode)
        object.__setattr__(self, "representation_cca_scaling", representation_cca_scaling)
        object.__setattr__(self, "anomaly_scaling", anomaly_scaling)
        object.__setattr__(self, "compositional_transform", str(self.compositional_transform).lower())
        object.__setattr__(self, "enabled_blocks", enabled_blocks)
        object.__setattr__(self, "manova_responses", manova_responses)
        object.__setattr__(self, "manova_factors", manova_factors)
        object.__setattr__(self, "manova_covariates", manova_covariates)
        object.__setattr__(self, "factorial_factors", factorial_factors)
        object.__setattr__(self, "factorial_covariates", factorial_covariates)
        object.__setattr__(self, "factorial_interactions", factorial_interactions)
        object.__setattr__(
            self,
            "factorial_formula",
            None if self.factorial_formula is None else str(self.factorial_formula).strip(),
        )
        object.__setattr__(
            self,
            "factorial_response",
            None if self.factorial_response is None else str(self.factorial_response),
        )
        object.__setattr__(self, "kind_overrides", MappingProxyType(kinds))
        object.__setattr__(self, "annotation_alignment", mode)
        object.__setattr__(self, "feature_alignment", feature_mode)
        object.__setattr__(self, "quantiles", quantiles)
        object.__setattr__(self, "correlations", correlations)
        object.__setattr__(self, "interval_correlations", interval_correlations)
        object.__setattr__(self, "comparison_tests", comparison_tests)
        object.__setattr__(self, "p_adjust", p_adjust)
        object.__setattr__(self, "factorial_p_adjust", factorial_p_adjust)
        object.__setattr__(self, "factorial_robust_covariance", robust)
        object.__setattr__(self, "outlier_methods", outlier_methods)
        object.__setattr__(self, "projection_scaling", projection_scaling)
        object.__setattr__(self, "multivariate_scaling", multivariate_scaling)

    def dataset_kwargs(self) -> dict[str, Any]:
        """Return arguments accepted directly by :class:`TabularDataset`."""
        return {
            "id_column": self.id_column,
            "role_overrides": dict(self.role_overrides),
            "kind_overrides": dict(self.kind_overrides),
        }

    def profiling_kwargs(self) -> dict[str, Any]:
        """Return controls used by descriptive profiling."""
        return {
            "max_missingness_patterns": self.max_missingness_patterns,
            "max_pairwise_columns": self.max_pairwise_columns,
        }

    def univariate_kwargs(self) -> dict[str, Any]:
        """Return controls used by univariate descriptive analysis."""
        return {
            "quantiles": self.quantiles,
            "min_numeric_n": self.min_numeric_n,
            "max_category_levels": self.max_category_levels,
            **self.profiling_kwargs(),
        }

    def outlier_kwargs(self) -> dict[str, Any]:
        """Return controls used by univariate outlier diagnostics."""
        return {
            "methods": self.outlier_methods,
            "min_numeric_n": self.min_numeric_n,
            "iqr_multiplier": self.outlier_iqr_multiplier,
            "robust_z_threshold": self.outlier_robust_z_threshold,
            "include_flags": self.include_outlier_flags,
        }

    def pca_kwargs(self) -> dict[str, Any]:
        """Return controls used by PCA."""
        return {
            "n_components": self.projection_n_components,
            "scaling": self.projection_scaling,
            "random_state": self.random_state,
        }

    def tsne_kwargs(self) -> dict[str, Any]:
        """Return controls used by t-SNE."""
        return {
            "n_components": self.projection_n_components,
            "scaling": self.projection_scaling,
            "metric": self.projection_metric,
            "perplexity": self.tsne_perplexity,
            "random_state": self.random_state,
            "max_iter": self.tsne_max_iter,
        }

    def umap_kwargs(self) -> dict[str, Any]:
        """Return controls used by UMAP."""
        return {
            "n_components": self.projection_n_components,
            "scaling": self.projection_scaling,
            "metric": self.projection_metric,
            "n_neighbors": self.umap_n_neighbors,
            "min_dist": self.umap_min_dist,
            "random_state": self.random_state,
        }

    def multivariate_kwargs(self) -> dict[str, Any]:
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

    def manova_kwargs(self) -> dict[str, Any]:
        """Return generic dimensionality/sample guards used by MANOVA."""
        return {
            "max_responses": self.manova_max_responses,
            "max_factor_levels": self.manova_max_factor_levels,
            "min_level_n": self.manova_min_level_n,
        }

    def factorial_kwargs(self) -> dict[str, Any]:
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

    def distribution_diagnostics_kwargs(self) -> dict[str, Any]:
        """Return controls for standalone distribution diagnostics."""
        return {
            "responses": self.responses,
            "groups": self.groups,
            "p_adjust": self.p_adjust,
            "min_group_n": self.min_group_n,
            "max_group_levels": self.max_group_levels,
            "max_shapiro_n": self.distribution_max_shapiro_n,
        }

    def dependence_kwargs(self) -> dict[str, Any]:
        """Return controls for nonlinear and partial dependence analysis."""
        return {
            "partial_covariates": self.dependence_partial_covariates,
            "min_complete_pairs": self.dependence_min_complete_pairs,
            "max_columns": self.dependence_max_columns,
            "n_permutations": self.dependence_n_permutations,
            "mutual_information_neighbors": self.dependence_mi_neighbors,
            "p_adjust": self.p_adjust,
            "random_state": self.random_state,
        }

    def contingency_kwargs(self) -> dict[str, Any]:
        """Return controls for cell-level contingency diagnostics."""
        return {
            "max_category_levels": self.max_category_levels,
            "p_adjust": self.p_adjust,
        }

    def interval_kwargs(self) -> dict[str, Any]:
        """Return controls for common confidence-interval estimands."""
        return {
            "confidence_level": self.confidence_level,
            "correlation_methods": self.interval_correlations,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_method": self.bootstrap_method,
            "min_group_n": self.min_group_n,
            "max_category_levels": self.max_group_levels,
            "random_state": self.random_state,
        }

    def permanova_kwargs(self) -> dict[str, Any]:
        return {
            "metric": self.permanova_metric,
            "n_permutations": self.permanova_permutations,
            "random_state": self.random_state,
            "min_group_n": self.permanova_min_group_n,
            "max_group_levels": self.permanova_max_group_levels,
            "alignment": self.feature_alignment,
        }

    def posthoc_kwargs(self) -> dict[str, Any]:
        return {
            "methods": self.posthoc_methods,
            "confidence_level": self.confidence_level,
            "min_group_n": self.min_group_n,
            "max_group_levels": self.max_group_levels,
        }

    def marginal_means_kwargs(self) -> dict[str, Any]:
        return {
            "confidence_level": self.confidence_level,
            "p_adjust": self.marginal_p_adjust,
            "max_interaction_order": self.factorial_max_interaction_order,
        }

    def mixed_effects_kwargs(self) -> dict[str, Any]:
        return {
            "random_slopes": self.mixed_random_slopes,
            "reml": self.mixed_reml,
            "optimizer": self.mixed_optimizer,
            "max_iter": self.mixed_max_iter,
            "confidence_level": self.confidence_level,
            "min_groups": self.mixed_min_groups,
            "min_group_n": self.mixed_min_group_n,
            "max_interaction_order": self.factorial_max_interaction_order,
        }

    def representation_kwargs(self) -> dict[str, Any]:
        return {
            "alignment": self.representation_alignment,
            "cca_components": self.representation_cca_components,
            "cca_scaling": self.representation_cca_scaling,
            "cca_max_iter": self.representation_cca_max_iter,
            "cca_tol": self.representation_cca_tol,
            "distance_metric": self.representation_distance_metric,
            "distance_similarity_method": self.representation_distance_similarity_method,
            "mantel_permutations": self.representation_mantel_permutations,
            "random_state": self.random_state,
        }

    def compositional_kwargs(self) -> dict[str, Any]:
        return {
            "transform": self.compositional_transform,
            "replace_zeros": self.compositional_replace_zeros,
            "zero_replacement_fraction": self.compositional_zero_replacement_fraction,
            "alr_denominator": self.compositional_alr_denominator,
        }

    def bayesian_kwargs(self) -> dict[str, Any]:
        return {
            "variables": self.bayesian_variables or None,
            "groups": self.bayesian_groups or self.groups,
            "credible_level": self.bayesian_credible_level,
            "rope": self.bayesian_rope,
            "draws": self.bayesian_draws,
            "min_n": self.bayesian_min_n,
            "random_state": self.random_state,
        }

    def anomaly_kwargs(self) -> dict[str, Any]:
        return {
            "methods": self.anomaly_methods,
            "scaling": self.anomaly_scaling,
            "contamination": self.anomaly_contamination,
            "isolation_estimators": self.anomaly_isolation_estimators,
            "lof_neighbors": self.anomaly_lof_neighbors,
            "random_state": self.random_state,
        }

    def grouped_kwargs(self) -> dict[str, Any]:
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

    def bivariate_kwargs(self) -> dict[str, Any]:
        """Return controls used by mixed-type bivariate analysis."""
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
