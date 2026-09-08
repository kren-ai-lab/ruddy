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
        object.__setattr__(self, "outlier_methods", outlier_methods)

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
