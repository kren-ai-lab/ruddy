"""Bivariate associations and mixed-type inference."""

from ruddy.bivariate.analysis import BivariateResult, analyze_bivariate
from ruddy.bivariate.associations import summarize_categorical_associations
from ruddy.bivariate.comparisons import summarize_numeric_categorical_comparisons
from ruddy.bivariate.correlations import summarize_correlations
from ruddy.bivariate.contingency import ContingencyDiagnosticsResult, analyze_contingency_diagnostics
from ruddy.bivariate.dependence import (
    DependenceResult,
    analyze_dependence,
    distance_correlation,
    summarize_general_dependence,
    summarize_partial_correlations,
)
from ruddy.bivariate.posthoc import PosthocResult, analyze_posthoc
from ruddy.bivariate.groups import (
    GroupAnalysisResult,
    analyze_grouped_responses,
    resolve_groups,
    resolve_responses,
    summarize_annotation_coverage,
    summarize_group_coverage,
    summarize_grouped_categorical_responses,
    summarize_grouped_numeric_responses,
    summarize_response_catalog,
)

__all__ = [
    "BivariateResult",
    "ContingencyDiagnosticsResult",
    "DependenceResult",
    "GroupAnalysisResult",
    "PosthocResult",
    "analyze_bivariate",
    "analyze_contingency_diagnostics",
    "analyze_dependence",
    "analyze_grouped_responses",
    "analyze_posthoc",
    "resolve_groups",
    "resolve_responses",
    "summarize_annotation_coverage",
    "summarize_group_coverage",
    "summarize_grouped_categorical_responses",
    "summarize_grouped_numeric_responses",
    "summarize_response_catalog",
    "distance_correlation",
    "summarize_categorical_associations",
    "summarize_general_dependence",
    "summarize_partial_correlations",
    "summarize_correlations",
    "summarize_numeric_categorical_comparisons",
]
