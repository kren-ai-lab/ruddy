"""Ruddy: domain-agnostic statistical exploratory data analysis."""

from ruddy._version import __version__
from ruddy.analysis import AnalysisConfig
from ruddy.core import AlignmentMode, ColumnKind, ColumnRole, ResultStatus
from ruddy.data import FeatureMatrix, TabularDataset
from ruddy.profiling import (
    ProfilingResult,
    pairwise_completeness,
    profile_columns,
    profile_dataset,
    summarize_missingness,
    summarize_missingness_patterns,
    summarize_overview,
)
from ruddy.results import Advisory, AnalysisProvenance, AnalysisResult
from ruddy.univariate import (
    UnivariateResult,
    UnivariateTables,
    analyze_univariate,
    summarize_categorical_statistics,
    summarize_datetime_statistics,
    summarize_numeric_statistics,
    summarize_univariate,
)

__all__ = [
    "Advisory",
    "AlignmentMode",
    "AnalysisConfig",
    "AnalysisProvenance",
    "AnalysisResult",
    "ColumnKind",
    "ColumnRole",
    "FeatureMatrix",
    "ProfilingResult",
    "ResultStatus",
    "TabularDataset",
    "UnivariateResult",
    "UnivariateTables",
    "analyze_univariate",
    "pairwise_completeness",
    "profile_columns",
    "profile_dataset",
    "summarize_categorical_statistics",
    "summarize_datetime_statistics",
    "summarize_missingness",
    "summarize_missingness_patterns",
    "summarize_numeric_statistics",
    "summarize_overview",
    "summarize_univariate",
    "__version__",
]
