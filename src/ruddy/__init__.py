"""Ruddy: domain-agnostic statistical exploratory data analysis."""

from ruddy._version import __version__
from ruddy.analysis import AnalysisConfig
from ruddy.core import AlignmentMode, ColumnKind, ColumnRole, ResultStatus
from ruddy.data import FeatureMatrix, TabularDataset
from ruddy.results import Advisory, AnalysisProvenance, AnalysisResult

__all__ = [
    "Advisory",
    "AlignmentMode",
    "AnalysisConfig",
    "AnalysisProvenance",
    "AnalysisResult",
    "ColumnKind",
    "ColumnRole",
    "FeatureMatrix",
    "ResultStatus",
    "TabularDataset",
    "__version__",
]
