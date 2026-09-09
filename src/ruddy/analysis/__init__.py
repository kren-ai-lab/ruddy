"""High-level analysis configuration and orchestration."""

from ruddy.analysis.api import analyze, analyze_groups
from ruddy.analysis.config import AnalysisConfig
from ruddy.analysis.intervals import ConfidenceIntervalResult, analyze_confidence_intervals
from ruddy.analysis.results import UnifiedAnalysisResult

__all__ = [
    "AnalysisConfig",
    "ConfidenceIntervalResult",
    "UnifiedAnalysisResult",
    "analyze",
    "analyze_confidence_intervals",
    "analyze_groups",
]
