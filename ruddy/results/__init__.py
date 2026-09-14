"""Structured analysis result contracts."""

from ruddy.results.base import Advisory, AnalysisResult
from ruddy.results.provenance import AnalysisProvenance
from ruddy.results.schemas import (
    RESULT_REASON_COLUMN,
    RESULT_STATUS_COLUMN,
    validate_result_table,
)

__all__ = [
    "Advisory",
    "AnalysisProvenance",
    "AnalysisResult",
    "RESULT_REASON_COLUMN",
    "RESULT_STATUS_COLUMN",
    "validate_result_table",
]
