"""Shared enumerations for Ruddy's scientific contracts."""

from __future__ import annotations

from enum import StrEnum


class ColumnRole(StrEnum):
    """Statistical role assigned to a tabular column."""

    IDENTIFIER = "identifier"
    VARIABLE = "variable"
    TARGET = "target"
    FACTOR = "factor"
    COVARIATE = "covariate"
    ANNOTATION = "annotation"
    EXCLUDED = "excluded"


class ColumnKind(StrEnum):
    """Observed data kind used by statistical dispatch."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    UNKNOWN = "unknown"


class AlignmentMode(StrEnum):
    """Policy used when aligning external annotations or metadata."""

    STRICT = "strict"
    PARTIAL = "partial"


class ResultStatus(StrEnum):
    """Scientific execution status for an analysis result."""

    OK = "ok"
    DEGENERATE = "degenerate"
    SKIPPED = "skipped"


class AdvisoryLevel(StrEnum):
    """Severity level for non-fatal scientific advisories."""

    INFO = "info"
    WARNING = "warning"
