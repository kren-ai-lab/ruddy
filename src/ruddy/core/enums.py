"""Shared enumerations for Ruddy's scientific contracts."""

from __future__ import annotations

from enum import StrEnum


class ColumnRole(StrEnum):
    """Statistical role assigned to a tabular column."""

    IDENTIFIER = "identifier"
    VARIABLE = "variable"
    RESPONSE = "response"
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


class AnnotationCoverage(StrEnum):
    """Observed coverage state for an external annotation source."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    ABSENT = "absent"


class ResultStatus(StrEnum):
    """Scientific execution status for an analysis result."""

    OK = "ok"
    DEGENERATE = "degenerate"
    SKIPPED = "skipped"


class AdvisoryLevel(StrEnum):
    """Severity level for non-fatal scientific advisories."""

    INFO = "info"
    WARNING = "warning"


class CorrelationMethod(StrEnum):
    """Numeric correlation methods supported by Ruddy."""

    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class ComparisonTest(StrEnum):
    """Inferential tests supported by Ruddy bivariate analysis."""

    WELCH_T = "welch_t"
    MANN_WHITNEY = "mann_whitney"
    WELCH_ANOVA = "welch_anova"
    KRUSKAL_WALLIS = "kruskal_wallis"
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"


class ScalingMethod(StrEnum):
    """Explicit feature scaling policies for projection analyses."""

    NONE = "none"
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"


class ProjectionMethod(StrEnum):
    """Supported nonlinear exploratory projection methods."""

    UMAP = "umap"
    TSNE = "tsne"


class OutlierMethod(StrEnum):
    """Univariate statistical outlier rules supported by Ruddy."""

    IQR = "iqr"
    ROBUST_Z = "robust_z"


class PAdjustMethod(StrEnum):
    """Multiple-testing correction policies."""

    NONE = "none"
    FDR_BH = "fdr_bh"
