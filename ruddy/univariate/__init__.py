"""Univariate descriptive statistics."""

from ruddy.univariate.categorical import (
    CATEGORICAL_FREQUENCY_COLUMNS,
    CATEGORICAL_STATISTICS_COLUMNS,
    summarize_categorical_statistics,
)
from ruddy.univariate.diagnostics import (
    DistributionDiagnosticsResult,
    analyze_distribution_diagnostics,
    summarize_dispersion_diagnostics,
    summarize_normality_diagnostics,
)
from ruddy.univariate.distributions import (
    DATETIME_STATISTICS_COLUMNS,
    UnivariateResult,
    UnivariateTables,
    analyze_univariate,
    summarize_datetime_statistics,
    summarize_univariate,
)
from ruddy.univariate.numeric import (
    NUMERIC_STATISTICS_BASE_COLUMNS,
    numeric_statistics_columns,
    quantile_column_name,
    summarize_numeric_statistics,
    validate_quantiles,
)
from ruddy.univariate.outliers import (
    NUMERIC_QUALITY_COLUMNS,
    OUTLIER_FLAG_COLUMNS,
    OUTLIER_SUMMARY_COLUMNS,
    OutlierResult,
    analyze_outliers,
    summarize_numeric_quality,
    summarize_outliers,
)

__all__ = [
    "CATEGORICAL_FREQUENCY_COLUMNS",
    "CATEGORICAL_STATISTICS_COLUMNS",
    "DATETIME_STATISTICS_COLUMNS",
    "NUMERIC_QUALITY_COLUMNS",
    "NUMERIC_STATISTICS_BASE_COLUMNS",
    "OUTLIER_FLAG_COLUMNS",
    "OUTLIER_SUMMARY_COLUMNS",
    "DistributionDiagnosticsResult",
    "OutlierResult",
    "UnivariateResult",
    "UnivariateTables",
    "analyze_distribution_diagnostics",
    "analyze_outliers",
    "analyze_univariate",
    "numeric_statistics_columns",
    "quantile_column_name",
    "summarize_categorical_statistics",
    "summarize_datetime_statistics",
    "summarize_dispersion_diagnostics",
    "summarize_normality_diagnostics",
    "summarize_numeric_quality",
    "summarize_numeric_statistics",
    "summarize_outliers",
    "summarize_univariate",
    "validate_quantiles",
]
