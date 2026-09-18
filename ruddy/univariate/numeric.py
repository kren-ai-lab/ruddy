"""Deterministic numerical univariate statistics.

Skewness and kurtosis use the adjusted Fisher–Pearson G1 and unbiased excess G2 estimators, matching the formulas historically provided by pandas.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from polars._typing import PolarsDataType
from scipy import stats

from ruddy.data import TabularDataset
from ruddy.univariate.categorical import _safe_float

NUMERIC_STATISTICS_BASE_COLUMNS: tuple[str, ...] = (
    "column",
    "role",
    "n_total",
    "n_finite",
    "n_missing",
    "n_non_finite",
    "mean",
    "std",
    "variance",
    "min",
    "max",
    "range",
    "median",
    "iqr",
    "mad",
    "skewness",
    "kurtosis",
    "zero_count",
    "zero_fraction",
    "status",
    "reason",
)


def quantile_column_name(quantile: float) -> str:
    """Return a stable output column name for a configured quantile."""
    percent = float(quantile) * 100.0
    rounded = round(percent)
    if abs(percent - rounded) < 1e-10:
        integer = int(rounded)
        return f"q{integer:02d}" if integer < 10 else f"q{integer}"
    label = format(percent, ".10g").replace(".", "_").replace("-", "m")
    return f"q{label}"


def numeric_statistics_columns(quantiles: tuple[float, ...]) -> tuple[str, ...]:
    """Return the deterministic numerical statistics schema."""
    quantile_columns = tuple(quantile_column_name(value) for value in quantiles)
    prefix = NUMERIC_STATISTICS_BASE_COLUMNS[:12]
    suffix = NUMERIC_STATISTICS_BASE_COLUMNS[12:]
    return (*prefix, *quantile_columns, *suffix)


def numeric_statistics_schema(quantiles: tuple[float, ...]) -> dict[str, PolarsDataType]:
    """Return the Polars schema dictionary for numerical statistics."""
    schema: dict[str, PolarsDataType] = {}
    for col in numeric_statistics_columns(quantiles):
        if col in {"column", "role", "status", "reason"}:
            schema[col] = pl.String
        elif col.startswith("n_") or col == "zero_count":
            schema[col] = pl.Int64
        else:
            schema[col] = pl.Float64
    return schema


def validate_quantiles(quantiles: tuple[float, ...]) -> tuple[float, ...]:
    values = tuple(float(value) for value in quantiles)
    if not values:
        raise ValueError("At least one quantile must be configured.")
    if any(value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("Quantiles must lie strictly between 0 and 1.")
    if len(set(values)) != len(values):
        raise ValueError("Quantiles cannot contain duplicates.")
    if values != tuple(sorted(values)):
        raise ValueError("Quantiles must be sorted in ascending order.")
    if not {0.25, 0.50, 0.75}.issubset(set(values)):
        raise ValueError("Quantiles must include 0.25, 0.50, and 0.75.")
    return values


def _finite_numeric_values(series: pl.Series) -> np.ndarray:
    values = series.drop_nulls().cast(pl.Float64).to_numpy()
    return values[np.isfinite(values)]


def _status(
    *, n_present: int, n_finite: int, is_constant: bool, min_numeric_n: int
) -> tuple[str, str | None]:
    if n_present == 0:
        return "skipped", "all_missing"
    if n_finite == 0:
        return "skipped", "no_finite_values"
    if n_finite < min_numeric_n:
        return "skipped", "insufficient_finite_observations"
    if is_constant:
        return "degenerate", "constant"
    return "ok", None


def _numeric_row(
    *,
    column: str,
    role: str,
    series: pl.Series,
    quantiles: tuple[float, ...],
    min_numeric_n: int,
) -> dict[str, Any]:
    values = _finite_numeric_values(series)
    n_total = series.len()
    n_missing = series.null_count()
    if series.dtype.is_float():
        nan_sum = series.is_nan().sum()
        if nan_sum is not None:
            n_missing += int(nan_sum)
    n_present = n_total - n_missing
    n_finite = int(values.size)
    n_non_finite = int(n_present - n_finite)
    is_constant = bool(n_finite > 0 and np.unique(values).size == 1)
    status, reason = _status(
        n_present=n_present,
        n_finite=n_finite,
        is_constant=is_constant,
        min_numeric_n=min_numeric_n,
    )

    row: dict[str, Any] = {
        "column": column,
        "role": role,
        "n_total": n_total,
        "n_finite": n_finite,
        "n_missing": n_missing,
        "n_non_finite": n_non_finite,
        "mean": None,
        "std": None,
        "variance": None,
        "min": None,
        "max": None,
        "range": None,
        "median": None,
        "iqr": None,
        "mad": None,
        "skewness": None,
        "kurtosis": None,
        "zero_count": 0,
        "zero_fraction": None,
        "status": status,
        "reason": reason,
    }
    for quantile in quantiles:
        row[quantile_column_name(quantile)] = None

    if n_finite == 0:
        return row

    minimum = _safe_float(np.min(values))
    maximum = _safe_float(np.max(values))
    row["mean"] = _safe_float(np.mean(values))
    row["min"] = minimum
    row["max"] = maximum
    if minimum is not None and maximum is not None:
        row["range"] = _safe_float(maximum - minimum)

    quantile_values = np.quantile(values, quantiles)
    quantile_lookup: dict[float, float] = {}
    for quantile, value in zip(quantiles, quantile_values, strict=True):
        converted = _safe_float(value)
        row[quantile_column_name(quantile)] = converted
        if converted is not None:
            quantile_lookup[float(quantile)] = converted

    median = quantile_lookup.get(0.50)
    q25 = quantile_lookup.get(0.25)
    q75 = quantile_lookup.get(0.75)
    row["median"] = median
    if q25 is not None and q75 is not None:
        row["iqr"] = _safe_float(q75 - q25)
    if median is not None:
        row["mad"] = _safe_float(np.median(np.abs(values - median)))

    zero_count = int(np.count_nonzero(values == 0.0))
    row["zero_count"] = zero_count
    row["zero_fraction"] = float(zero_count / n_finite)

    if n_finite >= 2:
        variance = _safe_float(np.var(values, ddof=1))
        row["variance"] = variance
        row["std"] = _safe_float(np.std(values, ddof=1))
    if not is_constant and n_finite >= 3:
        row["skewness"] = _safe_float(stats.skew(values, bias=False))
    if not is_constant and n_finite >= 4:
        row["kurtosis"] = _safe_float(stats.kurtosis(values, bias=False))
    return row


def summarize_numeric_statistics(
    dataset: TabularDataset,
    columns: pl.DataFrame,
    *,
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
    min_numeric_n: int = 3,
) -> pl.DataFrame:
    """Summarize eligible numerical variables using finite observations only."""
    quantiles = validate_quantiles(quantiles)
    if min_numeric_n < 2:
        raise ValueError("min_numeric_n must be at least 2.")

    selected = columns.filter(
        pl.col("analysis_eligible") & (pl.col("role") != "factor") & (pl.col("data_kind") == "numeric")
    )
    rows = [
        _numeric_row(
            column=str(profile["column"]),
            role=str(profile["role"]),
            series=dataset.frame.get_column(str(profile["column"])),
            quantiles=quantiles,
            min_numeric_n=min_numeric_n,
        )
        for profile in selected.iter_rows(named=True)
    ]
    return pl.DataFrame(rows, schema=numeric_statistics_schema(quantiles))
