"""Univariate orchestration and datetime descriptive statistics."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, cast

import polars as pl
from polars._typing import PolarsDataType

from ruddy.data import TabularDataset
from ruddy.profiling import ProfilingResult, profile_dataset
from ruddy.results import AnalysisProvenance
from ruddy.univariate.categorical import summarize_categorical_statistics
from ruddy.univariate.numeric import summarize_numeric_statistics, validate_quantiles

DATETIME_STATISTICS_SCHEMA: dict[str, PolarsDataType] = {
    "column": pl.String,
    "role": pl.String,
    "n_total": pl.Int64,
    "n_present": pl.Int64,
    "n_missing": pl.Int64,
    "min": pl.Datetime("us"),
    "max": pl.Datetime("us"),
    "range_seconds": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}

DATETIME_STATISTICS_COLUMNS: tuple[str, ...] = tuple(DATETIME_STATISTICS_SCHEMA)


@dataclass(frozen=True, slots=True)
class UnivariateTables:
    numeric_statistics: pl.DataFrame
    categorical_statistics: pl.DataFrame
    categorical_frequencies: pl.DataFrame
    datetime_statistics: pl.DataFrame


@dataclass(frozen=True, slots=True)
class UnivariateResult:
    profiling: ProfilingResult
    numeric_statistics: pl.DataFrame
    categorical_statistics: pl.DataFrame
    categorical_frequencies: pl.DataFrame
    datetime_statistics: pl.DataFrame
    provenance: AnalysisProvenance


def summarize_datetime_statistics(
    dataset: TabularDataset,
    columns: pl.DataFrame,
) -> pl.DataFrame:
    """Summarize eligible datetime variables without time-series interpretation."""
    selected = columns.filter(
        pl.col("analysis_eligible") & (pl.col("role") != "factor") & (pl.col("data_kind") == "datetime")
    )
    rows: list[dict[str, Any]] = []
    for profile in selected.iter_rows(named=True):
        column = str(profile["column"])
        role = str(profile["role"])
        series = dataset.frame.get_column(column)
        n_total = series.len()
        present = series.drop_nulls()
        if present.dtype != pl.Datetime("us"):
            present = present.cast(pl.Datetime("us"))
        n_present = present.len()
        n_missing = n_total - n_present
        if n_present == 0:
            minimum = None
            maximum = None
            status, reason = "skipped", "all_missing"
            range_seconds = None
        elif present.n_unique() == 1:
            minimum = present.min()
            maximum = present.max()
            status, reason = "degenerate", "constant"
            range_seconds = 0.0
        else:
            minimum = cast("datetime.datetime", present.min())
            maximum = cast("datetime.datetime", present.max())
            status, reason = "ok", None
            range_seconds = float((maximum - minimum).total_seconds())

        rows.append(
            {
                "column": column,
                "role": role,
                "n_total": n_total,
                "n_present": n_present,
                "n_missing": n_missing,
                "min": minimum,
                "max": maximum,
                "range_seconds": range_seconds,
                "status": status,
                "reason": reason,
            }
        )
    return pl.DataFrame(rows, schema=DATETIME_STATISTICS_SCHEMA)


def summarize_univariate(
    dataset: TabularDataset,
    columns: pl.DataFrame,
    *,
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
    min_numeric_n: int = 3,
    max_category_levels: int = 50,
) -> UnivariateTables:
    """Compute all univariate descriptive tables."""
    quantiles = validate_quantiles(quantiles)
    numeric = summarize_numeric_statistics(
        dataset,
        columns,
        quantiles=quantiles,
        min_numeric_n=min_numeric_n,
    )
    categorical, frequencies = summarize_categorical_statistics(
        dataset,
        columns,
        max_category_levels=max_category_levels,
    )
    datetime = summarize_datetime_statistics(dataset, columns)
    return UnivariateTables(
        numeric_statistics=numeric,
        categorical_statistics=categorical,
        categorical_frequencies=frequencies,
        datetime_statistics=datetime,
    )


def analyze_univariate(
    dataset: TabularDataset,
    *,
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
    min_numeric_n: int = 3,
    max_category_levels: int = 50,
    max_missingness_patterns: int = 20,
    max_pairwise_columns: int = 200,
) -> UnivariateResult:
    """Run profiling and complete univariate descriptive analysis."""
    quantiles = validate_quantiles(quantiles)
    profiling = profile_dataset(
        dataset,
        max_missingness_patterns=max_missingness_patterns,
        max_pairwise_columns=max_pairwise_columns,
    )
    tables = summarize_univariate(
        dataset,
        profiling.columns,
        quantiles=quantiles,
        min_numeric_n=min_numeric_n,
        max_category_levels=max_category_levels,
    )
    provenance = AnalysisProvenance(
        analysis="univariate",
        parameters={
            "quantiles": quantiles,
            "min_numeric_n": min_numeric_n,
            "max_category_levels": max_category_levels,
            "numeric_finite_policy": "finite_values_only",
            "numeric_standard_deviation_ddof": 1,
            "categorical_entropy_base": 2,
            "datetime_policy": "descriptive_only",
        },
        input_summary={
            "n_observations": dataset.n_observations,
            "n_columns": dataset.n_columns,
            "id_column": dataset.id_column,
        },
    )
    return UnivariateResult(
        profiling=profiling,
        numeric_statistics=tables.numeric_statistics,
        categorical_statistics=tables.categorical_statistics,
        categorical_frequencies=tables.categorical_frequencies,
        datetime_statistics=tables.datetime_statistics,
        provenance=provenance,
    )
