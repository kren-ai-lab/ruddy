"""Univariate orchestration and datetime descriptive statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from ruddy.core.enums import ColumnKind, ColumnRole
from ruddy.profiling import ProfilingResult, profile_dataset
from ruddy.results import AnalysisProvenance
from ruddy.univariate.categorical import summarize_categorical_statistics
from ruddy.univariate.numeric import summarize_numeric_statistics, validate_quantiles

if TYPE_CHECKING:
    from ruddy.data import TabularDataset

DATETIME_STATISTICS_COLUMNS: tuple[str, ...] = (
    "column",
    "role",
    "n_total",
    "n_present",
    "n_missing",
    "min",
    "max",
    "range_seconds",
    "status",
    "reason",
)


@dataclass(frozen=True, slots=True)
class UnivariateTables:
    """Collection of univariate descriptive statistics tables."""

    numeric_statistics: pd.DataFrame
    categorical_statistics: pd.DataFrame
    categorical_frequencies: pd.DataFrame
    datetime_statistics: pd.DataFrame


@dataclass(frozen=True, slots=True)
class UnivariateResult:
    """Result of a complete univariate descriptive analysis."""

    profiling: ProfilingResult
    numeric_statistics: pd.DataFrame
    categorical_statistics: pd.DataFrame
    categorical_frequencies: pd.DataFrame
    datetime_statistics: pd.DataFrame
    provenance: AnalysisProvenance


def _eligible_datetime(profile: pd.Series) -> bool:
    if not bool(profile["analysis_eligible"]):
        return False
    role = ColumnRole(str(profile["role"]))
    kind = ColumnKind(str(profile["data_kind"]))
    return role is not ColumnRole.FACTOR and kind is ColumnKind.DATETIME


def summarize_datetime_statistics(
    dataset: TabularDataset,
    columns: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize eligible datetime variables without time-series interpretation."""
    frame = dataset.to_frame()
    rows: list[dict[str, Any]] = []
    selected = columns.loc[columns.apply(_eligible_datetime, axis=1)]
    for _, profile in selected.iterrows():
        column = str(profile["column"])
        series = frame[column]
        present = series.dropna()
        n_total = len(series)
        n_present = len(present)
        n_missing = int(n_total - n_present)
        minimum = present.min() if n_present else None
        maximum = present.max() if n_present else None
        if n_present == 0:
            status, reason = "skipped", "all_missing"
            range_seconds = None
        elif minimum == maximum:
            status, reason = "degenerate", "constant"
            range_seconds = 0.0
        else:
            status, reason = "ok", None
            # Recomputed from `present` so the non-empty branch needs no None check.
            range_seconds = float((present.max() - present.min()).total_seconds())
        rows.append(
            {
                "column": column,
                "role": str(profile["role"]),
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
    return pd.DataFrame(rows, columns=pd.Index(DATETIME_STATISTICS_COLUMNS))


def summarize_univariate(
    dataset: TabularDataset,
    columns: pd.DataFrame,
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
