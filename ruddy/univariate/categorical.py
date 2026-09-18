"""Deterministic categorical and boolean univariate statistics."""

from __future__ import annotations

import collections
import datetime
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

    from ruddy.data import TabularDataset

CATEGORICAL_STATISTICS_SCHEMA: dict[str, PolarsDataType] = {
    "column": pl.String,
    "role": pl.String,
    "n_total": pl.Int64,
    "n_present": pl.Int64,
    "n_missing": pl.Int64,
    "n_levels": pl.Int64,
    "mode": pl.String,
    "mode_count": pl.Int64,
    "mode_fraction": pl.Float64,
    "entropy": pl.Float64,
    "normalized_entropy": pl.Float64,
    "entropy_base": pl.Int64,
    "n_levels_reported": pl.Int64,
    "frequencies_truncated": pl.Boolean,
    "unreported_count": pl.Int64,
    "unreported_fraction": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}

CATEGORICAL_STATISTICS_COLUMNS: tuple[str, ...] = tuple(CATEGORICAL_STATISTICS_SCHEMA)

CATEGORICAL_FREQUENCIES_SCHEMA: dict[str, PolarsDataType] = {
    "column": pl.String,
    "role": pl.String,
    "level": pl.String,
    "count": pl.Int64,
    "fraction": pl.Float64,
    "rank": pl.Int64,
}

CATEGORICAL_FREQUENCY_COLUMNS: tuple[str, ...] = tuple(CATEGORICAL_FREQUENCIES_SCHEMA)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _category_label(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        converted = float(value)
        if math.isfinite(converted):
            return repr(converted)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    return str(value)


def _profile(
    *,
    column: str,
    role: str,
    series: pl.Series,
    max_category_levels: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n_total = series.len()
    n_missing = series.null_count()
    if series.dtype.is_float():
        nan_sum = series.is_nan().sum()
        if nan_sum is not None:
            n_missing += int(nan_sum)
    n_present = n_total - n_missing

    present = series.drop_nulls()
    if present.dtype.is_float():
        present = present.filter(~present.is_nan())

    labels = [_category_label(v) for v in present.to_list()]
    counts = collections.Counter(labels)
    ordered = sorted(
        ((str(level), int(count)) for level, count in counts.items()),
        key=lambda item: (-item[1], item[0]),
    )
    n_levels = len(ordered)
    if n_present == 0:
        status, reason = "skipped", "all_missing"
    elif n_levels == 1:
        status, reason = "degenerate", "constant"
    else:
        status, reason = "ok", None

    mode: str | None = None
    mode_count = 0
    mode_fraction: float | None = None
    entropy: float | None = None
    normalized_entropy: float | None = None

    if ordered:
        mode, mode_count = ordered[0]
        mode_fraction = float(mode_count / n_present)
        probabilities = np.asarray([count / n_present for _, count in ordered], dtype=np.float64)
        entropy = _safe_float(-np.sum(probabilities * np.log2(probabilities)))
        if n_levels <= 1:
            normalized_entropy = 0.0
        elif entropy is not None:
            normalized_entropy = _safe_float(entropy / math.log2(n_levels))

    n_reported = min(n_levels, max_category_levels)
    reported = ordered[:n_reported]
    reported_count = int(sum(count for _, count in reported))
    unreported_count = int(n_present - reported_count)

    summary = {
        "column": column,
        "role": role,
        "n_total": n_total,
        "n_present": n_present,
        "n_missing": n_missing,
        "n_levels": n_levels,
        "mode": mode,
        "mode_count": mode_count,
        "mode_fraction": mode_fraction,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "entropy_base": 2,
        "n_levels_reported": n_reported,
        "frequencies_truncated": n_levels > max_category_levels,
        "unreported_count": unreported_count,
        "unreported_fraction": float(unreported_count / n_present) if n_present else None,
        "status": status,
        "reason": reason,
    }
    frequencies = [
        {
            "column": column,
            "role": role,
            "level": level,
            "count": count,
            "fraction": float(count / n_present),
            "rank": rank,
        }
        for rank, (level, count) in enumerate(reported, start=1)
    ]
    return summary, frequencies


def summarize_categorical_statistics(
    dataset: TabularDataset,
    columns: pl.DataFrame,
    *,
    max_category_levels: int = 50,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Summarize eligible categorical, boolean, and factor variables."""
    if max_category_levels < 2:
        raise ValueError("max_category_levels must be at least 2.")

    selected = columns.filter(
        pl.col("analysis_eligible")
        & ((pl.col("role") == "factor") | pl.col("data_kind").is_in(["categorical", "boolean"]))
    )
    summaries: list[dict[str, Any]] = []
    frequencies: list[dict[str, Any]] = []
    for profile in selected.iter_rows(named=True):
        column = str(profile["column"])
        role = str(profile["role"])
        series = dataset.frame.get_column(column)
        summary, rows = _profile(
            column=column,
            role=role,
            series=series,
            max_category_levels=max_category_levels,
        )
        summaries.append(summary)
        frequencies.extend(rows)

    return (
        pl.DataFrame(summaries, schema=CATEGORICAL_STATISTICS_SCHEMA),
        pl.DataFrame(frequencies, schema=CATEGORICAL_FREQUENCIES_SCHEMA),
    )
