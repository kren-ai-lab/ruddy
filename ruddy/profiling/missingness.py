"""Missing-data summaries and completeness diagnostics."""

from __future__ import annotations

import json
from itertools import combinations_with_replacement
from typing import Any

import polars as pl

from ruddy.data import TabularDataset

MISSINGNESS_SCHEMA: dict[str, pl.DataType] = {
    "column": pl.String,
    "role": pl.String,
    "data_kind": pl.String,
    "n_total": pl.Int64,
    "n_present": pl.Int64,
    "fraction_present": pl.Float64,
    "n_missing": pl.Int64,
    "fraction_missing": pl.Float64,
    "finite_count": pl.Int64,
    "non_finite_count": pl.Int64,
    "is_all_missing": pl.Boolean,
}

MISSINGNESS_COLUMNS: tuple[str, ...] = tuple(MISSINGNESS_SCHEMA)

PAIRWISE_COMPLETENESS_SCHEMA: dict[str, pl.DataType] = {
    "column_x": pl.String,
    "column_y": pl.String,
    "n_total": pl.Int64,
    "n_complete": pl.Int64,
    "n_incomplete": pl.Int64,
    "fraction_complete": pl.Float64,
}

PAIRWISE_COMPLETENESS_COLUMNS: tuple[str, ...] = tuple(PAIRWISE_COMPLETENESS_SCHEMA)

MISSINGNESS_PATTERN_SCHEMA: dict[str, pl.DataType] = {
    "rank": pl.Int64,
    "missing_columns": pl.String,
    "n_missing_columns": pl.Int64,
    "count": pl.Int64,
    "fraction": pl.Float64,
    "n_patterns_total": pl.Int64,
    "patterns_truncated": pl.Boolean,
    "unreported_count": pl.Int64,
    "unreported_fraction": pl.Float64,
}

MISSINGNESS_PATTERN_COLUMNS: tuple[str, ...] = tuple(MISSINGNESS_PATTERN_SCHEMA)


def summarize_missingness(columns: pl.DataFrame) -> pl.DataFrame:
    """Build deterministic per-column missingness output from column profiles."""
    required = {
        "column",
        "role",
        "data_kind",
        "n_total",
        "n_present",
        "n_missing",
        "missing_fraction",
        "finite_count",
        "non_finite_count",
        "is_all_missing",
    }
    missing = sorted(required - set(columns.columns))
    if missing:
        raise ValueError(f"Column profile is missing required fields: {missing}.")

    return columns.select(
        pl.col("column").cast(pl.String),
        pl.col("role").cast(pl.String),
        pl.col("data_kind").cast(pl.String),
        pl.col("n_total").cast(pl.Int64),
        pl.col("n_present").cast(pl.Int64),
        (1.0 - pl.col("missing_fraction").cast(pl.Float64)).alias("fraction_present"),
        pl.col("n_missing").cast(pl.Int64),
        pl.col("missing_fraction").cast(pl.Float64).alias("fraction_missing"),
        pl.col("finite_count").cast(pl.Int64),
        pl.col("non_finite_count").cast(pl.Int64),
        pl.col("is_all_missing").cast(pl.Boolean),
    )


def pairwise_completeness(
    dataset: TabularDataset,
    *,
    columns: tuple[str, ...] | None = None,
    max_columns: int = 200,
) -> pl.DataFrame:
    """Return upper-triangular pairwise non-missing completeness statistics.

    Missingness here means null or NaN missing values. Numeric infinities remain
    present observations and are handled separately by numerical analyses.
    """
    selected = columns or tuple(dataset.frame.columns)
    unknown = sorted(set(selected) - set(dataset.frame.columns))
    if unknown:
        raise ValueError(f"Unknown columns requested for completeness: {unknown}.")
    if max_columns < 1:
        raise ValueError("max_columns must be at least 1.")
    if len(selected) > max_columns:
        raise ValueError(
            f"Pairwise completeness requested for {len(selected)} columns, exceeding "
            f"max_columns={max_columns}."
        )

    presence_masks: dict[str, pl.Series] = {}
    for column in selected:
        series = dataset.frame.get_column(column)
        mask = series.is_not_null()
        if series.dtype.is_float():
            mask = mask & ~series.is_nan()
        presence_masks[column] = mask

    n_total = dataset.frame.height
    rows: list[dict[str, Any]] = []
    for column_x, column_y in combinations_with_replacement(selected, 2):
        n_complete = int((presence_masks[column_x] & presence_masks[column_y]).sum() or 0)
        rows.append(
            {
                "column_x": column_x,
                "column_y": column_y,
                "n_total": n_total,
                "n_complete": n_complete,
                "n_incomplete": int(n_total - n_complete),
                "fraction_complete": float(n_complete / n_total) if n_total else 0.0,
            }
        )
    return pl.DataFrame(rows, schema=PAIRWISE_COMPLETENESS_SCHEMA)


def summarize_missingness_patterns(
    dataset: TabularDataset,
    *,
    max_patterns: int = 20,
) -> pl.DataFrame:
    """Summarize common row-level missingness patterns with bounded output."""
    if max_patterns < 1:
        raise ValueError("max_patterns must be at least 1.")

    frame = dataset.frame
    n_total = frame.height
    cols = frame.columns
    counts: dict[tuple[str, ...], int] = {}

    if not cols:
        if n_total > 0:
            counts[()] = n_total
    else:
        exprs = [
            (pl.col(c).is_null() | pl.col(c).is_nan()).alias(c)
            if frame.schema[c].is_float()
            else pl.col(c).is_null().alias(c)
            for c in cols
        ]
        missing_frame = frame.select(exprs)
        for row in missing_frame.iter_rows():
            pattern = tuple(col for col, is_missing in zip(cols, row, strict=True) if is_missing)
            counts[pattern] = counts.get(pattern, 0) + 1

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    n_patterns_total = len(ordered)
    reported = ordered[:max_patterns]
    reported_count = sum(count for _, count in reported)
    unreported_count = int(n_total - reported_count)
    truncated = n_patterns_total > max_patterns

    rows = [
        {
            "rank": rank,
            "missing_columns": json.dumps(pattern, separators=(",", ":")),
            "n_missing_columns": len(pattern),
            "count": count,
            "fraction": float(count / n_total) if n_total else 0.0,
            "n_patterns_total": n_patterns_total,
            "patterns_truncated": truncated,
            "unreported_count": unreported_count,
            "unreported_fraction": float(unreported_count / n_total) if n_total else 0.0,
        }
        for rank, (pattern, count) in enumerate(reported, start=1)
    ]
    return pl.DataFrame(rows, schema=MISSINGNESS_PATTERN_SCHEMA)
