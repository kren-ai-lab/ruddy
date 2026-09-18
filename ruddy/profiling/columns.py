"""Deterministic column profiling for domain-agnostic tabular data."""

from __future__ import annotations

from typing import Any

import polars as pl
from polars._typing import PolarsDataType

from ruddy.core.enums import ColumnKind, ColumnRole
from ruddy.data import TabularDataset

COLUMN_PROFILE_SCHEMA: dict[str, PolarsDataType] = {
    "column": pl.String,
    "dtype": pl.String,
    "role": pl.String,
    "data_kind": pl.String,
    "excluded": pl.Boolean,
    "analysis_eligible": pl.Boolean,
    "n_total": pl.Int64,
    "n_present": pl.Int64,
    "n_missing": pl.Int64,
    "missing_fraction": pl.Float64,
    "n_unique": pl.Int64,
    "unique_fraction": pl.Float64,
    "finite_count": pl.Int64,
    "non_finite_count": pl.Int64,
    "is_all_missing": pl.Boolean,
    "is_constant": pl.Boolean,
}

COLUMN_PROFILE_COLUMNS: tuple[str, ...] = tuple(COLUMN_PROFILE_SCHEMA)

_ANALYSIS_ELIGIBLE_ROLES = {
    ColumnRole.VARIABLE,
    ColumnRole.RESPONSE,
    ColumnRole.FACTOR,
    ColumnRole.COVARIATE,
    ColumnRole.ANNOTATION,
}


def _safe_unique_count(series: pl.Series) -> int | None:
    try:
        clean = series.drop_nulls()
        if clean.dtype.is_float():
            clean = clean.filter(~clean.is_nan())
        return int(clean.n_unique())
    except (pl.exceptions.PolarsError, TypeError, ValueError):
        return None


def _finite_counts(series: pl.Series, kind: ColumnKind) -> tuple[int | None, int | None]:
    if kind is not ColumnKind.NUMERIC:
        return None, None
    clean = series.drop_nulls()
    if clean.dtype.is_float():
        clean = clean.filter(~clean.is_nan())
    if clean.len() == 0:
        return 0, 0
    values = clean.cast(pl.Float64)
    is_inf = values.is_infinite()
    non_finite = int(is_inf.sum() or 0)
    finite = values.len() - non_finite
    return finite, non_finite


def _is_analysis_eligible(
    *,
    role: ColumnRole,
    kind: ColumnKind,
    n_unique: int | None,
) -> bool:
    if role not in _ANALYSIS_ELIGIBLE_ROLES:
        return False
    if kind is ColumnKind.UNKNOWN:
        return False
    if n_unique is None and kind in {ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN}:
        return False
    return True


def profile_columns(dataset: TabularDataset) -> pl.DataFrame:
    """Profile every dataset column without transforming source values.

    The returned table preserves source column order. Statistical roles and data
    kinds come directly from :class:`TabularDataset`, so profiling cannot silently
    reinterpret or coerce the input.
    """
    if not dataset.frame.columns:
        return pl.DataFrame(schema=COLUMN_PROFILE_SCHEMA)

    rows: list[dict[str, Any]] = []

    for column in dataset.frame.columns:
        series = dataset.frame.get_column(column)
        role = dataset.role_of(column)
        kind = dataset.kind_of(column)

        n_total = series.len()
        n_missing = series.null_count()
        if series.dtype.is_float():
            nan_sum = series.is_nan().sum()
            if nan_sum is not None:
                n_missing += int(nan_sum)
        n_present = n_total - n_missing
        n_unique = _safe_unique_count(series)
        unique_fraction = float(n_unique / n_present) if n_unique is not None and n_present > 0 else None
        finite_count, non_finite_count = _finite_counts(series, kind)
        is_all_missing = n_present == 0
        is_constant = bool(n_present > 0 and n_unique is not None and n_unique == 1)
        excluded = role is ColumnRole.EXCLUDED

        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "role": role.value,
                "data_kind": kind.value,
                "excluded": excluded,
                "analysis_eligible": _is_analysis_eligible(
                    role=role,
                    kind=kind,
                    n_unique=n_unique,
                ),
                "n_total": n_total,
                "n_present": n_present,
                "n_missing": n_missing,
                "missing_fraction": float(n_missing / n_total) if n_total > 0 else 0.0,
                "n_unique": n_unique,
                "unique_fraction": unique_fraction,
                "finite_count": finite_count,
                "non_finite_count": non_finite_count,
                "is_all_missing": is_all_missing,
                "is_constant": is_constant,
            }
        )

    return pl.DataFrame(rows, schema=COLUMN_PROFILE_SCHEMA)
