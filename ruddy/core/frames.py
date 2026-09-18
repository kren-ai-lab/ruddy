"""Core operations for data frames and missingness arrays."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from polars._typing import PolarsDataType


def to_float_array(series: pl.Series) -> np.ndarray:
    """Convert series to float array with NaN for missing."""
    return series.cast(pl.Float64).fill_null(float("nan")).to_numpy()


def present_mask(series: pl.Series) -> pl.Series:
    """Return boolean mask of non-missing, non-NaN values."""
    mask = series.is_not_null()
    if series.dtype.is_float():
        mask = mask & ~series.is_nan().fill_null(value=False)
    return mask


def present_values(series: pl.Series) -> pl.Series:
    """Return series filtered to present values only."""
    return series.filter(present_mask(series))


def n_missing(series: pl.Series) -> int:
    """Return count of missing or NaN values."""
    return int(series.len() - int(present_mask(series).sum()))


def missing_expr(name: str, dtype: PolarsDataType) -> pl.Expr:
    """Return expression for missing/NaN values."""
    expr = pl.col(name).is_null() | pl.col(name).is_nan() if dtype.is_float() else pl.col(name).is_null()
    return expr.alias(name)


def id_dtype(ids: Sequence[Any]) -> PolarsDataType:
    """Determine Polars dtype for a sequence of IDs."""
    return pl.Series(list(ids)).dtype if len(ids) else pl.String


def finite_or_none(value: Any) -> float | None:
    """Convert value to float if finite, else return None."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def labeled_matrix_table(
    ids: np.ndarray | list[Any] | tuple[Any, ...],
    columns: list[str],
    matrix: np.ndarray,
    id_name: str = "observation_id",
    dtype: PolarsDataType = pl.Float64,
) -> pl.DataFrame:
    """Create a Polars DataFrame with labelled rows and columns."""
    id_list = list(ids)
    id_dt = id_dtype(id_list)
    return pl.DataFrame(
        {
            id_name: pl.Series(id_name, id_list, dtype=id_dt),
            **{cols: pl.Series(cols, matrix[:, i], dtype=dtype) for i, cols in enumerate(columns)},
        }
    )
