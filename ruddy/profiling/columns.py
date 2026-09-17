"""Deterministic column profiling for domain-agnostic tabular data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ruddy.core.enums import ColumnKind, ColumnRole

if TYPE_CHECKING:
    from ruddy.data import TabularDataset

COLUMN_PROFILE_COLUMNS: tuple[str, ...] = (
    "column",
    "dtype",
    "role",
    "data_kind",
    "excluded",
    "analysis_eligible",
    "n_total",
    "n_present",
    "n_missing",
    "missing_fraction",
    "n_unique",
    "unique_fraction",
    "finite_count",
    "non_finite_count",
    "is_all_missing",
    "is_constant",
)

_ANALYSIS_ELIGIBLE_ROLES = {
    ColumnRole.VARIABLE,
    ColumnRole.RESPONSE,
    ColumnRole.FACTOR,
    ColumnRole.COVARIATE,
    ColumnRole.ANNOTATION,
}


def _safe_unique_count(series: pd.Series) -> int | None:
    try:
        return int(series.nunique(dropna=True))
    except (TypeError, ValueError):
        return None


def _finite_counts(series: pd.Series, kind: ColumnKind) -> tuple[int | None, int | None]:
    if kind is not ColumnKind.NUMERIC:
        return None, None
    present = series.dropna()
    if present.empty:
        return 0, 0
    values = present.to_numpy(dtype=np.float64, na_value=np.nan)
    finite = np.isfinite(values)
    return int(finite.sum()), int((~finite).sum())


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
    return not (n_unique is None and kind in {ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN})


def profile_columns(dataset: TabularDataset) -> pd.DataFrame:
    """Profile every dataset column without transforming source values.

    The returned table preserves source column order. Statistical roles and data
    kinds come directly from :class:`TabularDataset`, so profiling cannot silently
    reinterpret or coerce the input.
    """
    frame = dataset.to_frame()
    rows: list[dict[str, Any]] = []

    for column in dataset.columns:
        series = frame[column]
        role = dataset.role_of(column)
        kind = dataset.kind_of(column)

        n_total = len(series)
        n_missing = int(series.isna().sum())
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
                "missing_fraction": float(n_missing / n_total) if n_total else 0.0,
                "n_unique": n_unique,
                "unique_fraction": unique_fraction,
                "finite_count": finite_count,
                "non_finite_count": non_finite_count,
                "is_all_missing": is_all_missing,
                "is_constant": is_constant,
            }
        )

    return pd.DataFrame(rows, columns=pd.Index(COLUMN_PROFILE_COLUMNS))
