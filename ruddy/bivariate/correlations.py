"""Numeric-to-numeric association analysis."""

from __future__ import annotations

import math
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from ruddy.core.enums import ColumnKind, ColumnRole, CorrelationMethod, PAdjustMethod
from ruddy.statistics import apply_multiple_testing

if TYPE_CHECKING:
    from ruddy.data import TabularDataset

CORRELATION_COLUMNS: tuple[str, ...] = (
    "method",
    "column_x",
    "column_y",
    "n_total",
    "n_complete",
    "n_missing_pair",
    "coefficient",
    "statistic",
    "p_value",
    "q_value",
    "family_id",
    "family_size",
    "correction",
    "status",
    "reason",
)


def _eligible_numeric_columns(dataset: TabularDataset) -> tuple[str, ...]:
    columns: list[str] = []
    for spec in dataset.schema:
        if spec.kind is not ColumnKind.NUMERIC:
            continue
        if spec.role in {ColumnRole.IDENTIFIER, ColumnRole.EXCLUDED, ColumnRole.FACTOR}:
            continue
        columns.append(spec.name)
    return tuple(columns)


def _pairwise_finite(left: pd.Series, right: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = left.to_numpy(dtype=np.float64, na_value=np.nan)
    y = right.to_numpy(dtype=np.float64, na_value=np.nan)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _constant_reason(x: np.ndarray, y: np.ndarray) -> str | None:
    x_constant = np.unique(x).size <= 1
    y_constant = np.unique(y).size <= 1
    if x_constant and y_constant:
        return "both_columns_constant"
    if x_constant:
        return "column_x_constant"
    if y_constant:
        return "column_y_constant"
    return None


def _run(method: CorrelationMethod, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if method is CorrelationMethod.PEARSON:
        result = stats.pearsonr(x, y)
    elif method is CorrelationMethod.SPEARMAN:
        result = stats.spearmanr(x, y)
    elif method is CorrelationMethod.KENDALL:
        result = stats.kendalltau(x, y)
    else:  # pragma: no cover
        msg = f"Unsupported correlation method: {method!r}."
        raise ValueError(msg)
    return float(result.statistic), float(result.pvalue)


def summarize_correlations(
    dataset: TabularDataset,
    *,
    methods: tuple[CorrelationMethod | str, ...] = (
        CorrelationMethod.PEARSON,
        CorrelationMethod.SPEARMAN,
        CorrelationMethod.KENDALL,
    ),
    min_complete_pairs: int = 3,
    max_columns: int = 100,
    p_adjust: PAdjustMethod | str = PAdjustMethod.FDR_BH,
) -> pd.DataFrame:
    """Compute all unordered numeric-numeric correlations.

    Missing and non-finite observations are removed pairwise. Ruddy never
    selects a method from normality diagnostics; requested methods are executed
    explicitly and corrected in separate hypothesis families.
    """
    if min_complete_pairs < 2:
        msg = "min_complete_pairs must be at least 2."
        raise ValueError(msg)
    if max_columns < 2:
        msg = "max_columns must be at least 2."
        raise ValueError(msg)
    resolved_methods = tuple(
        method if isinstance(method, CorrelationMethod) else CorrelationMethod(method) for method in methods
    )
    if not resolved_methods:
        msg = "At least one correlation method is required."
        raise ValueError(msg)
    if len(set(resolved_methods)) != len(resolved_methods):
        msg = "Correlation methods cannot contain duplicates."
        raise ValueError(msg)
    correction = p_adjust if isinstance(p_adjust, PAdjustMethod) else PAdjustMethod(p_adjust)

    candidates = _eligible_numeric_columns(dataset)
    if len(candidates) > max_columns:
        msg = f"Resolved {len(candidates)} numeric columns, exceeding max_columns={max_columns}."
        raise ValueError(msg)

    frame = dataset.to_frame()
    rows: list[dict[str, Any]] = []
    n_total = dataset.n_observations
    for method in resolved_methods:
        family_id = f"correlations:{method.value}"
        for column_x, column_y in combinations(candidates, 2):
            x, y = _pairwise_finite(frame[column_x], frame[column_y])
            n_complete = int(x.size)
            row: dict[str, Any] = {
                "method": method.value,
                "column_x": column_x,
                "column_y": column_y,
                "n_total": n_total,
                "n_complete": n_complete,
                "n_missing_pair": int(n_total - n_complete),
                "coefficient": np.nan,
                "statistic": np.nan,
                "p_value": np.nan,
                "q_value": np.nan,
                "family_id": family_id,
                "family_size": 0,
                "correction": correction.value,
                "status": "ok",
                "reason": None,
            }
            if n_complete < min_complete_pairs:
                row["status"] = "skipped"
                row["reason"] = "insufficient_complete_pairs"
                rows.append(row)
                continue
            reason = _constant_reason(x, y)
            if reason is not None:
                row["status"] = "degenerate"
                row["reason"] = reason
                rows.append(row)
                continue
            coefficient, p_value = _run(method, x, y)
            if not math.isfinite(coefficient) or not math.isfinite(p_value):
                row["status"] = "degenerate"
                row["reason"] = "undefined_correlation"
                rows.append(row)
                continue
            row["coefficient"] = coefficient
            row["statistic"] = coefficient
            row["p_value"] = p_value
            rows.append(row)

    table = pd.DataFrame(rows, columns=pd.Index(CORRELATION_COLUMNS))
    if table.empty:
        return table
    return apply_multiple_testing(table, correction)
