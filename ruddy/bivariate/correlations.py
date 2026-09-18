"""Numeric-to-numeric association analysis."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
import polars as pl
from polars._typing import PolarsDataType
from scipy import stats

from ruddy.core.enums import ColumnKind, ColumnRole, CorrelationMethod, PAdjustMethod
from ruddy.data import TabularDataset
from ruddy.statistics import apply_multiple_testing

CORRELATION_SCHEMA: dict[str, PolarsDataType] = {
    "method": pl.String,
    "column_x": pl.String,
    "column_y": pl.String,
    "n_total": pl.Int64,
    "n_complete": pl.Int64,
    "n_missing_pair": pl.Int64,
    "coefficient": pl.Float64,
    "statistic": pl.Float64,
    "p_value": pl.Float64,
    "q_value": pl.Float64,
    "family_id": pl.String,
    "family_size": pl.Int64,
    "correction": pl.String,
    "status": pl.String,
    "reason": pl.String,
}
CORRELATION_COLUMNS: tuple[str, ...] = tuple(CORRELATION_SCHEMA)


def _eligible_numeric_columns(dataset: TabularDataset) -> tuple[str, ...]:
    columns: list[str] = []
    for spec in dataset.schema:
        if spec.kind is not ColumnKind.NUMERIC:
            continue
        if spec.role in {ColumnRole.IDENTIFIER, ColumnRole.EXCLUDED, ColumnRole.FACTOR}:
            continue
        columns.append(spec.name)
    return tuple(columns)


def _pairwise_finite(left: pl.Series, right: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    x = left.cast(pl.Float64).fill_null(float("nan")).to_numpy()
    y = right.cast(pl.Float64).fill_null(float("nan")).to_numpy()
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
        raise ValueError(f"Unsupported correlation method: {method!r}.")
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
) -> pl.DataFrame:
    """Compute all unordered numeric-numeric correlations.

    Missing and non-finite observations are removed pairwise. Ruddy never
    selects a method from normality diagnostics; requested methods are executed
    explicitly and corrected in separate hypothesis families.
    """
    if min_complete_pairs < 2:
        raise ValueError("min_complete_pairs must be at least 2.")
    if max_columns < 2:
        raise ValueError("max_columns must be at least 2.")
    resolved_methods = tuple(
        method if isinstance(method, CorrelationMethod) else CorrelationMethod(method) for method in methods
    )
    if not resolved_methods:
        raise ValueError("At least one correlation method is required.")
    if len(set(resolved_methods)) != len(resolved_methods):
        raise ValueError("Correlation methods cannot contain duplicates.")
    correction = p_adjust if isinstance(p_adjust, PAdjustMethod) else PAdjustMethod(p_adjust)

    candidates = _eligible_numeric_columns(dataset)
    if len(candidates) > max_columns:
        raise ValueError(f"Resolved {len(candidates)} numeric columns, exceeding max_columns={max_columns}.")

    frame = dataset.frame
    rows: list[dict[str, Any]] = []
    n_total = frame.height
    for method in resolved_methods:
        family_id = f"correlations:{method.value}"
        for column_x, column_y in combinations(candidates, 2):
            x, y = _pairwise_finite(frame.get_column(column_x), frame.get_column(column_y))
            n_complete = int(x.size)
            row: dict[str, Any] = {
                "method": method.value,
                "column_x": column_x,
                "column_y": column_y,
                "n_total": n_total,
                "n_complete": n_complete,
                "n_missing_pair": int(n_total - n_complete),
                "coefficient": None,
                "statistic": None,
                "p_value": None,
                "q_value": None,
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

    table = pl.DataFrame(rows, schema=CORRELATION_SCHEMA)
    return table if table.height == 0 else apply_multiple_testing(table, correction)
