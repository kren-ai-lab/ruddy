"""Deterministic multiple-testing correction utilities."""

from __future__ import annotations

from typing import overload

import numpy as np
import pandas as pd
import polars as pl

from ruddy.core.enums import PAdjustMethod


def adjust_pvalues(
    p_values: np.ndarray,
    method: PAdjustMethod | str = PAdjustMethod.FDR_BH,
) -> np.ndarray:
    """Adjust one ordered family of finite p-values."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    if values.size == 0:
        return values.copy()
    if not np.isfinite(values).all():
        raise ValueError("p_values must contain only finite values.")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("p_values must lie in [0, 1].")

    resolved = method if isinstance(method, PAdjustMethod) else PAdjustMethod(method)
    if resolved is PAdjustMethod.NONE:
        return values.copy()

    m = values.size
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    adjusted_ranked = ranked * float(m) / ranks
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def family_sizes(
    table: pl.DataFrame | pd.DataFrame,
    *,
    family_column: str = "family_id",
    status_column: str = "status",
    ok_status: str = "ok",
) -> dict[str, int]:
    """Count inferential hypotheses in each explicitly declared family."""
    if isinstance(table, pl.DataFrame):
        if table.height == 0:
            return {}
        families = (
            table.filter(pl.col(status_column) == ok_status)
            .get_column(family_column)
            .cast(pl.String)
            .to_list()
        )
        counts: dict[str, int] = {}
        for family in families:
            counts[family] = counts.get(family, 0) + 1
        return counts

    # ponytail: temporary pandas adapter, removed in task 3D
    if table.empty:
        return {}
    mask = table[status_column].eq(ok_status)
    families = table.loc[mask, family_column].astype(str)
    return {family: int((families == family).sum()) for family in families.drop_duplicates().tolist()}


@overload
def apply_multiple_testing(
    table: pl.DataFrame,
    method: PAdjustMethod | str = PAdjustMethod.FDR_BH,
    *,
    family_column: str = "family_id",
    status_column: str = "status",
    p_column: str = "p_value",
    q_column: str = "q_value",
    family_size_column: str = "family_size",
    correction_column: str = "correction",
) -> pl.DataFrame: ...


@overload
def apply_multiple_testing(
    table: pd.DataFrame,
    method: PAdjustMethod | str = PAdjustMethod.FDR_BH,
    *,
    family_column: str = "family_id",
    status_column: str = "status",
    p_column: str = "p_value",
    q_column: str = "q_value",
    family_size_column: str = "family_size",
    correction_column: str = "correction",
) -> pd.DataFrame: ...


def apply_multiple_testing(
    table: pl.DataFrame | pd.DataFrame,
    method: PAdjustMethod | str = PAdjustMethod.FDR_BH,
    *,
    family_column: str = "family_id",
    status_column: str = "status",
    p_column: str = "p_value",
    q_column: str = "q_value",
    family_size_column: str = "family_size",
    correction_column: str = "correction",
) -> pl.DataFrame | pd.DataFrame:
    """Adjust p-values independently within explicit hypothesis families.

    Only ``status == 'ok'`` rows with finite p-values belong to the inferential
    family. This permits descriptive/statistic-only runs (for example zero
    permutations) to retain ``status='ok'`` while leaving p/q values missing.
    Non-inferential rows retain missing q-values.
    """
    if isinstance(table, pl.DataFrame):
        required = {family_column, status_column, p_column, correction_column}
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError("Missing multiple-testing columns: " + ", ".join(missing))

        resolved = method if isinstance(method, PAdjustMethod) else PAdjustMethod(method)

        correction_series = table.get_column(correction_column).cast(pl.String)
        if correction_series.null_count() > 0 or not bool((correction_series == resolved.value).all()):
            raise ValueError("correction column is inconsistent with configured method.")

        if table.height == 0:
            return table.with_columns(
                pl.Series(q_column, [], dtype=pl.Float64),
                pl.Series(family_size_column, [], dtype=pl.Int64),
            )

        families_series = table.get_column(family_column)
        if families_series.null_count() > 0:
            raise ValueError("family_id values must be present and non-empty.")
        stripped = families_series.cast(pl.String).str.strip_chars()
        if bool((stripped == "").any()):
            raise ValueError("family_id values must be present and non-empty.")

        n = table.height
        p = table.get_column(p_column).cast(pl.Float64).fill_null(float("nan")).to_numpy()
        status = table.get_column(status_column).cast(pl.String).to_numpy()
        family = families_series.cast(pl.String).to_numpy()
        q = np.full(n, np.nan, dtype=np.float64)
        sizes = np.zeros(n, dtype=np.int64)

        finite_p = np.isfinite(p)
        for family_id in dict.fromkeys(family):
            family_mask = family == family_id
            inferential = family_mask & (status == "ok") & finite_p
            k = int(inferential.sum())
            sizes[family_mask] = k
            if k > 0:
                q[inferential] = adjust_pvalues(p[inferential], resolved)

        return table.with_columns(
            pl.Series(q_column, q, dtype=pl.Float64).fill_nan(None),
            pl.Series(family_size_column, sizes, dtype=pl.Int64),
        )

    # ponytail: temporary pandas adapter, removed in task 3D
    required = {family_column, status_column, p_column, correction_column}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError("Missing multiple-testing columns: " + ", ".join(missing))

    resolved = method if isinstance(method, PAdjustMethod) else PAdjustMethod(method)
    result = table.copy(deep=True)
    if q_column not in result:
        result[q_column] = np.nan
    else:
        result[q_column] = pd.to_numeric(result[q_column], errors="coerce")
    result[family_size_column] = 0

    if result.empty:
        return result
    families = result[family_column]
    if families.isna().any() or families.astype(str).str.strip().eq("").any():
        raise ValueError("family_id values must be present and non-empty.")
    if not result[correction_column].astype(str).eq(resolved.value).all():
        raise ValueError("correction column is inconsistent with configured method.")

    normalized = result[family_column].astype(str)
    numeric_p = pd.to_numeric(result[p_column], errors="coerce")
    finite_p = np.isfinite(numeric_p.to_numpy(dtype=np.float64))
    for family_id in normalized.drop_duplicates().tolist():
        family_mask = normalized.eq(family_id)
        inferential = family_mask & result[status_column].eq("ok") & finite_p
        family_size = int(inferential.sum())
        result.loc[family_mask, family_size_column] = family_size
        if not bool(inferential.any()):
            continue
        p_values = numeric_p.loc[inferential].to_numpy(dtype=np.float64)
        result.loc[inferential, q_column] = adjust_pvalues(p_values, resolved)
    return result
