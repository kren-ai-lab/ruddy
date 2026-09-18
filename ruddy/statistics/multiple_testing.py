"""Deterministic multiple-testing correction utilities."""

from __future__ import annotations

import numpy as np
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
    table: pl.DataFrame,
    *,
    family_column: str = "family_id",
    status_column: str = "status",
    ok_status: str = "ok",
) -> dict[str, int]:
    """Count inferential hypotheses in each explicitly declared family."""
    if not isinstance(table, pl.DataFrame):
        raise TypeError("table must be a Polars DataFrame.")
    if table.height == 0:
        return {}
    families = (
        table.filter(pl.col(status_column) == ok_status).get_column(family_column).cast(pl.String).to_list()
    )
    counts: dict[str, int] = {}
    for family in families:
        counts[family] = counts.get(family, 0) + 1
    return counts


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
) -> pl.DataFrame:
    """Adjust p-values independently within explicit hypothesis families.

    Only ``status == 'ok'`` rows with finite p-values belong to the inferential
    family. This permits descriptive/statistic-only runs (for example zero
    permutations) to retain ``status='ok'`` while leaving p/q values missing.
    Non-inferential rows retain missing q-values.
    """
    if not isinstance(table, pl.DataFrame):
        raise TypeError("table must be a Polars DataFrame.")
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
