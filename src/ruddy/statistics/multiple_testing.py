"""Deterministic multiple-testing correction utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
    table: pd.DataFrame,
    *,
    family_column: str = "family_id",
    status_column: str = "status",
    ok_status: str = "ok",
) -> dict[str, int]:
    """Count inferential hypotheses in each explicitly declared family."""

    if table.empty:
        return {}
    mask = table[status_column].eq(ok_status)
    families = table.loc[mask, family_column].astype(str)
    return {
        family: int((families == family).sum())
        for family in families.drop_duplicates().tolist()
    }


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
) -> pd.DataFrame:
    """Adjust p-values independently within explicit hypothesis families.

    Only ``status == 'ok'`` rows belong to the inferential family. Non-ok rows
    retain missing q-values and a family size of zero if their family has no
    successful hypotheses.
    """

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

    sizes = family_sizes(result, family_column=family_column, status_column=status_column)
    normalized = result[family_column].astype(str)
    for family_id in normalized.drop_duplicates().tolist():
        family_mask = normalized.eq(family_id)
        result.loc[family_mask, family_size_column] = sizes.get(family_id, 0)
        inferential = family_mask & result[status_column].eq("ok")
        if not bool(inferential.any()):
            continue
        p_values = result.loc[inferential, p_column].to_numpy(dtype=np.float64)
        result.loc[inferential, q_column] = adjust_pvalues(p_values, resolved)
    return result
