"""Factorial model diagnostics and pathological-design accounting."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

from ruddy.core.enums import ResultStatus

DIAGNOSTIC_COLUMNS = (
    "diagnostic",
    "statistic",
    "p_value",
    "alpha",
    "flagged",
    "details",
    "status",
    "reason",
)

OBSERVATION_DIAGNOSTIC_COLUMNS = (
    "source_row_index",
    "observation_id",
    "fitted_value",
    "residual",
    "studentized_residual",
    "leverage",
    "cooks_distance",
    "large_residual_threshold",
    "high_leverage_threshold",
    "cooks_distance_threshold",
    "is_large_residual",
    "is_high_leverage",
    "is_influential",
    "status",
    "reason",
)


def _finite_or_none(value: object) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def build_factorial_cells(
    frame: pd.DataFrame,
    factors: Sequence[str],
    *,
    min_cell_n: int = 2,
    max_design_cells: int = 5000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Enumerate observed and structurally empty cells for selected factors."""
    if min_cell_n < 1:
        raise ValueError("min_cell_n must be at least 1.")
    if max_design_cells < 1:
        raise ValueError("max_design_cells must be at least 1.")
    factor_names = tuple(str(name) for name in factors)
    if not factor_names:
        return (
            pd.DataFrame(columns=("n", "is_empty", "below_min_cell_n", "status", "reason")),
            {
                "n_design_cells": 0,
                "n_observed_cells": 0,
                "n_empty_cells": 0,
                "n_small_cells": 0,
                "balanced": True,
            },
        )

    levels: list[tuple[Any, ...]] = []
    total_cells = 1
    for factor in factor_names:
        observed = tuple(pd.unique(frame[factor]))
        total_cells *= len(observed)
        if total_cells > max_design_cells:
            raise ValueError(f"Factorial cell enumeration exceeds max_design_cells={max_design_cells}.")
        levels.append(observed)

    observed_counts = frame.groupby(list(factor_names), dropna=False, observed=True).size()
    rows: list[dict[str, Any]] = []
    nonempty_counts: list[int] = []
    for combo in product(*levels):
        if len(factor_names) == 1:
            key: Any = combo[0]
        else:
            key = tuple(combo)
        try:
            count = int(observed_counts.get(key, 0))
        except TypeError:
            # Defensive fallback for mixed object categories that are awkward as an index key.
            mask = np.ones(len(frame), dtype=bool)
            for factor, level in zip(factor_names, combo, strict=True):
                mask &= frame[factor].eq(level).to_numpy()
            count = int(mask.sum())
        empty = count == 0
        small = 0 < count < min_cell_n
        if not empty:
            nonempty_counts.append(count)
        row = {factor: level for factor, level in zip(factor_names, combo, strict=True)}
        row.update(
            n=count,
            is_empty=empty,
            below_min_cell_n=small,
            status=(ResultStatus.DEGENERATE.value if empty or small else ResultStatus.OK.value),
            reason=("empty_factorial_cell" if empty else ("factorial_cell_too_small" if small else None)),
        )
        rows.append(row)

    table = pd.DataFrame(
        rows,
        columns=(*factor_names, "n", "is_empty", "below_min_cell_n", "status", "reason"),
    )
    n_empty = int(table["is_empty"].sum()) if not table.empty else 0
    n_small = int(table["below_min_cell_n"].sum()) if not table.empty else 0
    balanced = n_empty == 0 and (len(set(nonempty_counts)) <= 1)
    return table, {
        "n_design_cells": len(table),
        "n_observed_cells": len(nonempty_counts),
        "n_empty_cells": n_empty,
        "n_small_cells": n_small,
        "balanced": bool(balanced),
    }


def model_diagnostics(
    model,
    *,
    model_frame: pd.DataFrame,
    source_row_indices: np.ndarray,
    observation_ids: pd.Index,
    factors: Sequence[str],
    alpha: float = 0.05,
    condition_number_threshold: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute non-decision-making residual/design diagnostics for a fitted OLS model."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("diagnostic alpha must lie in (0, 1).")
    if condition_number_threshold <= 0.0:
        raise ValueError("condition_number_threshold must be greater than zero.")

    residuals = np.asarray(model.resid, dtype=np.float64)
    fitted = np.asarray(model.fittedvalues, dtype=np.float64)
    exog = np.asarray(model.model.exog, dtype=np.float64)
    n = residuals.size
    rows: list[dict[str, Any]] = []

    def append(
        diagnostic: str,
        *,
        statistic: float | None = None,
        p_value: float | None = None,
        flagged: bool | None = None,
        details: str | None = None,
        status: ResultStatus = ResultStatus.OK,
        reason: str | None = None,
    ) -> None:
        rows.append(
            {
                "diagnostic": diagnostic,
                "statistic": statistic,
                "p_value": p_value,
                "alpha": alpha if p_value is not None else None,
                "flagged": flagged,
                "details": details,
                "status": status.value,
                "reason": reason,
            }
        )

    if 3 <= n <= 5000:
        shapiro = stats.shapiro(residuals)
        p = _finite_or_none(shapiro.pvalue)
        append(
            "shapiro_wilk_residual_normality",
            statistic=_finite_or_none(shapiro.statistic),
            p_value=p,
            flagged=None if p is None else bool(p < alpha),
            details="Residual normality diagnostic; does not select or replace the fitted model.",
        )
    elif n > 5000:
        append(
            "shapiro_wilk_residual_normality",
            status=ResultStatus.SKIPPED,
            reason="sample_size_exceeds_shapiro_limit",
            details="Shapiro-Wilk is intentionally not reported beyond n=5000.",
        )
    else:
        append(
            "shapiro_wilk_residual_normality",
            status=ResultStatus.SKIPPED,
            reason="insufficient_residual_observations",
        )

    if n >= 8:
        jb_stat, jb_p, skew, kurtosis = jarque_bera(residuals)
        p = _finite_or_none(jb_p)
        append(
            "jarque_bera_residual_normality",
            statistic=_finite_or_none(jb_stat),
            p_value=p,
            flagged=None if p is None else bool(p < alpha),
            details=f"skew={float(skew):.12g}; kurtosis={float(kurtosis):.12g}",
        )
    else:
        append(
            "jarque_bera_residual_normality",
            status=ResultStatus.SKIPPED,
            reason="insufficient_residual_observations",
        )

    if n > exog.shape[1] and exog.shape[1] >= 2:
        try:
            lm_stat, lm_p, f_stat, f_p = het_breuschpagan(residuals, exog)
            p = _finite_or_none(lm_p)
            append(
                "breusch_pagan_heteroskedasticity",
                statistic=_finite_or_none(lm_stat),
                p_value=p,
                flagged=None if p is None else bool(p < alpha),
                details=f"f_value={float(f_stat):.12g}; f_p_value={float(f_p):.12g}",
            )
        except (ValueError, np.linalg.LinAlgError):
            append(
                "breusch_pagan_heteroskedasticity",
                status=ResultStatus.DEGENERATE,
                reason="breusch_pagan_not_estimable",
            )
    else:
        append(
            "breusch_pagan_heteroskedasticity",
            status=ResultStatus.SKIPPED,
            reason="insufficient_residual_degrees_of_freedom",
        )

    factor_names = tuple(str(name) for name in factors)
    if factor_names:
        grouped_residuals: list[np.ndarray] = []
        cell_frame = model_frame.loc[:, list(factor_names)].copy()
        cell_frame["__residual__"] = residuals
        for _key, group in cell_frame.groupby(list(factor_names), observed=True, sort=False):
            values = group["__residual__"].to_numpy(dtype=np.float64)
            if values.size >= 2:
                grouped_residuals.append(values)
        if len(grouped_residuals) >= 2:
            levene = stats.levene(*grouped_residuals, center="median")
            p = _finite_or_none(levene.pvalue)
            append(
                "levene_median_residual_homoscedasticity",
                statistic=_finite_or_none(levene.statistic),
                p_value=p,
                flagged=None if p is None else bool(p < alpha),
                details=f"n_cells_tested={len(grouped_residuals)}",
            )
        else:
            append(
                "levene_median_residual_homoscedasticity",
                status=ResultStatus.SKIPPED,
                reason="insufficient_factor_cells_for_levene",
            )
    else:
        append(
            "levene_median_residual_homoscedasticity",
            status=ResultStatus.SKIPPED,
            reason="no_factor_terms",
        )

    singular_values = np.linalg.svd(exog, compute_uv=False)
    positive = singular_values[singular_values > np.finfo(float).eps * max(exog.shape) * singular_values[0]]
    condition = float(positive.max() / positive.min()) if positive.size else math.inf
    append(
        "design_condition_number",
        statistic=_finite_or_none(condition),
        flagged=bool(math.isfinite(condition) and condition > condition_number_threshold),
        details=f"threshold={condition_number_threshold:.12g}",
    )

    try:
        influence = model.get_influence()
        leverage = np.asarray(influence.hat_matrix_diag, dtype=np.float64)
        studentized = np.asarray(influence.resid_studentized_internal, dtype=np.float64)
        cooks = np.asarray(influence.cooks_distance[0], dtype=np.float64)
        p_design = exog.shape[1]
        leverage_threshold = float(min(1.0, 2.0 * p_design / n)) if n else math.nan
        cooks_threshold = float(4.0 / n) if n else math.nan
        residual_threshold = 3.0
        obs_rows: list[dict[str, Any]] = []
        selected_ids = observation_ids.take(source_row_indices)
        for index in range(n):
            studentized_value = _finite_or_none(studentized[index])
            leverage_value = _finite_or_none(leverage[index])
            cooks_value = _finite_or_none(cooks[index])
            obs_rows.append(
                {
                    "source_row_index": int(source_row_indices[index]),
                    "observation_id": selected_ids[index],
                    "fitted_value": _finite_or_none(fitted[index]),
                    "residual": _finite_or_none(residuals[index]),
                    "studentized_residual": studentized_value,
                    "leverage": leverage_value,
                    "cooks_distance": cooks_value,
                    "large_residual_threshold": residual_threshold,
                    "high_leverage_threshold": leverage_threshold,
                    "cooks_distance_threshold": cooks_threshold,
                    "is_large_residual": (
                        None if studentized_value is None else abs(studentized_value) > residual_threshold
                    ),
                    "is_high_leverage": (
                        None if leverage_value is None else leverage_value > leverage_threshold
                    ),
                    "is_influential": (None if cooks_value is None else cooks_value > cooks_threshold),
                    "status": ResultStatus.OK.value,
                    "reason": None,
                }
            )
        observations = pd.DataFrame(obs_rows, columns=OBSERVATION_DIAGNOSTIC_COLUMNS)
    except (ValueError, np.linalg.LinAlgError):
        observations = pd.DataFrame(columns=OBSERVATION_DIAGNOSTIC_COLUMNS)

    return pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS), observations


__all__ = [
    "DIAGNOSTIC_COLUMNS",
    "OBSERVATION_DIAGNOSTIC_COLUMNS",
    "build_factorial_cells",
    "model_diagnostics",
]
