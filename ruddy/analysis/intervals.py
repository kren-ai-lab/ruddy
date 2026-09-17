"""High-level confidence intervals for common Ruddy EDA estimands."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats

from ruddy.bivariate.associations import _contingency, _eligible_categorical
from ruddy.core.enums import ColumnKind, ColumnRole, CorrelationMethod
from ruddy.results import AnalysisProvenance
from ruddy.statistics.bootstrap import bootstrap_confidence_interval
from ruddy.statistics.confidence_intervals import (
    mean_confidence_interval,
    odds_ratio_confidence_interval,
    pearson_confidence_interval,
    welch_mean_difference_confidence_interval,
)
from ruddy.statistics.effect_sizes import hedges_g
from ruddy.univariate.categorical import _category_label

if TYPE_CHECKING:
    from collections.abc import Callable

    from ruddy.data import TabularDataset


@dataclass(frozen=True, slots=True)
class ConfidenceIntervalResult:
    """Confidence intervals for dataset estimands."""

    means: pd.DataFrame
    correlations: pd.DataFrame
    mean_differences: pd.DataFrame
    effect_sizes: pd.DataFrame
    odds_ratios: pd.DataFrame
    provenance: AnalysisProvenance


def _hedges_g_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    """Return Hedges' g, or NaN when it is not estimable."""
    value = hedges_g(x, y)
    return float(np.nan if value is None else value)


def _numeric(dataset: TabularDataset) -> tuple[str, ...]:
    return tuple(
        spec.name
        for spec in dataset.schema
        if spec.kind is ColumnKind.NUMERIC
        and spec.role not in {ColumnRole.IDENTIFIER, ColumnRole.EXCLUDED, ColumnRole.FACTOR}
    )


def _correlation_statistic(method: CorrelationMethod) -> Callable[[np.ndarray, np.ndarray], float]:
    if method is CorrelationMethod.PEARSON:
        return lambda x, y: float(stats.pearsonr(x, y).statistic)
    if method is CorrelationMethod.SPEARMAN:
        return lambda x, y: float(stats.spearmanr(x, y).statistic)
    return lambda x, y: float(stats.kendalltau(x, y).statistic)


def analyze_confidence_intervals(
    dataset: TabularDataset,
    *,
    confidence_level: float = 0.95,
    correlation_methods: tuple[CorrelationMethod | str, ...] = (CorrelationMethod.PEARSON,),
    bootstrap_resamples: int = 1000,
    bootstrap_method: str = "bca",
    min_group_n: int = 3,
    max_category_levels: int = 20,
    random_state: int = 0,
) -> ConfidenceIntervalResult:
    """Estimate CIs for means, correlations, binary mean differences/effects, and ORs."""
    if not 0.0 < confidence_level < 1.0:
        msg = "confidence_level must lie in (0, 1)."
        raise ValueError(msg)
    if bootstrap_resamples < 100:
        msg = "bootstrap_resamples must be at least 100."
        raise ValueError(msg)
    methods = tuple(
        m if isinstance(m, CorrelationMethod) else CorrelationMethod(m) for m in correlation_methods
    )
    if len(set(methods)) != len(methods):
        msg = "correlation_methods cannot contain duplicates."
        raise ValueError(msg)
    numeric = _numeric(dataset)
    categorical = _eligible_categorical(dataset)
    frame = dataset.to_frame()

    mean_rows: list[dict[str, Any]] = []
    for column in numeric:
        x = frame[column].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        estimate, se, low, high = mean_confidence_interval(x, confidence_level)
        status = "ok" if np.isfinite(low) else "skipped"
        mean_rows.append(
            {
                "column": column,
                "n": int(x.size),
                "estimate": estimate,
                "standard_error": se,
                "confidence_level": confidence_level,
                "confidence_low": low,
                "confidence_high": high,
                "method": "student_t",
                "status": status,
                "reason": None if status == "ok" else "insufficient_finite_observations",
            }
        )

    correlation_rows: list[dict[str, Any]] = []
    for method in methods:
        for x_name, y_name in combinations(numeric, 2):
            values = frame[[x_name, y_name]].to_numpy(dtype=float)
            values = values[np.all(np.isfinite(values), axis=1)]
            n = int(values.shape[0])
            row: dict[str, Any] = {
                "method": method.value,
                "column_x": x_name,
                "column_y": y_name,
                "n": n,
                "estimate": np.nan,
                "confidence_level": confidence_level,
                "confidence_low": np.nan,
                "confidence_high": np.nan,
                "ci_method": None,
                "status": "ok",
                "reason": None,
            }
            if n < 4 or np.ptp(values[:, 0]) == 0.0 or np.ptp(values[:, 1]) == 0.0:
                row["status"] = "skipped"
                row["reason"] = "insufficient_or_constant_pair"
            else:
                estimate = _correlation_statistic(method)(values[:, 0], values[:, 1])
                row["estimate"] = estimate
                if method is CorrelationMethod.PEARSON:
                    low, high = pearson_confidence_interval(estimate, n, confidence_level)
                    row["confidence_low"] = low
                    row["confidence_high"] = high
                    row["ci_method"] = "fisher_z"
                    if not np.isfinite(low):
                        row["status"] = "degenerate"
                        row["reason"] = "perfect_or_undefined_correlation"
                else:
                    boot = bootstrap_confidence_interval(
                        (values[:, 0], values[:, 1]),
                        _correlation_statistic(method),
                        paired=True,
                        confidence_level=confidence_level,
                        n_resamples=bootstrap_resamples,
                        method=bootstrap_method,
                        random_state=random_state,
                    )
                    row["confidence_low"] = boot.confidence_low
                    row["confidence_high"] = boot.confidence_high
                    row["ci_method"] = f"bootstrap_{boot.method}"
                    row["status"] = boot.status
                    row["reason"] = boot.reason
            correlation_rows.append(row)

    difference_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    for group in categorical:
        pair_base = frame[[group]].dropna()
        levels = sorted(pair_base[group].map(_category_label).astype(str).unique().tolist())
        if len(levels) != 2 or len(levels) > max_category_levels:
            continue
        for response in numeric:
            pair = frame[[response, group]].dropna(subset=[group]).copy()
            pair["__group"] = pair[group].map(_category_label).astype(str)
            pair["__value"] = pd.to_numeric(pair[response], errors="coerce")
            pair = pair[np.isfinite(pair["__value"].to_numpy(dtype=float))]
            a = pair.loc[pair["__group"].eq(levels[0]), "__value"].to_numpy(dtype=float)
            b = pair.loc[pair["__group"].eq(levels[1]), "__value"].to_numpy(dtype=float)
            base: dict[str, Any] = {
                "response": response,
                "group": group,
                "level_a": levels[0],
                "level_b": levels[1],
                "n_a": int(a.size),
                "n_b": int(b.size),
                "confidence_level": confidence_level,
            }
            if a.size < min_group_n or b.size < min_group_n:
                difference_rows.append(
                    {
                        **base,
                        "estimate": np.nan,
                        "standard_error": np.nan,
                        "df": np.nan,
                        "confidence_low": np.nan,
                        "confidence_high": np.nan,
                        "method": "welch",
                        "status": "skipped",
                        "reason": "group_too_small",
                    }
                )
                effect_rows.append(
                    {
                        **base,
                        "estimate": np.nan,
                        "confidence_low": np.nan,
                        "confidence_high": np.nan,
                        "method": f"hedges_g_bootstrap_{bootstrap_method}",
                        "status": "skipped",
                        "reason": "group_too_small",
                    }
                )
                continue
            estimate, se, df, low, high = welch_mean_difference_confidence_interval(a, b, confidence_level)
            difference_rows.append(
                {
                    **base,
                    "estimate": estimate,
                    "standard_error": se,
                    "df": df,
                    "confidence_low": low,
                    "confidence_high": high,
                    "method": "welch",
                    "status": "ok",
                    "reason": None,
                }
            )
            effect = hedges_g(a, b)
            if effect is None:
                effect_rows.append(
                    {
                        **base,
                        "estimate": np.nan,
                        "confidence_low": np.nan,
                        "confidence_high": np.nan,
                        "method": f"hedges_g_bootstrap_{bootstrap_method}",
                        "status": "degenerate",
                        "reason": "undefined_hedges_g",
                    }
                )
            else:
                boot = bootstrap_confidence_interval(
                    (a, b),
                    _hedges_g_or_nan,
                    confidence_level=confidence_level,
                    n_resamples=bootstrap_resamples,
                    method=bootstrap_method,
                    random_state=random_state,
                )
                effect_rows.append(
                    {
                        **base,
                        "estimate": effect,
                        "confidence_low": boot.confidence_low,
                        "confidence_high": boot.confidence_high,
                        "method": f"hedges_g_bootstrap_{boot.method}",
                        "status": boot.status,
                        "reason": boot.reason,
                    }
                )

    odds_rows: list[dict[str, Any]] = []
    for x, y in combinations(categorical, 2):
        x_levels, y_levels, counts, n_used = _contingency(frame, x, y)
        if counts.shape != (2, 2):
            continue
        estimate, low, high = odds_ratio_confidence_interval(counts, confidence_level)
        status = "ok" if np.isfinite(estimate) else "degenerate"
        odds_rows.append(
            {
                "column_x": x,
                "column_y": y,
                "level_x_0": x_levels[0],
                "level_x_1": x_levels[1],
                "level_y_0": y_levels[0],
                "level_y_1": y_levels[1],
                "n_used": n_used,
                "estimate": estimate,
                "confidence_level": confidence_level,
                "confidence_low": low,
                "confidence_high": high,
                "method": "log_wald_no_zero_correction",
                "status": status,
                "reason": None if status == "ok" else "zero_cell_or_invalid_table",
            }
        )

    provenance = AnalysisProvenance(
        analysis="confidence_intervals",
        parameters={
            "confidence_level": confidence_level,
            "correlation_methods": tuple(m.value for m in methods),
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_method": bootstrap_method,
            "min_group_n": min_group_n,
            "odds_ratio_zero_cell_correction": False,
            "random_state": random_state,
        },
        input_summary={"n_observations": dataset.n_observations, "n_columns": dataset.n_columns},
        random_state=random_state,
    )
    return ConfidenceIntervalResult(
        means=pd.DataFrame(mean_rows),
        correlations=pd.DataFrame(correlation_rows),
        mean_differences=pd.DataFrame(difference_rows),
        effect_sizes=pd.DataFrame(effect_rows),
        odds_ratios=pd.DataFrame(odds_rows),
        provenance=provenance,
    )
