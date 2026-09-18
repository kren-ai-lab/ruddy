"""High-level confidence intervals for common Ruddy EDA estimands."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import polars as pl
from polars._typing import PolarsDataType
from scipy import stats

from ruddy.bivariate.associations import _contingency, _eligible_categorical
from ruddy.core.enums import ColumnKind, ColumnRole, CorrelationMethod
from ruddy.data import TabularDataset
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

MEANS_SCHEMA: dict[str, PolarsDataType] = {
    "column": pl.String,
    "n": pl.Int64,
    "estimate": pl.Float64,
    "standard_error": pl.Float64,
    "confidence_level": pl.Float64,
    "confidence_low": pl.Float64,
    "confidence_high": pl.Float64,
    "method": pl.String,
    "status": pl.String,
    "reason": pl.String,
}
MEANS_COLUMNS: tuple[str, ...] = tuple(MEANS_SCHEMA)

CORRELATION_CI_SCHEMA: dict[str, PolarsDataType] = {
    "method": pl.String,
    "column_x": pl.String,
    "column_y": pl.String,
    "n": pl.Int64,
    "estimate": pl.Float64,
    "confidence_level": pl.Float64,
    "confidence_low": pl.Float64,
    "confidence_high": pl.Float64,
    "ci_method": pl.String,
    "status": pl.String,
    "reason": pl.String,
}
CORRELATION_CI_COLUMNS: tuple[str, ...] = tuple(CORRELATION_CI_SCHEMA)

MEAN_DIFFERENCE_SCHEMA: dict[str, PolarsDataType] = {
    "response": pl.String,
    "group": pl.String,
    "level_a": pl.String,
    "level_b": pl.String,
    "n_a": pl.Int64,
    "n_b": pl.Int64,
    "confidence_level": pl.Float64,
    "estimate": pl.Float64,
    "standard_error": pl.Float64,
    "df": pl.Float64,
    "confidence_low": pl.Float64,
    "confidence_high": pl.Float64,
    "method": pl.String,
    "status": pl.String,
    "reason": pl.String,
}
MEAN_DIFFERENCE_COLUMNS: tuple[str, ...] = tuple(MEAN_DIFFERENCE_SCHEMA)

EFFECT_SIZE_SCHEMA: dict[str, PolarsDataType] = {
    "response": pl.String,
    "group": pl.String,
    "level_a": pl.String,
    "level_b": pl.String,
    "n_a": pl.Int64,
    "n_b": pl.Int64,
    "confidence_level": pl.Float64,
    "estimate": pl.Float64,
    "confidence_low": pl.Float64,
    "confidence_high": pl.Float64,
    "method": pl.String,
    "status": pl.String,
    "reason": pl.String,
}
EFFECT_SIZE_COLUMNS: tuple[str, ...] = tuple(EFFECT_SIZE_SCHEMA)

ODDS_RATIO_SCHEMA: dict[str, PolarsDataType] = {
    "column_x": pl.String,
    "column_y": pl.String,
    "level_x_0": pl.String,
    "level_x_1": pl.String,
    "level_y_0": pl.String,
    "level_y_1": pl.String,
    "n_used": pl.Int64,
    "estimate": pl.Float64,
    "confidence_level": pl.Float64,
    "confidence_low": pl.Float64,
    "confidence_high": pl.Float64,
    "method": pl.String,
    "status": pl.String,
    "reason": pl.String,
}
ODDS_RATIO_COLUMNS: tuple[str, ...] = tuple(ODDS_RATIO_SCHEMA)


@dataclass(frozen=True, slots=True)
class ConfidenceIntervalResult:
    means: pl.DataFrame
    correlations: pl.DataFrame
    mean_differences: pl.DataFrame
    effect_sizes: pl.DataFrame
    odds_ratios: pl.DataFrame
    provenance: AnalysisProvenance


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


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


def _correlation_statistic(method: CorrelationMethod):
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
        raise ValueError("confidence_level must lie in (0, 1).")
    if bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100.")
    methods = tuple(
        m if isinstance(m, CorrelationMethod) else CorrelationMethod(m) for m in correlation_methods
    )
    if len(set(methods)) != len(methods):
        raise ValueError("correlation_methods cannot contain duplicates.")
    numeric = _numeric(dataset)
    categorical = _eligible_categorical(dataset)
    frame = dataset.frame

    mean_rows: list[dict[str, Any]] = []
    for column in numeric:
        x = frame.get_column(column).cast(pl.Float64).fill_null(float("nan")).to_numpy()
        x = x[np.isfinite(x)]
        estimate, se, low, high = mean_confidence_interval(x, confidence_level)
        status = "ok" if np.isfinite(low) else "skipped"
        mean_rows.append(
            {
                "column": column,
                "n": int(x.size),
                "estimate": _safe_float(estimate) if status == "ok" else None,
                "standard_error": _safe_float(se) if status == "ok" else None,
                "confidence_level": confidence_level,
                "confidence_low": _safe_float(low) if status == "ok" else None,
                "confidence_high": _safe_float(high) if status == "ok" else None,
                "method": "student_t",
                "status": status,
                "reason": None if status == "ok" else "insufficient_finite_observations",
            }
        )

    correlation_rows: list[dict[str, Any]] = []
    for method in methods:
        for x_name, y_name in combinations(numeric, 2):
            values = frame.select(x_name, y_name).cast(pl.Float64).fill_null(float("nan")).to_numpy()
            values = values[np.all(np.isfinite(values), axis=1)]
            n = int(values.shape[0])
            row: dict[str, Any] = {
                "method": method.value,
                "column_x": x_name,
                "column_y": y_name,
                "n": n,
                "estimate": None,
                "confidence_level": confidence_level,
                "confidence_low": None,
                "confidence_high": None,
                "ci_method": None,
                "status": "ok",
                "reason": None,
            }
            if n < 4 or np.ptp(values[:, 0]) == 0.0 or np.ptp(values[:, 1]) == 0.0:
                row["status"] = "skipped"
                row["reason"] = "insufficient_or_constant_pair"
            else:
                raw_estimate = _correlation_statistic(method)(values[:, 0], values[:, 1])
                estimate = _safe_float(raw_estimate)
                row["estimate"] = estimate
                if method is CorrelationMethod.PEARSON:
                    if estimate is not None:
                        low, high = pearson_confidence_interval(estimate, n, confidence_level)
                    else:
                        low, high = float("nan"), float("nan")
                    row["confidence_low"] = _safe_float(low)
                    row["confidence_high"] = _safe_float(high)
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
                    row["confidence_low"] = _safe_float(boot.confidence_low)
                    row["confidence_high"] = _safe_float(boot.confidence_high)
                    row["ci_method"] = f"bootstrap_{boot.method}"
                    row["status"] = boot.status
                    row["reason"] = boot.reason
            correlation_rows.append(row)

    difference_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    for group in categorical:
        group_series = frame.get_column(group)
        if group_series.dtype.is_float():
            present_mask = ~(group_series.is_null() | group_series.is_nan()).to_numpy()
        else:
            present_mask = ~group_series.is_null().to_numpy()

        group_list = group_series.to_list()
        present_indices = np.flatnonzero(present_mask)
        labels_present = [_category_label(group_list[i]) for i in present_indices]
        levels = sorted(set(labels_present))
        if len(levels) != 2 or len(levels) > max_category_levels:
            continue

        label_array = np.empty(len(group_list), dtype=object)
        for idx, label in zip(present_indices, labels_present, strict=True):
            label_array[idx] = label

        mask_a = present_mask & (label_array == levels[0])
        mask_b = present_mask & (label_array == levels[1])

        for response in numeric:
            resp_values = frame.get_column(response).cast(pl.Float64).fill_null(float("nan")).to_numpy()
            vals_a = resp_values[mask_a]
            vals_b = resp_values[mask_b]
            a = vals_a[np.isfinite(vals_a)]
            b = vals_b[np.isfinite(vals_b)]
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
                        "estimate": None,
                        "standard_error": None,
                        "df": None,
                        "confidence_low": None,
                        "confidence_high": None,
                        "method": "welch",
                        "status": "skipped",
                        "reason": "group_too_small",
                    }
                )
                effect_rows.append(
                    {
                        **base,
                        "estimate": None,
                        "confidence_low": None,
                        "confidence_high": None,
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
                    "estimate": _safe_float(estimate),
                    "standard_error": _safe_float(se),
                    "df": _safe_float(df),
                    "confidence_low": _safe_float(low),
                    "confidence_high": _safe_float(high),
                    "method": "welch",
                    "status": "ok",
                    "reason": None,
                }
            )
            effect = hedges_g(a, b)
            if effect is None or not np.isfinite(effect):
                effect_rows.append(
                    {
                        **base,
                        "estimate": None,
                        "confidence_low": None,
                        "confidence_high": None,
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
                        "estimate": _safe_float(effect),
                        "confidence_low": _safe_float(boot.confidence_low),
                        "confidence_high": _safe_float(boot.confidence_high),
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
                "estimate": _safe_float(estimate),
                "confidence_level": confidence_level,
                "confidence_low": _safe_float(low),
                "confidence_high": _safe_float(high),
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
        input_summary={"n_observations": frame.height, "n_columns": frame.width},
        random_state=random_state,
    )
    return ConfidenceIntervalResult(
        means=pl.DataFrame(mean_rows, schema=MEANS_SCHEMA),
        correlations=pl.DataFrame(correlation_rows, schema=CORRELATION_CI_SCHEMA),
        mean_differences=pl.DataFrame(difference_rows, schema=MEAN_DIFFERENCE_SCHEMA),
        effect_sizes=pl.DataFrame(effect_rows, schema=EFFECT_SIZE_SCHEMA),
        odds_ratios=pl.DataFrame(odds_rows, schema=ODDS_RATIO_SCHEMA),
        provenance=provenance,
    )
