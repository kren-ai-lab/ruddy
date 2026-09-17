"""Standalone distribution and grouped dispersion diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import normal_ad

from ruddy.core.enums import ColumnKind, ColumnRole, PAdjustMethod
from ruddy.data import TabularDataset
from ruddy.results import AnalysisProvenance
from ruddy.statistics import apply_multiple_testing
from ruddy.univariate.categorical import _category_label

NORMALITY_COLUMNS = (
    "method",
    "column",
    "role",
    "n_total",
    "n_finite",
    "statistic",
    "p_value",
    "q_value",
    "family_id",
    "family_size",
    "correction",
    "status",
    "reason",
)
DISPERSION_COLUMNS = (
    "method",
    "response",
    "group",
    "n_total",
    "n_used",
    "n_groups",
    "group_sizes_json",
    "statistic",
    "p_value",
    "q_value",
    "family_id",
    "family_size",
    "correction",
    "status",
    "reason",
)


@dataclass(frozen=True, slots=True)
class DistributionDiagnosticsResult:
    normality: pd.DataFrame
    dispersion: pd.DataFrame
    provenance: AnalysisProvenance


def _eligible_numeric(dataset: TabularDataset) -> tuple[str, ...]:
    return tuple(
        spec.name
        for spec in dataset.schema
        if spec.kind is ColumnKind.NUMERIC
        and spec.role not in {ColumnRole.IDENTIFIER, ColumnRole.EXCLUDED, ColumnRole.FACTOR}
    )


def _eligible_groups(dataset: TabularDataset) -> tuple[str, ...]:
    return tuple(
        spec.name
        for spec in dataset.schema
        if spec.role not in {ColumnRole.IDENTIFIER, ColumnRole.EXCLUDED}
        and (spec.role is ColumnRole.FACTOR or spec.kind in {ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN})
    )


def _finite(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(dtype=float, copy=True)
    return values[np.isfinite(values)]


def summarize_normality_diagnostics(
    dataset: TabularDataset,
    *,
    methods: tuple[str, ...] = ("shapiro", "dagostino", "anderson_darling"),
    columns: tuple[str, ...] | None = None,
    max_shapiro_n: int = 5000,
    p_adjust: PAdjustMethod | str = PAdjustMethod.FDR_BH,
) -> pd.DataFrame:
    """Compute explicit normality diagnostics without altering downstream test choice."""
    allowed = {"shapiro", "dagostino", "anderson_darling"}
    methods = tuple(str(method).lower() for method in methods)
    if not methods or len(set(methods)) != len(methods) or any(m not in allowed for m in methods):
        raise ValueError("methods must be unique values from shapiro, dagostino, anderson_darling.")
    if max_shapiro_n < 3:
        raise ValueError("max_shapiro_n must be at least 3.")
    correction = p_adjust if isinstance(p_adjust, PAdjustMethod) else PAdjustMethod(p_adjust)
    candidates = _eligible_numeric(dataset)
    selected = candidates if columns is None else tuple(str(c) for c in columns)
    invalid = [c for c in selected if c not in candidates]
    if invalid:
        raise ValueError(f"Normality diagnostics require eligible numeric columns: {invalid}.")

    frame = dataset.to_frame()
    rows: list[dict[str, Any]] = []
    for method in methods:
        family_id = f"normality:{method}"
        for column in selected:
            x = _finite(frame[column])
            row: dict[str, Any] = {
                "method": method,
                "column": column,
                "role": dataset.role_of(column).value,
                "n_total": dataset.n_observations,
                "n_finite": int(x.size),
                "statistic": np.nan,
                "p_value": np.nan,
                "q_value": np.nan,
                "family_id": family_id,
                "family_size": 0,
                "correction": correction.value,
                "status": "ok",
                "reason": None,
            }
            minimum = 3 if method != "dagostino" else 8
            if x.size < minimum:
                row["status"] = "skipped"
                row["reason"] = "insufficient_finite_observations"
            elif np.ptp(x) == 0:
                row["status"] = "degenerate"
                row["reason"] = "constant"
            elif method == "shapiro" and x.size > max_shapiro_n:
                row["status"] = "skipped"
                row["reason"] = "sample_size_exceeds_shapiro_limit"
            else:
                if method == "shapiro":
                    result = stats.shapiro(x)
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                elif method == "dagostino":
                    result = stats.normaltest(x)
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                else:
                    statistic, p_value = (float(value) for value in normal_ad(x))
                if not np.isfinite(statistic) or not np.isfinite(p_value):
                    row["status"] = "degenerate"
                    row["reason"] = "undefined_diagnostic"
                else:
                    row["statistic"] = statistic
                    row["p_value"] = p_value
            rows.append(row)
    table = pd.DataFrame(rows, columns=pd.Index(NORMALITY_COLUMNS))
    return table if table.empty else apply_multiple_testing(table, correction)


def summarize_dispersion_diagnostics(
    dataset: TabularDataset,
    *,
    responses: tuple[str, ...],
    groups: tuple[str, ...],
    methods: tuple[str, ...] = ("brown_forsythe", "fligner_killeen"),
    min_group_n: int = 2,
    max_group_levels: int = 20,
    p_adjust: PAdjustMethod | str = PAdjustMethod.FDR_BH,
) -> pd.DataFrame:
    """Assess grouped dispersion using explicit robust tests."""
    import json

    allowed = {"brown_forsythe", "fligner_killeen"}
    methods = tuple(str(method).lower() for method in methods)
    if not methods or len(set(methods)) != len(methods) or any(m not in allowed for m in methods):
        raise ValueError("methods must be unique brown_forsythe/fligner_killeen values.")
    if min_group_n < 2:
        raise ValueError("min_group_n must be at least 2.")
    numeric = set(_eligible_numeric(dataset))
    groupable = set(_eligible_groups(dataset))
    bad_responses = [r for r in responses if r not in numeric]
    bad_groups = [g for g in groups if g not in groupable]
    if bad_responses:
        raise ValueError(f"Dispersion responses must be eligible numeric columns: {bad_responses}.")
    if bad_groups:
        raise ValueError(f"Dispersion groups must be eligible categorical/factor columns: {bad_groups}.")
    correction = p_adjust if isinstance(p_adjust, PAdjustMethod) else PAdjustMethod(p_adjust)
    frame = dataset.to_frame()
    rows: list[dict[str, Any]] = []
    for method in methods:
        for response in responses:
            family_id = f"dispersion:{method}:{response}"
            for group in groups:
                pair = frame[[response, group]].dropna(subset=[group]).copy()
                pair["__value"] = pd.to_numeric(pair[response], errors="coerce")
                pair = pair[np.isfinite(pair["__value"].to_numpy(dtype=float))]
                labels = pair[group].map(_category_label).astype(str)
                levels = sorted(labels.unique().tolist())
                samples = tuple(
                    pair.loc[labels.eq(level), "__value"].to_numpy(dtype=float) for level in levels
                )
                sizes = {level: int(sample.size) for level, sample in zip(levels, samples, strict=True)}
                row: dict[str, Any] = {
                    "method": method,
                    "response": response,
                    "group": group,
                    "n_total": dataset.n_observations,
                    "n_used": len(pair),
                    "n_groups": len(levels),
                    "group_sizes_json": json.dumps(sizes, sort_keys=True),
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "q_value": np.nan,
                    "family_id": family_id,
                    "family_size": 0,
                    "correction": correction.value,
                    "status": "ok",
                    "reason": None,
                }
                if len(levels) < 2:
                    row["status"] = "degenerate"
                    row["reason"] = "insufficient_group_levels"
                elif len(levels) > max_group_levels:
                    row["status"] = "skipped"
                    row["reason"] = "group_levels_exceed_max_group_levels"
                elif any(sample.size < min_group_n for sample in samples):
                    row["status"] = "skipped"
                    row["reason"] = "group_too_small"
                elif all(np.ptp(sample) == 0 for sample in samples):
                    row["status"] = "degenerate"
                    row["reason"] = "all_groups_constant"
                else:
                    result = (
                        stats.levene(*samples, center="median")
                        if method == "brown_forsythe"
                        else stats.fligner(*samples, center="median")
                    )
                    row["statistic"] = float(result.statistic)
                    row["p_value"] = float(result.pvalue)
                rows.append(row)
    table = pd.DataFrame(rows, columns=pd.Index(DISPERSION_COLUMNS))
    return table if table.empty else apply_multiple_testing(table, correction)


def analyze_distribution_diagnostics(
    dataset: TabularDataset,
    *,
    responses: tuple[str, ...] = (),
    groups: tuple[str, ...] = (),
    p_adjust: PAdjustMethod | str = PAdjustMethod.FDR_BH,
    min_group_n: int = 2,
    max_group_levels: int = 20,
    max_shapiro_n: int = 5000,
) -> DistributionDiagnosticsResult:
    """Run standalone distribution diagnostics; results never select another test automatically."""
    normality = summarize_normality_diagnostics(dataset, max_shapiro_n=max_shapiro_n, p_adjust=p_adjust)
    dispersion = (
        summarize_dispersion_diagnostics(
            dataset,
            responses=responses,
            groups=groups,
            min_group_n=min_group_n,
            max_group_levels=max_group_levels,
            p_adjust=p_adjust,
        )
        if responses and groups
        else pd.DataFrame(columns=pd.Index(DISPERSION_COLUMNS))
    )
    correction = p_adjust if isinstance(p_adjust, PAdjustMethod) else PAdjustMethod(p_adjust)
    provenance = AnalysisProvenance(
        analysis="distribution_diagnostics",
        parameters={
            "normality_methods": ("shapiro", "dagostino", "anderson_darling"),
            "dispersion_methods": ("brown_forsythe", "fligner_killeen"),
            "responses": tuple(responses),
            "groups": tuple(groups),
            "p_adjust": correction.value,
            "min_group_n": min_group_n,
            "max_group_levels": max_group_levels,
            "max_shapiro_n": max_shapiro_n,
            "automatic_test_selection": False,
        },
        input_summary={"n_observations": dataset.n_observations, "n_columns": dataset.n_columns},
    )
    return DistributionDiagnosticsResult(normality=normality, dispersion=dispersion, provenance=provenance)
