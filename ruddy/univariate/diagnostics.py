"""Standalone distribution and grouped dispersion diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from scipy import stats
from statsmodels.stats.diagnostic import normal_ad

from ruddy.core.enums import ColumnKind, ColumnRole, PAdjustMethod
from ruddy.core.frames import present_mask, to_float_array
from ruddy.results import AnalysisProvenance
from ruddy.statistics import apply_multiple_testing
from ruddy.univariate.categorical import _category_label

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

    from ruddy.data import TabularDataset

NORMALITY_SCHEMA: dict[str, PolarsDataType] = {
    "method": pl.String,
    "column": pl.String,
    "role": pl.String,
    "n_total": pl.Int64,
    "n_finite": pl.Int64,
    "statistic": pl.Float64,
    "p_value": pl.Float64,
    "q_value": pl.Float64,
    "family_id": pl.String,
    "family_size": pl.Int64,
    "correction": pl.String,
    "status": pl.String,
    "reason": pl.String,
}
NORMALITY_COLUMNS: tuple[str, ...] = tuple(NORMALITY_SCHEMA)

DISPERSION_SCHEMA: dict[str, PolarsDataType] = {
    "method": pl.String,
    "response": pl.String,
    "group": pl.String,
    "n_total": pl.Int64,
    "n_used": pl.Int64,
    "n_groups": pl.Int64,
    "group_sizes_json": pl.String,
    "statistic": pl.Float64,
    "p_value": pl.Float64,
    "q_value": pl.Float64,
    "family_id": pl.String,
    "family_size": pl.Int64,
    "correction": pl.String,
    "status": pl.String,
    "reason": pl.String,
}
DISPERSION_COLUMNS: tuple[str, ...] = tuple(DISPERSION_SCHEMA)


@dataclass(frozen=True, slots=True)
class DistributionDiagnosticsResult:
    """Results of univariate distribution diagnostics including normality and dispersion."""

    normality: pl.DataFrame
    dispersion: pl.DataFrame
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


def _finite(series: pl.Series) -> np.ndarray:
    values = to_float_array(series)
    return values[np.isfinite(values)]


def summarize_normality_diagnostics(
    dataset: TabularDataset,
    *,
    methods: tuple[str, ...] = ("shapiro", "dagostino", "anderson_darling"),
    columns: tuple[str, ...] | None = None,
    max_shapiro_n: int = 5000,
    p_adjust: PAdjustMethod | str = PAdjustMethod.FDR_BH,
) -> pl.DataFrame:
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

    frame = dataset.frame
    rows: list[dict[str, Any]] = []
    for method in methods:
        family_id = f"normality:{method}"
        for column in selected:
            x = _finite(frame.get_column(column))
            row: dict[str, Any] = {
                "method": method,
                "column": column,
                "role": dataset.role_of(column).value,
                "n_total": dataset.frame.height,
                "n_finite": int(x.size),
                "statistic": None,
                "p_value": None,
                "q_value": None,
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
    table = pl.DataFrame(rows, schema=NORMALITY_SCHEMA)
    return table if table.height == 0 else apply_multiple_testing(table, correction)


def summarize_dispersion_diagnostics(
    dataset: TabularDataset,
    *,
    responses: tuple[str, ...],
    groups: tuple[str, ...],
    methods: tuple[str, ...] = ("brown_forsythe", "fligner_killeen"),
    min_group_n: int = 2,
    max_group_levels: int = 20,
    p_adjust: PAdjustMethod | str = PAdjustMethod.FDR_BH,
) -> pl.DataFrame:
    """Assess grouped dispersion using explicit robust tests."""
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
    frame = dataset.frame
    rows: list[dict[str, Any]] = []
    for method in methods:
        for response in responses:
            family_id = f"dispersion:{method}:{response}"
            for group in groups:
                mask = present_mask(frame.get_column(group))
                pair = frame.select(response, group).filter(mask)
                raw_values = pair.get_column(response).cast(pl.Float64).fill_null(float("nan")).to_numpy()
                finite_mask = np.isfinite(raw_values)
                values = raw_values[finite_mask]
                all_labels = np.array(
                    [_category_label(v) for v in pair.get_column(group).to_list()],
                    dtype=object,
                )
                labels = all_labels[finite_mask]
                levels = sorted(set(labels.tolist()))
                samples = tuple(values[labels == level] for level in levels)
                sizes = {level: int(sample.size) for level, sample in zip(levels, samples, strict=True)}
                row: dict[str, Any] = {
                    "method": method,
                    "response": response,
                    "group": group,
                    "n_total": dataset.frame.height,
                    "n_used": int(values.size),
                    "n_groups": len(levels),
                    "group_sizes_json": json.dumps(sizes, sort_keys=True),
                    "statistic": None,
                    "p_value": None,
                    "q_value": None,
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
    table = pl.DataFrame(rows, schema=DISPERSION_SCHEMA)
    return table if table.height == 0 else apply_multiple_testing(table, correction)


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
        else pl.DataFrame(schema=DISPERSION_SCHEMA)
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
        input_summary={"n_observations": dataset.frame.height, "n_columns": dataset.frame.width},
    )
    return DistributionDiagnosticsResult(normality=normality, dispersion=dispersion, provenance=provenance)
