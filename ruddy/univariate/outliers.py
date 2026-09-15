"""Transparent univariate statistical outlier and data-quality diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ruddy.core.enums import ColumnKind, ColumnRole, OutlierMethod, ResultStatus
from ruddy.data import TabularDataset
from ruddy.profiling import profile_columns
from ruddy.results import AnalysisProvenance
from ruddy.statistics.robust import (
    MODIFIED_Z_CONSISTENCY,
    median_absolute_deviation,
    modified_z_scores,
    tukey_fences,
)

OUTLIER_SUMMARY_COLUMNS: tuple[str, ...] = (
    "column",
    "role",
    "method",
    "n_total",
    "n_finite",
    "n_missing",
    "n_non_finite",
    "center",
    "scale",
    "lower_bound",
    "upper_bound",
    "criterion",
    "threshold",
    "n_flagged",
    "flagged_fraction",
    "status",
    "reason",
)

OUTLIER_FLAG_COLUMNS: tuple[str, ...] = (
    "column",
    "role",
    "method",
    "observation_id",
    "source_row_index",
    "value",
    "score",
    "direction",
    "threshold",
    "status",
    "reason",
)

NUMERIC_QUALITY_COLUMNS: tuple[str, ...] = (
    "column",
    "role",
    "n_total",
    "n_present",
    "n_finite",
    "n_missing",
    "n_non_finite",
    "n_unique_finite",
    "missing_fraction",
    "non_finite_fraction",
    "iqr",
    "mad",
    "has_missing",
    "has_non_finite",
    "zero_iqr_nonconstant",
    "zero_mad_nonconstant",
    "status",
    "reason",
)


@dataclass(frozen=True, slots=True)
class OutlierResult:
    """Complete Phase 5 univariate outlier and quality output."""

    summaries: pd.DataFrame
    flags: pd.DataFrame
    quality: pd.DataFrame
    provenance: AnalysisProvenance


def _eligible_numeric(profile: pd.Series) -> bool:
    if not bool(profile["analysis_eligible"]):
        return False
    role = ColumnRole(str(profile["role"]))
    kind = ColumnKind(str(profile["data_kind"]))
    return role is not ColumnRole.FACTOR and kind is ColumnKind.NUMERIC


def _finite_values_with_positions(
    series: pd.Series,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    n_total = len(series)
    missing = series.isna().to_numpy(dtype=bool)
    numeric = series.to_numpy(dtype=np.float64, na_value=np.nan)
    finite = np.isfinite(numeric)
    positions = np.flatnonzero(finite).astype(np.int64)
    values = numeric[finite].astype(np.float64, copy=False)
    n_missing = int(missing.sum())
    n_finite = int(finite.sum())
    n_non_finite = int((~missing & ~finite).sum())
    return values, positions, n_total, n_finite, n_missing, n_non_finite


def _status_for_numeric(
    values: np.ndarray,
    *,
    n_total: int,
    n_finite: int,
    n_missing: int,
    min_numeric_n: int,
) -> tuple[ResultStatus, str | None]:
    n_present = n_total - n_missing
    if n_present == 0:
        return ResultStatus.SKIPPED, "all_missing"
    if n_finite == 0:
        return ResultStatus.SKIPPED, "no_finite_values"
    if n_finite < min_numeric_n:
        return ResultStatus.SKIPPED, "insufficient_finite_observations"
    if np.unique(values).size == 1:
        return ResultStatus.DEGENERATE, "constant"
    return ResultStatus.OK, None


def summarize_numeric_quality(
    dataset: TabularDataset,
    columns: pd.DataFrame | None = None,
    *,
    min_numeric_n: int = 3,
) -> pd.DataFrame:
    """Summarize numeric data-quality states without changing source values."""
    if min_numeric_n < 2:
        raise ValueError("min_numeric_n must be at least 2.")
    columns = profile_columns(dataset) if columns is None else columns
    selected = columns.loc[columns.apply(_eligible_numeric, axis=1)]
    frame = dataset.to_frame()
    rows: list[dict[str, Any]] = []

    for _, profile in selected.iterrows():
        column = str(profile["column"])
        role = str(profile["role"])
        values, _, n_total, n_finite, n_missing, n_non_finite = _finite_values_with_positions(frame[column])
        status, reason = _status_for_numeric(
            values,
            n_total=n_total,
            n_finite=n_finite,
            n_missing=n_missing,
            min_numeric_n=min_numeric_n,
        )
        n_present = n_total - n_missing
        n_unique_finite = int(np.unique(values).size) if n_finite else 0
        iqr: float | None = None
        mad: float | None = None
        zero_iqr_nonconstant = False
        zero_mad_nonconstant = False
        if n_finite:
            q25, q75 = np.quantile(values, [0.25, 0.75])
            iqr = float(q75 - q25)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            nonconstant = n_unique_finite > 1
            zero_iqr_nonconstant = bool(nonconstant and iqr == 0.0)
            zero_mad_nonconstant = bool(nonconstant and mad == 0.0)

        rows.append(
            {
                "column": column,
                "role": role,
                "n_total": n_total,
                "n_present": n_present,
                "n_finite": n_finite,
                "n_missing": n_missing,
                "n_non_finite": n_non_finite,
                "n_unique_finite": n_unique_finite,
                "missing_fraction": float(n_missing / n_total) if n_total else 0.0,
                "non_finite_fraction": float(n_non_finite / n_total) if n_total else 0.0,
                "iqr": iqr,
                "mad": mad,
                "has_missing": bool(n_missing),
                "has_non_finite": bool(n_non_finite),
                "zero_iqr_nonconstant": zero_iqr_nonconstant,
                "zero_mad_nonconstant": zero_mad_nonconstant,
                "status": status.value,
                "reason": reason,
            }
        )

    return pd.DataFrame(rows, columns=NUMERIC_QUALITY_COLUMNS)


def _summary_base(
    *,
    column: str,
    role: str,
    method: OutlierMethod,
    n_total: int,
    n_finite: int,
    n_missing: int,
    n_non_finite: int,
    center: float | None,
    scale: float | None,
    lower_bound: float | None,
    upper_bound: float | None,
    criterion: str,
    threshold: float,
    n_flagged: int | None,
    flagged_fraction: float | None,
    status: ResultStatus,
    reason: str | None,
) -> dict[str, Any]:
    return {
        "column": column,
        "role": role,
        "method": method.value,
        "n_total": n_total,
        "n_finite": n_finite,
        "n_missing": n_missing,
        "n_non_finite": n_non_finite,
        "center": center,
        "scale": scale,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "criterion": criterion,
        "threshold": float(threshold),
        "n_flagged": n_flagged,
        "flagged_fraction": flagged_fraction,
        "status": status.value,
        "reason": reason,
    }


def _flag_rows(
    *,
    summary: dict[str, Any],
    observation_ids: pd.Index,
    positions: np.ndarray,
    values: np.ndarray,
    flagged_indices: np.ndarray,
    scores: np.ndarray,
    directions: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    method = str(summary["method"])
    for local_index, direction in zip(flagged_indices, directions, strict=True):
        position = int(positions[local_index])
        rows.append(
            {
                "column": str(summary["column"]),
                "role": str(summary["role"]),
                "method": method,
                "observation_id": observation_ids[position],
                "source_row_index": position,
                "value": float(values[local_index]),
                "score": float(scores[local_index]),
                "direction": direction,
                "threshold": float(summary["threshold"]),
                "status": ResultStatus.OK.value,
                "reason": None,
            }
        )
    return rows


def _iqr_result(
    *,
    column: str,
    role: str,
    values: np.ndarray,
    positions: np.ndarray,
    observation_ids: pd.Index,
    n_total: int,
    n_finite: int,
    n_missing: int,
    n_non_finite: int,
    min_numeric_n: int,
    multiplier: float,
    include_flags: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    status, reason = _status_for_numeric(
        values,
        n_total=n_total,
        n_finite=n_finite,
        n_missing=n_missing,
        min_numeric_n=min_numeric_n,
    )
    criterion = "outside_tukey_fences"
    if status is not ResultStatus.OK:
        return (
            _summary_base(
                column=column,
                role=role,
                method=OutlierMethod.IQR,
                n_total=n_total,
                n_finite=n_finite,
                n_missing=n_missing,
                n_non_finite=n_non_finite,
                center=None,
                scale=None,
                lower_bound=None,
                upper_bound=None,
                criterion=criterion,
                threshold=multiplier,
                n_flagged=None,
                flagged_fraction=None,
                status=status,
                reason=reason,
            ),
            [],
        )

    q25, q75, lower, upper = tukey_fences(values, multiplier=multiplier)
    median = float(np.median(values))
    iqr = float(q75 - q25)
    mask = (values < lower) | (values > upper)
    flagged_indices = np.flatnonzero(mask)
    n_flagged = int(flagged_indices.size)
    summary = _summary_base(
        column=column,
        role=role,
        method=OutlierMethod.IQR,
        n_total=n_total,
        n_finite=n_finite,
        n_missing=n_missing,
        n_non_finite=n_non_finite,
        center=median,
        scale=iqr,
        lower_bound=lower,
        upper_bound=upper,
        criterion=criterion,
        threshold=multiplier,
        n_flagged=n_flagged,
        flagged_fraction=float(n_flagged / n_finite),
        status=ResultStatus.OK,
        reason=None,
    )
    if not include_flags or n_flagged == 0:
        return summary, []
    scores = np.where(values < lower, values - lower, values - upper)
    directions = ["low" if values[index] < lower else "high" for index in flagged_indices]
    return summary, _flag_rows(
        summary=summary,
        observation_ids=observation_ids,
        positions=positions,
        values=values,
        flagged_indices=flagged_indices,
        scores=scores,
        directions=directions,
    )


def _robust_z_result(
    *,
    column: str,
    role: str,
    values: np.ndarray,
    positions: np.ndarray,
    observation_ids: pd.Index,
    n_total: int,
    n_finite: int,
    n_missing: int,
    n_non_finite: int,
    min_numeric_n: int,
    threshold: float,
    include_flags: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    status, reason = _status_for_numeric(
        values,
        n_total=n_total,
        n_finite=n_finite,
        n_missing=n_missing,
        min_numeric_n=min_numeric_n,
    )
    criterion = "absolute_modified_z_exceeds_threshold"
    if status is not ResultStatus.OK:
        return (
            _summary_base(
                column=column,
                role=role,
                method=OutlierMethod.ROBUST_Z,
                n_total=n_total,
                n_finite=n_finite,
                n_missing=n_missing,
                n_non_finite=n_non_finite,
                center=None,
                scale=None,
                lower_bound=None,
                upper_bound=None,
                criterion=criterion,
                threshold=threshold,
                n_flagged=None,
                flagged_fraction=None,
                status=status,
                reason=reason,
            ),
            [],
        )

    median = float(np.median(values))
    mad = median_absolute_deviation(values)
    if mad <= 0.0:
        return (
            _summary_base(
                column=column,
                role=role,
                method=OutlierMethod.ROBUST_Z,
                n_total=n_total,
                n_finite=n_finite,
                n_missing=n_missing,
                n_non_finite=n_non_finite,
                center=median,
                scale=mad,
                lower_bound=None,
                upper_bound=None,
                criterion=criterion,
                threshold=threshold,
                n_flagged=None,
                flagged_fraction=None,
                status=ResultStatus.DEGENERATE,
                reason="zero_mad",
            ),
            [],
        )

    scores, _, _ = modified_z_scores(values)
    raw_delta = float(threshold * mad / MODIFIED_Z_CONSISTENCY)
    lower = float(median - raw_delta)
    upper = float(median + raw_delta)
    flagged_indices = np.flatnonzero(np.abs(scores) > threshold)
    n_flagged = int(flagged_indices.size)
    summary = _summary_base(
        column=column,
        role=role,
        method=OutlierMethod.ROBUST_Z,
        n_total=n_total,
        n_finite=n_finite,
        n_missing=n_missing,
        n_non_finite=n_non_finite,
        center=median,
        scale=mad,
        lower_bound=lower,
        upper_bound=upper,
        criterion=criterion,
        threshold=threshold,
        n_flagged=n_flagged,
        flagged_fraction=float(n_flagged / n_finite),
        status=ResultStatus.OK,
        reason=None,
    )
    if not include_flags or n_flagged == 0:
        return summary, []
    directions = ["low" if scores[index] < 0 else "high" for index in flagged_indices]
    return summary, _flag_rows(
        summary=summary,
        observation_ids=observation_ids,
        positions=positions,
        values=values,
        flagged_indices=flagged_indices,
        scores=scores,
        directions=directions,
    )


def summarize_outliers(
    dataset: TabularDataset,
    columns: pd.DataFrame | None = None,
    *,
    methods: Iterable[OutlierMethod | str] = (OutlierMethod.IQR, OutlierMethod.ROBUST_Z),
    min_numeric_n: int = 3,
    iqr_multiplier: float = 1.5,
    robust_z_threshold: float = 3.5,
    include_flags: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute deterministic univariate outlier summaries and optional row flags."""
    if min_numeric_n < 2:
        raise ValueError("min_numeric_n must be at least 2.")
    if iqr_multiplier <= 0:
        raise ValueError("iqr_multiplier must be greater than zero.")
    if robust_z_threshold <= 0:
        raise ValueError("robust_z_threshold must be greater than zero.")
    resolved_methods = tuple(
        method if isinstance(method, OutlierMethod) else OutlierMethod(method) for method in methods
    )
    if not resolved_methods:
        raise ValueError("At least one outlier method must be configured.")
    if len(set(resolved_methods)) != len(resolved_methods):
        raise ValueError("methods cannot contain duplicates.")

    columns = profile_columns(dataset) if columns is None else columns
    selected = columns.loc[columns.apply(_eligible_numeric, axis=1)]
    frame = dataset.to_frame()
    observation_ids = dataset.observation_ids
    summaries: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []

    for _, profile in selected.iterrows():
        column = str(profile["column"])
        role = str(profile["role"])
        values, positions, n_total, n_finite, n_missing, n_non_finite = _finite_values_with_positions(
            frame[column]
        )
        for method in resolved_methods:
            if method is OutlierMethod.IQR:
                summary, method_flags = _iqr_result(
                    column=column,
                    role=role,
                    values=values,
                    positions=positions,
                    observation_ids=observation_ids,
                    n_total=n_total,
                    n_finite=n_finite,
                    n_missing=n_missing,
                    n_non_finite=n_non_finite,
                    min_numeric_n=min_numeric_n,
                    multiplier=iqr_multiplier,
                    include_flags=include_flags,
                )
            else:
                summary, method_flags = _robust_z_result(
                    column=column,
                    role=role,
                    values=values,
                    positions=positions,
                    observation_ids=observation_ids,
                    n_total=n_total,
                    n_finite=n_finite,
                    n_missing=n_missing,
                    n_non_finite=n_non_finite,
                    min_numeric_n=min_numeric_n,
                    threshold=robust_z_threshold,
                    include_flags=include_flags,
                )
            summaries.append(summary)
            flags.extend(method_flags)

    return (
        pd.DataFrame(summaries, columns=OUTLIER_SUMMARY_COLUMNS),
        pd.DataFrame(flags, columns=OUTLIER_FLAG_COLUMNS),
    )


def analyze_outliers(
    dataset: TabularDataset,
    *,
    methods: Iterable[OutlierMethod | str] = (OutlierMethod.IQR, OutlierMethod.ROBUST_Z),
    min_numeric_n: int = 3,
    iqr_multiplier: float = 1.5,
    robust_z_threshold: float = 3.5,
    include_flags: bool = False,
) -> OutlierResult:
    """Run non-destructive univariate outlier and numeric quality diagnostics."""
    resolved_methods = tuple(
        method if isinstance(method, OutlierMethod) else OutlierMethod(method) for method in methods
    )
    columns = profile_columns(dataset)
    quality = summarize_numeric_quality(
        dataset,
        columns,
        min_numeric_n=min_numeric_n,
    )
    summaries, flags = summarize_outliers(
        dataset,
        columns,
        methods=resolved_methods,
        min_numeric_n=min_numeric_n,
        iqr_multiplier=iqr_multiplier,
        robust_z_threshold=robust_z_threshold,
        include_flags=include_flags,
    )
    provenance = AnalysisProvenance(
        analysis="outliers",
        parameters={
            "methods": tuple(method.value for method in resolved_methods),
            "min_numeric_n": min_numeric_n,
            "iqr_multiplier": float(iqr_multiplier),
            "robust_z_threshold": float(robust_z_threshold),
            "include_flags": bool(include_flags),
            "finite_policy": "finite_values_only",
            "missing_and_non_finite_policy": "reported_not_flagged",
            "record_removal": False,
            "value_modification": False,
            "automatic_cleaning": False,
            "multivariate_outliers": False,
        },
        input_summary={
            "n_observations": dataset.n_observations,
            "n_columns": dataset.n_columns,
            "id_column": dataset.id_column,
        },
    )
    return OutlierResult(
        summaries=summaries,
        flags=flags,
        quality=quality,
        provenance=provenance,
    )
