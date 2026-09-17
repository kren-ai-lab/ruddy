"""Deterministic categorical and boolean univariate statistics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ruddy.core.enums import ColumnKind, ColumnRole
from ruddy.data import TabularDataset

CATEGORICAL_STATISTICS_COLUMNS: tuple[str, ...] = (
    "column",
    "role",
    "n_total",
    "n_present",
    "n_missing",
    "n_levels",
    "mode",
    "mode_count",
    "mode_fraction",
    "entropy",
    "normalized_entropy",
    "entropy_base",
    "n_levels_reported",
    "frequencies_truncated",
    "unreported_count",
    "unreported_fraction",
    "status",
    "reason",
)

CATEGORICAL_FREQUENCY_COLUMNS: tuple[str, ...] = (
    "column",
    "role",
    "level",
    "count",
    "fraction",
    "rank",
)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _category_label(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        converted = float(value)
        if math.isfinite(converted):
            return repr(converted)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _eligible_categorical(profile: pd.Series) -> bool:
    if not bool(profile["analysis_eligible"]):
        return False
    role = ColumnRole(str(profile["role"]))
    kind = ColumnKind(str(profile["data_kind"]))
    if role is ColumnRole.FACTOR:
        return True
    return kind in {ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN}


def _profile(
    *,
    column: str,
    role: str,
    series: pd.Series,
    max_category_levels: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n_total = len(series)
    n_missing = int(series.isna().sum())
    n_present = n_total - n_missing

    labels = series.dropna().map(_category_label)
    counts = labels.value_counts(dropna=False, sort=False)
    ordered = sorted(
        ((str(level), int(count)) for level, count in counts.items()),
        key=lambda item: (-item[1], item[0]),
    )
    n_levels = len(ordered)
    if n_present == 0:
        status, reason = "skipped", "all_missing"
    elif n_levels == 1:
        status, reason = "degenerate", "constant"
    else:
        status, reason = "ok", None

    mode: str | None = None
    mode_count = 0
    mode_fraction: float | None = None
    entropy: float | None = None
    normalized_entropy: float | None = None

    if ordered:
        mode, mode_count = ordered[0]
        mode_fraction = float(mode_count / n_present)
        probabilities = np.asarray([count / n_present for _, count in ordered], dtype=np.float64)
        entropy = _safe_float(-np.sum(probabilities * np.log2(probabilities)))
        if n_levels <= 1:
            normalized_entropy = 0.0
        elif entropy is not None:
            normalized_entropy = _safe_float(entropy / math.log2(n_levels))

    n_reported = min(n_levels, max_category_levels)
    reported = ordered[:n_reported]
    reported_count = int(sum(count for _, count in reported))
    unreported_count = int(n_present - reported_count)

    summary = {
        "column": column,
        "role": role,
        "n_total": n_total,
        "n_present": n_present,
        "n_missing": n_missing,
        "n_levels": n_levels,
        "mode": mode,
        "mode_count": mode_count,
        "mode_fraction": mode_fraction,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "entropy_base": 2,
        "n_levels_reported": n_reported,
        "frequencies_truncated": n_levels > max_category_levels,
        "unreported_count": unreported_count,
        "unreported_fraction": float(unreported_count / n_present) if n_present else None,
        "status": status,
        "reason": reason,
    }
    frequencies = [
        {
            "column": column,
            "role": role,
            "level": level,
            "count": count,
            "fraction": float(count / n_present),
            "rank": rank,
        }
        for rank, (level, count) in enumerate(reported, start=1)
    ]
    return summary, frequencies


def summarize_categorical_statistics(
    dataset: TabularDataset,
    columns: pd.DataFrame | pl.DataFrame,
    *,
    max_category_levels: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize eligible categorical, boolean, and factor variables."""
    if max_category_levels < 2:
        raise ValueError("max_category_levels must be at least 2.")

    # ponytail: temporary pandas adapter, removed in task 3B
    if isinstance(columns, pl.DataFrame):
        columns = columns.to_pandas()
    selected = columns.loc[columns.apply(_eligible_categorical, axis=1)]
    frame = dataset.to_frame()
    summaries: list[dict[str, Any]] = []
    frequencies: list[dict[str, Any]] = []
    for _, profile in selected.iterrows():
        column = str(profile["column"])
        summary, rows = _profile(
            column=column,
            role=str(profile["role"]),
            series=frame[column],
            max_category_levels=max_category_levels,
        )
        summaries.append(summary)
        frequencies.extend(rows)

    return (
        pd.DataFrame(summaries, columns=pd.Index(CATEGORICAL_STATISTICS_COLUMNS)),
        pd.DataFrame(frequencies, columns=pd.Index(CATEGORICAL_FREQUENCY_COLUMNS)),
    )
