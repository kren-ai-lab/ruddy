"""Missing-data summaries and completeness diagnostics."""

from __future__ import annotations

from itertools import combinations_with_replacement
import json

import pandas as pd

from ruddy.data import TabularDataset

MISSINGNESS_COLUMNS: tuple[str, ...] = (
    "column",
    "role",
    "data_kind",
    "n_total",
    "n_present",
    "fraction_present",
    "n_missing",
    "fraction_missing",
    "finite_count",
    "non_finite_count",
    "is_all_missing",
)

PAIRWISE_COMPLETENESS_COLUMNS: tuple[str, ...] = (
    "column_x",
    "column_y",
    "n_total",
    "n_complete",
    "n_incomplete",
    "fraction_complete",
)

MISSINGNESS_PATTERN_COLUMNS: tuple[str, ...] = (
    "rank",
    "missing_columns",
    "n_missing_columns",
    "count",
    "fraction",
    "n_patterns_total",
    "patterns_truncated",
    "unreported_count",
    "unreported_fraction",
)


def summarize_missingness(columns: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic per-column missingness output from column profiles."""

    required = {
        "column",
        "role",
        "data_kind",
        "n_total",
        "n_present",
        "n_missing",
        "missing_fraction",
        "finite_count",
        "non_finite_count",
        "is_all_missing",
    }
    missing = sorted(required - set(columns.columns))
    if missing:
        raise ValueError(f"Column profile is missing required fields: {missing}.")

    rows = pd.DataFrame(
        {
            "column": columns["column"].astype(str),
            "role": columns["role"].astype(str),
            "data_kind": columns["data_kind"].astype(str),
            "n_total": columns["n_total"].astype(int),
            "n_present": columns["n_present"].astype(int),
            "fraction_present": 1.0 - columns["missing_fraction"].astype(float),
            "n_missing": columns["n_missing"].astype(int),
            "fraction_missing": columns["missing_fraction"].astype(float),
            "finite_count": columns["finite_count"],
            "non_finite_count": columns["non_finite_count"],
            "is_all_missing": columns["is_all_missing"].astype(bool),
        }
    )
    return rows.loc[:, list(MISSINGNESS_COLUMNS)]


def pairwise_completeness(
    dataset: TabularDataset,
    *,
    columns: tuple[str, ...] | None = None,
    max_columns: int = 200,
) -> pd.DataFrame:
    """Return upper-triangular pairwise non-missing completeness statistics.

    Missingness here means pandas missing values only. Numeric infinities remain
    present observations and are handled separately by numerical analyses.
    """

    selected = columns or dataset.columns
    unknown = sorted(set(selected) - set(dataset.columns))
    if unknown:
        raise ValueError(f"Unknown columns requested for completeness: {unknown}.")
    if max_columns < 1:
        raise ValueError("max_columns must be at least 1.")
    if len(selected) > max_columns:
        raise ValueError(
            f"Pairwise completeness requested for {len(selected)} columns, exceeding "
            f"max_columns={max_columns}."
        )

    frame = dataset.select(selected)
    present = frame.notna()
    n_total = len(frame)
    rows: list[dict[str, object]] = []
    for column_x, column_y in combinations_with_replacement(selected, 2):
        n_complete = int((present[column_x] & present[column_y]).sum())
        rows.append(
            {
                "column_x": column_x,
                "column_y": column_y,
                "n_total": n_total,
                "n_complete": n_complete,
                "n_incomplete": int(n_total - n_complete),
                "fraction_complete": float(n_complete / n_total) if n_total else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=PAIRWISE_COMPLETENESS_COLUMNS)


def summarize_missingness_patterns(
    dataset: TabularDataset,
    *,
    max_patterns: int = 20,
) -> pd.DataFrame:
    """Summarize common row-level missingness patterns with bounded output."""

    if max_patterns < 1:
        raise ValueError("max_patterns must be at least 1.")

    frame = dataset.to_frame()
    n_total = len(frame)
    counts: dict[tuple[str, ...], int] = {}
    for row in frame.isna().itertuples(index=False, name=None):
        pattern = tuple(column for column, is_missing in zip(frame.columns, row, strict=True) if is_missing)
        counts[pattern] = counts.get(pattern, 0) + 1

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    n_patterns_total = len(ordered)
    reported = ordered[:max_patterns]
    reported_count = sum(count for _, count in reported)
    unreported_count = int(n_total - reported_count)
    truncated = n_patterns_total > max_patterns

    rows = [
        {
            "rank": rank,
            "missing_columns": json.dumps(pattern, separators=(",", ":")),
            "n_missing_columns": len(pattern),
            "count": count,
            "fraction": float(count / n_total) if n_total else 0.0,
            "n_patterns_total": n_patterns_total,
            "patterns_truncated": truncated,
            "unreported_count": unreported_count,
            "unreported_fraction": float(unreported_count / n_total) if n_total else 0.0,
        }
        for rank, (pattern, count) in enumerate(reported, start=1)
    ]
    return pd.DataFrame(rows, columns=MISSINGNESS_PATTERN_COLUMNS)
