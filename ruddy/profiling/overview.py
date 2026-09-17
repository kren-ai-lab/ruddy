"""Dataset-level descriptive profiling orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from ruddy.core.enums import ColumnKind, ColumnRole
from ruddy.profiling.columns import profile_columns
from ruddy.profiling.missingness import (
    pairwise_completeness,
    summarize_missingness,
    summarize_missingness_patterns,
)
from ruddy.results import AnalysisProvenance

if TYPE_CHECKING:
    import pandas as pd

    from ruddy.data import TabularDataset


@dataclass(frozen=True, slots=True)
class ProfilingResult:
    """Complete descriptive profiling output."""

    overview: dict[str, Any]
    columns: pd.DataFrame
    missingness: pd.DataFrame
    pairwise_completeness: pd.DataFrame
    missingness_patterns: pd.DataFrame
    provenance: AnalysisProvenance


def _identifier_summary(dataset: TabularDataset) -> dict[str, Any]:
    ids = dataset.observation_ids
    return {
        "column": dataset.id_column,
        "source": "column" if dataset.id_column is not None else "index",
        "n_present": len(ids),
        "n_missing": 0,
        "n_unique": int(ids.nunique()),
        "n_duplicate_records": int(ids.duplicated(keep=False).sum()),
        "n_duplicate_values": int(ids[ids.duplicated(keep=False)].nunique()),
    }


def summarize_overview(dataset: TabularDataset, columns: pd.DataFrame) -> dict[str, Any]:
    """Create a structured dataset overview from source data and column profiles."""
    frame = dataset.to_frame()
    n_records = int(dataset.n_observations)
    n_columns = int(dataset.n_columns)
    total_cells = n_records * n_columns
    n_missing_cells = int(frame.isna().sum().sum())
    rows_with_missing = cast("pd.Series", frame.isna().any(axis=1))
    duplicated_all = frame.duplicated(keep=False)
    duplicated_extra = frame.duplicated(keep="first")

    role_counts = {role.value: int(columns["role"].eq(role.value).sum()) for role in ColumnRole}
    kind_counts = {kind.value: int(columns["data_kind"].eq(kind.value).sum()) for kind in ColumnKind}

    numeric_non_finite = (
        columns.loc[columns["data_kind"].eq(ColumnKind.NUMERIC.value), "non_finite_count"]
        .fillna(0)
        .astype(int)
    )

    return {
        "n_records": n_records,
        "n_columns": n_columns,
        "column_roles": role_counts,
        "column_kinds": kind_counts,
        "column_quality": {
            "n_analysis_eligible": int(columns["analysis_eligible"].sum()),
            "n_excluded": int(columns["excluded"].sum()),
            "n_with_missing": int(columns["n_missing"].gt(0).sum()),
            "n_all_missing": int(columns["is_all_missing"].sum()),
            "n_constant": int(columns["is_constant"].sum()),
            "n_unknown_kind": int(columns["data_kind"].eq(ColumnKind.UNKNOWN.value).sum()),
            "n_with_non_finite_numeric": int(numeric_non_finite.gt(0).sum()),
            "n_non_finite_numeric_observations": int(numeric_non_finite.sum()),
        },
        "missingness": {
            "n_missing_cells": n_missing_cells,
            "missing_fraction": float(n_missing_cells / total_cells) if total_cells else 0.0,
            "n_records_with_missing": int(rows_with_missing.sum()),
            "records_with_missing_fraction": float(rows_with_missing.mean()) if n_records else 0.0,
            "n_complete_records": int((~rows_with_missing).sum()),
            "complete_record_fraction": float((~rows_with_missing).mean()) if n_records else 0.0,
        },
        "duplicates": {
            "n_duplicate_rows": int(duplicated_extra.sum()),
            "n_records_in_duplicate_groups": int(duplicated_all.sum()),
        },
        "identifier": _identifier_summary(dataset),
    }


def profile_dataset(
    dataset: TabularDataset,
    *,
    max_missingness_patterns: int = 20,
    max_pairwise_columns: int = 200,
) -> ProfilingResult:
    """Run the complete descriptive profiling block."""
    columns = profile_columns(dataset)
    missingness = summarize_missingness(columns)
    pairwise = pairwise_completeness(dataset, max_columns=max_pairwise_columns)
    patterns = summarize_missingness_patterns(
        dataset,
        max_patterns=max_missingness_patterns,
    )
    overview = summarize_overview(dataset, columns)
    provenance = AnalysisProvenance(
        analysis="profiling",
        parameters={
            "max_missingness_patterns": max_missingness_patterns,
            "max_pairwise_columns": max_pairwise_columns,
            "missingness_policy": "pandas_missing_values_only",
            "numeric_non_finite_policy": "count_separately_from_missing",
        },
        input_summary={
            "n_observations": dataset.n_observations,
            "n_columns": dataset.n_columns,
            "id_column": dataset.id_column,
        },
    )
    return ProfilingResult(
        overview=overview,
        columns=columns,
        missingness=missingness,
        pairwise_completeness=pairwise,
        missingness_patterns=patterns,
        provenance=provenance,
    )
