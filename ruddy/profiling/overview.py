"""Dataset-level descriptive profiling orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from ruddy.core.enums import ColumnKind, ColumnRole
from ruddy.data import TabularDataset
from ruddy.profiling.columns import profile_columns
from ruddy.profiling.missingness import (
    pairwise_completeness,
    summarize_missingness,
    summarize_missingness_patterns,
)
from ruddy.results import AnalysisProvenance


@dataclass(frozen=True, slots=True)
class ProfilingResult:
    """Complete descriptive profiling output."""

    overview: dict[str, Any]
    columns: pl.DataFrame
    missingness: pl.DataFrame
    pairwise_completeness: pl.DataFrame
    missingness_patterns: pl.DataFrame
    provenance: AnalysisProvenance


def _identifier_summary(dataset: TabularDataset) -> dict[str, Any]:
    ids = dataset.observation_id_tuple
    source = "column" if dataset.id_column is not None else dataset.provenance["id_source"]
    n_present = len(ids)
    return {
        "column": dataset.id_column,
        "source": source,
        "n_present": n_present,
        "n_missing": 0,
        "n_unique": n_present,
        "n_duplicate_records": 0,
        "n_duplicate_values": 0,
    }


def summarize_overview(dataset: TabularDataset, columns: pl.DataFrame) -> dict[str, Any]:
    """Create a structured dataset overview from source data and column profiles."""
    frame = dataset.frame
    n_records = frame.height
    n_columns = frame.width
    total_cells = n_records * n_columns

    if n_columns == 0:
        n_missing_cells = 0
        n_records_with_missing = 0
        records_with_missing_fraction = 0.0
        n_complete_records = n_records
        complete_record_fraction = 1.0 if n_records else 0.0
        n_duplicate_rows = 0
        n_records_in_duplicate_groups = 0
    else:
        missing_exprs = [
            (pl.col(col).is_null() | pl.col(col).is_nan()).alias(col)
            if frame.schema[col].is_float()
            else pl.col(col).is_null().alias(col)
            for col in frame.columns
        ]
        mask = frame.select(missing_exprs)
        rows_with_missing = mask.select(pl.any_horizontal(pl.all()).alias("any")).get_column("any")
        n_missing_cells = int(columns.get_column("n_missing").sum() or 0)
        n_records_with_missing = int(rows_with_missing.sum() or 0)
        records_with_missing_fraction = float(n_records_with_missing / n_records) if n_records else 0.0
        n_complete_records = int(n_records - n_records_with_missing)
        complete_record_fraction = float(n_complete_records / n_records) if n_records else 0.0

        if n_records == 0:
            n_duplicate_rows = 0
            n_records_in_duplicate_groups = 0
        else:
            n_duplicate_rows = frame.height - frame.n_unique()
            n_records_in_duplicate_groups = int(frame.is_duplicated().sum())

    if columns.height == 0:
        role_counts = {role.value: 0 for role in ColumnRole}
        kind_counts = {kind.value: 0 for kind in ColumnKind}
        n_analysis_eligible = 0
        n_excluded = 0
        n_with_missing = 0
        n_all_missing = 0
        n_constant = 0
        n_unknown_kind = 0
        n_with_non_finite_numeric = 0
        n_non_finite_numeric_observations = 0
    else:
        role_counts = {
            role.value: int((columns.get_column("role") == role.value).sum() or 0) for role in ColumnRole
        }
        kind_counts = {
            kind.value: int((columns.get_column("data_kind") == kind.value).sum() or 0) for kind in ColumnKind
        }
        n_analysis_eligible = int(columns.get_column("analysis_eligible").sum() or 0)
        n_excluded = int(columns.get_column("excluded").sum() or 0)
        n_with_missing = int((columns.get_column("n_missing") > 0).sum() or 0)
        n_all_missing = int(columns.get_column("is_all_missing").sum() or 0)
        n_constant = int(columns.get_column("is_constant").sum() or 0)
        n_unknown_kind = int((columns.get_column("data_kind") == ColumnKind.UNKNOWN.value).sum() or 0)
        numeric_cols = columns.filter(pl.col("data_kind") == ColumnKind.NUMERIC.value)
        if numeric_cols.height > 0:
            numeric_non_finite = numeric_cols.get_column("non_finite_count").fill_null(0)
            n_with_non_finite_numeric = int((numeric_non_finite > 0).sum() or 0)
            n_non_finite_numeric_observations = int(numeric_non_finite.sum() or 0)
        else:
            n_with_non_finite_numeric = 0
            n_non_finite_numeric_observations = 0

    return {
        "n_records": n_records,
        "n_columns": n_columns,
        "column_roles": role_counts,
        "column_kinds": kind_counts,
        "column_quality": {
            "n_analysis_eligible": n_analysis_eligible,
            "n_excluded": n_excluded,
            "n_with_missing": n_with_missing,
            "n_all_missing": n_all_missing,
            "n_constant": n_constant,
            "n_unknown_kind": n_unknown_kind,
            "n_with_non_finite_numeric": n_with_non_finite_numeric,
            "n_non_finite_numeric_observations": n_non_finite_numeric_observations,
        },
        "missingness": {
            "n_missing_cells": n_missing_cells,
            "missing_fraction": float(n_missing_cells / total_cells) if total_cells else 0.0,
            "n_records_with_missing": n_records_with_missing,
            "records_with_missing_fraction": records_with_missing_fraction,
            "n_complete_records": n_complete_records,
            "complete_record_fraction": complete_record_fraction,
        },
        "duplicates": {
            "n_duplicate_rows": n_duplicate_rows,
            "n_records_in_duplicate_groups": n_records_in_duplicate_groups,
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
            "missingness_policy": "null_or_nan_missing_values",
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
