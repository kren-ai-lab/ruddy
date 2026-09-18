"""Response-centric grouped summaries and inference."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ruddy.bivariate.associations import (
    ASSOCIATION_SCHEMA,
    summarize_categorical_associations,
)
from ruddy.bivariate.comparisons import (
    COMPARISON_SCHEMA,
    _safe_float,
    summarize_numeric_categorical_comparisons,
)
from ruddy.core.enums import ColumnKind, ColumnRole, ComparisonTest, PAdjustMethod
from ruddy.data import AlignedAnnotations, TabularDataset, attach_annotations
from ruddy.results import AnalysisProvenance
from ruddy.statistics import apply_multiple_testing
from ruddy.univariate.categorical import _category_label

if TYPE_CHECKING:
    from collections.abc import Iterable

    from polars._typing import PolarsDataType

GROUP_COVERAGE_SCHEMA: dict[str, PolarsDataType] = {
    "group_column": pl.String,
    "group_role": pl.String,
    "data_kind": pl.String,
    "n_dataset": pl.Int64,
    "n_group_present": pl.Int64,
    "n_group_missing": pl.Int64,
    "coverage_fraction": pl.Float64,
    "n_levels": pl.Int64,
    "levels_json": pl.String,
    "status": pl.String,
    "reason": pl.String,
}
GROUP_COVERAGE_COLUMNS: tuple[str, ...] = tuple(GROUP_COVERAGE_SCHEMA)

RESPONSE_CATALOG_SCHEMA: dict[str, PolarsDataType] = {
    "response": pl.String,
    "declared_role": pl.String,
    "data_kind": pl.String,
    "n_dataset": pl.Int64,
    "n_present": pl.Int64,
    "n_missing": pl.Int64,
    "n_finite": pl.Int64,
    "n_non_finite": pl.Int64,
    "status": pl.String,
    "reason": pl.String,
}
RESPONSE_CATALOG_COLUMNS: tuple[str, ...] = tuple(RESPONSE_CATALOG_SCHEMA)

NUMERIC_GROUP_SUMMARY_SCHEMA: dict[str, PolarsDataType] = {
    "group_column": pl.String,
    "group_level": pl.String,
    "response": pl.String,
    "response_role": pl.String,
    "n_dataset": pl.Int64,
    "n_group_observations": pl.Int64,
    "n_response_present": pl.Int64,
    "n_response_finite": pl.Int64,
    "n_response_missing": pl.Int64,
    "n_response_non_finite": pl.Int64,
    "mean": pl.Float64,
    "std": pl.Float64,
    "median": pl.Float64,
    "q25": pl.Float64,
    "q75": pl.Float64,
    "iqr": pl.Float64,
    "min": pl.Float64,
    "max": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}
NUMERIC_GROUP_SUMMARY_COLUMNS: tuple[str, ...] = tuple(NUMERIC_GROUP_SUMMARY_SCHEMA)

CATEGORICAL_GROUP_SUMMARY_SCHEMA: dict[str, PolarsDataType] = {
    "group_column": pl.String,
    "group_level": pl.String,
    "response": pl.String,
    "response_role": pl.String,
    "response_level": pl.String,
    "n_dataset": pl.Int64,
    "n_group_observations": pl.Int64,
    "n_response_present": pl.Int64,
    "n_response_missing": pl.Int64,
    "count": pl.Int64,
    "fraction": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}
CATEGORICAL_GROUP_SUMMARY_COLUMNS: tuple[str, ...] = tuple(CATEGORICAL_GROUP_SUMMARY_SCHEMA)

ANNOTATION_COVERAGE_SCHEMA: dict[str, PolarsDataType] = {
    "source_name": pl.String,
    "coverage": pl.String,
    "base_count": pl.Int64,
    "annotation_count": pl.Int64,
    "covered_count": pl.Int64,
    "coverage_fraction": pl.Float64,
    "n_missing_ids": pl.Int64,
    "n_unmatched_ids": pl.Int64,
    "columns_json": pl.String,
}
ANNOTATION_COVERAGE_COLUMNS: tuple[str, ...] = tuple(ANNOTATION_COVERAGE_SCHEMA)

_GROUPED_NUMERIC_COMPARISON_SCHEMA: dict[str, PolarsDataType] = {
    "response": pl.String,
    "n_dataset": pl.Int64,
    "n_group_present": pl.Int64,
    "n_group_missing": pl.Int64,
    **COMPARISON_SCHEMA,
}
_GROUPED_NUMERIC_COMPARISON_COLUMNS: tuple[str, ...] = tuple(_GROUPED_NUMERIC_COMPARISON_SCHEMA)

_GROUPED_CATEGORICAL_COMPARISON_SCHEMA: dict[str, PolarsDataType] = {
    "response": pl.String,
    "group_column": pl.String,
    "n_dataset": pl.Int64,
    "n_group_present": pl.Int64,
    "n_group_missing": pl.Int64,
    **ASSOCIATION_SCHEMA,
}
_GROUPED_CATEGORICAL_COMPARISON_COLUMNS: tuple[str, ...] = tuple(_GROUPED_CATEGORICAL_COMPARISON_SCHEMA)

_NUMERIC_TESTS = (
    ComparisonTest.WELCH_T,
    ComparisonTest.MANN_WHITNEY,
    ComparisonTest.WELCH_ANOVA,
    ComparisonTest.KRUSKAL_WALLIS,
)
_CATEGORICAL_TESTS = (ComparisonTest.CHI_SQUARE, ComparisonTest.FISHER_EXACT)


@dataclass(frozen=True, slots=True)
class GroupAnalysisResult:
    """Complete response-centric grouped analysis output."""

    response_catalog: pl.DataFrame
    group_coverage: pl.DataFrame
    numeric_summaries: pl.DataFrame
    categorical_summaries: pl.DataFrame
    numeric_comparisons: pl.DataFrame
    categorical_comparisons: pl.DataFrame
    annotation_coverage: pl.DataFrame
    provenance: AnalysisProvenance


def _unique_columns(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    resolved = tuple(str(value) for value in values)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{label} cannot contain duplicates.")
    return resolved


def resolve_responses(
    dataset: TabularDataset,
    responses: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Resolve statistically analysable response columns."""
    if responses is None:
        selected = dataset.columns_with_role(ColumnRole.RESPONSE)
    else:
        selected = _unique_columns(responses, label="responses")
    unknown = [column for column in selected if column not in dataset.frame.columns]
    if unknown:
        raise ValueError(f"Unknown response columns: {unknown}.")
    invalid_role = [
        column
        for column in selected
        if dataset.role_of(column) in {ColumnRole.IDENTIFIER, ColumnRole.EXCLUDED}
    ]
    if invalid_role:
        raise ValueError(f"Response columns cannot be identifier/excluded columns: {invalid_role}.")
    invalid_kind = [
        column
        for column in selected
        if dataset.kind_of(column) not in {ColumnKind.NUMERIC, ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN}
    ]
    if invalid_kind:
        raise ValueError(
            f"Responses must be numeric, categorical, or boolean; invalid columns: {invalid_kind}."
        )
    return selected


def resolve_groups(
    dataset: TabularDataset,
    groups: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Resolve grouping variables without requiring a dedicated storage type."""
    if groups is None:
        selected = dataset.columns_with_role(ColumnRole.FACTOR)
    else:
        selected = _unique_columns(groups, label="groups")
    unknown = [column for column in selected if column not in dataset.frame.columns]
    if unknown:
        raise ValueError(f"Unknown group columns: {unknown}.")
    invalid: list[str] = []
    for column in selected:
        role = dataset.role_of(column)
        kind = dataset.kind_of(column)
        if role in {ColumnRole.IDENTIFIER, ColumnRole.EXCLUDED}:
            invalid.append(column)
            continue
        if role is ColumnRole.FACTOR:
            continue
        if kind not in {ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN}:
            invalid.append(column)
    if invalid:
        raise ValueError(
            "Group columns must be categorical/boolean or explicitly declared factors; "
            f"invalid columns: {invalid}."
        )
    return selected


def _n_missing(series: pl.Series) -> int:
    count = series.null_count()
    if series.dtype.is_float():
        nan_sum = series.is_nan().sum()
        if nan_sum is not None:
            count += int(nan_sum)
    return count


def _normalized_group(series: pl.Series) -> list[str | None]:
    """Label each row with ``_category_label``; missing (null or float NaN) rows become ``None``."""
    return [
        None if value is None or (isinstance(value, float) and math.isnan(value)) else _category_label(value)
        for value in series.to_list()
    ]


def summarize_group_coverage(
    dataset: TabularDataset,
    groups: Iterable[str] | None = None,
    *,
    max_group_levels: int = 20,
) -> pl.DataFrame:
    """Describe observation coverage for each grouping variable."""
    if max_group_levels < 2:
        raise ValueError("max_group_levels must be at least 2.")
    selected = resolve_groups(dataset, groups)
    frame = dataset.frame
    n_dataset = frame.height
    rows: list[dict[str, Any]] = []
    for group in selected:
        labels = _normalized_group(frame.get_column(group))
        non_none_labels = [label for label in labels if label is not None]
        levels = sorted(set(non_none_labels))
        n_present = len(non_none_labels)
        status = "ok"
        reason = None
        if len(levels) < 2:
            status = "degenerate"
            reason = "insufficient_group_levels"
        elif len(levels) > max_group_levels:
            status = "skipped"
            reason = "group_levels_exceed_max_group_levels"
        rows.append(
            {
                "group_column": group,
                "group_role": dataset.role_of(group).value,
                "data_kind": dataset.kind_of(group).value,
                "n_dataset": n_dataset,
                "n_group_present": n_present,
                "n_group_missing": n_dataset - n_present,
                "coverage_fraction": (float(n_present / n_dataset) if n_dataset else 1.0),
                "n_levels": len(levels),
                "levels_json": json.dumps(levels, separators=(",", ":")),
                "status": status,
                "reason": reason,
            }
        )
    return pl.DataFrame(rows, schema=GROUP_COVERAGE_SCHEMA)


def summarize_response_catalog(
    dataset: TabularDataset,
    responses: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Describe selected response variables and their usable observations."""
    selected = resolve_responses(dataset, responses)
    frame = dataset.frame
    n_dataset = frame.height
    rows: list[dict[str, Any]] = []
    for response in selected:
        series = frame.get_column(response)
        n_missing = _n_missing(series)
        n_present = n_dataset - n_missing
        kind = dataset.kind_of(response)
        n_finite: int | None = None
        n_non_finite: int | None = None
        status = "ok"
        reason = None
        if kind is ColumnKind.NUMERIC:
            non_missing = series.drop_nulls()
            if non_missing.dtype.is_float():
                non_missing = non_missing.filter(~non_missing.is_nan())
            values = non_missing.cast(pl.Float64).to_numpy()
            finite = np.isfinite(values)
            n_finite = int(finite.sum())
            n_non_finite = int((~finite).sum())
            if n_finite == 0:
                status = "skipped"
                reason = "no_finite_response_values"
        elif n_present == 0:
            status = "skipped"
            reason = "all_missing_response"
        rows.append(
            {
                "response": response,
                "declared_role": dataset.role_of(response).value,
                "data_kind": kind.value,
                "n_dataset": n_dataset,
                "n_present": n_present,
                "n_missing": n_missing,
                "n_finite": n_finite,
                "n_non_finite": n_non_finite,
                "status": status,
                "reason": reason,
            }
        )
    return pl.DataFrame(rows, schema=RESPONSE_CATALOG_SCHEMA)


def summarize_grouped_numeric_responses(
    dataset: TabularDataset,
    responses: Iterable[str] | None = None,
    groups: Iterable[str] | None = None,
    *,
    max_group_levels: int = 20,
) -> pl.DataFrame:
    """Compute descriptive summaries for numeric responses within groups."""
    selected_responses = tuple(
        response
        for response in resolve_responses(dataset, responses)
        if dataset.kind_of(response) is ColumnKind.NUMERIC
    )
    selected_groups = resolve_groups(dataset, groups)
    frame = dataset.frame
    n_dataset = frame.height
    coverage = (
        summarize_group_coverage(dataset, selected_groups, max_group_levels=max_group_levels)
        if selected_groups
        else None
    )
    coverage_status = (
        {row["group_column"]: row["status"] for row in coverage.iter_rows(named=True)}
        if coverage is not None
        else {}
    )
    rows: list[dict[str, Any]] = []
    for group in selected_groups:
        if coverage_status.get(group) == "skipped":
            continue
        group_labels = _normalized_group(frame.get_column(group))
        levels = sorted({label for label in group_labels if label is not None})
        for level in levels:
            mask = np.array([label == level for label in group_labels], dtype=bool)
            n_group = int(mask.sum())
            for response in selected_responses:
                resp_series = frame.get_column(response)
                raw_values = resp_series.cast(pl.Float64).fill_null(float("nan")).to_numpy()
                values = raw_values[mask]
                missing_mask = np.isnan(values)
                n_missing = int(missing_mask.sum())
                present = values[~missing_mask]
                finite = present[np.isfinite(present)]
                n_non_finite = int(len(present) - len(finite))
                row: dict[str, Any] = {
                    "group_column": group,
                    "group_level": level,
                    "response": response,
                    "response_role": dataset.role_of(response).value,
                    "n_dataset": n_dataset,
                    "n_group_observations": n_group,
                    "n_response_present": int(n_group - n_missing),
                    "n_response_finite": len(finite),
                    "n_response_missing": n_missing,
                    "n_response_non_finite": n_non_finite,
                    "mean": None,
                    "std": None,
                    "median": None,
                    "q25": None,
                    "q75": None,
                    "iqr": None,
                    "min": None,
                    "max": None,
                    "status": "ok",
                    "reason": None,
                }
                if len(finite) == 0:
                    row["status"] = "skipped"
                    row["reason"] = "no_finite_response_values"
                    rows.append(row)
                    continue
                q25, median, q75 = np.quantile(finite, (0.25, 0.5, 0.75))
                row.update(
                    mean=_safe_float(np.mean(finite)),
                    median=_safe_float(median),
                    q25=_safe_float(q25),
                    q75=_safe_float(q75),
                    iqr=_safe_float(q75 - q25),
                    min=_safe_float(np.min(finite)),
                    max=_safe_float(np.max(finite)),
                )
                if len(finite) >= 2:
                    row["std"] = _safe_float(np.std(finite, ddof=1))
                else:
                    row["status"] = "degenerate"
                    row["reason"] = "single_finite_response_observation"
                if len(finite) >= 2 and np.unique(finite).size == 1:
                    row["status"] = "degenerate"
                    row["reason"] = "constant_response_within_group"
                rows.append(row)
    return pl.DataFrame(rows, schema=NUMERIC_GROUP_SUMMARY_SCHEMA)


def summarize_grouped_categorical_responses(
    dataset: TabularDataset,
    responses: Iterable[str] | None = None,
    groups: Iterable[str] | None = None,
    *,
    max_group_levels: int = 20,
) -> pl.DataFrame:
    """Compute within-group frequency summaries for categorical responses."""
    selected_responses = tuple(
        response
        for response in resolve_responses(dataset, responses)
        if dataset.kind_of(response) in {ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN}
    )
    selected_groups = resolve_groups(dataset, groups)
    frame = dataset.frame
    n_dataset = frame.height
    coverage = (
        summarize_group_coverage(dataset, selected_groups, max_group_levels=max_group_levels)
        if selected_groups
        else None
    )
    coverage_status = (
        {row["group_column"]: row["status"] for row in coverage.iter_rows(named=True)}
        if coverage is not None
        else {}
    )
    rows: list[dict[str, Any]] = []
    for group in selected_groups:
        if coverage_status.get(group) == "skipped":
            continue
        group_labels = _normalized_group(frame.get_column(group))
        group_levels = sorted({label for label in group_labels if label is not None})
        for response in selected_responses:
            response_labels = _normalized_group(frame.get_column(response))
            response_levels = sorted({label for label in response_labels if label is not None})
            for group_level in group_levels:
                mask = np.array([label == group_level for label in group_labels], dtype=bool)
                n_group = int(mask.sum())
                indices = np.flatnonzero(mask)
                response_slice = [response_labels[i] for i in indices]
                n_present = sum(1 for v in response_slice if v is not None)
                n_missing = n_group - n_present
                if not response_levels:
                    rows.append(
                        {
                            "group_column": group,
                            "group_level": group_level,
                            "response": response,
                            "response_role": dataset.role_of(response).value,
                            "response_level": None,
                            "n_dataset": n_dataset,
                            "n_group_observations": n_group,
                            "n_response_present": n_present,
                            "n_response_missing": n_missing,
                            "count": 0,
                            "fraction": None,
                            "status": "skipped",
                            "reason": "all_missing_response",
                        }
                    )
                    continue
                for response_level in response_levels:
                    count = sum(1 for v in response_slice if v == response_level)
                    rows.append(
                        {
                            "group_column": group,
                            "group_level": group_level,
                            "response": response,
                            "response_role": dataset.role_of(response).value,
                            "response_level": response_level,
                            "n_dataset": n_dataset,
                            "n_group_observations": n_group,
                            "n_response_present": n_present,
                            "n_response_missing": n_missing,
                            "count": count,
                            "fraction": (float(count / n_present) if n_present else None),
                            "status": "ok" if n_present else "skipped",
                            "reason": None if n_present else "all_missing_response",
                        }
                    )
    return pl.DataFrame(rows, schema=CATEGORICAL_GROUP_SUMMARY_SCHEMA)


def summarize_annotation_coverage(
    sources: Iterable[AlignedAnnotations],
) -> pl.DataFrame:
    """Return one traceable coverage row per external annotation source."""
    rows: list[dict[str, Any]] = []
    for source in sources:
        summary = source.summary()
        rows.append(
            {
                "source_name": source.source_name,
                "coverage": source.coverage.value,
                "base_count": int(summary["base_count"]),
                "annotation_count": int(summary["annotation_count"]),
                "covered_count": int(summary["covered_count"]),
                "coverage_fraction": float(summary["coverage_fraction"]),
                "n_missing_ids": len(summary["missing_ids"]),
                "n_unmatched_ids": len(summary["unmatched_ids"]),
                "columns_json": json.dumps(list(source.columns), separators=(",", ":"), ensure_ascii=False),
            }
        )
    return pl.DataFrame(rows, schema=ANNOTATION_COVERAGE_SCHEMA)


def _group_counts(dataset: TabularDataset, group: str) -> tuple[int, int]:
    frame = dataset.frame
    missing = _n_missing(frame.get_column(group))
    present = frame.height - missing
    return present, missing


def _numeric_response_comparisons(
    dataset: TabularDataset,
    responses: tuple[str, ...],
    groups: tuple[str, ...],
    *,
    tests: tuple[ComparisonTest, ...],
    min_group_n: int,
    max_group_levels: int,
    pairwise: bool,
    p_adjust: PAdjustMethod,
) -> pl.DataFrame:
    tables: list[pl.DataFrame] = []
    n_dataset = dataset.frame.height
    for response in responses:
        if dataset.kind_of(response) is not ColumnKind.NUMERIC:
            continue
        for group in groups:
            table = summarize_numeric_categorical_comparisons(
                dataset,
                tests=tests,
                features=(response,),
                group_columns=(group,),
                min_group_n=min_group_n,
                max_group_levels=max_group_levels,
                pairwise=pairwise,
                p_adjust=PAdjustMethod.NONE,
            )
            if table.height == 0:
                continue
            n_present, n_missing = _group_counts(dataset, group)
            table = table.with_columns(
                pl.lit(response).alias("response"),
                pl.lit(n_dataset, dtype=pl.Int64).alias("n_dataset"),
                pl.lit(n_present, dtype=pl.Int64).alias("n_group_present"),
                pl.lit(n_missing, dtype=pl.Int64).alias("n_group_missing"),
                pl.format(
                    "grouped_numeric:{}:{}:{}:{}",
                    pl.lit(group),
                    pl.lit(response),
                    pl.col("scope"),
                    pl.col("test"),
                ).alias("family_id"),
                pl.lit(p_adjust.value).alias("correction"),
                pl.lit(None, dtype=pl.Float64).alias("q_value"),
                pl.lit(0, dtype=pl.Int64).alias("family_size"),
            ).select(list(_GROUPED_NUMERIC_COMPARISON_COLUMNS))
            tables.append(table)
    if not tables:
        return pl.DataFrame(schema=_GROUPED_NUMERIC_COMPARISON_SCHEMA)
    combined = pl.concat(tables, how="vertical")
    return apply_multiple_testing(combined, p_adjust)


def _categorical_response_comparisons(
    dataset: TabularDataset,
    responses: tuple[str, ...],
    groups: tuple[str, ...],
    *,
    tests: tuple[ComparisonTest, ...],
    max_category_levels: int,
    p_adjust: PAdjustMethod,
) -> pl.DataFrame:
    tables: list[pl.DataFrame] = []
    n_dataset = dataset.frame.height
    for response in responses:
        if dataset.kind_of(response) not in {ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN}:
            continue
        for group in groups:
            table = summarize_categorical_associations(
                dataset,
                tests=tests,
                pairs=((response, group),),
                max_category_levels=max_category_levels,
                p_adjust=PAdjustMethod.NONE,
            )
            if table.height == 0:
                continue
            n_present, n_missing = _group_counts(dataset, group)
            table = table.with_columns(
                pl.lit(response).alias("response"),
                pl.lit(group).alias("group_column"),
                pl.lit(n_dataset, dtype=pl.Int64).alias("n_dataset"),
                pl.lit(n_present, dtype=pl.Int64).alias("n_group_present"),
                pl.lit(n_missing, dtype=pl.Int64).alias("n_group_missing"),
                pl.format(
                    "grouped_categorical:{}:{}:{}",
                    pl.lit(group),
                    pl.lit(response),
                    pl.col("test"),
                ).alias("family_id"),
                pl.lit(p_adjust.value).alias("correction"),
                pl.lit(None, dtype=pl.Float64).alias("q_value"),
                pl.lit(0, dtype=pl.Int64).alias("family_size"),
            ).select(list(_GROUPED_CATEGORICAL_COMPARISON_COLUMNS))
            tables.append(table)
    if not tables:
        return pl.DataFrame(schema=_GROUPED_CATEGORICAL_COMPARISON_SCHEMA)
    combined = pl.concat(tables, how="vertical")
    return apply_multiple_testing(combined, p_adjust)


def analyze_grouped_responses(
    dataset: TabularDataset,
    *,
    responses: Iterable[str] | None = None,
    groups: Iterable[str] | None = None,
    annotations: Iterable[AlignedAnnotations] = (),
    comparison_tests: tuple[ComparisonTest | str, ...] = (
        ComparisonTest.WELCH_T,
        ComparisonTest.MANN_WHITNEY,
        ComparisonTest.WELCH_ANOVA,
        ComparisonTest.KRUSKAL_WALLIS,
        ComparisonTest.CHI_SQUARE,
        ComparisonTest.FISHER_EXACT,
    ),
    p_adjust: PAdjustMethod | str = PAdjustMethod.FDR_BH,
    min_group_n: int = 3,
    max_group_levels: int = 20,
    max_category_levels: int = 50,
    pairwise: bool = False,
) -> GroupAnalysisResult:
    """Analyze one or more responses independently across one or more groups.

    Each response is dispatched from its statistical data kind. Numeric responses
    receive numeric-vs-group inference, while categorical/boolean responses receive
    contingency-based inference. No classification/regression task label is used.
    """
    annotation_sources = tuple(annotations)
    augmented = attach_annotations(dataset, annotation_sources)
    selected_responses = resolve_responses(augmented, responses)
    selected_groups = resolve_groups(augmented, groups)
    overlap = sorted(set(selected_responses) & set(selected_groups))
    if overlap:
        raise ValueError(
            f"A column cannot be both response and grouping variable in one analysis: {overlap}."
        )
    if not selected_responses:
        raise ValueError(
            "At least one response must be selected explicitly or declared with role 'response'."
        )
    if not selected_groups:
        raise ValueError("At least one group must be selected explicitly or declared with role 'factor'.")

    tests = tuple(
        test if isinstance(test, ComparisonTest) else ComparisonTest(test) for test in comparison_tests
    )
    if len(set(tests)) != len(tests):
        raise ValueError("comparison_tests cannot contain duplicates.")
    correction = p_adjust if isinstance(p_adjust, PAdjustMethod) else PAdjustMethod(p_adjust)
    numeric_tests = tuple(test for test in tests if test in _NUMERIC_TESTS)
    categorical_tests = tuple(test for test in tests if test in _CATEGORICAL_TESTS)

    response_catalog = summarize_response_catalog(augmented, selected_responses)
    group_coverage = summarize_group_coverage(augmented, selected_groups, max_group_levels=max_group_levels)
    numeric_summaries = summarize_grouped_numeric_responses(
        augmented,
        selected_responses,
        selected_groups,
        max_group_levels=max_group_levels,
    )
    categorical_summaries = summarize_grouped_categorical_responses(
        augmented,
        selected_responses,
        selected_groups,
        max_group_levels=max_group_levels,
    )
    numeric_comparisons = _numeric_response_comparisons(
        augmented,
        selected_responses,
        selected_groups,
        tests=numeric_tests,
        min_group_n=min_group_n,
        max_group_levels=max_group_levels,
        pairwise=pairwise,
        p_adjust=correction,
    )
    categorical_comparisons = _categorical_response_comparisons(
        augmented,
        selected_responses,
        selected_groups,
        tests=categorical_tests,
        max_category_levels=max_category_levels,
        p_adjust=correction,
    )
    annotation_coverage = summarize_annotation_coverage(annotation_sources)
    provenance = AnalysisProvenance(
        analysis="grouped_responses",
        parameters={
            "responses": selected_responses,
            "groups": selected_groups,
            "comparison_tests": tuple(test.value for test in tests),
            "p_adjust": correction.value,
            "min_group_n": min_group_n,
            "max_group_levels": max_group_levels,
            "max_category_levels": max_category_levels,
            "pairwise": pairwise,
            "response_dispatch": "observed_or_declared_statistical_kind",
            "missing_group_policy": "exclude_only_from_affected_group_analysis",
            "numeric_response_policy": "finite_values_only",
        },
        input_summary={
            "n_observations": augmented.frame.height,
            "n_columns": augmented.frame.width,
            "n_responses": len(selected_responses),
            "n_groups": len(selected_groups),
            "n_annotation_sources": len(annotation_sources),
        },
    )
    return GroupAnalysisResult(
        response_catalog=response_catalog,
        group_coverage=group_coverage,
        numeric_summaries=numeric_summaries,
        categorical_summaries=categorical_summaries,
        numeric_comparisons=numeric_comparisons,
        categorical_comparisons=categorical_comparisons,
        annotation_coverage=annotation_coverage,
        provenance=provenance,
    )
