"""Response-centric grouped summaries and inference."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from ruddy.bivariate.associations import ASSOCIATION_COLUMNS, summarize_categorical_associations
from ruddy.bivariate.comparisons import COMPARISON_COLUMNS, summarize_numeric_categorical_comparisons
from ruddy.core.enums import ColumnKind, ColumnRole, ComparisonTest, PAdjustMethod
from ruddy.data import AlignedAnnotations, TabularDataset, attach_annotations
from ruddy.results import AnalysisProvenance
from ruddy.statistics import apply_multiple_testing
from ruddy.univariate.categorical import _category_label


GROUP_COVERAGE_COLUMNS: tuple[str, ...] = (
    "group_column",
    "group_role",
    "data_kind",
    "n_dataset",
    "n_group_present",
    "n_group_missing",
    "coverage_fraction",
    "n_levels",
    "levels_json",
    "status",
    "reason",
)

RESPONSE_CATALOG_COLUMNS: tuple[str, ...] = (
    "response",
    "declared_role",
    "data_kind",
    "n_dataset",
    "n_present",
    "n_missing",
    "n_finite",
    "n_non_finite",
    "status",
    "reason",
)

NUMERIC_GROUP_SUMMARY_COLUMNS: tuple[str, ...] = (
    "group_column",
    "group_level",
    "response",
    "response_role",
    "n_dataset",
    "n_group_observations",
    "n_response_present",
    "n_response_finite",
    "n_response_missing",
    "n_response_non_finite",
    "mean",
    "std",
    "median",
    "q25",
    "q75",
    "iqr",
    "min",
    "max",
    "status",
    "reason",
)

CATEGORICAL_GROUP_SUMMARY_COLUMNS: tuple[str, ...] = (
    "group_column",
    "group_level",
    "response",
    "response_role",
    "response_level",
    "n_dataset",
    "n_group_observations",
    "n_response_present",
    "n_response_missing",
    "count",
    "fraction",
    "status",
    "reason",
)

ANNOTATION_COVERAGE_COLUMNS: tuple[str, ...] = (
    "source_name",
    "coverage",
    "base_count",
    "annotation_count",
    "covered_count",
    "coverage_fraction",
    "n_missing_ids",
    "n_unmatched_ids",
    "columns_json",
)

_GROUPED_NUMERIC_COMPARISON_COLUMNS = (
    "response",
    "n_dataset",
    "n_group_present",
    "n_group_missing",
    *COMPARISON_COLUMNS,
)
_GROUPED_CATEGORICAL_COMPARISON_COLUMNS = (
    "response",
    "group_column",
    "n_dataset",
    "n_group_present",
    "n_group_missing",
    *ASSOCIATION_COLUMNS,
)

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

    response_catalog: pd.DataFrame
    group_coverage: pd.DataFrame
    numeric_summaries: pd.DataFrame
    categorical_summaries: pd.DataFrame
    numeric_comparisons: pd.DataFrame
    categorical_comparisons: pd.DataFrame
    annotation_coverage: pd.DataFrame
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
    unknown = [column for column in selected if column not in dataset.columns]
    if unknown:
        raise ValueError(f"Unknown response columns: {unknown}.")
    invalid_role = [
        column
        for column in selected
        if dataset.role_of(column) in {ColumnRole.IDENTIFIER, ColumnRole.EXCLUDED}
    ]
    if invalid_role:
        raise ValueError(
            f"Response columns cannot be identifier/excluded columns: {invalid_role}."
        )
    invalid_kind = [
        column
        for column in selected
        if dataset.kind_of(column)
        not in {ColumnKind.NUMERIC, ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN}
    ]
    if invalid_kind:
        raise ValueError(
            "Responses must be numeric, categorical, or boolean; invalid columns: "
            f"{invalid_kind}."
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
    unknown = [column for column in selected if column not in dataset.columns]
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


def _normalized_group(series: pd.Series) -> pd.Series:
    result = pd.Series(index=series.index, dtype="object")
    present = series.notna()
    result.loc[present] = series.loc[present].map(_category_label).astype(str)
    return result


def summarize_group_coverage(
    dataset: TabularDataset,
    groups: Iterable[str] | None = None,
    *,
    max_group_levels: int = 20,
) -> pd.DataFrame:
    """Describe observation coverage for each grouping variable."""

    if max_group_levels < 2:
        raise ValueError("max_group_levels must be at least 2.")
    selected = resolve_groups(dataset, groups)
    frame = dataset.to_frame()
    rows: list[dict[str, Any]] = []
    for group in selected:
        normalized = _normalized_group(frame[group])
        levels = tuple(sorted(normalized.dropna().unique().tolist()))
        n_present = int(normalized.notna().sum())
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
                "n_dataset": dataset.n_observations,
                "n_group_present": n_present,
                "n_group_missing": dataset.n_observations - n_present,
                "coverage_fraction": (
                    float(n_present / dataset.n_observations)
                    if dataset.n_observations
                    else 1.0
                ),
                "n_levels": len(levels),
                "levels_json": json.dumps(levels, separators=(",", ":")),
                "status": status,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=GROUP_COVERAGE_COLUMNS)


def summarize_response_catalog(
    dataset: TabularDataset,
    responses: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Describe selected response variables and their usable observations."""

    selected = resolve_responses(dataset, responses)
    frame = dataset.to_frame()
    rows: list[dict[str, Any]] = []
    for response in selected:
        series = frame[response]
        n_missing = int(series.isna().sum())
        n_present = int(len(series) - n_missing)
        kind = dataset.kind_of(response)
        n_finite: int | None = None
        n_non_finite: int | None = None
        status = "ok"
        reason = None
        if kind is ColumnKind.NUMERIC:
            values = series.dropna().to_numpy(dtype=np.float64, na_value=np.nan)
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
                "n_dataset": dataset.n_observations,
                "n_present": n_present,
                "n_missing": n_missing,
                "n_finite": n_finite,
                "n_non_finite": n_non_finite,
                "status": status,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=RESPONSE_CATALOG_COLUMNS)


def _safe_float(value: object) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def summarize_grouped_numeric_responses(
    dataset: TabularDataset,
    responses: Iterable[str] | None = None,
    groups: Iterable[str] | None = None,
    *,
    max_group_levels: int = 20,
) -> pd.DataFrame:
    """Compute descriptive summaries for numeric responses within groups."""

    selected_responses = tuple(
        response
        for response in resolve_responses(dataset, responses)
        if dataset.kind_of(response) is ColumnKind.NUMERIC
    )
    selected_groups = resolve_groups(dataset, groups)
    frame = dataset.to_frame()
    coverage = summarize_group_coverage(
        dataset, selected_groups, max_group_levels=max_group_levels
    ).set_index("group_column") if selected_groups else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for group in selected_groups:
        if not coverage.empty and coverage.loc[group, "status"] == "skipped":
            continue
        normalized = _normalized_group(frame[group])
        levels = tuple(sorted(normalized.dropna().unique().tolist()))
        for level in levels:
            mask = normalized.eq(level)
            n_group = int(mask.sum())
            for response in selected_responses:
                series = frame.loc[mask, response]
                n_missing = int(series.isna().sum())
                present = series.dropna().to_numpy(dtype=np.float64, na_value=np.nan)
                finite = present[np.isfinite(present)]
                n_non_finite = int(len(present) - len(finite))
                row: dict[str, Any] = {
                    "group_column": group,
                    "group_level": level,
                    "response": response,
                    "response_role": dataset.role_of(response).value,
                    "n_dataset": dataset.n_observations,
                    "n_group_observations": n_group,
                    "n_response_present": int(n_group - n_missing),
                    "n_response_finite": int(len(finite)),
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
    return pd.DataFrame(rows, columns=NUMERIC_GROUP_SUMMARY_COLUMNS)


def summarize_grouped_categorical_responses(
    dataset: TabularDataset,
    responses: Iterable[str] | None = None,
    groups: Iterable[str] | None = None,
    *,
    max_group_levels: int = 20,
) -> pd.DataFrame:
    """Compute within-group frequency summaries for categorical responses."""

    selected_responses = tuple(
        response
        for response in resolve_responses(dataset, responses)
        if dataset.kind_of(response) in {ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN}
    )
    selected_groups = resolve_groups(dataset, groups)
    frame = dataset.to_frame()
    coverage = summarize_group_coverage(
        dataset, selected_groups, max_group_levels=max_group_levels
    ).set_index("group_column") if selected_groups else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for group in selected_groups:
        if not coverage.empty and coverage.loc[group, "status"] == "skipped":
            continue
        normalized_group = _normalized_group(frame[group])
        group_levels = tuple(sorted(normalized_group.dropna().unique().tolist()))
        for response in selected_responses:
            normalized_response = _normalized_group(frame[response])
            response_levels = tuple(
                sorted(normalized_response.dropna().unique().tolist())
            )
            for group_level in group_levels:
                mask = normalized_group.eq(group_level)
                n_group = int(mask.sum())
                response_slice = normalized_response.loc[mask]
                n_present = int(response_slice.notna().sum())
                n_missing = n_group - n_present
                if not response_levels:
                    rows.append(
                        {
                            "group_column": group,
                            "group_level": group_level,
                            "response": response,
                            "response_role": dataset.role_of(response).value,
                            "response_level": None,
                            "n_dataset": dataset.n_observations,
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
                    count = int(response_slice.eq(response_level).sum())
                    rows.append(
                        {
                            "group_column": group,
                            "group_level": group_level,
                            "response": response,
                            "response_role": dataset.role_of(response).value,
                            "response_level": response_level,
                            "n_dataset": dataset.n_observations,
                            "n_group_observations": n_group,
                            "n_response_present": n_present,
                            "n_response_missing": n_missing,
                            "count": count,
                            "fraction": (
                                float(count / n_present) if n_present else None
                            ),
                            "status": "ok" if n_present else "skipped",
                            "reason": None if n_present else "all_missing_response",
                        }
                    )
    return pd.DataFrame(rows, columns=CATEGORICAL_GROUP_SUMMARY_COLUMNS)


def summarize_annotation_coverage(
    sources: Iterable[AlignedAnnotations],
) -> pd.DataFrame:
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
                "columns_json": json.dumps(
                    list(source.columns), separators=(",", ":"), ensure_ascii=False
                ),
            }
        )
    return pd.DataFrame(rows, columns=ANNOTATION_COVERAGE_COLUMNS)


def _group_counts(dataset: TabularDataset, group: str) -> tuple[int, int]:
    present = int(dataset.to_frame()[group].notna().sum())
    return present, dataset.n_observations - present


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
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
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
            if table.empty:
                continue
            n_present, n_missing = _group_counts(dataset, group)
            table = table.copy()
            table.insert(0, "response", response)
            table.insert(1, "n_dataset", dataset.n_observations)
            table.insert(2, "n_group_present", n_present)
            table.insert(3, "n_group_missing", n_missing)
            table["family_id"] = table.apply(
                lambda row: (
                    f"grouped_numeric:{group}:{response}:{row['scope']}:{row['test']}"
                ),
                axis=1,
            )
            table["correction"] = p_adjust.value
            table["q_value"] = np.nan
            table["family_size"] = 0
            rows.append(table)
    if not rows:
        return pd.DataFrame(columns=_GROUPED_NUMERIC_COMPARISON_COLUMNS)
    combined = pd.concat(rows, ignore_index=True)
    combined = apply_multiple_testing(combined, p_adjust)
    return combined.loc[:, _GROUPED_NUMERIC_COMPARISON_COLUMNS]


def _categorical_response_comparisons(
    dataset: TabularDataset,
    responses: tuple[str, ...],
    groups: tuple[str, ...],
    *,
    tests: tuple[ComparisonTest, ...],
    max_category_levels: int,
    p_adjust: PAdjustMethod,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
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
            if table.empty:
                continue
            n_present, n_missing = _group_counts(dataset, group)
            table = table.copy()
            table.insert(0, "response", response)
            table.insert(1, "group_column", group)
            table.insert(2, "n_dataset", dataset.n_observations)
            table.insert(3, "n_group_present", n_present)
            table.insert(4, "n_group_missing", n_missing)
            table["family_id"] = table["test"].map(
                lambda test: f"grouped_categorical:{group}:{response}:{test}"
            )
            table["correction"] = p_adjust.value
            table["q_value"] = np.nan
            table["family_size"] = 0
            rows.append(table)
    if not rows:
        return pd.DataFrame(columns=_GROUPED_CATEGORICAL_COMPARISON_COLUMNS)
    combined = pd.concat(rows, ignore_index=True)
    combined = apply_multiple_testing(combined, p_adjust)
    return combined.loc[:, _GROUPED_CATEGORICAL_COMPARISON_COLUMNS]


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
        raise ValueError(
            "At least one group must be selected explicitly or declared with role 'factor'."
        )

    tests = tuple(
        test if isinstance(test, ComparisonTest) else ComparisonTest(test)
        for test in comparison_tests
    )
    if len(set(tests)) != len(tests):
        raise ValueError("comparison_tests cannot contain duplicates.")
    correction = p_adjust if isinstance(p_adjust, PAdjustMethod) else PAdjustMethod(p_adjust)
    numeric_tests = tuple(test for test in tests if test in _NUMERIC_TESTS)
    categorical_tests = tuple(test for test in tests if test in _CATEGORICAL_TESTS)

    response_catalog = summarize_response_catalog(augmented, selected_responses)
    group_coverage = summarize_group_coverage(
        augmented, selected_groups, max_group_levels=max_group_levels
    )
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
            "n_observations": augmented.n_observations,
            "n_columns": augmented.n_columns,
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
