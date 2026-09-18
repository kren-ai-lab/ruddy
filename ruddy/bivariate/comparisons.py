"""Numeric-to-categorical comparison tests."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from ruddy.bivariate.associations import _eligible_categorical
from ruddy.core.enums import (
    ComparisonTest,
    PAdjustMethod,
)
from ruddy.data import TabularDataset
from ruddy.statistics import (
    apply_multiple_testing,
    cliffs_delta_from_u,
    epsilon_squared,
    eta_squared,
    hedges_g,
)
from ruddy.univariate.categorical import _category_label
from ruddy.univariate.diagnostics import _eligible_numeric

COMPARISON_SCHEMA: dict[str, pl.DataType] = {
    "group_column": pl.String,
    "group_role": pl.String,
    "scope": pl.String,
    "test": pl.String,
    "feature": pl.String,
    "feature_role": pl.String,
    "n_groups": pl.Int64,
    "group_a": pl.String,
    "group_b": pl.String,
    "group_a_n": pl.Int64,
    "group_b_n": pl.Int64,
    "min_group_n_used": pl.Int64,
    "max_group_n_used": pl.Int64,
    "n_total": pl.Int64,
    "n_used": pl.Int64,
    "n_missing_or_nonfinite": pl.Int64,
    "statistic": pl.Float64,
    "df1": pl.Float64,
    "df2": pl.Float64,
    "p_value": pl.Float64,
    "q_value": pl.Float64,
    "family_id": pl.String,
    "family_size": pl.Int64,
    "correction": pl.String,
    "effect_size_name": pl.String,
    "effect_size": pl.Float64,
    "status": pl.String,
    "reason": pl.String,
}
COMPARISON_COLUMNS: tuple[str, ...] = tuple(COMPARISON_SCHEMA)

_TWO_GROUP_TESTS = (ComparisonTest.WELCH_T, ComparisonTest.MANN_WHITNEY)
_OMNIBUS_TESTS = (ComparisonTest.WELCH_ANOVA, ComparisonTest.KRUSKAL_WALLIS)
_DEFAULT_TESTS = _TWO_GROUP_TESTS + _OMNIBUS_TESTS


def _finite_values(series: pl.Series) -> np.ndarray:
    values = series.cast(pl.Float64).fill_null(float("nan")).to_numpy()
    return values[np.isfinite(values)]


def _safe_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _welch_anova(groups: tuple[np.ndarray, ...]) -> tuple[float, float, float, float] | None:
    """Welch one-way ANOVA independent of SciPy version-specific APIs."""
    k = len(groups)
    if k < 2 or any(len(group) < 2 for group in groups):
        return None
    means = np.asarray([np.mean(group) for group in groups], dtype=np.float64)
    variances = np.asarray([np.var(group, ddof=1) for group in groups], dtype=np.float64)
    sizes = np.asarray([len(group) for group in groups], dtype=np.float64)
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
        return None
    weights = sizes / variances
    weight_sum = float(np.sum(weights))
    weighted_mean = float(np.sum(weights * means) / weight_sum)
    numerator = float(np.sum(weights * (means - weighted_mean) ** 2) / (k - 1))
    correction_sum = float(np.sum(((1.0 - weights / weight_sum) ** 2) / (sizes - 1.0)))
    if correction_sum <= 0.0:
        return None
    denominator = 1.0 + (2.0 * (k - 2.0) / (k**2 - 1.0)) * correction_sum
    f_value = numerator / denominator
    df1 = float(k - 1)
    df2 = float((k**2 - 1.0) / (3.0 * correction_sum))
    p_value = float(stats.f.sf(f_value, df1, df2))
    values = (f_value, p_value, df1, df2)
    return values if all(math.isfinite(value) for value in values) else None


def _all_values_identical(groups: tuple[np.ndarray, ...]) -> bool:
    if not groups:
        return True
    pooled = np.concatenate(groups)
    return bool(pooled.size == 0 or np.all(pooled == pooled[0]))


def _base_row(
    *,
    group_column: str,
    group_role: str,
    scope: str,
    test: ComparisonTest,
    feature: str,
    feature_role: str,
    labels: tuple[str, ...],
    arrays: tuple[np.ndarray, ...],
    group_sizes: tuple[int, ...],
    correction: PAdjustMethod,
    group_a: str | None = None,
    group_b: str | None = None,
) -> dict[str, Any]:
    used_sizes = tuple(len(values) for values in arrays)
    n_total = int(sum(group_sizes))
    n_used = int(sum(used_sizes))
    if scope == "pairwise":
        family_id = f"comparisons:{group_column}:pairwise:{feature}:{test.value}"
    else:
        family_id = f"comparisons:{group_column}:{scope}:{test.value}"
    return {
        "group_column": group_column,
        "group_role": group_role,
        "scope": scope,
        "test": test.value,
        "feature": feature,
        "feature_role": feature_role,
        "n_groups": len(labels),
        "group_a": group_a,
        "group_b": group_b,
        "group_a_n": used_sizes[0] if len(used_sizes) == 2 else None,
        "group_b_n": used_sizes[1] if len(used_sizes) == 2 else None,
        "min_group_n_used": min(used_sizes) if used_sizes else 0,
        "max_group_n_used": max(used_sizes) if used_sizes else 0,
        "n_total": n_total,
        "n_used": n_used,
        "n_missing_or_nonfinite": n_total - n_used,
        "statistic": None,
        "df1": None,
        "df2": None,
        "p_value": None,
        "q_value": None,
        "family_id": family_id,
        "family_size": 0,
        "correction": correction.value,
        "effect_size_name": None,
        "effect_size": None,
        "status": "ok",
        "reason": None,
    }


def _run_two_group(
    row: dict[str, Any],
    test: ComparisonTest,
    a: np.ndarray,
    b: np.ndarray,
    *,
    min_group_n: int,
) -> dict[str, Any]:
    if min(len(a), len(b)) < min_group_n:
        row["status"] = "skipped"
        row["reason"] = "insufficient_group_observations"
        return row
    if _all_values_identical((a, b)):
        row["status"] = "degenerate"
        row["reason"] = "all_values_identical"
        return row

    if test is ComparisonTest.WELCH_T:
        var_a = float(np.var(a, ddof=1))
        var_b = float(np.var(b, ddof=1))
        if var_a <= 0.0 and var_b <= 0.0:
            row["status"] = "degenerate"
            row["reason"] = "zero_within_group_variance"
            return row
        result = stats.ttest_ind(a, b, equal_var=False, alternative="two-sided")
        statistic = _safe_float(result.statistic)
        p_value = _safe_float(result.pvalue)
        df = _safe_float(getattr(result, "df", None))
        effect = hedges_g(a, b)
        if statistic is None or p_value is None or df is None or effect is None:
            row["status"] = "degenerate"
            row["reason"] = "non_finite_inference"
            return row
        row.update(
            statistic=statistic,
            df1=df,
            p_value=p_value,
            effect_size_name="hedges_g",
            effect_size=effect,
        )
        return row

    if test is ComparisonTest.MANN_WHITNEY:
        result = stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
        statistic = _safe_float(result.statistic)
        p_value = _safe_float(result.pvalue)
        effect = cliffs_delta_from_u(float(result.statistic), len(a), len(b))
        if statistic is None or p_value is None or effect is None:
            row["status"] = "degenerate"
            row["reason"] = "non_finite_inference"
            return row
        row.update(
            statistic=statistic,
            p_value=p_value,
            effect_size_name="cliffs_delta",
            effect_size=effect,
        )
        return row

    raise ValueError(f"Unsupported two-group test: {test.value}.")


def _run_omnibus(
    row: dict[str, Any],
    test: ComparisonTest,
    groups: tuple[np.ndarray, ...],
    *,
    min_group_n: int,
) -> dict[str, Any]:
    if any(len(group) < min_group_n for group in groups):
        row["status"] = "skipped"
        row["reason"] = "insufficient_group_observations"
        return row
    if _all_values_identical(groups):
        row["status"] = "degenerate"
        row["reason"] = "all_values_identical"
        return row

    if test is ComparisonTest.WELCH_ANOVA:
        result = _welch_anova(groups)
        if result is None:
            row["status"] = "degenerate"
            row["reason"] = "zero_or_invalid_within_group_variance"
            return row
        statistic, p_value, df1, df2 = result
        effect = eta_squared(groups)
        if effect is None:
            row["status"] = "degenerate"
            row["reason"] = "undefined_effect_size"
            return row
        row.update(
            statistic=statistic,
            df1=df1,
            df2=df2,
            p_value=p_value,
            effect_size_name="eta_squared",
            effect_size=effect,
        )
        return row

    if test is ComparisonTest.KRUSKAL_WALLIS:
        try:
            result = stats.kruskal(*groups, nan_policy="raise")
        except ValueError:
            row["status"] = "degenerate"
            row["reason"] = "all_values_identical"
            return row
        statistic = _safe_float(result.statistic)
        p_value = _safe_float(result.pvalue)
        effect = epsilon_squared(float(result.statistic), groups)
        if statistic is None or p_value is None or effect is None:
            row["status"] = "degenerate"
            row["reason"] = "non_finite_inference"
            return row
        row.update(
            statistic=statistic,
            p_value=p_value,
            effect_size_name="epsilon_squared",
            effect_size=effect,
        )
        return row

    raise ValueError(f"Unsupported omnibus test: {test.value}.")


def _group_values(
    frame: pl.DataFrame,
    group_column: str,
    feature: str,
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], tuple[int, ...]]:
    group_series = frame.get_column(group_column)
    if group_series.dtype.is_float():
        present_mask = ~(group_series.is_null() | group_series.is_nan()).to_numpy()
    else:
        present_mask = ~group_series.is_null().to_numpy()

    group_list = group_series.to_list()
    present_indices = np.flatnonzero(present_mask)
    labels_present = [_category_label(group_list[i]) for i in present_indices]
    labels = tuple(sorted(set(labels_present)))

    feature_values = frame.get_column(feature).cast(pl.Float64).fill_null(float("nan")).to_numpy()

    label_array = np.empty(len(group_list), dtype=object)
    for idx, label in zip(present_indices, labels_present, strict=True):
        label_array[idx] = label

    arrays: list[np.ndarray] = []
    sizes: list[int] = []
    for label in labels:
        mask = present_mask & (label_array == label)
        sizes.append(int(mask.sum()))
        vals = feature_values[mask]
        arrays.append(vals[np.isfinite(vals)])
    return labels, tuple(arrays), tuple(sizes)


def summarize_numeric_categorical_comparisons(
    dataset: TabularDataset,
    *,
    tests: tuple[ComparisonTest | str, ...] = _DEFAULT_TESTS,
    features: tuple[str, ...] | None = None,
    group_columns: tuple[str, ...] | None = None,
    min_group_n: int = 3,
    max_group_levels: int = 20,
    pairwise: bool = False,
    p_adjust: PAdjustMethod | str = PAdjustMethod.FDR_BH,
) -> pl.DataFrame:
    """Compare selected numeric features across selected categorical group variables."""
    if min_group_n < 2:
        raise ValueError("min_group_n must be at least 2.")
    if max_group_levels < 2:
        raise ValueError("max_group_levels must be at least 2.")
    resolved_tests = tuple(
        test if isinstance(test, ComparisonTest) else ComparisonTest(test) for test in tests
    )
    invalid = [test.value for test in resolved_tests if test not in set(_TWO_GROUP_TESTS + _OMNIBUS_TESTS)]
    if invalid:
        raise ValueError("Numeric comparisons received categorical tests: " + ", ".join(invalid))
    if len(set(resolved_tests)) != len(resolved_tests):
        raise ValueError("Comparison tests cannot contain duplicates.")
    correction = p_adjust if isinstance(p_adjust, PAdjustMethod) else PAdjustMethod(p_adjust)

    eligible_numeric = _eligible_numeric(dataset)
    eligible_categorical = _eligible_categorical(dataset)
    numeric = eligible_numeric if features is None else tuple(features)
    categorical = eligible_categorical if group_columns is None else tuple(group_columns)
    if len(set(numeric)) != len(numeric):
        raise ValueError("features cannot contain duplicates.")
    if len(set(categorical)) != len(categorical):
        raise ValueError("group_columns cannot contain duplicates.")
    invalid_numeric = [column for column in numeric if column not in eligible_numeric]
    invalid_groups = [column for column in categorical if column not in eligible_categorical]
    if invalid_numeric:
        raise ValueError("Selected numeric features are not eligible: " + ", ".join(invalid_numeric))
    if invalid_groups:
        raise ValueError(
            "Selected group columns are not eligible categorical/factor variables: "
            + ", ".join(invalid_groups)
        )
    frame = dataset.frame
    rows: list[dict[str, Any]] = []

    for group_column in categorical:
        group_role = dataset.role_of(group_column).value
        group_series = frame.get_column(group_column)
        if group_series.dtype.is_float():
            present_mask = ~(group_series.is_null() | group_series.is_nan()).to_numpy()
        else:
            present_mask = ~group_series.is_null().to_numpy()
        group_list = group_series.to_list()
        labels_present = [_category_label(group_list[i]) for i in np.flatnonzero(present_mask)]
        observed_levels = tuple(sorted(set(labels_present)))
        n_groups = len(observed_levels)
        if n_groups < 2:
            continue
        applicable = _TWO_GROUP_TESTS if n_groups == 2 else _OMNIBUS_TESTS
        applicable = tuple(test for test in resolved_tests if test in applicable)
        pairwise_tests = tuple(test for test in resolved_tests if test in _TWO_GROUP_TESTS)

        for feature in numeric:
            labels, arrays, sizes = _group_values(frame, group_column, feature)
            feature_role = dataset.role_of(feature).value
            if len(labels) > max_group_levels:
                for test in applicable:
                    row = _base_row(
                        group_column=group_column,
                        group_role=group_role,
                        scope="two_group" if len(labels) == 2 else "omnibus",
                        test=test,
                        feature=feature,
                        feature_role=feature_role,
                        labels=labels,
                        arrays=arrays,
                        group_sizes=sizes,
                        correction=correction,
                        group_a=labels[0] if len(labels) == 2 else None,
                        group_b=labels[1] if len(labels) == 2 else None,
                    )
                    row["status"] = "skipped"
                    row["reason"] = "group_levels_exceed_max_group_levels"
                    rows.append(row)
                continue

            if len(labels) == 2:
                for test in applicable:
                    row = _base_row(
                        group_column=group_column,
                        group_role=group_role,
                        scope="two_group",
                        test=test,
                        feature=feature,
                        feature_role=feature_role,
                        labels=labels,
                        arrays=arrays,
                        group_sizes=sizes,
                        correction=correction,
                        group_a=labels[0],
                        group_b=labels[1],
                    )
                    rows.append(_run_two_group(row, test, arrays[0], arrays[1], min_group_n=min_group_n))
                continue

            for test in applicable:
                row = _base_row(
                    group_column=group_column,
                    group_role=group_role,
                    scope="omnibus",
                    test=test,
                    feature=feature,
                    feature_role=feature_role,
                    labels=labels,
                    arrays=arrays,
                    group_sizes=sizes,
                    correction=correction,
                )
                rows.append(_run_omnibus(row, test, arrays, min_group_n=min_group_n))

            if pairwise:
                for left, right in combinations(range(len(labels)), 2):
                    pair_labels = (labels[left], labels[right])
                    pair_arrays = (arrays[left], arrays[right])
                    pair_sizes = (sizes[left], sizes[right])
                    for test in pairwise_tests:
                        row = _base_row(
                            group_column=group_column,
                            group_role=group_role,
                            scope="pairwise",
                            test=test,
                            feature=feature,
                            feature_role=feature_role,
                            labels=pair_labels,
                            arrays=pair_arrays,
                            group_sizes=pair_sizes,
                            correction=correction,
                            group_a=pair_labels[0],
                            group_b=pair_labels[1],
                        )
                        rows.append(
                            _run_two_group(
                                row,
                                test,
                                pair_arrays[0],
                                pair_arrays[1],
                                min_group_n=min_group_n,
                            )
                        )

    table = pl.DataFrame(rows, schema=COMPARISON_SCHEMA)
    return table if table.height == 0 else apply_multiple_testing(table, correction)
