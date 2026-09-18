"""Categorical-to-categorical association analysis."""

from __future__ import annotations

import json
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from scipy import stats

from ruddy.core.enums import (
    ColumnKind,
    ColumnRole,
    ComparisonTest,
    PAdjustMethod,
)
from ruddy.statistics import (
    apply_multiple_testing,
    bias_corrected_cramers_v,
    expected_count_diagnostics,
)
from ruddy.univariate.categorical import _category_label

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

    from ruddy.data import TabularDataset

ASSOCIATION_SCHEMA: dict[str, PolarsDataType] = {
    "test": pl.String,
    "column_x": pl.String,
    "column_y": pl.String,
    "column_x_role": pl.String,
    "column_y_role": pl.String,
    "n_x_levels": pl.Int64,
    "n_y_levels": pl.Int64,
    "n_total": pl.Int64,
    "n_used": pl.Int64,
    "n_missing": pl.Int64,
    "statistic": pl.Float64,
    "df": pl.Int64,
    "p_value": pl.Float64,
    "q_value": pl.Float64,
    "family_id": pl.String,
    "family_size": pl.Int64,
    "correction": pl.String,
    "effect_size_name": pl.String,
    "effect_size": pl.Float64,
    "min_expected_count": pl.Float64,
    "n_expected_lt5": pl.Int64,
    "fraction_expected_lt5": pl.Float64,
    "advisory_codes": pl.String,
    "contingency_json": pl.String,
    "status": pl.String,
    "reason": pl.String,
}
ASSOCIATION_COLUMNS: tuple[str, ...] = tuple(ASSOCIATION_SCHEMA)

_DEFAULT_TESTS = (ComparisonTest.CHI_SQUARE, ComparisonTest.FISHER_EXACT)


def _eligible_categorical(dataset: TabularDataset) -> tuple[str, ...]:
    selected: list[str] = []
    for spec in dataset.schema:
        if spec.role in {ColumnRole.IDENTIFIER, ColumnRole.EXCLUDED}:
            continue
        if spec.role is ColumnRole.FACTOR or spec.kind in {
            ColumnKind.CATEGORICAL,
            ColumnKind.BOOLEAN,
        }:
            selected.append(spec.name)
    return tuple(selected)


def _contingency(frame: pl.DataFrame, x: str, y: str) -> tuple[list[str], list[str], np.ndarray, int]:
    pair = frame.select(x, y).drop_nulls()
    if pair.schema[x].is_float():
        pair = pair.filter(~pl.col(x).is_nan())
    if pair.schema[y].is_float():
        pair = pair.filter(~pl.col(y).is_nan())

    n_used = pair.height
    counts_df = pair.group_by([x, y]).len()
    x_labels = [_category_label(v) for v in counts_df.get_column(x).to_list()]
    y_labels = [_category_label(v) for v in counts_df.get_column(y).to_list()]
    lens = counts_df.get_column("len").to_list()

    x_levels = sorted(set(x_labels))
    y_levels = sorted(set(y_labels))
    counts = np.zeros((len(x_levels), len(y_levels)), dtype=int)
    if x_labels:
        x_idx_map = {lvl: i for i, lvl in enumerate(x_levels)}
        y_idx_map = {lvl: j for j, lvl in enumerate(y_levels)}
        for xl, yl, n in zip(x_labels, y_labels, lens, strict=True):
            counts[x_idx_map[xl], y_idx_map[yl]] += n
    return x_levels, y_levels, counts, n_used


def _contingency_payload(x_levels: list[str], y_levels: list[str], counts: np.ndarray) -> str:
    return json.dumps(
        {
            "column_x_levels": x_levels,
            "column_y_levels": y_levels,
            "counts": counts.astype(int).tolist(),
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )


def summarize_categorical_associations(
    dataset: TabularDataset,
    *,
    tests: tuple[ComparisonTest | str, ...] = _DEFAULT_TESTS,
    pairs: tuple[tuple[str, str], ...] | None = None,
    max_category_levels: int = 50,
    p_adjust: PAdjustMethod | str = PAdjustMethod.FDR_BH,
) -> pl.DataFrame:
    """Compute selected or all unordered categorical-categorical associations."""
    if max_category_levels < 2:
        raise ValueError("max_category_levels must be at least 2.")
    resolved_tests = tuple(
        test if isinstance(test, ComparisonTest) else ComparisonTest(test) for test in tests
    )
    invalid = [test.value for test in resolved_tests if test not in _DEFAULT_TESTS]
    if invalid:
        raise ValueError("Categorical associations received numeric tests: " + ", ".join(invalid))
    if len(set(resolved_tests)) != len(resolved_tests):
        raise ValueError("Association tests cannot contain duplicates.")
    correction = p_adjust if isinstance(p_adjust, PAdjustMethod) else PAdjustMethod(p_adjust)

    frame = dataset.frame
    candidates = _eligible_categorical(dataset)
    if pairs is None:
        selected_pairs = tuple(combinations(candidates, 2))
    else:
        selected_pairs = tuple((str(x), str(y)) for x, y in pairs)
        if len(set(selected_pairs)) != len(selected_pairs):
            raise ValueError("pairs cannot contain duplicates.")
        invalid = [(x, y) for x, y in selected_pairs if x == y or x not in candidates or y not in candidates]
        if invalid:
            raise ValueError(
                f"Selected association pairs must contain distinct eligible categorical columns: {invalid}."
            )
    rows: list[dict[str, Any]] = []
    for column_x, column_y in selected_pairs:
        x_levels, y_levels, counts, n_used = _contingency(frame, column_x, column_y)
        n_total = dataset.frame.height
        for test in resolved_tests:
            base: dict[str, Any] = {
                "test": test.value,
                "column_x": column_x,
                "column_y": column_y,
                "column_x_role": dataset.role_of(column_x).value,
                "column_y_role": dataset.role_of(column_y).value,
                "n_x_levels": len(x_levels),
                "n_y_levels": len(y_levels),
                "n_total": n_total,
                "n_used": n_used,
                "n_missing": n_total - n_used,
                "statistic": None,
                "df": None,
                "p_value": None,
                "q_value": None,
                "family_id": f"associations:{test.value}",
                "family_size": 0,
                "correction": correction.value,
                "effect_size_name": None,
                "effect_size": None,
                "min_expected_count": None,
                "n_expected_lt5": None,
                "fraction_expected_lt5": None,
                "advisory_codes": None,
                "contingency_json": _contingency_payload(x_levels, y_levels, counts),
                "status": "ok",
                "reason": None,
            }
            if len(x_levels) > max_category_levels or len(y_levels) > max_category_levels:
                base["status"] = "skipped"
                base["reason"] = "feature_levels_exceed_max_category_levels"
                rows.append(base)
                continue
            if counts.shape[0] < 2 or counts.shape[1] < 2:
                base["status"] = "degenerate"
                base["reason"] = "insufficient_contingency_dimensions"
                rows.append(base)
                continue

            chi2, chi_p, dof, expected = stats.chi2_contingency(counts, correction=False)
            diagnostics = expected_count_diagnostics(expected)
            base.update(
                min_expected_count=diagnostics["min_expected_count"],
                n_expected_lt5=diagnostics["n_expected_lt5"],
                fraction_expected_lt5=diagnostics["fraction_expected_lt5"],
                advisory_codes=("low_expected_counts" if diagnostics["low_expected_counts"] else None),
            )

            if test is ComparisonTest.CHI_SQUARE:
                effect = bias_corrected_cramers_v(float(chi2), n_used, counts.shape[0], counts.shape[1])
                if effect is None:
                    base["status"] = "degenerate"
                    base["reason"] = "undefined_cramers_v"
                else:
                    base.update(
                        statistic=float(chi2),
                        df=int(dof),
                        p_value=float(chi_p),
                        effect_size_name="cramers_v_bias_corrected",
                        effect_size=effect,
                    )
                rows.append(base)
                continue

            if counts.shape != (2, 2):
                base["status"] = "skipped"
                base["reason"] = "fisher_exact_requires_2x2"
                rows.append(base)
                continue
            fisher = stats.fisher_exact(counts, alternative="two-sided")
            odds_ratio = float(fisher.statistic)
            base.update(
                statistic=odds_ratio,
                p_value=float(fisher.pvalue),
                effect_size_name="odds_ratio",
                effect_size=odds_ratio,
            )
            rows.append(base)

    table = pl.DataFrame(rows, schema=ASSOCIATION_SCHEMA)
    return table if table.height == 0 else apply_multiple_testing(table, correction)
