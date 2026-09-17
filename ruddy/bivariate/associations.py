"""Categorical-to-categorical association analysis."""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ruddy.core.enums import (
    ColumnKind,
    ColumnRole,
    ComparisonTest,
    PAdjustMethod,
)
from ruddy.data import TabularDataset
from ruddy.statistics import (
    apply_multiple_testing,
    bias_corrected_cramers_v,
    expected_count_diagnostics,
)
from ruddy.univariate.categorical import _category_label

ASSOCIATION_COLUMNS: tuple[str, ...] = (
    "test",
    "column_x",
    "column_y",
    "column_x_role",
    "column_y_role",
    "n_x_levels",
    "n_y_levels",
    "n_total",
    "n_used",
    "n_missing",
    "statistic",
    "df",
    "p_value",
    "q_value",
    "family_id",
    "family_size",
    "correction",
    "effect_size_name",
    "effect_size",
    "min_expected_count",
    "n_expected_lt5",
    "fraction_expected_lt5",
    "advisory_codes",
    "contingency_json",
    "status",
    "reason",
)

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


def _contingency(frame: pd.DataFrame, x: str, y: str) -> tuple[list[str], list[str], np.ndarray, int]:
    pair = frame[[x, y]].dropna()
    x_labels = pair[x].map(_category_label).astype(str)
    y_labels = pair[y].map(_category_label).astype(str)
    x_levels = sorted(x_labels.unique().tolist())
    y_levels = sorted(y_labels.unique().tolist())
    counts = np.zeros((len(x_levels), len(y_levels)), dtype=int)
    for i, x_level in enumerate(x_levels):
        for j, y_level in enumerate(y_levels):
            counts[i, j] = int((x_labels.eq(x_level) & y_labels.eq(y_level)).sum())
    return x_levels, y_levels, counts, len(pair)


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
) -> pd.DataFrame:
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

    frame = dataset.to_frame()
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
        n_total = dataset.n_observations
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
                "statistic": np.nan,
                "df": np.nan,
                "p_value": np.nan,
                "q_value": np.nan,
                "family_id": f"associations:{test.value}",
                "family_size": 0,
                "correction": correction.value,
                "effect_size_name": None,
                "effect_size": np.nan,
                "min_expected_count": np.nan,
                "n_expected_lt5": np.nan,
                "fraction_expected_lt5": np.nan,
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
                        df=float(dof),
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

    table = pd.DataFrame(rows, columns=pd.Index(ASSOCIATION_COLUMNS))
    if table.empty:
        return table
    return apply_multiple_testing(table, correction)
