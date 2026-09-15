from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ruddy import ColumnRole, TabularDataset, analyze_univariate
from ruddy.profiling import profile_columns
from ruddy.univariate import (
    numeric_statistics_columns,
    quantile_column_name,
    summarize_categorical_statistics,
    summarize_datetime_statistics,
    summarize_numeric_statistics,
)


def _dataset() -> TabularDataset:
    frame = pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(1, 9)],
            "measurement": [0.0, 1.0, 2.0, np.nan, np.inf, 4.0, 5.0, 6.0],
            "constant_numeric": [7.0] * 8,
            "tiny_numeric": [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "source": ["A", "A", "A", "B", "B", "C", "C", None],
            "flag": [True, True, False, False, True, False, True, False],
            "factor_code": [1, 1, 1, 2, 2, 3, 3, 3],
            "constant_category": ["only"] * 8,
            "all_missing_category": pd.Series([None] * 8, dtype="object"),
            "when": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    None,
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-08",
                ]
            ),
        }
    )
    return TabularDataset(
        frame,
        id_column="id",
        role_overrides={"factor_code": ColumnRole.FACTOR},
    )


def test_default_quantile_names_are_stable() -> None:
    quantiles = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    assert [quantile_column_name(q) for q in quantiles] == ["q01", "q05", "q25", "q50", "q75", "q95", "q99"]


def test_numeric_statistics_follow_finite_value_policy_and_add_variance_range() -> None:
    dataset = _dataset()
    stats = summarize_numeric_statistics(dataset, profile_columns(dataset)).set_index("column")
    row = stats.loc["measurement"]
    finite = np.asarray([0.0, 1.0, 2.0, 4.0, 5.0, 6.0])

    assert row["n_total"] == 8
    assert row["n_missing"] == 1
    assert row["n_non_finite"] == 1
    assert row["n_finite"] == 6
    assert row["mean"] == pytest.approx(np.mean(finite))
    assert row["std"] == pytest.approx(np.std(finite, ddof=1))
    assert row["variance"] == pytest.approx(np.var(finite, ddof=1))
    assert row["range"] == pytest.approx(6.0)
    assert row["iqr"] == pytest.approx(np.quantile(finite, 0.75) - np.quantile(finite, 0.25))
    assert row["mad"] == pytest.approx(np.median(np.abs(finite - np.median(finite))))
    assert row["zero_count"] == 1
    assert row["zero_fraction"] == pytest.approx(1 / 6)
    assert row["status"] == "ok"
    assert pd.isna(row["reason"])


def test_numeric_degenerate_and_insufficient_states_are_kept() -> None:
    dataset = _dataset()
    stats = summarize_numeric_statistics(dataset, profile_columns(dataset), min_numeric_n=3).set_index(
        "column"
    )
    constant = stats.loc["constant_numeric"]
    tiny = stats.loc["tiny_numeric"]
    assert constant["status"] == "degenerate"
    assert constant["reason"] == "constant"
    assert constant["std"] == pytest.approx(0.0)
    assert pd.isna(constant["skewness"])
    assert tiny["status"] == "skipped"
    assert tiny["reason"] == "insufficient_finite_observations"
    assert tiny["mean"] == pytest.approx(1.0)


def test_custom_quantiles_define_deterministic_schema() -> None:
    dataset = _dataset()
    quantiles = (0.10, 0.25, 0.50, 0.75, 0.90)
    stats = summarize_numeric_statistics(dataset, profile_columns(dataset), quantiles=quantiles)
    assert tuple(stats.columns) == numeric_statistics_columns(quantiles)
    assert "q10" in stats and "q90" in stats


def test_categorical_entropy_frequency_order_and_factor_semantics() -> None:
    dataset = _dataset()
    summary, frequencies = summarize_categorical_statistics(dataset, profile_columns(dataset))
    row = summary.set_index("column").loc["source"]
    probabilities = np.asarray([3 / 7, 2 / 7, 2 / 7])
    expected_entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    assert row["mode"] == "A"
    assert row["mode_count"] == 3
    assert row["entropy"] == pytest.approx(expected_entropy)
    assert row["normalized_entropy"] == pytest.approx(expected_entropy / math.log2(3))
    observed = frequencies.loc[frequencies["column"] == "source"]
    assert observed[["level", "count", "rank"]].to_dict("records") == [
        {"level": "A", "count": 3, "rank": 1},
        {"level": "B", "count": 2, "rank": 2},
        {"level": "C", "count": 2, "rank": 3},
    ]
    assert "factor_code" in set(summary["column"])
    assert "factor_code" not in set(summarize_numeric_statistics(dataset, profile_columns(dataset))["column"])


def test_boolean_constant_all_missing_and_high_cardinality_are_explicit() -> None:
    dataset = _dataset()
    summary, frequencies = summarize_categorical_statistics(
        dataset, profile_columns(dataset), max_category_levels=2
    )
    table = summary.set_index("column")
    assert set(frequencies.loc[frequencies["column"] == "flag", "level"]) == {"false", "true"}
    assert table.loc["constant_category", "status"] == "degenerate"
    assert table.loc["constant_category", "reason"] == "constant"
    assert table.loc["all_missing_category", "status"] == "skipped"
    assert table.loc["all_missing_category", "reason"] == "all_missing"


def test_datetime_summary_is_descriptive_only() -> None:
    dataset = _dataset()
    table = summarize_datetime_statistics(dataset, profile_columns(dataset)).set_index("column")
    row = table.loc["when"]
    assert row["n_total"] == 8
    assert row["n_missing"] == 1
    assert row["min"] == pd.Timestamp("2026-01-01")
    assert row["max"] == pd.Timestamp("2026-01-08")
    assert row["range_seconds"] == pytest.approx(7 * 24 * 3600)
    assert row["status"] == "ok"


def test_analyze_univariate_is_non_destructive_and_complete() -> None:
    dataset = _dataset()
    before = dataset.to_frame()
    result = analyze_univariate(dataset)
    pd.testing.assert_frame_equal(dataset.to_frame(), before)
    assert result.provenance.analysis == "univariate"
    assert result.provenance.parameters["numeric_finite_policy"] == "finite_values_only"
    assert not result.numeric_statistics.empty
    assert not result.categorical_statistics.empty
    assert not result.datetime_statistics.empty
