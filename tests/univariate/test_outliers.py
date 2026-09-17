from __future__ import annotations

import pandas as pd
import pytest

from ruddy import ColumnRole, TabularDataset, analyze_outliers, summarize_outliers


def _dataset() -> TabularDataset:
    frame = pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(1, 9)],
            "x": [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 100.0],
            "y": [10.0, 11.0, 9.0, 10.0, 10.0, 11.0, 9.0, 10.0],
            "response": [0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 9.0],
            "factor_code": [0, 0, 0, 0, 1, 1, 1, 1],
            "constant": [5.0] * 8,
            "label": ["a", "a", "a", "a", "b", "b", "b", "b"],
        }
    )
    return TabularDataset(
        frame,
        id_column="id",
        role_overrides={
            "response": ColumnRole.RESPONSE,
            "factor_code": ColumnRole.FACTOR,
        },
    )


def _summary(result, column: str, method: str) -> pd.Series:
    rows = result.summaries.loc[result.summaries["column"].eq(column) & result.summaries["method"].eq(method)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_iqr_flags_known_extreme_with_auditable_fences() -> None:
    result = analyze_outliers(_dataset(), methods=("iqr",), include_flags=True)
    row = _summary(result, "x", "iqr")
    assert row["status"] == "ok"
    assert row["center"] == pytest.approx(3.0)
    assert row["scale"] == pytest.approx(2.0)
    assert row["lower_bound"] == pytest.approx(-1.0)
    assert row["upper_bound"] == pytest.approx(7.0)
    assert row["n_flagged"] == 1
    assert row["flagged_fraction"] == pytest.approx(1 / 8)

    flags = result.flags.loc[(result.flags["column"] == "x") & (result.flags["method"] == "iqr")]
    assert len(flags) == 1
    flag = flags.iloc[0]
    assert flag["observation_id"] == "r8"
    assert flag["source_row_index"] == 7
    assert flag["value"] == pytest.approx(100.0)
    assert flag["direction"] == "high"
    assert flag["score"] == pytest.approx(93.0)


def test_robust_z_uses_modified_z_and_reports_raw_bounds() -> None:
    result = analyze_outliers(_dataset(), methods=("robust_z",), include_flags=True)
    row = _summary(result, "x", "robust_z")
    assert row["center"] == pytest.approx(3.0)
    assert row["scale"] == pytest.approx(1.0)
    assert row["threshold"] == pytest.approx(3.5)
    assert row["n_flagged"] == 1
    expected = 0.6744897501960817 * (100.0 - 3.0)
    flag = result.flags.loc[(result.flags["column"] == "x") & (result.flags["method"] == "robust_z")].iloc[0]
    assert flag["score"] == pytest.approx(expected)
    assert flag["direction"] == "high"


def test_flags_are_opt_in_but_counts_are_always_available() -> None:
    result = analyze_outliers(_dataset(), include_flags=False)
    assert result.flags.empty
    assert _summary(result, "x", "iqr")["n_flagged"] == 1
    assert _summary(result, "x", "robust_z")["n_flagged"] == 1


def test_numeric_response_is_eligible_but_numeric_factor_is_not() -> None:
    result = analyze_outliers(_dataset(), methods=("iqr",))
    observed = set(result.summaries["column"])
    assert "response" in observed
    assert "factor_code" not in observed
    assert "label" not in observed


def test_custom_thresholds_are_applied() -> None:
    result = analyze_outliers(
        _dataset(),
        iqr_multiplier=3.0,
        robust_z_threshold=5.0,
    )
    assert _summary(result, "x", "iqr")["threshold"] == pytest.approx(3.0)
    assert _summary(result, "x", "robust_z")["threshold"] == pytest.approx(5.0)


def test_method_order_and_output_are_deterministic() -> None:
    dataset = _dataset()
    left = analyze_outliers(dataset, methods=("robust_z", "iqr"), include_flags=True)
    right = analyze_outliers(dataset, methods=("robust_z", "iqr"), include_flags=True)
    pd.testing.assert_frame_equal(left.summaries, right.summaries, check_exact=True)
    pd.testing.assert_frame_equal(left.flags, right.flags, check_exact=True)
    first = list(zip(left.summaries["column"], left.summaries["method"], strict=True))[:4]
    assert first == [("x", "robust_z"), ("x", "iqr"), ("y", "robust_z"), ("y", "iqr")]


def test_analysis_never_mutates_or_removes_source_records() -> None:
    dataset = _dataset()
    before = dataset.to_frame()
    result = analyze_outliers(dataset, include_flags=True)
    pd.testing.assert_frame_equal(dataset.to_frame(), before)
    assert result.provenance.parameters["record_removal"] is False
    assert result.provenance.parameters["value_modification"] is False
    assert result.provenance.parameters["automatic_cleaning"] is False
    assert result.provenance.parameters["multivariate_outliers"] is False


def test_summarize_outliers_requires_explicit_nonempty_unique_methods() -> None:
    dataset = _dataset()
    with pytest.raises(ValueError, match="At least one"):
        summarize_outliers(dataset, methods=())
    with pytest.raises(ValueError, match="duplicates"):
        summarize_outliers(dataset, methods=("iqr", "iqr"))
