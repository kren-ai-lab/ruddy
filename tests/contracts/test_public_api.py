from __future__ import annotations

import ruddy


def test_phase1_contracts_are_available_from_public_api() -> None:
    expected = {
        "TabularDataset",
        "FeatureMatrix",
        "AnalysisConfig",
        "AnalysisResult",
        "AnalysisProvenance",
        "Advisory",
        "ColumnRole",
        "ColumnKind",
        "AlignmentMode",
        "ResultStatus",
    }
    assert expected.issubset(set(ruddy.__all__))


def test_phase2_api_is_public() -> None:
    expected = {
        "ProfilingResult",
        "UnivariateResult",
        "profile_columns",
        "profile_dataset",
        "summarize_missingness",
        "summarize_missingness_patterns",
        "pairwise_completeness",
        "summarize_numeric_statistics",
        "summarize_categorical_statistics",
        "summarize_datetime_statistics",
        "summarize_univariate",
        "analyze_univariate",
    }
    assert expected.issubset(set(ruddy.__all__))
