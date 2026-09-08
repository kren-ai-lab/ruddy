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
