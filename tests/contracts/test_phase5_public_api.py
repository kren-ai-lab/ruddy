from __future__ import annotations

import ruddy


def test_phase5_api_is_public() -> None:
    expected = {
        "OutlierMethod",
        "OutlierResult",
        "analyze_outliers",
        "summarize_outliers",
        "summarize_numeric_quality",
    }
    assert expected.issubset(set(ruddy.__all__))
