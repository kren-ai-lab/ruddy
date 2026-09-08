"""Lightweight statistical diagnostics used by inferential analyses."""

from __future__ import annotations

import numpy as np


def expected_count_diagnostics(expected: np.ndarray) -> dict[str, float | int | bool]:
    """Summarize conventional chi-square expected-count diagnostics."""

    values = np.asarray(expected, dtype=np.float64)
    if values.size == 0:
        return {
            "min_expected_count": float("nan"),
            "n_expected_lt5": 0,
            "fraction_expected_lt5": 0.0,
            "low_expected_counts": False,
        }
    under_five = values < 5.0
    return {
        "min_expected_count": float(np.min(values)),
        "n_expected_lt5": int(under_five.sum()),
        "fraction_expected_lt5": float(under_five.mean()),
        "low_expected_counts": bool(under_five.any()),
    }
