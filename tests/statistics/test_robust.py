from __future__ import annotations

import numpy as np
import pytest

from ruddy.statistics import (
    MODIFIED_Z_CONSISTENCY,
    median_absolute_deviation,
    modified_z_scores,
    tukey_fences,
)


def test_robust_primitives_match_known_definitions() -> None:
    values = np.asarray([1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 100.0])
    assert median_absolute_deviation(values) == pytest.approx(1.0)
    scores, median, mad = modified_z_scores(values)
    assert median == pytest.approx(3.0)
    assert mad == pytest.approx(1.0)
    assert scores[-1] == pytest.approx(MODIFIED_Z_CONSISTENCY * 97.0)
    q25, q75, lower, upper = tukey_fences(values)
    assert q25 == pytest.approx(2.0)
    assert q75 == pytest.approx(4.0)
    assert lower == pytest.approx(-1.0)
    assert upper == pytest.approx(7.0)


def test_modified_z_rejects_zero_mad() -> None:
    with pytest.raises(ValueError, match="MAD is zero"):
        modified_z_scores(np.asarray([0.0, 0.0, 0.0, 1.0]))
