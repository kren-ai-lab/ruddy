from __future__ import annotations

import numpy as np
import pytest

from ruddy import FeatureMatrix


@pytest.fixture
def well_conditioned_features() -> FeatureMatrix:
    rng = np.random.default_rng(20260908)
    values = rng.normal(size=(90, 4))
    values[:, 3] = 0.35 * values[:, 0] - 0.20 * values[:, 1] + rng.normal(scale=0.8, size=90)
    return FeatureMatrix(
        values,
        observation_ids=[f"obs_{index}" for index in range(90)],
        feature_names=("f1", "f2", "f3", "f4"),
    )
