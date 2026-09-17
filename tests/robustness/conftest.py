from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ruddy import TabularDataset


@pytest.fixture
def robust_tabular() -> TabularDataset:
    rng = np.random.default_rng(1701)
    n = 96
    group = np.repeat(["A", "B", "C"], n // 3)
    batch = np.tile(np.repeat(["X", "Y"], n // 6), 3)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 0.8 * x + np.select([group == "B", group == "C"], [0.7, 1.4], default=0.0) + rng.normal(0, 0.45, n)
    y2 = (
        -0.4 * x
        + 0.6 * z
        + np.select([group == "B", group == "C"], [0.4, 0.9], default=0.0)
        + rng.normal(0, 0.5, n)
    )
    cat = np.where(y > np.median(y), "high", "low")
    frame = pd.DataFrame(
        {
            "id": [f"o{i}" for i in range(n)],
            "x": x,
            "z": z,
            "y": y,
            "y2": y2,
            "group": group,
            "batch": batch,
            "class": cat,
        }
    )
    return TabularDataset(
        frame,
        id_column="id",
        role_overrides={
            "y": "response",
            "y2": "response",
            "group": "factor",
            "batch": "factor",
            "class": "factor",
            "x": "covariate",
            "z": "covariate",
        },
    )
