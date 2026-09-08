from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ruddy import ColumnRole, TabularDataset


@pytest.fixture
def mixed_dataset() -> TabularDataset:
    frame = pd.DataFrame(
        {
            "id": [f"s{i}" for i in range(1, 13)],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, np.inf, np.nan],
            "y": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0, 12.0, 11.0],
            "z": [10.0, 10.5, 9.5, 11.0, 20.0, 21.0, 19.5, 20.5, 30.0, 29.5, 31.0, 30.5],
            "binary": ["A"] * 6 + ["B"] * 6,
            "three": ["L"] * 4 + ["M"] * 4 + ["H"] * 4,
            "flag": [True, True, False, False, True, False, True, False, True, False, True, False],
            "numeric_factor": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
        }
    )
    return TabularDataset(
        frame,
        id_column="id",
        role_overrides={"numeric_factor": ColumnRole.FACTOR},
    )
