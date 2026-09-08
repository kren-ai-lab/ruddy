from __future__ import annotations

import pandas as pd
import pytest

from ruddy import ColumnRole, TabularDataset


@pytest.fixture
def mixed_response_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"obs_{i}" for i in range(1, 9)],
            "activity": [1.0, 1.3, 1.8, 2.1, 5.0, 5.3, 5.8, 6.1],
            "class_response": ["low", "low", "low", "high", "low", "high", "high", "high"],
            "family": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "category_group": ["x", "x", "y", "y", "x", "x", "y", None],
            "coded_factor": [0, 0, 0, 0, 1, 1, 1, 1],
            "continuous_group_candidate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
    )


@pytest.fixture
def response_dataset(mixed_response_frame: pd.DataFrame) -> TabularDataset:
    return TabularDataset(
        mixed_response_frame,
        id_column="id",
        role_overrides={
            "activity": ColumnRole.RESPONSE,
            "class_response": ColumnRole.RESPONSE,
            "family": ColumnRole.FACTOR,
            "coded_factor": ColumnRole.FACTOR,
        },
    )
