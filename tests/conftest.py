from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ruddy import ColumnRole, FeatureMatrix, TabularDataset


@pytest.fixture
def group_dataset():
    rng = np.random.default_rng(42)
    groups = np.repeat(np.array(["A", "B", "C"], dtype=object), 24)
    n = len(groups)
    x = rng.normal(size=n)
    y = 1.5 * (groups == "B") + 3.0 * (groups == "C") + 0.25 * x + rng.normal(scale=0.7, size=n)
    source = np.tile(np.repeat(["S1", "S2"], 12), 3)
    frame = pd.DataFrame({"id": np.arange(n), "group": groups, "source": source, "x": x, "y": y})
    return TabularDataset(
        frame,
        id_column="id",
        role_overrides={"group": "factor", "source": "factor", "x": "covariate", "y": "response"},
    )


@pytest.fixture
def separated_features(group_dataset):
    rng = np.random.default_rng(123)
    labels = group_dataset.select(["group"])["group"].to_numpy()
    matrix = rng.normal(scale=0.4, size=(len(labels), 5))
    matrix[labels == "B", 0] += 2.0
    matrix[labels == "C", 0] += 4.0
    return FeatureMatrix(matrix, observation_ids=group_dataset.observation_ids)


@pytest.fixture
def random_intercept_dataset():
    rng = np.random.default_rng(7)
    n_groups = 18
    per_group = 14
    n = n_groups * per_group
    batch = np.repeat(np.array([f"b{i}" for i in range(n_groups)], dtype=object), per_group)
    condition = np.tile(np.array(["A", "B"] * (per_group // 2), dtype=object), n_groups)
    x = rng.normal(size=n)
    random_intercept = rng.normal(scale=1.4, size=n_groups)
    y = (
        1.8 * (condition == "B")
        + 0.6 * x
        + np.repeat(random_intercept, per_group)
        + rng.normal(scale=0.5, size=n)
    )
    frame = pd.DataFrame({"id": np.arange(n), "condition": condition, "batch": batch, "x": x, "y": y})
    return TabularDataset(
        frame,
        id_column="id",
        role_overrides={"condition": "factor", "batch": "factor", "x": "covariate", "y": "response"},
    )


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
