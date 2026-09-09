import numpy as np
import pandas as pd
import pytest

from ruddy import FeatureMatrix, TabularDataset


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
def mixed_dataset():
    rng = np.random.default_rng(7)
    n_groups = 18
    per_group = 14
    n = n_groups * per_group
    batch = np.repeat(np.array([f"b{i}" for i in range(n_groups)], dtype=object), per_group)
    condition = np.tile(np.array(["A", "B"] * (per_group // 2), dtype=object), n_groups)
    x = rng.normal(size=n)
    random_intercept = rng.normal(scale=1.4, size=n_groups)
    y = 1.8 * (condition == "B") + 0.6 * x + np.repeat(random_intercept, per_group) + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame({"id": np.arange(n), "condition": condition, "batch": batch, "x": x, "y": y})
    return TabularDataset(
        frame,
        id_column="id",
        role_overrides={"condition": "factor", "batch": "factor", "x": "covariate", "y": "response"},
    )
