from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ruddy import FeatureMatrix, TabularDataset


@pytest.fixture
def pipeline_frame() -> pd.DataFrame:
    rng = np.random.default_rng(14)
    n = 48
    factor = np.repeat(["A", "B"], n // 2)
    batch = np.tile(np.repeat(["X", "Y"], n // 4), 2)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y1 = 1.2 * x + (factor == "B") * 1.5 + rng.normal(scale=0.45, size=n)
    y2 = -0.7 * x + (batch == "Y") * 1.0 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({
        "id": [f"obs_{i:03d}" for i in range(n)],
        "x": x,
        "z": z,
        "y1": y1,
        "y2": y2,
        "factor": factor,
        "batch": batch,
        "label": np.where(x > 0, "high", "low"),
        "sequence": ["AAAA"] * n,
    })


@pytest.fixture
def pipeline_dataset(pipeline_frame: pd.DataFrame) -> TabularDataset:
    return TabularDataset(
        pipeline_frame,
        id_column="id",
        role_overrides={
            "y1": "response",
            "y2": "response",
            "factor": "factor",
            "batch": "factor",
            "x": "covariate",
            "sequence": "excluded",
        },
    )


@pytest.fixture
def pipeline_features(pipeline_frame: pd.DataFrame) -> FeatureMatrix:
    matrix = np.column_stack([
        pipeline_frame["x"].to_numpy(),
        pipeline_frame["z"].to_numpy(),
        pipeline_frame["y1"].to_numpy(),
        pipeline_frame["y2"].to_numpy(),
    ])
    return FeatureMatrix(
        matrix,
        observation_ids=pipeline_frame["id"].tolist(),
        feature_names=("f1", "f2", "f3", "f4"),
    )
