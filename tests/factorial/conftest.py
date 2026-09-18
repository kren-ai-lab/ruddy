from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ruddy import TabularDataset


@pytest.fixture
def balanced_factorial_dataset() -> TabularDataset:
    rng = np.random.default_rng(20260909)
    rows = []
    index = 0
    for factor_a in ("A0", "A1"):
        for factor_b in ("B0", "B1", "B2"):
            for _ in range(18):
                cov = rng.normal()
                main_a = 0.9 if factor_a == "A1" else 0.0
                main_b = {"B0": 0.0, "B1": 0.6, "B2": 1.2}[factor_b]
                interaction = 0.8 if (factor_a, factor_b) == ("A1", "B2") else 0.0
                response = 2.0 + main_a + main_b + interaction + 0.45 * cov + rng.normal(scale=0.65)
                rows.append((f"obs_{index}", response, factor_a, factor_b, cov))
                index += 1
    frame = pd.DataFrame(rows, columns=pd.Index(("id", "response", "factor_a", "factor_b", "covariate")))
    return TabularDataset(
        frame,
        id_column="id",
        role_overrides={
            "response": "response",
            "factor_a": "factor",
            "factor_b": "factor",
            "covariate": "covariate",
        },
    )


@pytest.fixture
def unbalanced_factorial_dataset(balanced_factorial_dataset: TabularDataset) -> TabularDataset:
    frame = balanced_factorial_dataset.frame.to_pandas()
    drop = frame.index[(frame["factor_a"] == "A0") & (frame["factor_b"] == "B0")][:10]
    frame = frame.drop(drop).reset_index(drop=True)
    return TabularDataset(
        frame,
        id_column="id",
        role_overrides={
            "response": "response",
            "factor_a": "factor",
            "factor_b": "factor",
            "covariate": "covariate",
        },
    )
