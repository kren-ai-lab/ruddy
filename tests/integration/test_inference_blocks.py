from __future__ import annotations

import numpy as np
import pandas as pd

from ruddy import AnalysisBlock, AnalysisConfig, TabularDataset, analyze


def _dataset() -> TabularDataset:
    rng = np.random.default_rng(5)
    n = 40
    g = np.repeat(["A", "B"], 20)
    return TabularDataset(
        pd.DataFrame(
            {
                "id": range(n),
                "x": rng.normal(size=n),
                "y": rng.normal(size=n),
                "z": rng.normal(size=n),
                "group": g,
                "cat": np.where(np.arange(n) % 3 == 0, "X", "Y"),
            }
        ),
        id_column="id",
        role_overrides={"group": "factor", "z": "covariate"},
    )


def test_unified_runs_new_blocks_only_when_enabled() -> None:
    cfg = AnalysisConfig(
        enabled_blocks=("diagnostics", "dependence", "contingency", "intervals"),
        responses=("x",),
        groups=("group",),
        dependence_partial_covariates=("z",),
        dependence_n_permutations=9,
        bootstrap_resamples=100,
        bootstrap_method="percentile",
    )
    result = analyze(_dataset(), config=cfg)
    assert result.executed_blocks == (
        AnalysisBlock.DIAGNOSTICS,
        AnalysisBlock.DEPENDENCE,
        AnalysisBlock.CONTINGENCY,
        AnalysisBlock.INTERVALS,
    )
    assert result.diagnostics is not None
    assert result.dependence is not None
    assert result.contingency is not None
    assert result.intervals is not None
    assert result.profiling is None
    assert set(result.components) == set(result.executed_blocks)


def test_unified_default_is_unchanged() -> None:
    result = analyze(_dataset())
    assert result.executed_blocks == (AnalysisBlock.PROFILING, AnalysisBlock.UNIVARIATE)
    assert result.dependence is None
    assert result.intervals is None
