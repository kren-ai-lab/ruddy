from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.multivariate.manova import MANOVA

from ruddy import ResultStatus, TabularDataset, analyze_manova


def _dataset(seed: int = 3) -> TabularDataset:
    rng = np.random.default_rng(seed)
    n = 72
    group = np.repeat(["A", "B", "C"], n // 3)
    cov = rng.normal(size=n)
    y1 = rng.normal(size=n) + (group == "B") * 0.8 + (group == "C") * 1.1 + 0.2 * cov
    y2 = rng.normal(size=n) + (group == "B") * 0.3 + (group == "C") * 0.7 - 0.1 * cov
    frame = pd.DataFrame({"id": [f"o{i}" for i in range(n)], "y1": y1, "y2": y2, "group": group, "cov": cov})
    return TabularDataset(
        frame,
        id_column="id",
        role_overrides={"y1": "response", "y2": "response", "group": "factor", "cov": "covariate"},
    )


def test_manova_matches_statsmodels_reference():
    dataset = _dataset()
    result = analyze_manova(dataset, responses=("y1", "y2"), factors=("group",), covariates=("cov",))
    frame = dataset.frame.to_pandas().rename(columns={"y1": "Y0", "y2": "Y1", "group": "F0", "cov": "X0"})
    reference = MANOVA.from_formula("Y0 + Y1 ~ C(F0) + X0", data=frame).mv_test().results["C(F0)"]["stat"]
    observed = {row["statistic"]: row for row in result.tests.iter_rows(named=True) if row["term"] == "group"}
    for statistic in reference.index:
        assert observed[statistic]["value"] == pytest.approx(
            float(reference.loc[statistic, "Value"]), rel=1e-10
        )
        assert observed[statistic]["p_value"] == pytest.approx(
            float(reference.loc[statistic, "Pr > F"]), rel=1e-10
        )
    assert result.status is ResultStatus.OK
    assert result.model_summary["formula_basis"] == "main_effects_only"


def test_numeric_factor_is_categorical_when_declared_factor():
    df = _dataset().frame.to_pandas()
    df["batch"] = np.tile([0, 1, 2], len(df) // 3)
    wrapped = TabularDataset(
        df, id_column="id", role_overrides={"y1": "response", "y2": "response", "batch": "factor"}
    )
    result = analyze_manova(wrapped, responses=("y1", "y2"), factors=("batch",))
    assert result.status is ResultStatus.OK
    assert "batch" in set(result.tests.get_column("term").to_list())


def test_complete_case_exclusions_are_observable():
    frame = _dataset().frame.to_pandas()
    frame.loc[0, "y1"] = np.nan
    frame.loc[1, "y2"] = np.inf
    frame.loc[2, "group"] = None
    dataset = TabularDataset(
        frame, id_column="id", role_overrides={"y1": "response", "y2": "response", "group": "factor"}
    )
    result = analyze_manova(dataset, responses=("y1", "y2"), factors=("group",))
    assert len(result.exclusions) == 3
    assert set(result.exclusions.get_column("observation_id").to_list()) == {"o0", "o1", "o2"}


def test_rank_deficient_response_matrix_is_degenerate():
    frame = _dataset().frame.to_pandas()
    frame["y2"] = 2.0 * frame["y1"]
    dataset = TabularDataset(
        frame, id_column="id", role_overrides={"y1": "response", "y2": "response", "group": "factor"}
    )
    result = analyze_manova(dataset, responses=("y1", "y2"), factors=("group",))
    assert result.status is ResultStatus.DEGENERATE
    assert result.reason == "rank_deficient_response_matrix"


def test_rank_deficient_design_is_degenerate():
    frame = _dataset().frame.to_pandas()
    frame["cov2"] = frame["cov"]
    dataset = TabularDataset(
        frame,
        id_column="id",
        role_overrides={
            "y1": "response",
            "y2": "response",
            "group": "factor",
            "cov": "covariate",
            "cov2": "covariate",
        },
    )
    result = analyze_manova(dataset, responses=("y1", "y2"), factors=("group",), covariates=("cov", "cov2"))
    assert result.status is ResultStatus.DEGENERATE
    assert result.reason == "rank_deficient_design"


def test_small_factor_level_blocks_manova():
    frame = _dataset().frame.to_pandas()
    frame.loc[:1, "group"] = "tiny"
    dataset = TabularDataset(
        frame, id_column="id", role_overrides={"y1": "response", "y2": "response", "group": "factor"}
    )
    result = analyze_manova(dataset, responses=("y1", "y2"), factors=("group",), min_level_n=3)
    assert result.status is ResultStatus.DEGENERATE
    assert result.reason == "factor_level_too_small"


def test_high_dimensional_manova_requires_explicit_reduction():
    frame = _dataset().frame.to_pandas()
    for index in range(3, 8):
        frame[f"y{index}"] = np.random.default_rng(index).normal(size=len(frame))
    dataset = TabularDataset(frame, id_column="id", role_overrides={"group": "factor"})
    with pytest.raises(ValueError, match="dimensionality reduction"):
        analyze_manova(
            dataset, responses=tuple(f"y{i}" for i in range(1, 8)), factors=("group",), max_responses=4
        )


def test_manova_requires_numeric_responses():
    frame = _dataset().frame.to_pandas()
    frame["label"] = np.where(frame["y1"] > 0, "high", "low")
    dataset = TabularDataset(
        frame, id_column="id", role_overrides={"label": "response", "y2": "response", "group": "factor"}
    )
    with pytest.raises(ValueError, match="must be numeric"):
        analyze_manova(dataset, responses=("label", "y2"), factors=("group",))
