from __future__ import annotations

import pytest

from ruddy import AnalysisConfig, ColumnKind, ColumnRole


def test_analysis_config_normalizes_and_freezes_overrides() -> None:
    config = AnalysisConfig(
        id_column="id",
        role_overrides={"group": "factor"},
        kind_overrides={"code": "categorical"},
        annotation_alignment="partial",
        random_state=11,
    )

    assert config.role_overrides["group"] is ColumnRole.FACTOR
    assert config.kind_overrides["code"] is ColumnKind.CATEGORICAL
    assert config.annotation_alignment.value == "partial"
    assert config.dataset_kwargs()["id_column"] == "id"

    with pytest.raises(TypeError):
        config.role_overrides["x"] = ColumnRole.TARGET  # type: ignore[index]


def test_phase2_config_validates_descriptive_controls() -> None:
    config = AnalysisConfig(
        quantiles=(0.10, 0.25, 0.50, 0.75, 0.90),
        min_numeric_n=4,
        max_category_levels=10,
        max_missingness_patterns=5,
        max_pairwise_columns=50,
    )
    assert config.univariate_kwargs()["min_numeric_n"] == 4
    assert config.profiling_kwargs()["max_pairwise_columns"] == 50

    with pytest.raises(ValueError, match="include 0.25"):
        AnalysisConfig(quantiles=(0.1, 0.5, 0.9))
