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
        config.role_overrides["x"] = ColumnRole.RESPONSE  # type: ignore[index]


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


def test_phase4_config_keeps_response_and_group_selection_separate_from_roles() -> None:
    config = AnalysisConfig(
        responses=("response_a", "response_b"),
        groups=("group_a",),
    )
    assert config.responses == ("response_a", "response_b")
    assert config.groups == ("group_a",)
    assert config.grouped_kwargs()["responses"] == ("response_a", "response_b")
    assert config.grouped_kwargs()["groups"] == ("group_a",)

    with pytest.raises(ValueError, match="both response and group"):
        AnalysisConfig(responses=("x",), groups=("x",))
