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
