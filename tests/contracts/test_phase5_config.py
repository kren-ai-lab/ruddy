from __future__ import annotations

import pytest

from ruddy import AnalysisConfig, OutlierMethod


def test_phase5_config_normalizes_outlier_controls() -> None:
    config = AnalysisConfig(
        outlier_methods=("robust_z", "iqr"),
        outlier_iqr_multiplier=2.0,
        outlier_robust_z_threshold=4.0,
        include_outlier_flags=True,
    )
    assert config.outlier_methods == (OutlierMethod.ROBUST_Z, OutlierMethod.IQR)
    assert config.outlier_kwargs() == {
        "methods": (OutlierMethod.ROBUST_Z, OutlierMethod.IQR),
        "min_numeric_n": 3,
        "iqr_multiplier": 2.0,
        "robust_z_threshold": 4.0,
        "include_flags": True,
    }


def test_phase5_config_rejects_invalid_outlier_controls() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        AnalysisConfig(outlier_methods=())
    with pytest.raises(ValueError, match="duplicates"):
        AnalysisConfig(outlier_methods=("iqr", "iqr"))
    with pytest.raises(ValueError, match="greater than zero"):
        AnalysisConfig(outlier_iqr_multiplier=0)
    with pytest.raises(ValueError, match="greater than zero"):
        AnalysisConfig(outlier_robust_z_threshold=-1)
