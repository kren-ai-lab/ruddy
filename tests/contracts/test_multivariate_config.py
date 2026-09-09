from __future__ import annotations

import pytest

from ruddy import AnalysisConfig, ScalingMethod


def test_multivariate_config_kwargs_are_explicit():
    config = AnalysisConfig(
        multivariate_scaling="standard",
        multivariate_include_spearman=False,
        mahalanobis_threshold_quantile=0.99,
        mahalanobis_include_robust=False,
        multivariate_max_covariance_features=80,
        multivariate_max_collinearity_features=40,
        multivariate_max_mahalanobis_features=30,
        random_state=17,
    )
    kwargs = config.multivariate_kwargs()
    assert kwargs["scaling"] is ScalingMethod.STANDARD
    assert kwargs["include_spearman"] is False
    assert kwargs["mahalanobis_threshold_quantile"] == 0.99
    assert kwargs["include_robust_mahalanobis"] is False
    assert kwargs["random_state"] == 17


def test_manova_config_guards():
    config = AnalysisConfig(manova_max_responses=12, manova_max_factor_levels=8, manova_min_level_n=4)
    assert config.manova_kwargs() == {"max_responses": 12, "max_factor_levels": 8, "min_level_n": 4}
    with pytest.raises(ValueError):
        AnalysisConfig(mahalanobis_threshold_quantile=1.0)
