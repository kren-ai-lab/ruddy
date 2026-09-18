from __future__ import annotations

import numpy as np
import pytest

from ruddy import FeatureMatrix, analyze_multivariate, analyze_pca, analyze_representation_similarity
from ruddy.multivariate import analyze_collinearity, analyze_mahalanobis


def _features(n: int, p: int, seed: int = 1) -> FeatureMatrix:
    rng = np.random.default_rng(seed)
    return FeatureMatrix(rng.normal(size=(n, p)), observation_ids=[f"o{i}" for i in range(n)])


def test_pca_n_much_greater_than_p_runs():
    result = analyze_pca(_features(120, 8), n_components=5, scaling="standard")
    assert result.status.value == "ok"
    assert result.scores.shape == (120, 7)  # source row + id + 5 PCs


def test_pca_p_greater_than_n_respects_effective_rank():
    f = _features(15, 60)
    ok = analyze_pca(f, n_components=10)
    assert ok.status.value == "ok"
    with pytest.raises(ValueError):
        analyze_pca(f, n_components=15)


def test_collinearity_p_ge_n_is_not_fabricated():
    result = analyze_collinearity(_features(12, 12), max_features=20)
    assert result.status.value in {"skipped", "degenerate"}
    assert result.reason is not None


def test_mahalanobis_p_ge_n_refuses_inversion():
    result = analyze_mahalanobis(_features(20, 20), include_robust=False, max_features=30)
    assert result.status.value in {"skipped", "degenerate"}
    assert result.reason is not None


def test_perfect_collinearity_is_detected_before_mahalanobis():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(50, 3))
    x = np.column_stack([x, 2.0 * x[:, 0]])
    result = analyze_mahalanobis(FeatureMatrix(x), include_robust=False)
    assert result.status.value == "degenerate"
    assert result.methods["reason"][0] == "singular_covariance"


def test_multivariate_high_dimensional_limits_fail_loudly():
    with pytest.raises(ValueError):
        analyze_multivariate(_features(80, 205), max_covariance_features=200)


def test_cca_high_dimensional_space_requires_rank_respecting_components():
    rng = np.random.default_rng(11)
    n = 18
    x = FeatureMatrix(rng.normal(size=(n, 40)), observation_ids=range(n))
    y = FeatureMatrix(rng.normal(size=(n, 35)), observation_ids=range(n))
    result = analyze_representation_similarity(x, y, cca_components=18, mantel_permutations=0)
    assert result.cca.status.value == "skipped"
    assert result.cca.reason == "cca_components_exceed_effective_rank"
