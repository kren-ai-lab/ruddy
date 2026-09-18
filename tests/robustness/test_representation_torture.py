from __future__ import annotations

import numpy as np
import pytest

from ruddy import FeatureMatrix, analyze_representation_similarity, linear_cka


def test_identical_representation_all_similarity_metrics_peak():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(60, 6))
    f = FeatureMatrix(x, observation_ids=range(60))
    result = analyze_representation_similarity(f, f, cca_components=3, mantel_permutations=49, random_state=3)
    assert result.cka.item(0, "cka") == pytest.approx(1.0, abs=1e-12)
    assert result.procrustes.item(0, "disparity") == pytest.approx(0.0, abs=1e-12)
    assert result.distance_similarity.item(0, "coefficient") == pytest.approx(1.0, abs=1e-12)
    assert result.mantel.item(0, "correlation") == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(result.cca.correlations["canonical_correlation"].to_numpy(), 1.0, atol=1e-10)


def test_cka_is_invariant_to_positive_global_scaling():
    rng = np.random.default_rng(22)
    x = rng.normal(size=(50, 7))
    assert linear_cka(x, 17.3 * x) == pytest.approx(1.0, abs=1e-12)


def test_orthogonal_rotation_preserves_distance_geometry():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(70, 5))
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    ids = range(70)
    a = FeatureMatrix(x, observation_ids=ids)
    b = FeatureMatrix(x @ q, observation_ids=ids)
    result = analyze_representation_similarity(a, b, cca_components=3, mantel_permutations=29, random_state=8)
    assert result.cka.item(0, "cka") == pytest.approx(1.0, abs=1e-12)
    assert result.distance_similarity.item(0, "coefficient") == pytest.approx(1.0, abs=1e-12)
    assert result.procrustes.item(0, "disparity") < 1e-12


def test_feature_permutation_preserves_linear_cka():
    rng = np.random.default_rng(24)
    x = rng.normal(size=(50, 9))
    y = x[:, rng.permutation(x.shape[1])]
    assert linear_cka(x, y) == pytest.approx(1.0, abs=1e-12)


def test_independent_spaces_do_not_look_identical():
    rng = np.random.default_rng(25)
    x = rng.normal(size=(120, 10))
    y = rng.normal(size=(120, 8))
    result = analyze_representation_similarity(
        FeatureMatrix(x, observation_ids=range(120)),
        FeatureMatrix(y, observation_ids=range(120)),
        cca_components=2,
        mantel_permutations=49,
        random_state=7,
    )
    assert result.cka.item(0, "cka") < 0.3
    assert abs(float(result.distance_similarity.item(0, "coefficient"))) < 0.3


def test_zero_variance_cka_is_rejected():
    x = np.ones((20, 3))
    with pytest.raises(ValueError, match="zero-variance"):
        linear_cka(x, x)
