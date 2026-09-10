from __future__ import annotations

import numpy as np
import pytest

from ruddy import FeatureMatrix, analyze_anomalies, analyze_composition


def test_compositional_scale_invariance_under_closure():
    rng = np.random.default_rng(51)
    x = rng.uniform(.1, 3, size=(30, 6))
    scales = rng.uniform(.2, 10, size=(30, 1))
    a = analyze_composition(FeatureMatrix(x, observation_ids=range(30)), transform="clr")
    b = analyze_composition(FeatureMatrix(x*scales, observation_ids=range(30)), transform="clr")
    np.testing.assert_allclose(a.transformed.to_array(), b.transformed.to_array(), atol=1e-12)
    np.testing.assert_allclose(a.aitchison_distances, b.aitchison_distances, atol=1e-12)


def test_ilr_distance_geometry_matches_aitchison_for_many_pairs():
    rng = np.random.default_rng(52)
    x = rng.uniform(.05, 2, size=(20, 7))
    r = analyze_composition(FeatureMatrix(x, observation_ids=range(20)), transform="ilr")
    z = r.transformed.to_array()
    euclidean = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
    np.testing.assert_allclose(euclidean, r.aitchison_distances.to_numpy(), atol=1e-11)


def test_zero_replacement_preserves_positive_closed_compositions():
    x = np.array([[0., 2., 3., 5.], [1., 0., 4., 5.], [2., 3., 0., 5.]])
    r = analyze_composition(FeatureMatrix(x), transform="clr", replace_zeros=True)
    transformed_source = r.zero_replacement
    assert transformed_source["replaced"].all()
    assert np.isfinite(r.transformed.to_array()).all()


def test_all_zero_composition_is_rejected():
    x = np.array([[0., 0., 0.], [1., 2., 3.]])
    with pytest.raises(ValueError):
        analyze_composition(FeatureMatrix(x), replace_zeros=True)


def test_extreme_multivariate_anomaly_is_ranked_high_by_both_methods():
    rng = np.random.default_rng(53)
    x = rng.normal(size=(120, 4)); x[-1] = 12.0
    r = analyze_anomalies(FeatureMatrix(x, observation_ids=[f"o{i}" for i in range(120)]), contamination=.03, lof_neighbors=20, random_state=2)
    for method in ("isolation_forest", "lof"):
        subset = r.scores[r.scores.method == method].sort_values("anomaly_score", ascending=False)
        assert "o119" in set(subset.head(3).observation_id)


def test_nonfinite_anomaly_rows_are_excluded_from_all_methods_once():
    rng = np.random.default_rng(54)
    x = rng.normal(size=(50, 3)); x[4, 1] = np.inf
    r = analyze_anomalies(FeatureMatrix(x, observation_ids=[f"o{i}" for i in range(50)]))
    assert list(r.exclusions.observation_id) == ["o4"]
    assert "o4" not in set(r.scores.observation_id)
