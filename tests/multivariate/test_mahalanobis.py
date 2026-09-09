from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse, stats
from sklearn.covariance import EmpiricalCovariance, MinCovDet

from ruddy import FeatureMatrix, ResultStatus, analyze_mahalanobis


def test_classical_mahalanobis_matches_sklearn(well_conditioned_features):
    array = well_conditioned_features.to_array()
    reference = EmpiricalCovariance().fit(array).mahalanobis(array)
    result = analyze_mahalanobis(well_conditioned_features, include_robust=False)
    observed = result.distances.query("method == 'classical'")["squared_distance"].to_numpy()
    np.testing.assert_allclose(observed, reference, rtol=1e-11, atol=1e-12)
    assert result.status is ResultStatus.OK


def test_robust_mahalanobis_matches_mincovdet_under_fixed_seed(well_conditioned_features):
    array = well_conditioned_features.to_array()
    reference = MinCovDet(random_state=7).fit(array).mahalanobis(array)
    result = analyze_mahalanobis(well_conditioned_features, random_state=7)
    observed = result.distances.query("method == 'robust'")["squared_distance"].to_numpy()
    np.testing.assert_allclose(observed, reference, rtol=1e-10, atol=1e-10)


def test_threshold_policy_is_recorded_and_flags_reference_chi_square():
    rng = np.random.default_rng(10)
    values = rng.normal(size=(60, 2))
    values[-1] = [12.0, 12.0]
    result = analyze_mahalanobis(FeatureMatrix(values, feature_names=("x", "y")), include_robust=False, threshold_quantile=0.99)
    rows = result.distances.query("method == 'classical'")
    expected = stats.chi2.ppf(0.99, df=2)
    assert rows["threshold_squared"].iloc[0] == pytest.approx(expected)
    assert bool(rows.iloc[-1]["is_flagged"])
    assert result.provenance.parameters["degrees_of_freedom_policy"] == "n_features"


def test_singular_covariance_returns_degenerate_not_pseudoinverse_result():
    x = np.arange(1.0, 20.0)
    matrix = FeatureMatrix(np.column_stack([x, 2 * x]), feature_names=("x", "two_x"))
    result = analyze_mahalanobis(matrix)
    assert result.status is ResultStatus.DEGENERATE
    assert result.reason == "mahalanobis_not_feasible"
    assert result.distances.empty
    assert set(result.methods["reason"]) <= {"singular_covariance"}


def test_p_greater_than_n_is_explicitly_not_feasible():
    rng = np.random.default_rng(11)
    matrix = FeatureMatrix(rng.normal(size=(5, 8)))
    result = analyze_mahalanobis(matrix, max_features=10)
    assert result.status is ResultStatus.DEGENERATE
    assert result.distances.empty
    assert "insufficient_observations" in " ".join(result.methods["reason"].astype(str))


def test_mahalanobis_never_silently_densifies_sparse():
    with pytest.raises(ValueError, match="silently densify"):
        analyze_mahalanobis(FeatureMatrix(sparse.eye(8)))
