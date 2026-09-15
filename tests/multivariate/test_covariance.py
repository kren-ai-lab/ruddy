from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse, stats
from sklearn.preprocessing import StandardScaler

from ruddy import FeatureMatrix, ResultStatus, analyze_covariance_structure


def test_covariance_and_pearson_match_numpy(well_conditioned_features):
    result = analyze_covariance_structure(well_conditioned_features)
    array = well_conditioned_features.to_array()
    np.testing.assert_allclose(
        result.covariance.to_numpy(), np.cov(array, rowvar=False, ddof=1), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        result.pearson.to_numpy(), np.corrcoef(array, rowvar=False), rtol=1e-12, atol=1e-12
    )
    assert result.status is ResultStatus.OK
    assert result.summary["matrix_sample_policy"] == "common_complete_case"


def test_spearman_matches_scipy(well_conditioned_features):
    result = analyze_covariance_structure(well_conditioned_features)
    reference = stats.spearmanr(well_conditioned_features.to_array(), axis=0).statistic
    np.testing.assert_allclose(result.spearman.to_numpy(), reference, rtol=1e-12, atol=1e-12)


def test_spearman_two_features_has_square_shape():
    values = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
    result = analyze_covariance_structure(FeatureMatrix(values, feature_names=("x", "y")))
    assert result.spearman.shape == (2, 2)
    assert result.spearman.loc["x", "y"] == pytest.approx(-1.0)


def test_pairwise_counts_use_source_while_matrices_use_complete_cases():
    values = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, np.nan, 4.0],
            [3.0, 5.0, np.inf],
            [4.0, 6.0, 7.0],
        ]
    )
    result = analyze_covariance_structure(FeatureMatrix(values, feature_names=("a", "b", "c")))
    assert result.summary["n_complete_case_observations"] == 2
    assert result.pairwise_counts.loc["a", "b"] == 3
    assert result.pairwise_counts.loc["a", "c"] == 3
    assert result.pairwise_counts.loc["b", "c"] == 2
    assert len(result.exclusions) == 2


def test_explicit_standard_scaling_matches_reference(well_conditioned_features):
    array = well_conditioned_features.to_array()
    scaled = StandardScaler().fit_transform(array)
    result = analyze_covariance_structure(well_conditioned_features, scaling="standard")
    np.testing.assert_allclose(
        result.covariance.to_numpy(), np.cov(scaled, rowvar=False, ddof=1), rtol=1e-12, atol=1e-12
    )
    assert result.summary["scaling"] == "standard"


def test_constant_features_are_reported_not_hidden():
    values = np.column_stack([np.arange(8.0), np.ones(8), np.arange(8.0) ** 2])
    result = analyze_covariance_structure(FeatureMatrix(values, feature_names=("x", "constant", "z")))
    row = result.feature_diagnostics.set_index("feature").loc["constant"]
    assert row["status"] == "degenerate"
    assert row["reason"] == "constant_feature"
    assert np.isnan(result.pearson.loc["constant", "x"])


def test_covariance_rejects_high_dimensional_request_without_reduction():
    matrix = FeatureMatrix(np.ones((10, 5)), feature_names=[f"f{i}" for i in range(5)])
    with pytest.raises(ValueError, match="reduce dimensionality explicitly"):
        analyze_covariance_structure(matrix, max_features=4)


def test_covariance_never_silently_densifies_sparse():
    matrix = FeatureMatrix(sparse.csr_matrix(np.eye(5)))
    with pytest.raises(ValueError, match="silently densify"):
        analyze_covariance_structure(matrix)
