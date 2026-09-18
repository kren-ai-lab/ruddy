from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import sparse, stats
from sklearn.preprocessing import StandardScaler

from ruddy import FeatureMatrix, ResultStatus, analyze_covariance_structure
from ruddy.multivariate import square_table


def test_covariance_and_pearson_match_numpy(well_conditioned_features):
    result = analyze_covariance_structure(well_conditioned_features)
    array = well_conditioned_features.to_array()
    np.testing.assert_allclose(
        result.covariance.drop("feature").to_numpy(),
        np.cov(array, rowvar=False, ddof=1),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.pearson.drop("feature").to_numpy(),
        np.corrcoef(array, rowvar=False),
        rtol=1e-12,
        atol=1e-12,
    )
    assert result.status is ResultStatus.OK
    assert result.summary["matrix_sample_policy"] == "common_complete_case"


def test_spearman_matches_scipy(well_conditioned_features):
    result = analyze_covariance_structure(well_conditioned_features)
    reference = stats.spearmanr(well_conditioned_features.to_array(), axis=0).statistic
    np.testing.assert_allclose(result.spearman.drop("feature").to_numpy(), reference, rtol=1e-12, atol=1e-12)


def test_spearman_two_features_has_square_shape():
    values = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
    result = analyze_covariance_structure(FeatureMatrix(values, feature_names=("x", "y")))
    assert result.spearman.shape == (2, 3)
    assert result.spearman.columns == ["feature", "x", "y"]
    assert result.spearman.filter(pl.col("feature") == "x")["y"][0] == pytest.approx(-1.0)


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
    assert result.pairwise_counts.filter(pl.col("feature") == "a")["b"][0] == 3
    assert result.pairwise_counts.filter(pl.col("feature") == "a")["c"][0] == 3
    assert result.pairwise_counts.filter(pl.col("feature") == "b")["c"][0] == 2
    assert result.exclusions.height == 2


def test_explicit_standard_scaling_matches_reference(well_conditioned_features):
    array = well_conditioned_features.to_array()
    scaled = StandardScaler().fit_transform(array)
    result = analyze_covariance_structure(well_conditioned_features, scaling="standard")
    np.testing.assert_allclose(
        result.covariance.drop("feature").to_numpy(),
        np.cov(scaled, rowvar=False, ddof=1),
        rtol=1e-12,
        atol=1e-12,
    )
    assert result.summary["scaling"] == "standard"


def test_constant_features_are_reported_not_hidden():
    values = np.column_stack([np.arange(8.0), np.ones(8), np.arange(8.0) ** 2])
    result = analyze_covariance_structure(FeatureMatrix(values, feature_names=("x", "constant", "z")))
    row = result.feature_diagnostics.filter(pl.col("feature") == "constant").row(0, named=True)
    assert row["status"] == "degenerate"
    assert row["reason"] == "constant_feature"
    assert np.isnan(result.pearson.filter(pl.col("feature") == "constant")["x"][0])


def test_covariance_rejects_high_dimensional_request_without_reduction():
    matrix = FeatureMatrix(np.ones((10, 5)), feature_names=[f"f{i}" for i in range(5)])
    with pytest.raises(ValueError, match="reduce dimensionality explicitly"):
        analyze_covariance_structure(matrix, max_features=4)


def test_covariance_never_silently_densifies_sparse():
    matrix = FeatureMatrix(sparse.csr_matrix(np.eye(5)))
    with pytest.raises(ValueError, match="silently densify"):
        analyze_covariance_structure(matrix)


def test_square_table_roundtrip():
    cov_values = np.array([[1.0, 0.2, 0.3], [0.2, 2.0, 0.4], [0.3, 0.4, 3.0]])
    table = square_table(cov_values, ("a", "b", "c"), label_column="feature")
    assert table.columns == ["feature", "a", "b", "c"]
    assert table.schema["feature"] == pl.String
    assert table.filter(pl.col("feature") == "b").row(0)[2] == 2.0

    dist_values = np.zeros((3, 3))
    dist_table = square_table(dist_values, [0, 1, 2], label_column="observation_id")
    assert dist_table.schema["observation_id"] == pl.Int64
    assert dist_table.columns == ["observation_id", "0", "1", "2"]


def test_spearman_via_ranks_matches_scipy_on_random_matrix():
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(20, 4))
    res = analyze_covariance_structure(FeatureMatrix(matrix, feature_names=("a", "b", "c", "d")))
    expected = stats.spearmanr(matrix, axis=0).statistic
    np.testing.assert_allclose(res.spearman.drop("feature").to_numpy(), expected, rtol=1e-12, atol=1e-12)
