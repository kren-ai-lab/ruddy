import numpy as np
import pytest
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ruddy import FeatureMatrix, ResultStatus, analyze_pca


def _matrix(seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(24, 6))


def test_pca_matches_sklearn_reference():
    x = _matrix()
    result = analyze_pca(FeatureMatrix(x), n_components=4, random_state=7)
    reference = PCA(n_components=4, random_state=7).fit(x)
    np.testing.assert_allclose(
        result.scores.select(["PC1", "PC2", "PC3", "PC4"]).to_numpy(),
        reference.transform(x),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.loadings.select(["PC1", "PC2", "PC3", "PC4"]).to_numpy(),
        reference.components_.T,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.variance["explained_variance_ratio"].to_numpy(),
        reference.explained_variance_ratio_,
        rtol=1e-12,
        atol=1e-12,
    )


def test_pca_standard_scaling_only_when_requested():
    x = _matrix()
    raw = analyze_pca(FeatureMatrix(x), n_components=2, scaling="none")
    scaled = analyze_pca(FeatureMatrix(x), n_components=2, scaling="standard")
    expected = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(x))
    assert not np.allclose(
        raw.scores.select(["PC1", "PC2"]).to_numpy(),
        scaled.scores.select(["PC1", "PC2"]).to_numpy(),
    )
    np.testing.assert_allclose(
        scaled.scores.select(["PC1", "PC2"]).to_numpy(),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_full_rank_explained_variance_ratios_sum_to_one():
    x = _matrix()
    result = analyze_pca(FeatureMatrix(x), n_components=6)
    assert np.isclose(float(result.variance["explained_variance_ratio"].sum()), 1.0)
    assert np.isclose(float(result.variance["cumulative_explained_variance_ratio"][-1]), 1.0)


def test_pca_exclusions_preserve_row_mapping():
    x = _matrix()[:8]
    x[2, 1] = np.nan
    result = analyze_pca(FeatureMatrix(x, observation_ids=[f"r{i}" for i in range(8)]), n_components=2)
    assert result.scores["observation_id"].to_list() == ["r0", "r1", "r3", "r4", "r5", "r6", "r7"]
    assert result.exclusions["observation_id"].to_list() == ["r2"]
    assert result.scores["source_row_index"].to_list() == [0, 1, 3, 4, 5, 6, 7]


def test_pca_component_count_is_rank_bounded():
    x = np.column_stack([np.arange(8.0), np.arange(8.0) * 2.0, np.ones(8)])
    with pytest.raises(ValueError, match="at most 1 non-zero"):
        analyze_pca(FeatureMatrix(x), n_components=2)


def test_zero_rank_matrix_returns_skipped_result():
    result = analyze_pca(FeatureMatrix(np.ones((6, 3))), n_components=1)
    assert result.status is ResultStatus.SKIPPED
    assert result.reason == "zero_rank_matrix"


def test_sparse_pca_refuses_hidden_densification():
    x = sparse.csr_matrix(_matrix()[:10])
    with pytest.raises(ValueError, match="will not silently densify"):
        analyze_pca(FeatureMatrix(x), n_components=2)


def test_pca_scores_can_be_reused_with_attached_provenance():
    result = analyze_pca(FeatureMatrix(_matrix()), n_components=3)
    derived = result.to_feature_matrix()
    assert derived.feature_names == ("PC1", "PC2", "PC3")
    assert derived.provenance["derived_from"] == "pca"
    assert derived.provenance["inferential_allowed"] is True
