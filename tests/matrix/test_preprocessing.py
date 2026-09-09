import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ruddy import FeatureMatrix, prepare_features


def test_no_scaling_preserves_dense_values():
    x = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    result = prepare_features(FeatureMatrix(x), scaling="none")
    np.testing.assert_allclose(result.matrix, x)
    assert result.metadata["scaling"]["method"] == "none"


@pytest.mark.parametrize(
    ("method", "transformer"),
    [
        ("standard", StandardScaler()),
        ("robust", RobustScaler()),
        ("minmax", MinMaxScaler()),
    ],
)
def test_dense_scaling_matches_sklearn(method, transformer):
    x = np.array([[1.0, 2.0], [3.0, 8.0], [7.0, 11.0], [9.0, 12.0]])
    result = prepare_features(FeatureMatrix(x), scaling=method)
    expected = transformer.fit_transform(x)
    np.testing.assert_allclose(result.matrix, expected, rtol=1e-12, atol=1e-12)


def test_non_finite_rows_are_explicitly_excluded():
    x = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, np.inf], [7.0, 8.0]])
    result = prepare_features(
        FeatureMatrix(x, observation_ids=["a", "b", "c", "d"]), scaling="none"
    )
    assert result.observation_ids.tolist() == ["a", "d"]
    assert result.source_row_indices.tolist() == [0, 3]
    assert result.exclusions["observation_id"].tolist() == ["b", "c"]
    assert set(result.exclusions["reason"]) == {"non_finite_feature_row"}


def test_sparse_standard_scaling_stays_sparse_without_centering():
    x = sparse.csr_matrix([[0.0, 1.0], [2.0, 0.0], [3.0, 4.0]])
    result = prepare_features(FeatureMatrix(x), scaling="standard")
    assert sparse.issparse(result.matrix)
    assert result.metadata["scaling"]["centering_applied"] is False
    assert result.metadata["scaling"]["implicit_densification"] is False


def test_sparse_minmax_refuses_hidden_densification():
    x = sparse.csr_matrix([[0.0, 1.0], [2.0, 0.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="will not silently densify"):
        prepare_features(FeatureMatrix(x), scaling="minmax")


def test_sparse_non_finite_rows_are_excluded():
    x = sparse.csr_matrix([[0.0, 1.0], [2.0, np.inf], [3.0, 4.0]])
    result = prepare_features(FeatureMatrix(x, observation_ids=[10, 11, 12]))
    assert result.observation_ids.tolist() == [10, 12]
    assert result.exclusions["observation_id"].tolist() == [11]
