from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy import sparse

from ruddy import FeatureMatrix
from ruddy.core.exceptions import AlignmentError, FeatureMatrixValidationError


def test_dataframe_feature_matrix_infers_identity_and_feature_names() -> None:
    frame = pd.DataFrame(
        {"f1": [1.0, 2.0], "f2": [3.0, 4.0]},
        index=pd.Index(["a", "b"], name="sample"),
    )
    original = frame.copy(deep=True)

    matrix = FeatureMatrix(frame)

    assert matrix.shape == (2, 2)
    assert matrix.feature_names == ("f1", "f2")
    assert matrix.observation_ids.tolist() == ["a", "b"]
    frame.iloc[0, 0] = 999.0
    np.testing.assert_array_equal(matrix.to_array(), original.to_numpy())


def test_feature_matrix_rejects_non_numeric_dataframe() -> None:
    with pytest.raises(FeatureMatrixValidationError, match="only numeric"):
        FeatureMatrix(pd.DataFrame({"f1": [1.0], "label": ["x"]}))


def test_feature_matrix_supports_sparse_input() -> None:
    matrix = FeatureMatrix(
        sparse.csr_matrix([[1.0, 0.0], [0.0, 2.0]]),
        observation_ids=["a", "b"],
        feature_names=["x", "y"],
    )

    assert matrix.is_sparse
    np.testing.assert_array_equal(matrix.to_array(), np.array([[1.0, 0.0], [0.0, 2.0]]))


def test_matrix_metadata_cannot_silently_misalign() -> None:
    metadata = pd.DataFrame({"id": ["a", "c"], "group": [1, 2]})

    with pytest.raises(AlignmentError):
        FeatureMatrix(
            np.eye(2),
            observation_ids=["a", "b"],
            metadata=metadata,
            metadata_id_column="id",
        )


def test_partial_matrix_metadata_alignment_is_observable() -> None:
    metadata = pd.DataFrame({"id": ["a", "extra"], "group": [1, 2]})

    matrix = FeatureMatrix(
        np.eye(2),
        observation_ids=["a", "b"],
        metadata=metadata,
        metadata_id_column="id",
        metadata_alignment="partial",
    )

    assert matrix.alignment_report is not None
    assert matrix.alignment_report.missing_ids == ("b",)
    assert matrix.alignment_report.unmatched_ids == ("extra",)
    assert matrix.metadata is not None
    assert isinstance(matrix.metadata, pl.DataFrame)
    assert matrix.metadata["id"].to_list() == ["a", "b"]


def test_polars_numeric_frame_infers_feature_names_and_generated_ids() -> None:
    frame = pl.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    matrix = FeatureMatrix(frame)

    assert matrix.shape == (2, 2)
    assert matrix.feature_names == ("f1", "f2")
    assert matrix.observation_ids.to_list() == [0, 1]
    np.testing.assert_array_equal(matrix.to_array(), np.array([[1.0, 3.0], [2.0, 4.0]]))


def test_polars_frame_with_string_column_is_rejected() -> None:
    with pytest.raises(FeatureMatrixValidationError, match="only numeric"):
        FeatureMatrix(pl.DataFrame({"f1": [1.0, 2.0], "label": ["x", "y"]}))


def test_polars_metadata_with_id_column_aligns_partially() -> None:
    metadata = pl.DataFrame({"id": ["a", "extra"], "group": [1, 2]})

    matrix = FeatureMatrix(
        np.eye(2),
        observation_ids=["a", "b"],
        metadata=metadata,
        metadata_id_column="id",
        metadata_alignment="partial",
    )

    assert isinstance(matrix.metadata, pl.DataFrame)
    assert matrix.metadata["id"].to_list() == ["a", "b"]
    assert matrix.metadata["group"].to_list() == [1, None]
