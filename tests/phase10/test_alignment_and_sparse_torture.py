from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from ruddy import FeatureMatrix, TabularDataset, align_annotation_source, analyze_groups


def test_partial_annotation_coverage_is_preserved_without_global_row_drop():
    ids = [f"o{i}" for i in range(10)]
    ds = TabularDataset(
        pd.DataFrame({"id": ids, "y": np.arange(10, dtype=float)}),
        id_column="id",
        role_overrides={"y": "response"},
    )
    ann = pd.DataFrame({"id": ids[:6], "cohort": ["A", "A", "A", "B", "B", "B"]})
    aligned = align_annotation_source(
        ds,
        ann,
        id_column="id",
        mode="partial",
        role_overrides={"cohort": "factor"},
    )
    result = analyze_groups(ds, responses=("y",), groups=("cohort",), annotations=(aligned,), min_group_n=2)
    assert aligned.report is not None
    assert aligned.report.covered_count == 6
    assert aligned.report.missing_ids == tuple(ids[6:])
    coverage = result.group_coverage.set_index("group_column").loc["cohort"]
    assert coverage["n_dataset"] == 10
    assert coverage["n_group_present"] == 6
    assert coverage["n_group_missing"] == 4


def test_sparse_feature_matrix_roundtrip_remains_sparse():
    matrix = sparse.csr_matrix(np.eye(12))
    features = FeatureMatrix(matrix, observation_ids=range(12))
    assert features.is_sparse
    assert sparse.issparse(features.to_sparse())


def test_feature_matrix_duplicate_ids_remain_hard_error():
    with pytest.raises(Exception):  # noqa: B017 - the contract is that any hard error is raised
        FeatureMatrix(np.ones((3, 2)), observation_ids=["a", "a", "b"])
