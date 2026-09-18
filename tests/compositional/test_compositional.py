import numpy as np
import polars as pl
import pytest

from ruddy import (
    FeatureMatrix,
    analyze_composition,
    closure,
    clr_transform,
    ilr_transform,
    multiplicative_zero_replacement,
)
from ruddy.compositional.analysis import ZERO_REPLACEMENT_SCHEMA


def _features():
    x = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 5.0, 1.0], [4.0, 1.0, 3.0, 2.0], [3.0, 3.0, 2.0, 2.0]])
    return FeatureMatrix(x, observation_ids=["a", "b", "c", "d"], feature_names=["A", "B", "C", "D"])


def test_closure_rows_sum_to_one():
    x = _features().to_array()
    c = closure(x)
    np.testing.assert_allclose(c.sum(axis=1), 1.0)


def test_clr_rows_sum_to_zero():
    c = clr_transform(_features().to_array())
    np.testing.assert_allclose(c.sum(axis=1), 0.0, atol=1e-12)


def test_ilr_has_d_minus_one_dimensions():
    z = ilr_transform(_features().to_array())
    assert z.shape == (4, 3)


def test_analyze_clr_returns_aitchison_and_variation():
    r = analyze_composition(_features(), transform="clr")
    assert r.transformed.shape == (4, 4)
    assert r.variation_matrix.shape == (4, 5)
    assert r.aitchison_distances.shape == (4, 5)


def test_aitchison_distances_string_ids_and_zero_diagonal():
    features = _features()
    r = analyze_composition(features, transform="clr")
    table = r.aitchison_distances
    assert table.schema["observation_id"] == pl.String
    for i in range(table.height):
        assert table.row(i)[i + 1] == 0.0


def test_variation_matrix_columns():
    features = _features()
    r = analyze_composition(features, transform="clr")
    assert r.variation_matrix.columns == ["feature", *features.feature_names]


def test_zero_data_requires_explicit_replacement():
    x = _features().to_array()
    x[0, 0] = 0
    with pytest.raises(ValueError, match="replace_zeros=True"):
        analyze_composition(FeatureMatrix(x), transform="clr")


def test_multiplicative_replacement_preserves_closure():
    x = np.array([[0.0, 2.0, 3.0], [1.0, 0.0, 4.0]])
    replaced, report = multiplicative_zero_replacement(x)
    assert (replaced > 0).all()
    np.testing.assert_allclose(replaced.sum(1), 1)
    assert report["replaced"].all()
    for col, dtype in ZERO_REPLACEMENT_SCHEMA.items():
        assert report.schema[col] == dtype


def test_negative_parts_rejected():
    x = _features().to_array()
    x[0, 0] = -1
    with pytest.raises(ValueError, match="negative"):
        analyze_composition(FeatureMatrix(x))
