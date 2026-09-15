from __future__ import annotations

import numpy as np
import pytest
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ruddy import FeatureMatrix, ResultStatus, analyze_collinearity


def test_vif_and_tolerance_match_statsmodels(well_conditioned_features):
    array = well_conditioned_features.to_array()
    design = np.column_stack([np.ones(array.shape[0]), array])
    reference = [variance_inflation_factor(design, i) for i in range(1, design.shape[1])]
    result = analyze_collinearity(well_conditioned_features)
    np.testing.assert_allclose(result.features["vif"].to_numpy(), reference, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        result.features["tolerance"].to_numpy(), 1.0 / np.asarray(reference), rtol=1e-10, atol=1e-12
    )
    assert result.status is ResultStatus.OK
    assert result.summary["vif_intercept_policy"] == "included_in_auxiliary_regressions_not_reported"


def test_condition_number_matches_standardized_svd(well_conditioned_features):
    array = well_conditioned_features.to_array()
    z = (array - array.mean(axis=0)) / array.std(axis=0, ddof=1)
    singular = np.linalg.svd(z, compute_uv=False)
    result = analyze_collinearity(well_conditioned_features)
    assert result.summary["condition_number"] == pytest.approx(singular[0] / singular[-1])
    np.testing.assert_allclose(result.condition_spectrum["singular_value"], singular)


def test_rank_deficient_design_is_explicit_and_vif_not_fabricated():
    x = np.arange(1.0, 15.0)
    matrix = FeatureMatrix(np.column_stack([x, 2 * x, np.sin(x)]), feature_names=("x", "two_x", "sin_x"))
    result = analyze_collinearity(matrix)
    assert result.status is ResultStatus.DEGENERATE
    assert result.reason == "rank_deficient_design"
    assert result.features["vif"].isna().all()
    assert set(result.features.loc[~result.features["is_constant"], "reason"]) == {"rank_deficient_design"}


def test_constant_feature_has_own_degeneracy_reason():
    rng = np.random.default_rng(1)
    matrix = FeatureMatrix(
        np.column_stack([rng.normal(size=30), np.ones(30), rng.normal(size=30)]),
        feature_names=("a", "constant", "b"),
    )
    result = analyze_collinearity(matrix)
    row = result.features.set_index("feature").loc["constant"]
    assert row["status"] == "degenerate"
    assert row["reason"] == "constant_feature"


def test_insufficient_residual_df_is_skipped():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(4, 3))
    matrix = FeatureMatrix(values, feature_names=("a", "b", "c"))
    result = analyze_collinearity(matrix)
    assert result.status is ResultStatus.SKIPPED
    assert result.reason == "insufficient_residual_degrees_of_freedom"
