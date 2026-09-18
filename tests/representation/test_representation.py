import numpy as np
import polars as pl
import polars.testing as pl_testing
import pytest

from ruddy import FeatureMatrix, analyze_representation_similarity, linear_cka
from ruddy.core.enums import ResultStatus
from ruddy.core.exceptions import AlignmentError
from ruddy.representation.analysis import CCA_CORRELATION_SCHEMA


def _pair(n=60, seed=2):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    y = x @ rng.normal(size=(4, 6)) + 0.05 * rng.normal(size=(n, 6))
    ids = [f"o{i}" for i in range(n)]
    return FeatureMatrix(x, observation_ids=ids, feature_names=[f"x{i}" for i in range(4)]), FeatureMatrix(
        y, observation_ids=ids, feature_names=[f"y{i}" for i in range(6)]
    )


def test_linear_cka_identity_is_one():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 5))
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-12)


def test_linear_cka_invariant_to_orthogonal_rotation():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(50, 4))
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    assert linear_cka(x, x @ q) == pytest.approx(1.0, abs=1e-12)


def test_representation_related_spaces_have_high_cca_and_cka():
    x, y = _pair()
    r = analyze_representation_similarity(x, y, cca_components=2, mantel_permutations=19, random_state=7)
    assert r.cca.status.value == "ok"
    assert r.cca.correlations["canonical_correlation"].to_numpy().min() > 0.95
    assert 0 <= r.cka.item(0, "cka") <= 1
    assert r.cka.item(0, "cka") > 0.6
    assert r.cca.x_weights.columns == ["feature", "CC1", "CC2"]
    assert r.cca.y_weights.columns == ["feature", "CC1", "CC2"]
    assert r.cca.x_loadings.columns == ["feature", "CC1", "CC2"]
    assert r.cca.y_loadings.columns == ["feature", "CC1", "CC2"]
    assert r.cca.x_scores.columns == ["observation_id", "CC1", "CC2"]
    assert r.cca.y_scores.columns == ["observation_id", "CC1", "CC2"]
    assert r.cca.x_scores.schema["observation_id"] == pl.String
    assert r.cca.y_scores.schema["observation_id"] == pl.String
    for col, dtype in CCA_CORRELATION_SCHEMA.items():
        assert r.cca.correlations.schema[col] == dtype


def test_mantel_is_deterministic():
    x, y = _pair()
    a = analyze_representation_similarity(x, y, mantel_permutations=29, random_state=11)
    b = analyze_representation_similarity(x, y, mantel_permutations=29, random_state=11)
    pl_testing.assert_frame_equal(a.mantel, b.mantel)


def test_procrustes_skips_unequal_dimensions():
    x, y = _pair()
    r = analyze_representation_similarity(x, y, mantel_permutations=0)
    assert r.procrustes.item(0, "status") == "skipped"
    assert r.procrustes.item(0, "reason") == "procrustes_requires_equal_dimensions"
    assert r.procrustes.item(0, "disparity") is None
    assert r.procrustes.item(0, "similarity") is None
    assert r.procrustes.item(0, "n_dimensions") is None
    assert r.mantel.item(0, "p_value") is None


def test_procrustes_identical_equal_dimension_is_near_zero():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(40, 3))
    ids = range(40)
    x = FeatureMatrix(a, observation_ids=ids)
    y = FeatureMatrix(a.copy(), observation_ids=ids)
    r = analyze_representation_similarity(x, y, cca_components=2, mantel_permutations=0)
    assert r.procrustes.item(0, "disparity") < 1e-12


def test_strict_alignment_rejects_mismatch():
    rng = np.random.default_rng(5)
    x = FeatureMatrix(rng.normal(size=(5, 2)), observation_ids=["a", "b", "c", "d", "e"])
    y = FeatureMatrix(rng.normal(size=(5, 2)), observation_ids=["a", "b", "c", "d", "z"])
    with pytest.raises(AlignmentError):
        analyze_representation_similarity(x, y, mantel_permutations=0)


def test_partial_alignment_reports_mismatch():
    rng = np.random.default_rng(5)
    x = FeatureMatrix(rng.normal(size=(5, 2)), observation_ids=["a", "b", "c", "d", "e"])
    y = FeatureMatrix(rng.normal(size=(5, 2)), observation_ids=["a", "b", "c", "d", "z"])
    r = analyze_representation_similarity(x, y, alignment="partial", cca_components=1, mantel_permutations=0)
    assert r.alignment.covered_count == 4 and r.alignment.missing_ids == ("e",)


def test_nonfinite_joint_rows_are_reported():
    x, y = _pair(n=20)
    arr = y.to_array()
    arr[2, 0] = np.nan
    y = FeatureMatrix(arr, observation_ids=y.observation_ids)
    r = analyze_representation_similarity(x, y, cca_components=1, mantel_permutations=0)
    assert r.exclusions.height == 1 and r.exclusions.item(0, "observation_id") == "o2"
    assert r.exclusions.schema["observation_id"] == pl.String
    assert r.exclusions.schema["source_row_x"] == pl.Int64
    assert r.exclusions.schema["source_row_y"] == pl.Int64


def test_cca_effective_rank_guard():
    rng = np.random.default_rng(4)
    ids = range(20)
    base = rng.normal(size=(20, 1))
    x = FeatureMatrix(np.c_[base, base], observation_ids=ids)
    y = FeatureMatrix(np.c_[base, base], observation_ids=ids)
    r = analyze_representation_similarity(x, y, cca_components=2, mantel_permutations=0)
    assert r.cca.status is ResultStatus.SKIPPED
    assert r.cca.reason == "cca_components_exceed_effective_rank"
    assert r.cca.correlations.height == 0
    assert r.cca.x_weights.columns == ["feature"]
    assert r.cca.x_scores.columns == ["observation_id"]


def test_cka_zero_variance_degenerate():
    x = np.ones((20, 3))
    f1 = FeatureMatrix(x, observation_ids=range(20))
    f2 = FeatureMatrix(x, observation_ids=range(20))
    r = analyze_representation_similarity(f1, f2, cca_components=1, mantel_permutations=0)
    assert r.cka.item(0, "status") == "degenerate"
    assert r.cka.item(0, "cka") is None
    assert "zero-variance" in str(r.cka.item(0, "reason"))
