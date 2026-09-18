import numpy as np
import pandas as pd
import polars.testing as pl_testing
import pytest

from ruddy import (
    AnalysisConfig,
    FeatureMatrix,
    TabularDataset,
    analyze,
    analyze_bayesian_eda,
    analyze_composition,
    analyze_representation_similarity,
)


def _bundle():
    rng = np.random.default_rng(22)
    n = 36
    ids = [f"o{i}" for i in range(n)]
    frame = pd.DataFrame({"id": ids, "y": rng.normal(size=n), "g": ["A"] * (n // 2) + ["B"] * (n // 2)})
    ds = TabularDataset(frame, id_column="id", role_overrides={"y": "response", "g": "factor"})
    x = rng.normal(size=(n, 3))
    y = x @ rng.normal(size=(3, 4)) + 0.1 * rng.normal(size=(n, 4))
    return (
        ds,
        FeatureMatrix(x, observation_ids=ids, feature_names=["x1", "x2", "x3"]),
        FeatureMatrix(y, observation_ids=ids, feature_names=["y1", "y2", "y3", "y4"]),
    )


def test_unified_requires_second_representation():
    ds, x, _ = _bundle()
    cfg = AnalysisConfig(enabled_blocks=("representation",))
    with pytest.raises(ValueError, match="comparison_features"):
        analyze(ds, config=cfg, features=x)


def test_unified_bayesian_matches_standalone():
    ds, _, _ = _bundle()
    cfg = AnalysisConfig(
        enabled_blocks=("bayesian",),
        responses=("y",),
        groups=("g",),
        bayesian_variables=("y",),
        bayesian_groups=("g",),
        bayesian_draws=500,
        random_state=3,
    )
    u = analyze(ds, config=cfg).bayesian
    s = analyze_bayesian_eda(ds, variables=("y",), groups=("g",), draws=500, random_state=3)
    assert u is not None
    pl_testing.assert_frame_equal(u.mean_differences, s.mean_differences)


def test_unified_compositional_matches_standalone():
    ds, _, _ = _bundle()
    rng = np.random.default_rng(1)
    c = FeatureMatrix(np.abs(rng.normal(size=(36, 4))) + 0.1, observation_ids=ds.observation_ids)
    cfg = AnalysisConfig(enabled_blocks=("compositional",), compositional_transform="ilr")
    u = analyze(ds, config=cfg, features=c).compositional
    s = analyze_composition(c, transform="ilr")
    assert u is not None
    assert u.transformed is not None
    np.testing.assert_allclose(u.transformed.to_array(), s.transformed.to_array())
    pl_testing.assert_frame_equal(u.variation_matrix, s.variation_matrix)
    pl_testing.assert_frame_equal(u.aitchison_distances, s.aitchison_distances)
    pl_testing.assert_frame_equal(u.zero_replacement, s.zero_replacement)


def test_unified_representation_matches_standalone():
    ds, x, y = _bundle()
    cfg = AnalysisConfig(
        enabled_blocks=("representation",),
        representation_cca_components=2,
        representation_mantel_permutations=19,
        random_state=7,
    )
    u = analyze(ds, config=cfg, features=x, comparison_features=y).representation
    s = analyze_representation_similarity(x, y, cca_components=2, mantel_permutations=19, random_state=7)
    assert u is not None
    pl_testing.assert_frame_equal(u.cca.correlations, s.cca.correlations)
    pl_testing.assert_frame_equal(u.cka, s.cka)
    pl_testing.assert_frame_equal(u.procrustes, s.procrustes)
    pl_testing.assert_frame_equal(u.distance_similarity, s.distance_similarity)
    pl_testing.assert_frame_equal(u.mantel, s.mantel)
