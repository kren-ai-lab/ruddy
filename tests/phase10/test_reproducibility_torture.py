from __future__ import annotations

import numpy as np
import pandas as pd

from ruddy import (
    FeatureMatrix,
    TabularDataset,
    analyze_anomalies,
    analyze_bayesian_eda,
    analyze_dependence,
    analyze_permutation_group_structure,
    analyze_representation_similarity,
    bootstrap_confidence_interval,
)


def test_bootstrap_same_seed_is_bitwise_reproducible():
    x = np.linspace(-2, 2, 30)
    a = bootstrap_confidence_interval(x, np.mean, n_resamples=400, method="percentile", random_state=17)
    b = bootstrap_confidence_interval(x, np.mean, n_resamples=400, method="percentile", random_state=17)
    assert a == b


def test_dependence_permutations_same_seed_are_identical(robust_tabular):
    a = analyze_dependence(robust_tabular, n_permutations=29, random_state=5)
    b = analyze_dependence(robust_tabular, n_permutations=29, random_state=5)
    pd.testing.assert_frame_equal(a.distance_correlations, b.distance_correlations)
    pd.testing.assert_frame_equal(a.mutual_information, b.mutual_information)


def test_permanova_same_seed_is_identical():
    rng = np.random.default_rng(41)
    n = 45
    labels = np.repeat(["A", "B", "C"], 15)
    x = rng.normal(size=(n, 4)) + np.repeat([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], 15, axis=0)
    ids = [f"o{i}" for i in range(n)]
    f = FeatureMatrix(x, observation_ids=ids)
    ds = TabularDataset(
        pd.DataFrame({"id": ids, "g": labels}), id_column="id", role_overrides={"g": "factor"}
    )
    a = analyze_permutation_group_structure(f, ds, factor="g", n_permutations=39, random_state=8)
    b = analyze_permutation_group_structure(f, ds, factor="g", n_permutations=39, random_state=8)
    pd.testing.assert_frame_equal(a.summary, b.summary)
    pd.testing.assert_frame_equal(a.distances_to_centroid, b.distances_to_centroid)


def test_mantel_same_seed_is_identical(representation_pair):
    x, y = representation_pair
    a = analyze_representation_similarity(x, y, mantel_permutations=39, random_state=11)
    b = analyze_representation_similarity(x, y, mantel_permutations=39, random_state=11)
    pd.testing.assert_frame_equal(a.mantel, b.mantel)


def test_bayesian_same_seed_is_identical(robust_tabular):
    a = analyze_bayesian_eda(robust_tabular, variables=("y",), groups=("group",), draws=700, random_state=19)
    b = analyze_bayesian_eda(robust_tabular, variables=("y",), groups=("group",), draws=700, random_state=19)
    pd.testing.assert_frame_equal(a.means, b.means)
    pd.testing.assert_frame_equal(a.mean_differences, b.mean_differences)


def test_isolation_forest_same_seed_is_identical():
    rng = np.random.default_rng(43)
    f = FeatureMatrix(rng.normal(size=(90, 5)), observation_ids=range(90))
    a = analyze_anomalies(f, methods=("isolation_forest",), random_state=13)
    b = analyze_anomalies(f, methods=("isolation_forest",), random_state=13)
    pd.testing.assert_frame_equal(a.scores, b.scores)


def test_tsne_same_seed_is_identical():
    from ruddy import analyze_tsne

    rng = np.random.default_rng(45)
    f = FeatureMatrix(rng.normal(size=(45, 6)), observation_ids=range(45))
    a = analyze_tsne(f, n_components=2, perplexity=8, random_state=23, max_iter=350)
    b = analyze_tsne(f, n_components=2, perplexity=8, random_state=23, max_iter=350)
    pd.testing.assert_frame_equal(a.coordinates, b.coordinates)
