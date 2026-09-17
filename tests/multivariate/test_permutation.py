import numpy as np
import pandas as pd

from ruddy import FeatureMatrix
from ruddy.multivariate import analyze_permutation_group_structure


def test_permanova_detects_clear_group_structure(group_dataset, separated_features):
    result = analyze_permutation_group_structure(
        separated_features, group_dataset, factor="group", n_permutations=199, random_state=3
    )
    permanova = result.summary.query("analysis == 'permanova'").iloc[0]
    assert result.status.value == "ok"
    assert permanova["value"] > 20
    assert permanova["r_squared"] > 0.4
    assert permanova["p_value"] <= 0.01


def test_permdisp_is_reported_together(group_dataset, separated_features):
    result = analyze_permutation_group_structure(
        separated_features, group_dataset, factor="group", n_permutations=19, random_state=1
    )
    assert set(result.summary["analysis"]) == {"permanova", "permdisp"}
    assert len(result.distances_to_centroid) == separated_features.n_observations
    assert result.groups["n"].sum() == separated_features.n_observations


def test_permutation_analysis_is_deterministic(group_dataset, separated_features):
    a = analyze_permutation_group_structure(
        separated_features, group_dataset, factor="group", n_permutations=49, random_state=11
    )
    b = analyze_permutation_group_structure(
        separated_features, group_dataset, factor="group", n_permutations=49, random_state=11
    )
    pd.testing.assert_frame_equal(a.summary, b.summary)


def test_zero_distance_space_is_degenerate(group_dataset):
    matrix = FeatureMatrix(
        np.ones((group_dataset.n_observations, 3)), observation_ids=group_dataset.observation_ids
    )
    result = analyze_permutation_group_structure(matrix, group_dataset, factor="group", n_permutations=9)
    assert result.status.value == "degenerate"
    assert result.reason == "zero_distance_space"


def test_missing_feature_rows_are_excluded(group_dataset, separated_features):
    values = separated_features.to_array()
    values[2, 1] = np.nan
    matrix = FeatureMatrix(values, observation_ids=group_dataset.observation_ids)
    result = analyze_permutation_group_structure(matrix, group_dataset, factor="group", n_permutations=9)
    assert len(result.exclusions) == 1
    assert result.exclusions.iloc[0]["reason"] == "non_finite_feature_row"


def test_non_euclidean_metric_records_pcoa_advisory(group_dataset, separated_features):
    result = analyze_permutation_group_structure(
        separated_features, group_dataset, factor="group", metric="cosine", n_permutations=9
    )
    assert result.status.value == "ok"
    advisory = next(a for a in result.advisories if a.code == "negative_pcoa_eigenvalues")
    assert advisory.context["metric"] == "cosine"
    assert 0 < advisory.context["negative_eigenvalue_fraction"] < 1
    assert result.provenance.parameters["metric"] == "cosine"


def test_permanova_matches_direct_sum_of_squares_formula():
    from ruddy import TabularDataset

    frame = pd.DataFrame({"id": range(6), "group": ["A", "A", "A", "B", "B", "B"]})
    dataset = TabularDataset(frame, id_column="id", role_overrides={"group": "factor"})
    values = np.array([[0.0], [0.2], [0.4], [2.0], [2.2], [2.4]])
    features = FeatureMatrix(values, observation_ids=dataset.observation_ids)
    result = analyze_permutation_group_structure(features, dataset, factor="group", n_permutations=0)
    observed = result.summary.query("analysis == 'permanova'").iloc[0]
    d = np.abs(values[:, None, 0] - values[None, :, 0])
    ss_total = np.triu(d**2, 1).sum() / 6
    ss_within = np.triu(d[:3, :3] ** 2, 1).sum() / 3 + np.triu(d[3:, 3:] ** 2, 1).sum() / 3
    expected_f = ((ss_total - ss_within) / 1) / (ss_within / 4)
    assert np.isclose(observed.value, expected_f)
    assert np.isclose(observed.r_squared, (ss_total - ss_within) / ss_total)
