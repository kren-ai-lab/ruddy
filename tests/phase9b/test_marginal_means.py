import json
import numpy as np

from ruddy.factorial import analyze_marginal_means


def test_one_factor_emmeans_equal_observed_means_without_covariates(group_dataset):
    result = analyze_marginal_means(group_dataset, response="y", factors=("group",), p_adjust="none")
    observed = group_dataset.select(["y", "group"]).groupby("group")["y"].mean()
    for _, row in result.means.iterrows():
        level = json.loads(row.levels_json)["group"]
        assert np.isclose(row.estimate, observed.loc[level])


def test_emmeans_contrasts_are_difference_of_means(group_dataset):
    result = analyze_marginal_means(group_dataset, response="y", factors=("group",), p_adjust="fdr_bh")
    means = {json.loads(row.levels_json)["group"]: row.estimate for _, row in result.means.iterrows()}
    for _, row in result.contrasts.iterrows():
        a = json.loads(row.levels_a_json)["group"]
        b = json.loads(row.levels_b_json)["group"]
        assert np.isclose(row.estimate_difference, means[b] - means[a])
        assert row.q_value >= row.p_value


def test_emmeans_covariate_adjustment_runs(group_dataset):
    result = analyze_marginal_means(group_dataset, response="y", factors=("group",), covariates=("x",))
    assert len(result.means) == 3
    assert result.provenance.parameters["weighting"] == "equal"


def test_joint_factor_term_generates_cell_means(group_dataset):
    result = analyze_marginal_means(
        group_dataset,
        response="y",
        factors=("group", "source"),
        interactions=(("group", "source"),),
        terms=(("group", "source"),),
    )
    assert len(result.means) == 6
    assert set(result.means.term) == {"group:source"}


def test_emmeans_tracks_complete_case_exclusions(group_dataset):
    frame = group_dataset.to_frame()
    frame.loc[0, "y"] = np.nan
    from ruddy import TabularDataset
    ds = TabularDataset(frame, id_column="id", role_overrides={"group":"factor","source":"factor","x":"covariate","y":"response"})
    result = analyze_marginal_means(ds, response="y", factors=("group",))
    assert len(result.exclusions) == 1
