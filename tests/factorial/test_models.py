from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from ruddy import ResultStatus, TabularDataset, analyze_factorial


def _reference_table(dataset, typ: int, robust=None):
    frame = dataset.frame.to_pandas().rename(
        columns={"response": "Y", "factor_a": "A", "factor_b": "B", "covariate": "X"}
    )
    model = ols("Y ~ C(A, Sum) + C(B, Sum) + X + C(A, Sum):C(B, Sum)", data=frame).fit()
    return anova_lm(model, typ=typ, robust=robust)


def test_type_ii_matches_statsmodels_reference(balanced_factorial_dataset):
    result = analyze_factorial(
        balanced_factorial_dataset,
        formula="response ~ factor_a * factor_b + covariate",
        ss_type=2,
    )
    reference = _reference_table(balanced_factorial_dataset, 2)
    mapping = {
        "factor_a": "C(A, Sum)",
        "factor_b": "C(B, Sum)",
        "covariate": "X",
        "factor_a:factor_b": "C(A, Sum):C(B, Sum)",
    }
    observed = {row["term"]: row for row in result.effects.iter_rows(named=True)}
    for term, ref_term in mapping.items():
        assert observed[term]["sum_sq"] == pytest.approx(
            float(reference.loc[ref_term, "sum_sq"]),  # pyrefly: ignore[bad-argument-type]
            rel=1e-10,  # pyrefly: ignore[bad-argument-type]
        )
        assert observed[term]["f_value"] == pytest.approx(float(reference.loc[ref_term, "F"]), rel=1e-10)  # pyrefly: ignore[bad-argument-type]
        assert observed[term]["p_value"] == pytest.approx(
            float(reference.loc[ref_term, "PR(>F)"]),  # pyrefly: ignore[bad-argument-type]
            rel=1e-10,  # pyrefly: ignore[bad-argument-type]
        )
    assert result.status is ResultStatus.OK


def test_type_iii_unbalanced_matches_sum_contrast_reference(unbalanced_factorial_dataset):
    result = analyze_factorial(
        unbalanced_factorial_dataset,
        formula="response ~ factor_a * factor_b + covariate",
        ss_type=3,
    )
    reference = _reference_table(unbalanced_factorial_dataset, 3)
    observed = {row["term"]: row for row in result.effects.iter_rows(named=True)}
    mapping = {
        "factor_a": "C(A, Sum)",
        "factor_b": "C(B, Sum)",
        "covariate": "X",
        "factor_a:factor_b": "C(A, Sum):C(B, Sum)",
    }
    for term, ref_term in mapping.items():
        assert observed[term]["sum_sq"] == pytest.approx(
            float(reference.loc[ref_term, "sum_sq"]),  # pyrefly: ignore[bad-argument-type]
            rel=1e-10,  # pyrefly: ignore[bad-argument-type]
        )
        assert observed[term]["p_value"] == pytest.approx(
            float(reference.loc[ref_term, "PR(>F)"]),  # pyrefly: ignore[bad-argument-type]
            rel=1e-10,  # pyrefly: ignore[bad-argument-type]
        )
    assert "unbalanced_factorial_design" in {advisory.code for advisory in result.advisories}
    assert result.model_summary["factor_contrasts"] == "sum_to_zero"


def test_type_iii_is_invariant_to_factor_category_order(unbalanced_factorial_dataset):
    first = analyze_factorial(
        unbalanced_factorial_dataset, formula="response ~ factor_a * factor_b + covariate", ss_type=3
    )
    frame = unbalanced_factorial_dataset.frame.to_pandas()
    frame["factor_a"] = pd.Categorical(frame["factor_a"], categories=["A1", "A0"], ordered=True)
    frame["factor_b"] = pd.Categorical(frame["factor_b"], categories=["B2", "B1", "B0"], ordered=True)
    reordered = TabularDataset(
        frame,
        id_column="id",
        role_overrides={
            "response": "response",
            "factor_a": "factor",
            "factor_b": "factor",
            "covariate": "covariate",
        },
    )
    second = analyze_factorial(reordered, formula="response ~ factor_a * factor_b + covariate", ss_type=3)
    one = {row["term"]: row for row in first.effects.iter_rows(named=True)}
    two = {row["term"]: row for row in second.effects.iter_rows(named=True)}
    for term in one:
        assert two[term]["f_value"] == pytest.approx(one[term]["f_value"], rel=1e-10)
        assert two[term]["p_value"] == pytest.approx(one[term]["p_value"], rel=1e-10)


def test_ancova_coefficient_and_effect_are_reported(balanced_factorial_dataset):
    result = analyze_factorial(
        balanced_factorial_dataset,
        response="response",
        factors=("factor_a",),
        covariates=("covariate",),
        ss_type=2,
    )
    assert "covariate" in set(result.effects.get_column("term").to_list())
    assert "covariate" in set(result.coefficients.get_column("parameter").to_list())
    effects_by_term = {row["term"]: row for row in result.effects.iter_rows(named=True)}
    assert effects_by_term["covariate"]["p_value"] < 0.01


def test_robust_hc3_matches_statsmodels_reference(unbalanced_factorial_dataset):
    result = analyze_factorial(
        unbalanced_factorial_dataset,
        formula="response ~ factor_a * factor_b + covariate",
        ss_type=3,
        robust_covariance="hc3",
    )
    reference = _reference_table(unbalanced_factorial_dataset, 3, robust="hc3")
    observed = {row["term"]: row for row in result.effects.iter_rows(named=True)}
    assert observed["factor_a"]["f_value"] == pytest.approx(
        float(reference.loc["C(A, Sum)", "F"]),  # pyrefly: ignore[bad-argument-type]
        rel=1e-10,  # pyrefly: ignore[bad-argument-type]
    )
    assert result.model_summary["robust_covariance"] == "hc3"


def test_fdr_adjustment_is_applied_only_when_requested(balanced_factorial_dataset):
    result = analyze_factorial(
        balanced_factorial_dataset,
        formula="response ~ factor_a * factor_b + covariate",
        p_adjust="fdr_bh",
    )
    assert result.effects.get_column("family_size").n_unique() == 1
    assert result.effects.get_column("family_size")[0] == len(result.effects)
    assert result.effects.get_column("q_value").is_not_null().all()
    q_vals = result.effects.get_column("q_value")
    p_vals = result.effects.get_column("p_value")
    assert (q_vals >= p_vals - 1e-15).all()


def test_complete_case_exclusions_are_observable(balanced_factorial_dataset):
    frame = balanced_factorial_dataset.frame.to_pandas()
    frame.loc[0, "response"] = np.nan
    frame.loc[1, "covariate"] = np.inf
    frame.loc[2, "factor_a"] = None
    dataset = TabularDataset(
        frame,
        id_column="id",
        role_overrides={
            "response": "response",
            "factor_a": "factor",
            "factor_b": "factor",
            "covariate": "covariate",
        },
    )
    result = analyze_factorial(dataset, formula="response ~ factor_a * factor_b + covariate")
    assert len(result.exclusions) == 3
    assert set(result.exclusions.get_column("observation_id").to_list()) == {"obs_0", "obs_1", "obs_2"}
    assert result.model_summary["n_complete_case"] == dataset.frame.height - 3


def test_constant_response_is_degenerate(balanced_factorial_dataset):
    frame = balanced_factorial_dataset.frame.to_pandas()
    frame["response"] = 1.0
    dataset = TabularDataset(
        frame, id_column="id", role_overrides={"response": "response", "factor_a": "factor"}
    )
    result = analyze_factorial(dataset, response="response", factors=("factor_a",))
    assert result.status is ResultStatus.DEGENERATE
    assert result.reason == "constant_response"


def test_collinear_covariates_are_degenerate(balanced_factorial_dataset):
    frame = balanced_factorial_dataset.frame.to_pandas()
    frame["covariate_2"] = frame["covariate"]
    dataset = TabularDataset(
        frame,
        id_column="id",
        role_overrides={
            "response": "response",
            "factor_a": "factor",
            "covariate": "covariate",
            "covariate_2": "covariate",
        },
    )
    result = analyze_factorial(
        dataset, response="response", factors=("factor_a",), covariates=("covariate", "covariate_2")
    )
    assert result.status is ResultStatus.DEGENERATE
    assert result.reason == "rank_deficient_design"


def test_empty_factorial_cell_with_full_interaction_is_rank_deficient(balanced_factorial_dataset):
    frame = balanced_factorial_dataset.frame.to_pandas()
    mask = (frame["factor_a"] == "A1") & (frame["factor_b"] == "B2")
    frame = frame.loc[~mask].reset_index(drop=True)
    dataset = TabularDataset(
        frame,
        id_column="id",
        role_overrides={"response": "response", "factor_a": "factor", "factor_b": "factor"},
    )
    result = analyze_factorial(
        dataset,
        response="response",
        factors=("factor_a", "factor_b"),
        interactions=(("factor_a", "factor_b"),),
    )
    assert result.status is ResultStatus.DEGENERATE
    assert result.reason == "rank_deficient_design"
    assert result.cells.get_column("is_empty").any()
    assert "empty_factorial_cells" in {advisory.code for advisory in result.advisories}


def test_small_cells_are_advisory_not_automatic_model_switch(balanced_factorial_dataset):
    frame = balanced_factorial_dataset.frame.to_pandas()
    keep = ~((frame["factor_a"] == "A0") & (frame["factor_b"] == "B0"))
    one = frame.index[(frame["factor_a"] == "A0") & (frame["factor_b"] == "B0")][:1]
    frame = pd.concat([frame.loc[keep], frame.loc[one]], ignore_index=True)
    dataset = TabularDataset(
        frame,
        id_column="id",
        role_overrides={"response": "response", "factor_a": "factor", "factor_b": "factor"},
    )
    result = analyze_factorial(dataset, response="response", factors=("factor_a", "factor_b"), min_cell_n=3)
    assert result.status is ResultStatus.OK
    assert "small_factorial_cells" in {advisory.code for advisory in result.advisories}
    assert result.model_summary["n_small_cells"] == 1


def test_type_ii_with_interaction_emits_advisory(balanced_factorial_dataset):
    result = analyze_factorial(
        balanced_factorial_dataset, formula="response ~ factor_a * factor_b", ss_type=2
    )
    assert "type_ii_with_interactions" in {advisory.code for advisory in result.advisories}


def test_observation_influence_diagnostics_are_aligned(balanced_factorial_dataset):
    result = analyze_factorial(balanced_factorial_dataset, formula="response ~ factor_a + covariate")
    obs = result.observation_diagnostics
    assert len(obs) == balanced_factorial_dataset.frame.height
    assert obs.get_column("observation_id").to_list() == list(balanced_factorial_dataset.observation_ids)
    assert {"studentized_residual", "leverage", "cooks_distance"} <= set(obs.columns)


def test_polars_native_dataset_integer_factor_matches_pandas():
    import polars as pl
    import polars.testing as pl_testing

    rng = np.random.default_rng(42)
    n = 60
    batch = rng.choice([10, 20, 30], size=n)
    group = rng.choice(["ctrl", "treat"], size=n)
    cov = rng.normal(size=n)
    y = 0.5 * (batch == 20) + 1.2 * (group == "treat") + 0.3 * cov + rng.normal(scale=0.1, size=n)

    obs_ids = list(range(100, 100 + n))
    df_polars = pl.DataFrame(
        {
            "id": obs_ids,
            "batch": batch,
            "group": group,
            "cov": cov,
            "y": y,
        }
    )
    # Add a null row to verify exclusions table and observation_id dtype preservation
    df_polars = df_polars.with_columns(
        pl.when(pl.col("id") == 105).then(None).otherwise(pl.col("y")).alias("y")
    )
    df_pandas = df_polars.to_pandas()

    ds_polars = TabularDataset(
        df_polars,
        id_column="id",
        role_overrides={"y": "response", "batch": "factor", "group": "factor", "cov": "covariate"},
    )
    ds_pandas = TabularDataset(
        df_pandas,
        id_column="id",
        role_overrides={"y": "response", "batch": "factor", "group": "factor", "cov": "covariate"},
    )

    res_polars = analyze_factorial(ds_polars, response="y", factors=("batch", "group"), covariates=("cov",))
    res_pandas = analyze_factorial(ds_pandas, response="y", factors=("batch", "group"), covariates=("cov",))

    pl_testing.assert_frame_equal(res_polars.effects, res_pandas.effects)
    assert len(res_polars.exclusions) == 1
    assert res_polars.exclusions.schema["observation_id"] == df_polars.schema["id"]
