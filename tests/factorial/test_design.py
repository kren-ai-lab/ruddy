from __future__ import annotations

import pandas as pd
import pytest

from ruddy import TabularDataset
from ruddy.factorial import build_factorial_design


def test_star_formula_expands_main_effects_and_interaction(balanced_factorial_dataset):
    design = build_factorial_design(
        balanced_factorial_dataset,
        formula="response ~ factor_a * factor_b + covariate",
        ss_type="III",
    )
    assert design.ss_type == 3
    assert design.factors == ("factor_a", "factor_b")
    assert design.covariates == ("covariate",)
    assert design.interactions == (("factor_a", "factor_b"),)
    assert design.resolved_formula == "response ~ factor_a + factor_b + covariate + factor_a:factor_b"


def test_colon_formula_is_hierarchical(balanced_factorial_dataset):
    design = build_factorial_design(
        balanced_factorial_dataset,
        formula="response ~ factor_a:factor_b",
    )
    assert [term.label for term in design.terms] == [
        "factor_a", "factor_b", "factor_a:factor_b"
    ]


def test_three_way_interaction_includes_lower_order_hierarchy():
    frame = pd.DataFrame({
        "id": [f"o{i}" for i in range(16)],
        "y": list(range(16)),
        "a": ["a0", "a1"] * 8,
        "b": ["b0", "b0", "b1", "b1"] * 4,
        "c": ["c0"] * 8 + ["c1"] * 8,
    })
    dataset = TabularDataset(frame, id_column="id", role_overrides={"y": "response", "a": "factor", "b": "factor", "c": "factor"})
    design = build_factorial_design(dataset, formula="y ~ a * b * c", max_interaction_order=3)
    assert design.interactions == (
        ("a", "b"), ("a", "c"), ("b", "c"), ("a", "b", "c")
    )


def test_numeric_factor_is_inferred_from_explicit_role():
    frame = pd.DataFrame({"id": ["a", "b", "c", "d"], "y": [1., 2., 3., 4.], "batch": [0, 0, 1, 1]})
    dataset = TabularDataset(frame, id_column="id", role_overrides={"y": "response", "batch": "factor"})
    design = build_factorial_design(dataset, formula="y ~ batch")
    assert design.factors == ("batch",)
    assert design.covariates == ()


def test_numeric_predictor_is_covariate_by_default(balanced_factorial_dataset):
    design = build_factorial_design(balanced_factorial_dataset, formula="response ~ covariate")
    assert design.covariates == ("covariate",)


def test_formula_rejects_transform_functions(balanced_factorial_dataset):
    with pytest.raises(ValueError, match="exclude transforms"):
        build_factorial_design(balanced_factorial_dataset, formula="response ~ log(covariate)")


def test_formula_and_explicit_design_are_mutually_exclusive(balanced_factorial_dataset):
    with pytest.raises(ValueError, match="either formula"):
        build_factorial_design(
            balanced_factorial_dataset,
            formula="response ~ factor_a",
            response="response",
            factors=("factor_a",),
        )


def test_interaction_order_guard(balanced_factorial_dataset):
    with pytest.raises(ValueError, match="exceeds max_interaction_order"):
        build_factorial_design(
            balanced_factorial_dataset,
            response="response",
            factors=("factor_a", "factor_b"),
            covariates=("covariate",),
            interactions=(("factor_a", "factor_b", "covariate"),),
            max_interaction_order=2,
        )


def test_explicit_design_supports_column_names_not_valid_in_formula():
    frame = pd.DataFrame({"id": ["a", "b", "c", "d"], "my y": [1., 2., 2., 4.], "group label": ["x", "x", "y", "y"]})
    dataset = TabularDataset(frame, id_column="id", role_overrides={"my y": "response", "group label": "factor"})
    design = build_factorial_design(dataset, response="my y", factors=("group label",))
    assert design.response == "my y"
    assert design.factors == ("group label",)
