from __future__ import annotations

import pytest

from ruddy import (
    resolve_groups,
    summarize_group_coverage,
    summarize_grouped_numeric_responses,
)


def test_any_categorical_column_can_be_selected_as_group(response_dataset) -> None:
    groups = resolve_groups(response_dataset, ("category_group",))
    assert groups == ("category_group",)

    coverage = summarize_group_coverage(response_dataset, groups).iloc[0]
    assert coverage["n_dataset"] == 8
    assert coverage["n_group_present"] == 7
    assert coverage["n_group_missing"] == 1
    assert coverage["n_levels"] == 2


def test_numeric_coded_factor_is_valid_group(response_dataset) -> None:
    groups = resolve_groups(response_dataset, ("coded_factor",))
    assert groups == ("coded_factor",)

    table = summarize_grouped_numeric_responses(
        response_dataset,
        responses=("activity",),
        groups=groups,
    )
    assert set(table["group_level"]) == {"0", "1"}
    assert table["n_group_observations"].tolist() == [4, 4]


def test_plain_numeric_variable_is_not_silently_treated_as_group(response_dataset) -> None:
    with pytest.raises(ValueError, match="categorical/boolean"):
        resolve_groups(response_dataset, ("continuous_group_candidate",))


def test_grouped_numeric_summary_uses_only_group_covered_rows(response_dataset) -> None:
    table = summarize_grouped_numeric_responses(
        response_dataset,
        responses=("activity",),
        groups=("category_group",),
    )
    assert table["n_group_observations"].sum() == 7
    assert (table["n_dataset"] == 8).all()
