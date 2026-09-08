from __future__ import annotations

import pandas as pd

from ruddy import align_annotation_source, analyze_groups, resolve_responses


def test_multiple_responses_dispatch_independently_by_statistical_kind(
    response_dataset,
) -> None:
    assert resolve_responses(response_dataset) == ("activity", "class_response")

    result = analyze_groups(
        response_dataset,
        groups=("family",),
        min_group_n=2,
    )

    assert set(result.response_catalog["response"]) == {
        "activity",
        "class_response",
    }
    assert set(result.response_catalog["data_kind"]) == {"numeric", "categorical"}
    assert set(result.numeric_summaries["response"]) == {"activity"}
    assert set(result.categorical_summaries["response"]) == {"class_response"}
    assert set(result.numeric_comparisons["response"]) == {"activity"}
    assert set(result.categorical_comparisons["response"]) == {"class_response"}

    assert result.numeric_comparisons["family_id"].str.contains("activity").all()
    assert result.categorical_comparisons["family_id"].str.contains(
        "class_response"
    ).all()
    assert result.provenance.parameters["response_dispatch"] == (
        "observed_or_declared_statistical_kind"
    )


def test_partial_external_group_has_traceable_row_coverage_and_no_mutation(
    response_dataset,
) -> None:
    source = pd.DataFrame(
        {
            "id": ["obs_1", "obs_2", "obs_5", "obs_6", "outside"],
            "cohort": ["C1", "C1", "C2", "C2", "C9"],
        }
    )
    original_source = source.copy(deep=True)
    original_base = response_dataset.to_frame()

    aligned = align_annotation_source(
        response_dataset,
        source,
        source_name="cohorts",
        id_column="id",
        mode="partial",
        role_overrides={"cohort": "factor"},
    )
    result = analyze_groups(
        response_dataset,
        responses=("activity",),
        groups=("cohort",),
        annotations=(aligned,),
        min_group_n=2,
    )

    coverage = result.annotation_coverage.iloc[0]
    assert coverage["coverage"] == "partial"
    assert coverage["covered_count"] == 4
    assert coverage["n_unmatched_ids"] == 1

    group = result.group_coverage.iloc[0]
    assert group["n_dataset"] == 8
    assert group["n_group_present"] == 4
    assert group["n_group_missing"] == 4
    assert (result.numeric_comparisons["n_group_present"] == 4).all()
    assert (result.numeric_comparisons["n_group_missing"] == 4).all()

    pd.testing.assert_frame_equal(source, original_source)
    pd.testing.assert_frame_equal(response_dataset.to_frame(), original_base)


def test_explicit_responses_do_not_require_machine_learning_task_labels(
    mixed_response_frame,
) -> None:
    from ruddy import TabularDataset

    dataset = TabularDataset(mixed_response_frame, id_column="id")
    result = analyze_groups(
        dataset,
        responses=("activity", "class_response"),
        groups=("family",),
        min_group_n=2,
    )
    assert len(result.response_catalog) == 2
    assert result.provenance.analysis == "grouped_responses"
