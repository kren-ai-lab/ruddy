from __future__ import annotations

import pandas as pd
import polars as pl

from ruddy import TabularDataset, align_annotation_source, analyze_groups, resolve_responses


def test_multiple_responses_dispatch_independently_by_statistical_kind(
    response_dataset,
) -> None:
    assert resolve_responses(response_dataset) == ("activity", "class_response")

    result = analyze_groups(
        response_dataset,
        groups=("family",),
        min_group_n=2,
    )

    assert set(result.response_catalog["response"].to_list()) == {
        "activity",
        "class_response",
    }
    assert set(result.response_catalog["data_kind"].to_list()) == {"numeric", "categorical"}
    assert set(result.numeric_summaries["response"].to_list()) == {"activity"}
    assert set(result.categorical_summaries["response"].to_list()) == {"class_response"}
    assert set(result.numeric_comparisons["response"].to_list()) == {"activity"}
    assert set(result.categorical_comparisons["response"].to_list()) == {"class_response"}

    assert bool(result.numeric_comparisons["family_id"].str.contains("activity").all())
    assert bool(result.categorical_comparisons["family_id"].str.contains("class_response").all())
    assert result.provenance.parameters["response_dispatch"] == ("observed_or_declared_statistical_kind")


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

    coverage = result.annotation_coverage.row(0, named=True)
    assert coverage["coverage"] == "partial"
    assert coverage["covered_count"] == 4
    assert coverage["n_unmatched_ids"] == 1

    group = result.group_coverage.row(0, named=True)
    assert group["n_dataset"] == 8
    assert group["n_group_present"] == 4
    assert group["n_group_missing"] == 4
    assert bool((result.numeric_comparisons["n_group_present"] == 4).all())
    assert bool((result.numeric_comparisons["n_group_missing"] == 4).all())

    pd.testing.assert_frame_equal(source, original_source)
    pd.testing.assert_frame_equal(response_dataset.to_frame(), original_base)


def test_explicit_responses_do_not_require_machine_learning_task_labels(
    mixed_response_frame,
) -> None:
    dataset = TabularDataset(mixed_response_frame, id_column="id")
    result = analyze_groups(
        dataset,
        responses=("activity", "class_response"),
        groups=("family",),
        min_group_n=2,
    )
    assert result.response_catalog.height == 2
    assert result.provenance.analysis == "grouped_responses"


def test_numeric_comparisons_q_value_null_on_non_ok_and_family_size_int64() -> None:
    frame = pd.DataFrame(
        {
            "id": [f"obs_{i}" for i in range(1, 9)],
            "activity": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "family": ["A"] * 4 + ["B"] * 4,
        }
    )
    dataset = TabularDataset(
        frame,
        id_column="id",
        role_overrides={"activity": "response", "family": "factor"},
    )
    result = analyze_groups(dataset, responses=("activity",), groups=("family",))
    table = result.numeric_comparisons
    assert table.schema["family_size"] == pl.Int64
    assert table.schema["q_value"] == pl.Float64
    non_ok = table.filter(pl.col("status") != "ok")
    assert non_ok.height > 0
    assert non_ok["q_value"].null_count() == non_ok.height
