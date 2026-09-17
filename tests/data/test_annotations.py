from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from ruddy import (
    AnnotationCoverage,
    ColumnRole,
    align_annotation_source,
    attach_annotations,
)
from ruddy.core.exceptions import AlignmentError


def test_partial_annotations_preserve_coverage_and_do_not_mutate_inputs(
    response_dataset,
) -> None:
    annotations = pd.DataFrame(
        {
            "id": ["obs_1", "obs_2", "obs_5", "outside"],
            "batch": [1, 1, 2, 9],
            "site": ["north", "north", "south", "other"],
        }
    )
    original_annotations = annotations.copy(deep=True)
    original_dataset = response_dataset.to_frame()

    aligned = align_annotation_source(
        response_dataset,
        annotations,
        source_name="study_metadata",
        id_column="id",
        mode="partial",
        role_overrides={"batch": "factor"},
    )

    assert aligned.coverage is AnnotationCoverage.PARTIAL
    assert aligned.report is not None
    assert aligned.report.covered_count == 3
    assert aligned.report.missing_ids == (
        "obs_3",
        "obs_4",
        "obs_6",
        "obs_7",
        "obs_8",
    )
    assert aligned.report.unmatched_ids == ("outside",)
    assert aligned.roles["batch"] is ColumnRole.FACTOR
    assert aligned.frame.shape == (8, 2)
    assert aligned.to_frame().shape == (8, 2)

    augmented = attach_annotations(response_dataset, (aligned,))
    assert augmented.columns[-2:] == ("batch", "site")
    assert augmented.role_of("batch") is ColumnRole.FACTOR
    assert augmented.to_frame()["batch"].notna().sum() == 3

    pd.testing.assert_frame_equal(annotations, original_annotations)
    pd.testing.assert_frame_equal(response_dataset.to_frame(), original_dataset)


def test_strict_annotations_require_exact_one_to_one_coverage(response_dataset) -> None:
    annotations = pd.DataFrame({"id": ["obs_1", "obs_2"], "batch": [1, 2]})
    with pytest.raises(AlignmentError):
        align_annotation_source(
            response_dataset,
            annotations,
            id_column="id",
            mode="strict",
        )


def test_absent_annotation_source_is_explicit_and_attachment_is_noop(
    response_dataset,
) -> None:
    absent = align_annotation_source(
        response_dataset,
        None,
        source_name="optional_metadata",
    )
    assert absent.coverage is AnnotationCoverage.ABSENT
    assert absent.report is None
    summary = absent.summary()
    assert summary["coverage_fraction"] == 0.0
    assert summary["base_count"] == len(response_dataset.observation_id_tuple)
    assert summary["missing_ids"] == list(response_dataset.observation_id_tuple)

    augmented = attach_annotations(response_dataset, (absent,))
    pd.testing.assert_frame_equal(
        augmented.to_frame(),
        response_dataset.to_frame(),
    )


def test_annotation_column_collision_fails_explicitly(response_dataset) -> None:
    annotations = pd.DataFrame(
        {
            "id": response_dataset.observation_ids,
            "activity": range(response_dataset.n_observations),
        }
    )
    aligned = align_annotation_source(
        response_dataset,
        annotations,
        id_column="id",
    )
    with pytest.raises(ValueError, match="collides"):
        attach_annotations(response_dataset, (aligned,))


def test_aligned_annotations_frame_and_to_frame_with_polars(response_dataset) -> None:
    annotations = pl.DataFrame(
        {
            "id": ["obs_1", "obs_2"],
            "batch": [1, 2],
        }
    )
    aligned = align_annotation_source(
        response_dataset,
        annotations,
        id_column="id",
        mode="partial",
    )
    assert isinstance(aligned.frame, pl.DataFrame)
    assert isinstance(aligned.to_frame(), pd.DataFrame)
    assert aligned.frame.shape == (8, 1)
    assert aligned.to_frame().shape == (8, 1)
    assert "id" not in aligned.frame.columns
