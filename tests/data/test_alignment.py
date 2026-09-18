from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from ruddy import AlignmentMode
from ruddy.core.exceptions import (
    AlignmentError,
    DuplicateObservationIDError,
    MissingObservationIDError,
)
from ruddy.data import align_annotations
from ruddy.data.validation import validate_observation_ids


def test_strict_alignment_requires_exact_coverage() -> None:
    annotations = pd.DataFrame({"id": ["a", "b"], "group": [1, 2]})

    with pytest.raises(AlignmentError, match=r"missing=.*c"):
        align_annotations(
            ["a", "b", "c"],
            annotations,
            id_column="id",
            mode=AlignmentMode.STRICT,
        )


def test_partial_alignment_reports_missing_and_unmatched_ids() -> None:
    annotations = pd.DataFrame({"id": ["a", "b", "extra"], "group": [10, 20, 30]})

    aligned, report = align_annotations(
        ["a", "b", "c"],
        annotations,
        id_column="id",
        mode="partial",
    )

    assert report.covered_count == 2
    assert report.missing_ids == ("c",)
    assert report.unmatched_ids == ("extra",)
    assert report.coverage_fraction == pytest.approx(2 / 3)
    assert not report.complete
    assert aligned["id"].to_list() == ["a", "b", "c"]
    assert aligned.filter(pl.col("id") == "a")["group"][0] == 10
    assert aligned.filter(pl.col("id") == "c")["group"][0] is None


def test_annotation_duplicates_fail_instead_of_expanding_rows() -> None:
    annotations = pd.DataFrame({"id": ["a", "a"], "group": [1, 2]})

    with pytest.raises(DuplicateObservationIDError):
        align_annotations(["a"], annotations, id_column="id", mode="partial")


def test_partial_alignment_with_polars_annotations_by_id_column() -> None:
    annotations = pl.DataFrame({"id": ["b", "a", "extra"], "group": [20, 10, 30]})

    aligned, report = align_annotations(
        ["a", "b", "c"],
        annotations,
        id_column="id",
        mode="partial",
    )

    assert isinstance(aligned, pl.DataFrame)
    assert aligned["id"].to_list() == ["a", "b", "c"]
    assert aligned["group"].to_list() == [10, 20, None]
    assert report.covered_count == 2
    assert report.missing_ids == ("c",)
    assert report.unmatched_ids == ("extra",)
    assert report.coverage_fraction == pytest.approx(2 / 3)
    assert not report.complete


def test_polars_annotations_without_id_column_raises_value_error() -> None:
    annotations = pl.DataFrame({"group": [10, 20]})

    with pytest.raises(ValueError, match=r"Polars annotations require id_column: Polars has no row index\."):
        align_annotations(["a", "b"], annotations)


def test_pandas_annotations_without_id_column_raises() -> None:
    annotations = pd.DataFrame({"group": [10, 20]}, index=["a", "b"])

    with pytest.raises(
        ValueError,
        match=r"Annotations require id_column; a pandas index is not an identity",
    ):
        align_annotations(["a", "b"], annotations)


def test_pandas_annotations_with_explicit_id_column() -> None:
    annotations = pd.DataFrame({"id": ["a", "b"], "group": [10, 20]})

    aligned, report = align_annotations(["a", "b"], annotations, id_column="id")

    assert isinstance(aligned, pl.DataFrame)
    assert report.complete
    assert aligned["group"].to_list() == [10, 20]
    assert "id" in aligned.columns


def test_validate_observation_ids_error_and_return_cases() -> None:
    with pytest.raises(MissingObservationIDError):
        validate_observation_ids([1, None])

    with pytest.raises(MissingObservationIDError):
        validate_observation_ids([1.0, float("nan")])

    with pytest.raises(DuplicateObservationIDError):
        validate_observation_ids(pl.Series(["a", "a"]))

    result = validate_observation_ids(np.array([3, 1, 2]))
    assert result == (3, 1, 2)
    assert all(type(x) is int for x in result)
