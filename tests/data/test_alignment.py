from __future__ import annotations

import pandas as pd
import pytest

from ruddy import AlignmentMode
from ruddy.core.exceptions import AlignmentError, DuplicateObservationIDError
from ruddy.data import align_annotations


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
    assert aligned["id"].tolist() == ["a", "b", "c"]
    assert aligned.loc["a", "group"] == 10
    assert pd.isna(aligned.loc["c", "group"])


def test_annotation_duplicates_fail_instead_of_expanding_rows() -> None:
    annotations = pd.DataFrame({"id": ["a", "a"], "group": [1, 2]})

    with pytest.raises(DuplicateObservationIDError):
        align_annotations(["a"], annotations, id_column="id", mode="partial")
