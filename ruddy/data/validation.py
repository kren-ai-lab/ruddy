"""Input validation and explicit annotation-alignment rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ruddy.core.enums import AlignmentMode
from ruddy.core.exceptions import (
    AlignmentError,
    DuplicateObservationIDError,
    MissingObservationIDError,
    UnknownColumnError,
)


def validate_observation_ids(values: Any, *, source: str = "observations") -> pd.Index:
    """Validate non-missing, unique observation identifiers."""
    ids = pd.Index(values, copy=True)
    if ids.hasnans:
        msg = f"{source.capitalize()} contain missing observation identifiers."
        raise MissingObservationIDError(msg)
    duplicated = ids[ids.duplicated()].unique().tolist()
    if duplicated:
        preview = duplicated[:10]
        suffix = "" if len(duplicated) <= 10 else " ..."
        msg = f"{source.capitalize()} contain duplicate observation identifiers: {preview}{suffix}."
        raise DuplicateObservationIDError(msg)
    return ids


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    """Observable summary of annotation/metadata alignment."""

    mode: AlignmentMode
    base_count: int
    annotation_count: int
    covered_count: int
    missing_ids: tuple[Any, ...]
    unmatched_ids: tuple[Any, ...]

    @property
    def coverage_fraction(self) -> float:
        return self.covered_count / self.base_count if self.base_count else 1.0

    @property
    def complete(self) -> bool:
        return not self.missing_ids and not self.unmatched_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "base_count": self.base_count,
            "annotation_count": self.annotation_count,
            "covered_count": self.covered_count,
            "coverage_fraction": self.coverage_fraction,
            "missing_ids": list(self.missing_ids),
            "unmatched_ids": list(self.unmatched_ids),
            "complete": self.complete,
        }


def align_annotations(
    base_ids: Any,
    annotations: pd.DataFrame,
    *,
    id_column: str | None = None,
    mode: AlignmentMode | str = AlignmentMode.STRICT,
) -> tuple[pd.DataFrame, AlignmentReport]:
    """Align external metadata to base IDs without silently dropping mismatches.

    Strict mode requires an exact one-to-one ID set. Partial mode returns a left-aligned
    table with missing annotation fields represented as missing values and a report that
    explicitly lists missing and unmatched IDs.
    """
    if not isinstance(annotations, pd.DataFrame):
        msg = "annotations must be a pandas DataFrame."
        raise TypeError(msg)

    mode = mode if isinstance(mode, AlignmentMode) else AlignmentMode(mode)
    base_index = validate_observation_ids(base_ids, source="base data")

    source = annotations.copy(deep=True)
    if id_column is None:
        annotation_ids = validate_observation_ids(source.index, source="annotations")
        indexed = source.copy(deep=True)
        indexed.index = annotation_ids
    else:
        if id_column not in source.columns:
            msg = f"Unknown annotation ID column: {id_column!r}."
            raise UnknownColumnError(msg)
        annotation_ids = validate_observation_ids(source[id_column], source="annotations")
        indexed = source.set_index(id_column, drop=False)
        indexed.index = annotation_ids

    base_set = set(base_index.tolist())
    annotation_set = set(annotation_ids.tolist())
    missing = tuple(value for value in base_index if value not in annotation_set)
    unmatched = tuple(value for value in annotation_ids if value not in base_set)
    covered = len(base_index) - len(missing)

    report = AlignmentReport(
        mode=mode,
        base_count=len(base_index),
        annotation_count=len(annotation_ids),
        covered_count=covered,
        missing_ids=missing,
        unmatched_ids=unmatched,
    )

    if mode is AlignmentMode.STRICT and not report.complete:
        msg = (
            "Strict annotation alignment requires exact one-to-one ID coverage; "
            f"missing={list(missing)!r}, unmatched={list(unmatched)!r}."
        )
        raise AlignmentError(msg)

    aligned = indexed.reindex(base_index).copy(deep=True)
    aligned.index = base_index
    if id_column is not None:
        aligned[id_column] = base_index.to_numpy(copy=True)
    return aligned, report
