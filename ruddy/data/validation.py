"""Input validation and explicit annotation-alignment rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl

from ruddy.core.enums import AlignmentMode
from ruddy.core.exceptions import (
    AlignmentError,
    DuplicateObservationIDError,
    MissingObservationIDError,
    UnknownColumnError,
)

if TYPE_CHECKING:
    from ruddy.core.types import ObservationID


def validate_observation_ids(values: Any, *, source: str = "observations") -> tuple[ObservationID, ...]:
    """Validate non-missing, unique observation identifiers and return them as a tuple."""
    if hasattr(values, "to_list"):
        ids = values.to_list()
    elif hasattr(values, "tolist"):
        ids = values.tolist()
    else:
        ids = list(values)

    if any(v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)) for v in ids):
        raise MissingObservationIDError(f"{source.capitalize()} contain missing observation identifiers.")
    seen: set[Any] = set()
    duplicated = list(dict.fromkeys(v for v in ids if v in seen or seen.add(v)))
    if duplicated:
        preview = duplicated[:10]
        suffix = "" if len(duplicated) <= 10 else " ..."
        raise DuplicateObservationIDError(
            f"{source.capitalize()} contain duplicate observation identifiers: {preview}{suffix}."
        )
    return tuple(ids)


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
        """Return the proportion of base observations covered by the annotation."""
        return self.covered_count / self.base_count if self.base_count else 1.0

    @property
    def complete(self) -> bool:
        """Return True if all base and annotation IDs match with no omissions."""
        return not self.missing_ids and not self.unmatched_ids

    def to_dict(self) -> dict[str, Any]:
        """Serialize alignment diagnostics to a dictionary."""
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
    annotations: pl.DataFrame | pd.DataFrame,
    *,
    id_column: str | None = None,
    mode: AlignmentMode | str = AlignmentMode.STRICT,
) -> tuple[pl.DataFrame, AlignmentReport]:
    """Align external metadata to base IDs without silently dropping mismatches.

    Strict mode requires an exact one-to-one ID set. Partial mode returns a left-aligned
    table with missing annotation fields represented as null values and a report that
    explicitly lists missing and unmatched IDs.
    """
    mode = mode if isinstance(mode, AlignmentMode) else AlignmentMode(mode)

    if isinstance(annotations, pd.DataFrame):
        if id_column is None:
            raise ValueError(
                "Annotations require id_column; a pandas index is not an identity. "
                "Pass frame.reset_index() with the ID as a column."
            )
        annotation_id_values = annotations.get(id_column)
        annotations = pl.from_pandas(annotations, include_index=False)
    elif isinstance(annotations, pl.DataFrame):
        if id_column is None:
            raise ValueError("Polars annotations require id_column: Polars has no row index.")
        annotation_id_values = annotations.get_column(id_column, default=None)
    else:
        raise TypeError("annotations must be a Polars or pandas DataFrame.")

    if annotation_id_values is None:
        raise UnknownColumnError(f"Unknown annotation ID column: {id_column!r}.")

    base = validate_observation_ids(base_ids, source="base data")
    annotation_ids = validate_observation_ids(annotation_id_values, source="annotations")

    annotation_set = set(annotation_ids)
    base_set = set(base)
    missing = tuple(value for value in base if value not in annotation_set)
    unmatched = tuple(value for value in annotation_ids if value not in base_set)
    covered = len(base) - len(missing)

    report = AlignmentReport(
        mode=mode,
        base_count=len(base),
        annotation_count=len(annotation_ids),
        covered_count=covered,
        missing_ids=missing,
        unmatched_ids=unmatched,
    )

    if mode is AlignmentMode.STRICT and not report.complete:
        raise AlignmentError(
            "Strict annotation alignment requires exact one-to-one ID coverage; "
            f"missing={list(missing)!r}, unmatched={list(unmatched)!r}."
        )

    columns = annotations.drop(id_column) if id_column is not None else annotations
    if covered == 0:
        aligned = columns.clear(n=len(base))
    else:
        key = "__ruddy_id"
        while key in columns.columns:
            key += "_"
        base_key = pl.Series(key, list(base))
        annotation_key = pl.Series(key, list(annotation_ids), dtype=base_key.dtype)
        aligned = (
            base_key.to_frame()
            .join(columns.with_columns(annotation_key), on=key, how="left", maintain_order="left")
            .drop(key)
        )
    if id_column is not None:
        aligned = aligned.with_columns(pl.Series(id_column, list(base))).select(annotations.columns)
    return aligned, report
