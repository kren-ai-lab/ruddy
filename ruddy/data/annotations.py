"""External annotation alignment and non-mutating attachment."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import polars as pl

from ruddy.core.enums import AlignmentMode, AnnotationCoverage, ColumnKind, ColumnRole
from ruddy.core.exceptions import RoleConflictError, UnknownColumnError
from ruddy.data.dataset import TabularDataset
from ruddy.data.roles import infer_column_kind
from ruddy.data.validation import AlignmentReport, align_annotations

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from ruddy.core.types import KindOverrides, ObservationID, RoleOverrides, TableInput


class AlignedAnnotations:
    """One external annotation source aligned to a base observation index.

    The source data are stored as a Polars DataFrame. ``ABSENT`` represents an
    explicitly unavailable source, while ``PARTIAL`` preserves incomplete coverage
    without silently dropping observations.
    """

    def __init__(
        self,
        *,
        source_name: str,
        data: pl.DataFrame,
        coverage: AnnotationCoverage,
        report: AlignmentReport | None,
        roles: Mapping[str, ColumnRole],
        kinds: Mapping[str, ColumnKind],
        base_ids: tuple[ObservationID, ...],
    ) -> None:
        """Construct aligned annotations with metadata and coverage status."""
        name = str(source_name).strip()
        if not name:
            raise ValueError("source_name must be non-empty.")
        self._source_name = name
        self._data = data
        self._base_ids = tuple(base_ids)
        self._coverage = coverage
        self._report = report
        self._roles = MappingProxyType(dict(roles))
        self._kinds = MappingProxyType(dict(kinds))

    @property
    def source_name(self) -> str:
        """Return the name of the annotation source."""
        return self._source_name

    @property
    def base_ids(self) -> tuple[ObservationID, ...]:
        """Return the base observation identifiers."""
        return self._base_ids

    @property
    def coverage(self) -> AnnotationCoverage:
        """Return the annotation coverage classification."""
        return self._coverage

    @property
    def report(self) -> AlignmentReport | None:
        """Return the alignment diagnostic report if available."""
        return self._report

    @property
    def frame(self) -> pl.DataFrame:
        """Return the underlying Polars DataFrame."""
        return self._data

    @property
    def columns(self) -> tuple[str, ...]:
        """Return the annotation column names."""
        return tuple(str(column) for column in self._data.columns)

    @property
    def roles(self) -> Mapping[str, ColumnRole]:
        """Return the mapping from column names to semantic roles."""
        return self._roles

    @property
    def kinds(self) -> Mapping[str, ColumnKind]:
        """Return the mapping from column names to data kinds."""
        return self._kinds

    @property
    def available(self) -> bool:
        """Return whether the annotation source is present."""
        return self._coverage is not AnnotationCoverage.ABSENT

    def summary(self) -> dict[str, Any]:
        """Summarize annotation coverage and metadata as a dictionary."""
        payload: dict[str, Any] = {
            "source_name": self.source_name,
            "coverage": self.coverage.value,
            "columns": list(self.columns),
        }
        if self.report is None:
            payload.update(
                base_count=len(self._base_ids),
                annotation_count=0,
                covered_count=0,
                coverage_fraction=0.0,
                missing_ids=list(self._base_ids),
                unmatched_ids=[],
            )
        else:
            payload.update(self.report.to_dict())
        return payload


def _resolve_annotation_roles(
    columns: Iterable[str],
    overrides: RoleOverrides | None,
) -> dict[str, ColumnRole]:
    columns = tuple(columns)
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(columns))
    if unknown:
        raise UnknownColumnError(f"Annotation role overrides reference unknown columns: {unknown}.")
    resolved = dict.fromkeys(columns, ColumnRole.ANNOTATION)
    for column, value in overrides.items():
        role = value if isinstance(value, ColumnRole) else ColumnRole(value)
        if role is ColumnRole.IDENTIFIER:
            raise RoleConflictError(
                "External annotation columns cannot define the base observation identifier."
            )
        resolved[column] = role
    return resolved


def _resolve_annotation_kinds(
    frame: pl.DataFrame,
    overrides: KindOverrides | None,
) -> dict[str, ColumnKind]:
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(frame.columns))
    if unknown:
        raise UnknownColumnError(f"Annotation kind overrides reference unknown columns: {unknown}.")
    resolved: dict[str, ColumnKind] = {}
    for column in frame.columns:
        observed = infer_column_kind(frame.schema[column])
        if column not in overrides:
            resolved[column] = observed
            continue
        declared = overrides[column]
        requested = declared if isinstance(declared, ColumnKind) else ColumnKind(declared)
        if requested is ColumnKind.NUMERIC and observed is not ColumnKind.NUMERIC:
            raise ValueError(
                f"Annotation column {column!r} cannot be declared numeric without "
                "numeric dtype; Ruddy does not silently coerce values."
            )
        if requested is ColumnKind.BOOLEAN and observed is not ColumnKind.BOOLEAN:
            raise ValueError(
                f"Annotation column {column!r} cannot be declared boolean without boolean dtype."
            )
        if requested is ColumnKind.DATETIME and observed is not ColumnKind.DATETIME:
            raise ValueError(
                f"Annotation column {column!r} cannot be declared datetime without datetime dtype."
            )
        resolved[column] = requested
    return resolved


def align_annotation_source(
    dataset: TabularDataset,
    annotations: TableInput | None,
    *,
    source_name: str = "annotations",
    id_column: str | None = None,
    mode: AlignmentMode | str = AlignmentMode.STRICT,
    role_overrides: RoleOverrides | None = None,
    kind_overrides: KindOverrides | None = None,
) -> AlignedAnnotations:
    """Align one optional external annotation source to a dataset.

    ``annotations=None`` is preserved as an explicit ``ABSENT`` source instead of
    being conflated with an empty or fully missing table.
    """
    resolved_mode = mode if isinstance(mode, AlignmentMode) else AlignmentMode(mode)
    if annotations is None:
        return AlignedAnnotations(
            source_name=source_name,
            data=pl.DataFrame(),
            coverage=AnnotationCoverage.ABSENT,
            report=None,
            roles={},
            kinds={},
            base_ids=dataset.observation_ids,
        )

    aligned, report = align_annotations(
        dataset.observation_ids,
        annotations,
        id_column=id_column,
        mode=resolved_mode,
    )
    if id_column is not None and id_column in aligned.columns:
        aligned = aligned.drop(id_column)
    if len(set(aligned.columns)) != len(aligned.columns):
        raise ValueError("Annotation column names must be unique.")
    non_string = [column for column in aligned.columns if not isinstance(column, str)]
    if non_string:
        raise TypeError("Ruddy requires string annotation column names.")

    roles = _resolve_annotation_roles(aligned.columns, role_overrides)
    kinds = _resolve_annotation_kinds(aligned, kind_overrides)
    coverage = AnnotationCoverage.COMPLETE if report.complete else AnnotationCoverage.PARTIAL
    return AlignedAnnotations(
        source_name=source_name,
        data=aligned,
        coverage=coverage,
        report=report,
        roles=roles,
        kinds=kinds,
        base_ids=dataset.observation_ids,
    )


def attach_annotations(
    dataset: TabularDataset,
    sources: Iterable[AlignedAnnotations],
) -> TabularDataset:
    """Return a new dataset with aligned annotation columns attached.

    Base observations and their ordering are preserved exactly. Column-name
    collisions are rejected instead of overwritten.
    """
    sources = tuple(sources)
    base = dataset.frame

    roles = {spec.name: spec.role for spec in dataset.schema}
    kinds = {spec.name: spec.kind for spec in dataset.schema}

    for source in sources:
        if not isinstance(source, AlignedAnnotations):
            raise TypeError("sources must contain AlignedAnnotations objects.")
        if not source.available:
            continue
        if source.base_ids != dataset.observation_ids:
            raise ValueError(
                f"Annotation source {source.source_name!r} was aligned to a different observation "
                "ordering; realign it against this dataset before attaching."
            )
        frame = source.frame
        collisions = sorted(set(frame.columns) & set(base.columns))
        if collisions:
            raise ValueError(
                f"Annotation source {source.source_name!r} collides with existing columns: {collisions}."
            )
        if frame.width > 0:
            base = base.hstack(frame)
        roles.update(source.roles)
        kinds.update(source.kinds)

    return TabularDataset(
        base,
        id_column=dataset.id_column,
        observation_ids=None if dataset.id_column else dataset.observation_ids,
        role_overrides=roles,
        kind_overrides=kinds,
    )
