"""External annotation alignment and non-mutating attachment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

import pandas as pd

from ruddy.core.enums import AlignmentMode, AnnotationCoverage, ColumnKind, ColumnRole
from ruddy.core.exceptions import RoleConflictError, UnknownColumnError
from ruddy.core.types import KindOverrides, RoleOverrides
from ruddy.data.dataset import TabularDataset
from ruddy.data.roles import infer_column_kind
from ruddy.data.validation import AlignmentReport, align_annotations


class AlignedAnnotations:
    """One external annotation source aligned to a base observation index.

    The source data are copied on construction and every public accessor returns a
    defensive copy. ``ABSENT`` represents an explicitly unavailable source, while
    ``PARTIAL`` preserves incomplete coverage without silently dropping observations.
    """

    def __init__(
        self,
        *,
        source_name: str,
        data: pd.DataFrame,
        coverage: AnnotationCoverage,
        report: AlignmentReport | None,
        roles: Mapping[str, ColumnRole],
        kinds: Mapping[str, ColumnKind],
    ) -> None:
        name = str(source_name).strip()
        if not name:
            raise ValueError("source_name must be non-empty.")
        self._source_name = name
        self._data = data.copy(deep=True)
        self._coverage = coverage
        self._report = report
        self._roles = MappingProxyType(dict(roles))
        self._kinds = MappingProxyType(dict(kinds))

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def coverage(self) -> AnnotationCoverage:
        return self._coverage

    @property
    def report(self) -> AlignmentReport | None:
        return self._report

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(str(column) for column in self._data.columns)

    @property
    def roles(self) -> Mapping[str, ColumnRole]:
        return self._roles

    @property
    def kinds(self) -> Mapping[str, ColumnKind]:
        return self._kinds

    @property
    def available(self) -> bool:
        return self._coverage is not AnnotationCoverage.ABSENT

    def to_frame(self) -> pd.DataFrame:
        return self._data.copy(deep=True)

    def summary(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source_name": self.source_name,
            "coverage": self.coverage.value,
            "columns": list(self.columns),
        }
        if self.report is None:
            payload.update(
                base_count=len(self._data.index),
                annotation_count=0,
                covered_count=0,
                coverage_fraction=0.0,
                missing_ids=list(self._data.index),
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
    frame: pd.DataFrame,
    overrides: KindOverrides | None,
) -> dict[str, ColumnKind]:
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(frame.columns))
    if unknown:
        raise UnknownColumnError(f"Annotation kind overrides reference unknown columns: {unknown}.")
    resolved: dict[str, ColumnKind] = {}
    for column in frame.columns:
        observed = infer_column_kind(frame[column])
        if column not in overrides:
            resolved[column] = observed
            continue
        requested = (
            overrides[column] if isinstance(overrides[column], ColumnKind) else ColumnKind(overrides[column])
        )
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
    annotations: pd.DataFrame | None,
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
        empty = pd.DataFrame(index=dataset.observation_ids)
        return AlignedAnnotations(
            source_name=source_name,
            data=empty,
            coverage=AnnotationCoverage.ABSENT,
            report=None,
            roles={},
            kinds={},
        )

    aligned, report = align_annotations(
        dataset.observation_ids,
        annotations,
        id_column=id_column,
        mode=resolved_mode,
    )
    if id_column is not None and id_column in aligned.columns:
        aligned = aligned.drop(columns=[id_column])
    if not aligned.columns.is_unique:
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
    base = dataset.to_frame()
    original_index = base.index.copy()
    base.index = dataset.observation_ids

    roles = {spec.name: spec.role for spec in dataset.schema}
    kinds = {spec.name: spec.kind for spec in dataset.schema}

    for source in sources:
        if not isinstance(source, AlignedAnnotations):
            raise TypeError("sources must contain AlignedAnnotations objects.")
        if not source.available:
            continue
        frame = source.to_frame()
        collisions = sorted(set(frame.columns) & set(base.columns))
        if collisions:
            raise ValueError(
                f"Annotation source {source.source_name!r} collides with existing columns: {collisions}."
            )
        base = base.join(frame, how="left")
        roles.update(source.roles)
        kinds.update(source.kinds)

    if dataset.id_column is not None:
        base.index = original_index

    return TabularDataset(
        base,
        id_column=dataset.id_column,
        role_overrides=roles,
        kind_overrides=kinds,
    )
