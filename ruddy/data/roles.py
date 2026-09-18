"""Column kind inference and statistical-role normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from ruddy.core.enums import ColumnKind, ColumnRole
from ruddy.core.exceptions import (
    KindConflictError,
    RoleConflictError,
    UnknownColumnError,
)

if TYPE_CHECKING:
    from ruddy.core.types import KindOverrides, RoleOverrides


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """Resolved statistical metadata for one column."""

    name: str
    role: ColumnRole
    kind: ColumnKind
    dtype: str


def infer_column_kind(dtype: pl.DataType) -> ColumnKind:
    """Infer a statistical kind from a Polars dtype without coercing values."""
    if dtype == pl.Boolean:
        return ColumnKind.BOOLEAN
    if dtype.is_numeric() or dtype.is_decimal():
        return ColumnKind.NUMERIC
    if dtype in (pl.Date, pl.Datetime) or isinstance(dtype, pl.Datetime):
        return ColumnKind.DATETIME
    if dtype == pl.String or isinstance(dtype, (pl.Categorical, pl.Enum)):
        return ColumnKind.CATEGORICAL
    return ColumnKind.UNKNOWN


def _coerce_role(value: ColumnRole | str) -> ColumnRole:
    try:
        return value if isinstance(value, ColumnRole) else ColumnRole(value)
    except ValueError as exc:
        raise RoleConflictError(f"Unknown column role: {value!r}.") from exc


def _coerce_kind(value: ColumnKind | str) -> ColumnKind:
    try:
        return value if isinstance(value, ColumnKind) else ColumnKind(value)
    except ValueError as exc:
        raise KindConflictError(f"Unknown column kind: {value!r}.") from exc


def resolve_roles(
    frame: pl.DataFrame,
    *,
    id_column: str | None,
    overrides: RoleOverrides | None = None,
) -> dict[str, ColumnRole]:
    """Resolve one non-overlapping statistical role for each column."""
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(frame.columns))
    if unknown:
        raise UnknownColumnError(f"Role overrides reference unknown columns: {unknown}.")

    resolved = dict.fromkeys(frame.columns, ColumnRole.VARIABLE)
    for column, role in overrides.items():
        resolved[column] = _coerce_role(role)

    declared_identifiers = [column for column, role in resolved.items() if role is ColumnRole.IDENTIFIER]

    if id_column is not None:
        if id_column not in frame.columns:
            raise UnknownColumnError(f"Unknown observation ID column: {id_column!r}.")
        explicit = resolved[id_column]
        if id_column in overrides and explicit is not ColumnRole.IDENTIFIER:
            raise RoleConflictError(
                f"Column {id_column!r} is the observation ID but was explicitly "
                "assigned "
                f"role {explicit.value!r}."
            )
        for column in declared_identifiers:
            if column != id_column:
                raise RoleConflictError(
                    "Only one identifier column is supported; "
                    f"{column!r} conflicts with id_column={id_column!r}."
                )
        resolved[id_column] = ColumnRole.IDENTIFIER
    elif len(declared_identifiers) > 1:
        raise RoleConflictError(f"Only one identifier column is supported; found {declared_identifiers}.")

    return resolved


def _validate_kind_override(
    column: str,
    observed: ColumnKind,
    requested: ColumnKind,
) -> None:
    if requested is ColumnKind.NUMERIC and observed is not ColumnKind.NUMERIC:
        raise KindConflictError(
            f"Column {column!r} cannot be declared numeric without numeric dtype; "
            f"observed {observed.value!r}. Ruddy does not silently coerce strings."
        )
    if requested is ColumnKind.BOOLEAN and observed is not ColumnKind.BOOLEAN:
        raise KindConflictError(
            f"Column {column!r} cannot be declared boolean without boolean dtype; "
            f"observed {observed.value!r}."
        )
    if requested is ColumnKind.DATETIME and observed is not ColumnKind.DATETIME:
        raise KindConflictError(
            f"Column {column!r} cannot be declared datetime without datetime dtype; "
            f"observed {observed.value!r}."
        )


def resolve_kinds(
    frame: pl.DataFrame,
    *,
    overrides: KindOverrides | None = None,
) -> dict[str, ColumnKind]:
    """Resolve data kinds, allowing explicit non-coercive overrides."""
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(frame.columns))
    if unknown:
        raise UnknownColumnError(f"Kind overrides reference unknown columns: {unknown}.")

    resolved: dict[str, ColumnKind] = {}
    for column in frame.columns:
        observed = infer_column_kind(frame.schema[column])
        if column not in overrides:
            resolved[column] = observed
            continue
        requested = _coerce_kind(overrides[column])
        _validate_kind_override(column, observed, requested)
        resolved[column] = requested
    return resolved


def build_schema(
    frame: pl.DataFrame,
    *,
    roles: dict[str, ColumnRole],
    kinds: dict[str, ColumnKind],
) -> tuple[ColumnSpec, ...]:
    """Build an ordered immutable column schema."""
    return tuple(
        ColumnSpec(
            name=str(column),
            role=roles[column],
            kind=kinds[column],
            dtype=str(frame.schema[column]),
        )
        for column in frame.columns
    )
