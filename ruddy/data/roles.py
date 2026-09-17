"""Column kind inference and statistical-role normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_timedelta64_dtype,
)

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


def infer_column_kind(series: pd.Series) -> ColumnKind:
    """Infer a statistical kind without coercing or transforming values."""
    dtype = series.dtype
    if is_bool_dtype(dtype):
        return ColumnKind.BOOLEAN
    if is_complex_dtype(dtype) or is_timedelta64_dtype(dtype):
        return ColumnKind.UNKNOWN
    if is_datetime64_any_dtype(dtype):
        return ColumnKind.DATETIME
    if is_numeric_dtype(dtype):
        return ColumnKind.NUMERIC
    if isinstance(dtype, pd.CategoricalDtype) or is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype):
        return ColumnKind.CATEGORICAL
    return ColumnKind.UNKNOWN


def _coerce_role(value: ColumnRole | str) -> ColumnRole:
    try:
        return value if isinstance(value, ColumnRole) else ColumnRole(value)
    except ValueError as exc:
        msg = f"Unknown column role: {value!r}."
        raise RoleConflictError(msg) from exc


def _coerce_kind(value: ColumnKind | str) -> ColumnKind:
    try:
        return value if isinstance(value, ColumnKind) else ColumnKind(value)
    except ValueError as exc:
        msg = f"Unknown column kind: {value!r}."
        raise KindConflictError(msg) from exc


def resolve_roles(
    frame: pd.DataFrame,
    *,
    id_column: str | None,
    overrides: RoleOverrides | None = None,
) -> dict[str, ColumnRole]:
    """Resolve one non-overlapping statistical role for each column."""
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(frame.columns))
    if unknown:
        msg = f"Role overrides reference unknown columns: {unknown}."
        raise UnknownColumnError(msg)

    resolved = dict.fromkeys(frame.columns, ColumnRole.VARIABLE)
    for column, role in overrides.items():
        resolved[column] = _coerce_role(role)

    declared_identifiers = [column for column, role in resolved.items() if role is ColumnRole.IDENTIFIER]

    if id_column is not None:
        if id_column not in frame.columns:
            msg = f"Unknown observation ID column: {id_column!r}."
            raise UnknownColumnError(msg)
        explicit = resolved[id_column]
        if id_column in overrides and explicit is not ColumnRole.IDENTIFIER:
            msg = (
                f"Column {id_column!r} is the observation ID but was explicitly "
                "assigned "
                f"role {explicit.value!r}."
            )
            raise RoleConflictError(msg)
        for column in declared_identifiers:
            if column != id_column:
                msg = (
                    "Only one identifier column is supported; "
                    f"{column!r} conflicts with id_column={id_column!r}."
                )
                raise RoleConflictError(msg)
        resolved[id_column] = ColumnRole.IDENTIFIER
    elif len(declared_identifiers) > 1:
        msg = f"Only one identifier column is supported; found {declared_identifiers}."
        raise RoleConflictError(msg)

    return resolved


def _validate_kind_override(
    column: str,
    observed: ColumnKind,
    requested: ColumnKind,
) -> None:
    if requested is ColumnKind.NUMERIC and observed is not ColumnKind.NUMERIC:
        msg = (
            f"Column {column!r} cannot be declared numeric without numeric dtype; "
            f"observed {observed.value!r}. Ruddy does not silently coerce strings."
        )
        raise KindConflictError(msg)
    if requested is ColumnKind.BOOLEAN and observed is not ColumnKind.BOOLEAN:
        msg = (
            f"Column {column!r} cannot be declared boolean without boolean dtype; "
            f"observed {observed.value!r}."
        )
        raise KindConflictError(msg)
    if requested is ColumnKind.DATETIME and observed is not ColumnKind.DATETIME:
        msg = (
            f"Column {column!r} cannot be declared datetime without datetime dtype; "
            f"observed {observed.value!r}."
        )
        raise KindConflictError(msg)


def resolve_kinds(
    frame: pd.DataFrame,
    *,
    overrides: KindOverrides | None = None,
) -> dict[str, ColumnKind]:
    """Resolve data kinds, allowing explicit non-coercive overrides."""
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(frame.columns))
    if unknown:
        msg = f"Kind overrides reference unknown columns: {unknown}."
        raise UnknownColumnError(msg)

    resolved: dict[str, ColumnKind] = {}
    for column in frame.columns:
        observed = infer_column_kind(frame[column])
        if column not in overrides:
            resolved[column] = observed
            continue
        requested = _coerce_kind(overrides[column])
        _validate_kind_override(column, observed, requested)
        resolved[column] = requested
    return resolved


def build_schema(
    frame: pd.DataFrame,
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
            dtype=str(frame[column].dtype),
        )
        for column in frame.columns
    )
