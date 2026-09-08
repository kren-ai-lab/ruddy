"""Domain-agnostic tabular dataset contract."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ruddy.core.enums import ColumnKind, ColumnRole
from ruddy.core.types import KindOverrides, RoleOverrides
from ruddy.data.roles import ColumnSpec, build_schema, resolve_kinds, resolve_roles
from ruddy.data.validation import validate_observation_ids


class TabularDataset:
    """Immutable-by-contract wrapper around a pandas DataFrame.

    Ruddy never coerces values, imputes missing observations, or deletes rows while
    constructing this object. Internal data are copied from the caller and public
    accessors return copies, preventing accidental mutation of the scientific input.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        id_column: str | None = None,
        role_overrides: RoleOverrides | None = None,
        kind_overrides: KindOverrides | None = None,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        if not data.columns.is_unique:
            duplicates = data.columns[data.columns.duplicated()].tolist()
            raise ValueError(
                "DataFrame column names must be unique; "
                f"duplicates={duplicates}."
            )
        non_string = [column for column in data.columns if not isinstance(column, str)]
        if non_string:
            raise TypeError(
                "Ruddy requires string column names for stable schemas; "
                f"non-string labels={non_string}."
            )

        self._data = data.copy(deep=True)
        self._roles = resolve_roles(
            self._data,
            id_column=id_column,
            overrides=role_overrides,
        )
        self._kinds = resolve_kinds(self._data, overrides=kind_overrides)

        inferred_id_columns = [
            column
            for column, role in self._roles.items()
            if role is ColumnRole.IDENTIFIER
        ]
        self._id_column = (
            id_column
            if id_column is not None
            else (inferred_id_columns[0] if inferred_id_columns else None)
        )

        raw_ids = (
            self._data[self._id_column]
            if self._id_column is not None
            else self._data.index
        )
        self._observation_ids = validate_observation_ids(raw_ids)
        self._schema = build_schema(
            self._data,
            roles=self._roles,
            kinds=self._kinds,
        )

    @property
    def n_observations(self) -> int:
        return len(self._data)

    @property
    def n_columns(self) -> int:
        return self._data.shape[1]

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(str(column) for column in self._data.columns)

    @property
    def id_column(self) -> str | None:
        return self._id_column

    @property
    def observation_ids(self) -> pd.Index:
        return self._observation_ids.copy()

    @property
    def schema(self) -> tuple[ColumnSpec, ...]:
        return self._schema

    def role_of(self, column: str) -> ColumnRole:
        return self._roles[column]

    def kind_of(self, column: str) -> ColumnKind:
        return self._kinds[column]

    def columns_with_role(self, *roles: ColumnRole | str) -> tuple[str, ...]:
        wanted = {
            role if isinstance(role, ColumnRole) else ColumnRole(role) for role in roles
        }
        return tuple(
            column for column in self._data.columns if self._roles[column] in wanted
        )

    def columns_with_kind(self, *kinds: ColumnKind | str) -> tuple[str, ...]:
        wanted = {
            kind if isinstance(kind, ColumnKind) else ColumnKind(kind) for kind in kinds
        }
        return tuple(
            column for column in self._data.columns if self._kinds[column] in wanted
        )

    def select(self, columns: Iterable[str]) -> pd.DataFrame:
        return self._data.loc[:, list(columns)].copy(deep=True)

    def to_frame(self) -> pd.DataFrame:
        """Return a defensive copy of the stored table."""

        return self._data.copy(deep=True)

    def __len__(self) -> int:
        return self.n_observations

    def __repr__(self) -> str:
        return (
            f"TabularDataset(n_observations={self.n_observations}, "
            f"n_columns={self.n_columns}, id_column={self.id_column!r})"
        )
