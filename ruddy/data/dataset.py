"""Domain-agnostic tabular dataset contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import polars as pl
import pyarrow as pa

from ruddy.core.enums import ColumnKind, ColumnRole
from ruddy.data.roles import ColumnSpec, build_schema, resolve_kinds, resolve_roles
from ruddy.data.validation import validate_observation_ids

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ruddy.core.types import KindOverrides, ObservationID, RoleOverrides


class TabularDataset:
    """Immutable-by-contract wrapper around a polars DataFrame.

    Ruddy never coerces values, imputes missing observations, or deletes rows while
    constructing this object. Internal data are stored as polars frames and public
    accessors return copies, preventing accidental mutation of the scientific input.
    """

    def __init__(
        self,
        data: pl.DataFrame | pd.DataFrame,
        *,
        id_column: str | None = None,
        observation_ids: Sequence[ObservationID] | None = None,
        role_overrides: RoleOverrides | None = None,
        kind_overrides: KindOverrides | None = None,
    ) -> None:
        """Construct an immutable tabular dataset from a Polars or pandas DataFrame."""
        if isinstance(data, pd.DataFrame):
            default_index = (
                isinstance(data.index, pd.RangeIndex) and data.index.start == 0 and data.index.step == 1
            )
            if id_column is None and observation_ids is None and not default_index:
                raise ValueError(
                    "pandas DataFrame has a non-default index; "
                    "pass observation_ids=frame.index or reset_index()."
                )
            if not data.columns.is_unique:
                duplicates = data.columns[data.columns.duplicated()].tolist()
                raise ValueError(f"DataFrame column names must be unique; duplicates={duplicates}.")
            non_string = [column for column in data.columns if not isinstance(column, str)]
            if non_string:
                raise TypeError(
                    f"Ruddy requires string column names for stable schemas; non-string labels={non_string}."
                )
            input_backend = "pandas"
            non_string_categoricals = [
                column
                for column in data.columns
                if isinstance(data[column].dtype, pd.CategoricalDtype)
                and not pd.api.types.is_string_dtype(data[column].cat.categories)
            ]
            if non_string_categoricals:
                raise TypeError(
                    "pandas Categorical columns with non-string categories would be reinterpreted "
                    "as numeric by Polars; convert them to string categories or to Polars explicitly: "
                    f"{non_string_categoricals}."
                )
            try:
                # Deep-copy first: from_pandas may share numeric buffers with the caller's frame.
                self._frame = pl.from_pandas(data.copy(deep=True), include_index=False)
            except (pa.ArrowNotImplementedError, pa.ArrowInvalid, TypeError) as exc:
                raise TypeError(
                    "pandas DataFrame contains a dtype Polars cannot represent (e.g. complex); "
                    f"drop or convert it explicitly before constructing TabularDataset: {exc}"
                ) from exc
        elif isinstance(data, pl.DataFrame):
            input_backend = "polars"
            self._frame = data
        else:
            raise TypeError("data must be a polars or pandas DataFrame.")

        self._roles = resolve_roles(
            self._frame,
            id_column=id_column,
            overrides=role_overrides,
        )
        self._kinds = resolve_kinds(self._frame, overrides=kind_overrides)

        inferred_id_columns = [
            column for column, role in self._roles.items() if role is ColumnRole.IDENTIFIER
        ]
        self._id_column = (
            id_column if id_column is not None else (inferred_id_columns[0] if inferred_id_columns else None)
        )

        if self._id_column is not None and observation_ids is not None:
            raise ValueError("Cannot pass both id_column and observation_ids.")

        n = self._frame.height
        if self._id_column is not None:
            raw_ids = self._frame[self._id_column].to_list()
            id_source = "column"
        elif observation_ids is not None:
            if len(observation_ids) != n:
                raise ValueError("observation_ids length must equal frame.height.")
            raw_ids = observation_ids
            id_source = "argument"
        else:
            raw_ids = range(n)
            id_source = "generated"

        self._observation_ids = validate_observation_ids(raw_ids)

        self._schema = build_schema(
            self._frame,
            roles=self._roles,
            kinds=self._kinds,
        )

        self._provenance = {
            "input_backend": input_backend,
            "id_source": id_source,
        }

    @property
    def frame(self) -> pl.DataFrame:
        """Return the underlying Polars DataFrame."""
        return self._frame

    @property
    def observation_ids(self) -> tuple[ObservationID, ...]:
        """Return the sequence of observation identifiers."""
        return self._observation_ids

    @property
    def id_column(self) -> str | None:
        """Return the name of the identifier column, if one was specified."""
        return self._id_column

    @property
    def schema(self) -> tuple[ColumnSpec, ...]:
        """Return column specifications defining data types and semantic roles."""
        return self._schema

    @property
    def provenance(self) -> Mapping[str, Any]:
        """Return provenance metadata describing dataset origin and alignment."""
        return self._provenance

    def role_of(self, column: str) -> ColumnRole:
        """Return the semantic role assigned to the specified column."""
        return self._roles[column]

    def kind_of(self, column: str) -> ColumnKind:
        """Return the statistical data kind of the specified column."""
        return self._kinds[column]

    def columns_with_role(self, *roles: ColumnRole | str) -> tuple[str, ...]:
        """Return column names matching any of the specified roles."""
        wanted = {role if isinstance(role, ColumnRole) else ColumnRole(role) for role in roles}
        return tuple(column for column in self._frame.columns if self._roles[column] in wanted)

    def columns_with_kind(self, *kinds: ColumnKind | str) -> tuple[str, ...]:
        """Return column names matching any of the specified data kinds."""
        wanted = {kind if isinstance(kind, ColumnKind) else ColumnKind(kind) for kind in kinds}
        return tuple(column for column in self._frame.columns if self._kinds[column] in wanted)

    def __repr__(self) -> str:
        """Return a string representation of the tabular dataset."""
        return (
            f"TabularDataset(n_observations={self._frame.height}, "
            f"n_columns={self._frame.width}, id_column={self.id_column!r})"
        )
