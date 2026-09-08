"""Unified analysis configuration contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from ruddy.core.enums import AlignmentMode, ColumnKind, ColumnRole
from ruddy.core.types import KindOverrides, RoleOverrides


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Cross-cutting configuration shared by Ruddy analyses."""

    id_column: str | None = None
    role_overrides: RoleOverrides = field(default_factory=dict)
    kind_overrides: KindOverrides = field(default_factory=dict)
    annotation_alignment: AlignmentMode = AlignmentMode.STRICT
    random_state: int = 0

    def __post_init__(self) -> None:
        roles = {
            column: role if isinstance(role, ColumnRole) else ColumnRole(role)
            for column, role in self.role_overrides.items()
        }
        kinds = {
            column: kind if isinstance(kind, ColumnKind) else ColumnKind(kind)
            for column, kind in self.kind_overrides.items()
        }
        mode = (
            self.annotation_alignment
            if isinstance(self.annotation_alignment, AlignmentMode)
            else AlignmentMode(self.annotation_alignment)
        )
        object.__setattr__(self, "role_overrides", MappingProxyType(roles))
        object.__setattr__(self, "kind_overrides", MappingProxyType(kinds))
        object.__setattr__(self, "annotation_alignment", mode)

    def dataset_kwargs(self) -> dict[str, object]:
        """Return arguments accepted directly by :class:`TabularDataset`."""

        return {
            "id_column": self.id_column,
            "role_overrides": dict(self.role_overrides),
            "kind_overrides": dict(self.kind_overrides),
        }
