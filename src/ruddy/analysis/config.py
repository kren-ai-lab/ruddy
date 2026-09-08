"""Unified analysis configuration contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

from ruddy.core.enums import AlignmentMode, ColumnKind, ColumnRole
from ruddy.core.types import KindOverrides, RoleOverrides
from ruddy.univariate.numeric import validate_quantiles


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Cross-cutting configuration shared by Ruddy analyses."""

    id_column: str | None = None
    role_overrides: RoleOverrides = field(default_factory=dict)
    kind_overrides: KindOverrides = field(default_factory=dict)
    annotation_alignment: AlignmentMode = AlignmentMode.STRICT
    random_state: int = 0

    # Phase 2 descriptive controls.
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    min_numeric_n: int = 3
    max_category_levels: int = 50
    max_missingness_patterns: int = 20
    max_pairwise_columns: int = 200

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
        quantiles = validate_quantiles(tuple(self.quantiles))
        if self.min_numeric_n < 2:
            raise ValueError("min_numeric_n must be at least 2.")
        if self.max_category_levels < 2:
            raise ValueError("max_category_levels must be at least 2.")
        if self.max_missingness_patterns < 1:
            raise ValueError("max_missingness_patterns must be at least 1.")
        if self.max_pairwise_columns < 1:
            raise ValueError("max_pairwise_columns must be at least 1.")

        object.__setattr__(self, "role_overrides", MappingProxyType(roles))
        object.__setattr__(self, "kind_overrides", MappingProxyType(kinds))
        object.__setattr__(self, "annotation_alignment", mode)
        object.__setattr__(self, "quantiles", quantiles)

    def dataset_kwargs(self) -> dict[str, object]:
        """Return arguments accepted directly by :class:`TabularDataset`."""

        return {
            "id_column": self.id_column,
            "role_overrides": dict(self.role_overrides),
            "kind_overrides": dict(self.kind_overrides),
        }

    def profiling_kwargs(self) -> dict[str, object]:
        """Return controls used by descriptive profiling."""

        return {
            "max_missingness_patterns": self.max_missingness_patterns,
            "max_pairwise_columns": self.max_pairwise_columns,
        }

    def univariate_kwargs(self) -> dict[str, object]:
        """Return controls used by univariate descriptive analysis."""

        return {
            "quantiles": self.quantiles,
            "min_numeric_n": self.min_numeric_n,
            "max_category_levels": self.max_category_levels,
            **self.profiling_kwargs(),
        }
