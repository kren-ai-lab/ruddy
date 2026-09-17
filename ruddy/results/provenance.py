"""Analysis provenance and reproducibility metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from ruddy._version import __version__

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class AnalysisProvenance:
    """Minimal reproducibility record attached to scientific results."""

    analysis: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    input_summary: Mapping[str, Any] = field(default_factory=dict)
    random_state: int | None = None
    library_version: str = __version__
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Freeze provenance mappings to prevent mutation."""
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "input_summary", MappingProxyType(dict(self.input_summary)))

    def to_dict(self) -> dict[str, Any]:
        """Return the provenance record serialized as a native dictionary."""
        return {
            "analysis": self.analysis,
            "parameters": dict(self.parameters),
            "input_summary": dict(self.input_summary),
            "random_state": self.random_state,
            "library_version": self.library_version,
            "created_at": self.created_at.isoformat(),
        }
